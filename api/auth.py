"""
Authentication dependencies for FastAPI.
Merges legacy file-based auth with new graph-based user auth.
"""
from __future__ import annotations

import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Header, HTTPException, Depends, Cookie
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from api.services.token_service import token_service
from api.services.user_service import user_service

logger = logging.getLogger(__name__)

# --- Legacy Admin/Job Auth Support ---

USER_TOKEN_PATH = os.environ.get("JOBS_USER_TOKEN_PATH", "./data/job_user_token.txt")
ADMIN_ENV = "JOBS_ADMIN_TOKEN"

# Session TTL for admin sessions (seconds)
SESSION_TTL = 60 * 60 * 8

# In-memory store for admin sessions: {sid: expiry_epoch}
_admin_sessions: dict[str, float] = {}

def _load_admin_token() -> Optional[str]:
    return os.environ.get(ADMIN_ENV)

def verify_token(header_auth: Optional[str], require_admin: bool = False) -> str:
    """Verify an `Authorization: Bearer <token>` header against legacy static tokens."""
    if not header_auth:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if header_auth.lower().startswith("bearer "):
        token = header_auth.split(None, 1)[1].strip()
    else:
        token = header_auth.strip()

    admin = _load_admin_token()

    if require_admin:
        if admin and secrets.compare_digest(token, admin):
            return token
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # For now, this legacy function is mostly for admin or job routes.
    # Regular user auth is handled by get_current_user below.
    if admin and secrets.compare_digest(token, admin):
        return token
    raise HTTPException(status_code=401, detail="Invalid token")

def ensure_user_token() -> str:
    """Ensure a persistent user token exists for legacy jobs/scripts."""
    if os.path.exists(USER_TOKEN_PATH):
        try:
            with open(USER_TOKEN_PATH, "r") as f:
                content = f.read().strip()
                if content:
                    return content
        except Exception:
            pass
            
    token = secrets.token_urlsafe(32)
    try:
        os.makedirs(os.path.dirname(USER_TOKEN_PATH), exist_ok=True)
        with open(USER_TOKEN_PATH, "w") as f:
            f.write(token)
    except Exception as e:
        logger.warning(f"Failed to write user token to {USER_TOKEN_PATH}: {e}")
        
    return token


def require_user_or_admin(authorization: Optional[str] = Header(None)) -> str:
    return verify_token(authorization, require_admin=False)


def create_admin_session() -> str:
    """Create a short-lived admin session id and store it in-memory."""
    sid = secrets.token_urlsafe(32)
    _admin_sessions[sid] = time.time() + SESSION_TTL
    return sid


def invalidate_admin_session(sid: Optional[str]) -> None:
    """Invalidate an admin session id if present."""
    if not sid:
        return
    try:
        _admin_sessions.pop(sid, None)
    except Exception:
        pass


def validate_admin_session(sid: Optional[str]) -> bool:
    """Return True if session exists and is not expired."""
    if not sid:
        return False
    expiry = _admin_sessions.get(sid)
    if not expiry:
        return False
    if time.time() > expiry:
        # expired
        try:
            _admin_sessions.pop(sid, None)
        except Exception:
            pass
        return False
    return True


def require_admin_token(authorization: Optional[str] = Header(None), admin_session: Optional[str] = Cookie(None)) -> str:
    """FastAPI dependency that accepts either a valid `admin_session` cookie
    or an admin bearer token in the Authorization header.
    """
    # Prefer cookie-based session
    if admin_session and validate_admin_session(admin_session):
        return admin_session

    # Fallback to admin token via Authorization header
    return verify_token(authorization, require_admin=True)


def require_admin_cookie(admin_session: Optional[str] = Cookie(None)) -> str:
    """FastAPI dependency that requires a valid `admin_session` cookie."""
    if admin_session and validate_admin_session(admin_session):
        return admin_session
    raise HTTPException(status_code=401, detail="Missing or invalid admin session")


# --- New Graph-Based User Auth ---

security_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    authorization: Optional[str] = Header(None) # Fallback if not using Bearer scheme correctly
) -> Optional[str]:
    """
    Extract and validate user from Bearer token.
    Returns user_id or None (for anonymous/optional auth).
    
    Checks:
    1. HTTPBearer credentials
    2. Fallback to raw Authorization header (for legacy/custom clients)
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
    
    if not token:
        return None
        
    # Check if it's a legacy static admin token (optional interoperability)
    admin_token = _load_admin_token()
    if admin_token and token == admin_token:
        # Map admin to a special user ID or just let them pass as "admin"
        return "admin"
    
    # Validate against DB
    user_id = token_service.validate_token(token)
    
    if user_id:
        try:
            user_service.update_last_activity(user_id)
        except Exception:
            pass
            
    return user_id

async def require_current_user(
    user_id: Optional[str] = Depends(get_current_user)
) -> str:
    """Dependency demanding a valid user session."""
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )
    return user_id

async def require_admin(
    user_id: Optional[str] = Depends(get_current_user),
    authorization: Optional[str] = Header(None)
) -> str:
    """
    Require admin privileges. 
    Accepts:
    1. Legacy Admin Token (via Authorization header or env var matching)
    2. Authenticated User with role='admin'
    """
    # 1. Check Legacy Token
    # verify_token raises 401 if invalid/missing, so we catch it
    try:
        # We perform a manual check without raising to allow fallback
        if authorization:
            try:
                verify_token(authorization, require_admin=True)
                return "legacy_admin_token"
            except HTTPException:
                pass
    except Exception:
        pass
        
    # 2. Check User Role
    # get_current_user already validated the token and returned user_id
    if user_id and user_id != "admin": 
        # "admin" string is returned by get_current_user if it matched legacy admin token
        # but let's be explicit.
        from api.services.user_service import user_service
        user = user_service.get_user(user_id)
        if user and user.get("role") == "admin":
            return user_id

    # If user_id is "admin" (legacy admin token matched in get_current_user), allow it.
    if user_id == "admin":
        return "admin"

    # If we reached here, neither legacy token nor admin user checks passed.
    # We raise 403 Forbidden (or 401 if not authenticated at all, but get_current_user handles that partially)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    raise HTTPException(status_code=403, detail="Admin privileges required")
