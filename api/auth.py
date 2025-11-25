from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException

USER_TOKEN_PATH = os.environ.get("JOBS_USER_TOKEN_PATH", "./data/job_user_token.txt")
ADMIN_ENV = "JOBS_ADMIN_TOKEN"


def _load_admin_token() -> Optional[str]:
    return os.environ.get(ADMIN_ENV)


def ensure_user_token() -> str:
    """Ensure a persistent user token exists on disk and return it.

    This is intended to be run once (e.g., by the container on startup)
    to create a user-access token that can later be changed by editing
    the file. We keep this simple (file with token string).
    """
    try:
        os.makedirs(os.path.dirname(USER_TOKEN_PATH), exist_ok=True)
    except Exception:
        pass
    if os.path.exists(USER_TOKEN_PATH):
        try:
            with open(USER_TOKEN_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    token = secrets.token_hex(24)
    try:
        with open(USER_TOKEN_PATH, "w") as f:
            f.write(token)
    except Exception:
        # best-effort; token still returned
        pass
    return token


def _load_user_token() -> Optional[str]:
    if os.path.exists(USER_TOKEN_PATH):
        try:
            with open(USER_TOKEN_PATH, "r") as f:
                return f.read().strip()
        except Exception:
            return None
    return None


def verify_token(header_auth: Optional[str], require_admin: bool = False) -> str:
    """Verify an `Authorization: Bearer <token>` header.

    If `require_admin` is True, only the admin env token is accepted.
    Returns the token string when valid or raises HTTPException(401).
    """
    if not header_auth:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if header_auth.lower().startswith("bearer "):
        token = header_auth.split(None, 1)[1].strip()
    else:
        token = header_auth.strip()

    admin = _load_admin_token()
    user = _load_user_token()

    if require_admin:
        if admin and secrets.compare_digest(token, admin):
            return token
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # accept admin or user token
    if admin and secrets.compare_digest(token, admin):
        return token
    if user and secrets.compare_digest(token, user):
        return token
    raise HTTPException(status_code=401, detail="Invalid token")


def require_admin(authorization: Optional[str] = Header(None)) -> str:
    return verify_token(authorization, require_admin=True)


def require_user_or_admin(authorization: Optional[str] = Header(None)) -> str:
    return verify_token(authorization, require_admin=False)
