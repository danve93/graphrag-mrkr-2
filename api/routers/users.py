"""
User management and identification router.

Handles generation of persistent user tokens for the Frontend to access
personalized features (memory, history) without requiring a full login system.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from api.services.user_service import user_service
from api.services.token_service import token_service
from api.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

class IdentifyRequest(BaseModel):
    """Request model for identifying a user."""
    username: Optional[str] = Field(None, description="Optional username (for external integrations)")
    api_key: Optional[str] = Field(None, description="Optional API key (if using external auth)")


class IdentifyResponse(BaseModel):
    """Response model with user credentials."""
    user_id: str
    token: str
    role: str
    is_new: bool


@router.post("/identify", response_model=IdentifyResponse)
async def identify_user(request: IdentifyRequest, req: Request):
    """
    Identify a user and return a persistent token.
    
    If no credentials provided, generates a new anonymous user.
    If valid credentials provided, returns existing user.
    
    If the request has a valid 'admin_session' cookie (from /admin login),
    the user is automatically promoted to 'admin' role.
    """
    try:
        # Check for Admin Cookie to bootstrap Admin Role
        is_admin_session = False
        from api.auth import validate_admin_session
        sid = req.cookies.get("admin_session")
        if sid and validate_admin_session(sid):
            is_admin_session = True

        # Check for API Key Auth
        if request.api_key:
            from api.services.api_key_service import api_key_service
            
            # Validate Key
            key_info = api_key_service.validate_api_key(request.api_key)
            if not key_info:
                raise HTTPException(status_code=401, detail="Invalid API Key")
            
            if not request.username:
                raise HTTPException(status_code=400, detail="Username is required for API Key auth")
                
            # Get/Create External User
            user = user_service.get_or_create_external_user(
                username=request.username,
                api_key_id=key_info["id"]
            )
            
            # Create Token with Role
            token_data = token_service.create_token(
                user_id=user["id"],
                metadata={"role": "external"}
            )
        
        else:
            # Standard/Internal Auth
            request_username = request.username
            
            # 1. Get or Create User
            # If admin session is valid, we might want to ensure this user stays admin or becomes admin
            role_to_set = "user"
            if is_admin_session:
                role_to_set = "admin"
                
            user = user_service.get_or_create_user(username=request_username, role=role_to_set)
            
            # 2. Token (use stored role or default to 'user')
            # Fix: Default logic previously forced 'user' role in token even if user was admin.
            role = user.get("role", "user")
            
            # If admin session detected but user role isn't admin (e.g. existing user), upgrade them
            if is_admin_session and role != "admin":
                # Update user role in DB
                try:
                    query = "MATCH (u:User {id: $user_id}) SET u.role = 'admin' RETURN u"
                    from core.graph_db import graph_db
                    graph_db.driver.execute_query(query, user_id=user["id"])
                    role = "admin"
                    user["role"] = "admin"
                except Exception as e:
                    logger.error(f"Failed to promote user to admin: {e}")

            token_data = token_service.create_token(
                user_id=user["id"],
                metadata={"role": role}
            )
        
        user_id = user["id"]
        
        return IdentifyResponse(
            user_id=user_id,
            token=token_data["token"],
            role=role,
            is_new=False # We need to check if user was just created. Service doesn't tell us yet.
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/me")
async def get_me(user_id: str = Depends(get_current_user)):
    """Get current user details."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
        
    user = user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    return user
