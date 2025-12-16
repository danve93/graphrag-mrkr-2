"""
Token Service for generating and validating API tokens.
"""
import secrets
import logging
from typing import Optional, Dict, Any

from core.graph_db import graph_db

logger = logging.getLogger(__name__)

class TokenService:
    """
    Service for managing API tokens.
    """
    
    def create_token(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new token for a user.
        """
        # Generate a secure random token
        token = secrets.token_urlsafe(32)
        
        # In a real system, we'd hash this. For now, matching previous implementation logic
        # but wrapping it in service.
        # Ideally, we should store hash.
        
        # Update user with new token (single session/token/user model for simplicity per earlier impl)
        # OR add a Token node. Sticking to User property for now to match current schema
        
        query = """
        MATCH (u:User {id: $user_id})
        SET u.token_hash = $token
        RETURN u
        """
        
        graph_db.driver.execute_query(query, user_id=user_id, token=token)
        
        return {
            "token": token,
            "user_id": user_id,
            "type": "bearer",
            "role": metadata.get("role", "user") if metadata else "user",
            "expires_in": 3600 * 24 * 30 # 30 days dummy
        }

    def validate_token(self, token: str) -> Optional[str]:
        """
        Validate a token and return the user_id.
        """
        query = """
        MATCH (u:User {token_hash: $token})
        RETURN u.id as user_id
        """
        result = graph_db.driver.execute_query(query, token=token)
        
        if result and result.records:
            return result.records[0]["user_id"]
            
        return None

# Singleton instance
token_service = TokenService()
