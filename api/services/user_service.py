"""
User Service for managing user identities and metadata.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.graph_db import graph_db

logger = logging.getLogger(__name__)

class UserService:
    """
    Service for managing user identities in Neo4j.
    """
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        query = """
        MATCH (u:User {id: $user_id})
        RETURN u
        """
        result = graph_db.driver.execute_query(query, user_id=user_id)
        if result and result.records:
            node = result.records[0]["u"]
            return dict(node)
        return None

    def create_user(
        self,
        username: Optional[str] = None,
        email: Optional[str] = None,
        role: str = "user",
        api_key_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new user.
        Role: 'admin', 'user', 'external'
        """
        user_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Link to ApiKey if provided
        api_key_link_query = ""
        if api_key_id:
            api_key_link_query = """
            WITH u
            MATCH (k:ApiKey {id: $api_key_id})
            MERGE (u)-[:AUTHENTICATED_WITH]->(k)
            """

        query = f"""
        MERGE (u:User {{id: $user_id}})
        ON CREATE SET
            u.username = $username,
            u.email = $email,
            u.role = $role,
            u.created_at = $timestamp,
            u.updated_at = $timestamp,
            u.last_activity_at = $timestamp,
            u.is_active = true,
            u.metadata = $metadata_str
        {api_key_link_query}
        RETURN u
        """
        
        import json
        metadata_str = json.dumps(metadata or {})
        
        graph_db.driver.execute_query(
            query,
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            api_key_id=api_key_id,
            timestamp=timestamp,
            metadata_str=metadata_str
        )
        
        return self.get_user(user_id)

    def update_last_activity(self, user_id: str):
        """Update user's last activity timestamp."""
        timestamp = datetime.now(timezone.utc).isoformat()
        query = """
        MATCH (u:User {id: $user_id})
        SET u.last_activity_at = $timestamp
        """
        graph_db.driver.execute_query(query, user_id=user_id, timestamp=timestamp)

    def get_or_create_user(self, username: Optional[str] = None, role: str = "user") -> Dict[str, Any]:
        """
        Identify or create user based on username.
        """
        if username:
            query = "MATCH (u:User {username: $username}) RETURN u"
            result = graph_db.driver.execute_query(query, username=username)
            if result and result.records:
                return dict(result.records[0]["u"])
        
        return self.create_user(username=username, role=role)

    def get_or_create_external_user(self, username: str, api_key_id: str) -> Dict[str, Any]:
        """
        Get or create an external user authenticated via API Key.
        They are scoped to that API Key tenant concept.
        """
        # Search for user who is authenticated with this key AND has this username
        query = """
        MATCH (u:User {username: $username})-[:AUTHENTICATED_WITH]->(k:ApiKey {id: $api_key_id})
        RETURN u
        """
        result = graph_db.driver.execute_query(query, username=username, api_key_id=api_key_id)
        
        if result and result.records:
            return dict(result.records[0]["u"])
            
        # Create new external user
        return self.create_user(
            username=username,
            role="external",
            api_key_id=api_key_id,
            metadata={"source": "api_key_integration"}
        )

# Singleton instance
user_service = UserService()
