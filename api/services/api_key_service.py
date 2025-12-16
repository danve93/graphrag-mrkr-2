"""
API Key Service for managing external application authentication.
"""
import secrets
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import uuid

from core.graph_db import graph_db

logger = logging.getLogger(__name__)

class ApiKeyService:
    """
    Service for managing API keys.
    """

    def create_api_key(
        self,
        name: str,
        role: str = "external",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new API key.
        """
        key_id = str(uuid.uuid4())
        # Generate a secure random API key, e.g. "sk-..." or just random hex
        # Prefixing with 'sk-' helps identification
        key_secret = f"sk-{secrets.token_urlsafe(32)}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        query = """
        MERGE (k:ApiKey {id: $key_id})
        ON CREATE SET
            k.key = $key_secret,
            k.name = $name,
            k.role = $role,
            k.created_at = $timestamp,
            k.is_active = true,
            k.metadata = $metadata_str
        RETURN k
        """
        
        import json
        metadata_str = json.dumps(metadata or {})
        
        graph_db.driver.execute_query(
            query,
            key_id=key_id,
            key_secret=key_secret,
            name=name,
            role=role,
            timestamp=timestamp,
            metadata_str=metadata_str
        )
        
        return {
            "id": key_id,
            "key": key_secret,
            "name": name,
            "role": role,
            "created_at": timestamp
        }

    def validate_api_key(self, key_secret: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key and return its details (role, name, id).
        """
        query = """
        MATCH (k:ApiKey {key: $key_secret, is_active: true})
        RETURN k
        """
        result = graph_db.driver.execute_query(query, key_secret=key_secret)
        
        if result and result.records:
            record = result.records[0]["k"]
            # Neo4j node to dict
            return {
                "id": record["id"],
                "name": record["name"],
                "role": record.get("role", "external")
            }
            
        return None

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys.
        """
        query = """
        MATCH (k:ApiKey)
        RETURN k
        ORDER BY k.created_at DESC
        """
        result = graph_db.driver.execute_query(query)
        
        keys = []
        if result and result.records:
            for record in result.records:
                node = record["k"]
                keys.append({
                    "id": node["id"],
                    "name": node["name"],
                    "role": node.get("role", "external"),
                    "created_at": node.get("created_at"),
                    "is_active": node.get("is_active", True),
                    # Do not return the full secret key in list if possible, or maybe masked
                    # For simplicty now, returning full or masked
                    "key_masked": f"{node['key'][:6]}..." if node.get('key') else None
                })
        return keys

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke (delete or deactivate) an API key.
        """
        query = """
        MATCH (k:ApiKey {id: $key_id})
        SET k.is_active = false
        RETURN k
        """
        result = graph_db.driver.execute_query(query, key_id=key_id)
        return bool(result and result.records)

# Singleton
api_key_service = ApiKeyService()
