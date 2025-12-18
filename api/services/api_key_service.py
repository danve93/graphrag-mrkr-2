import secrets
import logging
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import uuid

from core.graph_db import graph_db

logger = logging.getLogger(__name__)

class ApiKeyService:
    """
    Service for managing API keys with secure hashing.
    """

    def _hash_key(self, key: str) -> str:
        """Generate SHA-256 hash of the key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_api_key(
        self,
        name: str,
        role: str = "external",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new API key. Only returns the raw key once.
        """
        # Issue #34: Enforce tenant isolation (one active key per user)
        check_query = "MATCH (k:ApiKey {name: $name, is_active: true}) RETURN count(k) as count"
        existing = graph_db.driver.execute_query(check_query, name=name)
        if existing and existing.records and existing.records[0]["count"] > 0:
            raise ValueError(f"Active API key already exists for user '{name}'")

        key_id = str(uuid.uuid4())
        # Generate a secure random API key prefixing with 'sk-'
        key_secret = f"sk-{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(key_secret)
        
        # Store a masked version for UI/Auditing
        key_masked = f"{key_secret[:7]}...{key_secret[-4:]}"
        
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {})
        
        query = """
        MERGE (k:ApiKey {id: $key_id})
        ON CREATE SET
            k.hash = $key_hash,
            k.mask = $key_masked,
            k.name = $name,
            k.role = $role,
            k.created_at = $timestamp,
            k.is_active = true,
            k.metadata = $metadata_str
        RETURN k
        """
        
        graph_db.driver.execute_query(
            query,
            key_id=key_id,
            key_hash=key_hash,
            key_masked=key_masked,
            name=name,
            role=role,
            timestamp=timestamp,
            metadata_str=metadata_str
        )
        
        return {
            "id": key_id,
            "key": key_secret, # Returned only ONCE
            "name": name,
            "role": role,
            "created_at": timestamp
        }

    def validate_api_key(self, key_secret: str) -> Optional[Dict[str, Any]]:
        """
        Validate an API key by hashing it and comparing against stored hashes.
        """
        key_hash = self._hash_key(key_secret)
        
        query = """
        MATCH (k:ApiKey {hash: $key_hash, is_active: true})
        RETURN k
        """
        result = graph_db.driver.execute_query(query, key_hash=key_hash)
        
        if result and result.records:
            record = result.records[0]["k"]
            return {
                "id": record["id"],
                "name": record["name"],
                "role": record.get("role", "external")
            }
            
        return None

    def list_api_keys(self) -> List[Dict[str, Any]]:
        """
        List all API keys with masked values.
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
                    "key_masked": node.get("mask") or node.get("key_masked") or "sk-***..."
                })
        return keys

    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke (deactivate) an API key.
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
