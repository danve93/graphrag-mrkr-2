
import asyncio
import os
import sys
import hashlib
import uuid
from datetime import datetime, timezone
import json

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db

def restore_admin_key():
    print("Connecting to Neo4j...")
    try:
        graph_db.ensure_connected()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    # 1. Get token from env
    token = os.environ.get("JOBS_ADMIN_TOKEN")
    if not token:
        # Try loading from .env file manually if not in environ
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("JOBS_ADMIN_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break
        except Exception:
            pass
            
    if not token:
        print("ERROR: JOBS_ADMIN_TOKEN not found in environment or .env file.")
        return

    print(f"Found token: {token[:5]}...{token[-5:]}")
    
    # 2. Compute Hash
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    key_masked = f"{token[:7]}...{token[-4:]}"
    key_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    admin_name = "admin"
    
    print(f"Restoring admin key for '{admin_name}'...")
    
    # 3. Insert into DB
    try:
        # First check if it exists by hash to avoid duplicates
        check_query = "MATCH (k:ApiKey {hash: $hash}) RETURN k"
        with graph_db.driver.session() as session:
            result = session.run(check_query, hash=key_hash).single()
            if result:
                print("Key already exists in database (hash match). Activating if inactive...")
                session.run("MATCH (k:ApiKey {hash: $hash}) SET k.is_active = true", hash=key_hash)
                print("Done.")
                return

            # Allow multiple admin keys? Usually yes, but let's check name collision
            # If name='admin' exists with different hash, we might want to revoke it or just add another?
            # Let's just create this one.
            
            query = """
            MERGE (k:ApiKey {id: $key_id})
            ON CREATE SET
                k.hash = $key_hash,
                k.mask = $key_masked,
                k.name = $name,
                k.role = 'admin',
                k.created_at = $timestamp,
                k.is_active = true,
                k.restored = true
            RETURN k
            """
            
            session.run(
                query,
                key_id=key_id,
                key_hash=key_hash,
                key_masked=key_masked,
                name=admin_name,
                timestamp=timestamp
            )
            
        print("SUCCESS! Admin API Key restored in database.")
        print("You should be able to login now.")

    except Exception as e:
        print(f"Error restoring key: {e}")

if __name__ == "__main__":
    restore_admin_key()
