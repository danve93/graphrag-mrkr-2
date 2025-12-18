
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from config.settings import settings
from core.graph_db import graph_db
from api.services.api_key_service import api_key_service

def generate_admin_key():
    print("Connecting to Neo4j...")
    try:
        graph_db.ensure_connected()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    admin_name = "admin"
    print(f"Creating new admin API key for user '{admin_name}'...")
    
    try:
        # Check if one exists and revoke it first? 
        # For safety, let's just try to create one.
        # If it fails due to existing active key, we'll let the user know.
        
        # Check for existing key
        query = "MATCH (k:ApiKey {name: $name, is_active: true}) RETURN k"
        with graph_db.driver.session() as session:
            result = session.run(query, name=admin_name)
            record = result.single()
            if record:
                print(f"WARNING: An active API key already exists for '{admin_name}'.")
                print("To create a new one, we must revoke the old one.")
                revoke = input("Revoke existing key? (y/n): ")
                if revoke.lower() == 'y':
                    node_id = record["k"]["id"]
                    api_key_service.revoke_api_key(node_id)
                    print("Revoked existing key.")
                else:
                    print("Aborted.")
                    return

        key_data = api_key_service.create_api_key(name=admin_name, role="admin")
        print("\nSUCCESS! New Admin API Key created:")
        print("---------------------------------------------------")
        print(f"Key: {key_data['key']}")
        print("---------------------------------------------------")
        print("Please copy this key immediately. It will not be shown again.")
        print("Use this key in the 'Authorization: Bearer <key>' header or login field.")

    except Exception as e:
        print(f"Error creating key: {e}")

if __name__ == "__main__":
    generate_admin_key()
