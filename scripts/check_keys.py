
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from core.graph_db import graph_db
from api.services.api_key_service import api_key_service

def list_keys():
    # Attempt to connect to localhost if neo4j fails (for local execution)
    os.environ["NEO4J_URI"] = os.environ.get("NEO4J_LOCAL_URI", "bolt://localhost:7687")
    
    print(f"Connecting to Neo4j at {os.environ['NEO4J_URI']}...")
    try:
        graph_db.driver.verify_connectivity()
    except Exception as e:
        print(f"Failed to connect: {e}")
        # Try original URI just in case
        print("Falling back to default environment URI...")
        # (This will likely fail again if it's neo4j:7687)
    
    try:
        keys = api_key_service.list_api_keys()
        print("\nActive API Keys:")
        for k in keys:
            if k.get('is_active'):
                print(f"- ID: {k['id']}, Name: {k['name']}, Role: {k['role']}, Mask: {k['key_masked']}, Created: {k['created_at']}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_keys()
