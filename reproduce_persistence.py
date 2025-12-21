
import asyncio
import os
import sys

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from core.graph_db import graph_db
from api.routers.graph_editor import EdgeCreateRequest

async def verify_persistence():
    print("Starting persistence verification...")
    
    # 1. identifying test nodes
    # We need two nodes to connect. Let's find two existing nodes.
    # We'll use a direct query to get two IDs.
    
    with graph_db.session_scope() as session:
        result = session.run("MATCH (n:Entity) RETURN n.id LIMIT 2")
        nodes = [record["n.id"] for record in result]
        
    if len(nodes) < 2:
        print("Not enough nodes in the database to test connection.")
        return

    source_id = nodes[0]
    target_id = nodes[1]
    
    print(f"Testing connection between {source_id} and {target_id}")
    
    # 2. Create an edge
    relation_type = "TEST_CONNECTION"
    print(f"Creating edge {relation_type}...")
    
    success = graph_db.create_relationship(
        source_id, 
        target_id, 
        relation_type, 
        {"created_by": "verification_script"}
    )
    
    if not success:
        print("Failed to create edge via graph_db.create_relationship")
        return

    print("Edge creation reported success.")

    # 3. Verify it exists
    print("Verifying edge existence in DB...")
    with graph_db.session_scope() as session:
        result = session.run(
            f"""
            MATCH (s:Entity {{id: $source_id}})-[r:`{relation_type}`]->(t:Entity {{id: $target_id}})
            RETURN count(r) as count
            """,
            source_id=source_id,
            target_id=target_id
        )
        count = result.single()["count"]
        
    if count == 1:
        print("SUCCESS: Edge correctly persisted in Neo4j.")
    else:
        print(f"FAILURE: Edge not found in Neo4j (count={count}).")
        
    # 4. Clean up
    print("Cleaning up test edge...")
    deleted = graph_db.delete_relationship(source_id, target_id, relation_type)
    
    if deleted:
        print("Test edge deleted successfully.")
    else:
        print("Failed to delete test edge.")

if __name__ == "__main__":
    # The graph_db driver is initialized on module import, but we need to ensure it's connected
    try:
        asyncio.run(verify_persistence())
    except Exception as e:
        print(f"An error occurred: {e}")
