import sys
import os
import asyncio
import logging

# Add project root to sys.path
sys.path.append(os.getcwd())

from core.graph_db import graph_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup_database():
    """
    Clean up orphaned nodes from the database.
    This script deletes:
    1. Chunks that are not connected to any Document.
    2. Entities that are not connected to any Chunk.
    """
    if not input("Are you sure you want to delete orphaned chunks and entities? (y/n): ").lower().startswith('y'):
        print("Aborted.")
        return

    logger.info("Starting database cleanup...")
    
    with graph_db.session_scope() as session:
        # 1. Delete orphaned chunks
        result = session.run("""
            MATCH (c:Chunk)
            WHERE NOT (:Document)-[:HAS_CHUNK]->(c)
            OPTIONAL MATCH (c)-[r]-()
            DELETE r, c
            RETURN count(c) as deleted
        """)
        deleted_chunks = result.single()["deleted"]
        logger.info(f"Deleted {deleted_chunks} orphaned chunks")
        
        # 2. Delete orphaned entities
        result = session.run("""
            MATCH (e:Entity)
            WHERE NOT (:Chunk)-[:CONTAINS_ENTITY]->(e)
            DETACH DELETE e
            RETURN count(e) as deleted
        """)
        deleted_entities = result.single()["deleted"]
        logger.info(f"Deleted {deleted_entities} orphaned entities")
        
    logger.info("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(cleanup_database())
