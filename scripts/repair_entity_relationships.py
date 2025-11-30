"""
Repair missing CONTAINS_ENTITY relationships.

This script finds all Entity nodes that have source_chunks metadata but are missing
CONTAINS_ENTITY relationships, then creates the missing relationships.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_db import graph_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def repair_entity_relationships():
    """Create missing CONTAINS_ENTITY relationships using source_chunks metadata."""
    
    logger.info("Starting entity relationship repair...")
    
    # Find all entities with source_chunks
    query = """
    MATCH (e:Entity)
    WHERE e.source_chunks IS NOT NULL AND size(e.source_chunks) > 0
    RETURN e.id as entity_id, e.name as entity_name, e.source_chunks as chunk_ids
    """
    
    with graph_db.session_scope() as session:
        result = session.run(query)
        entities = list(result)
    
    logger.info(f"Found {len(entities)} entities with source_chunks metadata")
    
    if not entities:
        logger.info("No entities found with source_chunks. Nothing to repair.")
        return
    
    # For each entity, create CONTAINS_ENTITY relationships for all source chunks
    total_relationships_created = 0
    total_relationships_skipped = 0
    
    for entity in entities:
        entity_id = entity["entity_id"]
        entity_name = entity["entity_name"]
        chunk_ids = entity["chunk_ids"]
        
        logger.debug(f"Processing entity {entity_name} ({entity_id}) with {len(chunk_ids)} chunks")
        
        for chunk_id in chunk_ids:
            try:
                # Check if relationship already exists
                check_query = """
                MATCH (c:Chunk {id: $chunk_id})-[r:CONTAINS_ENTITY]->(e:Entity {id: $entity_id})
                RETURN count(r) as rel_count
                """
                with graph_db.session_scope() as session:
                    result = session.run(check_query, chunk_id=chunk_id, entity_id=entity_id)
                    rel_count = result.single()["rel_count"]
                
                if rel_count > 0:
                    logger.debug(f"Relationship already exists: {chunk_id} -> {entity_id}")
                    total_relationships_skipped += 1
                    continue
                
                # Create the relationship
                create_query = """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:CONTAINS_ENTITY]->(e)
                """
                with graph_db.session_scope() as session:
                    session.run(create_query, chunk_id=chunk_id, entity_id=entity_id)
                
                logger.debug(f"Created relationship: {chunk_id} -> {entity_id}")
                total_relationships_created += 1
                
            except Exception as e:
                logger.error(f"Failed to create relationship {chunk_id} -> {entity_id}: {e}")
    
    logger.info(f"Repair complete!")
    logger.info(f"  - Relationships created: {total_relationships_created}")
    logger.info(f"  - Relationships skipped (already exist): {total_relationships_skipped}")
    
    # Verify the repair
    with graph_db.session_scope() as session:
        result = session.run("MATCH ()-[r:CONTAINS_ENTITY]->() RETURN count(r) as total")
        total_rels = result.single()["total"]
    
    logger.info(f"Total CONTAINS_ENTITY relationships in database: {total_rels}")


if __name__ == "__main__":
    repair_entity_relationships()
