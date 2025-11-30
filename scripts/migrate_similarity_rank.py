import logging
from neo4j import GraphDatabase
import os
from core.graph_db import graph_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("migrate_similarity_rank")


def migrate_similarity_rank():
    with graph_db.session_scope() as session:
        # Get all document IDs
        doc_ids = [r["id"] for r in session.run("MATCH (d:Document) RETURN d.id as id")]
        logger.info(f"Found {len(doc_ids)} documents.")
        for doc_id in doc_ids:
            logger.info(f"Processing document {doc_id}")
            # Get all chunk IDs for this document
            chunk_ids = [r["id"] for r in session.run(
                "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN c.id as id", doc_id=doc_id
            )]
            for chunk_id in chunk_ids:
                # Get all SIMILAR_TO relationships for this chunk, sorted by score desc
                rels = list(session.run(
                    "MATCH (c1:Chunk {id: $chunk_id})-[r:SIMILAR_TO]-(c2:Chunk) "
                    "RETURN id(r) as rel_id, r.score as score, c2.id as other_id "
                    "ORDER BY r.score DESC",
                    chunk_id=chunk_id
                ))
                for rank, rel in enumerate(rels):
                    rel_id = rel["rel_id"]
                    session.run(
                        "MATCH ()-[r]-() WHERE id(r) = $rel_id SET r.rank = $rank",
                        rel_id=rel_id,
                        rank=rank
                    )
                if rels:
                    logger.info(f"Updated {len(rels)} SIMILAR_TO relationships for chunk {chunk_id}")
    logger.info("Migration complete.")


if __name__ == "__main__":
    migrate_similarity_rank()
