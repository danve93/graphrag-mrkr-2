#!/usr/bin/env python3
"""
Migration script to add temporal nodes for existing documents.

This script creates temporal nodes (Date, Month, Quarter, Year) for all
existing documents that don't have them yet, based on their created_at timestamp.

Usage:
    python scripts/add_temporal_nodes.py [--dry-run] [--batch-size 100]

Options:
    --dry-run       Show what would be done without making changes
    --batch-size    Number of documents to process in each batch (default: 100)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.graph_db import GraphDB
from core.singletons import get_graph_db_driver
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_documents_without_temporal_nodes(graph_db: GraphDB, limit: int = None) -> list:
    """Get documents that don't have temporal nodes yet.

    Args:
        graph_db: GraphDB instance
        limit: Maximum number of documents to return

    Returns:
        List of document IDs
    """
    query = """
    MATCH (d:Document)
    WHERE NOT (d)-[:CREATED_AT]->(:Date)
      AND d.created_at IS NOT NULL
    RETURN d.id AS doc_id, d.created_at AS created_at
    """
    if limit:
        query += f" LIMIT {limit}"

    with graph_db.session_scope() as session:
        result = session.run(query)
        return [dict(record) for record in result]


def get_total_documents(graph_db: GraphDB) -> int:
    """Get total number of documents in the database.

    Args:
        graph_db: GraphDB instance

    Returns:
        Total document count
    """
    with graph_db.session_scope() as session:
        result = session.run("MATCH (d:Document) RETURN count(d) AS total")
        record = result.single()
        return record["total"] if record else 0


def get_documents_with_temporal_nodes(graph_db: GraphDB) -> int:
    """Get count of documents that already have temporal nodes.

    Args:
        graph_db: GraphDB instance

    Returns:
        Count of documents with temporal nodes
    """
    with graph_db.session_scope() as session:
        result = session.run("""
            MATCH (d:Document)-[:CREATED_AT]->(:Date)
            RETURN count(d) AS total
        """)
        record = result.single()
        return record["total"] if record else 0


def migrate_documents(
    graph_db: GraphDB,
    batch_size: int = 100,
    dry_run: bool = False
) -> dict:
    """Migrate documents to have temporal nodes.

    Args:
        graph_db: GraphDB instance
        batch_size: Number of documents to process per batch
        dry_run: If True, only show what would be done

    Returns:
        Dictionary with migration statistics
    """
    logger.info("Starting temporal nodes migration")

    # Get statistics
    total_docs = get_total_documents(graph_db)
    docs_with_temporal = get_documents_with_temporal_nodes(graph_db)
    docs_without_temporal_list = get_documents_without_temporal_nodes(graph_db)
    docs_without_temporal = len(docs_without_temporal_list)

    logger.info(f"Total documents: {total_docs}")
    logger.info(f"Documents with temporal nodes: {docs_with_temporal}")
    logger.info(f"Documents without temporal nodes: {docs_without_temporal}")

    if dry_run:
        logger.info("DRY RUN: No changes will be made")
        if docs_without_temporal > 0:
            logger.info(f"Would create temporal nodes for {docs_without_temporal} documents")
            # Show first 5 as examples
            for i, doc in enumerate(docs_without_temporal_list[:5]):
                logger.info(f"  Example {i+1}: {doc['doc_id']} (created_at: {doc['created_at']})")
        return {
            "total_documents": total_docs,
            "already_migrated": docs_with_temporal,
            "to_migrate": docs_without_temporal,
            "migrated": 0,
            "failed": 0,
            "dry_run": True
        }

    # Process documents in batches
    migrated = 0
    failed = 0
    failed_docs = []

    logger.info(f"Processing {docs_without_temporal} documents in batches of {batch_size}")

    for i in range(0, len(docs_without_temporal_list), batch_size):
        batch = docs_without_temporal_list[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} ({len(batch)} documents)")

        for doc in batch:
            doc_id = doc['doc_id']
            created_at = doc['created_at']

            try:
                graph_db.create_temporal_nodes_for_document(doc_id, timestamp=created_at)
                migrated += 1

                if migrated % 10 == 0:
                    logger.info(f"Progress: {migrated}/{docs_without_temporal} documents migrated")

            except Exception as e:
                logger.error(f"Failed to create temporal nodes for {doc_id}: {e}")
                failed += 1
                failed_docs.append({"doc_id": doc_id, "error": str(e)})

    logger.info(f"Migration complete!")
    logger.info(f"  Successfully migrated: {migrated}")
    logger.info(f"  Failed: {failed}")

    if failed_docs:
        logger.warning(f"Failed documents ({len(failed_docs)}):")
        for failed in failed_docs[:10]:  # Show first 10 failures
            logger.warning(f"  - {failed['doc_id']}: {failed['error']}")

    return {
        "total_documents": total_docs,
        "already_migrated": docs_with_temporal,
        "to_migrate": docs_without_temporal,
        "migrated": migrated,
        "failed": failed,
        "failed_docs": failed_docs,
        "dry_run": False
    }


def main():
    """Main migration script entry point."""
    parser = argparse.ArgumentParser(
        description="Add temporal nodes to existing documents"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process per batch (default: 100)"
    )
    args = parser.parse_args()

    # Initialize GraphDB
    logger.info("Connecting to Neo4j...")
    try:
        graph_db = GraphDB()
        graph_db.connect()
        logger.info("Connected to Neo4j successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Verify temporal filtering is enabled in settings
    if not settings.enable_temporal_filtering:
        logger.warning(
            "WARNING: enable_temporal_filtering is False in settings. "
            "Temporal nodes will be created but not used during ingestion. "
            "Set enable_temporal_filtering=True to use this feature."
        )

    # Run migration
    try:
        results = migrate_documents(
            graph_db,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )

        # Print summary
        print("\n" + "="*60)
        print("MIGRATION SUMMARY")
        print("="*60)
        print(f"Total documents:          {results['total_documents']}")
        print(f"Already had temporal:     {results['already_migrated']}")
        print(f"Needed migration:         {results['to_migrate']}")
        if not args.dry_run:
            print(f"Successfully migrated:    {results['migrated']}")
            print(f"Failed:                   {results['failed']}")
            if results['failed'] > 0:
                print(f"\nSee logs above for details on failed documents")
        print("="*60)

        if results['failed'] > 0:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if graph_db.driver:
            graph_db.close()
            logger.info("Closed Neo4j connection")


if __name__ == "__main__":
    main()
