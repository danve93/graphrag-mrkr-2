"""
Comprehensive end-to-end ingestion pipeline test.

Tests the full document processing flow:
1. Document upload and metadata extraction
2. Chunking and quality filtering
3. Embedding generation
4. Entity extraction
5. Similarity relationship creation
6. Auto-clustering and community assignment
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from core.graph_db import graph_db
from core.graph_clustering import run_auto_clustering
from ingestion.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_document_path():
    """Path to test document."""
    test_file = PROJECT_ROOT / "data" / "staged_uploads" / "137f69206b4dac08_README.md"
    if not test_file.exists():
        pytest.skip(f"Test document not found: {test_file}")
    return test_file


@pytest.fixture(scope="module")
def document_processor():
    """Initialize document processor."""
    return DocumentProcessor()


@pytest.fixture(scope="module")
def cleanup_test_data():
    """Cleanup any test data after tests complete."""
    yield
    # Cleanup happens after all tests
    logger.info("Test suite completed")


def test_neo4j_connection():
    """Verify Neo4j database connectivity."""
    logger.info("Testing Neo4j connection...")
    try:
        # Test connection by running a simple query
        with graph_db.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            assert result.single()["test"] == 1
        logger.info("✓ Neo4j connection verified")
    except Exception as e:
        pytest.fail(f"Neo4j connection test failed: {e}")


def test_settings_validation():
    """Verify critical settings are configured."""
    logger.info("Validating settings...")
    
    assert settings.neo4j_uri is not None, "NEO4J_URI not configured"
    assert settings.neo4j_username is not None, "NEO4J_USERNAME not configured"
    assert settings.neo4j_password is not None, "NEO4J_PASSWORD not configured"
    
    logger.info(f"✓ LLM provider: {settings.llm_provider}")
    logger.info(f"✓ Entity extraction: {settings.enable_entity_extraction}")
    logger.info(f"✓ Quality filtering: {settings.enable_quality_filtering}")
    logger.info(f"✓ Clustering: {settings.enable_clustering}")


def test_document_metadata_extraction(document_processor, test_document_path):
    """Test document metadata and ID generation."""
    logger.info("Testing metadata extraction...")
    
    doc_id = document_processor.compute_document_id(test_document_path)
    assert doc_id is not None, "Document ID generation failed"
    assert len(doc_id) == 32, f"Invalid document ID format: {doc_id}"
    
    metadata = document_processor.build_metadata(test_document_path)
    assert "file_path" in metadata
    assert "original_filename" in metadata or "file_name" in metadata
    # Metadata doesn't include document_id directly - it's computed separately
    logger.info(f"  Metadata keys: {list(metadata.keys())}")
    
    logger.info(f"✓ Document ID: {doc_id}")
    logger.info(f"✓ Metadata extracted: {len(metadata)} fields")


def test_full_ingestion_pipeline(document_processor, test_document_path, cleanup_test_data):
    """
    End-to-end ingestion pipeline test.
    
    Validates:
    - Document processing
    - Chunk creation and storage
    - Embedding generation
    - Entity extraction
    - Similarity relationships
    - Community clustering
    """
    logger.info("\n" + "="*80)
    logger.info("FULL INGESTION PIPELINE TEST")
    logger.info("="*80)
    
    # Step 1: Process document
    logger.info("\n[1/6] Processing document...")
    start_time = time.time()
    
    # Ensure local Neo4j URI during test (avoid docker service hostname)
    os.environ["NEO4J_URI"] = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    result = document_processor.process_file(test_document_path)
    
    process_time = time.time() - start_time
    logger.info(f"✓ Document processed in {process_time:.2f}s")
    
    assert result["status"] == "success", f"Processing failed: {result.get('error', 'Unknown error')}"
    doc_id = result["document_id"]
    chunks_created = result["chunks_created"]
    
    logger.info(f"  - Document ID: {doc_id}")
    logger.info(f"  - Chunks created: {chunks_created}")
    
    assert chunks_created > 0, "No chunks were created"
    
    # Step 2: Verify chunks in database
    logger.info("\n[2/6] Verifying chunks in database...")
    chunks = graph_db.get_document_chunks(doc_id)
    
    assert len(chunks) == chunks_created, f"Chunk count mismatch: expected {chunks_created}, got {len(chunks)}"
    logger.info(f"✓ All {len(chunks)} chunks verified in Neo4j")
    
    # Verify embeddings
    chunks_with_embeddings = sum(1 for c in chunks if c.get("embedding") is not None)
    logger.info(f"  - Chunks with embeddings: {chunks_with_embeddings}/{len(chunks)}")
    
    if settings.enable_entity_extraction:
        # Step 3: Extract entities synchronously
        logger.info("\n[3/6] Extracting entities synchronously...")
        entity_start = time.time()
        
        entity_result = document_processor.extract_entities_for_document(
            doc_id=doc_id,
            file_name=test_document_path.name
        )
        
        entity_time = time.time() - entity_start
        logger.info(f"✓ Entity extraction completed in {entity_time:.2f}s")
        logger.info(f"  - Status: {entity_result.get('status')}")
        logger.info(f"  - Entities created: {entity_result.get('entities_created', 0)}")
        logger.info(f"  - Relationships created: {entity_result.get('relationships_created', 0)}")
        
        assert entity_result["status"] == "success", f"Entity extraction failed: {entity_result.get('error', 'Unknown error')}"
        
        # Step 4: Verify entities in database
        logger.info("\n[4/6] Verifying entities in database...")
        
        # Query entities linked to this document
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN DISTINCT e.id AS entity_id, e.label AS label, e.type AS type, e.community_id AS community_id
        LIMIT 100
        """
        
        with graph_db.driver.session() as session:
            result_records = session.run(query, doc_id=doc_id)
            entities = [record.data() for record in result_records]
        
        logger.info(f"✓ Found {len(entities)} entities linked to document")
        
        if len(entities) > 0:
            # Show sample entities
            logger.info(f"  Sample entities:")
            for i, entity in enumerate(entities[:5], 1):
                logger.info(f"    {i}. {entity.get('label', 'N/A')} ({entity.get('type', 'N/A')}) - Community: {entity.get('community_id', 'None')}")
            
            # Step 5: Verify similarity relationships
            logger.info("\n[5/6] Verifying similarity relationships...")
            
            similarity_query = """
            MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e1:Entity)
            MATCH (e1)-[r:SIMILAR_TO]->(e2:Entity)
            RETURN count(r) AS similarity_count
            """
            
            with graph_db.driver.session() as session:
                sim_result = session.run(similarity_query, doc_id=doc_id)
                sim_record = sim_result.single()
                similarity_count = sim_record["similarity_count"] if sim_record else 0
            
            logger.info(f"✓ Found {similarity_count} similarity relationships")
            
            # Step 6: Verify clustering
            logger.info("\n[6/6] Verifying community clustering...")
            
            entities_with_community = sum(1 for e in entities if e.get("community_id") is not None)
            community_assignment_rate = (entities_with_community / len(entities) * 100) if len(entities) > 0 else 0
            
            logger.info(f"✓ Entities with community: {entities_with_community}/{len(entities)} ({community_assignment_rate:.1f}%)")
            
            if entities_with_community == 0:
                logger.warning("⚠ No entities have community assignments - clustering may not have run")
                
                # Try manual clustering
                logger.info("  Attempting manual clustering...")
                clustering_result = run_auto_clustering(graph_db.driver)
                logger.info(f"  Clustering result: {clustering_result}")
                
                # Re-query after manual clustering
                with graph_db.driver.session() as session:
                    result_records = session.run(query, doc_id=doc_id)
                    entities_after_clustering = [record.data() for record in result_records]
                
                entities_with_community_after = sum(1 for e in entities_after_clustering if e.get("community_id") is not None)
                logger.info(f"  After manual clustering: {entities_with_community_after}/{len(entities_after_clustering)} entities have communities")
            
        else:
            logger.warning("⚠ No entities found - entity extraction may have failed")
            pytest.fail("Entity extraction produced no entities")
    
    else:
        logger.info("\n[3-6] Entity extraction disabled in settings - skipping entity verification")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("="*80)
    logger.info(f"✓ Document processed: {doc_id}")
    logger.info(f"✓ Chunks created: {chunks_created}")
    logger.info(f"✓ Chunks with embeddings: {chunks_with_embeddings}/{chunks_created}")
    
    if settings.enable_entity_extraction:
        logger.info(f"✓ Entities extracted: {len(entities)}")
        logger.info(f"✓ Similarity relationships: {similarity_count}")
        logger.info(f"✓ Community assignment rate: {community_assignment_rate:.1f}%")
    
    logger.info("="*80 + "\n")


def test_clustering_algorithm():
    """Test clustering algorithm in isolation."""
    logger.info("\n" + "="*80)
    logger.info("CLUSTERING ALGORITHM TEST")
    logger.info("="*80)
    
    logger.info("Running clustering on current graph state...")
    result = run_auto_clustering(graph_db.driver)
    
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Communities found: {result.get('communities_count', 0)}")
    logger.info(f"Modularity: {result.get('modularity', 0.0):.4f}")
    logger.info(f"Nodes updated: {result.get('updated_nodes', 0)}")
    
    if result.get("status") == "success":
        logger.info("✓ Clustering algorithm executed successfully")
    else:
        logger.warning(f"⚠ Clustering status: {result.get('status')}")
    
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])
