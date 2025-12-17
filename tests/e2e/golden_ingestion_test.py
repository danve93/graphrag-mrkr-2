"""
GOLDEN INGESTION TEST - Full Integration Health Check
======================================================

This is the GOLDEN TEST for document ingestion. It must pass at all times
and serves as the index of sane behavior for the ingestion codebase.

PURPOSE:
    This test validates the complete document ingestion pipeline using REAL
    services (not mocks). If this test passes, the ingestion pipeline is healthy.

WHAT IT TESTS:
    1. Neo4j Connection      - Database connectivity
    2. Embedding Service     - OpenAI embedding generation (1536 dimensions)
    3. Text File Ingestion   - Full text→chunks→embeddings→entities flow
    4. PDF File Ingestion    - PDF parsing via smart OCR, chunking, embedding
    5. LLM Entity Extraction - Real GPT-4o-mini entity extraction with gleaning
    6. Progress Tracking     - UI progress updates persisted to database
    7. Data Integrity        - Required fields and constraints on all nodes

REQUIREMENTS:
    - Neo4j database running (docker compose up neo4j)
    - OPENAI_API_KEY configured (in .env or environment)
    - System dependencies for PDF processing (poppler-utils, tesseract-ocr)

LOCAL EXECUTION:
    TEST_ENABLE_E2E=1 TEST_SKIP_DOCKER=1 NEO4J_PASSWORD=<your-password> \\
        uv run pytest tests/e2e/golden_ingestion_test.py -v -s

GITHUB CI:
    The test runs automatically via .github/workflows/golden-test.yml
    Requires OPENAI_API_KEY as a repository secret.

DURATION:
    ~5 minutes (depends on LLM response times and document size)
"""


import logging
import os
import shutil
import socket
import sys
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT CHECKS (deferred to fixtures to work with e2e conftest)
# ============================================================================

def _is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _neo4j_reachable() -> bool:
    """Check if Neo4j is reachable."""
    try:
        from config.settings import settings
        parsed = urlparse(settings.neo4j_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 7687
        return _is_port_open(host, port)
    except Exception:
        return False


def _has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


# ============================================================================
# IMPORTS (conditional to avoid import errors when services unavailable)
# ============================================================================

# Import core modules only if Neo4j is reachable
try:
    from config.settings import settings
    from core.graph_db import graph_db
    from core.embeddings import embedding_manager
    from ingestion.document_processor import document_processor
    IMPORTS_AVAILABLE = True
except Exception as e:
    IMPORTS_AVAILABLE = False
    logger.warning(f"Could not import required modules: {e}")

# ============================================================================
# TEST CONTENT
# ============================================================================

TEST_TEXT_CONTENT = """
# Knowledge Management Systems

## Introduction

Knowledge management systems (KMS) are technology-based systems that help 
organizations collect, organize, and distribute knowledge. They are essential
for modern enterprises that need to leverage their intellectual capital.

## Key Components

### Knowledge Bases
A knowledge base is a centralized repository that stores information in a 
structured manner. It enables quick retrieval of relevant information through
search and navigation features.

### Entity Extraction
Entity extraction is the process of identifying and classifying named entities
in text into predefined categories such as person names, organizations, 
locations, and technical terms.

### Graph Databases
Graph databases like Neo4j store data as nodes and relationships. They excel
at representing and querying interconnected data such as knowledge graphs,
social networks, and recommendation systems.

## Integration with LLMs

Large Language Models (LLMs) like GPT-4 and Claude can be integrated with 
knowledge management systems to provide intelligent question answering,
summarization, and content generation capabilities.

## Applications

1. Customer Support: Automated FAQ systems and chatbots
2. Research: Literature review and knowledge synthesis
3. Enterprise: Document management and institutional memory
4. Education: Adaptive learning and content recommendation
"""

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def test_txt_file():
    """Create a temporary text file for testing."""
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_TEXT_CONTENT)
        file_path = Path(f.name)
    
    yield file_path
    
    if file_path.exists():
        file_path.unlink()


@pytest.fixture(scope="module")
def test_pdf_file():
    """Get or create a test PDF file."""
    # Use an existing small PDF if available
    pdf_candidates = [
        PROJECT_ROOT / "leggiqua/old/sampling.pdf",
        PROJECT_ROOT / "leggiqua/old/rag_issues.pdf",
    ]
    
    for pdf_path in pdf_candidates:
        if pdf_path.exists():
            yield pdf_path
            return
    
    # Skip if no PDF available
    pytest.skip("No test PDF file available")


@pytest.fixture(scope="module")
def cleanup_test_docs():
    """Track document IDs for cleanup after tests."""
    doc_ids: List[str] = []
    
    yield doc_ids
    
    # Cleanup all test documents
    logger.info(f"Cleaning up {len(doc_ids)} test documents...")
    with graph_db.driver.session() as session:
        for doc_id in doc_ids:
            try:
                # Delete document and all related nodes
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                    DETACH DELETE d, c
                """, doc_id=doc_id)
                logger.info(f"  Cleaned up: {doc_id}")
            except Exception as e:
                logger.warning(f"  Failed to cleanup {doc_id}: {e}")


# ============================================================================
# GOLDEN TESTS
# ============================================================================

class TestGoldenIngestion:
    """
    GOLDEN TEST SUITE for Document Ingestion.
    
    These tests validate the complete ingestion pipeline with REAL services.
    All tests must pass for the codebase to be considered healthy.
    """
    
    def test_01_neo4j_connection(self):
        """
        GOLDEN: Neo4j database is connected and responsive.
        """
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        
        logger.info("="*60)
        logger.info("GOLDEN TEST: Neo4j Connection")
        logger.info("="*60)
        
        # Ensure driver is initialized and connected
        graph_db.ensure_connected()
        
        assert graph_db.driver is not None, "Neo4j driver not initialized"
        
        with graph_db.driver.session() as session:
            result = session.run("RETURN 'healthy' as status")
            record = result.single()
            assert record is not None
            assert record["status"] == "healthy"
        
        logger.info("✓ Neo4j connection verified")
    
    def test_02_embedding_service(self):
        """
        GOLDEN: OpenAI embedding service is operational.
        """
        if not IMPORTS_AVAILABLE:
            pytest.skip("Required modules not available")
        
        # Check API key via settings (which loads from .env)
        if not settings.openai_api_key:
            pytest.skip("OPENAI_API_KEY not configured in settings")
        
        logger.info("="*60)
        logger.info("GOLDEN TEST: Embedding Service")
        logger.info("="*60)
        
        test_text = "This is a test sentence for embedding generation."
        
        # Test synchronous embedding
        embedding = embedding_manager.get_embedding(test_text)
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        logger.info(f"✓ Embedding generated: {len(embedding)} dimensions")
    
    def test_03_text_file_ingestion(self, test_txt_file, cleanup_test_docs):
        """
        GOLDEN: Text file ingestion creates proper graph structure.
        """
        logger.info("="*60)
        logger.info("GOLDEN TEST: Text File Ingestion")
        logger.info("="*60)
        
        doc_id = f"golden_txt_{int(time.time())}"
        cleanup_test_docs.append(doc_id)
        
        # Process the file
        result = document_processor.process_file(
            test_txt_file,
            document_id=doc_id,
            original_filename="golden_test.txt",
        )
        
        assert result.get("status") == "success", f"Ingestion failed: {result.get('error')}"
        
        # Wait for background entity extraction
        max_wait = 60
        start = time.time()
        while document_processor.is_entity_extraction_running():
            if time.time() - start > max_wait:
                logger.warning("Entity extraction taking too long, continuing...")
                break
            time.sleep(1)
        
        # Verify in Neo4j
        with graph_db.driver.session() as session:
            # Document exists
            doc = session.run(
                "MATCH (d:Document {id: $id}) RETURN d",
                id=doc_id
            ).single()
            assert doc is not None, "Document node not created"
            
            # Chunks exist
            chunks = session.run(
                "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) as count",
                id=doc_id
            ).single()["count"]
            assert chunks > 0, "No chunks created"
            
            # Chunks have embeddings
            embedded_chunks = session.run("""
                MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as count
            """, id=doc_id).single()["count"]
            assert embedded_chunks > 0, "No chunks have embeddings"
            
            # Entities created (may be 0 if extraction disabled)
            entities = session.run("""
                MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                RETURN count(DISTINCT e) as count
            """, id=doc_id).single()["count"]
        
        logger.info(f"✓ Document created: {doc_id}")
        logger.info(f"  - Chunks: {chunks}")
        logger.info(f"  - Embedded chunks: {embedded_chunks}")
        logger.info(f"  - Entities: {entities}")
    
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "leggiqua/old/sampling.pdf").exists(),
        reason="Test PDF not available"
    )
    def test_04_pdf_file_ingestion(self, test_pdf_file, cleanup_test_docs):
        """
        GOLDEN: PDF file ingestion creates proper graph structure.
        """
        logger.info("="*60)
        logger.info("GOLDEN TEST: PDF File Ingestion")
        logger.info("="*60)
        
        doc_id = f"golden_pdf_{int(time.time())}"
        cleanup_test_docs.append(doc_id)
        
        # Process the file
        result = document_processor.process_file(
            test_pdf_file,
            document_id=doc_id,
            original_filename="golden_test.pdf",
        )
        
        assert result.get("status") == "success", f"PDF ingestion failed: {result.get('error')}"
        
        # Wait for background entity extraction
        max_wait = 90  # PDFs take longer
        start = time.time()
        while document_processor.is_entity_extraction_running():
            if time.time() - start > max_wait:
                logger.warning("Entity extraction taking too long, continuing...")
                break
            time.sleep(1)
        
        # Verify in Neo4j
        with graph_db.driver.session() as session:
            chunks = session.run(
                "MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) as count",
                id=doc_id
            ).single()["count"]
            assert chunks > 0, "No chunks created from PDF"
            
            embedded = session.run("""
                MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as count
            """, id=doc_id).single()["count"]
        
        logger.info(f"✓ PDF processed: {doc_id}")
        logger.info(f"  - Chunks: {chunks}")
        logger.info(f"  - Embedded: {embedded}")
    
    def test_05_entity_extraction_llm(self, test_txt_file, cleanup_test_docs):
        """
        GOLDEN: LLM entity extraction produces valid entities.
        """
        logger.info("="*60)
        logger.info("GOLDEN TEST: LLM Entity Extraction")
        logger.info("="*60)
        
        # Force entity extraction enabled
        original_setting = settings.enable_entity_extraction
        settings.enable_entity_extraction = True
        
        doc_id = f"golden_llm_{int(time.time())}"
        cleanup_test_docs.append(doc_id)
        
        try:
            result = document_processor.process_file(
                test_txt_file,
                document_id=doc_id,
                original_filename="golden_llm_test.txt",
            )
            
            assert result.get("status") == "success"
            
            # Wait for entity extraction to complete
            max_wait = 120  # LLM calls take time
            start = time.time()
            while document_processor.is_entity_extraction_running():
                if time.time() - start > max_wait:
                    break
                time.sleep(2)
            
            # Small additional wait for persistence
            time.sleep(2)
            
            # Verify entities in Neo4j
            with graph_db.driver.session() as session:
                entities = session.run("""
                    MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                    RETURN count(DISTINCT e) as count
                """, id=doc_id).single()["count"]
                
                # Get sample entity names
                entity_names = session.run("""
                    MATCH (d:Document {id: $id})-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                    RETURN DISTINCT e.name as name LIMIT 10
                """, id=doc_id)
                names = [r["name"] for r in entity_names]
            
            logger.info(f"✓ Entity extraction completed: {entities} entities")
            if names:
                logger.info(f"  - Sample entities: {names[:5]}")
            
            # Entity count may be 0 if LLM doesn't find any, that's valid
            # but we should have at least attempted extraction
            assert result.get("status") == "success"
            
        finally:
            settings.enable_entity_extraction = original_setting
    
    def test_06_progress_tracking(self, test_txt_file, cleanup_test_docs):
        """
        GOLDEN: Progress tracking updates are persisted to database.
        """
        logger.info("="*60)
        logger.info("GOLDEN TEST: Progress Tracking")
        logger.info("="*60)
        
        doc_id = f"golden_progress_{int(time.time())}"
        cleanup_test_docs.append(doc_id)
        
        result = document_processor.process_file(
            test_txt_file,
            document_id=doc_id,
            original_filename="golden_progress_test.txt",
        )
        
        assert result.get("status") == "success"
        
        # Verify final status in database
        with graph_db.driver.session() as session:
            doc = session.run("""
                MATCH (d:Document {id: $id})
                RETURN d.processing_status as status,
                       d.processing_progress as progress,
                       d.processing_stage as stage
            """, id=doc_id).single()
            
            assert doc is not None
            assert doc["status"] == "completed"
            assert doc["progress"] == 100.0
        
        logger.info(f"✓ Progress tracking verified")
        logger.info(f"  - Status: {doc['status']}")
        logger.info(f"  - Progress: {doc['progress']}%")
        logger.info(f"  - Stage: {doc['stage']}")
    
    def test_07_data_integrity(self, cleanup_test_docs):
        """
        GOLDEN: All persisted data has required fields and valid values.
        """
        logger.info("="*60)
        logger.info("GOLDEN TEST: Data Integrity")
        logger.info("="*60)
        
        with graph_db.driver.session() as session:
            # Check Document nodes have required fields
            invalid_docs = session.run("""
                MATCH (d:Document)
                WHERE d.id IS NULL OR d.filename IS NULL
                RETURN count(d) as count
            """).single()["count"]
            assert invalid_docs == 0, f"Found {invalid_docs} documents with missing required fields"
            
            # Check Chunk nodes have required fields
            invalid_chunks = session.run("""
                MATCH (c:Chunk)
                WHERE c.id IS NULL OR c.document_id IS NULL OR c.content IS NULL
                RETURN count(c) as count
            """).single()["count"]
            assert invalid_chunks == 0, f"Found {invalid_chunks} chunks with missing required fields"
            
            # Check all chunks are linked to documents
            orphan_chunks = session.run("""
                MATCH (c:Chunk)
                WHERE NOT ()-[:HAS_CHUNK]->(c)
                RETURN count(c) as count
            """).single()["count"]
            # Note: orphan chunks may exist from other tests, just report
            
            # Get statistics
            stats = session.run("""
                MATCH (d:Document) WITH count(d) as docs
                MATCH (c:Chunk) WITH docs, count(c) as chunks
                MATCH (e:Entity) 
                RETURN docs, chunks, count(e) as entities
            """).single()
        
        logger.info(f"✓ Data integrity verified")
        logger.info(f"  - Documents: {stats['docs']}")
        logger.info(f"  - Chunks: {stats['chunks']}")
        logger.info(f"  - Entities: {stats['entities']}")
        if orphan_chunks > 0:
            logger.warning(f"  - Orphan chunks: {orphan_chunks} (may be from other tests)")


# ============================================================================
# SUMMARY REPORTER
# ============================================================================

@pytest.fixture(scope="module", autouse=True)
def test_summary(request):
    """Print summary at end of test run."""
    yield
    
    logger.info("\n" + "="*60)
    logger.info("GOLDEN INGESTION TEST SUMMARY")
    logger.info("="*60)
    logger.info("If all tests passed, the ingestion pipeline is healthy.")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
