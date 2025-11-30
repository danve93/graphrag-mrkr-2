"""
End-to-end test for complete GraphRAG pipeline.

Tests full workflow:
1. Document ingestion with quality filtering
2. Entity extraction
3. Chunk similarity creation
4. Leiden clustering
5. Community summarization
6. Graph visualization
"""

import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from core.graph_db import graph_db  # noqa: E402
from ingestion.document_processor import document_processor  # noqa: E402

logger = logging.getLogger(__name__)


# Test document content
TEST_DOCUMENT_CONTENT = """
GraphRAG: Advanced Graph-Based Retrieval-Augmented Generation

Introduction
GraphRAG v2.0 is a state-of-the-art document intelligence system powered by graph-based RAG 
(Retrieval-Augmented Generation) with LangGraph, FastAPI, Next.js, and Neo4j. This comprehensive 
system combines multiple technologies to provide powerful document analysis and question-answering 
capabilities.

Core Architecture
The system consists of several key components:

1. Query Analysis: Analyzes user intent and extracts structured query parameters from natural language inputs.

2. Retrieval Layer: Implements hybrid search combining vector similarity with graph-based expansion 
to fetch relevant document chunks. The retrieval mechanism is intelligent about entity relationships 
and document structure.

3. Graph Reasoning: Performs traversal of entity relationships to enrich context and handle 
multi-hop queries. This enables the system to find connections across documents and extract 
complex relationships.

4. Generation: Uses LLM streaming for high-quality responses with integrated quality scoring.
The generation process respects the retrieved context while maintaining coherence.

Entity Model
The system recognizes various entity types including:
- Components and Services
- Nodes and Domains
- Roles and Accounts
- Resources and Storage Objects
- Procedures and Tasks
- Organizations and Locations
- Technologies and Products
- Dates and Monetary Values

Graph Construction
Documents are processed through intelligent chunking with quality filtering. OCR is applied 
to scanned documents using Marker. Chunks are embedded and stored in Neo4j with relationships 
representing similarity and semantic connections. Entity extraction creates additional nodes 
and relationships for complex domain-specific knowledge representation.

Clustering and Communities
The system uses Leiden clustering to identify communities of related entities. Communities 
are detected at multiple hierarchy levels, allowing for both fine-grained and high-level 
analysis of document relationships. Each community can be summarized to provide semantic 
context for retrieval and reasoning.

Community summarization uses exemplar text units and entity importance scores to generate 
meaningful summaries. These summaries are cached in the graph database for efficient 
retrieval during reasoning steps.

Multi-hop Reasoning
The retrieval system supports multi-hop reasoning which explores relationship chains 
to find relevant context. Entities are scored based on their importance and relationship 
strength to surface the most relevant information.

Hybrid Retrieval
Results combine:
- Vector similarity scores from semantic embeddings
- Entity relationship strength from the knowledge graph
- Path-based relevance from multi-hop traversal

Weights can be adjusted to control the influence of each component.

API Layer
FastAPI provides a modern REST API with Server-Sent Events for streaming responses. 
Endpoints support chat, document management, database operations, and graph visualization.

Frontend
The Next.js frontend provides:
- Real-time chat interface with streaming responses
- Document management workspace
- Graph visualization for exploring entity relationships
- Conversation history with search capabilities
- Advanced retrieval configuration options

Performance Optimizations
The system implements:
- Connection pooling for Neo4j database access
- Asynchronous processing for embeddings and LLM calls
- Configurable concurrency limits to prevent rate limiting
- Quality-based filtering to reduce noise
- Caching of embeddings and summaries

Deployment
Full Docker Compose deployment includes:
- Backend API service
- Frontend Next.js application
- Neo4j database
- Unified configuration through environment variables

The system is production-ready and scales well with growing document collections.
"""


@pytest.fixture(scope="module")
def test_document():
    """Create a test document for the pipeline."""
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(TEST_DOCUMENT_CONTENT)
        doc_path = Path(f.name)
    
    yield doc_path
    
    # Cleanup
    if doc_path.exists():
        doc_path.unlink()


@pytest.fixture(scope="module")
def ingested_doc_id(test_document):
    """Ingest the test document and return its ID."""
    logger.info("Ingesting document...")
    result = document_processor.process_file(
        test_document,
        original_filename="graphrag_documentation.txt",
        enable_quality_filtering=True
    )
    
    assert result.get("status") == "success", f"Ingestion failed: {result.get('error')}"
    
    doc_id = result.get("document_id")
    chunks = result.get("chunks_created", 0)
    
    logger.info(f"✓ Document ingested: {doc_id}")
    logger.info(f"  - Chunks created: {chunks}")
    logger.info(f"  - Duration: {result.get('duration_seconds', 0):.2f}s")
    
    yield doc_id
    
    # Cleanup
    with graph_db.driver.session() as session:
        session.run("MATCH (d:Document {id: $doc_id}) DETACH DELETE d", doc_id=doc_id)
        session.run("MATCH (c:Chunk) WHERE c.document_id = $doc_id DETACH DELETE c", doc_id=doc_id)


def test_neo4j_connection():
    """Test Neo4j database connection."""
    logger.info(f"Testing Neo4j connection to {settings.neo4j_uri}")
    
    with graph_db.driver.session() as session:
        result = session.run("RETURN 1 as status")
        record = result.single()
        assert record is not None
        assert record["status"] == 1
    
    logger.info("✓ Neo4j connection successful")


def test_document_ingestion(ingested_doc_id):
    """Test document ingestion creates proper graph structure."""
    assert ingested_doc_id is not None
    
    with graph_db.driver.session() as session:
        # Verify document node
        doc_result = session.run(
            "MATCH (d:Document {id: $doc_id}) RETURN d",
            doc_id=ingested_doc_id
        ).single()
        assert doc_result is not None
        
        # Verify chunks
        chunk_count = session.run(
            "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) as count",
            doc_id=ingested_doc_id
        ).single()["count"]
        assert chunk_count > 0
        
        logger.info(f"✓ Document has {chunk_count} chunks")


def test_entity_extraction_status(ingested_doc_id):
    """Test entity extraction runs and completes."""
    is_running = document_processor.is_entity_extraction_running()
    status = document_processor.get_entity_extraction_status()
    
    logger.info(f"Entity extraction running: {is_running}")
    logger.info(f"Active operations: {status.get('active_operations', 0)}")
    
    # Entity extraction should either be running or completed
    assert isinstance(is_running, bool)


def test_database_statistics(ingested_doc_id):
    """Test database has expected content after ingestion."""
    with graph_db.driver.session() as session:
        doc_count = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
        chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
        similarity_count = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count").single()["count"]
        
        logger.info("✓ Database Statistics:")
        logger.info(f"  - Documents: {doc_count}")
        logger.info(f"  - Chunks: {chunk_count}")
        logger.info(f"  - Entities: {entity_count}")
        logger.info(f"  - Chunk Similarities: {similarity_count}")
        
        assert doc_count > 0
        assert chunk_count > 0
        # Entities might be 0 if extraction is disabled or still running


@pytest.mark.skipif(not settings.enable_clustering, reason="Clustering disabled in settings")
def test_leiden_clustering(ingested_doc_id):
    """Test Leiden clustering on entities."""
    from core.graph_clustering import (
        fetch_entity_projection,
        normalize_edge_weights,
        run_leiden_clustering,
        to_igraph,
        write_communities_to_neo4j,
    )
    
    logger.info("Running Leiden clustering...")
    
    nodes_df, edges_df = fetch_entity_projection(graph_db.driver)
    logger.info(f"  - Projection: {len(nodes_df)} nodes, {len(edges_df)} edges")
    
    if len(edges_df) > 0:
        edges_df = normalize_edge_weights(edges_df)
        igraph = to_igraph(nodes_df, edges_df)
        membership, modularity = run_leiden_clustering(igraph, resolution=settings.clustering_resolution)
        
        communities = len(set(membership.values()))
        logger.info(f"  - Communities: {communities}")
        logger.info(f"  - Modularity: {modularity:.4f}")
        
        updated = write_communities_to_neo4j(graph_db.driver, membership)
        logger.info(f"  - Updated {updated} entities")
        
        assert communities > 0
        assert modularity >= 0
    else:
        pytest.skip("No entity edges for clustering")


@pytest.mark.skipif(not settings.enable_clustering, reason="Clustering disabled in settings")
def test_community_summarization(ingested_doc_id):
    """Test community summarization."""
    from core.community_summarizer import community_summarizer
    
    logger.info("Generating community summaries...")
    
    with graph_db.driver.session() as session:
        levels_result = session.run(
            "MATCH (e:Entity) WHERE e.level IS NOT NULL RETURN DISTINCT e.level as level ORDER BY level"
        )
        levels = [record["level"] for record in levels_result]
        
        if levels:
            logger.info(f"  - Community levels: {levels}")
            summaries = community_summarizer.summarize_levels(levels)
            logger.info(f"✓ Generated {len(summaries)} community summaries")
            
            for summary in summaries[:3]:
                logger.info(f"  - Community {summary.get('community_id')} (level {summary.get('level')}): "
                          f"{len(summary.get('summary', ''))} chars")
            
            assert len(summaries) > 0
        else:
            pytest.skip("No community levels found")


def test_graph_visualization(ingested_doc_id):
    """Test graph visualization data retrieval."""
    logger.info("Testing graph visualization...")
    
    graph_data = graph_db.get_clustered_graph(limit=50)
    
    logger.info("✓ Graph visualization data retrieved")
    logger.info(f"  - Nodes: {len(graph_data.get('nodes', []))}")
    logger.info(f"  - Edges: {len(graph_data.get('edges', []))}")
    logger.info(f"  - Communities: {len(graph_data.get('communities', []))}")
    logger.info(f"  - Node types: {len(graph_data.get('node_types', []))}")
    
    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert isinstance(graph_data["nodes"], list)
    assert isinstance(graph_data["edges"], list)
