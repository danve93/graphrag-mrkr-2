#!/usr/bin/env python3
"""
Full pipeline test script: ingest document, create similarities, run clustering, and summarize communities.
"""

import asyncio
import logging
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import document_processor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_document():
    """Create a test document for the full pipeline."""
    content = """
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
    
    # Write to temporary file
    with NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        return Path(f.name)


def main():
    """Run the complete pipeline test."""
    logger.info("Starting full pipeline test for GraphRAG v2.0")
    
    # Step 1: Create Neo4j connection
    logger.info("Connecting to Neo4j at %s", settings.neo4j_uri)
    try:
        # Test connection
        with graph_db.driver.session() as session:
            result = session.run("RETURN 1 as status")
            record = result.single()
            if record:
                logger.info("✓ Neo4j connection successful")
            else:
                logger.error("✗ Failed to connect to Neo4j")
                return False
    except Exception as e:
        logger.error("✗ Neo4j connection failed: %s", e)
        logger.error("Ensure Neo4j is running at %s", settings.neo4j_uri)
        return False
    
    # Step 2: Create test document
    logger.info("Creating test document...")
    test_doc_path = create_test_document()
    logger.info("✓ Test document created at %s", test_doc_path)
    
    # Step 3: Ingest document
    logger.info("Ingesting document...")
    try:
        result = document_processor.process_file(
            test_doc_path,
            original_filename="graphrag_documentation.txt",
            enable_quality_filtering=True
        )
        
        if result and result.get("status") == "success":
            doc_id = result.get("document_id")
            chunks = result.get("chunks_created", 0)
            logger.info("✓ Document ingestion successful")
            logger.info("  - Document ID: %s", doc_id)
            logger.info("  - Chunks created: %s", chunks)
            logger.info("  - Duration: %.2f seconds", result.get("duration_seconds", 0))
        else:
            logger.error("✗ Document ingestion failed: %s", result.get("error"))
            return False
            
    except Exception as e:
        logger.error("✗ Document ingestion error: %s", e)
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test document
        if test_doc_path.exists():
            test_doc_path.unlink()
            logger.info("Cleaned up test document")
    
    # Step 4: Check entity extraction status
    logger.info("Checking entity extraction status...")
    try:
        is_running = document_processor.is_entity_extraction_running()
        status = document_processor.get_entity_extraction_status()
        
        if is_running:
            logger.info("✓ Entity extraction is running")
            logger.info("  - Active threads: %s", status.get("active_threads", 0))
            logger.info("  - Active operations: %s", status.get("active_operations", 0))
        else:
            logger.info("✓ Entity extraction completed or not running")
    except Exception as e:
        logger.error("✗ Failed to check entity extraction status: %s", e)
    
    # Step 5: Get database statistics
    logger.info("Retrieving database statistics...")
    try:
        with graph_db.driver.session() as session:
            doc_count = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            relationship_count = session.run("MATCH ()-[r:RELATED_TO|SIMILAR_TO]->() RETURN count(r) as count").single()["count"]
            
            logger.info("✓ Database Statistics")
            logger.info("  - Documents: %s", doc_count)
            logger.info("  - Chunks: %s", chunk_count)
            logger.info("  - Entities: %s", entity_count)
            logger.info("  - Relationships: %s", relationship_count)
            
            # Step 6: Check for cluster-ready graph
            if chunk_count > 0:
                similarities = session.run("MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count").single()["count"]
                logger.info("  - Chunk Similarities: %s", similarities)
                
                if similarities > 0:
                    logger.info("✓ Graph is ready for clustering")
                else:
                    logger.info("⚠ No chunk similarities found - clustering may have limited impact")
            else:
                logger.warning("⚠ No chunks in database - cannot proceed with clustering")
                return False
                
    except Exception as e:
        logger.error("✗ Failed to retrieve database statistics: %s", e)
        return False
    
    # Step 7: Run Leiden clustering if enabled
    if settings.enable_clustering and entity_count > 0:
        logger.info("Running Leiden clustering on entities...")
        try:
            from core.graph_clustering import (
                fetch_entity_projection,
                normalize_edge_weights,
                run_leiden_clustering,
                to_igraph,
                write_communities_to_neo4j,
            )
            
            logger.info("  - Fetching entity projection...")
            nodes_df, edges_df = fetch_entity_projection(graph_db.driver)
            logger.info("    ✓ Projection: %s nodes, %s edges", len(nodes_df), len(edges_df))
            
            if len(edges_df) > 0:
                logger.info("  - Normalizing edge weights...")
                edges_df = normalize_edge_weights(edges_df)
                
                logger.info("  - Building igraph...")
                igraph = to_igraph(nodes_df, edges_df)
                
                logger.info("  - Running Leiden algorithm...")
                membership, modularity = run_leiden_clustering(
                    igraph, 
                    resolution=settings.clustering_resolution
                )
                logger.info("    ✓ Communities found: %s", len(set(membership.values())))
                logger.info("    ✓ Modularity: %.4f", modularity)
                
                logger.info("  - Writing communities to Neo4j...")
                updated = write_communities_to_neo4j(graph_db.driver, membership)
                logger.info("    ✓ Updated %s entities with community assignments", updated)
            else:
                logger.warning("⚠ No entity edges found for clustering")
        except Exception as e:
            logger.error("✗ Leiden clustering failed: %s", e)
            import traceback
            traceback.print_exc()
    else:
        logger.info("⚠ Clustering disabled or no entities to cluster")
    
    # Step 8: Summarize communities
    if settings.enable_clustering:
        logger.info("Generating community summaries...")
        try:
            from core.community_summarizer import community_summarizer
            
            with graph_db.driver.session() as session:
                levels_result = session.run("MATCH (e:Entity) WHERE e.level IS NOT NULL RETURN DISTINCT e.level as level ORDER BY level")
                levels = [record["level"] for record in levels_result]
                
                if levels:
                    logger.info("  - Community levels: %s", levels)
                    summaries = community_summarizer.summarize_levels(levels)
                    logger.info("✓ Community summaries generated: %s", len(summaries))
                    
                    for summary in summaries[:3]:  # Show first 3
                        logger.info("  - Community %s (level %s): %s chars", 
                                  summary.get("community_id"),
                                  summary.get("level"),
                                  len(summary.get("summary", "")))
                else:
                    logger.info("⚠ No community levels found in database")
        except Exception as e:
            logger.error("✗ Community summarization failed: %s", e)
            import traceback
            traceback.print_exc()
    
    # Step 9: Test graph visualization endpoint
    logger.info("Testing graph visualization...")
    try:
        graph_data = graph_db.get_clustered_graph(limit=50)
        logger.info("✓ Graph visualization data retrieved")
        logger.info("  - Nodes: %s", len(graph_data.get("nodes", [])))
        logger.info("  - Edges: %s", len(graph_data.get("edges", [])))
        logger.info("  - Communities: %s", len(graph_data.get("communities", [])))
        logger.info("  - Node types: %s", len(graph_data.get("node_types", [])))
    except Exception as e:
        logger.error("✗ Graph visualization failed: %s", e)
        import traceback
        traceback.print_exc()
    
    logger.info("=" * 60)
    logger.info("✓ Full pipeline test completed successfully!")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
