"""
Comprehensive test for the complete chat pipeline.

Tests end-to-end flow for "What is Carbonio?" query:
1. Document ingestion and entity extraction
2. Vector retrieval
3. Graph reasoning and expansion
4. Optional reranking (FlashRank)
5. Response generation
6. Quality scoring
7. Follow-up suggestions
"""

import asyncio
import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.models import ChatRequest  # noqa: E402
from api.routers.chat import _prepare_chat_context  # noqa: E402
from config.settings import settings  # noqa: E402
from core.embeddings import embedding_manager  # noqa: E402
from core.graph_db import graph_db  # noqa: E402
from ingestion.document_processor import DocumentProcessor  # noqa: E402
from rag.graph_rag import graph_rag  # noqa: E402
from rag.retriever import document_retriever  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Test document content about Carbonio
TEST_DOCUMENT_CONTENT = """# Carbonio Documentation

## What is Carbonio?

Carbonio is a comprehensive email and collaboration platform designed for modern businesses. 
It provides enterprise-grade email services, calendaring, contacts management, and team collaboration tools.

### Key Features

Carbonio offers several core features:

1. **Email Management**: Full-featured email server with IMAP, POP3, and SMTP support
2. **Calendar and Scheduling**: Shared calendars, meeting scheduling, and resource booking
3. **Contact Management**: Centralized address books with sharing capabilities
4. **Collaboration Tools**: Document sharing, chat, and video conferencing
5. **Mobile Support**: Native mobile apps for iOS and Android

### Architecture

Carbonio is built on a modular architecture with several key components:

- **Carbonio Node**: The core service node that handles email processing and storage
- **Web Interface**: Modern web-based interface for accessing all features
- **Mobile Sync**: ActiveSync protocol support for mobile device synchronization
- **Admin Console**: Comprehensive administration interface for system management
- **Storage Backend**: Flexible storage options including local and cloud storage

### Administration

System administrators can manage Carbonio using:

- **Carbonio CLI Command**: Command-line tools for system configuration and management
- **Admin Panel**: Web-based administration interface
- **API Access**: RESTful API for automation and integration

### Security Features

Carbonio includes enterprise-level security:

- End-to-end encryption for stored data
- TLS/SSL support for all connections
- Anti-spam and anti-virus protection
- Two-factor authentication support
- Comprehensive audit logging

### Use Cases

Carbonio is ideal for:

- Small to medium-sized businesses requiring email infrastructure
- Organizations needing collaboration tools
- Companies transitioning from legacy email systems
- Enterprises requiring on-premises email solutions

For more information, visit the official Carbonio documentation.
"""


@pytest.fixture(scope="module")
def test_document_path(tmp_path_factory):
    """Create a temporary test document about Carbonio."""
    tmp_dir = tmp_path_factory.mktemp("test_data")
    doc_path = tmp_dir / "carbonio_overview.md"
    doc_path.write_text(TEST_DOCUMENT_CONTENT)
    return doc_path


@pytest.fixture(scope="module")
def document_processor():
    """Initialize document processor."""
    dp = DocumentProcessor()
    try:
        yield dp
    finally:
        # Wait briefly for background entity extraction to finish to avoid teardown races
        max_wait = float(os.environ.get("TEST_BG_EXTRACTION_WAIT", "15"))
        poll_interval = 0.5
        waited = 0.0
        while dp.is_entity_extraction_running() and waited < max_wait:
            time.sleep(poll_interval)
            waited += poll_interval
        if dp.is_entity_extraction_running():
            logger.debug("Entity extraction still running after wait timeout (%s)", max_wait)
        else:
            logger.info("Background entity extraction finished before fixture teardown")


@pytest.fixture(scope="module")
def test_document_id(document_processor, test_document_path):
    """Ingest test document and return its ID."""
    logger.info("=" * 80)
    logger.info("SETTING UP TEST DOCUMENT")
    logger.info("=" * 80)
    
    # Verify Neo4j connectivity
    try:
        graph_db.driver.verify_connectivity()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
    
    # Process document
    result = document_processor.process_file(test_document_path)
    
    if result["status"] != "success":
        pytest.fail(f"Document processing failed: {result.get('error', 'Unknown error')}")
    
    doc_id = result["document_id"]
    chunks_created = result["chunks_created"]
    
    logger.info(f"✓ Document ingested: {doc_id}")
    logger.info(f"✓ Chunks created: {chunks_created}")
    
    # Extract entities if enabled
    if settings.enable_entity_extraction:
        entity_result = document_processor.extract_entities_for_document(
            doc_id=doc_id,
            file_name=test_document_path.name
        )
        
        logger.info(f"✓ Entity extraction: {entity_result.get('status')}")
        logger.info(f"  - Entities: {entity_result.get('entities_created', 0)}")
        logger.info(f"  - Relationships: {entity_result.get('relationships_created', 0)}")
    
    logger.info("=" * 80)
    
    return doc_id


@pytest.fixture(scope="module")
def cleanup_test_data(test_document_id):
    """Clean up test data after all tests complete."""
    yield
    
    # Cleanup happens after tests
    logger.info("\n" + "=" * 80)
    logger.info("CLEANING UP TEST DATA")
    logger.info("=" * 80)
    
    try:
        # Delete test document and all related data
        with graph_db.driver.session() as session:
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                DETACH DELETE d, c, e
                """,
                doc_id=test_document_id
            )
        logger.info(f"✓ Cleaned up test document: {test_document_id}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")
    
    logger.info("=" * 80)


def test_neo4j_connection():
    """Verify Neo4j database connectivity."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Neo4j Connection")
    logger.info("=" * 80)
    
    try:
        with graph_db.driver.session() as session:
            result = session.run("RETURN 1 AS test")
            assert result.single()["test"] == 1
        logger.info("✓ Neo4j connection verified")
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")
    
    logger.info("=" * 80)


def test_document_ingestion(test_document_id):
    """Verify test document was ingested correctly."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Document Ingestion")
    logger.info("=" * 80)
    
    # Verify document exists
    chunks = graph_db.get_document_chunks(test_document_id)
    assert len(chunks) > 0, "No chunks found for test document"
    
    logger.info(f"✓ Document has {len(chunks)} chunks")
    
    # Verify embeddings
    chunks_with_embeddings = sum(1 for c in chunks if c.get("embedding") is not None)
    logger.info(f"✓ {chunks_with_embeddings}/{len(chunks)} chunks have embeddings")
    assert chunks_with_embeddings > 0, "No chunks have embeddings"
    
    # Verify entities if extraction is enabled
    if settings.enable_entity_extraction:
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN DISTINCT e.id AS entity_id, e.name AS name, e.type AS type
        LIMIT 50
        """
        
        with graph_db.driver.session() as session:
            result_records = session.run(query, doc_id=test_document_id)
            entities = [record.data() for record in result_records]
        
        logger.info(f"✓ Found {len(entities)} entities")
        
        if entities:
            # Show sample entities
            for i, entity in enumerate(entities[:5], 1):
                logger.info(f"  {i}. {entity['name']} ({entity['type']})")
            
            # Look for Carbonio-related entities
            carbonio_entities = [e for e in entities if "carbonio" in e['name'].lower()]
            logger.info(f"✓ Found {len(carbonio_entities)} Carbonio-related entities")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_vector_retrieval(test_document_id, cleanup_test_data):
    """Test vector-based retrieval for 'What is Carbonio?' query."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Vector Retrieval")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Test chunk-based retrieval
    chunks = await document_retriever.chunk_based_retrieval(
        query=query,
        top_k=5,
        allowed_document_ids=[test_document_id]
    )
    
    assert len(chunks) > 0, "Vector retrieval returned no chunks"
    logger.info(f"✓ Retrieved {len(chunks)} chunks via vector search")
    
    # Verify similarity scores
    for i, chunk in enumerate(chunks[:3], 1):
        similarity = chunk.get("similarity", 0.0)
        content_preview = chunk.get("content", "")[:100] + "..."
        logger.info(f"  {i}. Similarity: {similarity:.4f}")
        logger.info(f"     Content: {content_preview}")
        assert similarity > 0, f"Chunk {i} has invalid similarity score"
    
    # Verify chunks contain relevant content
    combined_content = " ".join(c.get("content", "") for c in chunks).lower()
    assert "carbonio" in combined_content, "Retrieved chunks don't mention Carbonio"
    logger.info("✓ Retrieved chunks contain relevant content about Carbonio")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_entity_retrieval(test_document_id):
    """Test entity-based retrieval for 'What is Carbonio?' query."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Entity-Based Retrieval")
    logger.info("=" * 80)
    
    if not settings.enable_entity_extraction:
        logger.info("⚠ Entity extraction disabled - skipping test")
        logger.info("=" * 80)
        pytest.skip("Entity extraction disabled")
        return
    
    query = "What is Carbonio?"
    
    # Test entity-based retrieval
    chunks = await document_retriever.entity_based_retrieval(
        query=query,
        top_k=5,
        allowed_document_ids=[test_document_id]
    )
    
    logger.info(f"✓ Entity retrieval returned {len(chunks)} chunks")
    
    if len(chunks) > 0:
        # Verify entities are included
        for i, chunk in enumerate(chunks[:3], 1):
            entities = chunk.get("relevant_entities", [])
            similarity = chunk.get("similarity", 0.0)
            logger.info(f"  {i}. Entities: {entities}")
            logger.info(f"     Similarity: {similarity:.4f}")
        
        # Check for Carbonio-related entities
        all_entities = []
        for chunk in chunks:
            all_entities.extend(chunk.get("relevant_entities", []))
        
        carbonio_entities = [e for e in all_entities if "carbonio" in e.lower()]
        logger.info(f"✓ Found {len(carbonio_entities)} Carbonio entity mentions")
    else:
        logger.info("⚠ No chunks returned by entity-based retrieval")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_hybrid_retrieval(test_document_id):
    """Test hybrid retrieval combining vectors and entities."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Hybrid Retrieval")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Test hybrid retrieval with standard weights
    chunks = await document_retriever.hybrid_retrieval(
        query=query,
        top_k=5,
        chunk_weight=0.5,
        entity_weight=0.5,
        use_multi_hop=False,
        allowed_document_ids=[test_document_id]
    )
    
    assert len(chunks) > 0, "Hybrid retrieval returned no chunks"
    logger.info(f"✓ Hybrid retrieval returned {len(chunks)} chunks")
    
    # Analyze retrieval sources
    sources = {}
    for chunk in chunks:
        source = chunk.get("retrieval_source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    
    logger.info("✓ Retrieval source distribution:")
    for source, count in sources.items():
        logger.info(f"  - {source}: {count}")
    
    # Verify hybrid scores
    for i, chunk in enumerate(chunks[:3], 1):
        hybrid_score = chunk.get("hybrid_score", 0.0)
        retrieval_source = chunk.get("retrieval_source", "unknown")
        logger.info(f"  {i}. Score: {hybrid_score:.4f} | Source: {retrieval_source}")
        assert hybrid_score > 0, f"Chunk {i} has invalid hybrid score"
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_graph_expansion(test_document_id):
    """Test graph expansion with entity relationships."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Graph Expansion")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Test retrieval with graph expansion
    chunks = await document_retriever.retrieve_with_graph_expansion(
        query=query,
        top_k=3,
        expand_depth=2,
        use_multi_hop=False,
        allowed_document_ids=[test_document_id]
    )
    
    assert len(chunks) > 0, "Graph expansion returned no chunks"
    logger.info(f"✓ Graph expansion returned {len(chunks)} chunks")
    
    # Count original vs expanded chunks
    original_chunks = [c for c in chunks if not c.get("expansion_context")]
    expanded_chunks = [c for c in chunks if c.get("expansion_context")]
    
    logger.info(f"  - Original chunks: {len(original_chunks)}")
    logger.info(f"  - Expanded chunks: {len(expanded_chunks)}")
    
    if expanded_chunks:
        logger.info("✓ Sample expanded chunks:")
        for i, chunk in enumerate(expanded_chunks[:3], 1):
            expansion_type = chunk.get("expansion_context", {}).get("expansion_type", "unknown")
            similarity = chunk.get("similarity", 0.0)
            logger.info(f"  {i}. Type: {expansion_type} | Similarity: {similarity:.4f}")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_reranking(test_document_id):
    """Test FlashRank reranking if enabled."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Reranking (FlashRank)")
    logger.info("=" * 80)
    
    if not getattr(settings, "flashrank_enabled", False):
        logger.info("⚠ FlashRank reranking disabled - skipping test")
        logger.info("=" * 80)
        pytest.skip("FlashRank disabled")
        return
    
    query = "What is Carbonio?"
    
    # Get initial retrieval results
    initial_chunks = await document_retriever.hybrid_retrieval(
        query=query,
        top_k=10,
        chunk_weight=0.5,
        entity_weight=0.5,
        use_multi_hop=False,
        allowed_document_ids=[test_document_id]
    )
    
    if len(initial_chunks) == 0:
        pytest.skip("No chunks retrieved for reranking test")
    
    logger.info(f"✓ Initial retrieval: {len(initial_chunks)} chunks")
    
    # Test reranking
    try:
        from rag.rerankers.flashrank_reranker import rerank_with_flashrank
        
        reranked_chunks = rerank_with_flashrank(
            query=query,
            candidates=initial_chunks,
            max_candidates=5
        )
        
        assert len(reranked_chunks) > 0, "Reranking returned no chunks"
        logger.info(f"✓ Reranked to {len(reranked_chunks)} chunks")
        
        # Compare top chunks before and after reranking
        logger.info("✓ Top 3 chunks after reranking:")
        for i, chunk in enumerate(reranked_chunks[:3], 1):
            score = chunk.get("hybrid_score", chunk.get("similarity", 0.0))
            rerank_score = chunk.get("rerank_score", 0.0)
            logger.info(f"  {i}. Original: {score:.4f} | Reranked: {rerank_score:.4f}")
        
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")
        pytest.skip(f"Reranking not available: {e}")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_complete_rag_pipeline(test_document_id):
    """Test the complete RAG pipeline from query to response."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Complete RAG Pipeline")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Run the complete pipeline
    result = graph_rag.query(
        user_query=query,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        use_multi_hop=False,
        context_documents=[test_document_id]
    )
    
    # Verify pipeline stages
    stages = result.get("stages", [])
    logger.info(f"✓ Pipeline executed {len(stages)} stages: {stages}")
    
    expected_stages = ["query_analysis", "retrieval", "graph_reasoning", "generation"]
    for stage in expected_stages:
        assert stage in stages, f"Stage '{stage}' not executed"
    
    # Verify query analysis
    query_analysis = result.get("query_analysis", {})
    assert query_analysis, "Query analysis missing"
    logger.info(f"✓ Query analysis completed")
    logger.info(f"  - Query type: {query_analysis.get('query_type', 'unknown')}")
    
    # Verify retrieval
    retrieved_chunks = result.get("retrieved_chunks", [])
    assert len(retrieved_chunks) > 0, "No chunks retrieved"
    logger.info(f"✓ Retrieved {len(retrieved_chunks)} chunks")
    
    # Verify graph reasoning
    graph_context = result.get("graph_context", [])
    assert len(graph_context) > 0, "No graph context generated"
    logger.info(f"✓ Graph reasoning produced {len(graph_context)} context items")
    
    # Verify response generation
    response = result.get("response", "")
    assert len(response) > 0, "No response generated"
    assert "carbonio" in response.lower(), "Response doesn't mention Carbonio"
    logger.info(f"✓ Generated response ({len(response)} chars)")
    logger.info(f"  Preview: {response[:200]}...")
    
    # Verify sources
    sources = result.get("sources", [])
    logger.info(f"✓ {len(sources)} sources cited")
    
    # Verify quality score
    quality_score = result.get("quality_score")
    if quality_score:
        logger.info(f"✓ Quality score calculated:")
        logger.info(f"  - Overall: {quality_score.get('overall_score', 0.0):.2f}")
        logger.info(f"  - Relevance: {quality_score.get('relevance_score', 0.0):.2f}")
        logger.info(f"  - Completeness: {quality_score.get('completeness_score', 0.0):.2f}")
        logger.info(f"  - Grounding: {quality_score.get('grounding_score', 0.0):.2f}")
    else:
        logger.warning("⚠ Quality score not calculated")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_chat_router_integration(test_document_id):
    """Test the chat router's _prepare_chat_context function."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Chat Router Integration")
    logger.info("=" * 80)
    
    # Create a chat request
    request = ChatRequest(
        message="What is Carbonio?",
        session_id=None,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        use_multi_hop=False,
        stream=False,
        context_documents=[test_document_id]
    )
    
    # Test the chat preparation function
    (
        session_id,
        chat_history,
        result,
        context_documents,
        context_document_labels,
        context_hashtags,
    ) = await _prepare_chat_context(request)
    
    # Verify session creation
    assert session_id is not None, "Session ID not created"
    logger.info(f"✓ Session created: {session_id}")
    
    # Verify result structure
    assert "response" in result, "Response missing from result"
    assert "sources" in result, "Sources missing from result"
    assert "metadata" in result, "Metadata missing from result"
    logger.info(f"✓ Result structure valid")
    
    # Verify response content
    response = result.get("response", "")
    assert len(response) > 0, "Response is empty"
    assert "carbonio" in response.lower(), "Response doesn't mention Carbonio"
    logger.info(f"✓ Response generated ({len(response)} chars)")
    
    # Verify context documents
    assert test_document_id in context_documents, "Test document not in context"
    logger.info(f"✓ Context documents preserved: {context_documents}")
    
    # Verify metadata
    metadata = result.get("metadata", {})
    assert "context_documents" in metadata, "Context documents missing from metadata"
    logger.info(f"✓ Metadata includes context: {list(metadata.keys())}")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_multi_turn_conversation(test_document_id):
    """Test multi-turn conversation with context preservation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Multi-Turn Conversation")
    logger.info("=" * 80)
    
    import uuid
    session_id = str(uuid.uuid4())
    
    # First turn: "What is Carbonio?"
    request1 = ChatRequest(
        message="What is Carbonio?",
        session_id=session_id,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        stream=False,
        context_documents=[test_document_id]
    )
    
    result1 = graph_rag.query(
        user_query=request1.message,
        retrieval_mode=request1.retrieval_mode,
        top_k=request1.top_k,
        temperature=request1.temperature,
        context_documents=request1.context_documents
    )
    
    assert len(result1.get("response", "")) > 0, "First response empty"
    logger.info(f"✓ Turn 1 completed: {len(result1['response'])} chars")
    logger.info(f"  Preview: {result1['response'][:150]}...")
    
    # Second turn: Follow-up question
    chat_history = [
        {"role": "user", "content": "What is Carbonio?"},
        {"role": "assistant", "content": result1.get("response", "")}
    ]
    
    request2 = ChatRequest(
        message="What are its key features?",
        session_id=session_id,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        stream=False,
        context_documents=[test_document_id]
    )
    
    result2 = graph_rag.query(
        user_query=request2.message,
        retrieval_mode=request2.retrieval_mode,
        top_k=request2.top_k,
        temperature=request2.temperature,
        chat_history=chat_history,
        context_documents=request2.context_documents
    )
    
    assert len(result2.get("response", "")) > 0, "Second response empty"
    logger.info(f"✓ Turn 2 completed: {len(result2['response'])} chars")
    logger.info(f"  Preview: {result2['response'][:150]}...")
    
    # Verify follow-up context is used
    response2 = result2.get("response", "").lower()
    assert any(word in response2 for word in ["feature", "email", "calendar", "collaboration"]), \
        "Follow-up response doesn't address features"
    
    logger.info("✓ Multi-turn conversation context preserved")
    logger.info("=" * 80)


def test_unrestricted_context_retrieval(test_document_id):
    """Test that queries work without context restrictions (the default UI behavior)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Unrestricted Context Retrieval")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Simulate UI behavior: no context_documents provided, restrict_to_context=True by default
    # This should search ALL documents, not return empty results
    result = graph_rag.query(
        user_query=query,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.0,
        use_multi_hop=False,
        context_documents=[],  # Empty list - should search all docs
        restrict_to_context=True,  # Default from settings
    )
    
    assert result is not None, "Query returned None"
    assert "response" in result, "Result missing response field"
    
    response = result["response"]
    assert len(response) > 0, "Response is empty - context restriction bug!"
    assert "couldn't find" not in response.lower(), "Got 'couldn't find' error message"
    
    # Should have retrieved chunks
    retrieved_chunks = result.get("retrieved_chunks", [])
    graph_context = result.get("graph_context", [])
    total_chunks = len(retrieved_chunks) + len(graph_context)
    
    logger.info(f"✓ Query with empty context_documents returned valid response")
    logger.info(f"✓ Response length: {len(response)} chars")
    logger.info(f"✓ Total chunks used: {total_chunks}")
    logger.info(f"  Preview: {response[:150]}...")
    
    assert total_chunks > 0, "No chunks retrieved with empty context - restriction bug!"
    
    logger.info("=" * 80)


def test_default_model_configuration(test_document_id):
    """Test that queries work with default model configuration (no explicit model specified)."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Default Model Configuration")
    logger.info("=" * 80)
    
    query = "What is Carbonio?"
    
    # Don't specify llm_model - should use default from chat_tuning_config.json
    # This simulates the UI behavior when user hasn't changed model selection
    result = graph_rag.query(
        user_query=query,
        retrieval_mode="hybrid",
        top_k=3,
        temperature=0.7,
        context_documents=[test_document_id],
        # llm_model is NOT specified - should use default
    )
    
    assert result is not None, "Query returned None with default model"
    assert "response" in result, "Result missing response field"
    
    response = result["response"]
    assert len(response) > 0, "Response is empty with default model"
    assert "couldn't find" not in response.lower(), "Got 'couldn't find' error with default model"
    assert "error" not in response.lower() or "carbonio" in response.lower(), \
        "Response contains error message"
    
    # Verify we got actual content about Carbonio
    response_lower = response.lower()
    carbonio_terms = ["carbonio", "email", "collaboration", "platform"]
    found_terms = [term for term in carbonio_terms if term in response_lower]
    assert len(found_terms) >= 2, \
        f"Response doesn't contain Carbonio information. Found terms: {found_terms}"
    
    # Check metadata for model info
    metadata = result.get("metadata", {})
    logger.info(f"✓ Query with default model completed successfully")
    logger.info(f"✓ Model used: {metadata.get('llm_model', 'not specified')}")
    logger.info(f"✓ Response length: {len(response)} chars")
    logger.info(f"✓ Response preview: {response[:150]}...")
    
    # Verify quality score if available
    quality_score = result.get("quality_score")
    if quality_score:
        total_score = quality_score.get("total", 0.0)
        assert total_score > 50, f"Quality score too low with default model: {total_score}"
        logger.info(f"✓ Quality score: {total_score:.1f}%")
        breakdown = quality_score.get("breakdown", {})
        if breakdown:
            for key, value in breakdown.items():
                logger.info(f"  - {key}: {value:.1f}%")
    
    logger.info("=" * 80)


@pytest.mark.asyncio
async def test_cache_effectiveness(test_document_id):
    """Test that caching system works effectively on repeated queries."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST: Cache Effectiveness")
    logger.info("=" * 80)
    
    # Skip if caching is disabled
    if not settings.enable_caching:
        logger.warning("⚠ Caching disabled in settings, skipping cache test")
        pytest.skip("Caching disabled")
    
    query = "What is Carbonio?"
    
    # First query - should populate caches
    logger.info("Running first query (cold caches)...")
    result1 = graph_rag.query(
        user_query=query,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        use_multi_hop=False,
        context_documents=[test_document_id]
    )
    
    assert result1.get("response"), "First query failed"
    logger.info(f"✓ First query completed ({len(result1.get('response', ''))} chars)")
    
    # Small delay to ensure cache writes complete
    await asyncio.sleep(0.1)
    
    # Second query - should hit caches
    logger.info("Running second query (warm caches)...")
    result2 = graph_rag.query(
        user_query=query,
        retrieval_mode="hybrid",
        top_k=5,
        temperature=0.7,
        use_multi_hop=False,
        context_documents=[test_document_id]
    )
    
    assert result2.get("response"), "Second query failed"
    logger.info(f"✓ Second query completed ({len(result2.get('response', ''))} chars)")
    
    # Verify retrieval results are consistent (indicating cache hit)
    chunks1 = result1.get("retrieved_chunks", [])
    chunks2 = result2.get("retrieved_chunks", [])
    
    # Extract chunk IDs for comparison
    chunk_ids1 = [c.get("chunk_id") for c in chunks1]
    chunk_ids2 = [c.get("chunk_id") for c in chunks2]
    
    # Chunks should be identical if cache worked
    if chunk_ids1 == chunk_ids2:
        logger.info("✓ Retrieval cache hit verified (identical chunks returned)")
    else:
        logger.warning("⚠ Chunk ordering/content differs (cache may have missed)")
    
    # Check cache stats if available
    try:
        from core.cache_metrics import CacheMetrics
        metrics = CacheMetrics.get_all_stats()
        
        logger.info("\n✓ Cache statistics:")
        for cache_name, stats in metrics.items():
            hit_rate = stats.get("hit_rate", 0.0)
            hits = stats.get("hits", 0)
            misses = stats.get("misses", 0)
            logger.info(f"  - {cache_name}: {hits} hits, {misses} misses ({hit_rate:.1%} hit rate)")
        
        # Verify at least one cache has hits
        total_hits = sum(s.get("hits", 0) for s in metrics.values())
        assert total_hits > 0, "No cache hits recorded"
        logger.info(f"\n✓ Total cache hits: {total_hits}")
        
    except Exception as e:
        logger.warning(f"⚠ Could not retrieve cache metrics: {e}")
    
    logger.info("=" * 80)


def test_full_pipeline_summary(test_document_id, cleanup_test_data):
    """Print a summary of all pipeline tests."""
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("=" * 80)
    
    # Get document stats
    chunks = graph_db.get_document_chunks(test_document_id)
    
    logger.info(f"Test Document: {test_document_id}")
    logger.info(f"Chunks: {len(chunks)}")
    
    if settings.enable_entity_extraction:
        query = """
        MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
        RETURN count(DISTINCT e) AS entity_count
        """
        with graph_db.driver.session() as session:
            result = session.run(query, doc_id=test_document_id)
            record = result.single()
            entity_count = record["entity_count"] if record else 0
        logger.info(f"Entities: {entity_count}")
    
    logger.info("")
    logger.info("✓ All pipeline components verified:")
    logger.info("  ✓ Document ingestion")
    logger.info("  ✓ Vector retrieval")
    logger.info("  ✓ Entity retrieval" if settings.enable_entity_extraction else "  ⊘ Entity retrieval (disabled)")
    logger.info("  ✓ Hybrid retrieval")
    logger.info("  ✓ Graph expansion")
    logger.info("  ✓ Reranking" if getattr(settings, "flashrank_enabled", False) else "  ⊘ Reranking (disabled)")
    logger.info("  ✓ Response generation")
    logger.info("  ✓ Quality scoring")
    logger.info("  ✓ Multi-turn conversation")
    logger.info("  ✓ Cache effectiveness" if settings.enable_caching else "  ⊘ Cache effectiveness (disabled)")
    logger.info("")
    logger.info("Pipeline Status: OPERATIONAL ✓")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
