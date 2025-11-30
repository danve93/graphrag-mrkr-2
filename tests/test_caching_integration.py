"""Integration tests for caching system with real services."""

import pytest
import asyncio
from core.graph_db import graph_db
from core.embeddings import embedding_manager
from rag.retriever import document_retriever


@pytest.mark.integration
def test_entity_label_cache_hit_rate():
    """Test entity label cache reduces database queries."""
    entity_id = "integration_test_entity_123"
    
    # Clear cache for clean test
    with graph_db._entity_label_lock:
        graph_db._entity_label_cache.clear()
    
    # First call - cache miss (database query)
    label1 = graph_db.get_entity_label_cached(entity_id)
    
    # Second call - cache hit (no database query)
    label2 = graph_db.get_entity_label_cached(entity_id)
    
    # Labels should match
    assert label1 == label2
    
    # Verify cache contains the entry
    with graph_db._entity_label_lock:
        assert entity_id in graph_db._entity_label_cache


@pytest.mark.integration
def test_entity_label_cache_multiple_entities():
    """Test entity label cache with multiple entities."""
    entity_ids = [f"integration_entity_{i}" for i in range(5)]
    
    # Clear cache
    with graph_db._entity_label_lock:
        graph_db._entity_label_cache.clear()
    
    # Fetch all entities (cache misses)
    labels1 = [graph_db.get_entity_label_cached(eid) for eid in entity_ids]
    
    # Fetch again (cache hits)
    labels2 = [graph_db.get_entity_label_cached(eid) for eid in entity_ids]
    
    # All labels should match
    assert labels1 == labels2
    
    # Cache should contain all entities
    with graph_db._entity_label_lock:
        for eid in entity_ids:
            assert eid in graph_db._entity_label_cache


@pytest.mark.integration
def test_embedding_cache_reduces_api_calls():
    """Test embedding cache reduces API calls."""
    from unittest.mock import patch
    
    text = "This is an integration test sentence for embedding cache"
    
    # Clear cache
    with embedding_manager._cache_lock:
        embedding_manager._cache.clear()
    
    # Mock the direct generation to track calls
    original_method = embedding_manager._generate_embedding_direct
    call_count = {"count": 0}
    
    def mock_generate(text):
        call_count["count"] += 1
        return original_method(text)
    
    with patch.object(embedding_manager, '_generate_embedding_direct', side_effect=mock_generate):
        # First call - cache miss (API call)
        emb1 = embedding_manager.get_embedding(text)
        first_call_count = call_count["count"]
        
        # Second call - cache hit (no API call)
        emb2 = embedding_manager.get_embedding(text)
        second_call_count = call_count["count"]
        
        # Verify embeddings match
        assert emb1 == emb2
        assert len(emb1) == len(emb2)
        
        # Verify cache hit (no additional API call)
        assert first_call_count == 1, "First call should generate embedding"
        assert second_call_count == 1, "Second call should use cache"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_cache_reuses_results():
    """Test retrieval cache reuses graph traversal results."""
    from unittest.mock import patch, AsyncMock
    
    query = "What is a graph database?"
    mock_results = [
        {"chunk_id": "int_1", "content": "Test chunk", "similarity": 0.85},
        {"chunk_id": "int_2", "content": "Another chunk", "similarity": 0.75}
    ]
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    # Track direct retrieval calls
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=mock_results) as mock_retrieval:
        # First call - full retrieval
        results1 = await document_retriever.hybrid_retrieval(query, top_k=5)
        first_call_count = mock_retrieval.call_count
        
        # Second call - cached results
        results2 = await document_retriever.hybrid_retrieval(query, top_k=5)
        second_call_count = mock_retrieval.call_count
        
        # Verify results match
        assert results1 == results2
        assert len(results1) > 0
        
        # Verify cache hit (no additional retrieval)
        assert first_call_count == 1, "First call should perform retrieval"
        assert second_call_count == 1, "Second call should use cache"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retrieval_cache_different_parameters():
    """Test retrieval cache creates separate entries for different parameters."""
    from unittest.mock import patch, AsyncMock
    
    query = "Test query for parameter variation"
    mock_results = [{"chunk_id": "test", "similarity": 0.8}]
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=mock_results) as mock_retrieval:
        # Call with different top_k values
        await document_retriever.hybrid_retrieval(query, top_k=5)
        await document_retriever.hybrid_retrieval(query, top_k=10)
        
        # Should call retrieval twice (different parameters)
        assert mock_retrieval.call_count == 2, "Different parameters should create separate cache entries"


@pytest.mark.integration
def test_feature_flag_disables_caching():
    """Test that caching can be disabled via feature flag."""
    from config.settings import settings
    from unittest.mock import patch
    
    entity_id = "feature_flag_test_entity"
    original_setting = settings.enable_caching
    
    try:
        # Disable caching
        settings.enable_caching = False
        
        # Clear cache
        with graph_db._entity_label_lock:
            graph_db._entity_label_cache.clear()
        
        # Mock the direct query method
        with patch.object(graph_db, '_get_entity_label_direct', return_value="Test Entity") as mock_direct:
            # Both calls should hit the direct method
            graph_db.get_entity_label_cached(entity_id)
            graph_db.get_entity_label_cached(entity_id)
            
            # Should call direct query twice (no caching)
            assert mock_direct.call_count == 2, "With caching disabled, should query directly each time"
    
    finally:
        settings.enable_caching = original_setting


@pytest.mark.integration
def test_cache_memory_usage():
    """Test that cache memory usage stays within reasonable limits."""
    import sys
    
    # Clear all caches
    with graph_db._entity_label_lock:
        graph_db._entity_label_cache.clear()
    with embedding_manager._cache_lock:
        embedding_manager._cache.clear()
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    # Fill entity label cache
    for i in range(1000):
        graph_db._entity_label_cache[f"entity_{i}"] = f"Entity Name {i}"
    
    # Fill embedding cache with typical embeddings (1536 dimensions)
    for i in range(100):
        embedding_manager._cache[f"text_{i}"] = [0.1] * 1536
    
    # Fill retrieval cache with typical results
    for i in range(50):
        document_retriever._cache[f"query_{i}"] = [
            {"chunk_id": f"chunk_{j}", "content": "x" * 500, "similarity": 0.8}
            for j in range(5)
        ]
    
    # Calculate approximate memory usage
    entity_size = sys.getsizeof(graph_db._entity_label_cache)
    embedding_size = sys.getsizeof(embedding_manager._cache)
    retrieval_size = sys.getsizeof(document_retriever._cache)
    
    total_size_mb = (entity_size + embedding_size + retrieval_size) / (1024 * 1024)
    
    # Memory should be reasonable (< 100MB for this test)
    assert total_size_mb < 100, f"Cache memory usage ({total_size_mb:.2f}MB) exceeds limit"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_concurrent_cache_access():
    """Test that caches handle concurrent access correctly."""
    from unittest.mock import patch
    
    text = "Concurrent test text"
    results = []
    
    # Clear cache
    with embedding_manager._cache_lock:
        embedding_manager._cache.clear()
    
    # Mock to ensure we only generate once
    with patch.object(embedding_manager, '_generate_embedding_direct', return_value=[0.1, 0.2, 0.3]) as mock_gen:
        async def fetch_embedding():
            emb = embedding_manager.get_embedding(text)
            results.append(emb)
        
        # Create concurrent tasks
        tasks = [fetch_embedding() for _ in range(20)]
        await asyncio.gather(*tasks)
        
        # All results should be identical
        assert all(r == results[0] for r in results), "All concurrent calls should get same result"
        
        # Should only generate once (subsequent calls use cache)
        assert mock_gen.call_count == 1, "Should only generate embedding once despite concurrent calls"


@pytest.mark.integration
def test_cache_persistence_across_operations():
    """Test that cache persists across multiple operations."""
    entity_id = "persistent_test_entity"
    
    # Clear and populate cache
    with graph_db._entity_label_lock:
        graph_db._entity_label_cache.clear()
    
    # Store value
    graph_db.get_entity_label_cached(entity_id)
    
    # Perform other operations
    text = "some random text"
    embedding_manager.get_embedding(text)
    
    # Cache should still contain original entity
    with graph_db._entity_label_lock:
        assert entity_id in graph_db._entity_label_cache, "Cache should persist across other operations"


@pytest.mark.integration
def test_cache_cleanup_on_shutdown():
    """Test that cache cleanup works correctly."""
    from core.singletons import cleanup_singletons
    
    # Populate caches
    graph_db.get_entity_label_cached("cleanup_test")
    embedding_manager.get_embedding("cleanup test")
    
    # Get cache sizes before cleanup
    entity_size_before = len(graph_db._entity_label_cache)
    embedding_size_before = len(embedding_manager._cache)
    
    # Cleanup
    cleanup_singletons()
    
    # Get new cache instances
    new_entity_cache = graph_db._entity_label_cache
    new_embedding_cache = embedding_manager._cache
    
    # Verify caches are cleared
    assert len(new_entity_cache) == 0, "Entity cache should be empty after cleanup"
    assert len(new_embedding_cache) == 0, "Embedding cache should be empty after cleanup"
