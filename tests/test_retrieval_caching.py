"""Unit tests for DocumentRetriever caching layer."""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock
from rag.retriever import document_retriever
from core.singletons import hash_retrieval_params


@pytest.mark.asyncio
async def test_retrieval_cache_hit():
    """Test that repeated retrieval requests use cache."""
    query = "What is Neo4j?"
    expected_results = [{"chunk_id": "1", "content": "Neo4j is a graph database"}]
    
    # Clear cache to start fresh
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    # First call - should perform retrieval
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=expected_results) as mock_retrieval:
        results1 = await document_retriever.hybrid_retrieval(query, top_k=5)
        assert mock_retrieval.call_count == 1
        assert results1 == expected_results
    
    # Second call - should use cache (no retrieval)
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=expected_results) as mock_retrieval:
        results2 = await document_retriever.hybrid_retrieval(query, top_k=5)
        assert mock_retrieval.call_count == 0  # No retrieval performed
        assert results2 == expected_results
        assert results1 == results2


@pytest.mark.asyncio
async def test_retrieval_cache_different_queries():
    """Test that different queries generate separate cache entries."""
    query1 = "What is Neo4j?"
    query2 = "What is GraphRAG?"
    results1 = [{"chunk_id": "1", "content": "Neo4j"}]
    results2 = [{"chunk_id": "2", "content": "GraphRAG"}]
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock) as mock_retrieval:
        mock_retrieval.side_effect = [results1, results2]
        
        res1 = await document_retriever.hybrid_retrieval(query1, top_k=5)
        res2 = await document_retriever.hybrid_retrieval(query2, top_k=5)
        
        assert res1 == results1
        assert res2 == results2
        assert res1 != res2
        assert mock_retrieval.call_count == 2


@pytest.mark.asyncio
async def test_retrieval_cache_key_includes_parameters():
    """Test that cache key includes all retrieval parameters."""
    query = "Test query"
    results = [{"chunk_id": "1"}]
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=results) as mock_retrieval:
        # Different parameters should result in different cache entries
        await document_retriever.hybrid_retrieval(query, top_k=5, chunk_weight=0.5)
        await document_retriever.hybrid_retrieval(query, top_k=10, chunk_weight=0.5)  # Different top_k
        await document_retriever.hybrid_retrieval(query, top_k=5, chunk_weight=0.7)  # Different weight
        
        # Should call retrieval 3 times (different parameters = different cache keys)
        assert mock_retrieval.call_count == 3


@pytest.mark.asyncio
async def test_retrieval_cache_ttl_expiration():
    """Test that cache entries expire after TTL."""
    query = "Test query with TTL"
    results = [{"chunk_id": "1"}]
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    # Set very short TTL for testing (need to modify cache TTL directly)
    original_ttl = document_retriever._cache.ttl
    
    try:
        # Replace cache with short TTL version
        from cachetools import TTLCache
        document_retriever._cache = TTLCache(maxsize=1000, ttl=1)  # 1 second TTL
        
        with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=results) as mock_retrieval:
            # First call - cache miss
            await document_retriever.hybrid_retrieval(query, top_k=5)
            assert mock_retrieval.call_count == 1
            
            # Second call immediately - cache hit
            await document_retriever.hybrid_retrieval(query, top_k=5)
            assert mock_retrieval.call_count == 1  # Still 1 (cache hit)
            
            # Wait for TTL expiration
            await asyncio.sleep(1.5)
            
            # Third call after expiration - cache miss
            await document_retriever.hybrid_retrieval(query, top_k=5)
            assert mock_retrieval.call_count == 2  # Cache expired, new retrieval
    
    finally:
        # Restore original cache
        from cachetools import TTLCache
        document_retriever._cache = TTLCache(maxsize=1000, ttl=original_ttl)


@pytest.mark.asyncio
async def test_retrieval_cache_disabled():
    """Test that caching can be disabled via feature flag."""
    from config.settings import settings
    
    query = "Test query"
    results = [{"chunk_id": "1"}]
    original_setting = settings.enable_caching
    
    try:
        settings.enable_caching = False
        
        # Clear cache
        with document_retriever._cache_lock:
            document_retriever._cache.clear()
        
        with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=results) as mock_retrieval:
            await document_retriever.hybrid_retrieval(query, top_k=5)
            await document_retriever.hybrid_retrieval(query, top_k=5)
            assert mock_retrieval.call_count == 2  # Both calls retrieve
    
    finally:
        settings.enable_caching = original_setting


@pytest.mark.asyncio
async def test_retrieval_cache_thread_safety():
    """Test that cache access is thread-safe."""
    query = "Thread safety test"
    results = [{"chunk_id": "1"}]
    result_list = []
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=results):
        async def fetch_results():
            res = await document_retriever.hybrid_retrieval(query, top_k=5)
            result_list.append(res)
        
        # Create multiple concurrent tasks
        tasks = [fetch_results() for _ in range(10)]
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        # All results should be identical
        assert len(result_list) == 10, "All tasks should complete"
        assert all(r == result_list[0] for r in result_list), "All tasks should get same result"


@pytest.mark.asyncio
async def test_retrieval_cache_empty_results():
    """Test that empty results are cached correctly."""
    query = "Query with no results"
    empty_results = []
    
    # Clear cache
    with document_retriever._cache_lock:
        document_retriever._cache.clear()
    
    with patch.object(document_retriever, '_hybrid_retrieval_direct', new_callable=AsyncMock, return_value=empty_results) as mock_retrieval:
        # First call
        results1 = await document_retriever.hybrid_retrieval(query, top_k=5)
        assert results1 == []
        assert mock_retrieval.call_count == 1
        
        # Second call - should use cached empty result
        results2 = await document_retriever.hybrid_retrieval(query, top_k=5)
        assert results2 == []
        assert mock_retrieval.call_count == 1  # No new retrieval
