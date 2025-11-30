"""Unit tests for EmbeddingManager caching layer."""

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock
from core.embeddings import embedding_manager
from core.singletons import hash_text


def test_embedding_cache_hit():
    """Test that repeated embedding requests use cache."""
    text = "This is a test document"
    
    # Clear cache to start fresh
    embedding_manager._cache.clear()
    
    # First call - should generate embedding
    with patch.object(embedding_manager, '_generate_embedding_direct', return_value=[0.1, 0.2, 0.3]) as mock_generate:
        emb1 = embedding_manager.get_embedding(text)
        assert mock_generate.call_count == 1
        assert emb1 == [0.1, 0.2, 0.3]
    
    # Second call - should use cache (no generation)
    with patch.object(embedding_manager, '_generate_embedding_direct', return_value=[0.1, 0.2, 0.3]) as mock_generate:
        emb2 = embedding_manager.get_embedding(text)
        assert mock_generate.call_count == 0  # No API call
        assert emb2 == [0.1, 0.2, 0.3]
        assert emb1 == emb2


def test_embedding_cache_different_texts():
    """Test that different texts generate different cache entries."""
    text1 = "First document"
    text2 = "Second document"
    
    # Clear cache to start fresh
    embedding_manager._cache.clear()
    
    with patch.object(embedding_manager, '_generate_embedding_direct') as mock_generate:
        mock_generate.side_effect = [
            [0.1, 0.2, 0.3],  # First text
            [0.4, 0.5, 0.6],  # Second text
        ]
        
        emb1 = embedding_manager.get_embedding(text1)
        emb2 = embedding_manager.get_embedding(text2)
        
        assert mock_generate.call_count == 2  # Both texts generated
        assert emb1 != emb2
        assert emb1 == [0.1, 0.2, 0.3]
        assert emb2 == [0.4, 0.5, 0.6]


def test_embedding_cache_model_in_key():
    """Test that cache key includes model name."""
    text = "Test document"
    
    # Clear cache
    embedding_manager._cache.clear()
    
    # Generate cache keys with different models
    key1 = hash_text(text, "text-embedding-ada-002")
    key2 = hash_text(text, "text-embedding-3-small")
    
    # Keys should be different for different models
    assert key1 != key2


def test_embedding_cache_lru_eviction():
    """Test LRU cache eviction behavior."""
    from config.settings import settings
    
    # Save original cache size
    original_size = settings.embedding_cache_size
    
    try:
        # Set small cache size for testing
        settings.embedding_cache_size = 3
        
        # Recreate cache with small size
        from cachetools import LRUCache
        embedding_manager._cache = LRUCache(maxsize=3)
        
        with patch.object(embedding_manager, '_generate_embedding_direct') as mock_generate:
            mock_generate.side_effect = [
                [0.1, 0.2, 0.3],  # text1
                [0.4, 0.5, 0.6],  # text2
                [0.7, 0.8, 0.9],  # text3
                [1.0, 1.1, 1.2],  # text4 (should evict text1)
            ]
            
            # Fill cache to capacity
            embedding_manager.get_embedding("text1")
            embedding_manager.get_embedding("text2")
            embedding_manager.get_embedding("text3")
            
            assert len(embedding_manager._cache) == 3
            
            # Add one more (should evict least recently used - text1)
            embedding_manager.get_embedding("text4")
            
            assert len(embedding_manager._cache) == 3
            
            # Verify text1 was evicted (should generate again)
            mock_generate.reset_mock()
            # Set side_effect to None and use return_value instead
            mock_generate.side_effect = None
            mock_generate.return_value = [0.1, 0.2, 0.3]
            embedding_manager.get_embedding("text1")
            assert mock_generate.call_count == 1  # Had to regenerate
    
    finally:
        # Restore original cache size
        settings.embedding_cache_size = original_size
        from cachetools import LRUCache
        embedding_manager._cache = LRUCache(maxsize=original_size)


def test_embedding_cache_disabled():
    """Test that caching can be disabled via feature flag."""
    from config.settings import settings
    
    text = "Test document"
    original_setting = settings.enable_caching
    
    try:
        settings.enable_caching = False
        
        # Clear cache
        embedding_manager._cache.clear()
        
        # Should call direct generation every time
        with patch.object(embedding_manager, '_generate_embedding_direct', return_value=[0.1, 0.2, 0.3]) as mock_generate:
            embedding_manager.get_embedding(text)
            embedding_manager.get_embedding(text)
            assert mock_generate.call_count == 2  # Both calls generate embeddings
    
    finally:
        settings.enable_caching = original_setting


def test_embedding_cache_thread_safety():
    """Test that cache access is thread-safe."""
    text = "Thread test document"
    results = []
    
    # Clear cache
    embedding_manager._cache.clear()
    
    # Mock the direct generation to return consistent value
    with patch.object(embedding_manager, '_generate_embedding_direct', return_value=[0.1, 0.2, 0.3]):
        def fetch_embedding():
            emb = embedding_manager.get_embedding(text)
            results.append(emb)
        
        # Create multiple threads
        threads = [threading.Thread(target=fetch_embedding) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All results should be identical
        assert len(results) == 10, "All threads should complete"
        assert all(r == [0.1, 0.2, 0.3] for r in results), "All threads should get same result"
        assert len(set(tuple(r) for r in results)) == 1, "All embeddings should be identical"
