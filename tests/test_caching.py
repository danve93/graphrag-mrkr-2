"""Unit tests for caching singleton manager."""

import pytest
import time
import threading
from core.singletons import (
    get_graph_db_driver,
    get_entity_label_cache,
    get_embedding_cache,
    get_retrieval_cache,
    hash_text,
    hash_retrieval_params,
    cleanup_singletons,
)


def test_singleton_driver():
    """Test Neo4j driver singleton pattern."""
    driver1 = get_graph_db_driver()
    driver2 = get_graph_db_driver()
    assert driver1 is driver2, "Should return same driver instance"


def test_singleton_driver_thread_safety():
    """Test that driver singleton is thread-safe."""
    drivers = []
    
    def get_driver():
        drivers.append(get_graph_db_driver())
    
    threads = [threading.Thread(target=get_driver) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    
    # All threads should get the same driver instance
    assert len(set(id(d) for d in drivers)) == 1, "All threads should get same driver"


def test_entity_label_cache_singleton():
    """Test entity label cache singleton pattern."""
    cache1 = get_entity_label_cache()
    cache2 = get_entity_label_cache()
    assert cache1 is cache2, "Should return same cache instance"


def test_entity_label_cache_operations():
    """Test entity label cache basic operations."""
    cache = get_entity_label_cache()
    
    # Clear cache for clean test
    cache.clear()
    
    # Test write
    cache["entity_123"] = "Test Entity"
    
    # Test read
    assert cache["entity_123"] == "Test Entity"
    
    # Test overwrite
    cache["entity_123"] = "Updated Entity"
    assert cache["entity_123"] == "Updated Entity"
    
    # Test deletion
    del cache["entity_123"]
    assert "entity_123" not in cache


def test_entity_label_cache_ttl():
    """Test entity label cache TTL expiration."""
    from cachetools import TTLCache
    
    # Create a temporary cache with short TTL for testing
    test_cache = TTLCache(maxsize=100, ttl=1)  # 1 second TTL
    
    # Add entry
    test_cache["temp_entity"] = "Temporary"
    assert "temp_entity" in test_cache
    
    # Wait for expiration
    time.sleep(1.5)
    
    # Entry should be expired
    assert "temp_entity" not in test_cache, "Should expire after TTL"


def test_embedding_cache_singleton():
    """Test embedding cache singleton pattern."""
    cache1 = get_embedding_cache()
    cache2 = get_embedding_cache()
    assert cache1 is cache2, "Should return same cache instance"


def test_embedding_cache_operations():
    """Test embedding cache basic operations."""
    cache = get_embedding_cache()
    
    # Clear cache for clean test
    cache.clear()
    
    # Test write
    cache["key1"] = [1.0, 2.0, 3.0]
    
    # Test read
    assert cache["key1"] == [1.0, 2.0, 3.0]
    
    # Test multiple entries
    cache["key2"] = [4.0, 5.0, 6.0]
    cache["key3"] = [7.0, 8.0, 9.0]
    
    assert len(cache) == 3


def test_embedding_cache_lru_eviction():
    """Test embedding cache LRU eviction."""
    from cachetools import LRUCache
    
    # Create a temporary cache with small size for testing
    test_cache = LRUCache(maxsize=3)
    
    # Fill cache to capacity
    test_cache["key1"] = [1.0, 2.0]
    test_cache["key2"] = [3.0, 4.0]
    test_cache["key3"] = [5.0, 6.0]
    
    assert len(test_cache) == 3
    assert "key1" in test_cache
    
    # Add one more (should evict key1 - least recently used)
    test_cache["key4"] = [7.0, 8.0]
    
    assert len(test_cache) == 3
    assert "key1" not in test_cache, "Oldest key should be evicted"
    assert "key4" in test_cache, "Newest key should exist"


def test_retrieval_cache_singleton():
    """Test retrieval cache singleton pattern."""
    cache1 = get_retrieval_cache()
    cache2 = get_retrieval_cache()
    assert cache1 is cache2, "Should return same cache instance"


def test_retrieval_cache_operations():
    """Test retrieval cache basic operations."""
    cache = get_retrieval_cache()
    
    # Clear cache for clean test
    cache.clear()
    
    # Test write
    cache["query_abc"] = [{"chunk": "result1"}]
    
    # Test read
    assert cache["query_abc"] == [{"chunk": "result1"}]
    
    # Test complex values
    complex_result = [
        {"chunk_id": "1", "content": "test", "similarity": 0.85},
        {"chunk_id": "2", "content": "test2", "similarity": 0.75},
    ]
    cache["query_xyz"] = complex_result
    assert cache["query_xyz"] == complex_result


def test_hash_text():
    """Test hash_text function for cache keys."""
    # Same text and model should produce same hash
    hash1 = hash_text("test text", "model-1")
    hash2 = hash_text("test text", "model-1")
    assert hash1 == hash2
    
    # Different text should produce different hash
    hash3 = hash_text("different text", "model-1")
    assert hash1 != hash3
    
    # Different model should produce different hash
    hash4 = hash_text("test text", "model-2")
    assert hash1 != hash4
    
    # Hash should be consistent
    assert len(hash1) == 32, "MD5 hash should be 32 characters"


def test_hash_retrieval_params():
    """Test hash_retrieval_params function for cache keys."""
    # Same parameters should produce same hash
    hash1 = hash_retrieval_params(
        query="test query",
        mode="hybrid",
        top_k=5,
        chunk_weight=0.5,
        entity_weight=0.3,
        path_weight=0.2,
    )
    hash2 = hash_retrieval_params(
        query="test query",
        mode="hybrid",
        top_k=5,
        chunk_weight=0.5,
        entity_weight=0.3,
        path_weight=0.2,
    )
    assert hash1 == hash2
    
    # Different query should produce different hash
    hash3 = hash_retrieval_params(
        query="different query",
        mode="hybrid",
        top_k=5,
        chunk_weight=0.5,
        entity_weight=0.3,
        path_weight=0.2,
    )
    assert hash1 != hash3
    
    # Different parameters should produce different hash
    hash4 = hash_retrieval_params(
        query="test query",
        mode="hybrid",
        top_k=10,  # Different top_k
        chunk_weight=0.5,
        entity_weight=0.3,
        path_weight=0.2,
    )
    assert hash1 != hash4
    
    # Hash should be consistent
    assert len(hash1) == 32, "MD5 hash should be 32 characters"


def test_cleanup_singletons():
    """Test cleanup of singletons."""
    # Get initial references
    driver = get_graph_db_driver()
    entity_cache = get_entity_label_cache()
    embedding_cache = get_embedding_cache()
    retrieval_cache = get_retrieval_cache()
    
    # Add some data to caches
    entity_cache["test"] = "value"
    embedding_cache["test"] = [1.0]
    retrieval_cache["test"] = []
    
    # Cleanup
    cleanup_singletons()
    
    # After cleanup, new instances should be created
    new_driver = get_graph_db_driver()
    new_entity_cache = get_entity_label_cache()
    new_embedding_cache = get_embedding_cache()
    new_retrieval_cache = get_retrieval_cache()
    
    # Verify new instances
    assert new_driver is not driver, "Should create new driver after cleanup"
    assert new_entity_cache is not entity_cache, "Should create new entity cache after cleanup"
    assert new_embedding_cache is not embedding_cache, "Should create new embedding cache after cleanup"
    assert new_retrieval_cache is not retrieval_cache, "Should create new retrieval cache after cleanup"
    
    # Verify caches are empty
    assert len(new_entity_cache) == 0, "New entity cache should be empty"
    assert len(new_embedding_cache) == 0, "New embedding cache should be empty"
    assert len(new_retrieval_cache) == 0, "New retrieval cache should be empty"


def test_cache_thread_safety():
    """Test that cache operations are thread-safe."""
    cache = get_embedding_cache()
    cache.clear()
    
    results = []
    errors = []
    
    def cache_operations():
        try:
            # Write operation
            cache[f"key_{threading.current_thread().name}"] = [1.0, 2.0, 3.0]
            
            # Read operation
            if "key_Thread-1" in cache:
                results.append(cache["key_Thread-1"])
        except Exception as e:
            errors.append(e)
    
    # Create multiple threads
    threads = [threading.Thread(target=cache_operations, name=f"Thread-{i}") for i in range(10)]
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # No errors should occur
    assert len(errors) == 0, f"Thread-safe operations should not cause errors: {errors}"
    
    # All keys should be written
    assert len(cache) == 10, "All threads should successfully write to cache"
