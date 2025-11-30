"""Unit tests for GraphDB caching layer."""

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock
from core.graph_db import graph_db


def test_entity_label_cache_hit():
    """Test that repeated entity label lookups use cache."""
    entity_id = "test_entity_123"
    
    # Clear cache to start fresh
    graph_db._entity_label_cache.clear()
    
    # First call - should query database
    with patch.object(graph_db, '_get_entity_label_direct', return_value="Test Entity") as mock_query:
        label1 = graph_db.get_entity_label_cached(entity_id)
        assert mock_query.call_count == 1
        assert label1 == "Test Entity"
    
    # Second call - should use cache (no database query)
    with patch.object(graph_db, '_get_entity_label_direct', return_value="Test Entity") as mock_query:
        label2 = graph_db.get_entity_label_cached(entity_id)
        assert mock_query.call_count == 0  # No database call
        assert label2 == "Test Entity"


def test_entity_label_cache_miss():
    """Test cache miss behavior."""
    entity_id = "new_entity_456"
    
    # Clear cache to ensure miss
    graph_db._entity_label_cache.clear()
    
    with patch.object(graph_db, '_get_entity_label_direct', return_value="New Entity") as mock_query:
        label = graph_db.get_entity_label_cached(entity_id)
        assert mock_query.call_count == 1
        assert label == "New Entity"


def test_entity_label_cache_disabled():
    """Test that caching can be disabled via feature flag."""
    from config.settings import settings
    
    entity_id = "test_entity_789"
    original_setting = settings.enable_caching
    
    try:
        settings.enable_caching = False
        
        # Clear cache
        graph_db._entity_label_cache.clear()
        
        # Should call direct query every time
        with patch.object(graph_db, '_get_entity_label_direct', return_value="Test") as mock_query:
            graph_db.get_entity_label_cached(entity_id)
            graph_db.get_entity_label_cached(entity_id)
            assert mock_query.call_count == 2  # Both calls hit database
    
    finally:
        settings.enable_caching = original_setting


def test_entity_label_cache_thread_safety():
    """Test that cache access is thread-safe."""
    entity_id = "thread_test_entity"
    results = []
    
    # Clear cache
    graph_db._entity_label_cache.clear()
    
    # Mock the direct query to return consistent value
    with patch.object(graph_db, '_get_entity_label_direct', return_value="Thread Safe Entity"):
        def fetch_label():
            label = graph_db.get_entity_label_cached(entity_id)
            results.append(label)
        
        # Create multiple threads
        threads = [threading.Thread(target=fetch_label) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All results should be identical
        assert len(results) == 10, "All threads should complete"
        assert len(set(results)) == 1, "All threads should get same result"
        assert results[0] == "Thread Safe Entity"
