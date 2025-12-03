"""
Tests for Milestone 2: Enhanced Cache Key Generation.

Verifies that cache keys properly include all parameters that affect
retrieval results, preventing stale cache hits when parameters differ.
"""

import pytest
from core.singletons import hash_retrieval_params


class TestCacheKeyParameterIsolation:
    """Test that different parameter values produce different cache keys."""

    def test_different_queries_produce_different_keys(self):
        """Different queries should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        key2 = hash_retrieval_params(
            query="What is NSX-T?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        assert key1 != key2, "Different queries should produce different cache keys"

    def test_different_top_k_produces_different_keys(self):
        """Different top_k values should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=10,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        assert key1 != key2, "Different top_k should produce different cache keys"

    def test_different_weights_produce_different_keys(self):
        """Different weight values should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.5,  # Different
            entity_weight=0.5,  # Different
            path_weight=0.2,
        )
        assert key1 != key2, "Different weights should produce different cache keys"

    def test_multi_hop_parameters_affect_cache_key(self):
        """Multi-hop parameters should be included in cache key."""
        # Without multi-hop
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=False,
        )
        # With multi-hop
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=True,
            max_hops=2,
            beam_size=10,
        )
        assert key1 != key2, "Multi-hop enabled should produce different cache key"

    def test_different_max_hops_produces_different_keys(self):
        """Different max_hops values should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=True,
            max_hops=2,
            beam_size=10,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=True,
            max_hops=3,  # Different
            beam_size=10,
        )
        assert key1 != key2, "Different max_hops should produce different cache keys"

    def test_different_beam_size_produces_different_keys(self):
        """Different beam_size values should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=True,
            max_hops=2,
            beam_size=10,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            use_multi_hop=True,
            max_hops=2,
            beam_size=20,  # Different
        )
        assert key1 != key2, "Different beam_size should produce different cache keys"


class TestCacheKeyDocumentFilters:
    """Test that document filter parameters affect cache keys."""

    def test_restrict_to_context_affects_cache_key(self):
        """restrict_to_context flag should be included in cache key."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=False,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=True,
        )
        assert key1 != key2, "restrict_to_context should affect cache key"

    def test_different_allowed_documents_produce_different_keys(self):
        """Different allowed_document_ids should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=True,
            allowed_document_ids=["doc1", "doc2"],
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=True,
            allowed_document_ids=["doc3", "doc4"],  # Different documents
        )
        assert key1 != key2, "Different allowed_document_ids should produce different cache keys"

    def test_same_documents_different_order_produce_same_key(self):
        """Same documents in different order should produce same cache key (sorted)."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=True,
            allowed_document_ids=["doc1", "doc2", "doc3"],
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            restrict_to_context=True,
            allowed_document_ids=["doc3", "doc1", "doc2"],  # Same docs, different order
        )
        assert key1 == key2, "Same documents in different order should produce same cache key"

    def test_no_allowed_documents_vs_empty_list(self):
        """None vs empty list should produce same cache key."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            allowed_document_ids=None,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            allowed_document_ids=[],  # Empty list
        )
        assert key1 == key2, "None and empty list should produce same cache key"


class TestCacheKeyModelParameters:
    """Test that model-related parameters affect cache keys."""

    def test_different_embedding_models_produce_different_keys(self):
        """Different embedding models should have different cache keys."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            embedding_model="text-embedding-3-small",
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            embedding_model="text-embedding-3-large",  # Different model
        )
        assert key1 != key2, "Different embedding models should produce different cache keys"

    def test_rrf_enabled_affects_cache_key(self):
        """RRF fusion flag should be included in cache key."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            enable_rrf=False,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            enable_rrf=True,
        )
        assert key1 != key2, "enable_rrf should affect cache key"

    def test_flashrank_enabled_affects_cache_key(self):
        """FlashRank reranking flag should be included in cache key."""
        key1 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            flashrank_enabled=False,
        )
        key2 = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
            flashrank_enabled=True,
        )
        assert key1 != key2, "flashrank_enabled should affect cache key"


class TestCacheKeyStability:
    """Test that cache keys are stable and repeatable."""

    def test_same_parameters_produce_same_key(self):
        """Identical parameters should always produce the same cache key."""
        params = {
            "query": "What is VxRail?",
            "mode": "hybrid",
            "top_k": 5,
            "chunk_weight": 0.7,
            "entity_weight": 0.3,
            "path_weight": 0.2,
            "use_multi_hop": True,
            "max_hops": 2,
            "beam_size": 10,
            "restrict_to_context": True,
            "allowed_document_ids": ["doc1", "doc2"],
            "embedding_model": "text-embedding-3-small",
            "enable_rrf": True,
            "flashrank_enabled": True,
        }
        
        # Generate key multiple times
        key1 = hash_retrieval_params(**params)
        key2 = hash_retrieval_params(**params)
        key3 = hash_retrieval_params(**params)
        
        assert key1 == key2 == key3, "Same parameters should always produce same cache key"

    def test_cache_key_is_string(self):
        """Cache key should be a string."""
        key = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        assert isinstance(key, str), "Cache key should be a string"

    def test_cache_key_is_md5_hash(self):
        """Cache key should be a 32-character MD5 hash."""
        key = hash_retrieval_params(
            query="What is VxRail?",
            mode="hybrid",
            top_k=5,
            chunk_weight=0.7,
            entity_weight=0.3,
            path_weight=0.2,
        )
        assert len(key) == 32, "MD5 hash should be 32 characters"
        assert key.isalnum(), "MD5 hash should be alphanumeric"
