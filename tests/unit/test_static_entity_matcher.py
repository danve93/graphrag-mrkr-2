"""
Unit tests for client-side vector matching for static entities.

Tests cover:
- StaticEntityMatcher class functionality (load, match, explain)
- Integration with query routing
- Configuration settings
- Error handling and fallbacks
"""

import gzip
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from core.static_entity_matcher import StaticEntityMatcher, get_static_matcher
from rag.nodes.query_router import route_query_to_categories


# Sample test data
SAMPLE_EMBEDDINGS_DATA = {
    "version": "1.0",
    "model": "text-embedding-3-small",
    "dimension": 1536,
    "categories": [
        {
            "id": "install",
            "title": "Installation",
            "description": "Setup and installation guides",
            "keywords": ["install", "setup", "prerequisites"],
            "embedding": [0.1] * 1536,  # Mock embedding
        },
        {
            "id": "configure",
            "title": "Configuration",
            "description": "Configuration and settings",
            "keywords": ["config", "settings", "options"],
            "embedding": [0.2] * 1536,  # Mock embedding
        },
        {
            "id": "troubleshoot",
            "title": "Troubleshooting",
            "description": "Error resolution and debugging",
            "keywords": ["error", "fix", "debug", "troubleshoot"],
            "embedding": [0.3] * 1536,  # Mock embedding
        },
    ],
}


@pytest.fixture
def temp_embeddings_file():
    """Create a temporary embeddings file for testing."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json.gz", delete=False) as f:
        json_bytes = json.dumps(SAMPLE_EMBEDDINGS_DATA).encode("utf-8")
        with gzip.open(f.name, "wb") as gz:
            gz.write(json_bytes)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def temp_embeddings_file_uncompressed():
    """Create a temporary uncompressed embeddings file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(SAMPLE_EMBEDDINGS_DATA, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


class TestStaticEntityMatcherLoad:
    """Test loading embeddings from files."""

    def test_load_compressed_file(self, temp_embeddings_file):
        """Test loading gzip-compressed embeddings."""
        matcher = StaticEntityMatcher()
        success = matcher.load(temp_embeddings_file)

        assert success is True
        assert matcher.is_loaded is True
        assert matcher.dimension == 1536
        assert matcher.model == "text-embedding-3-small"
        assert len(matcher.entities) == 3
        assert matcher.embeddings_matrix.shape == (3, 1536)

    def test_load_uncompressed_file(self, temp_embeddings_file_uncompressed):
        """Test loading uncompressed JSON embeddings."""
        matcher = StaticEntityMatcher()
        success = matcher.load(temp_embeddings_file_uncompressed)

        assert success is True
        assert matcher.is_loaded is True
        assert len(matcher.entities) == 3

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        matcher = StaticEntityMatcher()

        with pytest.raises(FileNotFoundError):
            matcher.load(Path("/nonexistent/embeddings.json.gz"))

    def test_load_invalid_format(self):
        """Test loading file with invalid format raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"invalid": "format"}, f)
            temp_path = Path(f.name)

        matcher = StaticEntityMatcher()

        with pytest.raises(ValueError, match="missing 'categories' field"):
            matcher.load(temp_path)

        temp_path.unlink()

    def test_auto_load_on_init(self, temp_embeddings_file):
        """Test that matcher auto-loads if path provided."""
        matcher = StaticEntityMatcher(embeddings_path=temp_embeddings_file)

        assert matcher.is_loaded is True
        assert len(matcher.entities) == 3

    def test_normalized_embeddings(self, temp_embeddings_file):
        """Test that embeddings are normalized after loading."""
        matcher = StaticEntityMatcher(embeddings_path=temp_embeddings_file)

        # Check that each embedding has norm ~1.0
        norms = np.linalg.norm(matcher.embeddings_matrix, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)


class TestStaticEntityMatcherMatch:
    """Test matching queries against static entities."""

    @pytest.fixture
    def loaded_matcher(self, temp_embeddings_file):
        """Provide a pre-loaded matcher."""
        return StaticEntityMatcher(embeddings_path=temp_embeddings_file)

    @pytest.mark.asyncio
    async def test_match_async_basic(self, loaded_matcher):
        """Test basic async matching."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            # Mock query embedding (similar to first category)
            mock_embed.return_value = [0.1] * 1536

            results = await loaded_matcher.match_async("how to install", top_k=3)

            assert len(results) > 0
            assert results[0]["id"] in ["install", "configure", "troubleshoot"]
            assert "similarity" in results[0]
            assert 0.0 <= results[0]["similarity"] <= 1.0

    @pytest.mark.asyncio
    async def test_match_async_top_k(self, loaded_matcher):
        """Test that top_k limits results."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            mock_embed.return_value = [0.15] * 1536

            results = await loaded_matcher.match_async("test query", top_k=2)

            assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_match_async_min_similarity(self, loaded_matcher):
        """Test that min_similarity filters results."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            results = await loaded_matcher.match_async(
                "test query", top_k=3, min_similarity=0.99
            )

            # High threshold should filter most results
            for result in results:
                assert result["similarity"] >= 0.99

    @pytest.mark.asyncio
    async def test_match_async_empty_query(self, loaded_matcher):
        """Test that empty query returns empty results."""
        results = await loaded_matcher.match_async("", top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_match_async_not_loaded(self):
        """Test that matching without loading raises error."""
        matcher = StaticEntityMatcher()

        with pytest.raises(RuntimeError, match="not loaded"):
            await matcher.match_async("test query")

    def test_match_sync_wrapper(self, loaded_matcher):
        """Test synchronous match wrapper."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            mock_embed.return_value = [0.2] * 1536

            results = loaded_matcher.match("configure settings", top_k=3)

            assert len(results) > 0
            assert isinstance(results, list)


class TestStaticEntityMatcherHelpers:
    """Test helper methods (get_entity, get_all_entities, explain_match)."""

    @pytest.fixture
    def loaded_matcher(self, temp_embeddings_file):
        """Provide a pre-loaded matcher."""
        return StaticEntityMatcher(embeddings_path=temp_embeddings_file)

    def test_get_entity_by_id(self, loaded_matcher):
        """Test retrieving entity by ID."""
        entity = loaded_matcher.get_entity("install")

        assert entity is not None
        assert entity["id"] == "install"
        assert entity["title"] == "Installation"
        assert "keywords" in entity
        assert "embedding" not in entity  # Should not include embedding

    def test_get_entity_not_found(self, loaded_matcher):
        """Test retrieving nonexistent entity."""
        entity = loaded_matcher.get_entity("nonexistent")
        assert entity is None

    def test_get_all_entities(self, loaded_matcher):
        """Test retrieving all entities."""
        entities = loaded_matcher.get_all_entities()

        assert len(entities) == 3
        assert all("id" in e for e in entities)
        assert all("title" in e for e in entities)
        assert all("embedding" not in e for e in entities)

    def test_explain_match(self, loaded_matcher):
        """Test match explanation."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            explanation = loaded_matcher.explain_match("how to install", "install")

            assert explanation["entity_id"] == "install"
            assert "similarity" in explanation
            assert "query" in explanation
            assert "matched_keywords" in explanation

    def test_explain_match_keyword_matching(self, loaded_matcher):
        """Test that keyword matching works in explanation."""
        with patch("core.embeddings.embedding_manager.aget_embedding") as mock_embed:
            mock_embed.return_value = [0.1] * 1536

            explanation = loaded_matcher.explain_match("setup prerequisites", "install")

            # Should match "setup" and "prerequisites" keywords
            assert len(explanation["matched_keywords"]) >= 1


class TestGlobalMatcherSingleton:
    """Test global matcher singleton pattern."""

    def test_get_static_matcher_singleton(self):
        """Test that get_static_matcher returns singleton."""
        with patch("core.static_entity_matcher.StaticEntityMatcher") as mock_class:
            mock_instance = MagicMock()
            mock_instance.is_loaded = False
            mock_class.return_value = mock_instance

            # Reset global singleton
            import core.static_entity_matcher as sem
            sem._static_matcher = None

            matcher1 = get_static_matcher()
            matcher2 = get_static_matcher()

            # Should return same instance
            assert matcher1 is matcher2
            # Should only instantiate once
            assert mock_class.call_count == 1


class TestQueryRoutingIntegration:
    """Test integration of static matcher with query routing."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for routing tests."""
        with patch("rag.nodes.query_router.settings") as mock_settings:
            mock_settings.enable_routing_cache = False
            mock_settings.enable_query_routing = True
            mock_settings.enable_static_entity_matching = True
            mock_settings.static_matching_min_similarity = 0.6
            yield mock_settings

    def test_static_routing_success(self, mock_settings):
        """Test successful routing via static matcher."""
        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            # Mock static matcher
            mock_matcher = MagicMock()
            mock_matcher.is_loaded = True
            mock_matcher.match.return_value = [
                {"id": "install", "title": "Installation", "similarity": 0.85}
            ]
            mock_get_matcher.return_value = mock_matcher

            result = route_query_to_categories(
                query="how to install",
                query_analysis={"query_type": "factual"},
                confidence_threshold=0.7,
            )

            assert result["categories"] == ["install"]
            assert result["confidence"] == 0.85
            assert result["should_filter"] is True
            assert "static_match" in result["reasoning"]

    def test_static_routing_below_threshold(self, mock_settings):
        """Test that low-confidence static match falls back to LLM."""
        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            with patch("rag.nodes.query_router.llm_manager") as mock_llm:
                # Mock static matcher with low confidence
                mock_matcher = MagicMock()
                mock_matcher.is_loaded = True
                mock_matcher.match.return_value = [
                    {"id": "install", "title": "Installation", "similarity": 0.5}
                ]
                mock_get_matcher.return_value = mock_matcher

                # Mock LLM fallback
                mock_llm.generate_response.return_value = json.dumps({
                    "categories": ["configure"],
                    "confidence": 0.9,
                    "reasoning": "llm routing"
                })

                result = route_query_to_categories(
                    query="how to setup",
                    query_analysis={"query_type": "factual"},
                    confidence_threshold=0.7,
                )

                # Should use LLM result (static match below threshold)
                assert result["categories"] == ["configure"]
                assert result["confidence"] == 0.9

    def test_static_routing_disabled(self, mock_settings):
        """Test that routing falls back to LLM when static matching disabled."""
        mock_settings.enable_static_entity_matching = False

        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            with patch("rag.nodes.query_router.llm_manager") as mock_llm:
                mock_llm.generate_response.return_value = json.dumps({
                    "categories": ["general"],
                    "confidence": 0.8,
                    "reasoning": "llm routing"
                })

                result = route_query_to_categories(
                    query="test query",
                    query_analysis={"query_type": "factual"},
                )

                # Should not call static matcher
                mock_get_matcher.assert_not_called()
                # Should use LLM
                assert result["categories"] == ["general"]

    def test_static_routing_not_loaded(self, mock_settings):
        """Test fallback when static matcher not loaded."""
        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            with patch("rag.nodes.query_router.llm_manager") as mock_llm:
                # Mock static matcher not loaded
                mock_matcher = MagicMock()
                mock_matcher.is_loaded = False
                mock_get_matcher.return_value = mock_matcher

                mock_llm.generate_response.return_value = json.dumps({
                    "categories": ["troubleshoot"],
                    "confidence": 0.75,
                    "reasoning": "llm routing"
                })

                result = route_query_to_categories(
                    query="error message",
                    query_analysis={"query_type": "factual"},
                )

                # Should fall back to LLM
                assert result["categories"] == ["troubleshoot"]

    def test_static_routing_exception_handling(self, mock_settings):
        """Test that exceptions in static matcher are handled gracefully."""
        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            with patch("rag.nodes.query_router.llm_manager") as mock_llm:
                # Mock static matcher raising exception
                mock_matcher = MagicMock()
                mock_matcher.is_loaded = True
                mock_matcher.match.side_effect = Exception("Test error")
                mock_get_matcher.return_value = mock_matcher

                mock_llm.generate_response.return_value = json.dumps({
                    "categories": ["general"],
                    "confidence": 0.6,
                    "reasoning": "fallback"
                })

                result = route_query_to_categories(
                    query="test query",
                    query_analysis={"query_type": "factual"},
                )

                # Should handle exception and fall back to LLM
                assert result["categories"] == ["general"]
                assert result["confidence"] == 0.6

    def test_static_routing_with_multiple_matches(self, mock_settings):
        """Test that alternative matches are included in reasoning."""
        with patch("rag.nodes.query_router.get_static_matcher") as mock_get_matcher:
            # Mock static matcher with multiple matches
            mock_matcher = MagicMock()
            mock_matcher.is_loaded = True
            mock_matcher.match.return_value = [
                {"id": "install", "title": "Installation", "similarity": 0.85},
                {"id": "configure", "title": "Configuration", "similarity": 0.72},
                {"id": "troubleshoot", "title": "Troubleshooting", "similarity": 0.65},
            ]
            mock_get_matcher.return_value = mock_matcher

            result = route_query_to_categories(
                query="setup and configure",
                query_analysis={"query_type": "factual"},
                confidence_threshold=0.7,
            )

            assert result["categories"] == ["install"]
            assert "alternatives" in result["reasoning"]
            assert "configure" in result["reasoning"]
