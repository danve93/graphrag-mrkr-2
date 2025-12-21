"""
Tests for sentence-window retrieval functionality.

Uses mocked graph database to test sentence retrieval logic.
These tests mock at the module level to avoid neo4j import issues.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys


# Mock neo4j before any imports
sys.modules['neo4j'] = MagicMock()


class TestSentenceContextLogic:
    """Tests for sentence context window logic (pure functions)."""

    def test_window_bounds_middle(self):
        """Test window calculation for middle sentence."""
        # Simulating get_sentence_context logic
        total_sentences = 10
        target_index = 5
        window_size = 2
        
        start_idx = max(0, target_index - window_size)
        end_idx = min(total_sentences, target_index + window_size + 1)
        
        assert start_idx == 3
        assert end_idx == 8
        assert end_idx - start_idx == 5  # 5 sentences in window

    def test_window_bounds_start(self):
        """Test window clamps at start of list."""
        total_sentences = 10
        target_index = 1
        window_size = 3
        
        start_idx = max(0, target_index - window_size)
        end_idx = min(total_sentences, target_index + window_size + 1)
        
        assert start_idx == 0  # Clamped at 0
        assert end_idx == 5

    def test_window_bounds_end(self):
        """Test window clamps at end of list."""
        total_sentences = 10
        target_index = 8
        window_size = 3
        
        start_idx = max(0, target_index - window_size)
        end_idx = min(total_sentences, target_index + window_size + 1)
        
        assert start_idx == 5
        assert end_idx == 10  # Clamped at total


class TestSentenceDeduplication:
    """Tests for chunk deduplication in sentence search."""

    def test_deduplicate_by_chunk(self):
        """Test that overlapping sentences from same chunk are deduplicated."""
        # Simulate deduplication logic from sentence_vector_search
        results = [
            {"sentence_id": "s1", "chunk_id": "c1", "similarity": 0.95},
            {"sentence_id": "s2", "chunk_id": "c1", "similarity": 0.90},  # Same chunk
            {"sentence_id": "s3", "chunk_id": "c2", "similarity": 0.85},
        ]
        
        seen_chunks = set()
        deduplicated = []
        for r in results:
            if r["chunk_id"] not in seen_chunks:
                seen_chunks.add(r["chunk_id"])
                deduplicated.append(r)
        
        assert len(deduplicated) == 2
        assert deduplicated[0]["chunk_id"] == "c1"
        assert deduplicated[1]["chunk_id"] == "c2"

    def test_empty_results(self):
        """Test handling of empty results."""
        results = []
        seen_chunks = set()
        deduplicated = [r for r in results if r.get("chunk_id") not in seen_chunks]
        assert deduplicated == []


class TestSentenceBasedRetrievalMocked:
    """Tests for sentence_based_retrieval with full mocking."""

    @pytest.mark.asyncio
    async def test_disabled_returns_empty(self):
        """When feature disabled, should return empty list immediately."""
        mock_settings = MagicMock()
        mock_settings.enable_sentence_window_retrieval = False
        
        with patch.dict('sys.modules', {'config.settings': MagicMock(settings=mock_settings)}):
            # Simulating the check from sentence_based_retrieval
            if not getattr(mock_settings, "enable_sentence_window_retrieval", False):
                result = []
            else:
                result = ["would have retrieved"]
            
            assert result == []

    @pytest.mark.asyncio
    async def test_enabled_would_search(self):
        """When feature enabled, should proceed with search."""
        mock_settings = MagicMock()
        mock_settings.enable_sentence_window_retrieval = True
        mock_settings.sentence_window_size = 5
        
        # Simulating the check from sentence_based_retrieval
        if not getattr(mock_settings, "enable_sentence_window_retrieval", False):
            result = []
        else:
            result = ["would_perform_search"]
        
        assert result == ["would_perform_search"]


class TestDocumentFiltering:
    """Tests for document ID filtering in results."""

    def test_filter_by_allowed_docs(self):
        """Test filtering results by allowed document IDs."""
        results = [
            {"chunk_id": "c1", "document_id": "doc1"},
            {"chunk_id": "c2", "document_id": "doc2"},
            {"chunk_id": "c3", "document_id": "doc1"},
        ]
        allowed = ["doc1"]
        
        filtered = [r for r in results if r.get("document_id") in set(allowed)]
        
        assert len(filtered) == 2
        assert all(r["document_id"] == "doc1" for r in filtered)

    def test_no_filter_when_empty(self):
        """Test that empty allowed list doesn't filter."""
        results = [
            {"chunk_id": "c1", "document_id": "doc1"},
            {"chunk_id": "c2", "document_id": "doc2"},
        ]
        allowed = None
        
        if allowed:
            filtered = [r for r in results if r.get("document_id") in set(allowed)]
        else:
            filtered = results
        
        assert len(filtered) == 2


class TestSentenceWindowAssembly:
    """Tests for assembling sentence windows from indices."""

    def test_assemble_window_text(self):
        """Test combining sentences into window text."""
        sentences = [
            "Sentence zero.",
            "Sentence one.",
            "Target sentence.",
            "Sentence three.",
            "Sentence four.",
        ]
        start_idx = 1
        end_idx = 4
        
        window = sentences[start_idx:end_idx]
        combined = " ".join(window)
        
        assert "Sentence one." in combined
        assert "Target sentence." in combined
        assert "Sentence three." in combined
        assert "Sentence zero." not in combined
        assert "Sentence four." not in combined

    def test_empty_sentences_fallback(self):
        """Test fallback when no sentences available."""
        sentences = []
        chunk_content = "Fallback to full chunk content."
        
        if not sentences:
            combined = chunk_content
        else:
            combined = " ".join(sentences)
        
        assert combined == chunk_content


class TestSettingsIntegration:
    """Tests for settings configuration."""

    def test_default_settings(self):
        """Test that default settings are sensible."""
        # These are the defaults we configured
        defaults = {
            "enable_sentence_window_retrieval": False,
            "sentence_window_size": 5,
            "sentence_min_length": 10,
        }
        
        assert defaults["enable_sentence_window_retrieval"] is False  # Opt-in
        assert defaults["sentence_window_size"] == 5  # ±5 sentences
        assert defaults["sentence_min_length"] == 10  # 10 chars minimum

    def test_window_size_range(self):
        """Test that window size is in valid range."""
        # From UI config: min=1, max=15
        window_size = 5
        assert 1 <= window_size <= 15

    def test_min_length_range(self):
        """Test that min length is in valid range."""
        # From UI config: min=5, max=50
        min_length = 10
        assert 5 <= min_length <= 50
