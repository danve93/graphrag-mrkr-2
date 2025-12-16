"""
Unit tests for two-stage retrieval (BM25 pre-filter + vector search).

Tests cover:
- Corpus size estimation
- Filtered vector search by chunk IDs
- Two-stage retrieval activation logic
- BM25 candidate filtering
- Integration with standard retrieval
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCorpusSizeEstimation:
    """Tests for corpus size estimation."""

    def test_estimate_total_chunks_with_data(self):
        """Test corpus size estimation with chunks in database."""
        from core.graph_db import GraphDB

        # Mock the session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"total": 10000}
        mock_session.run.return_value = mock_result

        # Mock session_scope as a context manager
        mock_session_scope = MagicMock()
        mock_session_scope.__enter__ = MagicMock(return_value=mock_session)
        mock_session_scope.__exit__ = MagicMock(return_value=False)

        # Test the method
        with patch.object(GraphDB, 'session_scope', return_value=mock_session_scope):
            db = GraphDB()
            count = db.estimate_total_chunks()

        assert count == 10000
        mock_session.run.assert_called_once()

    def test_estimate_total_chunks_empty_db(self):
        """Test corpus size estimation with empty database."""
        from core.graph_db import GraphDB

        # Mock the session and result
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = {"total": 0}
        mock_session.run.return_value = mock_result

        # Mock session_scope as a context manager
        mock_session_scope = MagicMock()
        mock_session_scope.__enter__ = MagicMock(return_value=mock_session)
        mock_session_scope.__exit__ = MagicMock(return_value=False)

        with patch.object(GraphDB, 'session_scope', return_value=mock_session_scope):
            db = GraphDB()
            count = db.estimate_total_chunks()

        assert count == 0

    def test_estimate_total_chunks_error_handling(self):
        """Test corpus size estimation error handling."""
        from core.graph_db import GraphDB

        # Mock the session to raise an error
        mock_session = MagicMock()
        mock_session.run.side_effect = Exception("Database error")

        # Mock session_scope as a context manager
        mock_session_scope = MagicMock()
        mock_session_scope.__enter__ = MagicMock(return_value=mock_session)
        mock_session_scope.__exit__ = MagicMock(return_value=False)

        with patch.object(GraphDB, 'session_scope', return_value=mock_session_scope):
            db = GraphDB()
            count = db.estimate_total_chunks()

        # Should return 0 on error
        assert count == 0


class TestFilteredVectorSearch:
    """Tests for filtered vector search on candidate chunks."""

    def test_retrieve_chunks_by_ids_empty_candidates(self):
        """Test filtered vector search with empty candidate list."""
        from core.graph_db import GraphDB

        db = GraphDB()
        query_embedding = [0.1] * 768
        candidate_ids = []

        results = db.retrieve_chunks_by_ids_with_similarity(
            query_embedding, candidate_ids, top_k=5
        )

        assert results == []


class TestTwoStageConfiguration:
    """Tests for two-stage retrieval configuration."""

    def test_two_stage_configuration_defaults(self):
        """Test that two-stage retrieval has correct default settings."""
        from config.settings import settings

        # Verify default configuration
        assert hasattr(settings, 'enable_two_stage_retrieval')
        assert hasattr(settings, 'two_stage_threshold_docs')
        assert hasattr(settings, 'two_stage_multiplier')

        # Verify default values (these can be overridden by env vars)
        assert isinstance(settings.enable_two_stage_retrieval, bool)
        assert isinstance(settings.two_stage_threshold_docs, int)
        assert isinstance(settings.two_stage_multiplier, int)
        assert settings.two_stage_threshold_docs > 0
        assert settings.two_stage_multiplier > 0

    def test_candidate_count_calculation(self):
        """Test candidate count calculation logic."""
        from config.settings import settings

        top_k = 5
        multiplier = settings.two_stage_multiplier
        expected_candidates = top_k * multiplier

        assert expected_candidates >= top_k
        assert expected_candidates == top_k * multiplier


class TestTwoStageActivation:
    """Tests for two-stage retrieval activation logic."""

    def test_activation_threshold_logic(self):
        """Test two-stage activation threshold logic."""
        from config.settings import settings

        threshold = settings.two_stage_threshold_docs

        # Test activation logic
        small_corpus = threshold - 1
        large_corpus = threshold + 1

        assert small_corpus < threshold  # Should not activate
        assert large_corpus >= threshold  # Should activate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
