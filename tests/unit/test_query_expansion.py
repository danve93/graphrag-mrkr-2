"""
Unit tests for query expansion functionality.

Tests cover:
- Abbreviation expansion
- LLM-based synonym expansion
- Expansion triggering logic
- Score penalty application
- Deduplication
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rag.nodes.query_expansion import (
    expand_query,
    _expand_abbreviations,
    _expand_with_llm,
    should_expand_query,
    ABBREVIATION_MAP,
)


class TestAbbreviationExpansion:
    """Tests for rule-based abbreviation expansion."""

    def test_expand_single_abbreviation(self):
        """Test expansion of a single common abbreviation."""
        result = _expand_abbreviations("What is an API?")
        assert "application programming interface" in result

    def test_expand_multiple_abbreviations(self):
        """Test expansion of multiple abbreviations in one query."""
        result = _expand_abbreviations("Connect DB to API using REST")
        # Should contain expansions for db, api, and rest
        assert any("database" in exp for exp in result)
        assert any("application programming interface" in exp for exp in result)
        assert any("representational state transfer" in exp or "restful" in exp for exp in result)

    def test_case_insensitive_expansion(self):
        """Test that abbreviations are matched case-insensitively."""
        result_lower = _expand_abbreviations("api")
        result_upper = _expand_abbreviations("API")
        result_mixed = _expand_abbreviations("Api")

        # All should return the same expansion
        assert result_lower == result_upper == result_mixed

    def test_no_expansion_for_unknown_abbrev(self):
        """Test that unknown abbreviations return empty list."""
        result = _expand_abbreviations("xyz123 unknown")
        assert result == []

    def test_word_boundary_matching(self):
        """Test that abbreviations only match whole words."""
        # "api" in "apiary" should not match
        result = _expand_abbreviations("apiary db")
        # Should only get db expansion (database), not API (apiary doesn't contain "api" as a word)
        assert any("database" in exp for exp in result)
        assert not any("application programming interface" in exp for exp in result)

    def test_technical_abbreviations(self):
        """Test expansion of technical domain abbreviations."""
        test_cases = [
            ("k8s cluster", "kubernetes"),
            ("CI/CD pipeline", "continuous integration"),
            ("ML model", "machine learning"),
            ("NLP processing", "natural language processing"),
        ]

        for query, expected in test_cases:
            result = _expand_abbreviations(query)
            assert any(expected.lower() in exp.lower() for exp in result), \
                f"Expected '{expected}' in expansions for '{query}', got {result}"


class TestLLMExpansion:
    """Tests for LLM-based query expansion."""

    @patch("rag.nodes.query_expansion.llm_manager")
    def test_llm_expansion_basic(self, mock_llm):
        """Test basic LLM expansion."""
        # Mock LLM to return comma-separated terms
        mock_llm.generate_response.return_value = "search, find, lookup, query, retrieve"

        query_analysis = {
            "query_type": "factual",
            "key_concepts": ["search", "function"],
        }

        result = _expand_with_llm("how to search", query_analysis)

        # Should return list of expanded terms
        assert isinstance(result, list)
        assert len(result) > 0
        assert "search" in result or "find" in result

    @patch("rag.nodes.query_expansion.llm_manager")
    def test_llm_expansion_filters_original(self, mock_llm):
        """Test that LLM expansion filters out the original query."""
        mock_llm.generate_response.return_value = "database, how to search, db storage"

        query_analysis = {
            "query_type": "factual",
            "key_concepts": ["search"],
        }

        result = _expand_with_llm("how to search", query_analysis)

        # Should not contain the original query
        assert "how to search" not in result

    @patch("rag.nodes.query_expansion.llm_manager")
    def test_llm_expansion_limits_results(self, mock_llm):
        """Test that LLM expansion limits to 5 terms."""
        # Return more than 5 terms
        mock_llm.generate_response.return_value = "term1, term2, term3, term4, term5, term6, term7"

        result = _expand_with_llm("test query", {})

        # Should limit to 5
        assert len(result) <= 5

    @patch("rag.nodes.query_expansion.llm_manager")
    def test_llm_expansion_handles_error(self, mock_llm):
        """Test that LLM expansion handles errors gracefully."""
        mock_llm.generate_response.side_effect = Exception("LLM error")

        result = _expand_with_llm("test query", {})

        # Should return empty list on error
        assert result == []


class TestExpansionTriggering:
    """Tests for determining when expansion should be triggered."""

    def test_should_expand_sparse_results(self):
        """Test expansion triggers on sparse results."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True
            mock_settings.query_expansion_threshold = 3

            query_analysis = {
                "query_type": "factual",
                "is_technical": False,
            }

            # Should trigger with 2 results (< threshold of 3)
            assert should_expand_query(query_analysis, initial_results_count=2)

            # Should not trigger with 5 results (>= threshold)
            assert not should_expand_query(query_analysis, initial_results_count=5)

    def test_should_expand_technical_query(self):
        """Test expansion triggers for technical queries."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            query_analysis = {
                "query_type": "factual",
                "is_technical": True,  # Technical query
            }

            # Should trigger for technical queries
            assert should_expand_query(query_analysis)

    def test_should_expand_complex_analytical(self):
        """Test expansion triggers for complex analytical queries."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            query_analysis = {
                "query_type": "analytical",
                "complexity": "complex",
            }

            # Should trigger for complex analytical queries
            assert should_expand_query(query_analysis)

    def test_should_not_expand_when_disabled(self):
        """Test expansion doesn't trigger when disabled."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = False

            query_analysis = {
                "query_type": "factual",
                "is_technical": True,
            }

            # Should not trigger when disabled
            assert not should_expand_query(query_analysis)


class TestQueryExpansionIntegration:
    """Integration tests for full query expansion flow."""

    def test_expand_query_with_abbreviations(self):
        """Test end-to-end expansion with abbreviations."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            query_analysis = {
                "query_type": "factual",
                "key_concepts": ["api", "database"],
            }

            result = expand_query(
                query="How to connect API to DB?",
                query_analysis=query_analysis,
                max_expansions=5,
                use_llm=False,
            )

            # Should contain abbreviation expansions
            assert len(result) > 0
            assert any("database" in exp for exp in result)

    def test_expand_query_respects_max_expansions(self):
        """Test that expansion respects max_expansions limit."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            query_analysis = {
                "query_type": "factual",
                "key_concepts": [],
            }

            # Query with multiple abbreviations
            result = expand_query(
                query="API REST HTTP JSON XML DB SQL",
                query_analysis=query_analysis,
                max_expansions=3,
                use_llm=False,
            )

            # Should respect max_expansions
            assert len(result) <= 3

    @patch("rag.nodes.query_expansion._expand_with_llm")
    def test_expand_query_with_llm(self, mock_llm_expansion):
        """Test expansion with LLM enabled."""
        mock_llm_expansion.return_value = ["search", "find", "lookup"]

        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            query_analysis = {
                "query_type": "factual",
                "key_concepts": ["search"],
            }

            result = expand_query(
                query="how to search",
                query_analysis=query_analysis,
                max_expansions=5,
                use_llm=True,
            )

            # Should call LLM expansion
            mock_llm_expansion.assert_called_once()

    def test_expand_query_disabled(self):
        """Test that expansion returns empty when disabled."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = False

            result = expand_query(
                query="API database",
                query_analysis={},
                max_expansions=5,
                use_llm=False,
            )

            # Should return empty list when disabled
            assert result == []

    def test_expand_query_handles_errors(self):
        """Test that expansion handles errors gracefully."""
        with patch("rag.nodes.query_expansion.settings") as mock_settings:
            mock_settings.enable_query_expansion = True

            # Simulate error in abbreviation expansion
            with patch("rag.nodes.query_expansion._expand_abbreviations") as mock_abbrev:
                mock_abbrev.side_effect = Exception("Expansion error")

                result = expand_query(
                    query="API database",
                    query_analysis={},
                    max_expansions=5,
                    use_llm=False,
                )

                # Should return empty list on error
                assert result == []


class TestAbbreviationMapping:
    """Tests for abbreviation mapping dictionary coverage."""

    def test_common_abbreviations_exist(self):
        """Test that common technical abbreviations are in the map."""
        common_abbrevs = [
            "api", "rest", "sql", "db", "http", "json",
            "ai", "ml", "nlp", "k8s", "cicd"
        ]

        for abbrev in common_abbrevs:
            assert abbrev in ABBREVIATION_MAP, \
                f"Common abbreviation '{abbrev}' missing from ABBREVIATION_MAP"

    def test_abbreviation_values_are_lists(self):
        """Test that all abbreviation values are lists."""
        for abbrev, expansions in ABBREVIATION_MAP.items():
            assert isinstance(expansions, list), \
                f"Abbreviation '{abbrev}' has non-list value: {expansions}"
            assert len(expansions) > 0, \
                f"Abbreviation '{abbrev}' has empty expansion list"

    def test_abbreviation_values_lowercase(self):
        """Test that all expansions are lowercase for consistency."""
        for abbrev, expansions in ABBREVIATION_MAP.items():
            for expansion in expansions:
                assert expansion == expansion.lower(), \
                    f"Expansion '{expansion}' for '{abbrev}' is not lowercase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
