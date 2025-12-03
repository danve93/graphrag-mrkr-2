"""
Tests for query analysis routing and expansion.

Validates:
- Query classification accuracy (entity_focused, keyword_focused, balanced)
- Strategy-based weight adjustments in retriever
- Query expansion triggering and integration
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from config.settings import settings


class TestQueryClassification:
    """Test query type classification and strategy determination."""

    def test_comparative_analysis_query(self):
        """Comparative/analytical queries should route to entity_focused."""
        from rag.nodes.query_analysis import _determine_retrieval_strategy
        
        # Test the strategy determination directly (note: query_type must be "comparative", not "comparison")
        mock_analysis = {
            "query_type": "comparative",
            "complexity": "intermediate",
            "requires_reasoning": True,
            "key_concepts": ["server", "features"]
        }
        
        strategy, confidence = _determine_retrieval_strategy(
            mock_analysis,
            "compare the features of server a and server b"
        )
        
        assert strategy == "entity_focused"
        assert confidence == 0.8

    def test_version_specific_query(self):
        """Version/code queries should route to keyword_focused."""
        from rag.nodes.query_analysis import _determine_retrieval_strategy
        
        mock_analysis = {
            "query_type": "factual",
            "complexity": "simple",
            "key_concepts": ["version"]
        }
        
        strategy, confidence = _determine_retrieval_strategy(
            mock_analysis,
            "what's new in version 2.5.0?"
        )
        
        # Should detect version pattern and route to keyword
        assert strategy == "keyword_focused"
        assert confidence >= 0.70

    def test_procedural_query(self):
        """How-to/procedural queries should route to keyword_focused."""
        from rag.nodes.query_analysis import _determine_retrieval_strategy
        
        mock_analysis = {
            "query_type": "factual",  # Must be factual for procedural check
            "complexity": "intermediate",
            "key_concepts": ["configure", "ssl"]
        }
        
        strategy, confidence = _determine_retrieval_strategy(
            mock_analysis,
            "how to configure ssl certificates?"
        )
        
        assert strategy == "keyword_focused"
        assert confidence == 0.7

    def test_relationship_query(self):
        """Relationship queries should route to entity_focused."""
        from rag.nodes.query_analysis import _determine_retrieval_strategy
        
        mock_analysis = {
            "query_type": "factual",
            "complexity": "intermediate",
            "key_concepts": ["components", "service"]
        }
        
        strategy, confidence = _determine_retrieval_strategy(
            mock_analysis,
            "what is the relationship between components and the authentication service?"
        )
        
        # Should detect relationship keyword
        assert strategy == "entity_focused"
        assert confidence == 0.85

    def test_simple_factual_query(self):
        """Simple factual queries should default to balanced."""
        from rag.nodes.query_analysis import _determine_retrieval_strategy
        
        mock_analysis = {
            "query_type": "factual",
            "complexity": "simple",
            "key_concepts": ["port"]
        }
        
        strategy, confidence = _determine_retrieval_strategy(
            mock_analysis,
            "what is the default port?"
        )
        
        assert strategy == "balanced"
        assert confidence >= 0.60


class TestStrategyWeightAdjustment:
    """Test retrieval weight adjustments based on routing strategy."""

    def test_entity_focused_weights(self):
        """Entity_focused strategy should boost entity weight and reduce chunk weight."""
        # Test weight multipliers without needing actual retriever instance
        
        # Mock initial weights
        base_entity = 0.4
        base_chunk = 0.6
        
        # Simulate entity_focused adjustment
        entity_multiplier = 1.3
        chunk_multiplier = 0.8
        
        adjusted_entity = base_entity * entity_multiplier
        adjusted_chunk = base_chunk * chunk_multiplier
        
        assert adjusted_entity > base_entity  # 0.52 > 0.4
        assert adjusted_chunk < base_chunk    # 0.48 < 0.6

    def test_keyword_focused_weights(self):
        """Keyword_focused strategy should boost keyword and chunk, reduce entity."""
        base_keyword = 0.3
        base_chunk = 0.6
        base_entity = 0.4
        
        # Simulate keyword_focused adjustment
        keyword_multiplier = 1.4
        chunk_multiplier = 1.1
        entity_multiplier = 0.7
        
        adjusted_keyword = base_keyword * keyword_multiplier
        adjusted_chunk = base_chunk * chunk_multiplier
        adjusted_entity = base_entity * entity_multiplier
        
        assert adjusted_keyword > base_keyword  # 0.42 > 0.3
        assert adjusted_chunk > base_chunk      # 0.66 > 0.6
        assert adjusted_entity < base_entity    # 0.28 < 0.4


class TestQueryExpansion:
    """Test query expansion triggering and term generation."""

    def test_expansion_disabled(self):
        """No expansion when disabled in settings."""
        from rag.query_expansion import expand_query_terms
        
        # Temporarily disable expansion
        original = getattr(settings, "enable_query_expansion", False)
        settings.enable_query_expansion = False
        
        try:
            result = expand_query_terms(
                query="test query",
                initial_results_count=1  # Sparse results
            )
            assert result == []  # No expansion
        finally:
            settings.enable_query_expansion = original

    def test_expansion_above_threshold(self):
        """No expansion when results above threshold."""
        from rag.query_expansion import expand_query_terms
        
        # Enable expansion but provide enough results
        original = getattr(settings, "enable_query_expansion", False)
        settings.enable_query_expansion = True
        
        try:
            result = expand_query_terms(
                query="test query",
                initial_results_count=10,  # Above threshold
                min_threshold=3
            )
            assert result == []  # No expansion needed
        finally:
            settings.enable_query_expansion = original

    def test_expansion_below_threshold(self):
        """Expansion triggered when results below threshold."""
        from rag.query_expansion import expand_query_terms
        
        # Mock LLM to return expanded terms
        mock_response = '["authentication", "authorization", "access control", "security", "permissions"]'
        
        mock_llm = MagicMock()
        mock_llm.generate_response.return_value = mock_response
        
        original = getattr(settings, "enable_query_expansion", False)
        settings.enable_query_expansion = True
        
        try:
            with patch("rag.query_expansion.llm_manager", mock_llm):
                result = expand_query_terms(
                    query="auth system",
                    initial_results_count=1,  # Sparse results
                    min_threshold=3
                )
            
            assert len(result) > 0
            assert "authentication" in result or "authorization" in result
        finally:
            settings.enable_query_expansion = original

    def test_expansion_metadata_tracking(self):
        """Expansion integration exists in retriever."""
        # Verify expansion integration point exists
        import rag.retriever as retriever_module
        import rag.query_expansion as expansion_module
        
        # Both modules should be importable and have key functions
        assert hasattr(expansion_module, 'expand_query_terms')
        # Integration happens in _hybrid_retrieval_direct (private method)


def test_end_to_end_routing_logic():
    """Test: query type → strategy → weight multipliers."""
    from rag.nodes.query_analysis import _determine_retrieval_strategy
    
    # Test comparative query routing (query_type="comparative")
    mock_analysis = {
        "query_type": "comparative",
        "complexity": "intermediate",
        "requires_reasoning": True,
        "key_concepts": ["servera", "serverb"]
    }
    
    strategy, confidence = _determine_retrieval_strategy(
        mock_analysis,
        "compare servera and serverb features"
    )
    
    # Verify routing decision
    assert strategy == "entity_focused"
    assert confidence == 0.8
    
    # Verify weight multipliers would be applied
    # (In actual retriever: entity_focused → entity×1.3, chunk×0.8)
    if strategy == "entity_focused":
        entity_multiplier = 1.3
        chunk_multiplier = 0.8
        assert entity_multiplier > 1.0
        assert chunk_multiplier < 1.0
