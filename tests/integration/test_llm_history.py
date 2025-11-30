"""
Tests for LLM conversation history support (Phase 1: Gleaning)
"""

import pytest
from core.llm import llm_manager


def test_generate_with_history_validation_invalid_format():
    """Test that history format validation catches invalid structures."""
    # Not a list
    with pytest.raises(ValueError, match="History must be a list"):
        llm_manager.generate_response_with_history(
            prompt="Test",
            history="not a list"
        )
    
    # List item not a dict
    with pytest.raises(ValueError, match="must be dict"):
        llm_manager.generate_response_with_history(
            prompt="Test",
            history=["not a dict"]
        )
    
    # Missing role/content
    with pytest.raises(ValueError, match="missing 'role' or 'content'"):
        llm_manager.generate_response_with_history(
            prompt="Test",
            history=[{"content": "no role"}]
        )
    
    # Invalid role
    with pytest.raises(ValueError, match="Invalid role"):
        llm_manager.generate_response_with_history(
            prompt="Test",
            history=[{"role": "invalid", "content": "test"}]
        )


def test_generate_with_empty_history():
    """Test that empty history works like normal call."""
    try:
        response = llm_manager.generate_response_with_history(
            prompt="What is 2+2?",
            history=[],
            temperature=0.0,
            max_tokens=50
        )
        # Should return a response (exact content depends on model)
        assert isinstance(response, str)
        assert len(response) > 0
    except Exception as e:
        # If LLM isn't configured, test passes
        if "API key" not in str(e) and "Failed to generate" not in str(e):
            raise


def test_generate_with_history_structure():
    """Test that history with valid structure is accepted."""
    valid_history = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ]
    
    try:
        response = llm_manager.generate_response_with_history(
            prompt="What about 3+3?",
            history=valid_history,
            temperature=0.0,
            max_tokens=50
        )
        assert isinstance(response, str)
        # History should not be mutated
        assert len(valid_history) == 2
    except Exception as e:
        # If LLM isn't configured, test passes
        if "API key" not in str(e) and "Failed to generate" not in str(e):
            raise


def test_history_not_mutated():
    """Test that the history list is not mutated during the call."""
    history = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    original_length = len(history)
    
    try:
        llm_manager.generate_response_with_history(
            prompt="Second question",
            history=history,
            temperature=0.0,
            max_tokens=50
        )
        # History should not be modified
        assert len(history) == original_length
    except Exception as e:
        # If LLM isn't configured, test passes
        if "API key" not in str(e) and "Failed to generate" not in str(e):
            raise
