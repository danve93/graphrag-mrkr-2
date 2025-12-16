"""Test evaluation feature flag overrides."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from api.models import ChatRequest
from api.routers.chat import _prepare_chat_context
from config.settings import settings


@pytest.mark.asyncio
async def test_eval_override_applies_and_restores():
    """Test override is applied during query and restored after."""
    original_routing = settings.enable_query_routing
    original_rrf = settings.enable_rrf

    # Set initial values
    settings.enable_query_routing = False
    settings.enable_rrf = False

    request = ChatRequest(
        message="test query",
        eval_enable_query_routing=True,  # Override to True
        eval_enable_rrf=True  # Override to True
    )

    captured_routing = None
    captured_rrf = None

    def mock_query(**kwargs):
        nonlocal captured_routing, captured_rrf
        captured_routing = settings.enable_query_routing
        captured_rrf = settings.enable_rrf
        return {
            "response": "test response",
            "sources": [],
            "metadata": {},
            "stages": []
        }

    # Mock graph_rag.query and chat history service
    with patch("api.routers.chat.graph_rag.query", side_effect=mock_query):
        with patch("api.routers.chat.chat_history_service.get_conversation", new_callable=AsyncMock) as mock_history:
            mock_history.return_value = MagicMock(messages=[])
            with patch("api.routers.chat.load_chat_tuning_config", return_value={}):
                with patch("api.routers.chat.quality_scorer.calculate_quality_score", return_value=None):
                    await _prepare_chat_context(request)

    # Verify overrides were applied during query
    assert captured_routing == True, "enable_query_routing should be True during query"
    assert captured_rrf == True, "enable_rrf should be True during query"

    # Verify settings were restored after query
    assert settings.enable_query_routing == False, "enable_query_routing should be restored to False"
    assert settings.enable_rrf == False, "enable_rrf should be restored to False"

    # Restore original values
    settings.enable_query_routing = original_routing
    settings.enable_rrf = original_rrf


@pytest.mark.asyncio
async def test_eval_override_restored_on_exception():
    """Test settings restored even if query raises exception."""
    original_structured_kg = settings.enable_structured_kg
    settings.enable_structured_kg = False

    request = ChatRequest(
        message="test query",
        eval_enable_structured_kg=True
    )

    captured_value = None

    def mock_raises(**kwargs):
        nonlocal captured_value
        captured_value = settings.enable_structured_kg
        raise ValueError("Test error")

    with patch("api.routers.chat.graph_rag.query", side_effect=mock_raises):
        with patch("api.routers.chat.chat_history_service.get_conversation", new_callable=AsyncMock) as mock_history:
            mock_history.return_value = MagicMock(messages=[])
            with patch("api.routers.chat.load_chat_tuning_config", return_value={}):
                with pytest.raises(ValueError, match="Test error"):
                    await _prepare_chat_context(request)

    # Verify override was applied
    assert captured_value == True, "enable_structured_kg should be True during query"

    # Verify setting was restored despite exception
    assert settings.enable_structured_kg == False, "enable_structured_kg should be restored even after exception"

    # Restore original
    settings.enable_structured_kg = original_structured_kg


@pytest.mark.asyncio
async def test_multiple_overrides_work_together():
    """Test multiple overrides can be applied simultaneously."""
    # Save originals
    original_routing = settings.enable_query_routing
    original_flashrank = settings.flashrank_enabled
    original_cache = settings.enable_routing_cache

    # Set initial values
    settings.enable_query_routing = True
    settings.flashrank_enabled = True
    settings.enable_routing_cache = True

    request = ChatRequest(
        message="test query",
        eval_enable_query_routing=False,
        eval_flashrank_enabled=False,
        eval_enable_routing_cache=False
    )

    captured = {}

    def mock_query(**kwargs):
        captured['routing'] = settings.enable_query_routing
        captured['flashrank'] = settings.flashrank_enabled
        captured['cache'] = settings.enable_routing_cache
        return {"response": "ok", "sources": [], "metadata": {}, "stages": []}

    with patch("api.routers.chat.graph_rag.query", side_effect=mock_query):
        with patch("api.routers.chat.chat_history_service.get_conversation", new_callable=AsyncMock) as mock_history:
            mock_history.return_value = MagicMock(messages=[])
            with patch("api.routers.chat.load_chat_tuning_config", return_value={}):
                with patch("api.routers.chat.quality_scorer.calculate_quality_score", return_value=None):
                    await _prepare_chat_context(request)

    # All overrides should have been applied
    assert captured['routing'] == False
    assert captured['flashrank'] == False
    assert captured['cache'] == False

    # All should be restored
    assert settings.enable_query_routing == True
    assert settings.flashrank_enabled == True
    assert settings.enable_routing_cache == True

    # Restore originals
    settings.enable_query_routing = original_routing
    settings.flashrank_enabled = original_flashrank
    settings.enable_routing_cache = original_cache


@pytest.mark.asyncio
async def test_none_values_dont_override():
    """Test that None values don't override (use defaults)."""
    original = settings.enable_rrf
    settings.enable_rrf = True

    request = ChatRequest(
        message="test query",
        eval_enable_rrf=None  # None means don't override
    )

    captured = None

    def mock_query(**kwargs):
        nonlocal captured
        captured = settings.enable_rrf
        return {"response": "ok", "sources": [], "metadata": {}, "stages": []}

    with patch("api.routers.chat.graph_rag.query", side_effect=mock_query):
        with patch("api.routers.chat.chat_history_service.get_conversation", new_callable=AsyncMock) as mock_history:
            mock_history.return_value = MagicMock(messages=[])
            with patch("api.routers.chat.load_chat_tuning_config", return_value={}):
                with patch("api.routers.chat.quality_scorer.calculate_quality_score", return_value=None):
                    await _prepare_chat_context(request)

    # Should remain at original value
    assert captured == True, "Setting should not be overridden when eval field is None"
    assert settings.enable_rrf == True

    settings.enable_rrf = original
