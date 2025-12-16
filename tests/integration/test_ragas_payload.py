"""Test Ragas payload builder maps flags correctly."""
import sys
from pathlib import Path

# Add evals/ragas to path so we can import ragas_runner
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evals" / "ragas"))

from ragas_runner import build_chat_payload


def test_payload_maps_all_feature_flags():
    """Verify all variant flags map to eval_* fields."""
    record = {"user_input": "test query"}
    defaults = {}
    variant_cfg = {
        "payload_flags": {
            "enable_query_routing": True,
            "enable_structured_kg": False,
            "enable_rrf": True,
            "enable_routing_cache": False,
            "flashrank_enabled": False,
            "enable_graph_clustering": True,
        }
    }

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Verify all flags are mapped correctly
    assert payload["eval_enable_query_routing"] == True
    assert payload["eval_enable_structured_kg"] == False
    assert payload["eval_enable_rrf"] == True
    assert payload["eval_enable_routing_cache"] == False
    assert payload["eval_flashrank_enabled"] == False
    assert payload["eval_enable_graph_clustering"] == True


def test_payload_vector_only_variant():
    """Test vector_only disables all flags."""
    record = {"user_input": "test question"}
    defaults = {}
    variant_cfg = {
        "retrieval_mode": "simple",
        "payload_flags": {
            "enable_query_routing": False,
            "enable_structured_kg": False,
            "enable_rrf": False,
            "flashrank_enabled": False,
            "enable_graph_clustering": False,
        }
    }

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Verify retrieval mode
    assert payload["retrieval_mode"] == "simple"

    # Verify all eval flags are False
    assert payload["eval_enable_query_routing"] == False
    assert payload["eval_enable_structured_kg"] == False
    assert payload["eval_enable_rrf"] == False
    assert payload["eval_flashrank_enabled"] == False
    assert payload["eval_enable_graph_clustering"] == False


def test_payload_graph_hybrid_variant():
    """Test graph_hybrid enables all flags."""
    record = {"user_input": "another test"}
    defaults = {}
    variant_cfg = {
        "retrieval_mode": "hybrid",
        "payload_flags": {
            "enable_query_routing": True,
            "enable_structured_kg": True,
            "enable_rrf": True,
            "enable_routing_cache": True,
            "flashrank_enabled": True,
            "enable_graph_clustering": True,
        }
    }

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Verify retrieval mode
    assert payload["retrieval_mode"] == "hybrid"

    # Verify all eval flags are True
    assert payload["eval_enable_query_routing"] == True
    assert payload["eval_enable_structured_kg"] == True
    assert payload["eval_enable_rrf"] == True
    assert payload["eval_enable_routing_cache"] == True
    assert payload["eval_flashrank_enabled"] == True
    assert payload["eval_enable_graph_clustering"] == True


def test_payload_mixed_configuration():
    """Test mixed configurations work correctly."""
    record = {"user_input": "mixed test"}
    defaults = {
        "top_k": 10,
        "temperature": 0.5,
    }
    variant_cfg = {
        "retrieval_mode": "hybrid",
        "payload_flags": {
            "enable_query_routing": True,
            "enable_structured_kg": False,
            "enable_rrf": True,
            "flashrank_enabled": False,
        }
    }

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Check defaults are included
    assert payload["top_k"] == 10
    assert payload["temperature"] == 0.5

    # Check mixed flags
    assert payload["eval_enable_query_routing"] == True
    assert payload["eval_enable_structured_kg"] == False
    assert payload["eval_enable_rrf"] == True
    assert payload["eval_flashrank_enabled"] == False

    # Flags not in variant should not be in payload
    assert "eval_enable_routing_cache" not in payload
    assert "eval_enable_graph_clustering" not in payload


def test_payload_preserves_basic_fields():
    """Test that basic fields (message, session_id, stream) are always set."""
    record = {"user_input": "basic test"}
    defaults = {}
    variant_cfg = {"payload_flags": {}}

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Basic fields should always be present
    assert "message" in payload
    assert payload["message"] == "basic test"
    assert "session_id" in payload
    assert payload["stream"] == False  # ragas always uses non-streaming


def test_payload_handles_missing_flags():
    """Test payload works when variant_flags is empty or missing."""
    record = {"user_input": "test"}
    defaults = {}
    variant_cfg = {}  # No payload_flags

    payload = build_chat_payload(record, defaults, variant_cfg)

    # Should not have any eval_* fields
    eval_fields = [k for k in payload.keys() if k.startswith("eval_")]
    assert len(eval_fields) == 0, "Should have no eval fields when payload_flags missing"


def test_payload_question_field_fallback():
    """Test that 'question' field can be used instead of 'user_input'."""
    record = {"question": "fallback test"}  # Using 'question' instead of 'user_input'
    defaults = {}
    variant_cfg = {"payload_flags": {}}

    payload = build_chat_payload(record, defaults, variant_cfg)

    assert payload["message"] == "fallback test"
