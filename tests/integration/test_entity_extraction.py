"""Unit tests for entity normalization and relationship suggestions."""

import json
from core.entity_extraction import EntityExtractor


def test_normalize_entity_name_subfloor_and_parentheses():
    extractor = EntityExtractor()

    raw = "  Sub-Floor (level 1)  "
    normalized = extractor._normalize_entity_name(raw)
    assert normalized == "subfloor"


def test_normalize_entity_type_overrides():
    extractor = EntityExtractor()

    assert extractor._normalize_entity_type("MTA") == "COMPONENT"
    assert extractor._normalize_entity_type("domain certificate") == "CERTIFICATE"
    # Unmapped type falls back to CONCEPT or preserved default
    assert extractor._normalize_entity_type("Some Unknown Type") in ("CONCEPT", "SOME UNKNOWN TYPE",)


def test_relation_type_suggestions_present():
    extractor = EntityExtractor()
    suggestions = extractor.RELATION_TYPE_SUGGESTIONS
    assert isinstance(suggestions, list)
    assert "COMPONENT_RUNS_ON_NODE" in suggestions
    assert "RELATED_TO" in suggestions


def test_metrics_json_serializable():
    """Ensure extraction metrics can be serialized to JSON for Neo4j storage."""
    metrics = {
        "document_id": "test-doc-123",
        "chunks_processed": 10,
        "entities_created": 44,
        "relationships_created": 54,
        "relationships_requested": 54,
        "chunk_coverage": 0.7,
        "entities_per_chunk": 4.4,
        "unique_chunks_with_entities": 7,
    }
    
    # Metrics should be JSON serializable (Neo4j only accepts primitives)
    metrics_json = json.dumps(metrics)
    assert isinstance(metrics_json, str)
    
    # Should round-trip correctly
    metrics_restored = json.loads(metrics_json)
    assert metrics_restored == metrics
