import asyncio
from unittest.mock import AsyncMock

from rag.nodes.structured_kg_executor import StructuredKGExecutor


def test_execute_query_rejects_unsuitable_type():
    executor = StructuredKGExecutor()

    result = asyncio.run(executor.execute_query("Tell me a joke"))

    assert result["success"] is False
    assert result.get("fallback_recommended") is True
    assert result["query_type"] == "general"


def test_execute_query_uses_mocked_cypher_path(monkeypatch):
    executor = StructuredKGExecutor()

    # Force a structured query type and stub heavy steps
    monkeypatch.setattr(executor, "_detect_query_type", lambda q: "aggregation")
    monkeypatch.setattr(executor, "_link_entities", AsyncMock(return_value=[{"name": "Neo4j", "confidence": 0.9}]))
    monkeypatch.setattr(
        executor,
        "_generate_cypher",
        AsyncMock(return_value={"success": True, "cypher": "MATCH (n) RETURN n LIMIT 1"}),
    )
    monkeypatch.setattr(
        executor,
        "_execute_with_correction",
        AsyncMock(return_value={"success": True, "results": [{"id": 1}], "corrections": 1, "final_cypher": "RETURN 1"}),
    )

    result = asyncio.run(executor.execute_query("How many documents mention Neo4j?"))

    assert result["success"] is True
    assert result["results"] == [{"id": 1}]
    assert result["corrections"] == 1
    # Ensure cypher from mocked generator is propagated
    assert "RETURN 1" in result["cypher"]
