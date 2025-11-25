import asyncio
import os
import sys
import types
import pytest

# Ensure repository root is on sys.path so tests can import the package modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.nodes import retrieval as retrieval_module


async def _fake_retrieve_async(**kwargs):
    # Return a recognizable result so the sync wrapper returns it
    return [{"chunk_id": "test_chunk", "similarity": 0.9, "_kw_args": kwargs}]


def test_retrieve_documents_calls_async_with_keyword_args(monkeypatch):
    """Verify retrieve_documents forwards keyword args to retrieve_documents_async and returns result."""
    monkeypatch.setattr(retrieval_module, "retrieve_documents_async", _fake_retrieve_async)

    result = retrieval_module.retrieve_documents(
        query="test query",
        query_analysis={"foo": "bar"},
        retrieval_mode="hybrid",
        top_k=3,
        chunk_weight=0.6,
        entity_weight=0.4,
        path_weight=0.2,
        use_multi_hop=False,
        max_hops=2,
        beam_size=4,
        restrict_to_context=True,
        expansion_depth=2,
        context_documents=["doc1"],
        embedding_model="embed-v1",
    )

    assert isinstance(result, list)
    assert result[0]["chunk_id"] == "test_chunk"
    # Ensure the mocked coroutine received our keyword args
    assert isinstance(result[0]["_kw_args"], dict)
    assert result[0]["_kw_args"].get("query") == "test query"
    assert result[0]["_kw_args"].get("top_k") == 3
    assert result[0]["_kw_args"].get("embedding_model") == "embed-v1"


def test_retrieve_documents_inside_running_loop(monkeypatch):
    """When called inside a running event loop, the wrapper should still return the async result.

    This test creates a temporary event loop and invokes the synchronous wrapper from inside
    that loop using `run_in_executor` to simulate the real behavior.
    """
    monkeypatch.setattr(retrieval_module, "retrieve_documents_async", _fake_retrieve_async)

    async def _runner():
        loop = asyncio.get_running_loop()
        # Pass minimal required positional args (query, query_analysis)
        return await loop.run_in_executor(None, retrieval_module.retrieve_documents, "test query 2", {})

    result = asyncio.run(_runner())

    assert isinstance(result, list)
    assert result[0]["chunk_id"] == "test_chunk"

