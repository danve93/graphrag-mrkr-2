import threading
import time

import pytest

from core.singletons import get_response_cache
from rag import graph_rag as graph_rag_module
from rag.graph_rag import graph_rag


def test_session_scoping(monkeypatch):
    """Ensure responses are scoped by session_id: different sessions shouldn't hit same cache entry."""
    cache = get_response_cache()
    cache.clear()

    call_count = {"c": 0}

    def fake_generate_response(query, graph_context, query_analysis, temperature, chat_history, llm_model=None):
        call_count["c"] += 1
        return {"response": f"resp-{call_count['c']}", "sources": [], "metadata": {}}

    # Patch the generation function used by the pipeline
    monkeypatch.setattr("rag.nodes.generation.generate_response", fake_generate_response)

    r1 = graph_rag.query("hello session", chat_history=[], session_id="session-A")
    r2 = graph_rag.query("hello session", chat_history=[], session_id="session-B")

    # Each different session should cause a separate generation (no cross-session reuse)
    assert call_count["c"] == 2
    assert r1["response"] != r2["response"]


def test_singleflight_concurrent(monkeypatch):
    """Multiple concurrent queries for same key should trigger only one generation (singleflight)."""
    cache = get_response_cache()
    cache.clear()

    call_count = {"c": 0}

    def fake_generate_response(query, graph_context, query_analysis, temperature, chat_history, llm_model=None):
        # Simulate an expensive generation
        call_count["c"] += 1
        time.sleep(0.4)
        return {"response": "expensive-result", "sources": [], "metadata": {}}

    monkeypatch.setattr("rag.nodes.generation.generate_response", fake_generate_response)

    results = []

    def worker():
        res = graph_rag.query("singleflight test", chat_history=[], session_id=None)
        results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(6)]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Only one underlying generation should have happened
    assert call_count["c"] == 1
    assert len(results) == 6
    for r in results:
        assert r["response"] == "expensive-result"


if __name__ == "__main__":
    pytest.main([__file__])
