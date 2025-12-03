import time
import json

from core.singletons import (
    get_response_cache,
    clear_response_cache,
    hash_response_params,
    ResponseKeyLock,
)


def test_response_cache_basic_put_get():
    clear_response_cache()
    cache = get_response_cache()
    key = hash_response_params(query="q1")
    payload = {"response": "hello", "metadata": {}}
    cache[key] = payload
    assert cache.get(key) == payload


def test_response_cache_clear_reinitializes():
    clear_response_cache()
    cache1 = get_response_cache()
    key = hash_response_params(query="q2")
    cache1[key] = {"response": "x"}
    assert cache1.get(key) is not None
    clear_response_cache()
    cache2 = get_response_cache()
    # After clear, old key should not exist
    assert cache2.get(key) is None


def test_hash_response_params_variations_change_key():
    k1 = hash_response_params(query="same", top_k=5)
    k2 = hash_response_params(query="same", top_k=10)
    assert k1 != k2
    # context documents affect key
    k3 = hash_response_params(query="same", top_k=5, context_documents=["docA"]) 
    assert k1 != k3
    # chat history hash affects key: simulate via different json
    chat_hash_a = ""  # empty
    chat_hash_b = "abcd"
    k4 = hash_response_params(query="same", chat_history_hash=chat_hash_a)
    k5 = hash_response_params(query="same", chat_history_hash=chat_hash_b)
    assert k4 != k5


def test_response_key_lock_singleflight():
    clear_response_cache()
    cache = get_response_cache()
    key = hash_response_params(query="q3")

    # Acquire lock in context and ensure others would block/timeout if concurrent.
    # Here we simply verify basic acquire/release behavior.
    with ResponseKeyLock(key, timeout=0.1) as acquired:
        assert acquired is True
        # While held, re-acquiring with 0 timeout should fail
        with ResponseKeyLock(key, timeout=0.0) as acquired2:
            assert acquired2 is False
    # After release, we can acquire again
    with ResponseKeyLock(key, timeout=0.1) as acquired3:
        assert acquired3 is True
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
