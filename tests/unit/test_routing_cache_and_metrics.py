import asyncio
from rag.nodes.routing_cache import RoutingCache
from core.routing_metrics import RoutingMetrics


def test_routing_cache_hit_and_stats(monkeypatch):
    cache = RoutingCache(maxsize=10, ttl=60, similarity_threshold=0.8)
    # Ensure cache enabled
    monkeypatch.setattr("rag.nodes.routing_cache.settings.enable_routing_cache", True, raising=False)
    # Deterministic embedding to force similarity=1.0
    monkeypatch.setattr("rag.nodes.routing_cache.embedding_manager.get_embedding", lambda q: [1.0, 0.0])

    asyncio.run(cache.set("what is neo4j?", {"categories": ["install"]}))
    result = asyncio.run(cache.get("what is neo4j?"))

    assert result == {"categories": ["install"]}
    stats = cache.get_stats()
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 0
    assert stats["hit_rate"] == 1.0
    assert stats["size"] == 1


def test_routing_metrics_compute_rates():
    metrics = RoutingMetrics()

    metrics.record_routing(categories=["install", "configure", "extra"], confidence=0.8, latency_ms=120, used_cache=True)
    metrics.record_routing(categories=["install"], confidence=0.5, latency_ms=80, used_cache=False, fallback_used=True)
    metrics.record_failure_point("FP1", {"reason": "test"})
    metrics.record_user_feedback(correct=True)
    metrics.record_user_feedback(correct=False)

    stats = metrics.get_stats()
    assert stats["total_queries"] == 2
    assert stats["cache_hit_rate"] == 0.5
    assert stats["fallback_rate"] == 0.5
    assert stats["multi_category_rate"] == 0.5
    assert stats["failure_point_counts"]["FP1"] == 1
