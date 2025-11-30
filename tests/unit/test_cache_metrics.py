import pytest

from core.cache_metrics import cache_metrics


def test_cache_metrics_hit_miss_invalidation_reset():
    # Ensure metrics start at zero
    cache_metrics.reset()

    assert cache_metrics.response_hits == 0
    assert cache_metrics.response_misses == 0
    assert cache_metrics.response_invalidations == 0

    # Record some hits/misses/invalidation
    cache_metrics.record_response_hit()
    cache_metrics.record_response_hit()
    cache_metrics.record_response_miss()
    cache_metrics.record_response_invalidation()

    report = cache_metrics.get_report()

    assert report["response"]["hits"] == 2
    assert report["response"]["misses"] == 1
    assert report["response"]["invalidations"] == 1

    # Reset and confirm cleared
    cache_metrics.reset()
    report2 = cache_metrics.get_report()
    assert report2["response"]["hits"] == 0
    assert report2["response"]["misses"] == 0
    assert report2["response"]["invalidations"] == 0
