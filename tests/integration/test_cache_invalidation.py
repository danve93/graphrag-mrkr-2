import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
def test_cache_invalidation_endpoint(monkeypatch):
    # Reset metrics
    from core.cache_metrics import cache_metrics

    cache_metrics.reset()

    client = TestClient(app)

    # Monkeypatch the graph_db.clear_database to avoid needing a live Neo4j instance
    import api.routers.database as db_mod

    try:
        monkeypatch.setattr(db_mod.graph_db, "clear_database", lambda *args, **kwargs: None)
    except Exception:
        # if graph_db not available for patching, just proceed (some test environments may differ)
        pass

    # Call the clear database endpoint which should clear the response cache and record invalidation
    resp = client.post("/api/database/clear")
    assert resp.status_code == 200

    # Verify invalidation metric incremented
    report = cache_metrics.get_report()
    assert report["response"]["invalidations"] >= 1
