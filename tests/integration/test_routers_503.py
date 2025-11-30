from fastapi.testclient import TestClient
import pytest

from api.main import app

from neo4j.exceptions import ServiceUnavailable


client = TestClient(app)


def test_graph_clustered_returns_503_on_db_unavailable(monkeypatch):
    from core import graph_db as gb

    def _raise(*args, **kwargs):
        raise ServiceUnavailable("simulated down")

    monkeypatch.setattr(gb.graph_db, "get_clustered_graph", _raise)

    resp = client.get("/api/graph/clustered")
    assert resp.status_code == 503
    assert "Graph database unavailable" in resp.json().get("detail", "")


def test_documents_metadata_returns_503_on_db_unavailable(monkeypatch):
    from core import graph_db as gb

    def _raise(*args, **kwargs):
        raise ServiceUnavailable("simulated down")

    monkeypatch.setattr(gb.graph_db, "get_document_details", _raise)

    resp = client.get("/api/documents/some-doc-id")
    assert resp.status_code == 503
    assert "Graph database unavailable" in resp.json().get("detail", "")


def test_database_stats_returns_503_on_db_unavailable(monkeypatch):
    from core import graph_db as gb

    def _raise(*args, **kwargs):
        raise ServiceUnavailable("simulated down")

    monkeypatch.setattr(gb.graph_db, "get_database_stats", _raise)

    resp = client.get("/api/database/stats")
    assert resp.status_code == 503
    assert "Graph database unavailable" in resp.json().get("detail", "")
