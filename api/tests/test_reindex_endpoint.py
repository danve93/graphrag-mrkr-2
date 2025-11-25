import json
from fastapi.testclient import TestClient
from api.main import app

import core.graph_db as graph_db_mod
import ingestion.document_processor as dp_mod
import core.graph_clustering as clustering_mod

client = TestClient(app)


def test_reindex_endpoint_with_mocks(monkeypatch):
    # Mock driver.session().run to return fake documents
    class FakeRecord(dict):
        def __getitem__(self, key):
            return self.get(key)

    class FakeResult(list):
        def __init__(self, items):
            super().__init__(items)

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def run(self, query):
            # Return a sequence that yields records with doc_id and filename
            return FakeResult([FakeRecord({"doc_id": "doc-1", "filename": "file1.md"})])

    class FakeDriver:
        def session(self):
            return FakeSession()

    # Patch the graph_db instance driver to avoid real Neo4j
    monkeypatch.setattr(graph_db_mod.graph_db, "driver", FakeDriver())

    # Patch reset_document_entities to return predictable results
    def fake_reset_document_entities(doc_id: str):
        return {"chunk_ids": 1, "entity_relationships": 0, "chunk_entity_relationships": 0, "entities_removed": 2}

    monkeypatch.setattr(graph_db_mod.graph_db, "reset_document_entities", fake_reset_document_entities)

    # Patch create_all_entity_similarities to no-op
    monkeypatch.setattr(graph_db_mod.graph_db, "create_all_entity_similarities", lambda *a, **k: {})

    # Patch DocumentProcessor.extract_entities_for_all_documents to return a quick result
    def fake_extract_all(self):
        return {"status": "completed", "processed_documents": 1}

    monkeypatch.setattr(dp_mod.DocumentProcessor, "extract_entities_for_all_documents", fake_extract_all)

    # Patch clustering run to avoid running real clustering
    monkeypatch.setattr(clustering_mod, "run_leiden_clustering", lambda *a, **k: {"clusters": 1})

    # Call the HTTP endpoint to validate background job behavior
    resp = client.post("/api/classification/reindex", json={"confirmed": True})
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert "job_id" in data and "status_url" in data
    job_id = data["job_id"]

    # Poll the job status until it completes (test environment runs background tasks quickly)
    import time
    for _ in range(20):
        status_resp = client.get(f"/api/classification/reindex/{job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        if status_data.get("status") in ("success", "partial", "failed"):
            break
        time.sleep(0.05)

    assert status_data.get("status") == "success"
    assert status_data.get("result") is not None
