import json
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_get_classification_config():
    r = client.get("/api/classification/config")
    assert r.status_code == 200
    data = r.json()
    assert "entity_types" in data
    assert isinstance(data["entity_types"], list)


def test_update_classification_config_and_restore():
    # Get original config
    r = client.get("/api/classification/config")
    assert r.status_code == 200
    orig = r.json()

    # Prepare modified config (append a temporary entity type)
    modified = json.loads(json.dumps(orig))
    temp = "TEST_ENTITY_TEMP"
    if temp in modified["entity_types"]:
        temp = temp + "_2"
    modified["entity_types"].append(temp)

    # Save modified
    r2 = client.post("/api/classification/config", json=modified)
    assert r2.status_code == 200

    # Verify persisted
    r3 = client.get("/api/classification/config")
    assert r3.status_code == 200
    assert temp in r3.json().get("entity_types", [])

    # Restore original
    r4 = client.post("/api/classification/config", json=orig)
    assert r4.status_code == 200


def test_get_chat_tuning_config():
    r = client.get("/api/chat-tuning/config")
    assert r.status_code == 200
    data = r.json()
    assert "parameters" in data and isinstance(data["parameters"], list)


def test_update_chat_tuning_config_and_restore():
    # Get original
    r = client.get("/api/chat-tuning/config")
    assert r.status_code == 200
    orig = r.json()

    # Prepare modified copy
    modified = json.loads(json.dumps(orig))

    # Find a numeric slider parameter to modify
    modified_key = None
    for p in modified.get("parameters", []):
        if p.get("type") == "slider":
            # Set to the midpoint between min and max when available
            _min = p.get("min", 0)
            _max = p.get("max", 1)
            p["value"] = (_min + _max) / 2
            modified_key = p.get("key")
            break

    if not modified_key:
        pytest.skip("No slider parameter found in chat tuning configuration to modify")

    # Save modified
    r2 = client.post("/api/chat-tuning/config", json=modified)
    assert r2.status_code == 200

    # Fetch compact values and verify
    r3 = client.get("/api/chat-tuning/config/values")
    assert r3.status_code == 200
    values = r3.json()
    assert modified_key in values

    # Restore original
    r4 = client.post("/api/chat-tuning/config", json=orig)
    assert r4.status_code == 200
