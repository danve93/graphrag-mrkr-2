# Backend Testing

Guide to testing the FastAPI backend.

## Prerequisites

- Python 3.10+
- Virtual environment activated
- Dependencies installed: `pip install -r requirements.txt`

## Running Tests

```bash
pytest tests/ -q
```

### Selecting Tests

```bash
# Run specific file
pytest tests/api/test_chat.py

# Run by keyword
pytest -k chat

# Stop on first failure
pytest -x
```

## Configuration for Tests

- Use `.env.test` or environment overrides
- Recommended test settings:
```bash
ENABLE_ENTITY_EXTRACTION=false
FLASHRANK_ENABLED=false
ENABLE_CLUSTERING=false
ENABLE_CACHING=false
LOG_LEVEL=DEBUG
SYNC_ENTITY_EMBEDDINGS=1
```

## Fixtures & Mocks

- Mock OpenAI with local stubs
- Use temporary Neo4j (Docker) or in-memory mocks
- Patch `core.embeddings.get_embedding` for deterministic output

## Patterns

- Avoid network calls in unit tests
- Test pure functions in `core/` directly
- Integration tests for routers under `api/routers/`

## Coverage

```bash
pytest --cov=api --cov=core --cov=config --cov=rag --cov-report=term-missing
```

## Example Test (Router)

```python
from fastapi.testclient import TestClient
from api.main import app

def test_health_endpoint():
    client = TestClient(app)
    r = client.get('/api/database/health')
    assert r.status_code in (200, 503)
```

## Tips

- Use `SYNC_ENTITY_EMBEDDINGS=1` for deterministic runs
- Seed random where needed
- Keep tests fast (< 2s per file)
