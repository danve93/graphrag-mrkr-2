# Local Runtime

Run Amber locally for development.

## Setup

```bash
# Python venv
python3 -m venv .venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Env vars
cp .env.example .env
# Fill OPENAI_API_KEY, Neo4j creds
```

## Start Services

```bash
# Backend (auto-reload)
python api/main.py

# Frontend
cd frontend
npm install
npm run dev
```

## Data Paths

- `data/documents/` - Uploaded files
- `data/chunks/` - Chunked text
- `data/extracted/` - Entity extraction artifacts
- `data/flashrank_cache/` - Reranker model cache (optional)

## Useful Endpoints

```bash
# Chat (SSE stream)
curl -N -H "Content-Type: application/json" \
  -d '{"message":"Hello","session_id":"local"}' \
  http://localhost:8000/api/chat

# Documents
curl http://localhost:8000/api/documents

# Processing status
curl http://localhost:8000/api/documents/processing-status
```

## Hot Reload & Debugging

- Backend auto-reloads on file change
- Use `LOG_LEVEL=DEBUG` for verbose logs
- Inspect entities: `python scripts/inspect_entities.py`

## Test & Lint

```bash
pytest tests/
cd frontend && npm test
```
