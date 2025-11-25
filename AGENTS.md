# Amber Contributor Guide

Use this file as the authoritative quick-reference for how to work in this repository. The guidance applies to the entire repo. When updating functionality, keep this document and the root README in sync.

## Documentation Expectations
- Keep the root `README.md` concise and actionable. Favor short, copyable bash snippets over prose.
- When you add or change pipeline behavior (ingestion, retrieval, clustering, reranking, chat tuning), update both `README.md` and this file with a brief summary and pointers to the relevant modules.
- Link to entry points instead of duplicating logic. Prefer paths like `api/routers/chat.py` or `ingestion/document_processor.py` over inline code blocks.

## Architecture Snapshot
- **RAG pipeline:** `rag/graph_rag.py` (LangGraph) orchestrates query analysis → retrieval (hybrid) → graph reasoning → generation. Each stage appends identifiers to `state["stages"]` so the API can stream progress via SSE.
- **Ingestion:** `ingestion/document_processor.py` coordinates loaders (`ingestion/loaders/`), conversion, chunking (`core/chunking.py`), embeddings (`core/embeddings.py`), entity extraction (`core/entity_extraction.py`), and graph persistence (`core/graph_db.py`).
- **Backend:** FastAPI routers live in `api/routers/`; Pydantic models are in `api/models.py`; services and utilities live in `api/services/` and `core/`.
- **Frontend:** `frontend/` (Next.js + TypeScript) renders chat, history, uploads, database explorer, and tuning controls using Zustand and React Context. SSE handling and model/type definitions live in `frontend/src/types/` and associated components.

## Coding Conventions
- Prefer explicit imports; avoid wildcard imports.
- Keep HTTP routers thin: validate, delegate to services, and format responses.
- Use parameterized Neo4j queries (no string concatenation) and ensure response payloads remain JSON-serializable for SSE.
- For async code, handle exceptions at the edge and return structured error data rather than throwing to clients.

## Testing & Quality
- Backend: `pytest api/tests/`, `ruff check .`, `black .`, and `isort .` before committing.
- Frontend: `npm run test` under `frontend/`.
- For deterministic entity embedding during tests, set `SYNC_ENTITY_EMBEDDINGS=1`.

## Operational Notes
- Docker Compose is the default way to bring up the stack locally; use `docker compose up -d --build` after Dockerfile changes.
- FlashRank prewarms at startup when enabled; failures should log but not block the app.
- Neo4j must have the GDS plugin and adequate memory for clustering utilities in `scripts/`.
