# Amber

Amber is a document intelligence platform that pairs graph-enhanced retrieval with large language models to deliver contextual, sourced answers over your documents. It combines hybrid search, Neo4j-backed graph expansion, entity reasoning, reranking, and streaming generation behind a FastAPI backend and a Next.js frontend.

## Contents
- [Features](#features)
- [Architecture](#architecture)
- [Quickstart (Docker Compose)](#quickstart-docker-compose)
- [Local Development](#local-development)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Configuration](#configuration)
- [Ingestion](#ingestion)
- [Chat & API](#chat--api)
- [Scripts](#scripts)
- [Testing](#testing)
- [Deployment Notes](#deployment-notes)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Hybrid retrieval** that blends embedding similarity with graph expansion over Documents, Chunks, and Entities stored in Neo4j.
- **Multi-hop reasoning** in `rag/graph_rag.py` using LangGraph to orchestrate query analysis → retrieval → graph reasoning → generation.
- **Streaming chat** with SSE events for tokens, stage updates, sources, quality scores, and follow-up suggestions.
- **Document ingestion** covering PDF, DOCX, PPTX, XLSX/CSV, images (OCR), and markdown with chunking, embeddings, and optional entity extraction.
- **Reranking & quality scoring** via FlashRank and optional scoring hooks.
- **Next.js frontend** for chat, history, upload, database explorer, and chat tuning controls.

## Architecture
- **Backend (FastAPI)** in `api/` exposes chat, documents, history, tuning, and job routes. Pydantic models live in `api/models.py`; business logic is under `api/services/`.
- **Ingestion pipeline** in `ingestion/document_processor.py` orchestrates loaders (`ingestion/loaders/`), converters, chunking (`core/chunking.py`), embeddings (`core/embeddings.py`), entity extraction (`core/entity_extraction.py`), and persistence (`core/graph_db.py`).
- **Graph storage** uses Neo4j to house Documents, Chunks, Entities, similarity edges, and clustering results.
- **Frontend** in `frontend/` (Next.js + TypeScript) consumes SSE streams, renders chat/history, manages uploads, and surfaces tuning controls via Zustand + React Context.
- **Configuration** is centralized in `config/settings.py` with environment variable overrides.

## Quickstart (Docker Compose)
Bring up the full stack locally (backend, frontend, Neo4j):

```bash
docker compose up -d
```

Rebuild images after Dockerfile changes:

```bash
docker compose up -d --build
```

Tail logs:

```bash
docker compose logs -f
```

The frontend runs on `http://localhost:3000`; backend docs are at `http://localhost:8000/docs`.

## Local Development
### Backend
1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy the env template and set secrets (OpenAI, Neo4j, optional FlashRank cache path):
   ```bash
   cp .env.example .env
   ```
3. Start the API with reload:
   ```bash
   python api/main.py
   ```

### Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL to http://localhost:8000
npm run dev
```

Production build:

```bash
cd frontend
npm ci
npm run build
npm run start
```

## Configuration
Set values through environment variables (see `config/settings.py` for defaults and full list).

- **LLM & embeddings:** `OPENAI_API_KEY`, `OPENAI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `EMBEDDING_MODEL`.
- **Neo4j:** `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
- **Retrieval & expansion:** `HYBRID_CHUNK_WEIGHT`, `HYBRID_ENTITY_WEIGHT`, `MAX_EXPANDED_CHUNKS`, `MAX_EXPANSION_DEPTH`, `EXPANSION_SIMILARITY_THRESHOLD`.
- **Reranking:** `FLASHRANK_ENABLED`, `FLASHRANK_MODEL_NAME`, `FLASHRANK_MAX_CANDIDATES`, `FLASHRANK_BLEND_WEIGHT`, `FLASHRANK_BATCH_SIZE`.
- **Entity & OCR:** `ENABLE_ENTITY_EXTRACTION`, `SYNC_ENTITY_EMBEDDINGS`, `ENABLE_OCR`, `OCR_QUALITY_THRESHOLD`.
- **Misc:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_CONCURRENCY`, `LLM_CONCURRENCY`, `EMBEDDING_DELAY_MIN/MAX`, `LLM_DELAY_MIN/MAX`.

## Ingestion
Use `ingestion/document_processor.py` (or the scripts below) to load documents:
- Loaders in `ingestion/loaders/` handle PDFs, Word, PowerPoint, spreadsheets/CSV, text/markdown, and images (OCR when needed).
- Conversion normalizes content and metadata; chunking uses `core/chunking.document_chunker` with overlap settings.
- Embeddings run via `core/embeddings.embedding_manager`; entity extraction (optional) creates Entity nodes and relationships.
- Graph persistence uses `core/graph_db` to store Documents, Chunks, Entities, similarity edges, and clustering artifacts.

## Chat & API
Common endpoints (full schema at `/docs`):
- `GET /api/health` — health, version, and feature flags.
- `POST /api/chat/query` — chat query with optional `stream` flag.
- `POST /api/chat/stream` — SSE streaming tokens and stage events.
- `POST /api/chat/follow-ups` — follow-up question generation.
- `GET /api/chat-tuning/config` and `/api/chat-tuning/config/values` — tuning defaults and current values.
- `GET /api/documents` — list documents; detail and analytics under `/api/documents/{document_id}`.
- `/api/documents/{document_id}/preview`, `/similarities`, `/generate-summary`, `/hashtags` — previews, similarity graphs, summaries, and hashtags.
- `GET /api/history/sessions` and `/api/history/{session_id}` — chat history retrieval.
- `/api/jobs` — background job management.

Chat responses stream SSE events of types `stage`, `token`, `sources`, `quality_score`, `follow_ups`, and `metadata`. Requests accept `llm_model` and `embedding_model` fields so clients can choose models per request.

## Scripts
- `scripts/ingest_documents.py` — ingest files or directories.
  ```bash
  # Single file
  python scripts/ingest_documents.py --file /path/to/document.pdf

  # Directory (recursive)
  python scripts/ingest_documents.py --input-dir /path/to/docs --recursive

  # Show supported extensions
  python scripts/ingest_documents.py --show-supported
  ```
- `scripts/run_clustering.py` — build projections and run Leiden clustering (inspect script for parameters).
- `scripts/build_leiden_projection.py` — prepare Neo4j projections for clustering.

## Testing
Backend:
```bash
source .venv/bin/activate
pytest api/tests/
```

Frontend:
```bash
cd frontend
npm run test
```

Formatting/linting examples:
```bash
source .venv/bin/activate
ruff check .
black .
isort .
```

## Deployment Notes
- Docker Compose is the simplest local/demo deployment. For production, place services behind ingress/load balancers and provision Neo4j with adequate memory and the GDS plugin if clustering is enabled.
- FlashRank is pre-warmed on startup when `FLASHRANK_ENABLED` is true; failures are logged but do not block startup.
- Use `SYNC_ENTITY_EMBEDDINGS=1` for deterministic entity embedding during tests.

## Contributing
1. Fork and create a feature branch.
2. Run tests and linters before opening a PR.
3. Include descriptions, relevant logs, and screenshots for UX changes.

## License
[MIT](./LICENSE.md)
