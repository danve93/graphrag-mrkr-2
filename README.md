# Amber

<!-- markdownlint-disable -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Amber is a document intelligence platform implementing graph-enhanced Retrieval-Augmented Generation (RAG). It combines vector search, graph expansion, entity reasoning, and LLMs to provide contextual, sourced, and high-quality answers over ingested document collections.

This repository contains:
- Backend API and ingestion pipeline (FastAPI + Python)
- Frontend UI (Next.js + TypeScript)
- Graph storage and analysis (Neo4j)
- Utilities and scripts for ingestion, clustering, and maintenance

For a short technical overview and architecture notes, see [`AGENTS.md`](./AGENTS.md).

## Table of Contents
- [Overview](#overview)
- [Pipeline Diagram](#pipeline-diagram)
- [Quick Start (Docker Compose)](#quick-start-docker-compose)
- [Local Development](#local-development)
  - [Backend](#backend)
  - [Frontend](#frontend)
- [Important CLI Scripts](#important-cli-scripts)
- [Selected API Endpoints](#selected-api-endpoints)
- [Ingestion & Processing](#ingestion--processing)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Testing and Code Quality](#testing-and-code-quality)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

## Overview

Key components:
- **Frontend:** `frontend/` (Next.js) — chat UI, history, upload, database explorer
- **Backend:** `api/` (FastAPI) — chat router, document endpoints, job management, chat tuning
- **Ingestion:** `ingestion/` — loaders, converters, and document processor handling chunking, OCR, embeddings, and entity extraction
- **Graph storage:** Neo4j — stores Documents, Chunks, Entities, and relationships for multi-hop retrieval
- **RAG pipeline:** `rag/graph_rag.py` — LangGraph pipeline (query analysis → retrieval → graph reasoning → generation)

## Pipeline Diagram

Mermaid diagram (for supported renderers):

```mermaid
flowchart LR
  subgraph FE [Frontend]
    UI[Next.js Chat UI]
  end

  subgraph BE [Backend]
    API[FastAPI API]
    RAG[LangGraph RAG Pipeline]
    RET[Retriever (vector search)]
    EXP[Graph Expansion / Multi-hop]
    RR[Reranker (FlashRank, optional)]
    GEN[LLM Generation]
    QS[Quality Scoring]
    FH[Follow-up Generation]
  end

  subgraph GRAPH [Graph]
    NEO[Neo4j (Docs / Chunks / Entities)]
  end

  subgraph LLM [LLM Provider]
    LLM[OpenAI / Ollama / Local Model]
    EMB[Embedding Service / Model]
  end

  UI -->|POST chat query| API
  API --> RAG
  RAG -->|embedding lookup| RET
  RET --> NEO
  RET --> EXP
  EXP --> NEO
  EXP -->|candidates| RR
  RR --> GEN
  GEN -->|tokens| API
  GEN --> QS
  QS --> API
  GEN --> FH
  API -->|SSE stream| UI
  NEO -->|graph data| RAG
  LLM --> GEN
  EMB --> RET
```

Fallback ASCII diagram:

```
Frontend (Next.js UI)
        |
        v
    FastAPI API
        |
        v
   LangGraph RAG Pipeline
        |
  -------------------------------
  |       Retrieval & Ranking    |
  |  - Vector Retriever (embeddings)
  |  - Graph Expansion (Neo4j multi-hop)
  |  - Optional Reranker (FlashRank)
  -------------------------------
        |
        v
    LLM Generation (OpenAI/Ollama)
        |
  -------------------------------
  |  Post-processing & UX events |
  |  - Quality Scoring           |
  |  - Follow-up Suggestion      |
  |  - SSE token streaming to UI |
  -------------------------------

Graph storage: Neo4j stores Documents, Chunks, Entities, and relationships used by the retriever and graph expansion.
```

## Quick Start (Docker Compose)

Run the full stack (backend, frontend, Neo4j) locally:

```bash
docker compose up -d
```

Rebuild images after changing Dockerfiles:

```bash
docker compose up -d --build
```

Stream logs:

```bash
docker compose logs -f
```

Open the frontend at `http://localhost:3000` and the backend docs at `http://localhost:8000/docs`.

## Local Development

### Backend

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment template and set secrets:
   ```bash
   cp .env.example .env
   # Set OPENAI_API_KEY, NEO4J_* credentials, etc.
   ```
4. Start the backend (with reload):
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

## Important CLI Scripts

- `scripts/ingest_documents.py` — ingest a file or directory.
  ```bash
  # Single file
  python scripts/ingest_documents.py --file /path/to/document.pdf

  # Directory (recursive)
  python scripts/ingest_documents.py --input-dir /path/to/docs --recursive

  # Show supported extensions
  python scripts/ingest_documents.py --show-supported
  ```
- `scripts/run_clustering.py` — run graph clustering and projections (see script for arguments).

## Selected API Endpoints

Common endpoints (see `/docs` for full schemas):
- `GET /api/health` — health, version, and feature flags
- `POST /api/chat/query` — chat query (structured response, supports `stream` flag)
- `POST /api/chat/stream` — SSE streaming tokens
- `POST /api/chat/follow-ups` — generate follow-up questions
- `GET /api/chat-tuning/config` — download chat tuning configuration (parameters)
- `GET /api/chat-tuning/config/values` — read current live tuning values
- `GET /api/documents` — list documents
- `GET /api/documents/{document_id}` — document metadata and analytics
- `POST /api/documents/{document_id}/generate-summary` — (re)generate summary
- `POST /api/documents/{document_id}/hashtags` — update document hashtags
- `GET /api/documents/{document_id}/similarities` — chunk-to-chunk similarities for a document
- `GET /api/documents/{document_id}/preview` — preview file or chunk content
- `GET /api/history/sessions` — list chat sessions
- `GET /api/history/{session_id}` — fetch conversation messages
- Job management under `/api/jobs` — list, trigger, and monitor background jobs

Notes:
- Chat requests accept `llm_model` and `embedding_model` fields so the UI can select models at runtime.
- Chat responses stream SSE events of types `stage`, `token`, `sources`, `quality_score`, `follow_ups`, and `metadata`.

## Ingestion & Processing

- Document ingestion is handled by `ingestion/document_processor.py`:
  - Converts files to text/markdown via `ingestion/converters`
  - Chunks text with `core/chunking.document_chunker`
  - Generates embeddings via `core/embeddings.embedding_manager`
  - Stores Document, Chunk, and Entity nodes in Neo4j via `core/graph_db`
- Entity extraction is optional (controlled by settings) and can run synchronously for deterministic tests or asynchronously in background threads.
- After ingestion, chunk similarity edges are created and optional clustering/summaries can run via scripts in `scripts/`.

## Advanced Features

- **Leiden / Graph Clustering** — utilities to build projections and run clustering (see `scripts/run_clustering.py` and `scripts/build_leiden_projection.py`); parameters in `config/settings.py`.
- **Reranking (FlashRank)** — optional post-retrieval reranker controlled by `flashrank_enabled`; settings in `config/settings.py`; pre-warmed at startup when enabled.
- **Classification** — document labeling via `api/routers/classification.py` and `config/classification_config.json`.
- **Entity Extraction & Graph** — entity extraction via `core/entity_extraction.py` with synchronous or async embedding persistence (`SYNC_ENTITY_EMBEDDINGS` flag).
- **Chat Tuning and Model Selection** — runtime retrieval/generation controls in `api/routers/chat_tuning.py`; tuning parameters and defaults available through `/api/chat-tuning/config*` endpoints.

## Configuration

Key environment variables (see `config/settings.py` for full list):
- **LLM & embeddings** — `OPENAI_API_KEY`, `OPENAI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `EMBEDDING_MODEL`
- **Neo4j** — `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- **Feature toggles** — `ENABLE_ENTITY_EXTRACTION`, `ENABLE_QUALITY_SCORING`, `FLASHRANK_ENABLED`
- **Misc** — `SYNC_ENTITY_EMBEDDINGS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

## Testing and Code Quality

Backend tests:

```bash
source .venv/bin/activate
pytest api/tests/
```

Frontend tests:

```bash
cd frontend
npm run test
```

Linting and formatting examples:

```bash
source .venv/bin/activate
ruff check .
black .
isort .
pytest api/tests/
```

## Deployment

Docker Compose is the recommended local/demo deployment mechanism. For production, deploy services behind suitable ingress or load-balancer and ensure Neo4j has required memory and the GDS plugin when clustering is enabled.

## Contributing

1. Fork the repository and create a feature branch.
2. Run tests and linters locally.
3. Open a PR with description, test results, and screenshots when applicable.

## License

This project is licensed under the MIT License. See [`LICENSE.md`](./LICENSE.md) for details.
