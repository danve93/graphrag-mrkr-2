````markdown
# Amber

<!-- markdownlint-disable -->

Amber is a document intelligence system powered by graph-enhanced Retrieval-Augmented Generation (RAG). Built with Next.js, FastAPI, and Neo4j.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

# Amber

Amber is a document intelligence platform implementing graph-enhanced Retrieval-Augmented Generation (RAG). It combines vector search, graph expansion, entity reasoning, and LLMs to provide contextual, sourced, and high-quality answers over ingested document collections.

This repository contains:
- Backend API and ingestion pipeline (FastAPI + Python)
- Frontend UI (Next.js + TypeScript)
- Graph storage and analysis (Neo4j)
- Utilities and scripts for ingestion, clustering, and maintenance

For a short technical overview and architecture notes, see `AGENTS.md`.

## Table of Contents
- Overview
- Quick Start (Docker Compose preferred)
- Local Development (backend + frontend)
- Important Scripts and Commands
- API (selected endpoints)
- Ingestion and Processing
- Configuration (important environment variables)
- Testing and Quality
- Deployment
- Contributing
- License

## Overview

Key components:
- Frontend: `frontend/` (Next.js app) — chat UI, history, upload, database explorer
- Backend: `api/` (FastAPI) — chat router, document endpoints, job management, chat-tuning
- Ingestion: `ingestion/` — loaders, converters, and document processor which handles chunking, OCR, embeddings, and entity extraction
- Graph storage: Neo4j — stores Documents, Chunks, Entities, and relationships used for multi-hop retrieval
- RAG pipeline: `rag/graph_rag.py` — LangGraph-based pipeline (query analysis → retrieval → graph reasoning → generation)

### Pipeline Diagram

Below is a visual representation of the core pipeline. A Mermaid diagram is provided for renderers that support it, followed by an ASCII fallback.

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


## Quick Start (recommended: Docker Compose)

The fastest way to run the full stack (backend, frontend, Neo4j) locally is Docker Compose.

Start the full stack:

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

## Local Development (backend)

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
# Edit .env and set OPENAI_API_KEY, NEO4J_* credentials, etc.
```

4. Start the backend (development reload enabled):

```bash
python api/main.py
```

The API will be available at `http://localhost:8000` and interactive docs at `/docs`.

## Local Frontend (development)

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

- `scripts/ingest_documents.py` — CLI to ingest a single file or directory into the pipeline. Examples:

  ```bash
  # Ingest a single file
  python scripts/ingest_documents.py --file /path/to/document.pdf

  # Ingest a directory recursively
  python scripts/ingest_documents.py --input-dir /path/to/docs --recursive

  # Show supported extensions
  python scripts/ingest_documents.py --show-supported
  ```

- `scripts/run_clustering.py` — run graph clustering and projections (see script for arguments)

## Selected API Endpoints

The backend exposes REST endpoints under `/api/*`. Below are commonly used endpoints; see `/docs` for full interactive schemas.

- `GET /api/health` — health, version and feature flags
- `POST /api/chat/query` — run a chat query (returns structured response, supports `stream` flag)
- `POST /api/chat/stream` — always returns SSE streaming tokens
- `POST /api/chat/follow-ups` — generate follow-up questions for a pair (request+response)
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

Notes on chat tuning and models:
- Chat requests accept `llm_model` and `embedding_model` fields; these are wired through the RAG pipeline so the UI can select models at runtime.
- Chat responses stream SSE events of types `stage`, `token`, `sources`, `quality_score`, `follow_ups`, and `metadata`.

## Ingestion & Processing

- Document ingestion is handled by `ingestion/document_processor.py`. It:
  - Converts files to text/markdown using `ingestion/converters`
  - Chunks text with `core/chunking.document_chunker`
  - Generates embeddings via `core/embeddings.embedding_manager`
  - Stores Document, Chunk, and Entity nodes in Neo4j via `core/graph_db`.

- Entity extraction is optional (controlled by settings) and can run synchronously (for deterministic tests) or in background threads.

- After ingestion, chunk similarity edges are created and optional clustering/summaries can run via scripts in `scripts/`.

## Advanced features

This project includes several advanced components for improving retrieval quality, organizing the graph, and controlling runtime behavior.

- Leiden / Graph Clustering
  - The repository contains utilities and scripts to build clustering projections and run Leiden clustering (see `scripts/run_clustering.py` and `scripts/build_leiden_projection.py`).
  - Clustering uses Neo4j (with GDS) to project relationships (e.g., `SIMILAR_TO`, `RELATED_TO`) and compute communities that are used for summarization and navigation in the UI.
  - Configuration for clustering can be found in `config/settings.py` (parameters like `clustering_resolution`, `clustering_relationship_types`, and `enable_graph_clustering`).

- Reranking (FlashRank)
  - An optional post-retrieval reranker (FlashRank) is available and guarded by the `flashrank_enabled` setting.
  - Reranker settings live in `config/settings.py` (`flashrank_model_name`, `flashrank_max_candidates`, `flashrank_blend_weight`, `flashrank_batch_size`, `flashrank_max_length`).
  - The backend attempts to pre-warm the reranker at startup when enabled (see `api/main.py` lifespan logic which calls `prewarm_ranker()` where configured).
  - Reranking is applied after initial hybrid retrieval and can be blended with the original hybrid score according to `flashrank_blend_weight`.

- Classification
  - The project includes a classification router and configuration (see `api/routers/classification.py` and `config/classification_config.json`).
  - Classification is used for document labeling and can be integrated into ingestion or used as a separate service to tag documents or chunks.

- Entity Extraction & Entity Graph
  - Entities are extracted from chunks using LLMs via `core/entity_extraction.py`. Extraction is optional and controlled by `ENABLE_ENTITY_EXTRACTION` and `sync_entity_embeddings` flags.
  - The `DocumentProcessor` schedules or runs entity extraction, persists entity nodes, creates chunk-entity relationships, and computes entity similarities (`graph_db.create_entity_similarities`).
  - Entity extraction can run synchronously for deterministic tests (`SYNC_ENTITY_EMBEDDINGS=1`) or asynchronously in background threads to avoid blocking ingestion.
  - The graph stores entities and relationships to enable multi-hop reasoning during retrieval and to support entity-based exploration in the UI.

- Chat Tuning and Model Selection
  - Chat tuning allows changing runtime retrieval and generation parameters from the UI without restarting the server. The chat router reads live tuning values (`api/routers/chat_tuning.py`) and the chat logic uses them as defaults when request values match application defaults.
  - Tuning parameters include retrieval weights (`chunk_weight`, `entity_weight`, `path_weight`), multi-hop options (`max_hops`, `beam_size`), expansion depth (`graph_expansion_depth`), and model selection fields.
  - The chat request schema accepts `llm_model` and `embedding_model` fields which are passed through to the RAG pipeline (`graph_rag.query(...)`) so the UI can select the generation model and embedding model at runtime.
  - Live tuning values and the tuning parameter definitions are available via `/api/chat-tuning/config` and `/api/chat-tuning/config/values`.

These building blocks work together: hybrid retrieval returns candidate chunks, the reranker can refine ordering, graph expansion and entity relationships enable multi-hop context, Leiden clustering groups content for summaries and navigation, and chat-tuning lets operators adjust behavior and models at request time.

## Configuration (important env vars and settings)

Configuration is provided via environment variables and documented in `config/settings.py`. Key variables include:

- LLM & embeddings
  - `OPENAI_API_KEY` — OpenAI API key (if using OpenAI provider)
  - `OPENAI_MODEL` — default OpenAI model (server default in settings)
  - `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL` — for local Ollama usage
  - `EMBEDDING_MODEL` — default embedding model

- Neo4j
  - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

- Feature toggles
  - `ENABLE_ENTITY_EXTRACTION` — enable entity extraction (default true)
  - `ENABLE_QUALITY_SCORING` — enable quality scoring
  - `FLASHRANK_ENABLED` — optional reranker

- Misc
  - `SYNC_ENTITY_EMBEDDINGS` — force synchronous embedding & persistence for deterministic runs
  - `CHUNK_SIZE`, `CHUNK_OVERLAP` — chunker params

Refer to `config/settings.py` for full list and defaults.

## Testing and Code Quality

Backend tests live under `api/tests/`. Run them with:

```bash
source .venv/bin/activate
pytest api/tests/
```

Frontend tests live in `frontend/` and run with the standard npm scripts:

```bash
cd frontend
npm run test
```

Repository tooling:

- Linting: `ruff`
- Formatting: `black`, `isort`
- Type checks: `mypy` (optional)

Example lint/test commands:

```bash
source .venv/bin/activate
ruff check .
black .
isort .
pytest api/tests/
```

## Deployment

- Docker Compose is the recommended local/demo deployment mechanism.
- For production, package services individually and run behind suitable ingress/load-balancer; ensure Neo4j is provisioned with required memory and GDS plugin when using clustering.

## Contributing

Please open issues and PRs. Recommended workflow:

1. Fork repository and create a feature branch
2. Run tests and linters locally
3. Open a PR with description, test results, and screenshots where applicable

## License

This project is licensed under the MIT License. See `LICENSE.md` for details.
# Optional type check
mypy .
```

## Docker

The preferred way to start all services for local demos is via Docker Compose as shown in Quick Start above.

Use `docker compose up -d` to bring up the full stack and `docker compose logs -f` to stream logs.

## Contributing

Contributions are welcome. Please follow the standard fork / branch / PR workflow and include tests for new behavior.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

````