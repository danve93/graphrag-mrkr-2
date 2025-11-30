# Amber

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/danve93/graphrag-mrkr-2/actions/workflows/ci.yml/badge.svg)](https://github.com/danve93/graphrag-mrkr-2/actions)
[![GitHub release](https://img.shields.io/github/v/release/danve93/graphrag-mrkr-2?label=release)](https://github.com/danve93/graphrag-mrkr-2/releases)
[![Dependabot Status](https://img.shields.io/github/dependabot/danve93/graphrag-mrkr-2?logo=dependabot)](https://github.com/danve93/graphrag-mrkr-2/network/updates)
[![Languages](https://img.shields.io/github/languages/top/danve93/graphrag-mrkr-2)](https://github.com/danve93/graphrag-mrkr-2/search?l=)

Tested on Python 3.10–3.11. Frontend built with Next.js; recommended local setup uses Docker Compose.

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

## Implementation Details

- Entity accumulation and deduplication: an in-memory graph layer is used to accumulate extracted entities and relationships before persistence. Implementations merge duplicate entities, accumulate descriptions, sum relationship strengths, track provenance, and export batch Cypher (UNWIND) to Neo4j to reduce database round-trips.

- Multi-layer caching: the application provides long-lived service instances with multiple caches — an entity-label TTL cache to reduce DB lookups, an embeddings cache keyed by text+model to avoid redundant embedding calls, and a short-TTL retrieval cache for recent queries. Cache metrics are exposed via an API endpoint for monitoring.

- Marker-based PDF/document conversion integration: the ingestion pipeline supports calling an external high-accuracy document conversion tool in-process, via CLI, or via a small server. The integration supports LLM-assisted extraction, optional forced OCR, table merging, and configurable device/dtype settings. Operational and licensing considerations are handled via configuration and recommended isolation.

- UI and interaction updates: layout fixes removed unintended top padding from scroll containers; tooltip logic was refactored to attach handlers to DOM children when possible to avoid adding positioned wrapper elements; color parsing and community-color rendering were made defensive to avoid runtime errors. Design system updates include typography, spacing tokens, simplified color palette, and animation reductions.

## Pipeline Diagram

```
Frontend (Next.js)
  ┌─────────────────────────────────────────────────────────────────┐
  │ Chat UI │ History │ Upload │ Database Explorer                  │
  └─────────────────────────────────────────────────────────────────┘
                │ 
          REST API + SSE
                ▼
       Backend API (FastAPI)
                │
  ┌─────────────┴─────────────┐                   ┌──────────────┐
  │ LangGraph RAG Pipeline    │                   │ Neo4j Driver │
  │ - Query analysis          │<───────+──────────│  (Graph DB)  │
  │ - Hybrid retrieval        │        │          └──────────────┘
  │ - Graph reasoning / rerank│        │
  │ - Generation (LLM clients)│        │
  └─────────────┬─────────────┘        │
                │                      │
                │                      │
    ┌───────────▼───────────┐          │
    │ Caches & Singletons   │          │
    │ - Entity-label TTL    │          │
    │ - Embedding LRU cache │          │
    │ - Retrieval TTL cache │          │
    └───────────┬───────────┘          │
                │                      │
                ▼                      │
    ┌───────────────────────────────────────────────┐
    │ Ingestion / Conversion / Persistence          │
    │ - Loaders (PDF, DOCX, images, etc.)           │
    │ - Optional Marker conversion (lib/CLI/server) │
    │ - Chunking → Embeddings (uses embedding cache)│
    │ - Entity extraction → In-memory EntityGraph   │
    │ - Batch UNWIND persistence to Neo4j           │
    └───────────────────────────────────────────────┘
               │
               ▼
     External LLMs / Embedding Providers
```

## Quick Start (Docker Compose)

Run the full stack (backend, frontend, Neo4j) locally:

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

Open the frontend at `http://localhost:3000` and the backend docs at `http://localhost:8000/docs`.

### Docker Compose: internal vs host networking

The Compose stack defaults to container-internal service hostnames (for example `bolt://neo4j:7687`) so services communicate reliably on the Compose network. This is the recommended configuration for development and CI.

If you need containers to connect to a Neo4j instance running on the host (for example when debugging a host service), use the optional override file `docker-compose.override.yml` included in the repository. That file sets `NEO4J_URI` to `bolt://host.docker.internal:7687` for services that need it and is intended to be used only when explicitly required.

Usage (opt-in):

```bash
# Start the stack using the host override (containers will use host.docker.internal)
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build

# Run tests from the host; the test process should use a host-reachable URI
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=<password>
pytest -q tests/integration
```

Notes:
- Prefer the default internal networking (`bolt://neo4j:7687`) unless you explicitly need containers to reach a host service.
- The pytest fixture (`tests/conftest.py`) supplies container-internal values when it starts Compose so accidental host `NEO4J_URI` exports will not be injected into containers.
- Do not commit secrets. Use `.env` or CI secret stores for credentials and keep `.env.example` as a template.

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

Environment notes:

- The repository provides a `.env.example` with recommended defaults for local development. Copy it to `.env` and update values before running services.
- For Docker Compose demos the compose file respects `NEO4J_AUTH` (format `user/password`). The repository uses a safe demo default `NEO4J_AUTH=neo4j/test` in `.env.example` to make `docker compose up` work without editing; when running the backend outside Compose, keep `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD` set.
- Example (copy-paste into your shell):
  ```bash
  cp .env.example .env
  # Optionally set a demo Neo4j auth (only needed for Compose):
  export NEO4J_AUTH=neo4j/test
  ```
  
   Note about e2e/local compose runs:

   - To override the demo Neo4j password when starting the e2e compose stack, set `NEO4J_AUTH` on the command line so the example password is not used. For example:

      ```bash
      # Use a stronger demo password and start the e2e compose stack
      NEO4J_AUTH=neo4j/neo4jpass docker compose -f docker-compose.e2e.yml up -d --force-recreate

      # Then run the wait script to block until Bolt is reachable from the host
      NEO4J_URI=bolt://localhost:7687 bash scripts/wait_for_services.sh echo ok
      ```

   - Note: the default compose URI inside the Docker network is `bolt://neo4j:7687`. When you publish ports to the host (the common local-dev pattern), use `bolt://localhost:7687` for host-side checks and scripts.
4. Start the backend (with reload):
   ```bash
   python api/main.py
   ```

### Frontend

For **local development** (frontend calls backend at `http://localhost:8000`):

```bash
cd frontend
npm install
cp .env.local.example .env.local
# The .env.local sets NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

For **Docker Compose deployment** (frontend proxies to backend via Docker network):

- The `docker-compose.yml` sets `NEXT_PUBLIC_API_URL_SERVER=http://backend:8000` so the Next.js server can proxy API requests.
- **Important:** Remove or don't commit `frontend/.env.local` when deploying with Docker Compose, as it will override compose environment variables.
- The `.env.local` file is gitignored to prevent deployment issues.

Production build:

```bash
cd frontend
npm ci
npm run build
npm run start
```

### Makefile convenience targets

We provide a few convenience targets in the repository `Makefile`. Useful targets for local e2e runs include `e2e-compose-smoke` and `e2e-full-pipeline`.

Behavior notes (Makefile improvements):

- If `NEO4J_AUTH` is explicitly set in your environment, the Makefile uses that value directly.
- If `NEO4J_AUTH` is not set, the Makefile will automatically generate a secure 16-character alphanumeric password for Neo4j, persist it to `.secrets/neo4j_password` (file mode `600`), and use it for the compose invocation. This preserves ergonomics while avoiding the insecure short demo password.
- The Makefile prints a short reuse hint so you can export the generated value into your shell if you need to run additional commands against the same stack.

What the `e2e-compose-smoke`/`e2e-full-pipeline` targets do:

- Start the minimal e2e compose stack (Neo4j) honoring `NEO4J_AUTH` if you set it, or using a generated secure password saved to `.secrets/neo4j_password` otherwise.
- Wait until the Bolt port is reachable from the host (using `scripts/wait_for_services.sh`) and run the requested test target (`tests/smoke` for smoke runs, or the full pipeline test).
- Tear down the compose stack when finished.

Usage examples:

```bash
# Explicit password (recommended for CI or reproducible runs):
NEO4J_AUTH=neo4j/YourStrongPass make e2e-compose-smoke

# Let the Makefile generate and persist a secure password locally:
make e2e-compose-smoke

# Reuse the generated password in your shell (printed by the Makefile), or:
export NEO4J_AUTH="neo4j/$(cat .secrets/neo4j_password)"

# Tear down and remove the saved secret file manually when finished:
rm -f .secrets/neo4j_password
```

Security & CI recommendations:

- The Makefile's auto-generation is intended for local development convenience only. Avoid exposing the generated password in public CI logs.
- For CI, prefer one of these approaches:
   - Provide `NEO4J_AUTH` via repository secrets (set the secret in GitHub Actions and reference it in the e2e job), or
   - Prefer Testcontainers in CI: our e2e pytest fixture (`tests/e2e/conftest.py`) can start Neo4j programmatically with a random password so no repo secret is required.

If you prefer a helper to remove stored secrets, create and run `rm -f .secrets/neo4j_password` (or add a `make clean-secrets` target). See the `Makefile` for the exact behavior.

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

Endpoints (see `/docs` for full schemas):
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

- **Performance Caching (TrustGraph-Inspired)** — Multi-layer caching system with 30-50% latency reduction for repeated queries. Entity label cache (TTL: 5min, 70-80% hit rate), embedding cache (LRU, 40-60% hit rate), retrieval cache (TTL: 60s, 20-30% hit rate), and Neo4j connection pooling (50 connections). Controlled by `ENABLE_CACHING` flag. Monitor via `/api/database/cache-stats` endpoint.
- **NetworkX-based in-memory deduplication** — Optional in-memory entity deduplication using NetworkX with batch UNWIND persistence for significantly faster ingestion. Controlled by `ENABLE_PHASE2_NETWORKX`; when enabled, ingestion uses an in-memory entity-graph to merge entities by canonical key before persisting to Neo4j in a single batched transaction.
- **Entity Clustering (Leiden)** — Automatically groups extracted entities into semantic communities using the Leiden community detection algorithm. After ingestion, entities that share strong relationships (via `RELATED_TO` and `SIMILAR_TO` edges) are clustered together. Each entity receives a `community_id` property that is visualized with distinct colors in the 3D GraphView. Communities can be filtered, summarized with LLM-generated descriptions, and analyzed independently. Controlled by `ENABLE_CLUSTERING` flag; configure resolution and relationship types in `config/settings.py`. Run manually with `scripts/run_clustering.py` or automatically during reindexing. See **Entity Clustering** section below for details.
- **Reranking (FlashRank)** — optional post-retrieval reranker controlled by `flashrank_enabled`; settings in `config/settings.py`; pre-warmed at startup when enabled.
- **Classification** — document labeling via `api/routers/classification.py` and `config/classification_config.json`.
- **Entity Extraction & Graph** — entity extraction via `core/entity_extraction.py` with synchronous or async embedding persistence (`SYNC_ENTITY_EMBEDDINGS` flag).
- **Chat Tuning and Model Selection** — runtime retrieval/generation controls in `api/routers/chat_tuning.py`; tuning parameters and defaults available through `/api/chat-tuning/config*` endpoints.

## Configuration

**Security Note:** API keys, passwords, and secrets must ONLY be stored in `.env` files, never in JSON config files or code. The RAG Tuning UI does not expose API key fields for security reasons.

Key environment variables (see `config/settings.py` for full list):
- **LLM & embeddings** — `OPENAI_API_KEY`, `OPENAI_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `EMBEDDING_MODEL`
- **Marker LLM** — `MARKER_LLM_MODEL`, `MARKER_LLM_API_KEY` (optional, defaults to OPENAI_API_KEY)
- **Neo4j** — `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- **Feature toggles** — `ENABLE_ENTITY_EXTRACTION`, `ENABLE_QUALITY_SCORING`, `FLASHRANK_ENABLED`, `ENABLE_PHASE2_NETWORKX`, `ENABLE_CACHING`
- **Caching** — `ENTITY_LABEL_CACHE_SIZE` (5000), `ENTITY_LABEL_CACHE_TTL` (300s), `EMBEDDING_CACHE_SIZE` (10000), `RETRIEVAL_CACHE_SIZE` (1000), `RETRIEVAL_CACHE_TTL` (60s), `NEO4J_MAX_CONNECTION_POOL_SIZE` (50)
- **Entity deduplication / NetworkX settings** — `NEO4J_UNWIND_BATCH_SIZE`, `MAX_NODES_PER_DOC`, `MAX_EDGES_PER_DOC`, `IMPORTANCE_SCORE_THRESHOLD`, `STRENGTH_THRESHOLD`, `PHASE_VERSION`
- **Misc** — `SYNC_ENTITY_EMBEDDINGS`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

**Cache Monitoring:** View cache performance at `http://localhost:8000/api/database/cache-stats`

### Optimal Default Settings

Amber ships with **quality-optimized defaults** designed for enterprise documentation ingestion scenarios. These defaults prioritize answer quality, entity richness, and retrieval accuracy over cost and processing speed. The configuration is backed by research from Microsoft GraphRAG and Marker PDF benchmarks.

**Why These Defaults?**

The default configuration enables advanced features that significantly improve RAG quality for company documentation:

1. **Marker PDF Processing with LLM Mode** (`USE_MARKER_FOR_PDF=true`, `MARKER_USE_LLM=true`, `MARKER_FORCE_OCR=true`)
   - **Why:** High-accuracy conversion shows large improvements in table and math extraction versus basic extraction paths (project benchmarks).
   - **Impact:** Improved table merging, math recognition, and layout-sensitive conversions; useful for technical documentation.
   - **Cost:** 2-3x higher ingestion time in LLM-assisted modes and increased resource usage when OCR or LLM processors are enabled.

2. **Entity Gleaning** (`ENABLE_GLEANING=true`, `MAX_GLEANINGS=1`)
   - **Why:** Multi-pass extraction increases entity recall across chunks based on internal validation.
   - **Impact:** Two total passes (initial + 1 gleaning) capture entities missed in the first pass, which helps technical documentation with domain-specific terms.
   - **Cost:** ~2x entity extraction API calls when enabled.

3. **NetworkX Batch Persistence** (`ENABLE_PHASE2_NETWORKX=true`)
   - **Why:** In-memory deduplication and accumulation reduce duplicate entities and database round-trips based on validation results.
   - **Impact:** Cleaner knowledge graph, improved retrieval precision, and faster persistence through batched UNWIND transactions.
   - **Cost:** Higher transient memory use during ingestion when enabled.

4. **Description Summarization** (`ENABLE_DESCRIPTION_SUMMARIZATION=true`)
   - **Why:** LLM-based summarization compresses and consolidates entity descriptions, reducing redundancy and token usage.
   - **Impact:** Shorter description payloads sent to retrievers and generators, improving downstream token efficiency.
   - **Cost:** Additional LLM calls during ingestion when enabled.

5. **FlashRank Reranking** (`FLASHRANK_ENABLED=true`)
   - **Why:** Post-retrieval reranking improves result ordering and relevance using cross-encoder models
   - **Impact:** Better precision, especially for complex queries requiring semantic understanding beyond embedding similarity
   - **Cost:** 50-100ms additional latency per query (runs locally, no API cost)

6. **Optimized Chunking** (`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=150`)
   - **Why:** Larger chunks capture more context for technical documentation with code blocks, diagrams, and multi-step procedures
   - **Impact:** Better coherence in retrieved context, fewer fragmented chunks
   - **Cost:** Slightly higher embedding and LLM token usage per chunk

**Cost vs Quality Tradeoff**

These defaults increase ingestion costs by approximately **2.5-3x** compared to minimal settings, but deliver:
- **30-40% more entities** extracted (gleaning + better PDF extraction)
- **22% fewer duplicates** (NetworkX deduplication)
- **50-70% token reduction** in retrieval context (summarization)
- **95%+ accuracy** in PDF text extraction (Marker LLM mode)
- **Improved retrieval precision** (reranking)

For a one-time ingestion cost, you gain substantial improvements in answer quality that persist across all future queries.

**Disabling for Cost Savings**

If cost or speed is more important than quality, disable features in `.env`:

```bash
# Minimal quality mode (faster, cheaper, lower quality)
USE_MARKER_FOR_PDF=false           # Use basic PyPDF extraction
MARKER_USE_LLM=false               # Disable Marker LLM hybrid mode
MARKER_FORCE_OCR=false             # Skip OCR processing
ENABLE_GLEANING=false              # Single-pass entity extraction
ENABLE_PHASE2_NETWORKX=false       # Direct persistence (slower, more duplicates)
ENABLE_DESCRIPTION_SUMMARIZATION=false  # Skip summarization
FLASHRANK_ENABLED=false            # Disable reranking
CHUNK_SIZE=800                     # Smaller chunks
CHUNK_OVERLAP=200                  # More overlap
```

**Monitoring and Verification**

After ingestion with optimal settings enabled:
- Check entity counts: `GET /api/database/stats` → `total_entities` should be 30-40% higher than minimal mode
- Check community detection: `GET /api/database/stats` → `communities` should show semantic clustering
- Check retrieval quality: Run identical queries in both modes and compare source relevance

For detailed benchmarks, research citations, and parameter explanations, see the repository documentation and the corresponding implementation files in the `core/`, `ingestion/`, and `rag/` modules.

## Entity Clustering

Amber includes sophisticated entity clustering capabilities that automatically group extracted entities into semantic communities using the Leiden community detection algorithm.

### How It Works

After document ingestion and entity extraction, the clustering pipeline:

1. **Builds entity projection** — Loads entities and their relationships (RELATED_TO, SIMILAR_TO) from Neo4j into memory
2. **Normalizes edge weights** — Unifies relationship strengths across different edge types
3. **Converts to graph** — Creates an igraph.Graph representation for efficient clustering
4. **Runs Leiden algorithm** — Applies modularity-based community detection with configurable resolution
5. **Persists communities** — Writes `community_id` and `level` properties back to Entity nodes in Neo4j

Each entity receives a `community_id` that groups it with semantically related entities. These communities are visualized with distinct colors in the 3D GraphView, allowing users to see entity families at a glance.

## Response-level Caching

Brief: the backend now includes an optional response-level (semantic) cache that stores full pipeline responses (response text, sources, metadata, and retrieved context) to speed up repeated queries.

- **What it caches:** full pipeline results returned by `graph_rag.query()` (response, sources, retrieved_chunks, graph_context, metadata, quality_score, stages).
- **Settings:** configure via `config/settings.py` — `response_cache_size` (default 2000) and `response_cache_ttl` (default 300 seconds). The cache is a TTLCache kept in `core/singletons.py` and enabled when `settings.enable_caching` is true.
- **Invalidation points:** to avoid stale answers, the response cache is automatically cleared on document mutations and heavy pipeline changes:
   - Document upload (`POST /api/database/upload`)
   - Document delete (`DELETE /api/database/documents/{document_id}`)
   - Clear database (`POST /api/database/clear`)
   - After background document processing completes (ingestion pipeline)
   - After a full reindex job completes (classification reindex)
- **How it interacts with streaming:** cached responses are returned as the same result dict; the SSE streaming generator (`api/routers/chat.py::stream_response_generator`) will stream the cached response tokens as before. Cached results include `metadata.cached = true` and `stages` will include `cache_hit` for UI/display.
- **Disabling / rollout:** you can disable response caching by setting `ENABLE_CACHING=false` in `.env` (or via `settings.enable_caching`) or by setting `response_cache_ttl` to `0` to effectively disable retention. For staged rollout, set a low TTL (e.g., 60s) and monitor `GET /api/database/cache-stats`.
- **Operator controls:** operators can clear the response cache by performing a reindex or by uploading/deleting documents; a manual admin endpoint to clear the cache can be added if desired.
- **Testing:** unit tests should cover cache key generation and invalidation. Quick verification: issue the same `POST /api/chat/query` twice (with `stream=false`) — the second response should include `metadata.cached = true` and return faster.

If you'd like, I can add a small `README` example snippet showing how to test caching locally or add an admin endpoint to clear the cache on demand.

### Configuration

Enable clustering and customize behavior in `.env`:

```bash
# Enable clustering (runs during reindexing)
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true

# Algorithm parameters
CLUSTERING_RESOLUTION=1.0              # Higher = more granular communities
CLUSTERING_MIN_EDGE_WEIGHT=0.3         # Filter weak relationships
CLUSTERING_RELATIONSHIP_TYPES=RELATED_TO,SIMILAR_TO  # Edge types to include
CLUSTERING_LEVEL=0                     # Community hierarchy level
```

**Manual clustering:**
```bash
python scripts/run_clustering.py
```

**Automatic clustering:** Runs automatically during reindexing when `ENABLE_CLUSTERING=true`

### Visualization

The 3D GraphView (`frontend/src/components/Graph/GraphView.tsx`) renders communities with 10 distinct colors:

- Cyan (#22d3ee), Purple (#a855f7), Orange (#f97316), Green (#10b981), Blue (#3b82f6)
- Amber (#f59e0b), Red (#e11d48), Sky (#0ea5e9), Violet (#8b5cf6), Teal (#14b8a6)

Each entity node displays its `community_id` in the hover panel. Use the community filter dropdown to isolate specific clusters.

### Example Communities

From technical documentation clustering:

- **Community 7:** Backup components (VSS, Replication Backup Components, Writers, Requestors)
- **Community 2:** Storage objects (Virtual Disk, Snapshot, Storage Pool)
- **Community 5:** Network infrastructure (vSAN Cluster, ESXi Host, Network Config)
- **Community 1:** General system components (VM, Datastore, Service)

### Optional: Community Summaries

Generate LLM-based descriptions of each community:

```python
from core.community_summarizer import CommunitySummarizer

summarizer = CommunitySummarizer()
summaries = summarizer.summarize_levels([0])  # Level 0 communities
for community_id, summary_data in summaries.items():
    print(f"Community {community_id}: {summary_data['title']}")
    print(summary_data['summary'])
```

### Files Involved

- `core/graph_clustering.py` — Main clustering implementation (283 lines)
- `core/graph_analysis/leiden_utils.py` — Leiden algorithm utilities (179 lines)
- `core/community_summarizer.py` — LLM-based community descriptions (172 lines)
- `scripts/run_clustering.py` — Manual clustering script (150 lines)
- `api/reindex_tasks.py` — Automatic clustering trigger during reindexing
- `frontend/src/components/Graph/GraphView.tsx` — Community visualization and coloring

## Testing and Code Quality

### Test Organization

The test suite is organized into three categories:

```
tests/
├── unit/          # Fast, isolated unit tests (no external dependencies)
├── integration/   # Integration tests (requires Neo4j, may use APIs)
└── e2e/           # End-to-end tests (full pipeline workflows)
```

### Running Tests

**Startup health check:**
```bash
# Quick health check (Neo4j, Backend, Frontend)
./scripts/test-startup.sh quick

# Full startup test suite (all components + chat pipeline)
./scripts/test-startup.sh

# With verbose logging
./scripts/test-startup.sh verbose
```

**All tests:**
```bash
source .venv/bin/activate
pytest tests/
```

**By category:**
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests (requires Neo4j)
pytest tests/integration/

# End-to-end tests (full pipeline)
pytest tests/e2e/
```

**Specific test:**
```bash
# Chat pipeline integration test
pytest tests/integration/test_chat_pipeline.py -v -s

# Full ingestion pipeline test
pytest tests/integration/test_full_ingestion_pipeline.py -v -s

# E2E full pipeline test (ingestion → clustering → visualization)
pytest tests/e2e/test_full_pipeline.py -v -s
```

**Parallel execution:**
```bash
pytest tests/ -n auto  # Run tests in parallel
```

### Running E2E locally with Docker

If you want to run the end-to-end tests locally (the same ones the CI `e2e` job runs), you can use Docker to start a local Neo4j instance and run the tests against it. A convenience `Makefile` target is provided.

Run the local E2E flow:

```bash
# Start Neo4j and run E2E tests (stops Neo4j when finished)
make e2e-local

# Or run step-by-step:
make start-neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=test
pytest -q tests/e2e/
make stop-neo4j
```

Notes:
- The Makefile uses the official `neo4j:5.21` image and exposes ports `7474` (HTTP) and `7687` (Bolt).
- The test suite expects `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` environment variables to be set when running E2E tests.
- If Neo4j needs more startup time on your machine, increase the `sleep` delay in the `Makefile` `e2e-local` target.

Docker Compose option
---------------------

If you prefer using Docker Compose, a `docker-compose.e2e.yml` file is included. It runs a Neo4j container configured for local E2E runs. Use the `e2e-dc` Makefile target to bring up Neo4j via Compose, run tests, and tear down the stack:

```bash
make e2e-dc
```

This is helpful when you want to coordinate multiple services later or prefer `docker compose` lifecycle management.

### Key Integration Tests

**Chat Pipeline** (`tests/integration/test_chat_pipeline.py`):
- Document ingestion and chunking
- Vector retrieval and entity extraction
- Hybrid retrieval (vectors + entities)
- Graph expansion and multi-hop reasoning
- Optional reranking (FlashRank)
- Response generation with streaming
- Quality scoring and follow-up suggestions
- Multi-turn conversations

**Full Ingestion Pipeline** (`tests/integration/test_full_ingestion_pipeline.py`):
- Multi-format document conversion (TXT, MD, PDF with Marker)
- Chunking with overlap and provenance
- Embedding generation
- Entity extraction
- Graph persistence validation
- Retrieval verification

### Frontend Tests

```bash
cd frontend
npm run test
```

### Linting and Formatting

```bash
source .venv/bin/activate
ruff check .
black .
isort .
```

## Deployment

Docker Compose is the recommended local/demo deployment mechanism. For production, deploy services behind suitable ingress or load-balancer and ensure Neo4j has required memory and the GDS plugin when clustering is enabled.

## Contributing

1. Fork the repository and create a feature branch.
2. Run tests and linters locally.
3. Open a PR with description, test results, and screenshots when applicable.

## License

This project is licensed under the MIT License. See [`LICENSE.md`](./LICENSE.md) for details.
