## Project Overview
Amber is a document intelligence platform implementing graph-enhanced Retrieval-Augmented Generation (RAG). It combines vector search, graph expansion, entity reasoning, and LLMs to provide contextual, sourced, and high-quality answers over ingested document collections. Amber is built with LangGraph, FastAPI, Next.js, and Neo4j.

## Documentation Updates

- The root `README.md` is the primary onboarding guide. It now highlights quick-start commands (Docker Compose), local development steps for both backend and frontend, and pointers to advanced features like clustering and reranking.
- Keep the README and this file aligned: when you add new capabilities to the RAG pipeline, ingestion flow, or clustering utilities, update both documents with a brief summary and the relevant entry points.
- When documenting commands, prefer concise bash snippets that can be copied verbatim.

## Architecture

### Core Pipeline (LangGraph State Machine)
The RAG pipeline in `rag/graph_rag.py` uses LangGraph's StateGraph to implement a modular pipeline that can be observed and tuned at runtime. Typical stages include:

1. **Query Analysis** (`query_analysis`) — Normalize the query, extract filters (context documents, hashtags), and prepare retrieval parameters.
2. **Retrieval** (`retrieval`) — Hybrid retrieval combining embedding similarity with entity-aware candidate selection.
3. **Graph Reasoning / Expansion** (`graph_reasoning`) — Neo4j-backed multi-hop expansion that traverses chunk similarities and entity relationships to enrich context.
4. **Reranking (optional)** — Post-retrieval reranking using FlashRank (when enabled) to refine candidate ordering.
5. **Generation** (`generation`) — LLM generation that streams tokens; downstream tasks include quality scoring and follow-up suggestion.

State Management: the pipeline uses plain dict-based `state` objects; each node appends readable stage identifiers to `state["stages"]`. The API streams these stage updates to the frontend (SSE) for progress UI and diagnostics.

### Ingestion Pipeline
`ingestion/document_processor.py` implements a robust pipeline for multi-format document ingestion and enrichment:

- **Format loaders** (`ingestion/loaders/`): PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, images; loaders convert content to text/Markdown and apply OCR selectively when warranted.
- **Conversion & Preprocessing**: normalization, metadata extraction, and filename handling.
- **Chunking** (`core/chunking.py`): configurable chunk size/overlap with provenance in chunk metadata.
- **Embedding generation** (`core/embeddings.py`): asynchronous manager with concurrency controls and model selection support.
- **Entity extraction** (`core/entity_extraction.py`): optional LLM-based extraction that creates Entity nodes, chunk-entity relationships, and entity embeddings. Extraction may run synchronously for tests or asynchronously in background threads for production ingestion.
- **Quality scoring & filtering** (`core/quality_scorer.py`): filters or marks low-quality chunks to improve retrieval and grounding.
- **Graph persistence** (`core/graph_db.py`): creates Document, Chunk, and Entity nodes and the relationship edges used for expansion and clustering.

The `DocumentProcessor` exposes synchronous and asynchronous entry points, supports chunk-only ingestion, and tracks background entity extraction operations so the UI can display progress and status.

### API Layer (FastAPI)
- **Routers**: `api/routers/` contains `chat.py`, `documents.py`, `database.py`, `history.py`, `classification.py`, `chat_tuning.py`, and `jobs.py`.
- **Models**: `api/models.py` defines Pydantic request/response models. The `ChatRequest` model exposes `llm_model` and `embedding_model` fields so the UI can select models at runtime.
- **Services**: `api/services/` contains business logic like chat history and follow-up question generation.
- **Streaming**: SSE streams emit structured payloads (`stage`, `token`, `sources`, `quality_score`, `follow_ups`, `metadata`) consumed by the frontend.

The lifecycle hook (`api.main.lifespan`) performs startup tasks (user token ensure, optional FlashRank prewarm) and logs provider resolution for debugging.

### Frontend (Next.js + React)
- **State Management**: Zustand plus React Context providers (branding, theme, chat tuning).
- **Features**: Chat interface with streaming, history panel, upload UI, database explorer, Chat Tuning panel (model and retrieval controls).
- **SSE Handling**: `ChatInterface` reconstructs token streams into messages, renders stage updates, shows sources and quality scores, and provides follow-up suggestions.
- **Type Safety**: TypeScript models in `src/types/index.ts` mirror backend Pydantic models for safe API consumption.

## Critical Data Flows

### Chat Query Flow
1. Frontend sends `ChatRequest` with message, session_id, retrieval_mode, top_k, temperature
2. Backend initializes `RAGState` dict with query + parameters
3. LangGraph invokes workflow: query → retrieve → reason → generate
4. Streaming response emits: stages → tokens → quality_score
5. Frontend reconstructs Message with sources, quality_score, follow_up_questions

### Document Ingestion Flow
1. Upload triggers `ingestion/document_processor.py::process_document_async()`
2. Loader extracts text, chunker segments with overlap
3. Entity extraction runs (optional, async)
4. All chunks embedded via `core/embeddings.py::embedding_manager`
5. Neo4j stores: Document → Chunk nodes + Entity nodes + relationships
6. UI polls `/api/documents/processing-status` to track progress

### Hybrid Retrieval (Key Algorithm)
Hybrid retrieval combines multiple signals and steps:

- **Vector search** (embeddings): initial top-K candidates by embedding similarity.
- **Graph expansion**: expand candidates by traversing `SIMILAR_TO` chunk edges and entity relationships in Neo4j (controlled by `expansion_similarity_threshold`, `max_expansion_depth`, `max_expanded_chunks`).
- **Optional reranking**: FlashRank reranker can re-order the top candidates for generation using `flashrank_blend_weight` to combine rerank scores with hybrid scores.

Multi-hop reasoning evaluates path strength using relationship strength and entity importance to surface context-rich multi-hop evidence for the generator.

## Key Configuration Patterns
Settings live in `config/settings.py` and can be set through environment variables or overridden by chat tuning at runtime:

- **LLM & Embeddings**: `llm_provider`, `openai_api_key`, `openai_model`, `ollama_model`, `embedding_model`.
- **Concurrency & Rate Limits**: `embedding_concurrency`, `llm_concurrency`, `embedding_delay_min/max`, `llm_delay_min/max`.
- **Retrieval & Expansion**: `hybrid_chunk_weight`, `hybrid_entity_weight`, `max_expanded_chunks`, `max_expansion_depth`, `expansion_similarity_threshold`.
- **Reranking**: `flashrank_enabled`, `flashrank_model_name`, `flashrank_max_candidates`, `flashrank_blend_weight`, `flashrank_batch_size`.
- **Entity / OCR**: `enable_entity_extraction`, `SYNC_ENTITY_EMBEDDINGS`, `enable_ocr`, `ocr_quality_threshold`.

### Development Workflow
- Create and activate a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install backend deps: `pip install -r requirements.txt`.
- Copy `.env.example` → `.env` and fill credentials (OpenAI, Neo4j, optional FlashRank cache path).
- Start backend: `python api/main.py` (reload enabled) and frontend: `cd frontend && npm run dev`.
- Tests: `pytest api/tests/` and `cd frontend && npm run test`.
- Use `SYNC_ENTITY_EMBEDDINGS=1` for deterministic entity extraction runs in test environments.

## Code Patterns & Conventions
- **Error Handling**: Async functions wrapped in try/except; errors returned in state/response dicts (never thrown to client)
- **Streaming**: All LLM responses use SSE; chunk data must be JSON serializable
- **Pydantic Models**: All API request/response types defined in `api/models.py` (e.g., ChatRequest has `message`, `session_id`, `context_documents`)
- **Neo4j Queries**: Use parameterized queries with `$param` syntax; avoid string concatenation
- **Service Isolation**: Business logic in `api/services/` and `core/`; routers only handle HTTP concerns (validation, response formatting)

### MVP Scope

Purpose: deliver an MVP that demonstrates Amber's graph-augmented RAG capabilities across a modern UI: ingest documents, run hybrid retrieval with optional reranking and multi-hop reasoning, and chat with streamed, sourced responses.

Core UX highlights:
- Responsive chat interface with streaming and sources
- Persistent conversation history and session restore
- Upload and ingestion tracking with chunk previews and entity displays
- Chat tuning UI to change retrieval/generation parameters and select LLM/embedding models at request time

Data flow highlights:
- Ingest → Chunk → Embed → Persist (Neo4j) → Hybrid retrieval → Optional rerank → Generate → Stream to UI

### Current Entity Model (from `core/entity_extraction.py`)
**Canonical Types:** Component, Service, Node, Domain, Class of Service, Account, Account Type, Role, Resource, Quota Object, Backup Object, Item, Storage Object, Migration Procedure, Certificate, Config Option, Security Feature, CLI Command, API Object, Task, Procedure, Concept, Document, Person, Organization, Location, Event, Technology, Product, Date, Money

**Relationship Patterns:** Component↔Node/Component/Feature, provisioning links, backup/storage coverage, config/security associations, migration/CLI/task/procedure edges, generic RELATED_TO fallback

### Operations & Setup
 **Docker deployment:** Unified entrypoint (`docker compose up -d`) for chat UI, backend API, and database upload. Use `docker compose up -d --build` when you need to rebuild images after Dockerfile changes.
- **Runtime:** Backend via `python api/main.py`, frontend via `npm run dev`, Neo4j local or Docker, SSE streams responses
- **Quality scoring:** Feature toggles in settings/env vars; scoring is non-blocking and cached
- **Docker deployment:** Unified entrypoint (`docker-compose up -d --build`) for chat UI, backend API, and database upload

### MVP Success Criteria
- Users upload documents → appear in database view with parsed metadata → can be removed
- Chat streams responses with sources, respects conversation context, displays quality scores
- Retrieval leverages hybrid + multi-hop reasoning to connect entities across documents with adjustable parameters
- Full setup via provided scripts without additional code changes

## Operational Notes & Do Not

Do NOT commit secrets or hardcoded API keys. Use environment variables and `.env` instead.

Operational notes:
- **Reranker prewarm**: `api/main.py` will attempt to import and prewarm FlashRank when `flashrank_enabled` is true; failures are logged and do not prevent startup.
- **Chat tuning**: tuning values are applied as runtime defaults and can be used to override retrieval/generation parameters without restarting the server.
- **Clustering**: Leiden clustering requires Neo4j with GDS and sufficient memory; use `scripts/run_clustering.py` and `scripts/build_leiden_projection.py`.

For deeper details see source files under `api/`, `ingestion/`, `core/`, and `rag/`.