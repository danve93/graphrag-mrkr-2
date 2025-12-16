## Project Overview
Amber is a document intelligence platform implementing graph-enhanced Retrieval-Augmented Generation (RAG). It combines vector search, graph expansion, entity reasoning, and LLMs to provide contextual, sourced, and high-quality answers over ingested document collections. Amber is built with LangGraph, FastAPI, Next.js, and Neo4j.

## Documentation Updates

- The root `README.md` is the primary onboarding guide. It now highlights quick-start commands (Docker Compose), local development steps for both backend and frontend, and pointers to advanced features like clustering and reranking.
- Keep the README and this file aligned: when you add new capabilities to the RAG pipeline, ingestion flow, or clustering utilities, update both documents with a brief summary and the relevant entry points.
- When documenting commands, prefer concise bash snippets that can be copied verbatim.

### Ingestion-Time Document Classification (M2.4)

Amber can classify documents into categories during ingestion and propagate metadata to chunks to improve retrieval.

Commands:

```bash
# Enable classification via env vars (backend)
export ENABLE_DOCUMENT_CLASSIFICATION=1
export CLASSIFICATION_MODEL=gpt-4o-mini
export CLASSIFICATION_CONFIDENCE_THRESHOLD=0.7

# Reindex a single document by id
docker compose exec backend python scripts/reindex_classification.py --doc-id <doc_id>

# Reindex a batch (first N documents)
docker compose exec backend python scripts/reindex_classification.py --limit 100
```

Notes:
- Document nodes receive: `category`, `categories`, `classification_confidence`, `keywords`, `difficulty`.
- Chunk nodes inherit `category` and include positional metadata like `chunk_number` and optional `semantic_group`.
- Classification falls back to the default `general` category when confidence is below threshold.

### Query Routing with Semantic Caching (M2.3) — Implemented

Amber intelligently routes queries to relevant document categories, improving retrieval precision and reducing latency.

**Architecture:**
- **Router module** (`rag/nodes/query_router.py`): LLM-based classification with confidence scoring
- **Semantic cache** (`rag/nodes/routing_cache.py`): Embedding-based similarity matching (TTLCache + embeddings)
- **Metrics tracking** (`core/routing_metrics.py`): Latency, accuracy, cache hit rates, fallback rates
- **Integration** (`rag/graph_rag.py`): Feature-flagged dual-path (new router or legacy CategoryManager)

**Enable routing:**

```bash
# Set environment variables
export ENABLE_QUERY_ROUTING=true
export QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.7

# Restart backend
docker compose up -d backend

# Monitor metrics
curl http://localhost:8000/api/database/routing-metrics | python3 -m json.tool
```

**Metrics API response:**
```json
{
  "total_queries": 10,
  "avg_routing_latency_ms": 5497.0,
  "cache_hit_rate": 0.4,
  "fallback_rate": 0.1,
  "multi_category_rate": 0.3,
  "top_categories": [["configure", 5], ["install", 3]],
  "routing_accuracy": null
}
```

**Key features:**
- Multi-category routing: Queries can match 2-3 categories simultaneously
- Semantic cache: 30%+ latency reduction on similar queries (0.92 cosine similarity threshold)
- Fallback validation: Auto-expands to all documents when <3 chunks retrieved
- Backward compatible: Falls back to CategoryManager when `ENABLE_QUERY_ROUTING=false`
- Zero-downtime rollback: Disable via environment variable without code changes

**Settings:**
- `enable_query_routing` (default: False)
- `query_routing_confidence_threshold` (default: 0.7)
- `query_routing_strict` (default: False) - no fallback when true
- `enable_routing_cache` (default: True)
- `routing_cache_similarity_threshold` (default: 0.92)
- `routing_cache_size` (default: 1000)
- `routing_cache_ttl` (default: 3600 seconds)
- `fallback_enabled` (default: True)
- `fallback_min_results` (default: 3)

See `config/settings.py` for complete configuration options.

### Neo4j Setup Flags (Indexes/Constraints)

The setup script `scripts/setup_neo4j.py` provides idempotent initialization for indexes and constraints.

Commands:

```bash
# Full setup (test → indexes → constraints → casefold → stats)
docker compose exec backend python scripts/setup_neo4j.py

# Indexes (alias supported)
docker compose exec backend python scripts/setup_neo4j.py --setup
docker compose exec backend python scripts/setup_neo4j.py --setup-indexes

# Constraints (unique Category.name)
docker compose exec backend python scripts/setup_neo4j.py --setup-constraints

# Case-insensitive uniqueness (name_lower)
docker compose exec backend python scripts/setup_neo4j.py --setup-casefold

# Dedupe categories (dry-run then apply)
docker compose exec backend python scripts/dedupe_categories.py --list
docker compose exec backend python scripts/dedupe_categories.py --apply --strategy merge-keep-first
```

Notes:
- `--setup-constraints` ensures a case-sensitive unique constraint on `Category.name`.
- `--setup-casefold` ensures a unique constraint on `Category.name_lower` and populates it from `name` for existing nodes.
- Dedupe script consolidates pre-existing duplicates safely before enforcing constraints.

## Implementation Highlights

The following implementation highlights summarize factual, developer-facing features currently present in the codebase:

- Entity accumulation and deduplication: an in-memory graph layer accumulates extracted entities and relationships before persistence. The implementation merges duplicate entities, accumulates descriptions, sums relationship strengths, tracks provenance, and exports batch Cypher (UNWIND) to Neo4j to reduce database round-trips.

- Multi-layer caching system: long-lived service instances include multiple caches — an entity-label TTL cache to reduce Neo4j lookups, an embeddings cache keyed by text+model to avoid duplicate embedding API calls, and a short-TTL retrieval cache for recent query results. Cache metrics are exported via an API endpoint for observation.

- Document conversion integration: the ingestion pipeline supports a high-accuracy document conversion tool integration that can be used in-process, via CLI, or via a small server. Modes include LLM-assisted extraction, OCR, and table merging, and the integration is configurable for device and runtime settings.

- UI robustness and design updates: layout corrections (removal of top padding on scroll containers), tooltip refactor to avoid adding positional wrapper elements, defensive color parsing, and community-color rendering improvements. The UI also adopts a tokenized design system (spacing, typography, color tokens) and simplified animations.

- Conversation context preservation: the chat system maintains conversation context across follow-up queries by preserving conversation history in the RAG state. Follow-up queries correctly reference previous messages without re-sending full context, reducing token usage and improving response relevance.

- Enhanced cache key generation: retrieval caching uses a comprehensive 14-parameter hash that includes query text, embedding model, retrieval mode, top-k, expansion parameters (depth, threshold, max chunks), similarity threshold, reranking settings, entity/chunk/keyword weights, RRF configuration, and fulltext search flags. This ensures cache hits only occur when all retrieval parameters match, preventing stale or mismatched results.

- Stage timing instrumentation: the RAG pipeline tracks execution time for all four stages (query analysis, retrieval, graph reasoning, generation) with millisecond precision. Each stage emits duration_ms, timestamp, and metadata (chunks retrieved, context items, response length, model used) enabling detailed performance monitoring and optimization.

- UI timing visualization: the frontend displays real-time stage progress with timing information in tooltips (e.g., "Retrieving Documents (350ms) - 5 chunks") and shows total query duration after completion. Stage metadata is rendered inline to provide visibility into pipeline behavior.

- Category-specific prompts: the generation stage selects LLM prompts tailored to document categories. The prompt selector module (`rag/nodes/prompt_selector.py`) provides 10 pre-configured templates (installation, configuration, troubleshooting, API, conceptual, quickstart, reference, example, best_practices, default) with category-specific retrieval strategies (step-back for procedural, PPR for technical reference, balanced for general), format instructions, and specificity levels (concise to comprehensive). Prompts are feature-flagged (`enable_category_prompts`), support conversation history integration, and fall back gracefully to default templates. The `/api/prompts` router enables CRUD operations on prompt templates.

- Structured knowledge graph queries: Amber supports direct graph database queries via natural language through Text-to-Cypher translation. The structured KG executor (`rag/nodes/structured_kg_executor.py`) detects suitable queries (aggregation, path, comparison, hierarchical, relationship types), links entities via embedding similarity (0.85 threshold), generates Cypher using LLM with graph schema context, and executes with iterative correction (max 2 attempts). Integrated into the RAG pipeline as a routing node between query analysis and retrieval: suitable queries execute via Cypher and skip standard retrieval, unsuitable queries fall back to hybrid retrieval. The `/api/structured-kg` router exposes execute, config, schema, and validate endpoints for direct API access. Feature-flagged (`enable_structured_kg`) with configurable entity threshold, correction limits, and timeout settings.

- User feedback and adaptive routing: The system collects user ratings (thumbs up/down) on responses and learns optimal retrieval weights over time. The `FeedbackLearner` class (`rag/nodes/adaptive_router.py`) tracks feedback events with routing metadata and adjusts `hybrid_chunk_weight`, `hybrid_entity_weight`, and other parameters using exponential moving average with configurable learning rates and minimum sample requirements. The feedback API (`api/routers/feedback.py`) exposes endpoints for submitting ratings, viewing metrics and current weights, resetting to defaults, and monitoring recent feedback. Weights are persisted across sessions and enable continuous quality improvement without manual retuning. Feature-flagged (`enable_adaptive_routing`) with configurable learning rate and minimum sample thresholds.

- Smart consolidation: Multi-category chunk consolidation ensures diverse representation when queries match multiple document categories. The `SmartConsolidator` class (`rag/nodes/smart_consolidation.py`) enforces per-category minimum chunk requirements (configurable, default 1 per category), removes semantic duplicates using embedding similarity (0.95 threshold), respects token budgets (8K max context default), and reorders results to maximize diversity while maintaining relevance. Integrated into the retrieval pipeline when query routing is enabled, improving result diversity and preventing redundant context.

These highlights are factual summaries of implemented behavior and do not reference external documentation paths.

## Architecture

### Performance Caching Layer (TrustGraph-Inspired)
Amber implements a multi-layer caching system based on TrustGraph's performance optimization patterns, providing 30-50% latency reduction for repeated queries:

- **Singleton Manager** (`core/singletons.py`): Thread-safe singleton pattern for long-lived service instances:
  - Neo4j driver with connection pooling (50 connections)
  - Entity label cache (TTLCache: 5000 entries, 300s TTL, 70-80% hit rate)
  - Embedding cache (LRUCache: 10000 entries, no TTL, 40-60% hit rate)
  - Retrieval cache (TTLCache: 1000 entries, 60s TTL, 20-30% hit rate)

- **Entity Label Caching** (`core/graph_db.py`): `get_entity_label_cached()` reduces database queries by caching entity name lookups. Used in multi-hop reasoning and graph expansion.

- **Embedding Caching** (`core/embeddings.py`): `get_embedding()` caches embeddings by text+model hash, eliminating redundant API calls for duplicate text.

- **Retrieval Caching** (`rag/retriever.py`): `hybrid_retrieval()` caches results by query+parameters hash with short TTL for consistency with document updates.

- **Cache Monitoring** (`core/cache_metrics.py`): CacheMetrics tracks hit/miss rates. API endpoint at `/api/database/cache-stats` provides real-time metrics.

- **Feature Flag**: `ENABLE_CACHING` setting allows instant rollback if issues occur.

### Core Pipeline (LangGraph State Machine)
The RAG pipeline in `rag/graph_rag.py` uses LangGraph's StateGraph to implement a modular pipeline that can be observed and tuned at runtime. Typical stages include:

1. **Query Analysis** (`query_analysis`) — Normalize the query, extract filters (context documents, hashtags), and prepare retrieval parameters.
2. **Structured KG Router** (`structured_kg_router`) — When enabled, detects queries suitable for direct graph database execution and routes them to Text-to-Cypher translation. Suitable queries skip standard retrieval and proceed directly to generation with structured results. Unsuitable queries fall back to standard retrieval path.
3. **Retrieval** (`retrieval`) — Hybrid retrieval combining embedding similarity with entity-aware candidate selection. Executed only if structured KG router passes through.
4. **Graph Reasoning / Expansion** (`graph_reasoning`) — Neo4j-backed multi-hop expansion that traverses chunk similarities and entity relationships to enrich context.
5. **Reranking (optional)** — Post-retrieval reranking using FlashRank (when enabled) to refine candidate ordering.
6. **Generation** (`generation`) — LLM generation that streams tokens; downstream tasks include quality scoring and follow-up suggestion. Receives either structured KG results or standard retrieval context.

State Management: the pipeline uses plain dict-based `state` objects; each node appends readable stage identifiers to `state["stages"]`. The API streams these stage updates to the frontend (SSE) for progress UI and diagnostics. Each stage tracks execution timing (duration_ms) and metadata (chunks_retrieved, context_items, response_length, model_used, query_type, cypher_query, linked_entities) for observability and performance tuning.

### Ingestion Pipeline
`ingestion/document_processor.py` implements a robust pipeline for multi-format document ingestion and enrichment:

- **Format loaders** (`ingestion/loaders/`): PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, images; loaders convert content to text/Markdown and apply OCR selectively when warranted.
- **Conversion & Preprocessing**: normalization, metadata extraction, and filename handling.
- **Chunking** (`core/chunking.py`): configurable chunk size/overlap with provenance in chunk metadata.
- **Embedding generation** (`core/embeddings.py`): asynchronous manager with concurrency controls and model selection support.
- **Entity extraction** (`core/entity_extraction.py`): optional LLM-based extraction that creates Entity nodes, chunk-entity relationships, and entity embeddings. Extraction may run synchronously for tests or asynchronously in background threads for production ingestion.
- **Quality scoring & filtering** (`core/quality_scorer.py`): filters or marks low-quality chunks to improve retrieval and grounding.
- **Graph persistence** (`core/graph_db.py`): creates Document, Chunk, and Entity nodes and the relationship edges used for expansion and clustering.
- **Entity clustering** (`core/graph_clustering.py`): optional Leiden community detection that groups related entities into semantic clusters. After ingestion, entities connected by strong relationships are assigned `community_id` properties. These communities are visualized with distinct colors in GraphView, can be filtered in the UI, and optionally summarized with LLM-generated descriptions via `core/community_summarizer.py`.

The `DocumentProcessor` exposes synchronous and asynchronous entry points, supports chunk-only ingestion, and tracks background entity extraction operations so the UI can display progress and status.

### API Layer (FastAPI)
- **Routers**: `api/routers/` contains `chat.py`, `documents.py`, `database.py`, `history.py`, `classification.py`, `chat_tuning.py`, `rag_tuning.py`, `prompts.py`, `structured_kg.py`, `feedback.py`, `documentation.py`, `admin_user_management.py`, and `jobs.py`.
- **Models**: `api/models.py` defines Pydantic request/response models. The `ChatRequest` model exposes `llm_model` and `embedding_model` fields so the UI can select models at runtime.
- **Services**: `api/services/` contains business logic like chat history and follow-up question generation.
- **Streaming**: SSE streams emit structured payloads (`stage`, `token`, `sources`, `quality_score`, `follow_ups`, `metadata`) consumed by the frontend.
- **Feedback API** (`api/routers/feedback.py`): Exposes endpoints for user feedback submission, metric tracking, weight viewing, and reset functionality for adaptive routing.
- **Admin Management** (`api/routers/admin_user_management.py`): Provides admin authentication and endpoints for user activity tracking, shared conversation management, and system metrics.
- **Documentation Serving** (`api/routers/documentation.py`): Serves markdown documentation files from the `documentation/` folder via HTTP, supporting dynamic documentation access.

The lifecycle hook (`api.main.lifespan`) performs startup tasks (user token ensure, optional FlashRank prewarm) and logs provider resolution for debugging.

### Frontend (Next.js + React)
- **State Management**: Zustand plus React Context providers (branding, theme, chat tuning).
- **Features**: Chat interface with streaming, history panel, upload UI, database explorer, Chat Tuning panel (model and retrieval controls).
- **SSE Handling**: `ChatInterface` reconstructs token streams into messages, renders stage updates, shows sources and quality scores, and provides follow-up suggestions.
- **Type Safety**: TypeScript models in `src/types/index.ts` mirror backend Pydantic models for safe API consumption.

## Critical Data Flows

### Chat Query Flow
1. Frontend sends `ChatRequest` with message, session_id, retrieval_mode, top_k, temperature
2. Backend initializes `RAGState` dict with query + parameters + conversation history
3. LangGraph invokes workflow: query → retrieve → reason → generate
4. Each stage tracks timing (time.time() measurements) and metadata
5. Streaming response emits: stages (with duration_ms, timestamp, metadata) → tokens → sources → quality_score → follow_ups
6. Frontend reconstructs Message with sources, quality_score, follow_up_questions, stages, total_duration_ms
7. LoadingIndicator displays real-time stage progress with timing in tooltips

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
- **Query routing**: intelligent query analysis determines optimal retrieval strategy:
  - **Entity-focused** (comparative/analytical/relationship queries): boost entity weight ×1.3, reduce chunk weight ×0.8 for better graph traversal
  - **Keyword-focused** (procedural/how-to/version queries): boost keyword ×1.4 and chunk ×1.1, reduce entity ×0.7 for exact-term matching
  - **Balanced** (simple factual queries): default weights for broad coverage
- **Optional query expansion**: when initial retrieval returns fewer than `query_expansion_threshold` results, LLM-based expansion generates 3-5 synonyms/related terms to improve recall
- **Optional RRF fusion**: when enabled, Reciprocal Rank Fusion combines the ranked lists from vector, entity, keyword (BM25), and path-based results into a single robust ordering before reranking.
- **Keyword tuning**: the RAG Tuning panel exposes `enable_chunk_fulltext` and `keyword_search_weight` to adjust lexical (BM25) influence alongside RRF.
- **Hybrid weights**: the panel also exposes `hybrid_chunk_weight` and `hybrid_entity_weight` so teams can balance semantic (vectors) and entity-aware signals per use case.
- **Optional reranking**: FlashRank reranker can re-order the top candidates for generation using `flashrank_blend_weight` to combine rerank scores with hybrid scores.

Multi-hop reasoning evaluates path strength using relationship strength and entity importance to surface context-rich multi-hop evidence for the generator.

## Key Configuration Patterns
Settings live in `config/settings.py` and can be set through environment variables or overridden by chat tuning at runtime:

- **LLM & Embeddings**: `llm_provider`, `openai_api_key`, `openai_model`, `ollama_model`, `embedding_model`.
- **Concurrency & Rate Limits**: `embedding_concurrency`, `llm_concurrency`, `embedding_delay_min/max`, `llm_delay_min/max`.
- **Retrieval & Expansion**: `hybrid_chunk_weight`, `hybrid_entity_weight`, `max_expanded_chunks`, `max_expansion_depth`, `expansion_similarity_threshold`.
- **Query Analysis & Routing**: `enable_query_expansion`, `query_expansion_threshold` (default: 3 results).
- **Query Routing (M2.3)**: `enable_query_routing`, `query_routing_confidence_threshold`, `query_routing_strict`, `enable_routing_cache`, `routing_cache_similarity_threshold`, `routing_cache_size`, `routing_cache_ttl`, `fallback_enabled`, `fallback_min_results`.
- **Structured KG (M3.3)**: `enable_structured_kg`, `structured_kg_entity_threshold`, `structured_kg_max_corrections`, `structured_kg_timeout`, `structured_kg_query_types`.
- **Adaptive Routing**: `enable_adaptive_routing`, `adaptive_routing_learning_rate` (default: 0.1), `adaptive_routing_min_samples` (default: 5) — controls exponential moving average learning and feedback event tracking.
- **RRF Fusion**: `enable_rrf`, `rrf_k` (rank discount constant; higher is flatter influence).
- **Reranking**: `flashrank_enabled`, `flashrank_model_name`, `flashrank_max_candidates`, `flashrank_blend_weight`, `flashrank_batch_size`.
- **Caching**: `enable_caching`, `entity_label_cache_size`, `entity_label_cache_ttl`, `embedding_cache_size`, `retrieval_cache_size`, `retrieval_cache_ttl`, `neo4j_max_connection_pool_size`.
- **Entity / OCR**: `enable_entity_extraction`, `SYNC_ENTITY_EMBEDDINGS`, `enable_ocr`, `ocr_quality_threshold`.
- **Clustering**: `enable_clustering`, `enable_graph_clustering`, `clustering_resolution`, `clustering_min_edge_weight`, `clustering_relationship_types`, `clustering_level`.

### Development Workflow
- Create and activate a virtualenv: `python3 -m venv .venv && source .venv/bin/activate`.
- Install backend deps: `pip install -r requirements.txt`.
- Copy `.env.example` → `.env` and fill credentials (OpenAI, Neo4j, optional FlashRank cache path).
- Start backend: `python api/main.py` (reload enabled) and frontend: `cd frontend && npm run dev`.
- Tests: `pytest tests/` and `cd frontend && npm run test`.
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
- **Clustering**: Leiden community detection groups related entities into semantic clusters. Entities connected by strong relationships (via `RELATED_TO`, `SIMILAR_TO` edges) receive `community_id` assignments that are visualized with distinct colors in GraphView. Run manually via `scripts/run_clustering.py` or automatically during reindexing when `ENABLE_CLUSTERING=true`. Configure resolution (granularity), min edge weight (filter weak connections), and relationship types in settings. Optional community summaries generated via `core/community_summarizer.py` provide LLM-generated descriptions of each cluster's theme and members.

For deeper details see source files under `api/`, `ingestion/`, `core/`, and `rag/`.
