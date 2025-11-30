```markdown
# Changelog — Implementation Summary

This changelog lists implemented features and components across the repository.

1) GraphRAG Improvements
- Gleaning
	- Multi-pass entity extraction (gleaning) to increase recall across chunks
	- Compatibility mitigations
	- Validation results and integration verification documents included
- NetworkX intermediate layer
	- `EntityGraph` (in-memory NetworkX graph) for accumulation, deduplication, and provenance
	- Merge/accumulate entity descriptions, sum relationship strengths, create orphan nodes
	- Batch Cypher export using UNWIND to persist entities/relationships in a single transaction
	- Tunable parameters such as `NEO4J_UNWIND_BATCH_SIZE`, mention-count tracking, importance scoring
- Tuple delimiters and schema changes
	- Tuple-delimiter handling and compatibility planning for next-stage schema changes
	- Migration and compatibility risk analyses included
- Description summarization
	- LLM-based description summarization step to compress and improve entity descriptions before persistence

2) cache (multi-layer caching & metrics)
- Singleton manager for long-lived service instances (Neo4j driver, embedding manager, etc.)
- Entity-label cache: TTL cache (configurable size & TTL) to avoid repeated DB lookups
- Embedding cache: LRU cache keyed by text+model hash to eliminate duplicate embedding calls
- Retrieval cache: short-TTL cache for hybrid retrieval results (query+params key)
- CacheManager patterns documented (multi-backend support, TTL, cleanup task)
- Cache metrics and monitoring: `core/cache_metrics.py` and `/api/database/cache-stats` endpoint

3) marker (PDF & document conversion integration)
- Marker library integration (three integration modes supported):
	- Library: call Marker converters directly in `ingestion/converters.py`
	- CLI: run Marker CLI (`marker_single`) for process-isolated conversion
	- Server: `marker_server` FastAPI service for remote conversion
- Supported inputs: PDF, images, PPT/PPTX, DOC/DOCX, XLS/XLSX, HTML, EPUB
- Outputs: Markdown, HTML, JSON, Chunks, OCR JSON, images
- Modes: LLM-assisted extraction, optional forced OCR, table merging and LLM processors
- Configuration knobs: `USE_MARKER_FOR_PDF`, `MARKER_USE_LLM`, `MARKER_FORCE_OCR`, device/dtype controls
- Licensing & operational notes: GPL considerations and recommended isolation patterns

4) ui (layout, design system, and interaction fixes)
- Typography system: new font choices, 9-level type scale, consistent headings and body text
- Color system: simplified palette, single accent color, component color updates
- Spacing system: 8‑point grid tokens and tokenized CSS variables (`--space-*`)
- Border radius standardization and tokenization
- Animation simplification: removed heavy animations and retained minimal transitions
- Component updates: message rendering simplification, input standardization, sidebar navigation cleanup
- Interaction improvements: drag-and-drop feedback, mention/autocomplete styling, button states
- Graph visualization: defensive community-color handling, hover panel and filter updates
- Chat tuning UI: layout and parameter UI improvements
- Theme & accessibility: dark mode refinements, improved contrast and focus indicators
- Performance: reduced dependencies (removed Framer Motion), CSS-only transitions, optimized rendering
- Cleanup: removed obsolete components and routes (classification panel, comblocks, unused pages)
- Layout bug fixes and interaction hardening implemented in code: removed top padding from scroll containers, fixed tooltip wrapper behavior to avoid overlay by attaching handlers to DOM children directly


## 2025-11-29 — Response-level caching, singleflight & cache-adapter

- **Response-level cache (in-process TTL):** Implemented an in-process TTL response cache to store full RAG outputs for repeated queries. Keys are session-scoped by default and include a compact fingerprint of recent chat history (to avoid returning personalized results from different sessions). Cached responses are annotated with `metadata.cached` and include a `cache_hit` stage when served.
  - Main files: `rag/graph_rag.py`, `core/singletons.py`, `tests/test_response_cache.py`

- **Singleflight (per-key) locking:** Added a per-response-key singleflight pattern to prevent duplicate expensive generation work within a process. The implementation provides a persistent lock registry and a `ResponseKeyLock` context manager; callers acquire the per-key lock, re-check the cache, compute the response if still missing, write the result into the cache while holding the lock, then release the lock so waiters read the cached response.
  - Main files: `core/singletons.py` (lock registry, ResponseKeyLock), `rag/graph_rag.py` (lock usage)

- **Monkeypatch-friendly generation node:** The generation node now imports `rag.nodes.generation` dynamically so tests can monkeypatch `rag.nodes.generation.generate_response` (used in the response-cache unit tests).
  - Main file: `rag/graph_rag.py`

- **Cache adapter abstraction:** Added `core/cache.py` with a `CacheAdapter` interface and an `InMemoryCacheAdapter` that wraps the existing `cachetools.TTLCache`. This creates a clear extension point to add a Redis-backed adapter in the future without changing call sites.
  - New file: `core/cache.py`

- **Tests:** Added `tests/test_response_cache.py` validating both session-scoped caching and singleflight behavior. Tests monkeypatch the generator to avoid LLM calls and simulate concurrent requests.

- **Invalidation hooks:** Added best-effort cache invalidation calls in reindex, document upload, and deletion flows so cached responses depending on changed documents are not served stale.
  - Touched files: `api/reindex_tasks.py`, `api/routers/database.py`, `ingestion/document_processor.py`

- **Docs & README:** Added documentation describing the response-level cache, its key composition, metrics to track (hits/misses, lock contention), and a roadmap to move to a distributed L2 cache (Redis) for multi-instance deployments.

- **Dev/runtime fixes:** Adjusted local Neo4j memory settings used during testing/dev (docker-compose) to avoid container startup failures on machines with limited memory.

Notes and next steps:
- The current implementation is optimized for single-instance deployments and provides immediate latency/LLM-cost benefits. For horizontal scaling, add a Redis-based L2 cache and distributed locking (or a Lua atomic get-or-set pattern) to enable cross-instance singleflight. The codebase is now prepared with a pluggable adapter point so a Redis adapter can be introduced with minimal changes.

```
