# Recent changes (2025-11-29)

This file lists recent development work intended to be merged into the main `CHANGELOG.md`.

## Response-level caching, singleflight & cache-adapter

- **Response-level cache (in-process TTL):** Implemented an in-process TTL response cache to store full RAG outputs for repeated queries. Keys are session-scoped by default and include a compact fingerprint of recent chat history to avoid returning personalized results from different sessions. Cached responses are annotated with `metadata.cached` and include a `cache_hit` stage when served.
  - Main files: `rag/graph_rag.py`, `core/singletons.py`, `tests/test_response_cache.py`

- **Singleflight (per-key) locking:** Added a per-response-key singleflight pattern to prevent duplicate expensive generation work within a process. The implementation uses a persistent lock registry and a `ResponseKeyLock` context manager; callers acquire the per-key lock before computing/writing a response, and waiting callers re-check the cache after the lock is released.
  - Main files: `core/singletons.py` (lock registry, `ResponseKeyLock`), `rag/graph_rag.py` (lock usage)

- **Monkeypatch-friendly generation node:** The generation node now dynamically imports `rag.nodes.generation` so test suites can monkeypatch `rag.nodes.generation.generate_response`.
  - Main file: `rag/graph_rag.py`

- **Cache adapter abstraction:** Added `core/cache.py` with a `CacheAdapter` interface and an `InMemoryCacheAdapter` that wraps the existing `cachetools.TTLCache`. This provides a clear extension point so a Redis adapter can be introduced later with minimal code changes.
  - New file: `core/cache.py`

- **Invalidation hooks:** Best-effort invalidation hooks added in reindex/upload/delete flows so cached responses depending on changed documents are not served stale.
  - Files touched: `api/reindex_tasks.py`, `api/routers/database.py`, `ingestion/document_processor.py`

- **Tests:** Added `tests/test_response_cache.py` validating session-scoped caching and singleflight behavior. Tests patch generation to avoid LLM calls.

- **Docs & README:** Added documentation describing the response-level cache, key composition, and a migration roadmap to a distributed L2 cache (Redis) if/when horizontal scaling is required.

- **Dev/runtime fixes:** During testing a local Neo4j memory configuration in `docker-compose.yml` was tuned to avoid container startup failures on machines with limited memory.

---

To merge this into the main `docs/CHANGELOG.md`, copy the section above and paste it in the desired location within that file.
