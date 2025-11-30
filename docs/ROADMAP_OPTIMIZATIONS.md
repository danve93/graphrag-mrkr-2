# Roadmap: Implementing Performance & Retrieval Optimizations

Purpose
- Provide a concrete, prioritized roadmap to implement the optimization ideas previously inspected in the codebase (streaming tokens, semantic/prompt caching, hybrid search improvements, RRF, reranking, metadata pre-filtering). This document lists milestones, concrete implementation steps, targeted files, test/validation guidance, estimated effort, rollout and monitoring plans, and known risks.

Constraints and approach
- All recommendations are implementation-focused and reference exact files and functions discovered during the code scan. No assumptions beyond the scanned code are made.
- We prioritize low-risk, high-impact changes first (response-level caching) and then larger architecture moves (LLM streaming and BM25 + RRF).
- Each milestone contains: goal, concrete implementation steps, where to change code (file references), tests to add, estimated complexity (S/M/L), and rollout notes.

---

Milestone 0 — Pre-work (already done)
- Completed: repository scan and factual findings file (`docs/OPTIMIZATION_FINDINGS.md`).
- Files: `docs/OPTIMIZATION_FINDINGS.md` (created)

---

Milestone 1 — Response-level (semantic) caching (Priority: High, Effort: Small)
Goal
- Cache full pipeline responses (response text, sources, metadata) for repeated or similar queries to reduce latency for common patterns.

Why first
- Low code changes, immediate impact for repeated queries (FAQs), minimal risk. Improves perceived performance while other, larger work proceeds.

Concrete steps
1. Add a response cache singleton in `core/singletons.py`:
   - `get_response_cache()` returns `TTLCache(maxsize=settings.response_cache_size, ttl=settings.response_cache_ttl)`.
   - Add `hash_response_params(query, retrieval_mode, top_k, chunk_weight, entity_weight, path_weight, context_document_ids, model)` helper to compute cache key.
2. Add cache lookup in `rag/graph_rag.py::GraphRAG.query()` before invoking the workflow:
   - Compute normalized key, check cache, if hit return cached dict exactly (include `stages` indicating cache hit).
3. After successful pipeline execution, store the final result dict in the response cache with the computed key.
4. Cache invalidation:
   - Whenever documents are added/updated/reindexed, flush response cache. Locate reindex endpoints (e.g., `api/routers/database.py` or `api/routers/documents.py`) and call a `clear_response_cache()` helper in `core/singletons.py`.
5. Metrics: extend `core/cache_metrics.py` to track `response_hits`/`response_misses`.

Files to modify
- `core/singletons.py` (add response cache helpers and `hash_response_params`)
- `rag/graph_rag.py` (check cache before workflow; set cache after)
- `core/cache_metrics.py` (add counters)
- `api/routers/database.py` or the reindex endpoints (call cache cleanup on document/pipeline changes)

Testing and validation
- Unit tests: caching behavior (hit/miss), cache key variation tests (different top_k, different context_documents produce different keys).
- Integration test: issue the same chat request twice; second response should be served from cache and be significantly faster.
- Add CLI script or test in `scripts/` to run benchmark (repeat query N times and log median latency).

Rollout
- Feature flag via `settings.enable_response_caching` (default off). Start with TTL small (e.g., 60s), monitor correctness, then increase TTL if safe.

Estimated time
- Dev + tests: 1-2 days.

---

Milestone 2 — True LLM token streaming (Priority: High, Effort: Large)
Goal
- Stream tokens from the LLM provider to the client SSE as the model generates them (reduces Time To First Token). This is distinct from streaming pre-generated text.

Why next
- Provides real TTFT improvements. Larger effort due to provider differences and wiring streaming through the existing pipeline.

Concrete steps
1. Add streaming LLM methods in `core/llm.py`:
   - Implement `stream_generate_openai(messages, ... )` that calls OpenAI streaming API (`openai.chat.completions.create(stream=True, ...)`) or OpenAI Streams if available, and yields token chunks.
   - Implement `stream_generate_ollama(...)` if Ollama supports streaming; otherwise note fallback.
   - Provide a unified generator interface: `LLMManager.stream_generate(... )` that yields dicts like `{ "type": "token", "token": str, "delta": {}}`.
2. Generation node adaptation:
   - Update `rag/nodes/generation.py::generate_response` (or create a streaming variant `generate_response_stream`) to accept a `stream=True` flag and return an async generator that yields tokens plus final metadata.
   - Because the LangGraph `workflow.invoke(...)` is synchronous, it may be simpler to leave `GraphRAG.query()` unchanged for non-streaming calls and add a dedicated streaming path: `graph_rag.stream_query(...)` which sets up retrieval and then returns an async generator that streams tokens by calling the streaming generation node.
3. API changes:
   - Update `api/routers/chat.py::stream_response_generator` to consume and forward tokens as they arrive from the LLM stream (instead of sending pre-generated words).
   - Ensure SSE keepalive, `X-Accel-Buffering: no` and `Connection: keep-alive` remain set.
4. Backpressure and error handling:
   - Handle LLM connection drops: emit SSE `error` event and close gracefully.
   - Add chunking and rate-limiting between model tokens and SSE to avoid socket overload (small buffer queue).
5. Tests:
   - Unit: test token generator with mocked openai streaming client.
   - Integration: run server and consume SSE stream, verify first token latency and that final response matches non-streaming.

Files to modify
- `core/llm.py` (add streaming methods and unified stream interface)
- `rag/nodes/generation.py` (add streaming generate variant)
- `rag/graph_rag.py` (add `stream_query` or adapt to support streaming)
- `api/routers/chat.py` (forward model tokens)

Testing and validation
- Mock-based tests for streaming generator.
- End-to-end: measure Time To First Token compared to baseline.

Rollout
- Feature-flag `settings.enable_llm_streaming` to toggle streaming.
- Start with providers/environments known to support streaming.

Estimated time
- Dev + tests + QA: 3–6 days (depends on provider streaming semantics and testing coverage).

Risks
- Provider streaming APIs differ; need careful abstraction.
- SSE and long-lived connections need resource tuning; scale testing required.

---

Milestone 3 — Chunk-level BM25 / full-text search (Priority: Medium, Effort: Medium)
Goal
- Add lexical keyword search over chunk content so exact matches (IDs, legal clauses) are retrievable by BM25-style ranking.

Concrete steps
1. Add chunk fulltext index in Neo4j and implement `core/graph_db.py::chunk_keyword_search(query_text, top_k)` using `CALL db.index.fulltext.queryNodes('chunk_text', $query_text)` and return `c.id, score, c.content, d.id`.
2. Ensure index created safely (idempotent) like the entity index is created in `entity_similarity_search`.
3. In `rag/retriever.py::_hybrid_retrieval_direct`, call `chunk_keyword_search` and produce a ranked list of chunk candidates (lexical list) alongside the vector list.
4. Add a fusion step (see Milestone 4) and use the fused ranking for candidate selection and reranking.

Files to modify
- `core/graph_db.py` (add chunk fulltext function)
- `rag/retriever.py` (call into keyword search and consume results)

Testing and validation
- Unit tests for the fulltext call (mock Neo4j) and integration tests that confirm chunk keyword matches surface as expected.

Estimated time
- Dev + tests: 2–4 days.

---

Milestone 4 — Reciprocal Rank Fusion (RRF) (Priority: Medium, Effort: Small)
Goal
- Implement RRF to combine ordered lists (vector, chunk-keyword, entity) robustly.

Concrete steps
1. Add helper in `rag/retriever.py`:
   - `_apply_rrf(list_of_ranked_ids, k=60)` computing scores: score[id] = sum(1 / (k + rank_i)) over all lists.
   - Return fused ordered list of candidate IDs with aggregated score.
2. In `_hybrid_retrieval_direct`, after obtaining vector_results, keyword_results, and entity_results, call `_apply_rrf` to create fused ordering and select top-N candidates to pass to reranker.
3. Maintain existing weighted scoring as a fallback if RRF disabled. Add settings flag `settings.enable_rrf`.

Files to modify
- `rag/retriever.py` (implement `_apply_rrf` and integrate)

Testing and validation
- Unit tests for `_apply_rrf` with synthetic ranked lists verifying expected ordering.

Estimated time
- Dev + tests: 1–2 days.

---

Milestone 5 — Re-Ranking pipeline (retrieve many, rerank with cross-encoder) (Priority: Medium, Effort: Medium)
Goal
- Retrieve a larger candidate set (e.g., 50), then rerank with a cross-encoder (or FlashRank) for better final ordering.

Concrete steps
1. Ensure retrieval returns N=50 candidates when reranker enabled (configurable via `settings.rerank_candidate_count`).
2. Implement an adapter for a cross-encoder reranker (if FlashRank is not adequate or to support BGE): create `rag/rerankers/cross_encoder_reranker.py` with `rerank(query, candidates)`.
3. Use existing pattern in `rag/retriever.py` to run reranker in executor (non-blocking) and blend scores using `flashrank_blend_weight` or `reranker_blend_weight`.
4. Prewarm: use `scripts/flashrank_prewarm_worker.py` pattern to pre-initialize the reranker during startup.

Files to modify
- `rag/retriever.py` (retrieve larger set and call reranker)
- `rag/rerankers/` (add cross-encoder wrapper)
- `scripts/` (add prewarm script if needed)

Testing and validation
- End-to-end tests measuring reranker latency and correctness. Compare IR metrics (MRR, MAP) on a validation set.

Estimated time
- Dev + tests: 3–5 days.

---

Milestone 6 — Metadata pre-filter helper & API hooks (Priority: Low-Medium, Effort: Small)
Goal
- Allow filter-by-metadata (year, category, source) before vector search to narrow search space.

Concrete steps
1. Add `core/graph_db.py::get_document_ids_by_metadata(filters: Dict)` that constructs parameterized Cypher and returns matching document IDs.
2. Add a `metadata_filters` optional field to `ChatRequest` in `api/models.py` or accept filters via a separate parameter (careful with API compatibility).
3. In `rag/nodes/retrieval.py`, if metadata filters provided, call `get_document_ids_by_metadata()` and pass `allowed_document_ids` into `document_retriever.retrieve(...)`.

Files to modify
- `core/graph_db.py` (helper), `rag/nodes/retrieval.py`, `api/models.py` (optional)

Testing and validation
- Unit tests for the Cypher query construction and integration tests that verify narrowing behavior.

Estimated time
- Dev + tests: 1–2 days.

---

Milestone 7 — Testing, CI, and Benchmarks (Priority: High for quality, Effort: Medium)
Goal
- Add unit/integration tests for caches, streaming, RRF, reranking and a small benchmark suite to measure TTFT, median and p90 latencies.

Concrete steps
1. Add tests under `tests/`:
   - `tests/test_response_cache.py` (unit tests for cache keys and invalidation)
   - `tests/test_rrf.py` (unit tests)
   - `tests/test_streaming_sse.py` (integration test using mocked streaming LLM)
   - `tests/integration_latency.py` (benchmark script to run representative queries)
2. Add `Makefile` targets and `scripts/bench_latency.sh` to run benchmarks and collect results.
3. Add CI workflow (if not present) to run unit tests and benchmarks for PRs.

Files to modify/add
- `tests/` (new tests), `scripts/benchmark_*`, CI config (e.g., `.github/workflows/ci.yml`).

Estimated time
- Tests & CI: 2–4 days.

---

Milestone 8 — Metrics, Monitoring & Rollout (Priority: High for production safety, Effort: Small-Medium)
Goal
- Track cache hit rates, streaming token latencies, reranker latencies, and overall query latency; provide feature flags for gradual rollout and safe rollback.

Concrete steps
1. Extend `core/cache_metrics.py` to include response cache metrics and streaming metrics counters (first-token latency histogram, token count per response).
2. Expose metrics via `/api/database/cache-stats` or a new `/api/metrics` endpoint. Consider Prometheus integration (exporter) if needed.
3. Feature flags: `settings.enable_response_caching`, `settings.enable_llm_streaming`, `settings.enable_chunk_fulltext`, `settings.enable_rrf`, `settings.rerank_enabled`.
4. Rollout plan:
   - Enable response cache on a small TTL and sample rate (e.g., 10% of requests), validate correctness.
   - Enable LLM streaming behind flag; start with internal users and validate connection stability.
   - Enable BM25 + RRF in staging; run A/B comparisons.

Estimated time
- Metrics & feature flags: 2–3 days.

---

Milestone 9 — Documentation and Handoff (Priority: Medium, Effort: Small)
Goal
- Update `docs/` with design notes for each change, runbooks for cache invalidation, and commands for prewarming rerankers.

Concrete steps
1. Add/update files in `docs/`:
   - `docs/OPTIMIZATION_FINDINGS.md` (already created)
   - `docs/ROADMAP_OPTIMIZATIONS.md` (this file)
   - `docs/IMPLEMENTATION_NOTES.md` (after implementation, per milestone)
2. Add developer runbook with commands to clear caches, prewarm rerankers, and enable feature flags in `.env`.

Estimated time
- Docs: 1–2 days (concurrent with implementation).

---

Risks and mitigations
- Streaming complexity: provider differences and SSE resource usage. Mitigate with feature flags and staged rollout.
- Cache correctness: stale responses after data changes. Mitigate by invalidating response cache on document ingestion, reindex, or explicit admin trigger.
- Reranker resource usage: reranking 50+ candidates can be CPU/GPU intensive. Mitigate by capping reranked candidates, running reranker in a worker pool, and prewarming models.

Ownership and estimated timeline (example staffed by 1-2 engineers)
- Week 1: Milestone 1 (response cache) + metrics scaffolding + tests.
- Week 2–3: Milestone 2 (LLM streaming) — implement streaming for OpenAI + integrate SSE path; heavy testing.
- Week 4: Milestone 3 & 4 (chunk fulltext + RRF) + add small integration tests.
- Week 5: Milestone 5 (reranking pipeline improvements) + prewarm worker.
- Week 6: Metadata pre-filter, docs, polish, and rollout to production with feature gating.

If you want me to start implementing now
- I can implement Milestone 1 (response-level cache) immediately (small change, low risk). I will:
  - Add `get_response_cache()` + `hash_response_params` in `core/singletons.py`.
  - Add cache checks in `rag/graph_rag.py::query()`.
  - Add invalidation calls to the reindex document API endpoint.
  - Add unit tests under `tests/test_response_cache.py`.

Tell me which milestone to start with and I will create specific commits/patches and run tests locally (or in the workspace) as requested.