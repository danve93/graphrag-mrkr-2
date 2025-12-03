# Feature Flags Reference

This document provides a comprehensive reference for all feature flags available in Amber's GraphRAG system. Feature flags enable safe rollout, A/B testing, and instant rollback of optimizations without code changes.

## Table of Contents

- [Core Retrieval & Fusion](#core-retrieval--fusion)
- [Caching System](#caching-system)
- [Entity Extraction & Processing](#entity-extraction--processing)
- [Document Processing](#document-processing)
- [Quality & Monitoring](#quality--monitoring)
- [Advanced Features](#advanced-features)

---

## Core Retrieval & Fusion

### `ENABLE_RRF`
**Type:** Boolean  
**Default:** `false`  
**Location:** `config/settings.py::enable_rrf`

Enables Reciprocal Rank Fusion (RRF) to combine ranked lists from multiple retrieval modalities (vector, entity, keyword, path-based) into a robust unified ranking.

**How it works:**
- Collects separate ranked lists from chunk embedding search, entity-filtered search, BM25 keyword search, and multi-hop path traversal
- Applies RRF formula: `score[id] = sum(1 / (k + rank_i))` across all lists
- Uses configurable `RRF_K` constant (default: 60) to control rank discount

**When to enable:**
- Comparative/analytical queries benefit from RRF by balancing semantic and lexical signals
- Use with `ENABLE_CHUNK_FULLTEXT=true` for best results
- Recommended for production after A/B validation

**Related flags:**
- `RRF_K`: RRF constant controlling rank discount (higher = flatter influence curve)

**Tuning:**
- Set via Chat Tuning UI or environment variable
- Can be toggled per-query via API request

**Testing:**
- Unit tests: `tests/test_rrf.py`
- Integration: `tests/test_query_routing.py::test_rrf_integration_flag`

---

### `ENABLE_CHUNK_FULLTEXT`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_chunk_fulltext`

Enables BM25-style full-text keyword search on chunk content for exact-term matching (IDs, version numbers, legal clauses, etc.).

**How it works:**
- Creates Neo4j full-text index on chunk content
- Runs lexical (BM25) search alongside vector search
- Weighted by `KEYWORD_SEARCH_WEIGHT` (default: 0.3)
- Query routing automatically boosts keyword weight for procedural/how-to queries

**When to enable:**
- Always recommended for production
- Critical for queries with specific terms, IDs, or version strings
- Works best with `ENABLE_RRF=true` for optimal fusion

**Related flags:**
- `KEYWORD_SEARCH_WEIGHT`: Weight for keyword results in hybrid scoring (0.0-1.0)
- `ENABLE_RRF`: Fusion method for combining keyword with vector results

**Performance:**
- Adds ~50-100ms latency for keyword index query
- Minimal memory overhead (Neo4j managed index)

**Testing:**
- Integration: Query routing tests verify keyword weight adjustments

---

### `ENABLE_QUERY_EXPANSION`
**Type:** Boolean  
**Default:** `false`  
**Location:** `config/settings.py::enable_query_expansion`

Automatically expands queries with synonyms and related terms when initial retrieval returns fewer than `QUERY_EXPANSION_THRESHOLD` results.

**How it works:**
- After initial retrieval, checks result count
- If below threshold (default: 3), uses LLM to generate 3-5 expansion terms
- Re-runs retrieval with expanded query
- Merges results and deduplicates

**When to enable:**
- Sparse document collections
- Domain-specific terminology with many synonyms
- User queries with uncommon phrasing

**Trade-offs:**
- Adds LLM call latency (~200-500ms)
- May reduce precision (more false positives)
- Increases token usage

**Related flags:**
- `QUERY_EXPANSION_THRESHOLD`: Trigger expansion when results < N (default: 3)

**Testing:**
- Unit tests: `tests/test_query_routing.py::TestQueryExpansion`

---

## Caching System

### `ENABLE_CACHING`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_caching`

Master toggle for entire caching system. Disabling provides instant rollback if cache-related issues occur.

**Subsystems controlled:**
- Entity label cache (Neo4j lookups)
- Embedding cache (API call deduplication)
- Retrieval cache (query result caching)
- Response cache (full pipeline response caching)

**Performance impact when disabled:**
- 30-50% latency increase for repeated queries
- Higher API costs (no embedding deduplication)
- Increased database load

**Monitoring:**
- Cache metrics: `GET /api/database/cache-stats`
- Hit rates, miss rates, size, TTL per cache type

**Related flags:**
- `ENTITY_LABEL_CACHE_SIZE`, `ENTITY_LABEL_CACHE_TTL`
- `EMBEDDING_CACHE_SIZE`
- `RETRIEVAL_CACHE_SIZE`, `RETRIEVAL_CACHE_TTL`
- `RESPONSE_CACHE_SIZE`, `RESPONSE_CACHE_TTL`

---

### Entity Label Cache

**Environment variables:**
- `ENTITY_LABEL_CACHE_SIZE` (default: 5000)
- `ENTITY_LABEL_CACHE_TTL` (default: 300 seconds)

**Purpose:** Cache entity name → label lookups to reduce Neo4j queries during multi-hop reasoning and graph expansion.

**Hit rate:** 70-80% typical for stable entity sets

---

### Embedding Cache

**Environment variables:**
- `EMBEDDING_CACHE_SIZE` (default: 10000)

**Purpose:** Deduplicate embedding API calls for identical text+model combinations (no TTL, LRU eviction).

**Hit rate:** 40-60% typical during ingestion; higher for repeated queries

---

### Retrieval Cache

**Environment variables:**
- `RETRIEVAL_CACHE_SIZE` (default: 1000)
- `RETRIEVAL_CACHE_TTL` (default: 60 seconds)

**Purpose:** Cache hybrid retrieval results for recent queries. Short TTL maintains consistency with document updates.

**Hit rate:** 20-30% typical for user queries

**Invalidation:** Automatic on TTL expiry; manual flush on document ingestion/reindex

---

### Response Cache

**Environment variables:**
- `RESPONSE_CACHE_SIZE` (default: 2000)
- `RESPONSE_CACHE_TTL` (default: 300 seconds)

**Purpose:** Cache complete RAG pipeline responses (answer + sources + metadata) for FAQ-style repeated queries.

**Cache key includes:**
- Query text (normalized)
- Retrieval mode, top_k, weights
- Context document IDs
- Model selection

**Invalidation:** Automatic on TTL expiry; manual flush on document changes

**Testing:**
- Unit tests: `tests/test_response_cache.py`

---

## Entity Extraction & Processing

### `ENABLE_ENTITY_EXTRACTION`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_entity_extraction`

Master toggle for entity extraction during document ingestion.

**When disabled:**
- Chunks created without entity graph
- Graph reasoning and entity-based retrieval unavailable
- Significantly faster ingestion (~3x speedup)
- Use for testing or pure vector search scenarios

---

### `ENABLE_GLEANING`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_gleaning`

Enables multi-pass entity extraction with gleaning (Microsoft GraphRAG approach).

**How it works:**
- Initial extraction pass
- LLM reviews extraction and identifies missed entities
- Additional passes refine and complete entity graph
- Controlled by `MAX_GLEANINGS` (default: 1 = 2 total passes)

**Quality impact:**
- 15-25% more entities extracted
- Better relationship coverage
- Higher entity description quality

**Performance:**
- Adds 1 LLM call per gleaning pass per document
- Total: (1 + MAX_GLEANINGS) × extraction_latency

**Related flags:**
- `MAX_GLEANINGS`: Number of additional passes (0-5, default: 1)

---

### `ENABLE_PHASE2_NETWORKX`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_phase2_networkx`

Enables NetworkX intermediate graph layer for batch entity persistence (Phase 2 optimization).

**Benefits:**
- 22% reduction in duplicate entities
- Entity/relationship deduplication before Neo4j persistence
- Batch UNWIND queries reduce database round-trips
- Memory-efficient accumulation with provenance tracking

**When to disable:**
- Memory-constrained environments
- Documents with >2000 entities (exceeds `MAX_NODES_PER_DOC`)

**Related settings:**
- `NEO4J_UNWIND_BATCH_SIZE`: Entities per batch (default: 500)
- `MAX_NODES_PER_DOC`: Memory limit (default: 2000)
- `MAX_EDGES_PER_DOC`: Memory limit (default: 5000)

---

### `ENABLE_DESCRIPTION_SUMMARIZATION`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_description_summarization`

Enables LLM-based entity/relationship description summarization (Microsoft GraphRAG Phase 4).

**Benefits:**
- 50-70% reduction in description verbosity
- Improved relevance and readability
- Better embedding quality (less noise)

**How it works:**
- Accumulates descriptions during extraction
- After persistence, identifies entities/relationships with multiple mentions
- Batches summarization requests (default: 5 per LLM call)
- Replaces verbose concatenated descriptions with concise summaries

**Trigger conditions:**
- Entity/relationship has ≥ `SUMMARIZATION_MIN_MENTIONS` (default: 3)
- Description length ≥ `SUMMARIZATION_MIN_LENGTH` characters (default: 200)

**Performance:**
- Adds LLM calls post-ingestion (async, non-blocking)
- Caching enabled by default (`SUMMARIZATION_CACHE_ENABLED=true`)

---

## Document Processing

### `USE_MARKER_FOR_PDF`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::use_marker_for_pdf`

Enables Marker for high-accuracy PDF → Markdown conversion.

**Benefits:**
- Superior table extraction
- Inline math/LaTeX support
- Better layout preservation
- OCR integration for scanned PDFs

**Related flags:**
- `MARKER_USE_LLM`: Enable LLM hybrid processors (highest accuracy)
- `MARKER_FORCE_OCR`: Force OCR on all pages
- `MARKER_OUTPUT_FORMAT`: Output format (markdown/json/html/chunks)

**Performance:**
- 2-5x slower than basic PDF extraction
- GPU acceleration recommended for large batches

---

### `ENABLE_OCR`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_ocr`

Enables OCR processing for scanned documents and images.

**Related flags:**
- `OCR_QUALITY_THRESHOLD`: Quality threshold (0.0-1.0, default: 0.6)

---

## Quality & Monitoring

### `ENABLE_QUALITY_SCORING`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_quality_scoring`

Enables LLM-based quality scoring for generated answers.

**Dimensions scored:**
- Context relevance (30%)
- Answer completeness (25%)
- Factual grounding (25%)
- Coherence (10%)
- Citation quality (10%)

**Output:** Aggregate quality score (0.0-1.0) displayed in UI

**Performance:**
- Adds 1 LLM call per response (~200-300ms)
- Non-blocking (parallel with streaming)

---

### `FLASHRANK_ENABLED`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::flashrank_enabled`

Enables FlashRank cross-encoder reranking for post-retrieval candidate refinement.

**How it works:**
- Retrieves N candidates (controlled by `FLASHRANK_MAX_CANDIDATES`)
- Reranks using cross-encoder (semantic relevance scoring)
- Blends rerank scores with hybrid scores via `FLASHRANK_BLEND_WEIGHT`

**Models available:**
- `ms-marco-TinyBERT-L-2-v2` (default, fastest)
- `ms-marco-MiniLM-L-6-v2` (balanced)
- `ms-marco-MiniLM-L-12-v2` (highest quality)

**Performance:**
- TinyBERT: ~15-20ms for 50 candidates
- MiniLM-L-6: ~30-50ms
- MiniLM-L-12: ~60-100ms

**Related flags:**
- `FLASHRANK_MODEL_NAME`: Model selection
- `FLASHRANK_MAX_CANDIDATES`: Candidates to rerank (default: 100)
- `FLASHRANK_BLEND_WEIGHT`: Blend factor (0.0 = pure rerank, 1.0 = pure hybrid)
- `FLASHRANK_BATCH_SIZE`: Batch size (default: 32)

**Testing:**
- Unit tests: `tests/test_reranking.py`

---

## Advanced Features

### `ENABLE_GRAPH_CLUSTERING`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_graph_clustering`

Enables Leiden community detection to group related entities into semantic clusters.

**Benefits:**
- Visualize entity relationships with color-coded communities
- Filter entities by community in GraphView
- Optional LLM-generated community summaries

**Related flags:**
- `CLUSTERING_RESOLUTION`: Granularity (default: 1.0)
- `CLUSTERING_MIN_EDGE_WEIGHT`: Filter weak connections (default: 0.0)
- `CLUSTERING_RELATIONSHIP_TYPES`: Edge types to include (default: `["SIMILAR_TO", "RELATED_TO"]`)

**Execution:**
- Manual: `python scripts/run_clustering.py`
- Automatic: enabled during reindexing when `ENABLE_CLUSTERING=true`

---

### `ENABLE_DOCUMENT_SUMMARIES`
**Type:** Boolean  
**Default:** `true`  
**Location:** `config/settings.py::enable_document_summaries`

Precomputes document-level summaries for faster document detail page loads.

**Related flags:**
- `DOCUMENT_SUMMARY_TTL`: Cache TTL (default: 300s)
- `DOCUMENT_SUMMARY_TOP_N_COMMUNITIES`: Top communities to store (default: 10)
- `DOCUMENT_SUMMARY_TOP_N_SIMILARITIES`: Top similarities to store (default: 20)

---

## Feature Flag Rollout Guide

### Safe Rollout Pattern

1. **Local validation:**
   ```bash
   # Set flag in .env
   ENABLE_RRF=true
   
   # Run tests
   pytest tests/test_rrf.py -v
   
   # Run benchmarks
   make bench-compare PREV=baseline.json
   ```

2. **Staging deployment:**
   - Enable flag in staging environment
   - Monitor metrics at `/api/database/cache-stats`
   - Run latency benchmarks
   - Validate quality with test queries

3. **Production rollout:**
   - Enable via environment variable (requires restart)
   - OR enable via Chat Tuning UI (runtime, no restart)
   - Monitor error rates, latency p90, cache hit rates
   - Prepare instant rollback plan

4. **Rollback:**
   ```bash
   # Emergency rollback via environment
   ENABLE_RRF=false
   docker compose restart backend
   
   # OR instant rollback via Chat Tuning UI (preferred)
   ```

### A/B Testing

Use Chat Tuning UI to toggle flags per-session for A/B testing:
1. Open Chat Tuning panel
2. Toggle feature flag
3. Issue test queries
4. Compare results, latency, quality scores

### Monitoring

**Key metrics per feature:**

| Feature | Metric | Endpoint | Threshold |
|---------|--------|----------|-----------|
| Caching | Hit rate | `/api/database/cache-stats` | >60% |
| RRF | Latency delta | Benchmark suite | <50ms |
| Reranking | Latency delta | Benchmark suite | <100ms |
| Query expansion | Expansion rate | Logs | 5-10% |

---

## Environment Variable Summary

**Retrieval & Fusion:**
```bash
ENABLE_RRF=false
RRF_K=60
ENABLE_CHUNK_FULLTEXT=true
KEYWORD_SEARCH_WEIGHT=0.3
ENABLE_QUERY_EXPANSION=false
QUERY_EXPANSION_THRESHOLD=3
```

**Caching:**
```bash
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

**Entity Extraction:**
```bash
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=1
ENABLE_PHASE2_NETWORKX=true
ENABLE_DESCRIPTION_SUMMARIZATION=true
SUMMARIZATION_MIN_MENTIONS=3
SUMMARIZATION_MIN_LENGTH=200
SUMMARIZATION_BATCH_SIZE=5
SUMMARIZATION_CACHE_ENABLED=true
```

**Document Processing:**
```bash
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=true
MARKER_OUTPUT_FORMAT=markdown
ENABLE_OCR=true
OCR_QUALITY_THRESHOLD=0.6
```

**Quality & Reranking:**
```bash
ENABLE_QUALITY_SCORING=true
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-TinyBERT-L-2-v2
FLASHRANK_MAX_CANDIDATES=100
FLASHRANK_BLEND_WEIGHT=0.0
FLASHRANK_BATCH_SIZE=32
```

**Advanced:**
```bash
ENABLE_GRAPH_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.0
ENABLE_DOCUMENT_SUMMARIES=true
```

---

## Troubleshooting

### Flag not taking effect

**Symptom:** Changed flag in `.env` but behavior unchanged.

**Solutions:**
1. Restart backend: `docker compose restart backend`
2. Check Chat Tuning overrides (they take precedence)
3. Verify `.env` loaded: check logs for "Applied RAG tuning configuration overrides"
4. Use runtime tuning: toggle via Chat Tuning UI without restart

### Cache hit rate too low

**Symptom:** Cache hit rate <30% at `/api/database/cache-stats`.

**Solutions:**
1. Increase cache size: `RETRIEVAL_CACHE_SIZE=2000`
2. Increase TTL: `RETRIEVAL_CACHE_TTL=120`
3. Check query diversity (high diversity = low hit rate is expected)
4. Verify caching enabled: `ENABLE_CACHING=true`

### Performance regression after enabling flag

**Symptom:** Latency increased after enabling feature.

**Solutions:**
1. Run benchmarks to quantify: `make bench-compare PREV=baseline.json`
2. Check specific metrics: TTFT, E2E latency, breakdown
3. Adjust parameters (e.g., reduce `FLASHRANK_MAX_CANDIDATES`)
4. Rollback via Chat Tuning UI or environment variable

---

## References

- **Source Code:** `config/settings.py` (canonical flag definitions)
- **Chat Tuning UI:** Modify flags at runtime without restart
- **Benchmarks:** `tests/integration_latency.py`, `scripts/bench_latency.sh`
- **Metrics:** `GET /api/database/cache-stats`
- **Tests:** `tests/test_rrf.py`, `tests/test_reranking.py`, `tests/test_query_routing.py`

---

**Last Updated:** December 1, 2025  
**Version:** Milestone 8 (Feature Flag Wiring & Documentation)
