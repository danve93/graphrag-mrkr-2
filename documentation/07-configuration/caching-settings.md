# Caching Settings

Performance tuning for Amber's multi-layer, unified caching system.

## Overview

Amber implements a robust caching architecture using a unified `CacheService` that supports both in-memory and disk-based persistence. This system significantly reduces latency and API costs for repeated operations:

1.  **Entity Label Cache** - Persists entity name → label lookups (GraphDB offload).
2.  **Embedding Cache** - Persists text → embedding vectors (OpenAI API offload).
3.  **Retrieval Cache** - Caches hybrid retrieval results for identical queries.
4.  **Response Cache** - Caches full RAG responses for identical queries.

## Architecture

The system uses `diskcache` for high-performance, persistent caching and `cachetools` for purely in-memory operations if configured.

```
┌─────────────────────────────────────────────┐
│          Request Flow                       │
├─────────────────────────────────────────────┤
│  Query → Retrieval Cache (Disk/Mem)         │
│           ↓ miss                            │
│         Vector Search                       │
│           ↓                                 │
│         Entity Label Cache (Disk/Mem)       │
│           ↓ miss                            │
│         Neo4j Lookup                        │
│           ↓                                 │
│         Embedding Cache (Disk/Mem)          │
│           ↓ miss                            │
│         OpenAI API                          │
│           ↓                                 │
│         Response Cache (Disk/Mem)           │
└─────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

```bash
# Master toggle
ENABLE_CACHING=true

# Cache Backend Selection: 'disk' (persistent) or 'memory' (volatile)
CACHE_TYPE=disk

# Entity Label Cache
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300

# Embedding Cache
# (Recommended: Disk cache for persistence across restarts)
EMBEDDING_CACHE_SIZE=10000
EMBEDDING_CACHE_TTL=604800  # 7 days (default)

# Retrieval Cache
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60

# Response Cache
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300

# Document detail caches
DOCUMENT_SUMMARY_TTL=300
DOCUMENT_DETAIL_CACHE_TTL=60

# Neo4j connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### UI Configuration

Cache settings can be adjusted at runtime via the **Chat Tuning** panel under the **Performance & Caching** category.

-   **Enable Caching**: Master switch.
-   **Cache Backend**: Switch between `disk` (persistent) and `memory` (volatile).
-   **TTLs**: Adjust time-to-live for embeddings and responses.

Changes made in the UI take effect immediately for the running instance.

---

## Cache Details

### 1. Unified `CacheService`

All caches are instances of `Core.CacheService`, which handles:
-   **Backend abstraction**: Seamlessly switches between `diskcache.Cache` and `cachetools.TTLCache`.
-   **Workspace Isolation**: Automatically namespaces keys by workspace/session if required.
-   **Metrics**: Automatically records hit/miss stats.

### 2. Embedding Cache (Critical)

**Purpose**: Avoid duplicate embedding API calls. This is the most valuable cache to persist.

-   **Type**: Disk-based (SQLite via `diskcache`)
-   **Policy**: LRU (Least Recently Used) automatic eviction.
-   **Persistence**: Survives container restarts.

**Tuning**:
-   **TTL**: Set high (e.g., 7 days or 30 days) as embeddings are expensive and immutable.
-   **Size**: Increase to cover your entire document corpus size.

### 3. Response Cache

**Purpose**: Instant answers for repeated questions.

-   **Type**: Disk or Memory (configurable).
-   **Key**: Hash of (Query + Chat History + RAG Parameters).
-   **Behavior**:
    -   Exact matches only.
    -   Includes associated metadata and source links.

---

## Monitoring

### Cache Stats API

```bash
curl http://localhost:8000/api/database/cache-stats
```

Response:

```json
{
  "cache_config": {
    "entity_labels": {
      "type": "disk",
      "size": 124,
      "hits": 450,
      "misses": 12
    },
    ...
  },
  "cache_metrics": { ... }
}
```

### Metrics Dashboard

Check the **Metrics > Routing** panel in the frontend for "Cache Hit Rate" statistics on RAG queries.

---

## Troubleshooting

### Cache Not Persisting?

Ensure `CACHE_TYPE=disk` is set in `.env` or in the UI. Disk caches are stored in `/app/data/cache` inside the container.

### High Disk Usage

If `diskcache` grows too large:
1.  Reduce `*_CACHE_SIZE` limits.
2.  Reduce `EMBEDDING_CACHE_TTL`.
3.  Manually clear cache:
    ```python
    from core.singletons import get_embedding_cache
    get_embedding_cache().clear()
    ```
