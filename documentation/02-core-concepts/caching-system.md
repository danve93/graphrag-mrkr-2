# Caching System

Multi-layer, unified caching architecture for performance optimization.

## Overview

Amber implements a robust caching system using a unified `CacheService`. This service abstracts over both **persistent disk caching** (using SQLite via `diskcache`) and **in-memory caching** (using `cachetools`).

By enabling disk persistence for expensive operations (like embeddings), the system survives container restarts, significantly reducing API costs and reprocessing time.

## Cache Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    ▼            ▼            ▼              ▼
┌────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐
│ Entity │  │Embedding│  │Retrieval │  │ Response │
│ Label  │  │  Cache  │  │  Cache   │  │  Cache   │
│ Cache  │  │(SQLite) │  │  (TTL)   │  │ (SQLite) │
└────┬───┘  └────┬────┘  └─────┬────┘  └─────┬────┘
     │           │             │              │
     └───────────┴─────────────┴──────────────┘
```

## Unified CacheService

All caches are backed by the `Core.CacheService` class which provides:

1.  **Backend Agnostic Strategy**:
    -   **Disk Mode**: Uses `diskcache` (SQLite) for persistence. Best for Embeddings and Responses.
    -   **Memory Mode**: Uses `cachetools` for ephemeral speed. Best for short-lived Retrieval results.

2.  **Workspace Isolation**:
    -   Keys can be automatically namespaced by `workspace_id` or `session_id`.

3.  **Metrics**:
    -   Automatic tracking of hits, misses, and eviction rates.

## Cache Layers

### 1. Entity Label Cache

**Purpose**: Cache entity name → label lookups.

-   **Backend**: `disk` or `memory` (default: memory)
-   **TTL**: 5 minutes
-   **Impact**: Eliminates Neo4j lookups for frequently accessed entities.

### 2. Embedding Cache (Critical)

**Purpose**: Cache text → embedding vector.

-   **Backend**: `disk` (SQLite)
-   **TTL**: 7 days (default)
-   **Impact**:
    -   **Cost**: Prevents paying for the same OpenAI embedding twice.
    -   **Performance**: <1ms lookup vs 200ms API call.
    -   **Persistence**: Survives restarts (crucial for large corpuses).

### 3. Retrieval Cache

**Purpose**: Cache query → retrieval results (nodes/chunks).

-   **Backend**: `memory` (usually sufficient)
-   **TTL**: 60 seconds
-   **Impact**: Speeds up pagination and repeated identical queries.

### 4. Response Cache

**Purpose**: Cache complete LLM responses.

-   **Backend**: `disk` (default)
-   **TTL**: 2 hours (default)
-   **Impact**: Instant answers for identical questions.

## Configuration

### Environment Variables

```bash
# Core
ENABLE_CACHING=true
CACHE_TYPE=disk  # 'disk' or 'memory'

# Embedding Cache
EMBEDDING_CACHE_SIZE=10000
EMBEDDING_CACHE_TTL=604800 # 7 days

# Response Cache
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=7200    # 2 hours
```

## Monitoring

### API Check

```bash
curl http://localhost:8000/api/database/cache-stats
```

### Response

```json
{
  "cache_config": {
    "embeddings": {
      "type": "disk",
      "directory": "/app/data/cache/embeddings",
      "size": 1500,
      "hits": 450,
      "misses": 1500
    }
  },
  "cache_metrics": { ... }
}
```

## Related Documentation

-   [Caching Settings](../07-configuration/caching-settings.md)
-   [Singletons](../03-components/backend/singletons.md)
