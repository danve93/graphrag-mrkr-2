# Singletons and Cache Management

Thread-safe singleton pattern for long-lived service instances and multi-layer caching.

## Overview

The singletons component provides thread-safe singleton instances for database connections, cache managers, and other long-lived services. It implements a unified `CacheService` that abstracts over `diskcache` (persistent) and `cachetools` (in-memory).

**Location**: `core/singletons.py`, `core/cache_service.py`
**Pattern**: Thread-safe singleton accessors
**Cache Backend**: `diskcache.Cache` (SQLite) or `cachetools.TTLCache`

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Singleton Accessors                       │
├──────────────────────────────────────────────────┤
│                                                   │
│  Global Functions (core/singletons.py)           │
│  ┌─────────────────────────────────────────────┐ │
│  │  get_graph_db_driver()                      │ │
│  │  get_entity_label_cache() -> CacheService   │ │
│  │  get_embedding_cache() -> CacheService      │ │
│  │  get_retrieval_cache() -> CacheService      │ │
│  │  get_response_cache() -> CacheService       │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│         Unified CacheService                      │
├──────────────────────────────────────────────────┤
│                                                   │
│  Features:                                        │
│  • Backend Agnostic (Disk vs Memory)             │
│  • Workspace/Session Namespacing                 │
│  • Automatic Metric Recording                    │
│  • Thread-safe                                   │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Singleton Implementation

### Base Singleton Class

```python
import threading
from typing import Optional

class Singleton:
    """
    Thread-safe singleton base class using double-checked locking.
    
    Usage:
        class MyService(Singleton):
            def __init__(self):
                self.value = 42
        
        instance = MyService.get_instance()
    """
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance."""
        # First check (no lock)
        if cls not in cls._instances:
            # Acquire lock for creation
            with cls._lock:
                # Second check (with lock)
                if cls not in cls._instances:
                    cls._instances[cls] = cls()
        
        return cls._instances[cls]
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        with cls._lock:
            if cls in cls._instances:
                del cls._instances[cls]
```

### GraphDB Singleton

```python
from neo4j import GraphDatabase
from config.settings import settings

class GraphDB(Singleton):
    """Singleton Neo4j database connection."""
    
    def __init__(self):
        self.driver = None
        self._initialized = False
    
    def connect(self):
        """Initialize database driver."""
        if self._initialized:
            return
        
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size
        )
        
        self.driver.verify_connectivity()
        self._initialized = True
        
        logger.info(
            f"Neo4j connected: {settings.neo4j_uri} "
            f"(pool size: {settings.neo4j_max_connection_pool_size})"
        )
    
    def close(self):
        """Close database connection."""
        if self.driver:
            self.driver.close()
            self._initialized = False

# Usage
def get_db() -> GraphDB:
    """Get singleton database instance."""
    db = GraphDB.get_instance()
    if not db._initialized:
        db.connect()
    return db
```

### Access Pattern

Instead of a monolithic `CacheManager` class, we now use specific accessor functions that lazily initialize the required service.

```python
# core/singletons.py

def get_embedding_cache() -> CacheService:
    """Get or create singleton embedding cache service."""
    global _embedding_cache
    
    if _embedding_cache is not None:
        return _embedding_cache

    with _embedding_lock:
        if _embedding_cache is not None:
            return _embedding_cache

        _embedding_cache = CacheService(
            name="embeddings",
            ttl=settings.embedding_cache_ttl,
            max_size=settings.embedding_cache_size,
            use_disk=True  # Prefer disk for embeddings
        )
        return _embedding_cache
```

### CacheService Usage

The `CacheService` provides a standard dictionary-like interface with extra features.

```python
from core.singletons import get_embedding_cache

# 1. Get the cache instance
cache = get_embedding_cache()

# 2. Put item (automatically handles serialization)
cache.set("key", value)

# 3. Get item
value = cache.get("key")

# 4. Workspace-aware caching
# Generates namespaced key: "workspace_123:my_key"
cache.set("my_key", value, workspace_id="workspace_123")
val = cache.get("my_key", workspace_id="workspace_123")
```

## Cache Layers

### 1. Entity Label Cache
-   **Accessor**: `get_entity_label_cache()`
-   **Default Backend**: Memory (or Disk if configured)
-   **TTL**: 5 minutes
-   **Use**: Maps Entity Name -> Entity Type

### 2. Embedding Cache
-   **Accessor**: `get_embedding_cache()`
-   **Default Backend**: Disk (SQLite)
-   **TTL**: 7 days (configurable)
-   **Use**: Maps Text Hash -> Vector List
-   **Note**: Persisting this saves significant OpenAI costs.

### 3. Retrieval Cache
-   **Accessor**: `get_retrieval_cache()`
-   **Default Backend**: Memory
-   **TTL**: 60 seconds
-   **Use**: Maps Query + Params -> List[Nodes]

### 4. Response Cache
-   **Accessor**: `get_response_cache()`
-   **Default Backend**: Disk
-   **TTL**: 5 minutes
-   **Use**: Maps Query Hash -> Full RAG Response

## Cache Metrics

### Metrics Tracking

```python
from collections import defaultdict
from typing import Dict

class CacheMetrics(Singleton):
    """
    Track cache hit/miss rates for monitoring.
    
    Tracks separate metrics for:
        - entity_label
        - embedding
        - retrieval
    """
    
    def __init__(self):
        self.cache_hits: Dict[str, int] = defaultdict(int)
        self.cache_misses: Dict[str, int] = defaultdict(int)
    
    def record_hit(self, cache_name: str):
        """Record cache hit."""
        self.cache_hits[cache_name] += 1
    
    def record_miss(self, cache_name: str):
        """Record cache miss."""
        self.cache_misses[cache_name] += 1
    
    def get_hit_rate(self, cache_name: str) -> float:
        """Calculate hit rate for a cache."""
        hits = self.cache_hits.get(cache_name, 0)
        misses = self.cache_misses.get(cache_name, 0)
        
        total = hits + misses
        if total == 0:
            return 0.0
        
        return hits / total
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all caches."""
        return {
            cache_name: {
                "hits": self.cache_hits[cache_name],
                "misses": self.cache_misses[cache_name],
                "hit_rate": self.get_hit_rate(cache_name)
            }
            for cache_name in set(
                list(self.cache_hits.keys()) + list(self.cache_misses.keys())
            )
        }
    
    def reset(self):
        """Reset all metrics."""
        self.cache_hits.clear()
        self.cache_misses.clear()

# Global instance
cache_metrics = CacheMetrics.get_instance()
```

### API Endpoint for Metrics

```python
# api/routers/database.py
from fastapi import APIRouter
from core.singletons import cache_metrics

router = APIRouter(prefix="/api/database", tags=["database"])

@router.get("/cache-stats")
def get_cache_stats():
    """Get cache statistics and metrics."""
    metrics = cache_metrics.get_all_stats()
    
    return {
        "cache_metrics": metrics
    }
```

## Cache Invalidation

### Manual Clearing

```python
from core.singletons import get_response_cache

# Clear specific cache
get_response_cache().clear()
```

### Response Cache Invalidation

The response cache often needs invalidation when underlying documents change.

```python
from core.singletons import clear_response_cache

# Safely clears and re-initializes the cache
clear_response_cache()
```

## Related Documentation

-   [Caching Settings](../../07-configuration/caching-settings.md)
-   [Cache Metrics](cache-metrics.md)
