# Singletons and Cache Management

Thread-safe singleton pattern for long-lived service instances and multi-layer caching.

## Overview

The singletons component provides thread-safe singleton instances for database connections, cache managers, and other long-lived services. It implements TrustGraph-inspired multi-layer caching with TTL and LRU policies to reduce database queries and API calls by 30-50%.

**Location**: `core/singletons.py`, `core/cache_metrics.py`
**Pattern**: Thread-safe singleton with double-checked locking
**Cache Types**: TTLCache (entity labels, retrieval), LRUCache (embeddings)

## Architecture

```
┌──────────────────────────────────────────────────┐
│         Singleton Manager                         │
├──────────────────────────────────────────────────┤
│                                                   │
│  Thread-Safe Singleton Pattern                   │
│  ┌─────────────────────────────────────────────┐ │
│  │  Double-checked locking                     │ │
│  │  • Instance creation lock                   │ │
│  │  • Atomic initialization                    │ │
│  │  • Safe across threads                      │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Global Instances                                 │
│  ┌─────────────────────────────────────────────┐ │
│  │  • Neo4j driver (connection pooling)        │ │
│  │  • Cache manager (multi-layer caches)       │ │
│  │  • Cache metrics (hit/miss tracking)        │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│         Multi-Layer Cache System                  │
├──────────────────────────────────────────────────┤
│                                                   │
│  Layer 1: Entity Label Cache                     │
│  ┌─────────────────────────────────────────────┐ │
│  │  Type: TTLCache                             │ │
│  │  Size: 5000 entries                         │ │
│  │  TTL: 300 seconds (5 minutes)               │ │
│  │  Hit Rate: 70-80%                           │ │
│  │  Use: Entity type lookups                   │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 2: Embedding Cache                        │
│  ┌─────────────────────────────────────────────┐ │
│  │  Type: LRUCache                             │ │
│  │  Size: 10000 entries                        │ │
│  │  TTL: None (LRU eviction)                   │ │
│  │  Hit Rate: 40-60%                           │ │
│  │  Use: Embedding API results                 │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 3: Retrieval Cache                        │
│  ┌─────────────────────────────────────────────┐ │
│  │  Type: TTLCache                             │ │
│  │  Size: 1000 entries                         │ │
│  │  TTL: 60 seconds (1 minute)                 │ │
│  │  Hit Rate: 20-30%                           │ │
│  │  Use: Query results                         │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Singleton Pattern Implementation

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

## Cache Manager

### Multi-Layer Cache Manager

```python
from cachetools import TTLCache, LRUCache
from config.settings import settings

class CacheManager(Singleton):
    """
    Multi-layer cache manager with different policies.
    
    Layers:
        - entity_label_cache: TTL cache for entity type lookups
        - embedding_cache: LRU cache for embedding API results
        - retrieval_cache: TTL cache for query results
    """
    
    def __init__(self):
        self._initialized = False
        self.entity_label_cache = None
        self.embedding_cache = None
        self.retrieval_cache = None
    
    def initialize(self):
        """Initialize caches based on settings."""
        if self._initialized or not settings.enable_caching:
            return
        
        # Entity label cache (TTL)
        self.entity_label_cache = TTLCache(
            maxsize=settings.entity_label_cache_size,
            ttl=settings.entity_label_cache_ttl
        )
        
        # Embedding cache (LRU, no TTL)
        self.embedding_cache = LRUCache(
            maxsize=settings.embedding_cache_size
        )
        
        # Retrieval cache (TTL)
        self.retrieval_cache = TTLCache(
            maxsize=settings.retrieval_cache_size,
            ttl=settings.retrieval_cache_ttl
        )
        
        self._initialized = True
        
        logger.info(
            f"Cache manager initialized: "
            f"entity_label={settings.entity_label_cache_size}/{settings.entity_label_cache_ttl}s, "
            f"embedding={settings.embedding_cache_size}, "
            f"retrieval={settings.retrieval_cache_size}/{settings.retrieval_cache_ttl}s"
        )
    
    def clear_all(self):
        """Clear all caches."""
        if not self._initialized:
            return
        
        self.entity_label_cache.clear()
        self.embedding_cache.clear()
        self.retrieval_cache.clear()
        
        logger.info("All caches cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._initialized:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "entity_label_cache": {
                "size": len(self.entity_label_cache),
                "maxsize": self.entity_label_cache.maxsize,
                "ttl": self.entity_label_cache.ttl
            },
            "embedding_cache": {
                "size": len(self.embedding_cache),
                "maxsize": self.embedding_cache.maxsize
            },
            "retrieval_cache": {
                "size": len(self.retrieval_cache),
                "maxsize": self.retrieval_cache.maxsize,
                "ttl": self.retrieval_cache.ttl
            }
        }

# Global instance
cache_manager = CacheManager.get_instance()

# Initialize on import if caching enabled
if settings.enable_caching:
    cache_manager.initialize()
```

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
from core.singletons import cache_manager, cache_metrics

router = APIRouter(prefix="/api/database", tags=["database"])

@router.get("/cache-stats")
def get_cache_stats():
    """Get cache statistics and metrics."""
    stats = cache_manager.get_stats()
    metrics = cache_metrics.get_all_stats()
    
    return {
        "cache_config": stats,
        "cache_metrics": metrics
    }
```

## Cache Usage Examples

### Entity Label Caching

```python
from core.singletons import cache_manager, cache_metrics
from core.graph_db import get_db

async def get_entity_label_cached(entity_name: str) -> Optional[str]:
    """
    Get entity type with caching.
    
    Flow:
        1. Check cache
        2. If miss, query database
        3. Store in cache
        4. Track metrics
    """
    from config.settings import settings
    
    if not settings.enable_caching:
        return await _get_entity_label_from_db(entity_name)
    
    # Check cache
    if entity_name in cache_manager.entity_label_cache:
        cache_metrics.record_hit("entity_label")
        return cache_manager.entity_label_cache[entity_name]
    
    # Cache miss - query database
    cache_metrics.record_miss("entity_label")
    label = await _get_entity_label_from_db(entity_name)
    
    # Store in cache
    if label:
        cache_manager.entity_label_cache[entity_name] = label
    
    return label

async def _get_entity_label_from_db(entity_name: str) -> Optional[str]:
    """Query entity label from database."""
    db = get_db()
    query = """
    MATCH (e:Entity {name: $name})
    RETURN e.type as label
    LIMIT 1
    """
    
    results = await db.execute_read_async(query, {"name": entity_name})
    return results[0]["label"] if results else None
```

### Embedding Caching

```python
import hashlib
from core.singletons import cache_manager, cache_metrics

class EmbeddingManager:
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key from text and model."""
        content = f"{text}::{self.model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching."""
        if not settings.enable_caching:
            return await self._generate_embedding(text)
        
        # Generate cache key
        cache_key = self._generate_cache_key(text)
        
        # Check cache
        if cache_key in cache_manager.embedding_cache:
            cache_metrics.record_hit("embedding")
            return cache_manager.embedding_cache[cache_key]
        
        # Cache miss - generate embedding
        cache_metrics.record_miss("embedding")
        embedding = await self._generate_embedding(text)
        
        # Store in cache
        cache_manager.embedding_cache[cache_key] = embedding
        
        return embedding
```

### Retrieval Caching

```python
import hashlib
import json
from core.singletons import cache_manager, cache_metrics

def _generate_retrieval_cache_key(
    query: str,
    top_k: int,
    retrieval_mode: str,
    context_documents: List[str]
) -> str:
    """Generate cache key for retrieval."""
    data = {
        "query": query,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "context_documents": sorted(context_documents)
    }
    content = json.dumps(data, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

async def hybrid_retrieval_cached(
    query: str,
    top_k: int = 20,
    retrieval_mode: str = "hybrid",
    context_documents: List[str] = None
) -> List[Dict]:
    """Hybrid retrieval with caching."""
    if not settings.enable_caching:
        return await hybrid_retrieval(query, top_k, retrieval_mode, context_documents)
    
    # Generate cache key
    cache_key = _generate_retrieval_cache_key(
        query, top_k, retrieval_mode, context_documents or []
    )
    
    # Check cache
    if cache_key in cache_manager.retrieval_cache:
        cache_metrics.record_hit("retrieval")
        return cache_manager.retrieval_cache[cache_key]
    
    # Cache miss - perform retrieval
    cache_metrics.record_miss("retrieval")
    results = await hybrid_retrieval(
        query, top_k, retrieval_mode, context_documents
    )
    
    # Store in cache (short TTL for consistency)
    cache_manager.retrieval_cache[cache_key] = results
    
    return results
```

## Configuration

### Environment Variables

```bash
# Master cache toggle
ENABLE_CACHING=true

# Entity label cache (TTL)
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300  # 5 minutes

# Embedding cache (LRU, no TTL)
EMBEDDING_CACHE_SIZE=10000

# Retrieval cache (TTL)
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60  # 1 minute

# Neo4j connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### Settings Access

```python
from config.settings import settings

# Check if caching is enabled
if settings.enable_caching:
    # Use cached version
    label = await get_entity_label_cached(entity_name)
else:
    # Direct query
    label = await _get_entity_label_from_db(entity_name)
```

## Performance Impact

### Expected Hit Rates

Based on TrustGraph implementation:

```python
# Entity label cache
Hit rate: 70-80%
Latency reduction: Neo4j query (~10ms) → cache lookup (<1ms)
Use case: Multi-hop reasoning, graph expansion

# Embedding cache
Hit rate: 40-60%
Latency reduction: API call (~200ms) → cache lookup (<1ms)
Use case: Repeated queries, entity re-ingestion

# Retrieval cache
Hit rate: 20-30%
Latency reduction: Full retrieval (~100-500ms) → cache lookup (<1ms)
Use case: Repeated queries, pagination
```

### Overall Impact

- **Total latency reduction**: 30-50% for typical queries
- **Database load reduction**: 70-80% fewer entity lookups
- **API call reduction**: 40-60% fewer embedding requests
- **Query throughput**: 2-3x improvement with warm cache

## Cache Invalidation

### Manual Invalidation

```python
from core.singletons import cache_manager

# Clear all caches
cache_manager.clear_all()

# Clear specific cache
cache_manager.entity_label_cache.clear()
cache_manager.embedding_cache.clear()
cache_manager.retrieval_cache.clear()
```

### Automatic Invalidation

```python
# Entity label cache: TTL-based (5 minutes)
# Automatically expires stale entries

# Retrieval cache: TTL-based (1 minute)
# Short TTL ensures consistency with document updates

# Embedding cache: LRU eviction
# No TTL - embeddings don't change unless model changes
```

### Document Update Invalidation

```python
def invalidate_document_caches(document_id: str):
    """Invalidate caches after document update."""
    # Clear retrieval cache (queries may return different results)
    cache_manager.retrieval_cache.clear()
    
    # Entity label cache can remain (entity types don't change often)
    # Embedding cache can remain (text hashes ensure correctness)
```

## Testing

### Unit Tests

```python
import pytest
from core.singletons import Singleton, CacheManager, cache_metrics

def test_singleton_pattern():
    """Test thread-safe singleton."""
    instance1 = CacheManager.get_instance()
    instance2 = CacheManager.get_instance()
    
    assert instance1 is instance2

def test_cache_manager_initialization():
    """Test cache initialization."""
    manager = CacheManager.get_instance()
    manager.initialize()
    
    assert manager.entity_label_cache is not None
    assert manager.embedding_cache is not None
    assert manager.retrieval_cache is not None

def test_cache_metrics():
    """Test metrics tracking."""
    metrics = cache_metrics
    metrics.reset()
    
    # Record hits and misses
    metrics.record_hit("test_cache")
    metrics.record_hit("test_cache")
    metrics.record_miss("test_cache")
    
    # Verify metrics
    assert metrics.cache_hits["test_cache"] == 2
    assert metrics.cache_misses["test_cache"] == 1
    assert metrics.get_hit_rate("test_cache") == 2/3

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons after each test."""
    yield
    CacheManager.reset_instance()
    CacheMetrics.reset_instance()
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_entity_label_caching():
    """Test entity label cache integration."""
    from core.singletons import cache_manager, cache_metrics
    
    cache_metrics.reset()
    cache_manager.clear_all()
    
    entity_name = "OpenAI"
    
    # First call - cache miss
    label1 = await get_entity_label_cached(entity_name)
    assert cache_metrics.cache_misses["entity_label"] == 1
    assert cache_metrics.cache_hits["entity_label"] == 0
    
    # Second call - cache hit
    label2 = await get_entity_label_cached(entity_name)
    assert label1 == label2
    assert cache_metrics.cache_hits["entity_label"] == 1
```

## Monitoring and Observability

### Log Cache Performance

```python
import logging

logger = logging.getLogger(__name__)

def log_cache_stats():
    """Log cache statistics."""
    stats = cache_manager.get_stats()
    metrics = cache_metrics.get_all_stats()
    
    for cache_name, cache_metrics_data in metrics.items():
        hit_rate = cache_metrics_data["hit_rate"]
        
        if hit_rate < 0.3:
            logger.warning(
                f"Low cache hit rate for {cache_name}: {hit_rate:.2%} "
                f"(hits={cache_metrics_data['hits']}, misses={cache_metrics_data['misses']})"
            )
        else:
            logger.info(
                f"{cache_name} hit rate: {hit_rate:.2%}"
            )
```

### Periodic Monitoring

```python
import asyncio

async def monitor_caches_periodically(interval: int = 300):
    """Monitor cache performance every interval seconds."""
    while True:
        await asyncio.sleep(interval)
        log_cache_stats()
```

## Troubleshooting

### Common Issues

**Issue**: Low cache hit rate
```python
# Solution: Increase cache size
ENTITY_LABEL_CACHE_SIZE=10000
EMBEDDING_CACHE_SIZE=20000

# Or increase TTL
ENTITY_LABEL_CACHE_TTL=600  # 10 minutes
```

**Issue**: High memory usage
```python
# Solution: Reduce cache sizes
ENTITY_LABEL_CACHE_SIZE=1000
EMBEDDING_CACHE_SIZE=5000
RETRIEVAL_CACHE_SIZE=500
```

**Issue**: Stale cache results
```python
# Solution: Reduce TTL or clear caches after updates
ENTITY_LABEL_CACHE_TTL=60  # 1 minute
RETRIEVAL_CACHE_TTL=30     # 30 seconds

# Clear caches manually
cache_manager.clear_all()
```

## Related Documentation

- [Caching System](02-core-concepts/caching-system.md)
- [Performance Optimization](../../../docs/PERFORMANCE_OPTIMIZATION.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Embeddings](03-components/backend/embeddings.md)
- [Retriever](03-components/backend/retriever.md)
