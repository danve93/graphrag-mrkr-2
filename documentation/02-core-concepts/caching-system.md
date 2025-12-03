# Caching System

Multi-layer caching architecture for performance optimization.

## Overview

Amber implements a comprehensive caching system inspired by TrustGraph's performance patterns. The multi-layer cache provides 30-50% latency reduction for repeated queries by eliminating redundant:
- Database lookups (entity labels)
- API calls (embeddings)
- Graph queries (retrieval results)
- LLM responses (full answers)

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
│ Cache  │  │         │  │          │  │          │
│ (TTL)  │  │  (LRU)  │  │  (TTL)   │  │  (TTL)   │
└────┬───┘  └────┬────┘  └─────┬────┘  └─────┬────┘
     │           │             │              │
     └───────────┴─────────────┴──────────────┘
                 │
                 ▼
         ┌───────────────┐
         │     Neo4j     │
         │   Database    │
         └───────────────┘
```

## Cache Layers

### 1. Entity Label Cache

**Purpose**: Cache entity name → label lookups

**Implementation**: TTLCache (Time-To-Live)

**Location**: `core/singletons.py::EntityLabelCache`

**Configuration**:
```bash
ENTITY_LABEL_CACHE_SIZE=5000      # Maximum entries
ENTITY_LABEL_CACHE_TTL=300        # Time-to-live (seconds)
```

**Usage**:
```python
from core.graph_db import get_entity_label_cached

# First call: Database query
label = await get_entity_label_cached("vCenter")  # ~10ms

# Subsequent calls: Cache hit
label = await get_entity_label_cached("vCenter")  # ~0.1ms
```

**Cache Key**: Entity name (string)

**Cache Value**: Entity label (string)

**Invalidation**: TTL-based (5 minutes)

**Hit Rate**: 70-80% typical

**Use Cases**:
- Multi-hop graph expansion
- Entity relationship queries
- Graph reasoning node

**Performance Impact**:
- Without cache: 5-10ms per lookup
- With cache: <1ms per lookup
- Latency reduction: 90-95%

### 2. Embedding Cache

**Purpose**: Cache text → embedding vector

**Implementation**: LRUCache (Least Recently Used)

**Location**: `core/singletons.py::EmbeddingCache`

**Configuration**:
```bash
EMBEDDING_CACHE_SIZE=10000        # Maximum entries
```

**Usage**:
```python
from core.embeddings import embedding_manager

# First call: API request
embedding = await embedding_manager.get_embedding(text)  # ~100-300ms

# Subsequent calls: Cache hit
embedding = await embedding_manager.get_embedding(text)  # ~0.1ms
```

**Cache Key**: Hash of (text + model name)
```python
cache_key = hashlib.sha256(f"{text}:{model}".encode()).hexdigest()
```

**Cache Value**: Embedding vector (List[float])

**Invalidation**: LRU eviction (no TTL)

**Hit Rate**: 40-60% typical

**Use Cases**:
- Query embedding
- Chunk embedding (reprocessing)
- Entity embedding
- Duplicate text prevention

**Performance Impact**:
- Without cache: 100-300ms per embedding
- With cache: <1ms per embedding
- Cost savings: API call avoided

### 3. Retrieval Cache

**Purpose**: Cache query → retrieval results

**Implementation**: TTLCache

**Location**: `core/singletons.py::RetrievalCache`

**Configuration**:
```bash
RETRIEVAL_CACHE_SIZE=1000         # Maximum entries
RETRIEVAL_CACHE_TTL=60            # Time-to-live (seconds)
```

**Usage**:
```python
from rag.retriever import hybrid_retrieval

# First call: Database queries + graph expansion
results = await hybrid_retrieval(query, top_k=10)  # ~200-500ms

# Subsequent calls (within TTL): Cache hit
results = await hybrid_retrieval(query, top_k=10)  # ~1ms
```

**Cache Key**: Hash of parameters
```python
cache_key = hashlib.sha256(json.dumps({
    "query": query,
    "top_k": top_k,
    "context_documents": context_documents,
    "retrieval_mode": retrieval_mode,
    "max_expansion_depth": max_expansion_depth
}, sort_keys=True).encode()).hexdigest()
```

**Cache Value**: List of retrieved chunks with scores

**Invalidation**: 
- TTL-based (1 minute)
- Manual invalidation on document ingestion

**Hit Rate**: 20-30% typical

**Use Cases**:
- Repeated queries
- Query variations with same parameters
- Development/testing

**Performance Impact**:
- Without cache: 200-500ms retrieval
- With cache: <5ms retrieval
- Latency reduction: 95-99%

**TTL Rationale**: Short TTL (60s) ensures consistency with document updates while still providing benefit for rapid repeated queries.

### 4. Response Cache

**Purpose**: Cache complete LLM responses

**Implementation**: TTLCache

**Location**: `core/singletons.py::ResponseCache`

**Configuration**:
```bash
RESPONSE_CACHE_SIZE=2000          # Maximum entries
RESPONSE_CACHE_TTL=300            # Time-to-live (seconds)
```

**Usage**:
```python
from core.cache import get_cached_response, cache_response

# Check cache
cached = get_cached_response(query, context_hash)
if cached:
    return cached

# Generate and cache
response = await llm.generate(prompt)
cache_response(query, context_hash, response)
```

**Cache Key**: Hash of (query + context chunks + parameters)
```python
cache_key = hashlib.sha256(json.dumps({
    "query": query,
    "context_hash": hash(tuple(chunk_ids)),
    "temperature": temperature,
    "model": model
}, sort_keys=True).encode()).hexdigest()
```

**Cache Value**: Complete response with sources and metadata

**Invalidation**: TTL-based (5 minutes)

**Hit Rate**: 10-20% typical (varies by query diversity)

**Use Cases**:
- Identical queries (rare)
- Development/testing
- Demo environments

**Performance Impact**:
- Without cache: 2-5 seconds generation
- With cache: <10ms response
- Cost savings: LLM API call avoided

**Feature Flag**: `ENABLE_RESPONSE_CACHING` (default: false in production)

## Singleton Manager

**Purpose**: Thread-safe singleton instances for long-lived services

**Implementation**: `core/singletons.py`

```python
from threading import Lock
from typing import Optional
from cachetools import TTLCache, LRUCache

class Singleton:
    _instances = {}
    _lock = Lock()
    
    @classmethod
    def get_instance(cls):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = cls()
        return cls._instances[cls]

class CacheManager(Singleton):
    def __init__(self):
        self.entity_label_cache = TTLCache(
            maxsize=settings.entity_label_cache_size,
            ttl=settings.entity_label_cache_ttl
        )
        self.embedding_cache = LRUCache(
            maxsize=settings.embedding_cache_size
        )
        self.retrieval_cache = TTLCache(
            maxsize=settings.retrieval_cache_size,
            ttl=settings.retrieval_cache_ttl
        )
        self.response_cache = TTLCache(
            maxsize=settings.response_cache_size,
            ttl=settings.response_cache_ttl
        )
```

**Benefits**:
- Single cache instance per process
- Thread-safe initialization
- Memory efficiency
- Shared across requests

## Cache Monitoring

### Metrics Collection

**Implementation**: `core/cache_metrics.py`

```python
class CacheMetrics:
    def __init__(self):
        self.hits = defaultdict(int)
        self.misses = defaultdict(int)
    
    def record_hit(self, cache_name: str):
        self.hits[cache_name] += 1
    
    def record_miss(self, cache_name: str):
        self.misses[cache_name] += 1
    
    def get_stats(self) -> Dict[str, Dict]:
        stats = {}
        for cache_name in set(list(self.hits.keys()) + list(self.misses.keys())):
            hits = self.hits[cache_name]
            misses = self.misses[cache_name]
            total = hits + misses
            hit_rate = hits / total if total > 0 else 0
            
            stats[cache_name] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hit_rate
            }
        return stats
```

### API Endpoint

**Endpoint**: `GET /api/database/cache-stats`

**Response**:
```json
{
  "entity_label_cache": {
    "hits": 1523,
    "misses": 378,
    "total": 1901,
    "hit_rate": 0.801,
    "size": 1234,
    "max_size": 5000
  },
  "embedding_cache": {
    "hits": 892,
    "misses": 1456,
    "total": 2348,
    "hit_rate": 0.380,
    "size": 2103,
    "max_size": 10000
  },
  "retrieval_cache": {
    "hits": 45,
    "misses": 167,
    "total": 212,
    "hit_rate": 0.212,
    "size": 123,
    "max_size": 1000
  },
  "response_cache": {
    "hits": 12,
    "misses": 89,
    "total": 101,
    "hit_rate": 0.119,
    "size": 67,
    "max_size": 2000
  }
}
```

### Monitoring Dashboard

Access cache statistics:
```bash
curl http://localhost:8000/api/database/cache-stats | jq
```

**Key Metrics**:
- **Hit Rate**: Percentage of cache hits (target: >50% for entity/embedding)
- **Size**: Current cache entries
- **Eviction Rate**: How often LRU/TTL evictions occur

## Cache Invalidation

### Manual Invalidation

**Clear all caches**:
```python
from core.singletons import cache_manager

cache_manager.clear_all()
```

**Clear specific cache**:
```python
cache_manager.entity_label_cache.clear()
cache_manager.embedding_cache.clear()
cache_manager.retrieval_cache.clear()
cache_manager.response_cache.clear()
```

### Automatic Invalidation

**On document ingestion**:
```python
# After document processed
cache_manager.retrieval_cache.clear()
cache_manager.response_cache.clear()
```

**TTL expiration**:
- Entity label cache: 5 minutes
- Retrieval cache: 1 minute
- Response cache: 5 minutes

**LRU eviction**:
- Embedding cache: Automatic when full

### Invalidation API

**Endpoint**: `POST /api/database/clear-cache`

**Request**:
```json
{
  "cache_name": "retrieval"  // or "all"
}
```

**Response**:
```json
{
  "status": "success",
  "cache_name": "retrieval",
  "cleared_entries": 123
}
```

## Configuration

### Master Switch

```bash
ENABLE_CACHING=true  # Enable all caches
```

When disabled, all cache operations become no-ops.

### Individual Cache Settings

```bash
# Entity Label Cache
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300

# Embedding Cache
EMBEDDING_CACHE_SIZE=10000

# Retrieval Cache
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60

# Response Cache
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300
```

### Neo4j Connection Pool

```bash
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

Larger pool supports higher concurrency but uses more memory.

## Performance Impact

### Latency Reduction

**Cold Query** (no cache):
```
Query Analysis:     10ms
Retrieval:          200ms
  ├─ Entity Labels: 50ms (5 lookups × 10ms)
  ├─ Vector Search: 100ms
  └─ Expansion:     50ms
Generation:         2000ms
─────────────────────────
Total:              2210ms
```

**Warm Query** (cache hit):
```
Query Analysis:     10ms
Retrieval:          100ms (cached)
  ├─ Entity Labels: 0.5ms (5 lookups × 0.1ms)
  ├─ Vector Search: 100ms
  └─ Expansion:     Not needed (retrieval cached)
Generation:         2000ms
─────────────────────────
Total:              2110ms (4.5% improvement)
```

**Hot Query** (full cache):
```
Query Analysis:     10ms
Retrieval:          5ms (cached)
Generation:         10ms (cached)
─────────────────────────
Total:              25ms (98.9% improvement)
```

### Cost Reduction

**Embedding API Calls**:
- Cost per call: ~$0.0001 (text-embedding-3-small)
- Cache hit rate: 40-60%
- Monthly queries: 100,000
- Savings: $4-6/month per 100K queries

**LLM API Calls** (with response cache):
- Cost per call: ~$0.01 (gpt-4o-mini)
- Cache hit rate: 10-20%
- Monthly queries: 10,000
- Savings: $10-20/month per 10K queries

## Best Practices

### Cache Warming

Pre-populate caches for common queries:
```python
# Warm entity label cache
common_entities = ["vCenter", "ESXi", "NSX", "vSAN"]
for entity in common_entities:
    await get_entity_label_cached(entity)

# Warm embedding cache
common_queries = load_common_queries()
for query in common_queries:
    await embedding_manager.get_embedding(query)
```

### Cache Sizing

**Entity Label Cache**:
- Small documents: 1000-2000
- Medium documents: 3000-5000
- Large documents: 5000-10000

**Embedding Cache**:
- Development: 1000-5000
- Production: 10000-50000

**Retrieval Cache**:
- Low diversity: 500-1000
- High diversity: 1000-5000

**Response Cache**:
- Demo/testing: 1000-5000
- Production: Disable or 500-1000

### Monitoring

**Check hit rates regularly**:
```bash
watch -n 5 'curl -s http://localhost:8000/api/database/cache-stats | jq ".entity_label_cache.hit_rate"'
```

**Alert on low hit rates**:
- Entity label cache: <50%
- Embedding cache: <30%
- Retrieval cache: <15%

**Increase cache sizes** if eviction rate is high.

### Testing

**Disable caching for tests**:
```bash
ENABLE_CACHING=false pytest tests/
```

**Test cache invalidation**:
```python
def test_cache_invalidation():
    # Populate cache
    result1 = get_entity_label_cached("test")
    
    # Invalidate
    cache_manager.entity_label_cache.clear()
    
    # Verify cache miss
    result2 = get_entity_label_cached("test")
    assert result2 is not result1  # Fresh lookup
```

## Troubleshooting

### High Memory Usage

**Check cache sizes**:
```python
from core.singletons import cache_manager
import sys

print(f"Entity cache: {sys.getsizeof(cache_manager.entity_label_cache)} bytes")
print(f"Embedding cache: {sys.getsizeof(cache_manager.embedding_cache)} bytes")
```

**Reduce cache sizes**:
```bash
ENTITY_LABEL_CACHE_SIZE=2000
EMBEDDING_CACHE_SIZE=5000
```

### Low Hit Rates

**Causes**:
- Query diversity too high
- Cache sizes too small
- TTL too short

**Solutions**:
- Increase cache sizes
- Increase TTL (if data consistency allows)
- Analyze query patterns

### Stale Data

**Symptoms**: Outdated responses after document updates

**Solutions**:
- Reduce TTL
- Clear caches on ingestion
- Disable response cache

## Related Documentation

- [Performance Optimization](../../docs/PERFORMANCE_OPTIMIZATION.md)
- [Singleton Manager](03-components/backend/singletons.md)
- [Graph Database](03-components/backend/graph-database.md)
- [Embeddings Component](03-components/backend/embeddings.md)
