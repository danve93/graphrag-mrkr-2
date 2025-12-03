# Caching Settings

Performance tuning for Amber's multi-layer caching system.

## Overview

Amber implements a multi-layer caching architecture inspired by TrustGraph, providing 30-50% latency reduction for repeated queries. The caching system includes:

1. **Entity Label Cache** - TTL cache for entity name → label lookups
2. **Embedding Cache** - LRU cache for text → embedding vectors
3. **Retrieval Cache** - TTL cache for query → retrieval results
4. **Response Cache** - TTL cache for semantic response caching

## Architecture

```
┌─────────────────────────────────────────────┐
│          Request Flow                       │
├─────────────────────────────────────────────┤
│  Query → Retrieval Cache (60s TTL)          │
│           ↓ miss                            │
│         Vector Search                       │
│           ↓                                 │
│         Entity Label Cache (300s TTL)       │
│           ↓ miss                            │
│         Neo4j Lookup                        │
│           ↓                                 │
│         Embedding Cache (LRU, no TTL)       │
│           ↓ miss                            │
│         OpenAI API                          │
│           ↓                                 │
│         Response Cache (300s TTL)           │
└─────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

```bash
# Master toggle
ENABLE_CACHING=true

# Entity label cache
ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300

# Embedding cache
EMBEDDING_CACHE_SIZE=10000

# Retrieval cache
RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60

# Response cache
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300

# Document detail caches
DOCUMENT_SUMMARY_TTL=300
DOCUMENT_DETAIL_CACHE_TTL=60

# Neo4j connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### Settings Class

```python
from config.settings import Settings

settings = Settings()

# Check caching status
if settings.enable_caching:
    print(f"Entity cache: {settings.entity_label_cache_size} entries")
    print(f"Embedding cache: {settings.embedding_cache_size} entries")
    print(f"Retrieval cache: {settings.retrieval_cache_size} entries")
```

---

## Cache Details

### Entity Label Cache

**Purpose**: Reduce Neo4j queries for entity name → label lookups during multi-hop reasoning.

**Type**: TTLCache (time-based expiration)  
**Default Size**: 5000 entries  
**Default TTL**: 300 seconds (5 minutes)

**Cache Key**: Entity name (string)  
**Cache Value**: Entity label (string)

**Hit Rate**: 70-80% in typical workloads

```python
# Implementation (simplified)
from cachetools import TTLCache

entity_label_cache = TTLCache(
    maxsize=5000,
    ttl=300
)

def get_entity_label_cached(name: str) -> str:
    if name in entity_label_cache:
        return entity_label_cache[name]
    
    # Cache miss - query Neo4j
    label = query_neo4j_for_label(name)
    entity_label_cache[name] = label
    return label
```

**Tuning**:
- Increase size for large knowledge bases (10,000+ entities)
- Increase TTL for stable datasets (600s)
- Decrease TTL for frequently updated data (60s)

### Embedding Cache

**Purpose**: Avoid duplicate embedding API calls for repeated text.

**Type**: LRUCache (no TTL, evicts least recently used)  
**Default Size**: 10,000 entries  
**No TTL**: Embeddings are deterministic and don't expire

**Cache Key**: Hash of (text + model_name)  
**Cache Value**: Embedding vector (List[float])

**Hit Rate**: 40-60% (depends on query diversity)

```python
# Implementation (simplified)
from cachetools import LRUCache
import hashlib

embedding_cache = LRUCache(maxsize=10000)

def get_embedding(text: str, model: str) -> List[float]:
    cache_key = hashlib.md5(f"{text}:{model}".encode()).hexdigest()
    
    if cache_key in embedding_cache:
        return embedding_cache[cache_key]
    
    # Cache miss - call OpenAI
    embedding = openai.embeddings.create(
        input=text,
        model=model
    ).data[0].embedding
    
    embedding_cache[cache_key] = embedding
    return embedding
```

**Tuning**:
- Increase size for diverse query patterns (20,000+)
- Monitor memory usage (~4KB per embedding)
- No TTL needed (embeddings are immutable)

### Retrieval Cache

**Purpose**: Cache hybrid retrieval results for recent queries.

**Type**: TTLCache  
**Default Size**: 1000 entries  
**Default TTL**: 60 seconds

**Cache Key**: Hash of (query + parameters)  
**Cache Value**: List of retrieved chunks

**Hit Rate**: 20-30% (higher for repeated queries)

```python
# Implementation (simplified)
from cachetools import TTLCache
import hashlib
import json

retrieval_cache = TTLCache(
    maxsize=1000,
    ttl=60
)

def hybrid_retrieval(query: str, **params) -> List[Chunk]:
    cache_key = hashlib.md5(
        f"{query}:{json.dumps(params)}".encode()
    ).hexdigest()
    
    if cache_key in retrieval_cache:
        return retrieval_cache[cache_key]
    
    # Cache miss - perform retrieval
    results = perform_hybrid_retrieval(query, **params)
    retrieval_cache[cache_key] = results
    return results
```

**Tuning**:
- Short TTL (60s) ensures fresh results after document updates
- Increase size for high query volume (2000+)
- Decrease TTL for rapidly changing knowledge base (30s)

### Response Cache

**Purpose**: Cache complete LLM responses for identical queries.

**Type**: TTLCache  
**Default Size**: 2000 entries  
**Default TTL**: 300 seconds

**Cache Key**: Hash of (query + context + parameters)  
**Cache Value**: Complete response object

**Hit Rate**: 10-20% (exact query matches only)

```python
# Implementation (simplified)
response_cache = TTLCache(
    maxsize=2000,
    ttl=300
)

def generate_response(query: str, context: List[Chunk], **params):
    cache_key = hashlib.md5(
        f"{query}:{context_hash}:{json.dumps(params)}".encode()
    ).hexdigest()
    
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    # Cache miss - generate response
    response = llm.generate(query, context, **params)
    response_cache[cache_key] = response
    return response
```

**Tuning**:
- Increase TTL for stable answers (600s)
- Increase size for FAQ-style workloads (5000+)
- Disable if responses must be fresh (`RESPONSE_CACHE_SIZE=0`)

---

## Performance Impact

### Latency Reduction

**Entity Label Cache**:
- Miss: 10-20ms (Neo4j query)
- Hit: <1ms
- Impact: 30-40% reduction in multi-hop reasoning time

**Embedding Cache**:
- Miss: 100-200ms (OpenAI API)
- Hit: <1ms
- Impact: 50-70% reduction in embedding generation time

**Retrieval Cache**:
- Miss: 200-500ms (full retrieval pipeline)
- Hit: <1ms
- Impact: 40-50% reduction for repeated queries

**Response Cache**:
- Miss: 2-5s (LLM generation)
- Hit: <1ms
- Impact: 90%+ reduction for exact query matches

### Memory Usage

**Entity Label Cache**: ~1MB for 5000 entries (200 bytes per entry)

**Embedding Cache**: ~40MB for 10,000 entries (4KB per embedding)

**Retrieval Cache**: ~50MB for 1000 entries (50KB per result set)

**Response Cache**: ~100MB for 2000 entries (50KB per response)

**Total**: ~200MB for default configuration

---

## Monitoring

### Cache Stats API

```bash
curl http://localhost:8000/api/database/cache-stats
```

Response:
```json
{
  "entity_label_cache": {
    "size": 1247,
    "max_size": 5000,
    "hits": 8934,
    "misses": 1247,
    "hit_rate": 0.877,
    "ttl_seconds": 300
  },
  "embedding_cache": {
    "size": 7823,
    "max_size": 10000,
    "hits": 15234,
    "misses": 8456,
    "hit_rate": 0.643
  },
  "retrieval_cache": {
    "size": 234,
    "max_size": 1000,
    "hits": 456,
    "misses": 1567,
    "hit_rate": 0.225,
    "ttl_seconds": 60
  },
  "total_memory_mb": 142.3
}
```

### Metrics Module

```python
from core.cache_metrics import CacheMetrics

metrics = CacheMetrics()

# Record cache hit/miss
metrics.record_hit("entity_label")
metrics.record_miss("embedding")

# Get statistics
stats = metrics.get_stats()
print(f"Entity cache hit rate: {stats['entity_label']['hit_rate']:.2%}")
```

---

## Tuning Guidelines

### High Volume (1000+ queries/hour)

```bash
ENTITY_LABEL_CACHE_SIZE=10000
ENTITY_LABEL_CACHE_TTL=600
EMBEDDING_CACHE_SIZE=20000
RETRIEVAL_CACHE_SIZE=2000
RETRIEVAL_CACHE_TTL=90
RESPONSE_CACHE_SIZE=5000
RESPONSE_CACHE_TTL=600
NEO4J_MAX_CONNECTION_POOL_SIZE=100
```

### Low Memory Environment

```bash
ENTITY_LABEL_CACHE_SIZE=1000
EMBEDDING_CACHE_SIZE=2000
RETRIEVAL_CACHE_SIZE=200
RESPONSE_CACHE_SIZE=500
```

### Rapidly Changing Data

```bash
ENTITY_LABEL_CACHE_TTL=60
RETRIEVAL_CACHE_TTL=30
RESPONSE_CACHE_TTL=60
```

### FAQ/Support Workload (many repeated queries)

```bash
RETRIEVAL_CACHE_SIZE=5000
RETRIEVAL_CACHE_TTL=300
RESPONSE_CACHE_SIZE=10000
RESPONSE_CACHE_TTL=1800
```

---

## Cache Invalidation

### Manual Cache Clear

```python
from core.singletons import SingletonManager

manager = SingletonManager()

# Clear specific cache
manager.entity_label_cache.clear()
manager.embedding_cache.clear()
manager.retrieval_cache.clear()

# Reset all singletons (nuclear option)
manager.reset()
```

### Automatic Invalidation

Caches automatically invalidate on:
- **TTL expiration** (entity_label, retrieval, response)
- **LRU eviction** (embedding)
- **Document deletion** (all caches cleared)
- **Reindexing** (all caches cleared)

---

## Troubleshooting

### Low Hit Rates

**Entity Label Cache < 50%**:
- Increase cache size
- Increase TTL
- Check for entity naming inconsistencies

**Embedding Cache < 30%**:
- Increase cache size
- Check query diversity (high diversity = low hit rate is normal)

**Retrieval Cache < 10%**:
- Increase TTL
- Check for parameter variation in queries

### High Memory Usage

```bash
# Check current usage
curl http://localhost:8000/api/database/cache-stats | jq '.total_memory_mb'

# Reduce cache sizes
ENTITY_LABEL_CACHE_SIZE=2000
EMBEDDING_CACHE_SIZE=5000
RETRIEVAL_CACHE_SIZE=500
RESPONSE_CACHE_SIZE=1000
```

### Stale Results

```bash
# Reduce TTLs
ENTITY_LABEL_CACHE_TTL=60
RETRIEVAL_CACHE_TTL=30
RESPONSE_CACHE_TTL=60

# Or disable specific caches
RETRIEVAL_CACHE_SIZE=0
```

---

## Rollback

Disable caching entirely if issues occur:

```bash
ENABLE_CACHING=false
```

Individual caches can be disabled by setting size to 0:

```bash
ENTITY_LABEL_CACHE_SIZE=0
EMBEDDING_CACHE_SIZE=0
RETRIEVAL_CACHE_SIZE=0
RESPONSE_CACHE_SIZE=0
```

---

## Related Documentation

- [Environment Variables](07-configuration/environment-variables.md)
- [Optimal Defaults](07-configuration/optimal-defaults.md)
- [Cache Metrics](03-components/backend/cache-metrics.md)
- [Singletons](03-components/backend/singletons.md)
- [Performance Optimization](02-core-concepts/performance.md)
