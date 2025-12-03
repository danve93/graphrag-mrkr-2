# Optimal Defaults

Quality vs cost tradeoff analysis and recommended configurations.

## Overview

Amber's default configuration balances quality, cost, and performance. This guide explains the tradeoffs and provides optimized presets for different use cases.

---

## Default Configuration

### Current Defaults

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_CONCURRENCY=3
LLM_CONCURRENCY=2

# Document Processing
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=1

# Retrieval
RETRIEVAL_TOP_K=10
HYBRID_CHUNK_WEIGHT=0.6
HYBRID_ENTITY_WEIGHT=0.4
EXPANSION_DEPTH=1
MAX_EXPANDED_CHUNKS=50

# Reranking
FLASHRANK_ENABLED=true
FLASHRANK_BLEND_WEIGHT=0.5
FLASHRANK_MAX_CANDIDATES=30

# Generation
TEMPERATURE=0.7
MAX_TOKENS=2000

# Caching
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=5000
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000

# Clustering
ENABLE_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.0
```

### Cost Profile (1000 queries/month)

**OpenAI Costs**:
- Embeddings (ada-002): $0.10 per 1M tokens
  - Document ingestion (50 docs, 5M tokens): ~$0.50
  - Query embeddings (1000 queries, 10K tokens): ~$0.001
- LLM (gpt-4o-mini): $0.15 per 1M input, $0.60 per 1M output
  - Entity extraction (50 docs, 2M tokens): ~$0.30
  - Query responses (1000 queries, 500K input, 200K output): ~$0.20

**Total**: ~$1.00/month for moderate usage

**Latency Profile**:
- Document ingestion: 15-60s per document
- Query TTFT: 700-1000ms
- Query streaming: 20-50 tokens/sec

---

## Quality-First Configuration

### High Quality Preset

Maximum accuracy, higher cost and latency.

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_CONCURRENCY=5
LLM_CONCURRENCY=3

# Document Processing
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=2
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=true

# Retrieval
RETRIEVAL_TOP_K=20
HYBRID_CHUNK_WEIGHT=0.6
HYBRID_ENTITY_WEIGHT=0.4
EXPANSION_DEPTH=2
MAX_EXPANDED_CHUNKS=150
EXPANSION_SIMILARITY_THRESHOLD=0.65

# Reranking
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2
FLASHRANK_BLEND_WEIGHT=0.0
FLASHRANK_MAX_CANDIDATES=50

# Generation
TEMPERATURE=0.3
MAX_TOKENS=3000

# Quality Features
ENABLE_QUALITY_SCORING=true
ENABLE_DESCRIPTION_SUMMARIZATION=true
ENABLE_DOCUMENT_SUMMARIES=true

# Clustering
ENABLE_CLUSTERING=true
CLUSTERING_RESOLUTION=1.5
CLUSTERING_MIN_EDGE_WEIGHT=0.3
```

**Cost Profile (1000 queries/month)**:
- Embeddings (large): ~$2.00
- Entity extraction (gpt-4o): ~$5.00
- Query responses (gpt-4o): ~$15.00
- **Total**: ~$22/month (22x default)

**Quality Improvements**:
- Entity extraction: +15-20% recall
- Retrieval: +10-15% relevance
- Generation: +5-10% coherence
- Overall: +25-30% quality score

**Latency**:
- Ingestion: 60-180s per document (3x slower)
- Query TTFT: 1000-1500ms (1.5x slower)

---

## Cost-Optimized Configuration

### Low Cost Preset

Minimize API costs while maintaining acceptable quality.

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_CONCURRENCY=2
LLM_CONCURRENCY=1

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
ENABLE_ENTITY_EXTRACTION=false
ENABLE_GLEANING=false
USE_MARKER_FOR_PDF=false

# Retrieval
RETRIEVAL_TOP_K=5
HYBRID_CHUNK_WEIGHT=0.8
HYBRID_ENTITY_WEIGHT=0.2
EXPANSION_DEPTH=0
MAX_EXPANDED_CHUNKS=20

# Reranking
FLASHRANK_ENABLED=false

# Generation
TEMPERATURE=0.7
MAX_TOKENS=1500

# Caching (aggressive)
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=10000
EMBEDDING_CACHE_SIZE=20000
RETRIEVAL_CACHE_SIZE=5000
RETRIEVAL_CACHE_TTL=300
RESPONSE_CACHE_SIZE=10000
RESPONSE_CACHE_TTL=1800

# Clustering
ENABLE_CLUSTERING=false
```

**Cost Profile (1000 queries/month)**:
- Embeddings (small): ~$0.05
- Entity extraction: $0 (disabled)
- Query responses (gpt-3.5): ~$0.10
- **Total**: ~$0.15/month (15% of default)

**Quality Impact**:
- Entity extraction: N/A (disabled)
- Retrieval: -15-20% relevance (no expansion, no reranking)
- Generation: -10-15% quality (simpler model)
- Overall: -25-35% quality score

**Latency**:
- Ingestion: 5-15s per document (3x faster)
- Query TTFT: 400-600ms (faster, especially with caching)

---

## Performance-Optimized Configuration

### Low Latency Preset

Minimize query latency for real-time applications.

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_CONCURRENCY=10
LLM_CONCURRENCY=5

# Retrieval
RETRIEVAL_TOP_K=5
EXPANSION_DEPTH=0
MAX_EXPANDED_CHUNKS=20

# Reranking
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-TinyBERT-L-2-v2
FLASHRANK_MAX_CANDIDATES=15

# Generation
TEMPERATURE=0.7
MAX_TOKENS=1500

# Caching (maximal)
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=10000
ENTITY_LABEL_CACHE_TTL=600
EMBEDDING_CACHE_SIZE=20000
RETRIEVAL_CACHE_SIZE=5000
RETRIEVAL_CACHE_TTL=300
RESPONSE_CACHE_SIZE=10000
RESPONSE_CACHE_TTL=1800

# Connection pooling
NEO4J_MAX_CONNECTION_POOL_SIZE=100
```

**Latency Profile**:
- Query TTFT: 300-500ms (cold), 50-100ms (cached)
- Streaming: 30-60 tokens/sec

**Cost**: Similar to default (~$1/month)

**Quality**: -10-15% (reduced context from no expansion)

---

## Enterprise Configuration

### Production Preset

Balanced for enterprise workloads with high quality requirements.

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_CONCURRENCY=10
LLM_CONCURRENCY=5
EMBEDDING_DELAY_MIN=0.2
EMBEDDING_DELAY_MAX=0.5

# Document Processing
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=1
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=true

# Retrieval
RETRIEVAL_TOP_K=15
HYBRID_CHUNK_WEIGHT=0.6
HYBRID_ENTITY_WEIGHT=0.4
EXPANSION_DEPTH=1
MAX_EXPANDED_CHUNKS=75
EXPANSION_SIMILARITY_THRESHOLD=0.7

# Reranking
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2
FLASHRANK_BLEND_WEIGHT=0.3
FLASHRANK_MAX_CANDIDATES=40
FLASHRANK_PREWARM_IN_PROCESS=false

# Generation
TEMPERATURE=0.5
MAX_TOKENS=2500

# Caching (production-optimized)
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=10000
ENTITY_LABEL_CACHE_TTL=600
EMBEDDING_CACHE_SIZE=20000
RETRIEVAL_CACHE_SIZE=2000
RETRIEVAL_CACHE_TTL=120
RESPONSE_CACHE_SIZE=5000
RESPONSE_CACHE_TTL=600
NEO4J_MAX_CONNECTION_POOL_SIZE=100

# Quality Features
ENABLE_QUALITY_SCORING=true
ENABLE_DESCRIPTION_SUMMARIZATION=true
ENABLE_DOCUMENT_SUMMARIES=true

# Clustering
ENABLE_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.2

# Observability
LOG_LEVEL=INFO
```

**Cost Profile (10,000 queries/month)**:
- ~$20-30/month

**Quality**: +10-15% vs default

**Latency**: 600-900ms TTFT

---

## Development Configuration

### Dev/Testing Preset

Fast iteration with deterministic behavior.

```bash
# LLM & Embeddings
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_CONCURRENCY=1
LLM_CONCURRENCY=1

# Document Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=100
ENABLE_ENTITY_EXTRACTION=true
SYNC_ENTITY_EMBEDDINGS=1
SKIP_ENTITY_EMBEDDINGS=0
ENABLE_GLEANING=false

# Retrieval
RETRIEVAL_TOP_K=5
EXPANSION_DEPTH=0
MAX_EXPANDED_CHUNKS=20

# Reranking
FLASHRANK_ENABLED=false

# Generation
TEMPERATURE=0.0
MAX_TOKENS=1000

# Caching (disabled for determinism)
ENABLE_CACHING=false

# Clustering
ENABLE_CLUSTERING=false

# Logging
LOG_LEVEL=DEBUG
```

**Benefits**:
- Deterministic results (temperature=0, no caching)
- Fast ingestion (small chunks, no gleaning)
- Simplified debugging (sequential processing)

---

## Parameter Impact Analysis

### Chunk Size

| Size | Ingestion Time | Retrieval Precision | Context Richness |
|------|----------------|---------------------|------------------|
| 800 | Fast (10s) | High (narrow match) | Low |
| 1200 | Medium (15s) | Medium | Medium (default) |
| 1600 | Slow (20s) | Low (broad match) | High |

**Recommendation**: 1200 for balanced performance

### Entity Extraction

| Setting | Cost | Quality Gain | Latency |
|---------|------|--------------|---------|
| Disabled | $0 | Baseline | Fast |
| Enabled, no gleaning | +$0.30/doc | +10% | +30s |
| Enabled, gleaning=1 | +$0.50/doc | +20% | +45s |
| Enabled, gleaning=2 | +$0.80/doc | +25% | +60s |

**Recommendation**: Gleaning=1 for best quality/cost ratio

### Expansion Depth

| Depth | Retrieval Time | Context Size | Quality Gain |
|-------|----------------|--------------|--------------|
| 0 | 200ms | Small (5-10 chunks) | Baseline |
| 1 | 350ms | Medium (20-40 chunks) | +10% |
| 2 | 600ms | Large (50-100 chunks) | +15% |
| 3 | 1200ms | XLarge (100+ chunks) | +18% |

**Recommendation**: Depth=1 for balanced, depth=2 for quality

### Reranking

| Model | Inference Time | Memory | Quality Gain |
|-------|----------------|--------|--------------|
| Disabled | 0ms | 0MB | Baseline |
| TinyBERT | 50-100ms | 50MB | +8% |
| MiniLM | 200-500ms | 200MB | +15% |
| rank-T5 | 1000-2000ms | 1GB | +20% |

**Recommendation**: MiniLM for production, TinyBERT for latency-sensitive

---

## Monitoring & Tuning

### Key Metrics

```bash
# Check quality scores
curl http://localhost:8000/api/chat -d '{"message": "...", ...}' \
  | jq '.quality_score'

# Check cache hit rates
curl http://localhost:8000/api/database/cache-stats \
  | jq '.embedding_cache.hit_rate, .retrieval_cache.hit_rate'

# Check latency breakdown
# (monitor SSE stage events)
```

### A/B Testing

```python
configs = {
    "default": {...},
    "high_quality": {...},
    "low_cost": {...}
}

for name, config in configs.items():
    results = []
    for query in test_queries:
        response = chat(query, **config)
        results.append({
            "config": name,
            "quality": response.quality_score,
            "latency": response.latency_ms,
            "sources": len(response.sources)
        })
    
    analyze(results)
```

---

## Migration Paths

### From Default → High Quality

```bash
# Phase 1: Improve retrieval
RETRIEVAL_TOP_K=15
EXPANSION_DEPTH=2
FLASHRANK_MAX_CANDIDATES=50

# Phase 2: Better models
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# Phase 3: Multi-pass extraction
MAX_GLEANINGS=2
```

### From Default → Low Cost

```bash
# Phase 1: Disable expensive features
ENABLE_ENTITY_EXTRACTION=false
FLASHRANK_ENABLED=false

# Phase 2: Cheaper models
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small

# Phase 3: Aggressive caching
RETRIEVAL_CACHE_SIZE=5000
RESPONSE_CACHE_SIZE=10000
```

---

## Related Documentation

- [Environment Variables](07-configuration/environment-variables.md)
- [RAG Tuning](07-configuration/rag-tuning.md)
- [Caching Settings](07-configuration/caching-settings.md)
- [Clustering Settings](07-configuration/clustering-settings.md)
- [Performance Optimization](02-core-concepts/performance.md)
