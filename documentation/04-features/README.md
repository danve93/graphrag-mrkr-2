# Features

Advanced features and capabilities of the Amber platform.

## Contents

- [README](04-features/README.md) - Features overview
- [Entity Clustering](04-features/community-detection.md) - Leiden community detection for semantic grouping
- [Reranking](05-data-flows/reranking-flow.md) - FlashRank cross-encoder post-retrieval reranking
- [Chat Tuning](04-features/chat-tuning.md) - Runtime model and retrieval parameter controls
- [Document Classification](04-features/document-upload.md) - Automatic document labeling
- [Gleaning](04-features/entity-reasoning.md) - Multi-pass entity extraction for improved recall
- [Response Caching](02-core-concepts/caching-system.md) - Semantic caching for complete pipeline responses

## Feature Overview

### Entity Clustering

**Status**: Production-ready

Leiden community detection automatically groups related entities into semantic communities based on relationship strength. Each entity receives a `community_id` property used for:
- Graph visualization with distinct colors (10-color palette)
- Filtered retrieval by community
- Topic discovery and analysis
- Optional LLM-generated community summaries

**Configuration**:
```bash
ENABLE_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.3
```

**Usage**:
```bash
python scripts/run_clustering.py
```

See [Entity Clustering](04-features/community-detection.md) for implementation details.

### Reranking

**Status**: Production-ready (optional)

FlashRank cross-encoder reranking improves retrieval precision by:
- Re-scoring top-K candidates with cross-encoder model
- Blending rerank scores with hybrid retrieval scores
- Running locally without API calls (50-100ms latency)

**Configuration**:
```bash
FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2
FLASHRANK_MAX_CANDIDATES=50
FLASHRANK_BLEND_WEIGHT=0.7
```

Pre-warmed at startup when enabled. Falls back gracefully if import fails.

See [Reranking](05-data-flows/reranking-flow.md) for algorithm details.

### Chat Tuning

**Status**: Production-ready

Runtime controls for model selection and retrieval parameters without code changes:
- LLM model selection (OpenAI, Ollama)
- Embedding model selection
- Temperature, top_p, top_k
- Retrieval mode (vector, hybrid, graph)
- Hybrid weights (chunks, entities, paths)
- Graph expansion parameters (depth, beam size)

**API**:
- `GET /api/chat-tuning/config` - Download current configuration
- `GET /api/chat-tuning/config/values` - Get runtime values
- UI panel in frontend for interactive tuning

See [Chat Tuning](04-features/chat-tuning.md) for parameter reference.

### Query Routing

**Status**: Production-ready

Intelligent query classification to document categories:
- LLM-based routing with confidence scoring (0.7 threshold)
- Multi-category routing (queries can match 2-3 categories)
- Semantic caching for 30-50% latency reduction (0.92 similarity threshold)
- Automatic fallback to full search when <3 chunks retrieved
- Feature-flagged with zero-downtime rollback

**Configuration**:
```bash
ENABLE_QUERY_ROUTING=true
QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.7
ENABLE_ROUTING_CACHE=true
ROUTING_CACHE_SIMILARITY_THRESHOLD=0.92
```

**API**:
- `GET /api/database/routing-metrics` - Performance metrics
- `/routing` dashboard for real-time monitoring

See [Query Routing](04-features/query-routing.md) for architecture details.

### Routing Metrics

**Status**: Production-ready

Comprehensive routing performance tracking:
- Latency tracking (avg 150-180ms)
- Cache hit rates (typical 35-45%)
- Fallback rates and category usage
- 7 failure point types (FP1-FP7)
- User feedback accuracy measurement

**Metrics**:
- Real-time dashboard at `/routing`
- JSON API at `/api/database/routing-metrics`
- Grafana integration support

See [Routing Metrics](04-features/routing-metrics.md) for complete metrics reference.

### Smart Consolidation

**Status**: Production-ready

Category-aware result ranking and deduplication:
- Ensures minimum 1 chunk per target category
- Semantic deduplication (>0.95 similarity threshold)
- Token budget enforcement (8K max)
- Preserves diversity while maximizing relevance

**Configuration**:
```python
SmartConsolidator(
    max_tokens=8000,
    semantic_threshold=0.95,
    ensure_category_representation=True,
    min_chunks_per_category=1
)
```

See [Smart Consolidation](04-features/smart-consolidation.md) for algorithm details.

### Category-Specific Prompts

**Status**: Production-ready

Intelligent prompt selection based on document categories:
- 10 pre-configured category templates (installation, API, troubleshooting, etc.)
- Category-specific retrieval strategies (step-back, PPR, balanced)
- Configurable specificity levels (concise to comprehensive)
- Conversation history integration (last 3 turns)

**Configuration**:
```bash
ENABLE_CATEGORY_PROMPTS=true
ENABLE_CATEGORY_PROMPT_INSTRUCTIONS=true
```

**Categories**:
- Installation, Configuration, Troubleshooting, API, Conceptual
- Quickstart, Reference, Example, Best Practices, Default

See [Category Prompts](04-features/category-prompts.md) for template reference.

### Structured Knowledge Graph Queries

**Status**: Production-ready

Text-to-Cypher translation for direct graph queries:
- Automatic detection of suitable query types (aggregation, path, relationship)
- Entity linking with 0.85 similarity threshold
- Iterative Cypher correction (max 2 attempts)
- 60-80% faster for aggregation queries vs standard retrieval
- UI interface at `/structured-kg`

**Configuration**:
```bash
ENABLE_STRUCTURED_KG=true
STRUCTURED_KG_ENTITY_THRESHOLD=0.85
STRUCTURED_KG_MAX_CORRECTIONS=2
STRUCTURED_KG_TIMEOUT=5000
```

**API**:
- `POST /api/structured-kg/execute` - Execute query
- `GET /api/structured-kg/schema` - Get graph schema

See [Structured KG](04-features/structured-kg.md) for query types and examples.

### Adaptive Routing

**Status**: Production-ready

Feedback-based weight adjustment:
- Learns optimal retrieval weights from user ratings (thumbs up/down)
- Exponential moving average updates (0.1 learning rate)
- Per-query-type performance tracking
- 10-15% accuracy improvement after 50+ samples
- Bounded weight adjustments (0.1-0.9 range)

**Configuration**:
```bash
ENABLE_ADAPTIVE_ROUTING=true
ADAPTIVE_LEARNING_RATE=0.1
ADAPTIVE_MIN_SAMPLES=5
```

**API**:
- `POST /api/feedback` - Submit rating
- `GET /api/feedback/weights` - Current weights
- `GET /api/feedback/stats` - Performance by query type

See [Adaptive Routing](04-features/adaptive-routing.md) for learning algorithm.

### Document Classification

**Status**: Production-ready

Automatic labeling system:
- Rule-based classification via regex patterns
- LLM-based classification for complex logic
- Configurable categories and rules
- Reindex operation to apply classification to existing documents

**Configuration**: `config/classification_config.json`

**API**:
- `POST /api/classification/reindex` - Reclassify all documents

See [Document Classification](04-features/document-upload.md) for configuration schema.

### Gleaning

**Status**: Production-ready (optional)

Multi-pass entity extraction:
- Initial extraction pass identifies entities
- Gleaning pass captures missed entities using first-pass context
- Configurable number of gleaning iterations (default 1)
- Increases recall by 30-40% at 2x API cost

**Configuration**:
```bash
ENABLE_GLEANING=true
MAX_GLEANINGS=1
```

Controlled in entity extraction settings.

See [Gleaning](04-features/entity-reasoning.md) for research background.

### Response Caching

**Status**: Production-ready (optional)

Semantic caching of complete pipeline responses:
- Caches full response (text, sources, metadata)
- TTL-based expiration (default 300s)
- Automatic invalidation on document mutations
- Cache key includes query text and retrieval parameters

**Configuration**:
```bash
ENABLE_CACHING=true
RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300
```

**Monitoring**: `GET /api/database/cache-stats`

See [Response Caching](02-core-concepts/caching-system.md) for implementation.

## Feature Flags

All features are controlled via environment variables or runtime configuration:

| Feature | Flag | Default | Impact |
|---------|------|---------|--------|
| Query Routing | `ENABLE_QUERY_ROUTING` | `false` | Automatic query-to-category routing |
| Routing Cache | `ENABLE_ROUTING_CACHE` | `true` | Semantic cache for routing (30%+ latency reduction) |
| Structured KG | `ENABLE_STRUCTURED_KG` | `true` | Text-to-Cypher for graph queries |
| Category Prompts | `ENABLE_CATEGORY_PROMPTS` | `true` | Category-specific prompt selection |
| Adaptive Routing | `ENABLE_ADAPTIVE_ROUTING` | `true` | Feedback-based weight learning |
| Entity Extraction | `ENABLE_ENTITY_EXTRACTION` | `true` | Enables entity extraction during ingestion |
| Gleaning | `ENABLE_GLEANING` | `true` | Multi-pass entity extraction |
| Clustering | `ENABLE_CLUSTERING` | `true` | Community detection |
| Reranking | `FLASHRANK_ENABLED` | `true` | Post-retrieval reranking |
| Response Cache | `ENABLE_CACHING` | `true` | Response-level caching |
| Quality Scoring | `ENABLE_QUALITY_SCORING` | `true` | Chunk quality assessment |
| NetworkX Dedup | `ENABLE_PHASE2_NETWORKX` | `true` | In-memory entity deduplication |

## Performance Impact

**Quality vs Cost Tradeoffs**:

| Configuration | Ingestion Time | API Calls | Quality Improvement |
|---------------|----------------|-----------|---------------------|
| Minimal | 1x | 1x | Baseline |
| Default (all features) | 2.5-3x | 2-3x | +30-40% entities, +50-70% token efficiency |
| Maximum accuracy | 4-5x | 4-5x | +95% extraction accuracy |

See [Optimal Defaults](../../07-configuration/optimal-defaults.md) for detailed benchmarks.

## Experimental Features

Features under development:
- Hierarchical clustering (multi-level communities)
- Adaptive chunking (dynamic chunk size based on content)
- Query expansion (automatic query reformulation)
- Multi-modal embeddings (text + image)

## Feature Development

To add new features:
1. Implement core logic in `core/` or `rag/`
2. Add feature flag to `config/settings.py`
3. Add configuration to `.env.example`
4. Update relevant routers in `api/routers/`
5. Add tests in `tests/integration/`
6. Document in this directory

See [Development Guide](09-development/feature-flag-wiring.md).

## Related Documentation

- [Configuration Reference](07-configuration)
- [Backend Components](03-components/backend)
- [Operations Guide](08-operations)
