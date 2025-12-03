# Configuration Guide

Comprehensive configuration reference for Amber platform.

## Configuration Hierarchy

Settings are applied in the following precedence order (highest to lowest):

1. **Chat Tuning** (runtime) - UI-based parameter overrides
2. **RAG Tuning** (config file) - `config/rag_tuning_config.json`
3. **Environment Variables** - `.env` file
4. **Default Values** - Hard-coded in `config/settings.py`

## Environment Variables

### Core Configuration

**File**: `.env` (copy from `.env.example`)

### LLM Configuration

```bash
# Provider Selection
LLM_PROVIDER=openai              # "openai" or "ollama"

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...       # Required for OpenAI
OPENAI_MODEL=gpt-4o-mini         # Model name
OPENAI_BASE_URL=https://api.openai.com/v1  # API endpoint

# Ollama Configuration (optional)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1            # Model name
```

**Supported OpenAI Models**:
- `gpt-4o` (most capable, slower, expensive)
- `gpt-4o-mini` (balanced, recommended)
- `gpt-3.5-turbo` (fast, cheaper, less capable)

**Supported Ollama Models**:
- `llama3.1` (recommended)
- `mistral`
- `phi3`
- Any model pulled via `ollama pull`

### Embedding Configuration

```bash
# Embedding Model Selection
EMBEDDING_MODEL=text-embedding-3-small  # Default

# OpenAI Embeddings
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Supported Models**:
- `text-embedding-3-small` (1536 dims, fast, cheap)
- `text-embedding-3-large` (3072 dims, higher quality)
- `text-embedding-ada-002` (legacy, 1536 dims)

### Neo4j Configuration

```bash
# Connection
NEO4J_URI=bolt://localhost:7687  # Database URI
NEO4J_USERNAME=neo4j             # Username
NEO4J_PASSWORD=password          # Password

# Performance
NEO4J_MAX_CONNECTION_POOL_SIZE=50  # Connection pool size
NEO4J_UNWIND_BATCH_SIZE=500        # Batch insert size
```

### Caching Configuration

```bash
# Master Switch
ENABLE_CACHING=true              # Enable multi-layer caching

# Entity Label Cache
ENTITY_LABEL_CACHE_SIZE=5000     # Max entries
ENTITY_LABEL_CACHE_TTL=300       # Time-to-live (seconds)

# Embedding Cache
EMBEDDING_CACHE_SIZE=10000       # Max entries (LRU)

# Retrieval Cache
RETRIEVAL_CACHE_SIZE=1000        # Max entries
RETRIEVAL_CACHE_TTL=60           # Time-to-live (seconds)

# Response Cache
RESPONSE_CACHE_SIZE=2000         # Max entries
RESPONSE_CACHE_TTL=300           # Time-to-live (seconds)
```

### Retrieval Configuration

```bash
# Hybrid Retrieval Weights
HYBRID_CHUNK_WEIGHT=0.7          # Vector similarity weight (0.0-1.0)
HYBRID_ENTITY_WEIGHT=0.3         # Entity match weight (0.0-1.0)

# Graph Expansion
MAX_EXPANDED_CHUNKS=30           # Maximum chunks after expansion
MAX_EXPANSION_DEPTH=2            # Maximum hops (1-3)
EXPANSION_SIMILARITY_THRESHOLD=0.7  # Minimum edge strength

# Top-K
TOP_K=10                         # Initial retrieval count
```

### Reranking Configuration

```bash
# FlashRank
FLASHRANK_ENABLED=false          # Enable reranking
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2  # Model name
FLASHRANK_MAX_CANDIDATES=50      # Max chunks to rerank
FLASHRANK_BLEND_WEIGHT=0.5       # Blend with hybrid scores (0.0-1.0)
FLASHRANK_BATCH_SIZE=32          # Batch size for inference
FLASHRANK_CACHE_DIR=data/flashrank_cache  # Model cache path
```

### Entity Extraction Configuration

```bash
# Entity Features
ENABLE_ENTITY_EXTRACTION=true    # Enable entity extraction
SYNC_ENTITY_EMBEDDINGS=false     # Synchronous vs async extraction

# Phase 2 Features
ENABLE_PHASE2_NETWORKX=true      # In-memory deduplication
ACCUMULATION_SIMILARITY_THRESHOLD=0.85  # Entity merge threshold
```

### Ingestion Configuration

```bash
# Chunking
CHUNK_SIZE=1000                  # Characters per chunk
CHUNK_OVERLAP=200                # Overlap between chunks

# OCR
ENABLE_OCR=false                 # Enable OCR for images/scans
OCR_QUALITY_THRESHOLD=0.7        # Minimum OCR confidence

# Quality Scoring
QUALITY_SCORE_ENABLED=true       # Enable chunk quality scoring

# Concurrency
EMBEDDING_CONCURRENCY=5          # Concurrent embedding requests
EMBEDDING_DELAY_MIN=0.0          # Min delay between requests (seconds)
EMBEDDING_DELAY_MAX=0.1          # Max delay between requests (seconds)
```

### Clustering Configuration

```bash
# Leiden Clustering
ENABLE_CLUSTERING=false          # Enable at ingestion time
ENABLE_GRAPH_CLUSTERING=true     # Enable clustering feature
CLUSTERING_RESOLUTION=1.0        # Granularity (0.1-2.0)
CLUSTERING_MIN_EDGE_WEIGHT=0.3   # Minimum edge strength
CLUSTERING_RELATIONSHIP_TYPES=RELATED_TO,SIMILAR_TO  # Edge types
CLUSTERING_LEVEL=0               # Hierarchy level
```

### Logging Configuration

```bash
# Log Level
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR

# Log Files
LOG_FILE=backend.log             # Backend log file path
```

## RAG Tuning Configuration

**File**: `config/rag_tuning_config.json`

This file provides defaults for RAG pipeline parameters. Values can be overridden at runtime via Chat Tuning UI.

```json
{
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "temperature": 0.1,
  "top_k": 10,
  "max_expanded_chunks": 30,
  "max_expansion_depth": 2,
  "expansion_similarity_threshold": 0.7,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  "enable_reranking": false,
  "flashrank_max_candidates": 50,
  "flashrank_blend_weight": 0.5
}
```

### Parameter Reference

**llm_model**: LLM for generation
- Type: string
- Options: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `llama3.1`
- Default: `gpt-4o-mini`

**embedding_model**: Model for embeddings
- Type: string
- Options: `text-embedding-3-small`, `text-embedding-3-large`
- Default: `text-embedding-3-small`

**temperature**: Generation randomness
- Type: float
- Range: 0.0 (deterministic) to 2.0 (creative)
- Default: 0.1

**top_k**: Initial retrieval count
- Type: integer
- Range: 1-100
- Default: 10

**max_expanded_chunks**: Maximum chunks after expansion
- Type: integer
- Range: 10-100
- Default: 30

**max_expansion_depth**: Graph traversal depth
- Type: integer
- Range: 1-3
- Default: 2
- Note: Higher values increase latency

**expansion_similarity_threshold**: Minimum edge strength
- Type: float
- Range: 0.0-1.0
- Default: 0.7

**hybrid_chunk_weight**: Vector similarity weight
- Type: float
- Range: 0.0-1.0
- Default: 0.7

**hybrid_entity_weight**: Entity match weight
- Type: float
- Range: 0.0-1.0
- Default: 0.3
- Note: Must sum to 1.0 with chunk_weight

**enable_reranking**: Enable FlashRank
- Type: boolean
- Default: false

**flashrank_max_candidates**: Chunks to rerank
- Type: integer
- Range: 10-100
- Default: 50

**flashrank_blend_weight**: Blend rerank scores
- Type: float
- Range: 0.0 (ignore rerank) to 1.0 (use only rerank)
- Default: 0.5

## Chat Tuning Configuration

**File**: `config/chat_tuning_config.json`

Runtime overrides applied via UI. These override both RAG tuning and environment variables.

```json
{
  "enabled": true,
  "overrides": {
    "llm_model": "gpt-4o",
    "temperature": 0.3,
    "top_k": 15,
    "max_expansion_depth": 3
  }
}
```

Access via UI: Settings icon > Chat Tuning panel

## Performance Tuning Presets

### Fast (Low Latency)

```bash
# Environment
MAX_EXPANSION_DEPTH=1
MAX_EXPANDED_CHUNKS=15
ENABLE_RERANKING=false
FLASHRANK_ENABLED=false

# RAG Tuning
{
  "top_k": 5,
  "max_expansion_depth": 1,
  "max_expanded_chunks": 15,
  "enable_reranking": false
}
```

**Use case**: Quick queries, real-time chat

### Balanced (Default)

```bash
# Environment
MAX_EXPANSION_DEPTH=2
MAX_EXPANDED_CHUNKS=30
FLASHRANK_ENABLED=false

# RAG Tuning
{
  "top_k": 10,
  "max_expansion_depth": 2,
  "max_expanded_chunks": 30,
  "enable_reranking": false
}
```

**Use case**: General purpose, production default

### High Quality (Best Results)

```bash
# Environment
MAX_EXPANSION_DEPTH=3
MAX_EXPANDED_CHUNKS=50
FLASHRANK_ENABLED=true
FLASHRANK_MAX_CANDIDATES=50

# RAG Tuning
{
  "llm_model": "gpt-4o",
  "top_k": 20,
  "max_expansion_depth": 3,
  "max_expanded_chunks": 50,
  "enable_reranking": true,
  "flashrank_max_candidates": 50
}
```

**Use case**: Complex queries, research, high accuracy requirements

## Environment-Specific Configuration

### Development

```bash
# .env.development
LOG_LEVEL=DEBUG
ENABLE_CACHING=false
SYNC_ENTITY_EMBEDDINGS=true
EMBEDDING_CONCURRENCY=1
LLM_CONCURRENCY=1
```

### Testing

```bash
# .env.test
LOG_LEVEL=WARNING
ENABLE_CACHING=false
ENABLE_ENTITY_EXTRACTION=false
NEO4J_URI=bolt://localhost:7688
```

### Production

```bash
# .env.production
LOG_LEVEL=INFO
ENABLE_CACHING=true
ENABLE_ENTITY_EXTRACTION=true
NEO4J_MAX_CONNECTION_POOL_SIZE=100
EMBEDDING_CONCURRENCY=10
LLM_CONCURRENCY=10
```

## Configuration Validation

Backend validates configuration at startup and logs resolved values:

```
INFO: LLM Provider: openai (model: gpt-4o-mini)
INFO: Embedding Model: text-embedding-3-small
INFO: Neo4j URI: bolt://neo4j:7687
INFO: Caching: enabled (entity: 5000, embedding: 10000, retrieval: 1000)
INFO: FlashRank: disabled
```

### Check Current Configuration

```bash
# Via API
curl http://localhost:8000/api/settings

# Via logs
grep "Config" backend.log
```

## Security Best Practices

### API Keys

**Never commit secrets**:
```bash
# Add to .gitignore
.env
.env.local
.env.production
```

**Use environment-specific files**:
```bash
.env.development
.env.test
.env.production
```

**Rotate keys regularly**: Update `OPENAI_API_KEY` every 90 days

### Neo4j Security

**Use strong passwords**:
```bash
NEO4J_PASSWORD=$(openssl rand -base64 32)
```

**Restrict network access**: In production, bind Neo4j to internal network only

**Enable authentication**: Never use `NEO4J_AUTH=none` in production

## Troubleshooting Configuration

### Values Not Applied

Check precedence order:
1. Chat Tuning (highest)
2. RAG Tuning
3. Environment Variables
4. Defaults (lowest)

### Missing Environment Variables

Backend will use defaults but log warnings:
```
WARNING: OPENAI_API_KEY not set, LLM features disabled
```

### Invalid Values

Backend validates and rejects invalid values:
```
ERROR: MAX_EXPANSION_DEPTH must be 1-3, got 5
```

### Cache Not Working

Verify master switch:
```bash
ENABLE_CACHING=true
```

Check metrics:
```bash
curl http://localhost:8000/api/database/cache-stats
```

## Related Documentation

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Optimal Settings](../../docs/OPTIMAL_SETTINGS.md)
- [Performance Tuning](08-operations/performance-tuning.md)
- [Chat Tuning Feature](04-features/chat-tuning.md)
