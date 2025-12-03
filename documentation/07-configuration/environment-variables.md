# Environment Variables

Complete reference for all environment variables in Amber.

## Quick Reference

```bash
# Copy example to create .env
cp .env.example .env
```

## Core Services

### OpenAI Configuration

```bash
# Required for LLM and embeddings
OPENAI_API_KEY=sk-...

# Optional: Custom base URL
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Model selection
OPENAI_MODEL=gpt-4o-mini

# Optional: Proxy URL
OPENAI_PROXY=http://proxy.example.com:8080
```

### Ollama Configuration (Alternative to OpenAI)

```bash
# Ollama server URL
OLLAMA_BASE_URL=http://localhost:11434

# Model names
OLLAMA_MODEL=llama2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Set provider to ollama
LLM_PROVIDER=ollama
```

### Neo4j Database

```bash
# Connection URI
NEO4J_URI=bolt://localhost:7687

# Credentials
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_secure_password

# Optional: Connection pool size
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### Redis (Optional - for job management)

```bash
REDIS_URL=redis://localhost:6379/0
```

---

## LLM & Embedding Models

### Model Selection

```bash
# Primary LLM model
OPENAI_MODEL=gpt-4o-mini

# Embedding model
EMBEDDING_MODEL=text-embedding-ada-002

# Provider selection
LLM_PROVIDER=openai  # or 'ollama'
```

**Supported OpenAI Models**:
- `gpt-4o` - Most capable, higher cost
- `gpt-4o-mini` - Balanced performance/cost (default)
- `gpt-3.5-turbo` - Faster, lower cost

**Supported Embedding Models**:
- `text-embedding-ada-002` - OpenAI default
- `text-embedding-3-small` - Lower cost
- `text-embedding-3-large` - Highest quality
- `nomic-embed-text` - Ollama local embedding

### Concurrency & Rate Limits

```bash
# Concurrent requests
EMBEDDING_CONCURRENCY=3
LLM_CONCURRENCY=2

# Rate limiting delays (seconds)
EMBEDDING_DELAY_MIN=0.5
EMBEDDING_DELAY_MAX=1.0
LLM_DELAY_MIN=0.5
LLM_DELAY_MAX=1.0
```

---

## Document Processing

### Chunking

```bash
# Chunk size in characters
CHUNK_SIZE=1200

# Overlap between chunks
CHUNK_OVERLAP=150
```

**Recommended Values**:
- Technical docs: `CHUNK_SIZE=1200`, `CHUNK_OVERLAP=150`
- General text: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`
- Short-form: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`

### Entity Extraction

```bash
# Enable entity extraction
ENABLE_ENTITY_EXTRACTION=true

# Multi-pass extraction (gleaning)
ENABLE_GLEANING=true
MAX_GLEANINGS=1

# Synchronous mode (for testing)
SYNC_ENTITY_EMBEDDINGS=0
SKIP_ENTITY_EMBEDDINGS=0
```

**Gleaning Configuration**:
```bash
# Document-type-specific gleaning (JSON format)
GLEANING_BY_DOC_TYPE='{"admin":2,"user":1,"support":0}'
```

### OCR & Quality

```bash
# Enable OCR for scanned documents
ENABLE_OCR=true
OCR_QUALITY_THRESHOLD=0.6

# Enable quality filtering
ENABLE_QUALITY_FILTERING=true
```

### Marker PDF Conversion

```bash
# Use Marker for high-quality PDF extraction
USE_MARKER_FOR_PDF=true
MARKER_OUTPUT_FORMAT=markdown

# LLM-enhanced processing (tables, complex layouts)
MARKER_USE_LLM=true
MARKER_LLM_MODEL=gpt-4o-mini

# OCR settings
MARKER_FORCE_OCR=true
MARKER_STRIP_EXISTING_OCR=false

# Performance tuning
MARKER_PDFTEXT_WORKERS=4
MARKER_PAGINATE_OUTPUT=true
```

---

## Retrieval Configuration

### Hybrid Retrieval Weights

```bash
# Vector vs entity weights (must sum to 1.0)
HYBRID_CHUNK_WEIGHT=0.6
HYBRID_ENTITY_WEIGHT=0.4

# Minimum similarity threshold
MIN_RETRIEVAL_SIMILARITY=0.1
```

### Graph Expansion

```bash
# Enable graph expansion
ENABLE_GRAPH_EXPANSION=true

# Expansion limits
MAX_EXPANDED_CHUNKS=500
MAX_EXPANSION_DEPTH=2
EXPANSION_SIMILARITY_THRESHOLD=0.1

# Connection limits
MAX_ENTITY_CONNECTIONS=20
MAX_CHUNK_CONNECTIONS=10
```

### Multi-hop Reasoning

```bash
# Multi-hop parameters
MULTI_HOP_MAX_HOPS=2
MULTI_HOP_BEAM_SIZE=8
MULTI_HOP_MIN_EDGE_STRENGTH=0.0
HYBRID_PATH_WEIGHT=0.6
```

### Context Restriction

```bash
# Restrict retrieval to context documents by default
DEFAULT_CONTEXT_RESTRICTION=true
```

---

## Reranking (FlashRank)

```bash
# Enable reranking
FLASHRANK_ENABLED=true

# Model selection
FLASHRANK_MODEL_NAME=ms-marco-TinyBERT-L-2-v2

# Performance tuning
FLASHRANK_MAX_CANDIDATES=100
FLASHRANK_BATCH_SIZE=32
FLASHRANK_MAX_LENGTH=128

# Score blending (0.0 = use reranker ordering)
FLASHRANK_BLEND_WEIGHT=0.0

# Cache directory
FLASHRANK_CACHE_DIR=/app/data/flashrank_cache

# Prewarm in process (false for production workers)
FLASHRANK_PREWARM_IN_PROCESS=true
```

**Model Options**:
- `ms-marco-TinyBERT-L-2-v2` - Fast, lightweight (default)
- `ms-marco-MiniLM-L-12-v2` - Better quality, slower
- `rank-T5-flan` - Highest quality, requires more memory

---

## Caching

```bash
# Enable caching system
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

# Document detail cache
DOCUMENT_SUMMARY_TTL=300
DOCUMENT_DETAIL_CACHE_TTL=60
```

---

## Graph Clustering

```bash
# Enable clustering
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true

# Leiden algorithm parameters
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.0
CLUSTERING_LEVEL=0
DEFAULT_EDGE_WEIGHT=1.0

# Relationship types to include
CLUSTERING_RELATIONSHIP_TYPES='["SIMILAR_TO","RELATED_TO"]'
```

**Resolution Guidelines**:
- `0.5` - Fewer, larger communities
- `1.0` - Balanced (default)
- `2.0` - More, smaller communities

---

## Quality Scoring

```bash
# Enable quality scoring
ENABLE_QUALITY_SCORING=true

# Component weights (must sum to 1.0)
QUALITY_SCORE_WEIGHTS='{
  "context_relevance": 0.30,
  "answer_completeness": 0.25,
  "factual_grounding": 0.25,
  "coherence": 0.10,
  "citation_quality": 0.10
}'
```

---

## Advanced Features

### Phase 2: NetworkX Intermediate Layer

```bash
# Enable NetworkX graph layer (reduces duplicates)
ENABLE_PHASE2_NETWORKX=true

# Batch persistence settings
NEO4J_UNWIND_BATCH_SIZE=500
MAX_NODES_PER_DOC=2000
MAX_EDGES_PER_DOC=5000

# Filtering thresholds
IMPORTANCE_SCORE_THRESHOLD=0.3
STRENGTH_THRESHOLD=0.4

# Phase version tag
PHASE_VERSION=phase2_v1
```

### Phase 3: Tuple-Delimited Format

```bash
# Entity extraction format
ENTITY_EXTRACTION_FORMAT=tuple_v1

# Tuple format settings
TUPLE_FORMAT_VALIDATION=true
TUPLE_DELIMITER=<|>
TUPLE_MAX_DESCRIPTION_LENGTH=500
```

### Phase 4: Description Summarization

```bash
# Enable LLM-based description summarization
ENABLE_DESCRIPTION_SUMMARIZATION=true

# Summarization thresholds
SUMMARIZATION_MIN_MENTIONS=3
SUMMARIZATION_MIN_LENGTH=200

# Batch processing
SUMMARIZATION_BATCH_SIZE=5
SUMMARIZATION_CACHE_ENABLED=true
```

### Document Summaries

```bash
# Enable precomputed document summaries
ENABLE_DOCUMENT_SUMMARIES=true

# Preview limits
DOCUMENT_SUMMARY_TOP_N_COMMUNITIES=10
DOCUMENT_SUMMARY_TOP_N_SIMILARITIES=20
```

---

## Application Settings

### General

```bash
# Logging level
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# Upload limits
MAX_UPLOAD_SIZE=104857600  # 100MB in bytes
```

### Database Operations

```bash
# Enable delete operations
ENABLE_DELETE_OPERATIONS=true
```

### Similarity Computation

```bash
# Similarity thresholds
SIMILARITY_THRESHOLD=0.7
MAX_SIMILARITY_CONNECTIONS=5
```

---

## Production Recommendations

```bash
# Core services
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=strong_random_password

# Performance
EMBEDDING_CONCURRENCY=5
LLM_CONCURRENCY=3
NEO4J_MAX_CONNECTION_POOL_SIZE=50

# Quality features
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=1
FLASHRANK_ENABLED=true
ENABLE_QUALITY_SCORING=true

# Caching
ENABLE_CACHING=true
ENTITY_LABEL_CACHE_SIZE=5000
EMBEDDING_CACHE_SIZE=10000
RETRIEVAL_CACHE_SIZE=1000

# Graph expansion
ENABLE_GRAPH_EXPANSION=true
MAX_EXPANDED_CHUNKS=500
MAX_EXPANSION_DEPTH=2

# Clustering
ENABLE_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0

# Marker PDF
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=true

# Prewarm in external worker
FLASHRANK_PREWARM_IN_PROCESS=false
```

---

## Development Overrides

```bash
# Faster iteration
ENABLE_ENTITY_EXTRACTION=false
ENABLE_GLEANING=false
FLASHRANK_ENABLED=false
ENABLE_CLUSTERING=false

# Verbose logging
LOG_LEVEL=DEBUG

# Synchronous entity processing for tests
SYNC_ENTITY_EMBEDDINGS=1
```

---

## Docker Compose

Variables in `.env` are automatically loaded by `docker-compose.yml`:

```yaml
services:
  backend:
    env_file:
      - .env
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
```

---

## Validation

Check loaded settings:

```bash
# View current configuration
curl http://localhost:8000/api/settings

# Check cache stats
curl http://localhost:8000/api/database/cache-stats
```

---

## Related Documentation

- [RAG Tuning](07-configuration/rag-tuning.md)
- [Caching Settings](07-configuration/caching-settings.md)
- [Clustering Settings](07-configuration/clustering-settings.md)
- [Optimal Defaults](07-configuration/optimal-defaults.md)
- [Settings Module](03-components/backend/settings.md)
