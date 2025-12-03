# Configuration

Comprehensive configuration reference for Amber.

## Contents

- [README](07-configuration/README.md) - Configuration overview
- [Environment Variables](07-configuration/environment-variables.md) - Complete environment variable reference
- [Feature Flags](07-configuration/feature-flags.md) - Feature flag reference and rollout guide
- [RAG Tuning](07-configuration/rag-tuning.md) - Retrieval and generation parameters
- [Caching Settings](07-configuration/caching-settings.md) - Cache sizes, TTLs, and performance tuning
- [Clustering Settings](07-configuration/clustering-settings.md) - Leiden algorithm parameters
- [Optimal Defaults](07-configuration/optimal-defaults.md) - Quality vs cost tradeoff analysis

## Configuration Methods

Amber supports multiple configuration sources in order of precedence:

1. **Runtime Chat Tuning** - Highest priority, per-request overrides
2. **Environment Variables** - Set via `.env` file or system environment
3. **Configuration Files** - JSON files in `config/` directory
4. **Default Values** - Hardcoded in `config/settings.py`

## Quick Configuration

### Minimal Setup

```bash
cp .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

Start services:
```bash
docker compose up -d
```

### Recommended Production Setup

```bash
OPENAI_API_KEY=sk-...
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=strong_random_password

ENABLE_CACHING=true
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
FLASHRANK_ENABLED=true

CHUNK_SIZE=1200
CHUNK_OVERLAP=150

USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
```

## Key Configuration Areas

### LLM and Embeddings

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Neo4j Database

```bash
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_MAX_CONNECTION_POOL_SIZE=50
```

### Performance and Caching

```bash
ENABLE_CACHING=true

ENTITY_LABEL_CACHE_SIZE=5000
ENTITY_LABEL_CACHE_TTL=300

EMBEDDING_CACHE_SIZE=10000

RETRIEVAL_CACHE_SIZE=1000
RETRIEVAL_CACHE_TTL=60

RESPONSE_CACHE_SIZE=2000
RESPONSE_CACHE_TTL=300
```

### Ingestion Features

```bash
ENABLE_ENTITY_EXTRACTION=true
ENABLE_GLEANING=true
MAX_GLEANINGS=1

ENABLE_PHASE2_NETWORKX=true
ENABLE_DESCRIPTION_SUMMARIZATION=true

ENABLE_QUALITY_SCORING=true

CHUNK_SIZE=1200
CHUNK_OVERLAP=150
```

### Marker PDF Processing

```bash
USE_MARKER_FOR_PDF=true
MARKER_USE_LLM=true
MARKER_FORCE_OCR=false
MARKER_DEVICE=cpu
MARKER_DTYPE=float32
```

### Retrieval and RAG

```bash
RETRIEVAL_MODE=hybrid
HYBRID_CHUNK_WEIGHT=0.7
HYBRID_ENTITY_WEIGHT=0.3
HYBRID_PATH_WEIGHT=0.0

MAX_EXPANDED_CHUNKS=50
MAX_EXPANSION_DEPTH=2
EXPANSION_SIMILARITY_THRESHOLD=0.7

FLASHRANK_ENABLED=true
FLASHRANK_MODEL_NAME=ms-marco-MiniLM-L-12-v2
FLASHRANK_MAX_CANDIDATES=50
FLASHRANK_BLEND_WEIGHT=0.7
```

### Clustering

```bash
ENABLE_CLUSTERING=true
ENABLE_GRAPH_CLUSTERING=true
CLUSTERING_RESOLUTION=1.0
CLUSTERING_MIN_EDGE_WEIGHT=0.3
CLUSTERING_RELATIONSHIP_TYPES=RELATED_TO,SIMILAR_TO
CLUSTERING_LEVEL=0
```

## Configuration Files

### RAG Tuning Config

**File**: `config/rag_tuning_config.json`

Runtime-tunable parameters exposed via Chat Tuning UI:
```json
{
  "retrieval_mode": "hybrid",
  "top_k": 10,
  "temperature": 0.7,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  "max_expansion_depth": 2
}
```

### Classification Config

**File**: `config/classification_config.json`

Document classification rules:
```json
{
  "categories": [
    {
      "name": "Technical Documentation",
      "rules": [
        {"type": "filename_pattern", "pattern": ".*\\.md$"},
        {"type": "content_pattern", "pattern": "API|SDK|Tutorial"}
      ]
    }
  ]
}
```

### Chat Tuning Config

**File**: `config/chat_tuning_config.json`

Default chat parameters:
```json
{
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 10
}
```

## Environment-Specific Configuration

### Development

```bash
DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_CACHING=false
SYNC_ENTITY_EMBEDDINGS=true
```

### Testing

```bash
NEO4J_URI=bolt://localhost:7687
ENABLE_CACHING=false
SYNC_ENTITY_EMBEDDINGS=true
ENABLE_GLEANING=false
```

### Production

```bash
DEBUG=false
LOG_LEVEL=INFO
ENABLE_CACHING=true
SYNC_ENTITY_EMBEDDINGS=false
```

## Configuration Validation

Settings are validated at startup via Pydantic:

```python
from config.settings import settings

settings.validate()
```

Invalid configurations will raise descriptive errors.

## Monitoring Configuration

Check current configuration:
```bash
curl http://localhost:8000/api/health
```

Response includes:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "features": {
    "entity_extraction": true,
    "clustering": true,
    "reranking": true,
    "caching": true
  },
  "llm_provider": "openai",
  "embedding_model": "text-embedding-3-small"
}
```

Cache performance:
```bash
curl http://localhost:8000/api/database/cache-stats
```

## Security Best Practices

1. **Never commit secrets** - Use `.env` files (gitignored)
2. **Use strong passwords** - Generate random passwords for Neo4j
3. **Rotate API keys** - Regularly update OpenAI and other API keys
4. **Restrict access** - Use firewall rules to limit database access
5. **Use HTTPS** - Enable TLS for production deployments
6. **Environment isolation** - Separate dev/staging/prod configurations

## Configuration Troubleshooting

**Issue**: Settings not taking effect
- Check `.env` file exists and is readable
- Verify environment variable names (case-sensitive)
- Restart services after configuration changes
- Check logs for validation errors

**Issue**: Performance degradation
- Review cache hit rates at `/api/database/cache-stats`
- Increase cache sizes if hit rates are low
- Adjust TTLs based on document update frequency
- Monitor Neo4j connection pool usage

**Issue**: High API costs
- Disable gleaning: `ENABLE_GLEANING=false`
- Disable Marker LLM mode: `MARKER_USE_LLM=false`
- Reduce chunk size: `CHUNK_SIZE=800`
- Disable description summarization: `ENABLE_DESCRIPTION_SUMMARIZATION=false`

## Related Documentation

- [Environment Variables](07-configuration/environment-variables.md)
- [RAG Tuning Parameters](07-configuration/rag-tuning.md)
- [Optimal Defaults Analysis](07-configuration/optimal-defaults.md)
- [Performance Tuning](08-operations/monitoring.md)
