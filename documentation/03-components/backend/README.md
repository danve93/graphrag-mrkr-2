# Backend Components

FastAPI application layer, RAG pipeline implementation, and core services.

## Contents

- [README](03-components/backend/README.md) - Backend overview
- [API Layer](03-components/backend/api-layer.md) - FastAPI routers, endpoints, and request handling
- [RAG Pipeline](03-components/backend/rag-pipeline.md) - LangGraph state machine orchestrating retrieval and generation
- [Retriever](03-components/backend/retriever.md) - Hybrid retrieval combining vector search and graph expansion
- [Graph Database](03-components/backend/graph-database.md) - Neo4j integration and query patterns
- [Embeddings](03-components/backend/embeddings.md) - Embedding generation and caching
- [Entity Extraction](03-components/backend/entity-extraction.md) - LLM-based entity extraction and deduplication
- [Clustering](03-components/backend/clustering.md) - Leiden community detection algorithm
- [Job Management](03-components/backend/job-management.md) - Background job processing and scheduling

## Architecture

The backend is organized into layers:

### API Layer (`api/`)
- FastAPI application with dependency injection
- Routers for chat, documents, database, history, jobs
- Pydantic models for request/response validation
- SSE streaming for real-time updates
- Lifecycle hooks for startup/shutdown

### RAG Pipeline (`rag/`)
- LangGraph StateGraph implementing query → retrieve → reason → generate
- Modular nodes for each pipeline stage
- State management using plain dicts
- Optional reranking with FlashRank
- Quality scoring and follow-up generation

### Core Services (`core/`)
- Graph database operations (Neo4j driver wrapper)
- Embedding manager with async batching
- Entity extraction with NetworkX deduplication
- Graph clustering with Leiden algorithm
- Caching layer (entity labels, embeddings, retrieval, responses)
- Singletons for shared resources

### Background Jobs (`api/`)
- Job manager for async task execution
- Document reindexing and classification
- Entity extraction jobs
- Progress tracking and status reporting

## Key Components

### API Layer
**File**: `api/main.py`, `api/routers/*.py`

FastAPI application with modular routers. Each router handles a specific domain:
- `chat.py` - Query handling and SSE streaming
- `documents.py` - Document operations, metadata, chunks, entities, similarities
- `database.py` - Stats, upload, reindexing, cache metrics
- `history.py` - Conversation persistence and retrieval
- `jobs.py` - Background job management

### RAG Pipeline
**File**: `rag/graph_rag.py`

LangGraph state machine with stages:
1. Query analysis - Parse query, extract filters
2. Retrieval - Hybrid search (vectors + entities)
3. Graph reasoning - Multi-hop expansion
4. Reranking - Optional FlashRank cross-encoder
5. Generation - LLM streaming with sources

State flows through nodes, each appending metadata.

### Hybrid Retriever
**File**: `rag/retriever.py`

Combines multiple signals:
- Vector search via Neo4j embeddings
- Entity-aware candidate expansion
- Graph traversal (SIMILAR_TO, RELATED_TO edges)
- Configurable weights and thresholds
- Optional caching with TTL

### Graph Database
**File**: `core/graph_db.py`

Neo4j driver wrapper providing:
- Connection pooling (50 connections)
- Parameterized queries
- Batch operations (UNWIND for bulk inserts)
- Entity label caching
- Document/chunk/entity CRUD operations

### Embeddings Manager
**File**: `core/embeddings.py`

Async embedding generation with:
- OpenAI or local provider support
- LRU caching by text + model hash
- Concurrency limits and rate limiting
- Batch processing
- Retry logic

### Entity Extraction
**File**: `core/entity_extraction.py`

LLM-based extraction:
- Structured output with type and description
- Optional gleaning (multi-pass for recall)
- NetworkX in-memory graph for deduplication
- Batch UNWIND persistence
- Async or sync embedding generation

### Clustering
**File**: `core/graph_clustering.py`

Leiden community detection:
- Loads entity projection from Neo4j
- Converts to igraph.Graph
- Runs modularity-based clustering
- Assigns community_id to entities
- Supports hierarchical levels

### Job Manager
**File**: `api/job_manager.py`

Background task coordinator:
- Job queue with priority
- Progress tracking
- Status reporting (pending/running/completed/failed)
- Job cancellation
- Reindex and classification workflows

## Configuration

Backend configuration via `config/settings.py` and environment variables:

```python
NEO4J_URI=bolt://neo4j:7687
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
ENABLE_CACHING=true
FLASHRANK_ENABLED=true
ENABLE_ENTITY_EXTRACTION=true
```

See [Environment Variables](07-configuration/environment-variables.md) for complete reference.

## Development

**Running backend locally:**
```bash
source .venv/bin/activate
python api/main.py
```

**Testing:**
```bash
pytest tests/integration/
```

**API docs:**
http://localhost:8000/docs (Swagger UI)

## Related Documentation

- [API Reference](06-api-reference)
- [Data Flows](05-data-flows)
- [Configuration](07-configuration)
- [Development Guide](09-development)
