# Data Flows

End-to-end traces of key operations through the Amber system.

## Contents

- [README](05-data-flows/README.md) - Data flows overview
- [Chat Query Flow](05-data-flows/chat-query-flow.md) - Complete request lifecycle from frontend to response
- [Ingestion Flow](05-data-flows/document-ingestion-flow.md) - Document upload through persistence
- [Retrieval Flow](05-data-flows/graph-expansion-flow.md) - Query to hybrid retrieval to reranking
- [Entity Deduplication](05-data-flows/entity-extraction-flow.md) - NetworkX in-memory graph merging
- [Streaming SSE](05-data-flows/streaming-sse-flow.md) - Server-Sent Events token streaming

## Overview

This section traces data through the system to show:
- Which components are involved
- What transformations occur
- Where state is maintained
- How errors are handled
- Performance characteristics

## Key Flows

### Chat Query Flow

Complete request path from user query to streamed response:

```
Frontend Input
    ↓
POST /api/chat/query
    ↓
RAG Pipeline (LangGraph)
    ├─ Query Analysis
    ├─ Hybrid Retrieval
    ├─ Graph Expansion
    ├─ Optional Reranking
    └─ LLM Generation
    ↓
SSE Stream Events
    ├─ stage: query_analysis
    ├─ stage: retrieval  
    ├─ stage: generation
    ├─ token: <text>
    ├─ sources: [...]
    └─ quality_score: 0.85
    ↓
Frontend Display
```

**Timeline**: 1-5 seconds depending on query complexity

See [Chat Query Flow](05-data-flows/chat-query-flow.md) for detailed sequence diagram.

### Ingestion Flow

Document upload through graph persistence:

```
File Upload
    ↓
POST /api/database/upload
    ↓
DocumentProcessor
    ├─ Format Detection
    ├─ Loader Selection
    ├─ Text Extraction
    ├─ Chunking
    ├─ Embedding Generation
    ├─ Entity Extraction
    │   ├─ LLM Call
    │   ├─ Optional Gleaning
    │   └─ NetworkX Deduplication
    ├─ Quality Scoring
    └─ Batch Persistence
    ↓
Neo4j Graph
    ├─ Document Node
    ├─ Chunk Nodes
    ├─ Entity Nodes
    └─ Relationships
    ↓
Background: Similarity Calculation
    ↓
Background: Optional Clustering
```

**Timeline**: 10 seconds - 20 minutes depending on document size and features enabled

See [Ingestion Flow](05-data-flows/document-ingestion-flow.md) for detailed breakdown.

### Retrieval Flow

Query through hybrid retrieval:

```
User Query
    ↓
Query Embedding
    ↓ (cache check)
Embedding API or Cache
    ↓
Vector Search (Neo4j)
    ├─ Chunk Similarity (cosine)
    └─ Entity Mention Search
    ↓
Graph Expansion (Optional)
    ├─ Traverse SIMILAR_TO edges
    ├─ Traverse RELATED_TO edges  
    └─ Multi-hop reasoning
    ↓
Hybrid Scoring
    ├─ chunk_weight * chunk_score
    ├─ entity_weight * entity_score
    └─ path_weight * path_score
    ↓
Reranking (Optional)
    ├─ FlashRank Cross-Encoder
    └─ Blend with hybrid scores
    ↓
Top-K Results
```

**Latency Breakdown**:
- Embedding: 100-200ms (or <1ms if cached)
- Vector search: 50-100ms
- Graph expansion: 100-300ms (depends on depth)
- Reranking: 50-100ms
- **Total**: 300-700ms typical

See [Retrieval Flow](05-data-flows/graph-expansion-flow.md) for algorithm details.

### Entity Deduplication Flow

In-memory graph merging before persistence:

```
Entity Extraction Results
    ↓
EntityGraph (NetworkX)
    ├─ Add nodes (canonical key)
    ├─ Merge duplicate entities
    │   ├─ Concatenate descriptions
    │   └─ Track source chunks
    ├─ Add edges (relationships)
    └─ Sum edge weights
    ↓
Export to Cypher
    ├─ UNWIND nodes batch
    ├─ UNWIND edges batch
    └─ Single transaction
    ↓
Neo4j Persistence
```

**Benefits**:
- 22% fewer duplicate entities
- 10x faster persistence (batch vs individual inserts)
- Cleaner knowledge graph

See [Entity Deduplication](05-data-flows/entity-extraction-flow.md) for NetworkX implementation.

### SSE Streaming Flow

Real-time token delivery:

```
LLM API Stream
    ↓
Generator Function
    ├─ Decode chunks
    ├─ Parse tokens
    └─ Format SSE events
    ↓
HTTP Response (text/event-stream)
    ├─ data: {"type": "stage", ...}
    ├─ data: {"type": "token", ...}
    └─ data: {"type": "sources", ...}
    ↓
Frontend EventSource
    ├─ Parse event data
    ├─ Update UI state
    └─ Append to message
    ↓
User sees incremental response
```

**Characteristics**:
- Streaming starts immediately (no buffering)
- Tokens arrive every 20-50ms
- Sources sent after generation completes
- Connection closed on completion or error

See [Streaming SSE](05-data-flows/streaming-sse-flow.md) for implementation details.

## Observability

Each flow includes:
- **Logging**: Structured logs at INFO level
- **Metrics**: Performance timings and cache hit rates
- **Tracing**: Stage events in pipeline responses
- **Error Context**: Stack traces and state snapshots

Monitor flows via:
- Backend logs: `docker compose logs backend`
- Cache metrics: `GET /api/database/cache-stats`
- Pipeline stages: SSE `stage` events
- Database stats: `GET /api/database/stats`

## Performance Characteristics

| Flow | Typical Latency | Bottlenecks | Optimizations |
|------|----------------|-------------|---------------|
| Chat Query | 1-5s | LLM generation | Streaming, caching |
| Ingestion | 10s-20m | Entity extraction | Async, batching |
| Retrieval | 300-700ms | Graph expansion | Caching, depth limits |
| Deduplication | <100ms | Graph operations | NetworkX in-memory |
| SSE Streaming | 20-50ms/token | Network latency | Compression, keep-alive |

## Error Handling

Flows implement graceful degradation:
- **Retrieval**: Falls back to vector-only if graph expansion fails
- **Entity Extraction**: Continues with chunks if extraction fails
- **Reranking**: Uses hybrid scores if reranker unavailable
- **Streaming**: Sends error event and closes connection

## Testing Flows

```bash
pytest tests/integration/test_chat_pipeline.py
pytest tests/integration/test_full_ingestion_pipeline.py
pytest tests/e2e/test_full_pipeline.py
```

## Related Documentation

- [Backend Components](03-components/backend)
- [API Reference](06-api-reference)
- [Performance Tuning](08-operations/monitoring.md)
