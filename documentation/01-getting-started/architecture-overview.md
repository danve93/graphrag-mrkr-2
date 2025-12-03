# Architecture Overview

High-level architecture and design principles of the Amber platform.

## System Architecture

Amber implements a three-tier microservices architecture for graph-enhanced RAG:

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend Layer                        │
│  Next.js 14 + React + TypeScript + Zustand + Tailwind CSS   │
│  - Chat Interface  - Document View  - Graph Visualization   │
└─────────────────────────────────────────────────────────────┘
                              │
                         HTTP/SSE
                              │
┌─────────────────────────────────────────────────────────────┐
│                        Backend Layer                         │
│         FastAPI + Python 3.10+ + LangGraph + Pydantic       │
│  - API Routers  - RAG Pipeline  - Ingestion  - Services     │
└─────────────────────────────────────────────────────────────┘
                              │
                    Neo4j Bolt Protocol
                              │
┌─────────────────────────────────────────────────────────────┐
│                         Data Layer                          │
│              Neo4j 5.x Graph Database + Cypher              │
│  - Documents  - Chunks  - Entities  - Relationships         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Frontend (Next.js)

**Purpose**: User-facing interface for document interaction and chat

**Technology Stack**:
- Next.js 14 with App Router
- TypeScript for type safety
- Zustand for state management
- Tailwind CSS for styling
- Force-Graph 3D for graph visualization
- SSE client for streaming responses

**Key Modules**:
- `src/app/` - Page routes and layouts
- `src/components/` - Reusable UI components
- `src/lib/api.ts` - Backend API client
- `src/store/` - Global state management
- `src/types/` - TypeScript type definitions

**Communication**: HTTP REST + Server-Sent Events (SSE)

### Backend (FastAPI)

**Purpose**: Business logic, orchestration, and API layer

**Technology Stack**:
- FastAPI for REST endpoints
- LangGraph for RAG state machine
- Pydantic for data validation
- asyncio for concurrency
- Neo4j driver for graph access

**Key Modules**:
- `api/routers/` - REST endpoint definitions
- `api/services/` - Business logic services
- `rag/graph_rag.py` - LangGraph RAG pipeline
- `rag/retriever.py` - Hybrid retrieval logic
- `core/` - Shared utilities and services
- `ingestion/` - Document processing pipeline

**Communication**: Neo4j Bolt + OpenAI/Ollama APIs

### Database (Neo4j)

**Purpose**: Graph storage for documents, chunks, entities, and relationships

**Data Model**:
```cypher
(Document)-[:HAS_CHUNK]->(Chunk)
(Chunk)-[:CONTAINS_ENTITY]->(Entity)
(Chunk)-[:SIMILAR_TO {strength}]->(Chunk)
(Entity)-[:RELATED_TO {strength}]->(Entity)
(Entity)-[:IN_COMMUNITY {community_id}]->(Community)
```

**Capabilities**:
- Vector search via HNSW indexes
- Graph traversal via Cypher
- ACID transactions
- Connection pooling

## Design Principles

### 1. Graph-Enhanced RAG

Traditional RAG relies solely on vector similarity. Amber augments this with graph relationships:

**Vector Search**: Initial candidate retrieval by embedding similarity
```python
MATCH (c:Chunk)
CALL db.index.vector.queryNodes('chunk_embeddings', k, query_embedding)
YIELD node, score
RETURN node, score
```

**Graph Expansion**: Multi-hop reasoning through chunk similarities and entity relationships
```python
MATCH (c:Chunk)-[:SIMILAR_TO*1..2]-(related:Chunk)
MATCH (c)-[:CONTAINS_ENTITY]->()-[:RELATED_TO]->()<-[:CONTAINS_ENTITY]-(related)
RETURN DISTINCT related
```

**Result**: Context-aware retrieval that surfaces semantically and structurally related information.

### 2. Multi-Layer Caching

Performance optimization through strategic caching:

**Entity Label Cache** (TTL: 300s):
- Maps entity names to labels
- Reduces database lookups during expansion
- 70-80% hit rate

**Embedding Cache** (LRU):
- Caches embeddings by text+model hash
- Eliminates duplicate API calls
- 40-60% hit rate

**Retrieval Cache** (TTL: 60s):
- Caches query results
- Short TTL for data consistency
- 20-30% hit rate

**Response Cache** (TTL: 300s):
- Caches complete LLM responses
- Reduces latency for repeated queries
- Configurable via feature flag

### 3. Streaming-First Architecture

Real-time feedback through Server-Sent Events:

**Backend**: Token-by-token LLM generation
```python
async for chunk in llm.astream(prompt):
    yield {"type": "token", "content": chunk}
```

**Frontend**: Progressive rendering
```typescript
eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data)
  if (data.type === 'token') {
    appendToken(data.content)
  }
})
```

**Benefits**: Immediate user feedback, perceived performance improvement, early cancellation support.

### 4. Modular Pipeline (LangGraph)

RAG pipeline as a directed acyclic graph:

```
query_analysis → retrieval → graph_reasoning → reranking → generation
                                                              ↓
                                               quality_scoring + follow_ups
```

Each node is independently testable, configurable, and replaceable. State flows through a plain dict, enabling observability and debugging.

### 5. Asynchronous Processing

Non-blocking operations for I/O-bound tasks:

**Ingestion**: Background entity extraction
```python
async def process_document_async(file_path: str):
    chunks = await chunker.chunk_async(text)
    await embedder.embed_batch_async(chunks)
    asyncio.create_task(extract_entities_async(chunks))
```

**Retrieval**: Concurrent database queries
```python
results = await asyncio.gather(
    vector_search(query),
    graph_expansion(candidates),
    entity_search(query)
)
```

**Benefits**: Higher throughput, better resource utilization, responsive API.

### 6. Summary-First + Lazy-Load

UI optimization pattern:

**Summary**: Precomputed statistics (fast)
```cypher
MATCH (d:Document)
RETURN d.precomputed_chunk_count, d.precomputed_entity_count
```

**Detail**: Paginated on-demand loading (lazy)
```http
GET /api/documents/{id}/chunks?limit=50&offset=0
```

**Benefits**: Fast initial render, reduced memory usage, scalable to large documents.

## Data Flow

### Chat Query Flow

1. **User Input**: Message submitted via chat interface
2. **API Request**: POST to `/api/chat` with context parameters
3. **Query Analysis**: Extract filters, normalize query
4. **Retrieval**: Hybrid search (vector + graph)
5. **Reranking**: FlashRank scores (optional)
6. **Generation**: LLM generates response with sources
7. **Streaming**: SSE delivers tokens to frontend
8. **Quality Scoring**: Background quality evaluation
9. **Follow-ups**: Generated suggestion questions
10. **History**: Persisted to conversation store

### Document Ingestion Flow

1. **Upload**: File uploaded via API
2. **Conversion**: Format-specific loader extracts text
3. **Chunking**: Text segmented with overlap
4. **Embedding**: Chunks vectorized via API
5. **Entity Extraction**: LLM extracts entities (async)
6. **Graph Persistence**: Cypher UNWIND batch writes
7. **Similarity Calculation**: Chunk-chunk similarity edges
8. **Clustering**: Leiden community detection (optional)
9. **Indexing**: HNSW vector index updates
10. **Status Update**: UI polls processing status

## Integration Points

### External Services

**OpenAI API**:
- GPT-4/3.5 for generation
- text-embedding-3-small/large for vectors
- Configurable via `OPENAI_API_KEY`

**Ollama** (optional):
- Local LLM inference
- Local embeddings
- Configurable via `OLLAMA_BASE_URL`

**Marker** (optional):
- High-accuracy document conversion
- OCR for images/scanned PDFs
- Configurable via `MARKER_API_URL`

**FlashRank** (optional):
- Fast reranking model
- CPU/GPU execution
- Configurable via `FLASHRANK_ENABLED`

### Configuration Management

**Environment Variables** (`.env`):
- Database credentials
- API keys
- Feature flags

**Chat Tuning** (`config/chat_tuning_config.json`):
- Runtime model selection
- Retrieval parameters
- Generation settings

**RAG Tuning** (`config/rag_tuning_config.json`):
- Pipeline node parameters
- Expansion thresholds
- Caching settings

**Precedence**: Chat Tuning > RAG Tuning > Environment Variables > Defaults

## Deployment Models

### Docker Compose (Development/Single-Host)

```yaml
services:
  backend:
    build: .
    ports: ["8000:8000"]
    depends_on: [neo4j]
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
```

**Advantages**: Simple setup, easy teardown, consistent environments

### Kubernetes (Production/Scale)

```yaml
Deployment: backend (3 replicas)
StatefulSet: neo4j (1 primary, 2 read replicas)
Deployment: frontend (2 replicas)
Service: LoadBalancer/Ingress
PersistentVolume: neo4j-data
```

**Advantages**: High availability, horizontal scaling, rolling updates

### Bare Metal (Custom)

Direct installation with process managers (systemd/supervisor) and reverse proxy (Nginx/Traefik).

**Advantages**: Full control, no containerization overhead

## Security Architecture

### Authentication

**User Tokens**: UUID-based tokens stored in `data/job_user_token.txt`
- Generated on first startup
- Required for job submission endpoints
- Passed via `X-User-Token` header

### Network Security

**Port Exposure**:
- Frontend: 3000 (HTTP)
- Backend: 8000 (HTTP)
- Neo4j: 7474 (HTTP), 7687 (Bolt)

**Production**: Use reverse proxy with HTTPS, restrict Neo4j ports to internal network

### Data Security

**API Keys**: Stored in environment variables, never committed
**Neo4j Credentials**: Configurable via `.env`, use strong passwords
**Sensitive Data**: No PII storage in default configuration

## Performance Characteristics

### Latency Targets

**Chat Query** (end-to-end):
- Cold query: 2-5 seconds
- Cached query: 0.5-1 second
- Streaming start: <500ms

**Document Ingestion**:
- Small doc (10 pages): 5-10 seconds
- Large doc (1000 pages): 2-5 minutes
- Entity extraction: +50% time (async)

**Graph Expansion**:
- 1-hop: 50-100ms
- 2-hop: 200-500ms
- 3-hop: 1-2 seconds

### Scalability

**Vertical Limits** (single host):
- Documents: 10,000+
- Chunks: 1,000,000+
- Entities: 5,000,000+
- Concurrent users: 50+

**Horizontal Scaling**:
- Backend: Stateless, scale replicas
- Neo4j: Causal clustering with read replicas
- Frontend: CDN + multiple instances

## Related Documentation

- [Docker Setup](01-getting-started/docker-setup.md)
- [Local Development](01-getting-started/local-development.md)
- [Core Concepts](02-core-concepts)
- [Components](03-components)
