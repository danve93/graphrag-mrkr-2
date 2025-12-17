# Amber

<img width="1280" height="500" alt="ChatGPT Image 17 dic 2025, 13_04_11" src="https://github.com/user-attachments/assets/e53987c0-72e7-47dc-a327-f7b53355fa7b" />
A production-ready Graph-Enhanced Retrieval-Augmented Generation (GraphRAG) platform that combines vector similarity search with knowledge graph reasoning to deliver contextual, sourced, and high-quality answers over document collections.

## Overview

Traditional RAG systems rely solely on embedding similarity to retrieve context. Amber augments this with graph-based reasoning that leverages:

- **Chunk similarity relationships** - Semantic connections between text segments
- **Entity co-occurrence** - Shared entities across documents
- **Multi-hop traversal** - Indirect relationships through graph paths
- **Community detection** - Clustered semantic neighborhoods

This graph-enhanced approach surfaces contextually relevant information that pure vector search might miss.

## Architecture

```
                         Frontend Layer
    ┌─────────────────────────────────────────────────────────────┐
    │    Next.js 14 + React + TypeScript + Zustand + Tailwind     │
    │    Chat Interface  |  Document View  |  Graph Visualization │
    └─────────────────────────────────────────────────────────────┘
                                  │
                             HTTP / SSE
                                  │
                         Backend Layer
    ┌─────────────────────────────────────────────────────────────┐
    │         FastAPI + Python 3.10+ + LangGraph + Pydantic       │
    │    API Routers  |  RAG Pipeline  |  Ingestion  |  Services  │
    └─────────────────────────────────────────────────────────────┘
                                  │
                        Neo4j Bolt Protocol
                                  │
                          Data Layer
    ┌─────────────────────────────────────────────────────────────┐
    │              Neo4j 5.x Graph Database + Cypher              │
    │    Documents  |  Chunks  |  Entities  |  Relationships      │
    └─────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Next.js 14, TypeScript, Zustand, Tailwind CSS, Force-Graph 3D |
| **Backend** | FastAPI, LangGraph, Pydantic, asyncio, Neo4j driver |
| **Database** | Neo4j 5.x with HNSW vector indexes |
| **AI/ML** | OpenAI (GPT-4/3.5, embeddings), Ollama (optional), FlashRank reranking |
| **Document Processing** | Multi-format loaders, Marker (optional OCR) |

## Features

### Core GraphRAG Capabilities

| Feature | Description |
|---------|-------------|
| **Hybrid Retrieval** | Combined vector search (70%) and entity search (30%) with configurable weights |
| **Graph Expansion** | Multi-hop traversal (1-3 hops) through similarity and entity relationship edges |
| **FlashRank Reranking** | Cross-encoder reranking with 10-15% relevance improvement |
| **Leiden Community Detection** | Automatic entity clustering into semantic communities |
| **Entity Extraction with Gleaning** | Multi-pass extraction with 30-40% recall improvement |
| **Quality Scoring** | Chunk quality assessment for filtering low-quality content |

### Advanced Intelligence

| Feature | Description |
|---------|-------------|
| **Query Routing** | LLM-based query classification with semantic caching (30-50% latency reduction) |
| **Adaptive Routing** | Feedback-based weight learning that improves 10-15% after 50+ samples |
| **Category-Specific Prompts** | 10 pre-configured templates with tailored retrieval strategies |
| **Smart Consolidation** | Category-aware ranking with semantic deduplication |
| **Structured KG Queries** | Text-to-Cypher translation (60-80% faster for aggregation queries) |

### Platform Features

| Feature | Description |
|---------|-------------|
| **Chat Tuning** | Runtime controls for model selection and retrieval parameters |
| **Document Classification** | Rule-based and LLM-based automatic labeling |
| **Incremental Updates** | Content-hash diffing for efficient document updates (only changed chunks reprocessed) |
| **Multi-Layer Caching** | Persistent embedding/response cache (disk), in-memory entity/retrieval cache |
| **External User Integration** | API key authentication with minimal chat interface |
| **SSE Streaming** | Real-time token streaming (20-50ms per token) |

## Quick Start

### Docker Compose (Recommended)

Run the full stack locally:

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Rebuild after changes
docker compose up -d --build
```

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

### Local Development

**Backend:**

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and credentials

# Start backend
python api/main.py
```

**Frontend:**

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

## GraphRAG Pipeline

The RAG pipeline is implemented as a LangGraph StateGraph with the following stages:

```
Query Analysis     Parse query, extract filters, normalize
       │
       ▼
   Retrieval       Hybrid vector + entity search
       │
       ▼
Graph Reasoning    Multi-hop expansion via chunk/entity relationships
       │
       ▼
   Reranking       Optional FlashRank cross-encoder (if enabled)
       │
       ▼
  Generation       LLM generates response with source citations
       │
       ▼
Quality Scoring    Evaluate response quality (background)
       │
       ▼
  Follow-ups       Generate suggested questions (background)
```

Each node is independently testable, configurable, and replaceable. State flows through a typed dictionary, enabling observability and debugging.

## Knowledge Graph Schema

```
Document ──[HAS_CHUNK]──> Chunk ──[SIMILAR_TO]──> Chunk
                            │
                    [CONTAINS_ENTITY]
                            │
                            ▼
                         Entity ──[RELATED_TO]──> Entity
                            │
                      [IN_COMMUNITY]
                            │
                            ▼
                        Community
```

| Relationship | Description | Key Properties |
|-------------|-------------|----------------|
| `HAS_CHUNK` | Document to Chunk | `chunk_index` |
| `CONTAINS_ENTITY` | Chunk to Entity | `mention_count` |
| `SIMILAR_TO` | Chunk to Chunk | `strength` (0-1 cosine similarity) |
| `RELATED_TO` | Entity to Entity | `strength`, `co_occurrence` |
| `IN_COMMUNITY` | Entity to Community | - |

## Configuration

Configuration follows a clear precedence hierarchy:

```
Chat Tuning > RAG Tuning > Environment Variables > Defaults
```

### Key Environment Variables

```bash
# LLM and Embeddings
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# Feature Toggles
ENABLE_ENTITY_EXTRACTION=true
ENABLE_CLUSTERING=true
FLASHRANK_ENABLED=true
ENABLE_CACHING=true
CACHE_TYPE=disk
ENABLE_QUERY_ROUTING=false
ENABLE_STRUCTURED_KG=true
ENABLE_ADAPTIVE_ROUTING=true
```

### Chat Tuning vs RAG Tuning

**Chat Tuning** (`/api/chat-tuning`) - Affects query execution at runtime:
- Retrieval weights, top_k, expansion depth
- LLM temperature, max tokens, model selection
- Changes take effect immediately

**RAG Tuning** (`/api/rag-tuning`) - Affects document ingestion:
- Entity extraction, PDF processing, clustering
- Chunking strategy, embedding model
- Requires reindexing to apply to existing documents

## Performance

### Latency Targets

| Operation | Cold | Cached |
|-----------|------|--------|
| Chat Query (end-to-end) | 2-5 seconds | 0.5-1 second |
| Streaming Start | <500ms | <200ms |
| 1-hop Expansion | 50-100ms | - |
| 2-hop Expansion | 200-500ms | - |

### Ingestion Performance

| Document Size | Time |
|---------------|------|
| Small (10 pages) | 5-10 seconds |
| Large (1000 pages) | 2-5 minutes |
| + Entity Extraction | +50% (async) |

### Scalability (Single Host)

- Documents: 10,000+
- Chunks: 1,000,000+
- Entities: 5,000,000+
- Concurrent Users: 50+

## API Reference

### Chat and Retrieval

| Endpoint | Description |
|----------|-------------|
| `POST /api/chat/query` | Chat query with structured response |
| `POST /api/chat/stream` | SSE streaming tokens |
| `POST /api/chat/follow-ups` | Generate follow-up questions |

### Documents

| Endpoint | Description |
|----------|-------------|
| `GET /api/documents` | List documents |
| `GET /api/documents/{id}` | Document metadata and analytics |
| `POST /api/database/upload` | Upload and ingest document |
| `PUT /api/documents/{id}` | Incremental update (only changed chunks reprocessed) |
| `DELETE /api/database/documents/{id}` | Delete document |

### Configuration

| Endpoint | Description |
|----------|-------------|
| `GET /api/chat-tuning/config/values` | Current retrieval tuning values |
| `GET /api/rag-tuning/config/values` | Current ingestion tuning values |
| `GET /api/database/stats` | Database statistics |
| `GET /api/database/cache-stats` | Cache performance metrics |

### Structured Queries

| Endpoint | Description |
|----------|-------------|
| `POST /api/structured-kg/execute` | Execute Text-to-Cypher query |
| `GET /api/structured-kg/schema` | Get graph schema |

For the complete API reference, see the interactive documentation at `/docs`.

## Document Processing

### Supported Formats

- PDF (with optional Marker OCR)
- DOCX, PPTX, XLSX
- TXT, Markdown
- CSV
- Images (with Marker)

### Ingestion Pipeline

1. Format detection and loader selection
2. Text extraction with format-specific processing
3. Chunking with configurable size and overlap
4. Async embedding generation with caching
5. LLM-based entity extraction with optional gleaning
6. Quality scoring and filtering
7. Batch persistence to Neo4j
8. Similarity edge calculation
9. Optional Leiden community detection

## Testing

```bash
# All tests
pytest tests/

# By category
pytest tests/unit/           # Fast, isolated
pytest tests/integration/    # Requires Neo4j
pytest tests/e2e/            # Full pipeline

# Specific tests
pytest tests/integration/test_chat_pipeline.py -v
pytest tests/integration/test_full_ingestion_pipeline.py -v

# With parallel execution
pytest tests/ -n auto
```

### E2E with Docker

```bash
# Start Neo4j and run E2E tests
make e2e-local

# Or use Docker Compose
make e2e-dc
```

## Project Structure

```
amber/
├── api/                    # FastAPI routers and services
│   ├── routers/            # REST endpoint definitions
│   └── services/           # Business logic
├── core/                   # Shared utilities
│   ├── embeddings.py       # Embedding manager
│   ├── graph_db.py         # Neo4j operations
│   └── entity_extraction.py
├── rag/                    # RAG pipeline
│   ├── graph_rag.py        # LangGraph state machine
│   ├── retriever.py        # Hybrid retrieval
│   └── nodes/              # Pipeline nodes
├── ingestion/              # Document processing
│   ├── document_processor.py
│   └── loaders/            # Format-specific loaders
├── frontend/               # Next.js application
│   ├── src/app/            # Page routes
│   ├── src/components/     # UI components
│   └── src/lib/api.ts      # Backend API client
├── config/                 # Configuration files
├── scripts/                # Utility scripts
├── tests/                  # Test suite
└── documentation/          # Detailed documentation
```

## Documentation

Comprehensive documentation is available in the `documentation/` directory:

1. **Getting Started** - Architecture, setup, configuration
2. **Core Concepts** - Data model, pipeline, caching, retrieval strategies
3. **Components** - Backend, frontend, ingestion implementation details
4. **Features** - Detailed documentation for 20+ features
5. **Data Flows** - End-to-end traces of key operations
6. **API Reference** - REST endpoint documentation
7. **Configuration** - Complete parameter reference
8. **Operations** - Monitoring and maintenance guides
9. **Development** - Testing and feature development guides
10. **Scripts** - Utility scripts and CLI tools

## Deployment

Docker Compose is recommended for local and demo deployments. For production:

- Deploy services behind a load balancer with HTTPS
- Ensure Neo4j has adequate memory (8GB+ recommended)
- Install the GDS plugin when clustering is enabled
- Use environment variables or secrets management for credentials
- Configure appropriate connection pool sizes

## Contributing

1. Fork the repository and create a feature branch
2. Run tests and linters locally (`pytest`, `ruff check .`, `black .`)
3. Open a pull request with description and test results

## License

This project is licensed under the MIT License. See [LICENSE.md](./LICENSE.md) for details.
