# Amber

<img width="1280" height="640" alt="amber_avatar" src="https://github.com/user-attachments/assets/d86b3e7f-ca0b-488f-8ce6-9090bdb2a6a9" />

A production-ready Graph-Enhanced Retrieval-Augmented Generation (GraphRAG) platform that combines vector similarity search with knowledge graph reasoning to deliver contextual, sourced, and high-quality answers over document collections.

## Overview

Traditional RAG systems rely solely on embedding similarity to retrieve context. Amber augments this with graph-based reasoning that leverages:

- **Chunk similarity relationships** - Semantic connections between text segments
- **Entity co-occurrence** - Shared entities across documents
- **Multi-hop traversal** - Indirect relationships through graph paths
- **Community detection** - Clustered semantic neighborhoods

This graph-enhanced approach surfaces contextually relevant information that pure vector search might miss.

## Architecture

<img width="640" alt="architecture" src="https://github.com/user-attachments/assets/fcc010b5-5cb0-4e4c-be11-56ff12b87949" />

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
| **AI/ML** | OpenAI (GPT-4/3.5, embeddings), Google Gemini (optional), Ollama (optional), FlashRank reranking |
| **Document Processing** | Multi-format loaders, Docling (optional, PDF/DOCX/PPTX/XLSX/HTML), Marker (optional OCR), HTML heading chunker |

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
| **Token-Aware Chunking** | HTML heading chunker + Docling hybrid with tunable token budgets |

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
| **Automatic Orphan Cleanup** | Startup cleanup of disconnected chunks/entities with configurable grace period |
| **Multi-Layer Caching** | Persistent embedding/response cache (disk), in-memory entity/retrieval cache |
| **Secure API Key Management** | SHA-256 hashed API keys with tenant isolation and database-backed authentication |
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

**First Time Setup (Important):**
Since v2.1.0, admin keys are no longer set via environment variables. You must generate the first admin key manually:

```bash
# Generate admin key for the first login
docker compose exec backend python scripts/generate_admin_key.py
```
Copy the generated key (starts with `sk-...`) and use it to log in.

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

**Generate Admin Key:**
In a separate terminal, while the backend and Neo4j are running:
```bash
python scripts/generate_admin_key.py
```

**Frontend:**

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

### Recent Updates

#### v2.2.0 (December 2025)

**New Features:**
- **Docling Integration**: Optional Docling library support for state-of-the-art document conversion (PDF, DOCX, PPTX, XLSX, HTML, images) with layout analysis and structure preservation
- **LLM Token Usage Metrics**: Comprehensive analytics dashboard tracking token usage, costs (USD/EUR), breakdowns by operation/provider/model/conversation, time trends, and efficiency metrics
- **Selective Database Clearing**: Granular control to clear Knowledge Base or Conversation History independently
- **Google Gemini Support**: Added Gemini as LLM provider option with full configuration support
- **HTML Heading Chunker**: New strategy for HTML documents with semantic structure and heading path extraction

**Infrastructure:**
- **Token Management Enhancements**: Comprehensive token tracking and context management
  - New `token_counter.py` utility and enhanced `token_manager.py` with intelligent context splitting
  - Standardized `include_usage` parameter across all LLM providers
  - Updated model context sizes for 2024-2025 models

**Configuration:**
- **TruLens Toggle Persistence**: TruLens state now persists to `config.yaml` and defaults to disabled

**UI Improvements:**
- **Search UI Enhancement**: Added fuzzy search to Chat and RAG Tuning panels
- **Bottom Panel Padding**: Consistent padding across all scrollable panels

#### v2.1.1 (December 2024)

**Bug Fixes:**
- **Orphaned Chunks Resolution**: Fixed RAG retrieval returning 0 results due to orphaned chunks not connected to documents
- **Document Update Progress**: Fixed progress bar stuck at 5% during document updates with proper status reporting
- **UI Cleanup**: Removed redundant "Processing in progress..." banner from Database panel

**New Features:**
- **Automatic Orphan Cleanup**: Startup cleanup of orphaned chunks and entities with configurable grace period
  - Configure via `ENABLE_ORPHAN_CLEANUP_ON_STARTUP` (default: true)
  - Grace period via `ORPHAN_CLEANUP_GRACE_PERIOD_MINUTES` (default: 5 minutes)
- **Manual Cleanup API**: New `POST /api/database/cleanup-orphans` endpoint for on-demand cleanup
- **Cleanup UI Button**: Added "Cleanup" button in Database panel toolbar with confirmation dialog

#### v2.1.0 (December 2024)

**Major Security Improvements:**
- **Static Admin Token Removed**: Authentication now strictly enforces database-backed API keys (no more `JOBS_ADMIN_TOKEN`)
- **API Key Hashing**: Implemented SHA-256 hashing for secure API key storage
- **Tenant Isolation**: Enforced one active API key per user
- **Session Security**: Configured `Secure` and `HttpOnly` flags for admin cookies
- **Access Control**: Fixed broken access control on conversation endpoints

**Stability & Reliability:**
- **Persistent Processing State**: Refactored in-memory state to persist in Neo4j (prevents data loss on restart)
- **Community Detection**: Replaced Neo4j GDS dependency with `igraph` for better stability
- **Settings Synchronization**: Full alignment of RAG tuning parameters, LLM overrides, and static matching thresholds
- **Memory Leak Fixes**: Added bounds to routing cache and adaptive router feedback history

**Logic & Correctness:**
- **Progress Calculation**: Fixed "jumping" progress bars with proper stage interpolation
- **Metadata Merging**: Changed to map merging instead of complete overwrite
- **Orphan Detection**: Improved algorithm for detecting disconnected components
- **Async Improvements**: Fixed unsafe `asyncio.run()` nesting

50+ issues addressed from the December 2024 audit. See [`CHANGELOG.md`](./CHANGELOG.md) for complete details.

## GraphRAG Pipeline

<img width="640" alt="pipeline" src="https://github.com/user-attachments/assets/f0d6b915-16c8-472f-bc44-9b7c24128246" />

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

<img width="640" alt="api" src="https://github.com/user-attachments/assets/807e87f8-b8bd-4f20-b376-7fa690f7e4fc" />

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
| `POST /api/database/cleanup-orphans` | Manual cleanup of orphaned chunks and entities |

### Configuration

| Endpoint | Description |
|----------|-------------|
| `GET /api/chat-tuning/config/values` | Current retrieval tuning values |
| `GET /api/rag-tuning/config/values` | Current ingestion tuning values |
| `GET /api/database/stats` | Database statistics (includes orphan counts) |
| `GET /api/database/cache-stats` | Cache performance metrics |

**Note:** Admin endpoints now support pagination via `limit` and `offset` query parameters (v2.1.0).

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


<img width="640" alt="documentation" src="https://github.com/user-attachments/assets/17fdc7ef-467f-4fde-a3e7-53093c4514ef" />

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
