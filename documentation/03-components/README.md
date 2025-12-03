# Components

Technical reference for all Amber system components organized by architectural layer.

## Directory Structure

### [Backend](03-components/backend)
FastAPI application, RAG pipeline, retrieval engine, graph database integration, and background job management.

### [Ingestion](03-components/ingestion)
Document processing pipeline including format loaders, chunking strategies, embedding generation, and entity extraction.

### [Frontend](03-components/frontend)
Next.js application with chat interface, document explorer, graph visualization, and SSE streaming.

## Component Overview

### Backend Layer

The backend provides:
- RESTful API with FastAPI routers
- LangGraph-based RAG pipeline orchestration
- Hybrid retrieval combining vector and graph signals
- Neo4j driver for graph operations
- Embedding management with caching
- Entity extraction and deduplication
- Background job processing
- SSE streaming for real-time updates

Key files:
- `api/main.py` - Application entry point and lifecycle
- `rag/graph_rag.py` - LangGraph state machine pipeline
- `rag/retriever.py` - Hybrid retrieval implementation
- `core/graph_db.py` - Neo4j operations
- `core/embeddings.py` - Embedding manager
- `core/entity_extraction.py` - Entity extraction logic

### Ingestion Layer

The ingestion pipeline handles:
- Multi-format document conversion (PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, images)
- Optional Marker integration for high-accuracy PDF extraction
- Configurable chunking with overlap and provenance
- Async embedding generation with concurrency controls
- Entity extraction with optional gleaning (multi-pass)
- Quality scoring and filtering
- Batch persistence to Neo4j

Key files:
- `ingestion/document_processor.py` - Main processing coordinator
- `ingestion/loaders/` - Format-specific loaders
- `core/chunking.py` - Chunking strategy
- `core/quality_scorer.py` - Chunk quality assessment

### Frontend Layer

The frontend provides:
- Responsive chat interface with streaming responses
- Persistent conversation history
- Document upload and ingestion tracking
- Database explorer with pagination
- 3D graph visualization with community colors
- Chat tuning panel for runtime parameter control

Key files:
- `frontend/src/app/page.tsx` - Main application page
- `frontend/src/components/Chat/ChatInterface.tsx` - Chat UI with SSE
- `frontend/src/components/Document/DocumentView.tsx` - Document explorer
- `frontend/src/components/Graph/GraphView.tsx` - 3D graph visualization
- `frontend/src/lib/api.ts` - API client wrapper

## Component Interaction

```
User Request
    ↓
Frontend (Next.js)
    ↓ REST API
Backend API Layer (FastAPI)
    ↓
RAG Pipeline (LangGraph)
    ↓
Retriever ← Embeddings Cache
    ↓
Neo4j Graph Database
    ↑
Ingestion Pipeline
    ↑
Document Loaders
```

## Component Development

When modifying components:

1. **Backend**: Update `api/routers/` for endpoints, `rag/` for pipeline logic, `core/` for shared utilities
2. **Ingestion**: Update `ingestion/loaders/` for new formats, `ingestion/document_processor.py` for workflow changes
3. **Frontend**: Update `frontend/src/components/` for UI changes, `frontend/src/lib/api.ts` for API integration

## Testing Components

- **Backend**: `pytest tests/integration/test_*.py`
- **Ingestion**: `pytest tests/integration/test_full_ingestion_pipeline.py`
- **Frontend**: `cd frontend && npm run test`

See [Testing Guide](09-development/testing-backend.md) for detailed procedures.

## Related Documentation

- [Architecture Overview](01-getting-started/architecture-overview.md)
- [Data Flows](05-data-flows)
- [API Reference](06-api-reference)
- [Development Guide](09-development)
