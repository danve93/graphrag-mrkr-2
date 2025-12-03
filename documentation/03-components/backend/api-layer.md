# API Layer

FastAPI-based REST API serving as the backend entry point.

## Overview

The API layer provides HTTP endpoints for all Amber functionality including chat, document management, database operations, and job management. Built with FastAPI, it offers automatic OpenAPI documentation, request validation via Pydantic, and async request handling.

**Framework**: FastAPI 0.104+
**Location**: `api/`
**Entry Point**: `api/main.py`

## Application Structure

```
api/
├── main.py              # Application initialization and lifecycle
├── auth.py              # User token authentication
├── models.py            # Pydantic request/response models
├── job_manager.py       # In-memory job tracking
├── redis_job_manager.py # Redis-based job tracking (optional)
├── reindex_manager.py   # Reindexing workflow coordination
├── reindex_tasks.py     # Background reindexing tasks
├── routers/             # API route handlers
│   ├── chat.py          # Chat and RAG endpoints
│   ├── documents.py     # Document CRUD and upload
│   ├── database.py      # Database stats and operations
│   ├── history.py       # Conversation history
│   ├── classification.py # Document classification
│   ├── chat_tuning.py   # Runtime parameter tuning
│   └── jobs.py          # Job status and management
├── services/            # Business logic
│   ├── chat_service.py  # Chat orchestration
│   ├── history_service.py # History persistence
│   └── follow_up_generator.py # Follow-up questions
└── utils/               # Utilities
    └── file_handler.py  # File operations
```

## Application Lifecycle

### Startup

**File**: `api/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Starting Amber API...")
    
    # Initialize user token
    token_manager.ensure_user_token()
    
    # Log provider configuration
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    
    # Optional: Prewarm FlashRank
    if settings.flashrank_enabled:
        try:
            from rag.rerankers.flashrank_reranker import FlashRankReranker
            _ = FlashRankReranker()
            logger.info("FlashRank prewarmed successfully")
        except Exception as e:
            logger.warning(f"FlashRank prewarm failed: {e}")
    
    yield  # Application runs
    
    # Shutdown tasks
    logger.info("Shutting down Amber API...")
    # Close database connections, cleanup resources

app = FastAPI(
    title="Amber API",
    version="1.0.0",
    lifespan=lifespan
)
```

### Router Registration

```python
from api.routers import (
    chat, documents, database, history,
    classification, chat_tuning, jobs
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(classification.router, prefix="/api/classification", tags=["classification"])
app.include_router(chat_tuning.router, prefix="/api/chat-tuning", tags=["chat-tuning"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
```

### CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://frontend:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Request/Response Models

**File**: `api/models.py`

### Chat Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context_documents: List[str] = Field(default_factory=list, description="Document filter")
    
    # RAG parameters (runtime overrides)
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    max_expansion_depth: Optional[int] = Field(None, ge=0, le=3)
    max_expanded_chunks: Optional[int] = Field(None, ge=1, le=100)
    retrieval_mode: Optional[str] = Field(None, pattern="^(vector|hybrid)$")

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    sources: Optional[List[Source]] = None
    quality_score: Optional[float] = None
    follow_up_questions: Optional[List[str]] = None
    timestamp: Optional[str] = None

class Source(BaseModel):
    chunk_id: str
    text: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    score: float
```

### Document Models

```python
class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str = "processing"
    message: str

class DocumentMetadata(BaseModel):
    id: str
    filename: str
    file_path: str
    file_type: str
    file_size: int
    title: str
    created_at: str
    page_count: Optional[int]
    word_count: Optional[int]
    precomputed_chunk_count: int
    precomputed_entity_count: int
    precomputed_community_count: int
    precomputed_similarity_count: int

class ChunksPaginatedResponse(BaseModel):
    document_id: str
    total: int
    limit: int
    offset: int
    has_more: bool
    chunks: List[Dict]

class EntitySummaryResponse(BaseModel):
    document_id: str
    total: int
    groups: List[Dict]  # [{type: str, count: int}]
```

### Job Models

```python
class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
```

## Router Endpoints

### Chat Router

**File**: `api/routers/chat.py`

```python
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Stream chat response using RAG pipeline.
    
    Returns SSE stream with:
    - stage: Pipeline stage updates
    - token: Generated tokens
    - sources: Retrieved sources
    - quality_score: Response quality
    - follow_ups: Suggested questions
    """
    try:
        # Initialize RAG state
        state = {
            "query": request.message,
            "session_id": request.session_id or str(uuid.uuid4()),
            "context_documents": request.context_documents,
            "llm_model": request.llm_model,
            "temperature": request.temperature,
            "top_k": request.top_k,
            # ... other parameters
        }
        
        # Stream RAG pipeline execution
        async def generate():
            async for chunk in rag_pipeline.astream(state):
                if "stage" in chunk:
                    yield f"data: {json.dumps({'type': 'stage', 'stage': chunk['stage']})}\n\n"
                elif "token" in chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk['token']})}\n\n"
                elif "sources" in chunk:
                    yield f"data: {json.dumps({'type': 'sources', 'sources': chunk['sources']})}\n\n"
                # ... other chunk types
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### Documents Router

**File**: `api/routers/documents.py`

```python
from fastapi import APIRouter, UploadFile, File, Query

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document."""
    try:
        # Save uploaded file
        file_path = await save_upload(file)
        
        # Process document asynchronously
        job_id = await document_processor.process_document_async(file_path)
        
        return DocumentUploadResponse(
            document_id=job_id,
            filename=file.filename,
            status="processing",
            message="Document processing started"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_documents():
    """List all documents with metadata."""
    documents = await graph_db.get_all_documents()
    return documents

@router.get("/{document_id}")
async def get_document(document_id: str):
    """Get document metadata and precomputed stats."""
    doc = await graph_db.get_document_details(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@router.get("/{document_id}/chunks")
async def get_document_chunks_paginated(
    document_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get document chunks with pagination."""
    chunks = await graph_db.get_chunks_paginated(document_id, limit, offset)
    total = await graph_db.count_document_chunks(document_id)
    
    return ChunksPaginatedResponse(
        document_id=document_id,
        total=total,
        limit=limit,
        offset=offset,
        has_more=(offset + limit < total),
        chunks=chunks
    )

@router.get("/{document_id}/entity-summary")
async def get_entity_summary(document_id: str):
    """Get aggregated entity type counts."""
    summary = await graph_db.get_entity_summary(document_id)
    return EntitySummaryResponse(
        document_id=document_id,
        total=summary["total"],
        groups=summary["groups"]
    )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete document and cascade delete chunks/entities."""
    success = await graph_db.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Clear relevant caches
    cache_manager.retrieval_cache.clear()
    cache_manager.response_cache.clear()
    
    return {"status": "success", "document_id": document_id}
```

### Database Router

**File**: `api/routers/database.py`

```python
router = APIRouter()

@router.get("/stats")
async def get_database_stats():
    """Get database statistics."""
    stats = await graph_db.get_database_stats()
    return stats

@router.get("/cache-stats")
async def get_cache_stats():
    """Get cache performance metrics."""
    from core.cache_metrics import cache_metrics
    return cache_metrics.get_stats()

@router.post("/clear-cache")
async def clear_cache(cache_name: str = "all"):
    """Clear specified cache or all caches."""
    if cache_name == "all":
        cache_manager.clear_all()
    elif cache_name == "entity_label":
        cache_manager.entity_label_cache.clear()
    elif cache_name == "embedding":
        cache_manager.embedding_cache.clear()
    elif cache_name == "retrieval":
        cache_manager.retrieval_cache.clear()
    elif cache_name == "response":
        cache_manager.response_cache.clear()
    else:
        raise HTTPException(status_code=400, detail="Invalid cache name")
    
    return {"status": "success", "cache_name": cache_name}
```

## Authentication

**File**: `api/auth.py`

```python
from fastapi import Header, HTTPException
from core.token_manager import token_manager

async def verify_user_token(x_user_token: str = Header(...)):
    """Verify user token for protected endpoints."""
    if not token_manager.verify_token(x_user_token):
        raise HTTPException(
            status_code=401,
            detail="Invalid user token"
        )
    return x_user_token
```

**Usage**:
```python
from fastapi import Depends
from api.auth import verify_user_token

@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(verify_user_token)])
async def cancel_job(job_id: str):
    # Only accessible with valid token
    pass
```

## Error Handling

### Global Exception Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An error occurred"
        }
    )
```

### Custom Exceptions

```python
class DocumentNotFoundError(Exception):
    pass

class InsufficientChunksError(Exception):
    pass

@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc)})
```

## Streaming SSE

### Event Types

**stage**: Pipeline stage update
```json
{"type": "stage", "stage": "retrieval"}
```

**token**: Generated token
```json
{"type": "token", "content": "The backup"}
```

**sources**: Retrieved sources
```json
{
  "type": "sources",
  "sources": [
    {
      "chunk_id": "abc123",
      "text": "Backup procedure...",
      "document_name": "VMware Guide",
      "score": 0.92
    }
  ]
}
```

**quality_score**: Response quality
```json
{"type": "quality_score", "score": 0.87}
```

**follow_ups**: Suggested questions
```json
{
  "type": "follow_ups",
  "questions": ["How often should backups run?", "What is the retention policy?"]
}
```

**error**: Error occurred
```json
{"type": "error", "message": "Retrieval failed"}
```

### Client Consumption

**JavaScript/TypeScript**:
```typescript
const eventSource = new EventSource('/api/chat', {
  method: 'POST',
  body: JSON.stringify({message: 'What is the backup procedure?'})
});

eventSource.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'token':
      appendToken(data.content);
      break;
    case 'sources':
      displaySources(data.sources);
      break;
    case 'quality_score':
      showQualityScore(data.score);
      break;
  }
});
```

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_documents():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_upload_document():
    with open("test.pdf", "rb") as f:
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    assert response.status_code == 200
    assert "document_id" in response.json()
```

### Integration Tests

```bash
pytest tests/integration/test_api_routers.py -v
```

## Performance

### Request Metrics

**Average Latency**:
- Health check: <5ms
- List documents: 10-50ms
- Get document: 10-30ms
- Chat (first token): 400-900ms
- Upload: 100-500ms (file I/O)

### Concurrency

**FastAPI Async**:
- Non-blocking I/O for database, API calls
- uvicorn workers for parallelism

**Configuration**:
```bash
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

## OpenAPI Documentation

**Interactive Docs**: http://localhost:8000/docs

**Features**:
- Auto-generated from Pydantic models
- Interactive request testing
- Schema inspection
- Example requests/responses

**Alternative UI**: http://localhost:8000/redoc

## Related Documentation

- [RAG Pipeline](03-components/backend/rag-pipeline.md)
- [Chat Service](05-data-flows/chat-query-flow.md)
- [Document Processing](03-components/ingestion/document-processor.md)
- [API Reference](06-api-reference)
