# Models and Schemas

Pydantic request/response models for the Amber API.

## Chat Models

### ChatRequest

Request model for POST /api/chat.

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: str = Field(..., description="Conversation session ID")
    
    # Model selection
    llm_model: Optional[str] = Field(None, description="LLM model name")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    
    # Generation parameters
    temperature: Optional[float] = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(2000, ge=100, le=8000)
    
    # Retrieval parameters
    retrieval_mode: Optional[str] = Field("hybrid", pattern="^(vector|hybrid|entity)$")
    retrieval_top_k: Optional[int] = Field(10, ge=1, le=100)
    
    # Hybrid retrieval weights
    hybrid_chunk_weight: Optional[float] = Field(0.7, ge=0, le=1)
    hybrid_entity_weight: Optional[float] = Field(0.3, ge=0, le=1)
    
    # Graph expansion
    expansion_depth: Optional[int] = Field(1, ge=0, le=3)
    expansion_similarity_threshold: Optional[float] = Field(0.7, ge=0, le=1)
    max_expanded_chunks: Optional[int] = Field(50, ge=0, le=200)
    
    # Reranking
    flashrank_blend_weight: Optional[float] = Field(0.5, ge=0, le=1)
    flashrank_max_candidates: Optional[int] = Field(30, ge=5, le=100)
    
    # Context
    context_documents: Optional[List[str]] = Field(None, description="Document IDs to restrict search")
```

**Example**:
```json
{
  "message": "How do I configure VxRail backups?",
  "session_id": "session-abc-123",
  "llm_model": "gpt-4",
  "temperature": 0.7,
  "retrieval_mode": "hybrid",
  "retrieval_top_k": 15,
  "expansion_depth": 2,
  "flashrank_blend_weight": 0.6,
  "context_documents": ["doc-001", "doc-005"]
}
```

### ChatMessage

Individual message in conversation history.

```python
class ChatMessage(BaseModel):
    message_id: str
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: str
    sources: Optional[List['Source']] = None
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    metadata: Optional[dict] = None
```

**Example**:
```json
{
  "message_id": "msg-001",
  "role": "assistant",
  "content": "VxRail backup configuration involves...",
  "timestamp": "2024-01-15T14:30:15Z",
  "sources": [...],
  "quality_score": 0.92,
  "metadata": {
    "llm_model": "gpt-4",
    "tokens_used": 234
  }
}
```

### Source

Source chunk referenced in response.

```python
class Source(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    relevance_score: float = Field(..., ge=0, le=1)
    content_preview: Optional[str] = None
    expansion_path: Optional[List[str]] = None
```

**Example**:
```json
{
  "chunk_id": "chunk-123",
  "document_id": "doc-001",
  "document_name": "VxRail_Admin_Guide.pdf",
  "page_number": 87,
  "relevance_score": 0.89,
  "content_preview": "Configure backup schedules in the VxRail Manager...",
  "expansion_path": ["chunk-123", "entity-backup", "chunk-456"]
}
```

---

## Document Models

### DocumentMetadata

Document information and statistics.

```python
class DocumentMetadata(BaseModel):
    id: str
    filename: str
    title: Optional[str] = None
    page_count: Optional[int] = None
    chunk_count: int
    entity_count: int
    upload_date: str
    file_size: int
    file_type: str
    metadata: Optional[dict] = None
```

**Example**:
```json
{
  "id": "doc-001",
  "filename": "VxRail_Admin_Guide.pdf",
  "title": "VxRail Administration Guide",
  "page_count": 350,
  "chunk_count": 147,
  "entity_count": 83,
  "upload_date": "2024-01-15T14:00:00Z",
  "file_size": 8456192,
  "file_type": "pdf",
  "metadata": {
    "author": "Dell Technologies",
    "created_date": "2023-06-15"
  }
}
```

### ChunkData

Individual chunk with content and relationships.

```python
class ChunkData(BaseModel):
    chunk_id: str
    content: str
    page_number: Optional[int] = None
    position: int
    chunk_size: int
    entities: List['EntityReference']
    metadata: Optional[dict] = None
```

**Example**:
```json
{
  "chunk_id": "chunk-123",
  "content": "VxRail backup configuration involves...",
  "page_number": 87,
  "position": 0,
  "chunk_size": 947,
  "entities": [
    {
      "entity_id": "entity-vxrail",
      "name": "VxRail",
      "type": "Component"
    },
    {
      "entity_id": "entity-backup",
      "name": "Backup",
      "type": "Procedure"
    }
  ],
  "metadata": {
    "quality_score": 0.87
  }
}
```

### UploadResponse

Response from document upload.

```python
class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str = Field(..., pattern="^(processing|queued|failed)$")
    message: Optional[str] = None
```

**Example**:
```json
{
  "job_id": "job-123-abc",
  "filename": "VxRail_Admin_Guide.pdf",
  "status": "processing",
  "message": "Document ingestion started"
}
```

---

## Database Models

### DatabaseStats

Neo4j database statistics.

```python
class DatabaseStats(BaseModel):
    documents: 'DocumentStats'
    chunks: 'ChunkStats'
    entities: 'EntityStats'
    relationships: 'RelationshipStats'
    communities: Optional['CommunityStats'] = None
    storage: 'StorageStats'

class DocumentStats(BaseModel):
    total: int
    total_size_mb: float

class ChunkStats(BaseModel):
    total: int
    avg_per_document: float
    with_embeddings: int

class EntityStats(BaseModel):
    total: int
    by_type: dict[str, int]

class RelationshipStats(BaseModel):
    CONTAINS: int
    MENTIONS: int
    RELATED_TO: int
    SIMILAR_TO: int
    total: int

class CommunityStats(BaseModel):
    total: int
    largest_size: int
    avg_size: float

class StorageStats(BaseModel):
    node_count: int
    relationship_count: int
    property_count: int
    index_count: int
```

### GraphData

Graph visualization data.

```python
class GraphData(BaseModel):
    nodes: List['GraphNode']
    edges: List['GraphEdge']
    communities: Optional[List['CommunityInfo']] = None
    total_nodes: int
    total_edges: int

class GraphNode(BaseModel):
    id: str
    label: str
    type: str = Field(..., pattern="^(Document|Chunk|Entity)$")
    properties: dict

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    properties: Optional[dict] = None

class CommunityInfo(BaseModel):
    id: int
    size: int
    summary: Optional[str] = None
```

---

## History Models

### ConversationSession

Full conversation session with messages.

```python
class ConversationSession(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    title: str
    messages: List[ChatMessage]
    metadata: Optional[dict] = None

class SessionSummary(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    title: str
    last_message_preview: Optional[str] = None
```

---

## Job Models

### JobStatus

Background job status and progress.

```python
class JobStatus(BaseModel):
    job_id: str
    job_type: str = Field(..., pattern="^(ingestion|reindex|clustering)$")
    status: str = Field(..., pattern="^(pending|running|completed|failed|cancelled)$")
    progress: float = Field(..., ge=0, le=1)
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Optional[dict] = None
    stages: Optional[List['JobStage']] = None
    errors: Optional[List[str]] = None

class JobStage(BaseModel):
    name: str
    status: str
    progress: float = Field(..., ge=0, le=1)
    duration_seconds: Optional[float] = None
```

---

## Entity Models

### Entity

Entity extracted from documents.

```python
class Entity(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = None
    importance: float = Field(..., ge=0, le=1)
    mention_count: int
    relationship_count: int
    community_id: Optional[int] = None
    source_documents: List[str]

class EntityReference(BaseModel):
    entity_id: str
    name: str
    type: str
```

### Relationship

Relationship between entities.

```python
class Relationship(BaseModel):
    source_id: str
    target_id: str
    type: str
    strength: float = Field(..., ge=0, le=1)
    provenance: List[str]  # Chunk IDs where relationship appears
```

---

## Error Models

### ErrorResponse

Standard error response format.

```python
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
```

**Example**:
```json
{
  "error": "Document not found",
  "detail": "No document with ID: doc-999",
  "code": "DOCUMENT_NOT_FOUND"
}
```

### ValidationError

Pydantic validation error (422 response).

```json
{
  "detail": [
    {
      "loc": ["body", "temperature"],
      "msg": "ensure this value is less than or equal to 2",
      "type": "value_error.number.not_le",
      "ctx": {"limit_value": 2}
    }
  ]
}
```

---

## Configuration Models

### ChatTuningConfig

Chat tuning panel settings.

```python
class ChatTuningConfig(BaseModel):
    llm_model: str
    embedding_model: str
    temperature: float
    max_tokens: int
    retrieval_mode: str
    retrieval_top_k: int
    hybrid_chunk_weight: float
    hybrid_entity_weight: float
    expansion_depth: int
    expansion_similarity_threshold: float
    max_expanded_chunks: int
    flashrank_enabled: bool
    flashrank_blend_weight: float
    flashrank_max_candidates: int
```

---

## Common Patterns

### Pagination

Standard pagination response wrapper.

```python
class PaginatedResponse(BaseModel):
    items: List[T]  # Generic type
    total: int
    limit: int
    offset: int
    has_more: bool
```

**Example**:
```json
{
  "items": [...],
  "total": 147,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

### Timestamps

All timestamps use ISO 8601 format:
```
2024-01-15T14:30:00Z
```

Python:
```python
from datetime import datetime
timestamp = datetime.utcnow().isoformat() + 'Z'
```

JavaScript:
```javascript
const timestamp = new Date().toISOString();
```

### Field Validation

Common validation patterns:

```python
# String patterns
pattern="^(vector|hybrid|entity)$"  # Enum via regex
min_length=1, max_length=255        # String length

# Numeric ranges
ge=0, le=1      # Greater/less than or equal (0 to 1)
gt=0, lt=100    # Greater/less than (0 to 100, exclusive)

# Optional with default
Field(0.7, ge=0, le=1)  # Default 0.7, range [0, 1]
```

---

## Related Documentation

- [Chat Endpoints](06-api-reference/chat-endpoints.md)
- [Document Endpoints](06-api-reference/document-endpoints.md)
- [Database Endpoints](06-api-reference/database-endpoints.md)
- [History Endpoints](06-api-reference/history-endpoints.md)
- [Jobs Endpoints](06-api-reference/jobs-endpoints.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)
