# API Reference

Complete REST API documentation for the Amber platform.

## Contents

- [README](06-api-reference/README.md) - API overview
- [Chat Endpoints](06-api-reference/chat-endpoints.md) - Chat query and streaming endpoints
- [Chat API Details](06-api-reference/chat-api.md) - Comprehensive Chat API reference
- [Document Endpoints](06-api-reference/document-endpoints.md) - Document operations and metadata
- [Database Endpoints](06-api-reference/database-endpoints.md) - Upload, stats, and maintenance
- [Graph Editor Endpoints](06-api-reference/graph-editor-endpoints.md) - Curation, healing, and backups
- [History Endpoints](06-api-reference/history-endpoints.md) - Conversation persistence
- [Jobs Endpoints](06-api-reference/jobs-endpoints.md) - Background job management
- [Models and Schemas](06-api-reference/models-schemas.md) - Pydantic request/response models

## Base URL

- **Local Development**: `http://localhost:8000`
- **Docker Compose**: `http://backend:8000` (internal) or `http://localhost:8000` (host)

## Authentication

Currently, Amber uses a simple user token system. Most endpoints do not require authentication for local development.

For production deployments, implement appropriate authentication (JWT, OAuth2, etc.) via FastAPI dependency injection.

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Common Headers

```http
Content-Type: application/json
Accept: application/json
```

For streaming endpoints:
```http
Accept: text/event-stream
```

## Response Format

Standard JSON responses:
```json
{
  "status": "success",
  "data": {...},
  "message": "Operation completed"
}
```

Error responses:
```json
{
  "detail": "Error message",
  "status_code": 400
}
```

## Pagination

Endpoints supporting pagination use offset-based pagination:

```http
GET /api/documents/{document_id}/chunks?limit=100&offset=0
```

Response includes pagination metadata:
```json
{
  "document_id": "abc123",
  "total": 2453,
  "limit": 100,
  "offset": 0,
  "has_more": true,
  "chunks": [...]
}
```

## Streaming Responses

Chat endpoints support Server-Sent Events (SSE) for real-time streaming:

```http
POST /api/chat/stream
Content-Type: application/json

{
  "message": "What is RAG?",
  "session_id": "session123",
  "stream": true
}
```

Response format:
```
data: {"type": "stage", "stage": "query_analysis"}
data: {"type": "token", "token": "RAG"}
data: {"type": "token", "token": " stands"}
data: {"type": "sources", "sources": [...]}
```

Event types:
- `stage` - Pipeline progress
- `token` - Generated text
- `sources` - Retrieved documents
- `quality_score` - Response quality
- `follow_ups` - Suggested questions
- `metadata` - Additional context

## Rate Limiting

No built-in rate limiting currently. Implement via:
- Nginx/reverse proxy
- API gateway
- FastAPI middleware (slowapi)

## Versioning

API version is currently embedded in the codebase. Future versions may use:
- Path-based: `/api/v1/chat/query`
- Header-based: `Accept: application/vnd.amber.v1+json`

## Quick Reference

### Chat
- `POST /api/chat/query` - Structured chat query
- `POST /api/chat/stream` - SSE streaming chat
- `POST /api/chat/follow-ups` - Generate follow-up questions

### Documents
- `GET /api/documents` - List all documents
- `GET /api/documents/{id}` - Get document metadata
- `GET /api/documents/{id}/summary` - Get precomputed summary
- `GET /api/documents/{id}/chunks` - Get document chunks (paginated)
- `GET /api/documents/{id}/entities` - Get document entities (paginated)
- `GET /api/documents/{id}/entity-summary` - Get entity counts by type
- `GET /api/documents/{id}/similarities` - Get chunk similarities (paginated)
- `DELETE /api/documents/{id}` - Delete document
- `POST /api/documents/{id}/generate-summary` - Generate LLM summary
- `PATCH /api/documents/{id}/hashtags` - Update hashtags

### Database
- `GET /api/database/stats` - Get database statistics
- `POST /api/database/upload` - Upload document
- `POST /api/database/clear` - Clear all data
- `GET /api/database/cache-stats` - Get cache performance metrics

### History
- `GET /api/history/sessions` - List all sessions
- `GET /api/history/{session_id}` - Get session messages
- `DELETE /api/history/{session_id}` - Delete session
- `POST /api/history/clear` - Clear all history

### Jobs
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job status
- `POST /api/jobs/{job_id}/cancel` - Cancel job

### Health
- `GET /api/health` - Health check and system info

## Code Examples

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat/query",
    json={
        "message": "What is graph-enhanced RAG?",
        "session_id": "session123",
        "llm_model": "gpt-4o-mini",
        "retrieval_mode": "hybrid",
        "stream": False
    }
)
result = response.json()
print(result["response"])
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/api/chat/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What is graph-enhanced RAG?',
    session_id: 'session123',
    llm_model: 'gpt-4o-mini',
    retrieval_mode: 'hybrid',
    stream: false
  })
});
const result = await response.json();
console.log(result.response);
```

### cURL

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is graph-enhanced RAG?",
    "session_id": "session123",
    "llm_model": "gpt-4o-mini",
    "retrieval_mode": "hybrid",
    "stream": false
  }'
```

## Testing API Endpoints

```bash
pytest tests/integration/test_api_endpoints.py -v
```

## Related Documentation

- [Chat Endpoints Details](06-api-reference/chat-endpoints.md)
- [Document Endpoints Details](06-api-reference/document-endpoints.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
- [Models and Schemas](06-api-reference/models-schemas.md)
