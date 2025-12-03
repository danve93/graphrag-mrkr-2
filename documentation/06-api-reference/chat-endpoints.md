# Chat Endpoints

Chat query and streaming response endpoints.

## POST /api/chat

Stream chat response with RAG pipeline execution.

### Request

**URL**: `POST /api/chat`

**Headers**:
```http
Content-Type: application/json
Accept: text/event-stream
```

**Body**:
```json
{
  "message": "What are the backup procedures for VxRail?",
  "session_id": "session-abc-123",
  "context_documents": [],
  
  "llm_model": "gpt-4",
  "embedding_model": "text-embedding-3-small",
  "temperature": 0.7,
  "max_tokens": 2000,
  
  "retrieval_mode": "hybrid",
  "retrieval_top_k": 10,
  "hybrid_chunk_weight": 0.7,
  "hybrid_entity_weight": 0.3,
  
  "expansion_depth": 1,
  "expansion_similarity_threshold": 0.7,
  "max_expanded_chunks": 50,
  
  "flashrank_blend_weight": 0.5,
  "flashrank_max_candidates": 30
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User query text |
| `session_id` | string | Yes | - | Conversation session ID |
| `context_documents` | string[] | No | `[]` | Filter to specific document IDs |
| `llm_model` | string | No | `settings.llm_model` | LLM model name |
| `embedding_model` | string | No | `settings.embedding_model` | Embedding model name |
| `temperature` | float | No | `0.7` | Sampling temperature (0.0-2.0) |
| `max_tokens` | int | No | `2000` | Max output tokens (100-8000) |
| `retrieval_mode` | string | No | `"hybrid"` | Retrieval mode: `"vector"`, `"hybrid"`, `"entity"` |
| `retrieval_top_k` | int | No | `10` | Number of chunks to retrieve (1-100) |
| `hybrid_chunk_weight` | float | No | `0.7` | Weight for vector component (0.0-1.0) |
| `hybrid_entity_weight` | float | No | `0.3` | Weight for entity component (0.0-1.0) |
| `expansion_depth` | int | No | `1` | Graph expansion depth (0-3) |
| `expansion_similarity_threshold` | float | No | `0.7` | Min similarity for expansion (0.0-1.0) |
| `max_expanded_chunks` | int | No | `50` | Max expanded chunks (0-200) |
| `flashrank_blend_weight` | float | No | `0.5` | Rerank score weight (0.0-1.0) |
| `flashrank_max_candidates` | int | No | `30` | Max chunks to rerank (5-100) |

### Response

**Status**: `200 OK`

**Headers**:
```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

**Body** (Server-Sent Events):

#### Stage Event
```
data: {"type":"stage","stage":"query_analysis","message":"Analyzing query"}
```

Stages: `query_analysis`, `retrieval`, `graph_reasoning`, `generation`, `post_generation`

#### Token Event
```
data: {"type":"token","token":"VxRail"}
```

#### Sources Event
```
data: {"type":"sources","sources":[{"chunk_id":"chunk-047","document_id":"doc-001","document_name":"VxRail_Admin.pdf","page_number":47,"relevance_score":0.778}]}
```

#### Quality Score Event
```
data: {"type":"quality_score","score":0.87}
```

#### Follow-Up Questions Event
```
data: {"type":"follow_ups","questions":["What retention policies are recommended?","How to restore from backup?"]}
```

#### Error Event
```
data: {"type":"error","message":"An error occurred during generation"}
```

### Example

**cURL**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "message": "What are the backup procedures?",
    "session_id": "test-session",
    "temperature": 0.7,
    "retrieval_top_k": 10
  }'
```

**JavaScript**:
```javascript
const response = await fetch('http://localhost:8000/api/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'text/event-stream',
  },
  body: JSON.stringify({
    message: 'What are the backup procedures?',
    session_id: 'test-session',
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const text = decoder.decode(value);
  console.log(text);
}
```

**Python**:
```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream(
        'POST',
        'http://localhost:8000/api/chat',
        json={
            'message': 'What are the backup procedures?',
            'session_id': 'test-session',
        },
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith('data: '):
                event_data = line[6:]
                event = json.loads(event_data)
                print(event)
```

### Error Responses

**400 Bad Request**:
```json
{
  "detail": "message field is required"
}
```

**422 Unprocessable Entity**:
```json
{
  "detail": [
    {
      "loc": ["body", "temperature"],
      "msg": "ensure this value is less than or equal to 2.0",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Internal server error"
}
```

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [RAG Pipeline](03-components/backend/rag-pipeline.md)
- [Chat Query Flow](05-data-flows/chat-query-flow.md)
- [Streaming SSE](05-data-flows/streaming-sse-flow.md)
