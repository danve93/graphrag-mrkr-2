# Chat API Reference

## Overview

The Chat API provides endpoints for conversational RAG queries with support for streaming responses, conversation history, and detailed timing instrumentation.

## Base URL

```
http://localhost:8000/api/chat
```

## Endpoints

### POST /query

Non-streaming chat endpoint that returns a complete response with all metadata.

#### Request

```json
{
  "message": "What are the backup options in Carbonio?",
  "session_id": "user-session-123",
  "context_documents": [],
  "retrieval_mode": "hybrid",
  "top_k": 10,
  "temperature": 0.7,
  "llm_model": "gpt-4",
  "embedding_model": "text-embedding-3-small",
  "similarity_threshold": 0.7,
  "enable_reranking": true,
  "expansion_similarity_threshold": 0.75,
  "max_expanded_chunks": 15,
  "max_expansion_depth": 2,
  "hybrid_chunk_weight": 1.0,
  "hybrid_entity_weight": 1.0,
  "hybrid_keyword_weight": 0.5,
  "enable_rrf": true,
  "rrf_k": 60,
  "enable_chunk_fulltext": true
}
```

#### Response

```json
{
  "response": "Carbonio provides several backup options...",
  "sources": [
    {
      "chunk_id": "chunk_001",
      "document_id": "doc_123",
      "document_name": "CarbonioAdminGuide.pdf",
      "content": "Backup procedures include...",
      "page_number": 42,
      "score": 0.89
    }
  ],
  "session_id": "user-session-123",
  "quality_score": {
    "overall": 0.85,
    "relevance": 0.88,
    "completeness": 0.82,
    "clarity": 0.86
  },
  "follow_up_questions": [
    "How do I schedule automated backups?",
    "What are the backup retention policies?",
    "Can I backup to cloud storage?"
  ],
  "stages": [
    {
      "name": "Query Analysis",
      "duration_ms": 234,
      "timestamp": 1701234567.123,
      "metadata": {}
    },
    {
      "name": "Retrieval",
      "duration_ms": 456,
      "timestamp": 1701234567.357,
      "metadata": {
        "chunks_retrieved": 10
      }
    },
    {
      "name": "Graph Reasoning",
      "duration_ms": 189,
      "timestamp": 1701234567.813,
      "metadata": {
        "context_items": 8
      }
    },
    {
      "name": "Generation",
      "duration_ms": 3421,
      "timestamp": 1701234571.002,
      "metadata": {
        "response_length": 342,
        "model_used": "gpt-4"
      }
    }
  ],
  "total_duration_ms": 4300
}
```

### POST /stream

Streaming chat endpoint that emits Server-Sent Events (SSE) for real-time updates.

#### Request

Same as `/query` endpoint.

#### Response (SSE Stream)

The stream emits multiple event types:

**Stage Event:**
```
data: {"type": "stage", "stage": "Query Analysis", "duration_ms": 234, "timestamp": 1701234567.123, "metadata": {}}
```

**Token Event:**
```
data: {"type": "token", "token": "Carbonio"}
```

**Sources Event:**
```
data: {"type": "sources", "sources": [...]}
```

**Quality Score Event:**
```
data: {"type": "quality_score", "quality_score": {...}}
```

**Follow-ups Event:**
```
data: {"type": "follow_ups", "follow_ups": [...]}
```

**Metadata Event:**
```
data: {"type": "metadata", "metadata": {"total_duration_ms": 4300, "stages": [...]}}
```

**Done Event:**
```
data: {"type": "done"}
```

## Request Parameters

### Required Parameters

- **message** (string): The user's query text
- **session_id** (string): Unique identifier for the conversation session

### Optional Parameters

- **context_documents** (array): List of document IDs to restrict retrieval scope (default: `[]`)
- **retrieval_mode** (string): Retrieval strategy - `"hybrid"`, `"vector"`, or `"graph"` (default: `"hybrid"`)
- **top_k** (integer): Number of chunks to retrieve (default: `10`)
- **temperature** (float): LLM generation temperature 0.0-1.0 (default: `0.7`)
- **llm_model** (string): LLM model name (default: from settings)
- **embedding_model** (string): Embedding model name (default: from settings)
- **similarity_threshold** (float): Minimum similarity score 0.0-1.0 (default: `0.7`)
- **enable_reranking** (boolean): Enable FlashRank reranking (default: from settings)
- **expansion_similarity_threshold** (float): Graph expansion edge threshold (default: `0.75`)
- **max_expanded_chunks** (integer): Maximum chunks after expansion (default: `15`)
- **max_expansion_depth** (integer): Graph traversal depth (default: `2`)
- **hybrid_chunk_weight** (float): Weight for chunk similarity (default: `1.0`)
- **hybrid_entity_weight** (float): Weight for entity similarity (default: `1.0`)
- **hybrid_keyword_weight** (float): Weight for keyword (BM25) matching (default: `0.5`)
- **enable_rrf** (boolean): Enable Reciprocal Rank Fusion (default: from settings)
- **rrf_k** (integer): RRF rank discount constant (default: `60`)
- **enable_chunk_fulltext** (boolean): Enable fulltext keyword search (default: from settings)

## Response Fields

### ChatResponse

- **response** (string): The generated answer text
- **sources** (array): List of source chunks used for generation
- **session_id** (string): Conversation session identifier
- **quality_score** (object, optional): Quality metrics for the response
- **follow_up_questions** (array): Suggested follow-up questions
- **stages** (array): Timing and metadata for each pipeline stage
- **total_duration_ms** (integer, optional): Total execution time in milliseconds

### Stage Object

- **name** (string): Stage name (`"Query Analysis"`, `"Retrieval"`, `"Graph Reasoning"`, `"Generation"`)
- **duration_ms** (integer): Execution time in milliseconds
- **timestamp** (float): Unix timestamp when stage completed
- **metadata** (object): Stage-specific data
  - **chunks_retrieved** (integer): Number of chunks retrieved (Retrieval stage)
  - **context_items** (integer): Number of context items after reasoning (Graph Reasoning stage)
  - **response_length** (integer): Character count of response (Generation stage)
  - **model_used** (string): LLM model name (Generation stage)

### Source Object

- **chunk_id** (string): Unique chunk identifier
- **document_id** (string): Parent document identifier
- **document_name** (string): Human-readable document name
- **content** (string): Chunk text content
- **page_number** (integer, optional): Page number in source document
- **score** (float): Similarity/relevance score 0.0-1.0

## Conversation Context

The chat API maintains conversation history within each session:

1. **Initial Query**: Send message with session_id
2. **Follow-up Queries**: Send subsequent messages with the same session_id
3. **Context Preservation**: The pipeline automatically includes previous messages when generating responses
4. **Token Efficiency**: Follow-ups reference history without re-sending full context

Example conversation flow:

```json
// First message
{
  "message": "What is Carbonio?",
  "session_id": "session-123"
}

// Follow-up (context automatically preserved)
{
  "message": "How do I install it?",
  "session_id": "session-123"
}
// The system knows "it" refers to Carbonio from previous context
```

## Cache Behavior

The retrieval cache uses a comprehensive 14-parameter hash:

1. Query text
2. Embedding model
3. Retrieval mode
4. Top-k
5. Expansion depth
6. Expansion threshold
7. Max expanded chunks
8. Similarity threshold
9. Reranking enabled
10. Chunk weight
11. Entity weight
12. Keyword weight
13. RRF configuration
14. Fulltext search enabled

**Cache Isolation**: Different parameter combinations will NOT share cache entries, ensuring parameter changes always yield fresh results.

**Cache TTL**: 60 seconds (configurable via `RETRIEVAL_CACHE_TTL`)

## Performance Monitoring

### Stage Timing

Monitor stage durations to identify bottlenecks:

- **Query Analysis**: Typically 100-300ms
- **Retrieval**: Typically 300-800ms
- **Graph Reasoning**: Typically 150-500ms
- **Generation**: Typically 2000-10000ms (varies by model and response length)

### Cache Metrics

Query cache statistics via:

```bash
curl http://localhost:8000/api/database/cache-stats
```

Response:
```json
{
  "entity_label_cache": {
    "size": 1234,
    "hits": 5678,
    "misses": 890,
    "hit_rate": 0.864
  },
  "embedding_cache": {
    "size": 4567,
    "hits": 12345,
    "misses": 2345,
    "hit_rate": 0.840
  },
  "retrieval_cache": {
    "size": 234,
    "hits": 567,
    "misses": 123,
    "hit_rate": 0.822
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

- **400 Bad Request**: Invalid parameters or missing required fields
- **404 Not Found**: Session or document not found
- **500 Internal Server Error**: Backend processing error (check logs)
- **503 Service Unavailable**: LLM provider or database unavailable

## Example: Python Client

```python
import requests
import json

# Non-streaming request
def chat_query(message, session_id):
    response = requests.post(
        "http://localhost:8000/api/chat/query",
        json={
            "message": message,
            "session_id": session_id,
            "top_k": 10,
            "temperature": 0.7
        }
    )
    return response.json()

# Streaming request
def chat_stream(message, session_id):
    response = requests.post(
        "http://localhost:8000/api/chat/stream",
        json={
            "message": message,
            "session_id": session_id
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if data['type'] == 'token':
                    print(data['token'], end='', flush=True)
                elif data['type'] == 'stage':
                    print(f"\n[{data['stage']}: {data['duration_ms']}ms]")
                elif data['type'] == 'done':
                    break
    print()

# Usage
result = chat_query("What is Carbonio?", "session-123")
print(f"Response: {result['response']}")
print(f"Duration: {result['total_duration_ms']}ms")
print(f"Sources: {len(result['sources'])}")

# Streaming
chat_stream("How do I install it?", "session-123")
```

## Example: JavaScript/TypeScript Client

```typescript
// Non-streaming
async function chatQuery(message: string, sessionId: string) {
  const response = await fetch('http://localhost:8000/api/chat/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      top_k: 10,
      temperature: 0.7
    })
  });
  return await response.json();
}

// Streaming
async function chatStream(message: string, sessionId: string) {
  const response = await fetch('http://localhost:8000/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, session_id: sessionId })
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = JSON.parse(line.slice(6));
        if (data.type === 'token') {
          process.stdout.write(data.token);
        } else if (data.type === 'stage') {
          console.log(`\n[${data.stage}: ${data.duration_ms}ms]`);
        } else if (data.type === 'done') {
          return;
        }
      }
    }
  }
}

// Usage
const result = await chatQuery("What is Carbonio?", "session-123");
console.log(`Response: ${result.response}`);
console.log(`Duration: ${result.total_duration_ms}ms`);

await chatStream("How do I install it?", "session-123");
```

## Best Practices

1. **Session Management**: Generate unique session IDs for each conversation; reuse for follow-ups
2. **Context Documents**: Restrict retrieval to specific documents when known to improve precision
3. **Parameter Tuning**: Use Chat Tuning UI to experiment with retrieval parameters before hardcoding
4. **Cache Warming**: Initial queries are slower; repeated queries benefit from caching
5. **Error Handling**: Always handle network errors and parse SSE events defensively
6. **Streaming UX**: Show stage progress and tokens incrementally for better user experience
7. **Performance Monitoring**: Track stage durations and cache hit rates to identify optimization opportunities

## Related Documentation

- [Configuration Reference](07-configuration/settings.md)
- [Operations Runbook](08-operations/monitoring.md)
- [Deployment Guide](08-operations/deployment.md)
