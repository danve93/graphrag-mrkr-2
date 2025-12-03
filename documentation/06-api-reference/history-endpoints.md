# History Endpoints

Conversation history management and session operations.

## GET /api/history/sessions

List all conversation sessions.

### Request

**URL**: `GET /api/history/sessions?limit=50&offset=0`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `50` | Max sessions to return |
| `offset` | int | No | `0` | Pagination offset |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "sessions": [
    {
      "session_id": "session-abc-123",
      "created_at": "2024-01-15T14:30:00Z",
      "updated_at": "2024-01-15T15:45:00Z",
      "message_count": 12,
      "title": "VxRail backup procedures",
      "last_message_preview": "How do I configure incremental backups?"
    },
    {
      "session_id": "session-def-456",
      "created_at": "2024-01-14T09:15:00Z",
      "updated_at": "2024-01-14T09:30:00Z",
      "message_count": 4,
      "title": "Storage configuration",
      "last_message_preview": "What's the recommended RAID level?"
    }
  ],
  "total": 47,
  "limit": 50,
  "offset": 0,
  "has_more": false
}
```

### Example

```bash
curl "http://localhost:8000/api/history/sessions?limit=20"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(
        'http://localhost:8000/api/history/sessions',
        params={'limit': 20}
    )
    data = response.json()
    
    for session in data['sessions']:
        print(f"{session['title']} ({session['message_count']} messages)")
```

---

## GET /api/history/sessions/{session_id}

Get all messages for a session.

### Request

**URL**: `GET /api/history/sessions/{session_id}`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "session_id": "session-abc-123",
  "created_at": "2024-01-15T14:30:00Z",
  "updated_at": "2024-01-15T15:45:00Z",
  "title": "VxRail backup procedures",
  "messages": [
    {
      "message_id": "msg-001",
      "role": "user",
      "content": "How do I configure VxRail backups?",
      "timestamp": "2024-01-15T14:30:00Z"
    },
    {
      "message_id": "msg-002",
      "role": "assistant",
      "content": "VxRail backup configuration involves several steps...",
      "timestamp": "2024-01-15T14:30:15Z",
      "sources": [
        {
          "chunk_id": "chunk-123",
          "document_id": "doc-001",
          "document_name": "VxRail_Admin_Guide.pdf",
          "page_number": 87,
          "relevance_score": 0.89
        }
      ],
      "quality_score": 0.92
    }
  ],
  "metadata": {
    "document_context": ["doc-001", "doc-005"],
    "total_tokens": 3456
  }
}
```

### Example

```bash
curl http://localhost:8000/api/history/sessions/session-abc-123
```

---

## POST /api/history/sessions

Create a new conversation session.

### Request

**URL**: `POST /api/history/sessions`

**Body**:
```json
{
  "title": "New conversation",
  "metadata": {
    "document_context": ["doc-001"]
  }
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | No | Session title (auto-generated if omitted) |
| `metadata` | object | No | Session metadata |

### Response

**Status**: `201 Created`

**Body**:
```json
{
  "session_id": "session-xyz-789",
  "created_at": "2024-01-15T16:00:00Z",
  "title": "New conversation",
  "message_count": 0
}
```

### Example

```bash
curl -X POST http://localhost:8000/api/history/sessions \
  -H "Content-Type: application/json" \
  -d '{"title": "Storage questions"}'
```

---

## PUT /api/history/sessions/{session_id}

Update session metadata.

### Request

**URL**: `PUT /api/history/sessions/{session_id}`

**Body**:
```json
{
  "title": "Updated title",
  "metadata": {
    "document_context": ["doc-001", "doc-002"],
    "tags": ["backup", "storage"]
  }
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "session_id": "session-abc-123",
  "title": "Updated title",
  "updated_at": "2024-01-15T16:15:00Z"
}
```

---

## DELETE /api/history/sessions/{session_id}

Delete a conversation session.

### Request

**URL**: `DELETE /api/history/sessions/{session_id}`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Session deleted",
  "session_id": "session-abc-123",
  "messages_deleted": 12
}
```

### Example

```bash
curl -X DELETE http://localhost:8000/api/history/sessions/session-abc-123
```

---

## POST /api/history/sessions/{session_id}/messages

Add a message to a session (manual entry).

### Request

**URL**: `POST /api/history/sessions/{session_id}/messages`

**Body**:
```json
{
  "role": "user",
  "content": "What's the backup retention policy?",
  "metadata": {
    "client_timestamp": "2024-01-15T16:20:00Z"
  }
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | `"user"` or `"assistant"` |
| `content` | string | Yes | Message content |
| `metadata` | object | No | Message metadata |

### Response

**Status**: `201 Created`

**Body**:
```json
{
  "message_id": "msg-045",
  "session_id": "session-abc-123",
  "role": "user",
  "content": "What's the backup retention policy?",
  "timestamp": "2024-01-15T16:20:05Z"
}
```

---

## DELETE /api/history/sessions/{session_id}/messages/{message_id}

Delete a specific message from a session.

### Request

**URL**: `DELETE /api/history/sessions/{session_id}/messages/{message_id}`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Message deleted",
  "message_id": "msg-045"
}
```

---

## GET /api/history/search

Search across conversation history.

### Request

**URL**: `GET /api/history/search?query=backup&limit=20`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | int | No | Max results (default: 20) |
| `role` | string | No | Filter by role (`user`/`assistant`) |
| `session_id` | string | No | Filter to specific session |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "results": [
    {
      "message_id": "msg-023",
      "session_id": "session-abc-123",
      "role": "assistant",
      "content": "VxRail backup configuration involves...",
      "timestamp": "2024-01-15T14:30:15Z",
      "relevance_score": 0.87,
      "context": {
        "session_title": "VxRail backup procedures",
        "previous_message": "How do I configure VxRail backups?"
      }
    }
  ],
  "total": 5,
  "query": "backup"
}
```

### Example

```bash
curl "http://localhost:8000/api/history/search?query=backup&limit=10"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(
        'http://localhost:8000/api/history/search',
        params={'query': 'backup configuration', 'limit': 10}
    )
    results = response.json()
    
    for result in results['results']:
        print(f"Session: {result['context']['session_title']}")
        print(f"Score: {result['relevance_score']:.2f}")
        print(f"Content: {result['content'][:100]}...")
```

---

## GET /api/history/export/{session_id}

Export session to JSON or Markdown.

### Request

**URL**: `GET /api/history/export/{session_id}?format=markdown`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | string | No | `json` | Export format (`json` or `markdown`) |

### Response (JSON format)

**Status**: `200 OK`

**Content-Type**: `application/json`

**Body**: Full session data as JSON (same structure as GET session)

### Response (Markdown format)

**Status**: `200 OK`

**Content-Type**: `text/markdown`

**Body**:
```markdown
# VxRail backup procedures

**Session ID**: session-abc-123  
**Created**: 2024-01-15 14:30:00  
**Messages**: 12

---

## Message 1 (User)
**Timestamp**: 2024-01-15 14:30:00

How do I configure VxRail backups?

---

## Message 2 (Assistant)
**Timestamp**: 2024-01-15 14:30:15  
**Quality Score**: 0.92

VxRail backup configuration involves several steps...

**Sources**:
- VxRail_Admin_Guide.pdf (page 87, score: 0.89)
```

### Example

```bash
# Export as JSON
curl http://localhost:8000/api/history/export/session-abc-123 > session.json

# Export as Markdown
curl "http://localhost:8000/api/history/export/session-abc-123?format=markdown" > session.md
```

---

## Related Documentation

- [Chat Interface](03-components/frontend/chat-interface.md)
- [History Panel](03-components/frontend/history-panel.md)
- [Chat Service](03-components/backend/chat-service.md)
- [Chat Endpoints](06-api-reference/chat-endpoints.md)
