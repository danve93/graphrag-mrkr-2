# Jobs Endpoints

Background job management and status tracking.

## GET /api/jobs

List all background jobs.

### Request

**URL**: `GET /api/jobs?status=running&limit=50&offset=0`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `status` | string | No | - | Filter by status (`pending`/`running`/`completed`/`failed`) |
| `job_type` | string | No | - | Filter by type (`ingestion`/`reindex`/`clustering`) |
| `limit` | int | No | `50` | Max jobs to return |
| `offset` | int | No | `0` | Pagination offset |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "jobs": [
    {
      "job_id": "job-123-abc",
      "job_type": "ingestion",
      "status": "running",
      "progress": 0.65,
      "message": "Processing chunks (95/147)",
      "created_at": "2024-01-15T16:30:00Z",
      "started_at": "2024-01-15T16:30:05Z",
      "metadata": {
        "filename": "VxRail_Admin_Guide.pdf",
        "document_id": "doc-001"
      }
    },
    {
      "job_id": "job-456-def",
      "job_type": "reindex",
      "status": "completed",
      "progress": 1.0,
      "message": "Reindexing complete",
      "created_at": "2024-01-15T15:00:00Z",
      "started_at": "2024-01-15T15:00:02Z",
      "completed_at": "2024-01-15T15:23:45Z",
      "metadata": {
        "rebuild_similarities": true,
        "run_clustering": true
      }
    }
  ],
  "total": 127,
  "limit": 50,
  "offset": 0,
  "has_more": true
}
```

### Example

```bash
curl "http://localhost:8000/api/jobs?status=running"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(
        'http://localhost:8000/api/jobs',
        params={'status': 'running', 'limit': 10}
    )
    data = response.json()
    
    for job in data['jobs']:
        print(f"{job['job_type']}: {job['progress']*100:.1f}% - {job['message']}")
```

---

## GET /api/jobs/{job_id}

Get detailed status for a specific job.

### Request

**URL**: `GET /api/jobs/{job_id}`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "job_id": "job-123-abc",
  "job_type": "ingestion",
  "status": "running",
  "progress": 0.65,
  "message": "Processing chunks (95/147)",
  "created_at": "2024-01-15T16:30:00Z",
  "started_at": "2024-01-15T16:30:05Z",
  "metadata": {
    "filename": "VxRail_Admin_Guide.pdf",
    "document_id": "doc-001",
    "file_size": 8456192,
    "page_count": 350
  },
  "stages": [
    {
      "name": "loading",
      "status": "completed",
      "progress": 1.0,
      "duration_seconds": 3.2
    },
    {
      "name": "chunking",
      "status": "completed",
      "progress": 1.0,
      "duration_seconds": 1.8
    },
    {
      "name": "embedding",
      "status": "running",
      "progress": 0.65,
      "duration_seconds": 12.4
    },
    {
      "name": "entity_extraction",
      "status": "pending",
      "progress": 0.0
    },
    {
      "name": "persistence",
      "status": "pending",
      "progress": 0.0
    }
  ],
  "errors": []
}
```

### Example

```bash
curl http://localhost:8000/api/jobs/job-123-abc
```

---

## POST /api/jobs/{job_id}/cancel

Cancel a running job.

### Request

**URL**: `POST /api/jobs/{job_id}/cancel`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "job_id": "job-123-abc",
  "status": "cancelled",
  "message": "Job cancelled successfully"
}
```

**Status**: `400 Bad Request` (if job cannot be cancelled)

**Body**:
```json
{
  "error": "Cannot cancel completed job"
}
```

### Example

```bash
curl -X POST http://localhost:8000/api/jobs/job-123-abc/cancel
```

---

## DELETE /api/jobs/{job_id}

Delete a job record (completed or failed only).

### Request

**URL**: `DELETE /api/jobs/{job_id}`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Job deleted",
  "job_id": "job-123-abc"
}
```

**Status**: `400 Bad Request` (if job is still running)

**Body**:
```json
{
  "error": "Cannot delete running job. Cancel it first."
}
```

---

## GET /api/jobs/{job_id}/logs

Get detailed logs for a job.

### Request

**URL**: `GET /api/jobs/{job_id}/logs?level=info&limit=100`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `level` | string | No | - | Filter by level (`debug`/`info`/`warning`/`error`) |
| `limit` | int | No | `100` | Max log entries |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "job_id": "job-123-abc",
  "logs": [
    {
      "timestamp": "2024-01-15T16:30:05.123Z",
      "level": "info",
      "message": "Starting document ingestion",
      "context": {
        "filename": "VxRail_Admin_Guide.pdf"
      }
    },
    {
      "timestamp": "2024-01-15T16:30:08.456Z",
      "level": "info",
      "message": "Loaded 350 pages",
      "context": {
        "duration_seconds": 3.2
      }
    },
    {
      "timestamp": "2024-01-15T16:30:10.789Z",
      "level": "info",
      "message": "Created 147 chunks",
      "context": {
        "chunk_size": 1000,
        "overlap": 200
      }
    },
    {
      "timestamp": "2024-01-15T16:30:15.234Z",
      "level": "warning",
      "message": "Rate limit approaching, adding delay",
      "context": {
        "requests_per_minute": 55,
        "delay_ms": 500
      }
    }
  ],
  "total": 47
}
```

### Example

```bash
curl "http://localhost:8000/api/jobs/job-123-abc/logs?level=warning"
```

---

## POST /api/jobs/cleanup

Clean up old completed/failed jobs.

### Request

**URL**: `POST /api/jobs/cleanup`

**Body**:
```json
{
  "older_than_days": 7,
  "status": ["completed", "failed"]
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `older_than_days` | int | No | `7` | Delete jobs older than N days |
| `status` | string[] | No | `["completed","failed"]` | Job statuses to clean up |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Cleanup complete",
  "jobs_deleted": 23,
  "space_freed_mb": 145.7
}
```

### Example

```bash
curl -X POST http://localhost:8000/api/jobs/cleanup \
  -H "Content-Type: application/json" \
  -d '{"older_than_days": 30}'
```

---

## WebSocket: /api/jobs/{job_id}/stream

Real-time job progress updates via WebSocket.

### Connection

**URL**: `ws://localhost:8000/api/jobs/{job_id}/stream`

### Messages (Server → Client)

```json
{
  "type": "progress",
  "job_id": "job-123-abc",
  "progress": 0.67,
  "message": "Processing chunks (98/147)",
  "timestamp": "2024-01-15T16:30:18.456Z"
}
```

```json
{
  "type": "stage_complete",
  "job_id": "job-123-abc",
  "stage": "embedding",
  "duration_seconds": 15.2,
  "timestamp": "2024-01-15T16:30:20.123Z"
}
```

```json
{
  "type": "completed",
  "job_id": "job-123-abc",
  "message": "Document ingestion complete",
  "metadata": {
    "chunks_created": 147,
    "entities_extracted": 83
  },
  "timestamp": "2024-01-15T16:31:05.789Z"
}
```

```json
{
  "type": "error",
  "job_id": "job-123-abc",
  "error": "Embedding API rate limit exceeded",
  "timestamp": "2024-01-15T16:30:25.456Z"
}
```

### Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8000/api/jobs/job-123-abc/stream');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'progress':
      console.log(`Progress: ${(data.progress * 100).toFixed(1)}%`);
      break;
    case 'stage_complete':
      console.log(`Stage ${data.stage} completed in ${data.duration_seconds}s`);
      break;
    case 'completed':
      console.log('Job completed:', data.metadata);
      ws.close();
      break;
    case 'error':
      console.error('Job error:', data.error);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};
```

### Example (Python)

```python
import asyncio
import websockets
import json

async def monitor_job(job_id: str):
    uri = f'ws://localhost:8000/api/jobs/{job_id}/stream'
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            
            if data['type'] == 'progress':
                print(f"Progress: {data['progress']*100:.1f}% - {data['message']}")
            elif data['type'] == 'completed':
                print(f"Completed: {data['message']}")
                break
            elif data['type'] == 'error':
                print(f"Error: {data['error']}")
                break

asyncio.run(monitor_job('job-123-abc'))
```

---

## Job Types

### Ingestion Jobs

**Type**: `ingestion`

**Stages**:
1. `loading` - Extract text from document
2. `chunking` - Split into chunks
3. `embedding` - Generate embeddings
4. `entity_extraction` - Extract entities (optional)
5. `persistence` - Save to Neo4j

**Typical Duration**: 15s (small) - 5min (large)

### Reindex Jobs

**Type**: `reindex`

**Stages**:
1. `rebuild_embeddings` - Regenerate embeddings (optional)
2. `rebuild_similarities` - Compute SIMILAR_TO edges
3. `clustering` - Run community detection (optional)

**Typical Duration**: 5min (small) - 30min (large database)

### Clustering Jobs

**Type**: `clustering`

**Stages**:
1. `build_projection` - Create graph projection
2. `run_leiden` - Execute Leiden algorithm
3. `assign_communities` - Update entity properties
4. `generate_summaries` - Create community descriptions (optional)

**Typical Duration**: 2-10min depending on entity count

---

## Related Documentation

- [Document Processing](03-components/backend/document-processing.md)
- [Job Manager](03-components/backend/job-manager.md)
- [Upload Component](03-components/frontend/upload.md)
- [Document Endpoints](06-api-reference/document-endpoints.md)
