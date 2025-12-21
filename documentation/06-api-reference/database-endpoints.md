# Database Endpoints

Database inspection, statistics, and management operations.

## GET /api/database/stats

Get Neo4j database statistics.

### Request

**URL**: `GET /api/database/stats`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "documents": {
    "total": 47,
    "total_size_mb": 125.4
  },
  "chunks": {
    "total": 6891,
    "avg_per_document": 146.6,
    "with_embeddings": 6891
  },
  "entities": {
    "total": 1247,
    "by_type": {
      "Component": 234,
      "Service": 189,
      "Procedure": 156,
      "Product": 143,
      "Concept": 128,
      "Node": 97,
      "Other": 300
    }
  },
  "relationships": {
    "CONTAINS": 6891,
    "MENTIONS": 8932,
    "RELATED_TO": 2847,
    "SIMILAR_TO": 12456,
    "total": 31126
  },
  "communities": {
    "total": 87,
    "largest_size": 42,
    "avg_size": 14.3
  },
  "storage": {
    "node_count": 8185,
    "relationship_count": 31126,
    "property_count": 94523,
    "index_count": 3
  }
}
```

### Example

```bash
curl http://localhost:8000/api/database/stats
```

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get('http://localhost:8000/api/database/stats')
    stats = response.json()
    
    print(f"Documents: {stats['documents']['total']}")
    print(f"Chunks: {stats['chunks']['total']}")
    print(f"Entities: {stats['entities']['total']}")
```

---

## GET /api/database/graph

Get graph data for visualization.

### Request

**URL**: `GET /api/database/graph?limit=100&document_id=doc-001&include_chunks=true&include_entities=true`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `100` | Max nodes to return |
| `document_id` | string | No | - | Filter to specific document |
| `include_chunks` | bool | No | `true` | Include chunk nodes |
| `include_entities` | bool | No | `true` | Include entity nodes |
| `include_documents` | bool | No | `true` | Include document nodes |
| `community_ids` | int[] | No | - | Filter to specific communities |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "nodes": [
    {
      "id": "doc-001",
      "label": "VxRail_Admin_Guide.pdf",
      "type": "Document",
      "properties": {
        "page_count": 350,
        "chunk_count": 147
      }
    },
    {
      "id": "chunk-001",
      "label": "Chunk 1 (Page 15)",
      "type": "Chunk",
      "properties": {
        "page_number": 15,
        "content_preview": "VxRail is a hyper-converged..."
      }
    },
    {
      "id": "entity-vxrail",
      "label": "VxRail",
      "type": "Entity",
      "properties": {
        "entity_type": "Component",
        "importance": 0.95,
        "community_id": 5
      }
    }
  ],
  "edges": [
    {
      "source": "doc-001",
      "target": "chunk-001",
      "type": "CONTAINS"
    },
    {
      "source": "chunk-001",
      "target": "entity-vxrail",
      "type": "MENTIONS"
    },
    {
      "source": "entity-vxrail",
      "target": "entity-backup",
      "type": "RELATED_TO",
      "properties": {
        "strength": 0.85
      }
    }
  ],
  "communities": [
    {
      "id": 5,
      "size": 23,
      "summary": "VxRail infrastructure and backup procedures"
    }
  ],
  "total_nodes": 847,
  "total_edges": 2156
}
```

### Example

```bash
curl "http://localhost:8000/api/database/graph?limit=50&document_id=doc-001"
```

---

## GET /api/database/cache-stats

Get cache performance metrics.

### Request

**URL**: `GET /api/database/cache-stats`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "entity_label_cache": {
    "size": 1247,
    "max_size": 5000,
    "hits": 8934,
    "misses": 1247,
    "hit_rate": 0.877,
    "ttl_seconds": 300
  },
  "embedding_cache": {
    "size": 7823,
    "max_size": 10000,
    "hits": 15234,
    "misses": 8456,
    "hit_rate": 0.643
  },
  "retrieval_cache": {
    "size": 234,
    "max_size": 1000,
    "hits": 456,
    "misses": 1567,
    "hit_rate": 0.225,
    "ttl_seconds": 60
  },
  "total_memory_mb": 142.3
}
```

### Example

```bash
curl http://localhost:8000/api/database/cache-stats
```

---

## POST /api/database/reindex

Trigger database reindexing.

### Request

**URL**: `POST /api/database/reindex`

**Body**:
```json
{
  "rebuild_embeddings": false,
  "rebuild_similarities": true,
  "run_clustering": true
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `rebuild_embeddings` | bool | No | `false` | Regenerate all embeddings |
| `rebuild_similarities` | bool | No | `true` | Recompute SIMILAR_TO edges |
| `run_clustering` | bool | No | `true` | Run community detection |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "job_id": "reindex-job-123",
  "status": "running",
  "message": "Reindexing started"
}
```

### Example

```bash
curl -X POST http://localhost:8000/api/database/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "rebuild_similarities": true,
    "run_clustering": true
  }'
```

---

## DELETE /api/database/clear

Clear all data from database (dev only).

### Request

**URL**: `DELETE /api/database/clear`

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `confirm` | string | Yes | Must be `"yes"` to proceed |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Database cleared",
  "nodes_deleted": 8185,
  "relationships_deleted": 31126
}
```

### Example

```bash
curl -X DELETE "http://localhost:8000/api/database/clear?confirm=yes"
```

**Warning**: This operation is irreversible and should only be used in development.

---

## GET /api/database/health

Check database connection health.

### Request

**URL**: `GET /api/database/health`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "healthy",
  "neo4j": {
    "connected": true,
    "version": "5.13.0",
    "database": "neo4j",
    "latency_ms": 12
  },
  "redis": {
    "connected": true,
    "version": "7.0.11",
    "latency_ms": 3
  }
}
```

**Status**: `503 Service Unavailable` (if unhealthy)

**Body**:
```json
{
  "status": "unhealthy",
  "neo4j": {
    "connected": false,
    "error": "Connection refused"
  }
}
```

### Example

```bash
curl http://localhost:8000/api/database/health
```

---

## GET /api/database/folders

List all folders with document counts.

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "folders": [
    {
      "id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
      "name": "Release Notes",
      "created_at": 1717687421.02,
      "document_count": 4
    }
  ]
}
```

---

## POST /api/database/folders

Create a new folder.

### Request

**Body**:
```json
{
  "name": "Release Notes"
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
  "name": "Release Notes",
  "created_at": 1717687421.02,
  "document_count": 0
}
```

---

## PATCH /api/database/folders/{folder_id}

Rename a folder.

### Request

**Body**:
```json
{
  "name": "Updated Release Notes"
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
  "name": "Updated Release Notes",
  "created_at": 1717687421.02,
  "document_count": 4
}
```

---

## DELETE /api/database/folders/{folder_id}?mode=move_to_root|delete_documents

Delete a folder and choose what happens to its documents.

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "folder_id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
  "documents_deleted": 0,
  "documents_moved": 4
}
```

---

## PATCH /api/database/documents/{document_id}/folder

Move a document into a folder or back to root.

### Request

**Body**:
```json
{
  "folder_id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df"
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "document_id": "doc-123",
  "folder_id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
  "folder_name": "Release Notes",
  "folder_order": 3
}
```

---

## POST /api/database/documents/order

Persist manual ordering for documents within a folder or root.

### Request

**Body**:
```json
{
  "folder_id": "0c1f2c6e-9a90-4a2f-b1d8-88a0a7a7e6df",
  "document_ids": ["doc-1", "doc-2", "doc-3"]
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "updated": 3
}
```

---

## Related Documentation

- [Graph Database](03-components/backend/graph-database.md)
- [Graph Visualization](03-components/frontend/graph-visualization.md)
- [Community Detection](04-features/community-detection.md)
- [Caching System](02-core-concepts/caching.md)
