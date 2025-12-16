# Graph Editor Endpoints

Endpoints for manual and AI-assisted graph curation, including healing, edge management, and backups.

## POST /api/graph/editor/heal

Trigger AI Graph Healing to find missing connections for a specific node.

### Request

**URL**: `POST /api/graph/editor/heal`
**Auth**: Required (Admin)

**Body**:
```json
{
  "node_id": "node-123"
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "suggestions": [
    {
      "id": "node-456",
      "name": "Related Entity",
      "score": 0.89,
      "type": "Concept"
    },
    ...
  ]
}
```

---

## POST /api/graph/editor/edge

Create a new relationship betwen two entities.

### Request

**URL**: `POST /api/graph/editor/edge`
**Auth**: Required (Admin)

**Body**:
```json
{
  "source_id": "node-123",
  "target_id": "node-456",
  "relation_type": "RELATED_TO",
  "properties": {
    "source": "manual_curation",
    "confidence": 1.0
  }
}
```

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "status": "success",
  "edge_id": "edge-789"
}
```

---

## DELETE /api/graph/editor/edge

Remove an existing relationship.

### Request

**URL**: `DELETE /api/graph/editor/edge`
**Auth**: Required (Admin)
**Query Parameters**: `source_id`, `target_id`, `type`

**Example**:
`DELETE /api/graph/editor/edge?source_id=node-A&target_id=node-B&type=RELATED_TO`

### Response

**Status**: `200 OK`

---

## POST /api/graph/editor/nodes/merge

Merge multiple source nodes into a single target node.

**Behavior**:
1. All edges `to`/`from` source nodes are moved `to`/`from` the target node.
2. Source descriptions are appended to the target node's description.
3. Source nodes are permanently deleted.

### Request

**URL**: `POST /api/graph/editor/nodes/merge`
**Auth**: Required (Admin)

**Body**:
```json
{
  "target_id": "node-primary",
  "source_ids": ["node-duplicate-1", "node-duplicate-2"]
}
```

### Response

**Status**: `200 OK`

---

## PATCH /api/graph/editor/node

Update node properties (e.g. rename).

### Request

**URL**: `PATCH /api/graph/editor/node`
**Auth**: Required (Admin)

**Body**:
```json
{
  "node_id": "node-123",
  "properties": {
    "name": "New Name",
    "type": "NewType"
  }
}
```

### Response

**Status**: `200 OK`

---

## GET /api/graph/editor/snapshot

Download a full backup of the graph (Nodes + Edges) as JSON.

### Request

**URL**: `GET /api/graph/editor/snapshot`
**Auth**: Required (Admin)

### Response

**Status**: `200 OK`
**Content-Type**: `application/json`
**Body**: (Large JSON object containing `nodes` and `edges` arrays)

---

## POST /api/graph/editor/restore

Restore the graph from a JSON snapshot. **Destructive Action**: Wipes existing graph first.

### Request

**URL**: `POST /api/graph/editor/restore`
**Auth**: Required (Admin)
**Content-Type**: `multipart/form-data`

**Body**:
- `file`: (The JSON file)

### Response

**Status**: `200 OK`

---

## GET /api/graph/editor/orphans

Detect orphan nodes that are disconnected from the main graph.

**Behavior**:
Returns entity IDs that either:
1. Have no relationships (MENTIONS or RELATED_TO), OR
2. Are not connected to any Document node within 3 hops

### Request

**URL**: `GET /api/graph/editor/orphans`
**Auth**: Required (Admin)

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "orphan_ids": ["node-123", "node-456", "node-789"]
}
```

---

## Error Responses

All endpoints may return:

| Status | Description |
|--------|-------------|
| 401 | Unauthorized - Missing or invalid admin token |
| 404 | Node not found |
| 422 | Validation error - Missing required fields |
| 500 | Internal server error |
