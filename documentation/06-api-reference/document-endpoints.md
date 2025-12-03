# Document Endpoints

Document management operations and metadata retrieval.

## GET /api/documents

List all documents in the knowledge base.

### Request

**URL**: `GET /api/documents?limit=20&offset=0&search=VxRail`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | int | No | `20` | Max documents to return |
| `offset` | int | No | `0` | Number of documents to skip |
| `search` | string | No | - | Search term for filtering |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "documents": [
    {
      "id": "doc-001",
      "filename": "VxRail_Admin_Guide.pdf",
      "title": "VxRail Administration Guide",
      "page_count": 350,
      "chunk_count": 147,
      "entity_count": 83,
      "upload_date": "2025-12-01T10:00:00Z",
      "file_size": 2457600,
      "file_type": "pdf"
    }
  ],
  "total": 47,
  "limit": 20,
  "offset": 0,
  "has_more": true
}
```

### Example

```bash
curl http://localhost:8000/api/documents?limit=10
```

---

## POST /api/upload

Upload document for processing.

### Request

**URL**: `POST /api/upload`

**Headers**:
```http
Content-Type: multipart/form-data
```

**Body** (form-data):
```
file: <binary file data>
```

**Supported Formats**:
- PDF: `.pdf`
- Word: `.docx`
- Text: `.txt`, `.md`
- Presentations: `.pptx`
- Spreadsheets: `.xlsx`, `.csv`
- Images: `.jpg`, `.jpeg`, `.png` (with OCR)

**Size Limit**: 50MB

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "job_id": "job-abc-123",
  "filename": "VxRail_Admin_Guide.pdf",
  "status": "processing"
}
```

### Example

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/document.pdf"
```

```python
import httpx

async with httpx.AsyncClient() as client:
    with open('document.pdf', 'rb') as f:
        response = await client.post(
            'http://localhost:8000/api/upload',
            files={'file': f},
        )
    
    result = response.json()
    job_id = result['job_id']
```

### Error Responses

**400 Bad Request** (unsupported format):
```json
{
  "detail": "Unsupported file type: .exe"
}
```

**400 Bad Request** (file too large):
```json
{
  "detail": "File too large (max 50MB)"
}
```

---

## GET /api/documents/{id}

Get document metadata.

### Request

**URL**: `GET /api/documents/{id}`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Document ID |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "id": "doc-001",
  "filename": "VxRail_Admin_Guide.pdf",
  "title": "VxRail Administration Guide",
  "page_count": 350,
  "chunk_count": 147,
  "entity_count": 83,
  "upload_date": "2025-12-01T10:00:00Z",
  "file_size": 2457600,
  "file_type": "pdf",
  "metadata": {
    "author": "Dell EMC",
    "created_date": "2024-06-15"
  }
}
```

### Error Responses

**404 Not Found**:
```json
{
  "detail": "Document not found"
}
```

---

## GET /api/documents/{id}/chunks

Get document chunks with pagination.

### Request

**URL**: `GET /api/documents/{id}/chunks?limit=10&offset=0`

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `id` | string | Yes | - | Document ID |
| `limit` | int | No | `10` | Max chunks to return |
| `offset` | int | No | `0` | Number of chunks to skip |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "document_id": "doc-001",
  "chunks": [
    {
      "chunk_id": "chunk-001",
      "content": "VxRail is a hyper-converged infrastructure...",
      "page_number": 15,
      "position": 0,
      "chunk_size": 987,
      "entities": ["VxRail", "Infrastructure"]
    },
    {
      "chunk_id": "chunk-002",
      "content": "infrastructure appliance that integrates...",
      "page_number": 15,
      "position": 1,
      "chunk_size": 1024,
      "entities": ["VxRail", "Compute", "Storage"]
    }
  ],
  "total": 147,
  "limit": 10,
  "offset": 0,
  "has_more": true
}
```

### Example

```bash
curl "http://localhost:8000/api/documents/doc-001/chunks?limit=3&offset=0"
```

---

## DELETE /api/documents/{id}

Delete document and all associated data.

### Request

**URL**: `DELETE /api/documents/{id}`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `id` | string | Yes | Document ID |

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "message": "Document deleted successfully",
  "document_id": "doc-001",
  "chunks_deleted": 147,
  "entities_affected": 83
}
```

### Example

```bash
curl -X DELETE http://localhost:8000/api/documents/doc-001
```

### Error Responses

**404 Not Found**:
```json
{
  "detail": "Document not found"
}
```

---

## GET /api/documents/processing-status

Get processing status for all documents.

### Request

**URL**: `GET /api/documents/processing-status`

### Response

**Status**: `200 OK`

**Body**:
```json
{
  "processing": [
    {
      "document_id": "doc-002",
      "filename": "vSphere_Guide.pdf",
      "status": "running",
      "progress": 65,
      "message": "Extracting entities"
    }
  ],
  "completed": 15,
  "failed": 0,
  "total": 16
}
```

---

## Related Documentation

- [Document Upload](04-features/document-upload.md)
- [Document Processor](03-components/ingestion/document-processor.md)
- [Document Ingestion Flow](05-data-flows/document-ingestion-flow.md)
- [Jobs API](06-api-reference/jobs-endpoints.md)
