# Document Upload Feature

Multi-format document ingestion with progress tracking and validation.

## Overview

Document Upload enables users to ingest documents into Amber's knowledge graph through a web interface or API. It supports multiple file formats, provides real-time progress tracking, validates uploads, and manages background processing with status updates.

**Supported Formats**:
- PDF, DOCX, TXT, MD
- PPTX, XLSX, CSV
- Images (with OCR)

**Key Features**:
- Drag-and-drop upload
- Multi-file batch processing
- Progress tracking and status updates
- File validation and size limits
- Background processing with job tracking

## Architecture

```
┌────────────────────────────────────────────────────────┐
│          Document Upload Architecture                   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Frontend Upload Flow                   │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ 1. User selects file(s)                   │  │   │
│  │  │ 2. Client validates (size, format)        │  │   │
│  │  │ 3. POST /api/upload (multipart/form-data) │  │   │
│  │  │ 4. Track upload progress (XHR)            │  │   │
│  │  │ 5. Receive job_id                         │  │   │
│  │  │ 6. Poll /api/jobs/{job_id} for status     │  │   │
│  │  │ 7. Display completion                     │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Backend Processing Pipeline            │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Upload Endpoint                           │  │   │
│  │  │   ├─ Receive file                         │  │   │
│  │  │   ├─ Validate format & size               │  │   │
│  │  │   ├─ Save to staging directory            │  │   │
│  │  │   └─ Create processing job                │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  │    ↓                                              │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ Background Worker                         │  │   │
│  │  │   ├─ Load document                        │  │   │
│  │  │   ├─ Chunk text                           │  │   │
│  │  │   ├─ Generate embeddings                  │  │   │
│  │  │   ├─ Extract entities (optional)          │  │   │
│  │  │   ├─ Persist to Neo4j                     │  │   │
│  │  │   └─ Update job status                    │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │          Job Status Tracking                    │   │
│  │  ┌───────────────────────────────────────────┐  │   │
│  │  │ States:                                   │  │   │
│  │  │   • pending: Queued for processing        │  │   │
│  │  │   • running: Active processing            │  │   │
│  │  │   • completed: Successfully ingested      │  │   │
│  │  │   • failed: Error occurred                │  │   │
│  │  │                                            │  │   │
│  │  │ Progress: 0-100% with stage messages      │  │   │
│  │  └───────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└────────────────────────────────────────────────────────┘
```

## Backend Implementation

### Upload Endpoint

```python
# api/routers/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import hashlib
from api.job_manager import get_job_manager
from ingestion.document_processor import DocumentProcessor

router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".pptx", ".xlsx", ".csv",
    ".jpg", ".jpeg", ".png",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process document.
    
    Args:
        file: Uploaded file (multipart/form-data)
    
    Returns:
        Job ID for tracking processing status
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}",
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large (max {MAX_FILE_SIZE / 1024 / 1024}MB)",
        )
    
    # Generate unique filename
    file_hash = hashlib.md5(content).hexdigest()
    staged_filename = f"{file_hash}_{file.filename}"
    
    # Save to staging directory
    staging_dir = Path("data/staged_uploads")
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    staged_path = staging_dir / staged_filename
    staged_path.write_bytes(content)
    
    # Create background processing job
    job_manager = get_job_manager()
    job_id = await job_manager.create_job(
        job_type="document_ingestion",
        params={
            "file_path": str(staged_path),
            "filename": file.filename,
        },
    )
    
    # Start background processing
    await process_document_background(job_id, str(staged_path), file.filename)
    
    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "processing",
    }
```

### Background Processing

```python
# api/routers/documents.py (continued)
import asyncio
from typing import Optional

async def process_document_background(
    job_id: str,
    file_path: str,
    filename: str,
):
    """
    Process document in background thread.
    
    Args:
        job_id: Job tracking ID
        file_path: Path to staged file
        filename: Original filename
    """
    job_manager = get_job_manager()
    processor = DocumentProcessor()
    
    async def _process():
        try:
            # Update status: running
            await job_manager.update_job(
                job_id,
                status="running",
                progress=0,
                message="Starting document processing",
            )
            
            # Process document with progress callbacks
            result = await processor.process_document_async(
                file_path=file_path,
                filename=filename,
                progress_callback=lambda p, msg: asyncio.create_task(
                    job_manager.update_job(job_id, progress=p, message=msg)
                ),
            )
            
            # Update status: completed
            await job_manager.update_job(
                job_id,
                status="completed",
                progress=100,
                message="Processing complete",
                result={
                    "document_id": result["document_id"],
                    "chunk_count": result["chunk_count"],
                    "entity_count": result.get("entity_count", 0),
                },
            )
            
            # Cleanup staged file
            Path(file_path).unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            
            await job_manager.update_job(
                job_id,
                status="failed",
                message=f"Processing failed: {str(e)}",
                error=str(e),
            )
            
            # Cleanup on error
            Path(file_path).unlink(missing_ok=True)
    
    # Run in background
    asyncio.create_task(_process())
```

### Job Status Endpoint

```python
# api/routers/jobs.py
from fastapi import APIRouter, HTTPException
from api.job_manager import get_job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """
    Get processing job status.
    
    Args:
        job_id: Job identifier
    
    Returns:
        Job status with progress information
    """
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
    }
```

## Frontend Implementation

### Upload Component

```typescript
// frontend/src/components/upload/DocumentUpload.tsx
'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';
import { uploadDocument, pollJobStatus } from '@/lib/api-client';

interface UploadFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  message?: string;
  error?: string;
  jobId?: string;
}

export function DocumentUpload() {
  const [files, setFiles] = useState<UploadFile[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadFile[] = acceptedFiles.map((file) => ({
      id: `${Date.now()}-${file.name}`,
      file,
      status: 'pending',
      progress: 0,
    }));

    setFiles((prev) => [...prev, ...newFiles]);

    // Start uploads
    newFiles.forEach((uploadFile) => handleUpload(uploadFile));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/csv': ['.csv'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
    },
    maxSize: 50 * 1024 * 1024, // 50MB
  });

  const handleUpload = async (uploadFile: UploadFile) => {
    try {
      // Update status: uploading
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? { ...f, status: 'uploading', progress: 0 }
            : f
        )
      );

      // Upload file with progress
      const result = await uploadDocument(
        uploadFile.file,
        (progress) => {
          setFiles((prev) =>
            prev.map((f) =>
              f.id === uploadFile.id
                ? { ...f, progress: Math.round(progress / 2) } // 0-50% for upload
                : f
            )
          );
        }
      );

      // Update status: processing
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? {
                ...f,
                status: 'processing',
                jobId: result.job_id,
                progress: 50,
                message: 'Processing document...',
              }
            : f
        )
      );

      // Poll job status
      await pollJobStatus(
        result.job_id,
        (status) => {
          setFiles((prev) =>
            prev.map((f) =>
              f.id === uploadFile.id
                ? {
                    ...f,
                    progress: 50 + Math.round(status.progress / 2), // 50-100%
                    message: status.message,
                  }
                : f
            )
          );
        }
      );

      // Update status: completed
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? { ...f, status: 'completed', progress: 100 }
            : f
        )
      );
    } catch (error) {
      // Update status: error
      setFiles((prev) =>
        prev.map((f) =>
          f.id === uploadFile.id
            ? {
                ...f,
                status: 'error',
                error: error instanceof Error ? error.message : 'Upload failed',
              }
            : f
        )
      );
    }
  };

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
          isDragActive
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
            : 'border-neutral-300 hover:border-primary-400 dark:border-neutral-700'
        }`}
      >
        <input {...getInputProps()} />
        
        <Upload className="mx-auto mb-4 h-12 w-12 text-neutral-400" />
        
        {isDragActive ? (
          <p className="text-primary-600 dark:text-primary-400">
            Drop files here...
          </p>
        ) : (
          <>
            <p className="mb-2 text-neutral-700 dark:text-neutral-300">
              Drag & drop files here, or click to select
            </p>
            <p className="text-sm text-neutral-500">
              Supports: PDF, DOCX, TXT, MD, PPTX, XLSX, CSV, Images
            </p>
            <p className="text-xs text-neutral-400">Max size: 50MB per file</p>
          </>
        )}
      </div>

      {/* Upload list */}
      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file) => (
            <UploadFileItem
              key={file.id}
              file={file}
              onRemove={() => removeFile(file.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
```

### Upload File Item

```typescript
// frontend/src/components/upload/UploadFileItem.tsx
interface UploadFileItemProps {
  file: UploadFile;
  onRemove: () => void;
}

function UploadFileItem({ file, onRemove }: UploadFileItemProps) {
  const getStatusIcon = () => {
    switch (file.status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-success-500" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-error-500" />;
      default:
        return <File className="h-5 w-5 text-neutral-500" />;
    }
  };

  const getStatusColor = () => {
    switch (file.status) {
      case 'completed':
        return 'bg-success-500';
      case 'error':
        return 'bg-error-500';
      case 'uploading':
      case 'processing':
        return 'bg-primary-500';
      default:
        return 'bg-neutral-300';
    }
  };

  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <div className="flex items-start gap-3">
        {getStatusIcon()}
        
        <div className="flex-1 space-y-2">
          <div className="flex items-start justify-between">
            <div>
              <p className="font-medium text-neutral-900 dark:text-neutral-100">
                {file.file.name}
              </p>
              {file.message && (
                <p className="text-sm text-neutral-500">{file.message}</p>
              )}
              {file.error && (
                <p className="text-sm text-error-600 dark:text-error-400">
                  {file.error}
                </p>
              )}
            </div>
            
            {file.status === 'completed' || file.status === 'error' ? (
              <button
                onClick={onRemove}
                className="rounded p-1 hover:bg-neutral-100 dark:hover:bg-neutral-800"
              >
                <X className="h-4 w-4 text-neutral-500" />
              </button>
            ) : null}
          </div>
          
          {/* Progress bar */}
          {(file.status === 'uploading' || file.status === 'processing') && (
            <div className="h-2 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
              <div
                className={`h-full transition-all ${getStatusColor()}`}
                style={{ width: `${file.progress}%` }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

## Validation

### File Validation

```python
# api/utils/file_validation.py
from pathlib import Path
from typing import Optional

def validate_file(
    filename: str,
    content: bytes,
    max_size: int = 50 * 1024 * 1024,
) -> Optional[str]:
    """
    Validate uploaded file.
    
    Args:
        filename: Original filename
        content: File content bytes
        max_size: Maximum file size in bytes
    
    Returns:
        Error message if invalid, None if valid
    """
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"Unsupported file type: {ext}"
    
    # Check size
    if len(content) > max_size:
        return f"File too large (max {max_size / 1024 / 1024}MB)"
    
    # Check for empty file
    if len(content) == 0:
        return "Empty file"
    
    return None
```

## Configuration

### Upload Settings

```python
# config/settings.py
class Settings(BaseSettings):
    # Upload limits
    max_upload_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: List[str] = [
        ".pdf", ".docx", ".txt", ".md",
        ".pptx", ".xlsx", ".csv",
        ".jpg", ".jpeg", ".png",
    ]
    
    # Storage
    staging_dir: str = "data/staged_uploads"
    documents_dir: str = "data/documents"
    
    # Processing
    background_processing: bool = True
    job_timeout: int = 600  # 10 minutes
```

## Testing

### Upload Tests

```python
# tests/test_upload.py
import pytest
from fastapi.testclient import TestClient
from api.main import app
from io import BytesIO

client = TestClient(app)

def test_upload_pdf():
    """Test PDF upload."""
    content = b"%PDF-1.4 mock content"
    files = {"file": ("test.pdf", BytesIO(content), "application/pdf")}
    
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 200
    assert "job_id" in response.json()

def test_upload_unsupported_format():
    """Test rejection of unsupported format."""
    content = b"test content"
    files = {"file": ("test.exe", BytesIO(content), "application/x-executable")}
    
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

def test_upload_too_large():
    """Test rejection of oversized file."""
    content = b"x" * (51 * 1024 * 1024)  # 51MB
    files = {"file": ("large.pdf", BytesIO(content), "application/pdf")}
    
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()
```

## Troubleshooting

### Common Issues

**Issue**: Upload fails with timeout
```python
# Solution: Increase timeout
job_timeout = 1200  # 20 minutes

# Or process in chunks for large files
chunk_size = 1024 * 1024  # 1MB chunks
```

**Issue**: Out of disk space
```python
# Solution: Clean up staged files
import shutil
from pathlib import Path

staging_dir = Path("data/staged_uploads")
for file in staging_dir.glob("*"):
    if file.stat().st_mtime < time.time() - 3600:  # 1 hour old
        file.unlink()
```

**Issue**: Job status stuck
```python
# Solution: Implement job timeout
async def check_stale_jobs():
    jobs = await job_manager.list_jobs(status="running")
    
    for job in jobs:
        age = time.time() - job["created_at"]
        if age > settings.job_timeout:
            await job_manager.update_job(
                job["id"],
                status="failed",
                error="Job timeout",
            )
```

## Related Documentation

- [Document Processor](03-components/ingestion/document-processor.md)
- [Loaders](03-components/ingestion/loaders.md)
- [Job Management](03-components/backend/job-management.md)
- [Documents API](06-api-reference/documents.md)
