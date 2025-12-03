# Job Management Component

Background job tracking, progress monitoring, and status updates.

## Overview

The job management component provides infrastructure for tracking long-running background tasks such as document ingestion, entity extraction, reindexing, and clustering. It maintains job status, progress, and results through Redis or in-memory storage, and exposes REST endpoints for monitoring.

**Location**: `api/job_manager.py`, `api/redis_job_manager.py`
**Storage**: Redis (production) or in-memory (development)
**API**: `/api/jobs/*` endpoints

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Job Management System                   │
├──────────────────────────────────────────────────┤
│                                                   │
│  Job Lifecycle                                    │
│  ┌─────────────────────────────────────────────┐ │
│  │  pending → running → completed/failed       │ │
│  │                                              │ │
│  │  • Progress updates (0.0 - 1.0)             │ │
│  │  • Status messages                          │ │
│  │  • Error capture                            │ │
│  │  • Result storage                           │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Storage Backend                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  RedisJobManager (Production)               │ │
│  │  • Persistent across restarts               │ │
│  │  • Shared across workers                    │ │
│  │  • TTL-based cleanup                        │ │
│  │                                              │ │
│  │  InMemoryJobManager (Development)           │ │
│  │  • Simple dict storage                      │ │
│  │  • No external dependencies                 │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  API Endpoints                                    │
│  ┌─────────────────────────────────────────────┐ │
│  │  GET  /api/jobs                             │ │
│  │  GET  /api/jobs/{job_id}                    │ │
│  │  POST /api/jobs/{job_id}/cancel             │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
└──────────────────────────────────────────────────┘
```

## Job Model

### Job Status Enum

```python
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### Job Data Structure

```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class Job:
    """Background job representation."""
    job_id: str
    job_type: str           # "ingestion", "extraction", "reindex", "clustering"
    status: JobStatus
    progress: float         # 0.0 - 1.0
    message: str            # Current status message
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format
        for field in ["created_at", "started_at", "completed_at"]:
            if data[field]:
                data[field] = data[field].isoformat()
        return data
```

## Job Manager Interface

### Base Abstract Class

```python
from abc import ABC, abstractmethod

class JobManager(ABC):
    """Abstract base class for job management."""
    
    @abstractmethod
    def create_job(
        self,
        job_id: str,
        job_type: str,
        metadata: Optional[Dict] = None
    ) -> Job:
        """Create a new job."""
        pass
    
    @abstractmethod
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        pass
    
    @abstractmethod
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict] = None
    ):
        """Update job status and progress."""
        pass
    
    @abstractmethod
    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filtering."""
        pass
    
    @abstractmethod
    def delete_job(self, job_id: str):
        """Delete a job."""
        pass
```

## In-Memory Implementation

### InMemoryJobManager

```python
from typing import Dict, List, Optional
import uuid
from datetime import datetime

class InMemoryJobManager(JobManager):
    """In-memory job storage for development."""
    
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
    
    def create_job(
        self,
        job_id: Optional[str] = None,
        job_type: str = "generic",
        metadata: Optional[Dict] = None
    ) -> Job:
        """Create a new job."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            progress=0.0,
            message="Job created",
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job by ID."""
        return self.jobs.get(job_id)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict] = None
    ):
        """Update job status and progress."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        if status:
            job.status = status
            
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
        
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        
        if message:
            job.message = message
        
        if error:
            job.error = error
        
        if result:
            job.result = result
    
    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs with optional filtering."""
        jobs = list(self.jobs.values())
        
        # Filter by type
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        
        # Filter by status
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    def delete_job(self, job_id: str):
        """Delete a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
```

## Redis Implementation

### RedisJobManager

```python
import redis
import json
from typing import Optional, List, Dict

class RedisJobManager(JobManager):
    """Redis-backed job storage for production."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.key_prefix = "job:"
        self.index_key = "jobs:index"
        self.ttl = 86400 * 7  # 7 days
    
    def _make_key(self, job_id: str) -> str:
        """Generate Redis key for job."""
        return f"{self.key_prefix}{job_id}"
    
    def create_job(
        self,
        job_id: Optional[str] = None,
        job_type: str = "generic",
        metadata: Optional[Dict] = None
    ) -> Job:
        """Create a new job in Redis."""
        if job_id is None:
            job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.PENDING,
            progress=0.0,
            message="Job created",
            created_at=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Store in Redis
        key = self._make_key(job_id)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(job.to_dict())
        )
        
        # Add to index
        self.redis.zadd(
            self.index_key,
            {job_id: job.created_at.timestamp()}
        )
        
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Retrieve job from Redis."""
        key = self._make_key(job_id)
        data = self.redis.get(key)
        
        if not data:
            return None
        
        job_dict = json.loads(data)
        
        # Convert datetime strings back to datetime objects
        for field in ["created_at", "started_at", "completed_at"]:
            if job_dict[field]:
                job_dict[field] = datetime.fromisoformat(job_dict[field])
        
        return Job(**job_dict)
    
    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict] = None
    ):
        """Update job in Redis."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        # Apply updates
        if status:
            job.status = status
            
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.utcnow()
            
            if status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.utcnow()
        
        if progress is not None:
            job.progress = max(0.0, min(1.0, progress))
        
        if message:
            job.message = message
        
        if error:
            job.error = error
        
        if result:
            job.result = result
        
        # Save back to Redis
        key = self._make_key(job_id)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(job.to_dict())
        )
    
    def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs from Redis."""
        # Get job IDs from index (sorted by timestamp, newest first)
        job_ids = self.redis.zrevrange(self.index_key, 0, limit - 1)
        
        jobs = []
        for job_id in job_ids:
            job = self.get_job(job_id)
            if job:
                # Apply filters
                if job_type and job.job_type != job_type:
                    continue
                if status and job.status != status:
                    continue
                
                jobs.append(job)
        
        return jobs
    
    def delete_job(self, job_id: str):
        """Delete job from Redis."""
        key = self._make_key(job_id)
        self.redis.delete(key)
        self.redis.zrem(self.index_key, job_id)
```

## Singleton Access

### Global Job Manager

```python
from config.settings import settings

# Global instance
_job_manager: Optional[JobManager] = None

def get_job_manager() -> JobManager:
    """Get global job manager instance."""
    global _job_manager
    
    if _job_manager is None:
        if settings.redis_url:
            _job_manager = RedisJobManager(settings.redis_url)
        else:
            _job_manager = InMemoryJobManager()
    
    return _job_manager
```

## API Endpoints

### Router Definition

```python
# api/routers/jobs.py
from fastapi import APIRouter, HTTPException
from api.job_manager import get_job_manager, JobStatus

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.get("")
def list_jobs(
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """List all jobs with optional filtering."""
    manager = get_job_manager()
    
    status_enum = JobStatus(status) if status else None
    
    jobs = manager.list_jobs(
        job_type=job_type,
        status=status_enum,
        limit=limit
    )
    
    return {
        "jobs": [job.to_dict() for job in jobs],
        "total": len(jobs)
    }

@router.get("/{job_id}")
def get_job(job_id: str):
    """Get job details."""
    manager = get_job_manager()
    job = manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_dict()

@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    """Cancel a running job."""
    manager = get_job_manager()
    job = manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status {job.status}"
        )
    
    manager.update_job(
        job_id=job_id,
        status=JobStatus.CANCELLED,
        message="Job cancelled by user"
    )
    
    return {"message": "Job cancelled", "job_id": job_id}

@router.delete("/{job_id}")
def delete_job(job_id: str):
    """Delete a job."""
    manager = get_job_manager()
    manager.delete_job(job_id)
    
    return {"message": "Job deleted", "job_id": job_id}
```

## Usage Examples

### Document Ingestion Job

```python
from api.job_manager import get_job_manager, JobStatus

async def ingest_document_with_job(file_path: str) -> str:
    """Ingest document with job tracking."""
    manager = get_job_manager()
    
    # Create job
    job = manager.create_job(
        job_type="ingestion",
        metadata={
            "file_path": file_path,
            "filename": os.path.basename(file_path)
        }
    )
    
    job_id = job.job_id
    
    try:
        # Update to running
        manager.update_job(
            job_id=job_id,
            status=JobStatus.RUNNING,
            message="Starting document ingestion"
        )
        
        # Load document
        manager.update_job(job_id, progress=0.1, message="Loading document")
        doc_processor = DocumentProcessor()
        doc_data = doc_processor.load_document(file_path)
        
        # Chunk document
        manager.update_job(job_id, progress=0.3, message="Chunking document")
        chunks = doc_processor.chunk_document(doc_data)
        
        # Generate embeddings
        manager.update_job(job_id, progress=0.5, message="Generating embeddings")
        from core.embeddings import EmbeddingManager
        emb_manager = EmbeddingManager()
        chunks = await emb_manager.process_with_concurrency(
            items=chunks,
            text_extractor=lambda c: c["text"]
        )
        
        # Persist to database
        manager.update_job(job_id, progress=0.8, message="Persisting to database")
        from core.graph_db import get_db
        db = get_db()
        db.create_document(doc_data)
        db.create_chunks_batch(doc_data["id"], chunks)
        
        # Complete
        manager.update_job(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Document ingestion complete",
            result={
                "document_id": doc_data["id"],
                "chunk_count": len(chunks)
            }
        )
        
        return job_id
    
    except Exception as e:
        # Mark failed
        manager.update_job(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e),
            message=f"Ingestion failed: {str(e)}"
        )
        
        raise
```

### Entity Extraction Job

```python
async def extract_entities_with_job(document_id: str, chunks: List[Dict]) -> str:
    """Extract entities with job tracking."""
    manager = get_job_manager()
    
    job = manager.create_job(
        job_type="extraction",
        metadata={"document_id": document_id}
    )
    
    job_id = job.job_id
    
    try:
        manager.update_job(
            job_id=job_id,
            status=JobStatus.RUNNING,
            message="Extracting entities"
        )
        
        # Extract
        from core.entity_extraction import extract_entities_batch
        entities, relationships = await extract_entities_batch(chunks)
        
        manager.update_job(job_id, progress=0.6, message="Adding embeddings")
        
        # Add embeddings
        from core.entity_extraction import add_entity_embeddings
        entities = await add_entity_embeddings(entities)
        
        manager.update_job(job_id, progress=0.9, message="Persisting entities")
        
        # Persist
        from core.entity_extraction import persist_entities_batch, persist_relationships_batch
        for chunk in chunks:
            chunk_entities = [e for e in entities if chunk["id"] in e["chunk_ids"]]
            persist_entities_batch(chunk_entities, chunk["id"])
        
        persist_relationships_batch(relationships)
        
        manager.update_job(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Entity extraction complete",
            result={
                "entity_count": len(entities),
                "relationship_count": len(relationships)
            }
        )
        
        return job_id
    
    except Exception as e:
        manager.update_job(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e)
        )
        raise
```

## Configuration

### Environment Variables

```bash
# Redis (optional, defaults to in-memory)
REDIS_URL=redis://localhost:6379/0

# Job TTL (seconds)
JOB_TTL=604800  # 7 days
```

### Settings

```python
from config.settings import settings

# Check Redis availability
if settings.redis_url:
    manager = RedisJobManager(settings.redis_url)
else:
    manager = InMemoryJobManager()
```

## Testing

### Unit Tests

```python
import pytest
from api.job_manager import InMemoryJobManager, JobStatus

@pytest.fixture
def manager():
    return InMemoryJobManager()

def test_create_job(manager):
    job = manager.create_job(job_type="test")
    
    assert job.job_id
    assert job.status == JobStatus.PENDING
    assert job.progress == 0.0

def test_update_job(manager):
    job = manager.create_job(job_type="test")
    
    manager.update_job(
        job.job_id,
        status=JobStatus.RUNNING,
        progress=0.5,
        message="In progress"
    )
    
    updated = manager.get_job(job.job_id)
    assert updated.status == JobStatus.RUNNING
    assert updated.progress == 0.5
    assert updated.message == "In progress"

def test_list_jobs(manager):
    manager.create_job(job_type="ingestion")
    manager.create_job(job_type="extraction")
    manager.create_job(job_type="ingestion")
    
    all_jobs = manager.list_jobs()
    assert len(all_jobs) == 3
    
    ingestion_jobs = manager.list_jobs(job_type="ingestion")
    assert len(ingestion_jobs) == 2
```

## Related Documentation

- [API Layer](03-components/backend/api-layer.md)
- [Document Processor](03-components/ingestion/document-processor.md)
- [Entity Extraction](03-components/backend/entity-extraction.md)
- [Reindexing](08-operations/reindexing.md)
