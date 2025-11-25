"""Simple in-memory reindex job manager used for background reindex operations.

This is intentionally lightweight for the test/dev environment. Jobs are
stored in a dict and updated by the background worker. For production,
replace with a durable job queue (Redis/RQ, Celery, or similar).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class JobRecord:
    job_id: str
    status: str
    message: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None


_jobs: Dict[str, JobRecord] = {}
_lock = threading.Lock()
_paused = False


def create_job(job_id: str, message: str = "queued") -> JobRecord:
    now = time.time()
    rec = JobRecord(job_id=job_id, status="queued", message=message, created_at=now)
    with _lock:
        _jobs[job_id] = rec
    return rec


def set_job_started(job_id: str):
    with _lock:
        rec = _jobs.get(job_id)
        if rec:
            rec.status = "running"
            rec.started_at = time.time()


def set_job_result(job_id: str, result: Dict[str, Any], status: str = "success", message: str | None = None):
    with _lock:
        rec = _jobs.get(job_id)
        if rec:
            rec.status = status
            rec.finished_at = time.time()
            rec.result = result
            rec.message = message or rec.message


def set_job_failed(job_id: str, message: str):
    with _lock:
        rec = _jobs.get(job_id)
        if rec:
            rec.status = "failed"
            rec.finished_at = time.time()
            rec.message = message


def get_job(job_id: str) -> Optional[JobRecord]:
    with _lock:
        rec = _jobs.get(job_id)
        return asdict(rec) if rec else None



def list_jobs(limit: int | None = None, offset: int = 0) -> list[Dict[str, Any]]:
    """Paginated listing for in-memory jobs.

    Note: maintains newest-first ordering.
    """
    with _lock:
        items = [asdict(r) for r in _jobs.values()]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    if limit is None:
        return items[offset:]
    return items[offset : offset + limit]


def count_jobs() -> int:
    with _lock:
        return len(_jobs)


def cancel_job(job_id: str) -> bool:
    with _lock:
        rec = _jobs.get(job_id)
        if not rec:
            return False
        rec.status = "cancelled"
        rec.message = "cancelled by user"
    return True


def cancel_all() -> int:
    """Mark all known jobs as cancelled and return how many were updated."""
    with _lock:
        count = 0
        for rec in _jobs.values():
            if rec.status not in ("finished", "failed", "cancelled"):
                rec.status = "cancelled"
                rec.message = "cancelled by admin"
                count += 1
    return count


def purge_jobs() -> int:
    """Remove all jobs from in-memory store. Returns number removed."""
    with _lock:
        n = len(_jobs)
        _jobs.clear()
    return n


def retry_job(job_id: str) -> bool:
    with _lock:
        rec = _jobs.get(job_id)
        if not rec:
            return False
        # only retry if not currently running
        if rec.status == "running":
            return False
        rec.status = "queued"
        rec.attempts = 0 if not hasattr(rec, "attempts") else getattr(rec, "attempts", 0)
    return True


def pause_queue():
    global _paused
    with _lock:
        _paused = True


def resume_queue():
    global _paused
    with _lock:
        _paused = False


def is_paused() -> bool:
    with _lock:
        return bool(_paused)


def requeue_stuck_jobs(older_than_seconds: int = 300, max_retries: int = 5) -> int:
    # In-memory manager doesn't support worker interruption handling beyond no-op
    return 0
