from __future__ import annotations

from fastapi import APIRouter, HTTPException
from typing import Optional

from api import job_manager
from api import auth
from fastapi import Depends

router = APIRouter()


@router.get("", summary="List jobs")
def list_jobs(page: int = 1, page_size: int = 50, _token: str = Depends(auth.require_user_or_admin)):
    if page < 1 or page_size < 1:
        raise HTTPException(status_code=400, detail="page and page_size must be >= 1")
    offset = (page - 1) * page_size
    try:
        total = job_manager.count_jobs()
        items = job_manager.list_jobs(limit=page_size, offset=offset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"jobs": items, "page": page, "page_size": page_size, "total": int(total)}



@router.get("/token", summary="Get persistent user token (admin only)")
def get_user_token(_admin: str = Depends(auth.require_admin)):
    try:
        token = auth.ensure_user_token()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"user_token": token}


@router.get("/{job_id}", summary="Get job details")
def get_job(job_id: str, _token: str = Depends(auth.require_user_or_admin)):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.post("/pause", summary="Pause job processing")
def pause_jobs(_admin: str = Depends(auth.require_admin)):
    try:
        job_manager.pause_queue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "paused"}


@router.post("/resume", summary="Resume job processing")
def resume_jobs(_admin: str = Depends(auth.require_admin)):
    try:
        job_manager.resume_queue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "running"}


@router.post("/requeue", summary="Requeue stuck jobs")
def requeue_jobs(older_than_seconds: Optional[int] = 300, max_retries: Optional[int] = 5, _admin: str = Depends(auth.require_admin)):
    try:
        count = job_manager.requeue_stuck_jobs(older_than_seconds=older_than_seconds, max_retries=max_retries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"requeued": int(count)}


@router.post("/{job_id}/cancel", summary="Cancel a job")
def cancel_job(job_id: str, _admin: str = Depends(auth.require_admin)):
    try:
        ok = job_manager.cancel_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="job not found")
    return {"status": "cancelled", "job_id": job_id}



@router.post("/cancel-all", summary="Cancel all jobs")
def cancel_all(_admin: str = Depends(auth.require_admin)):
    try:
        count = job_manager.cancel_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"cancelled": int(count)}


@router.post("/purge", summary="Purge all jobs and queue")
def purge_all(_admin: str = Depends(auth.require_admin)):
    try:
        removed = job_manager.purge_jobs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"removed": int(removed)}


@router.post("/{job_id}/retry", summary="Retry a job")
def retry_job(job_id: str, _token: str = Depends(auth.require_user_or_admin)):
    try:
        ok = job_manager.retry_job(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if not ok:
        raise HTTPException(status_code=404, detail="job not found or running")
    return {"status": "requeued", "job_id": job_id}


@router.get("/token", summary="Get persistent user token (admin only)")
def get_user_token(_admin: str = Depends(auth.require_admin)):
    try:
        token = auth.ensure_user_token()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"user_token": token}

