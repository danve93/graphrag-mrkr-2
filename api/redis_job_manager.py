"""Redis-backed job manager.

Stores job records in Redis hashes and enqueues job IDs on a list
`reindex:queue` so external workers can process them.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

_redis = None


def _get_redis():
    global _redis
    if _redis is None:
        try:
            import redis
        except Exception as e:
            raise RuntimeError("redis package not installed") from e
        _redis = redis.from_url(os.environ.get("REDIS_URL"))
    return _redis


def _job_key(job_id: str) -> str:
    return f"jobs:{job_id}"


def create_job(job_id: str, message: str = "queued") -> Dict[str, Any]:
    r = _get_redis()
    key = _job_key(job_id)
    data = {
        "job_id": job_id,
        "status": "queued",
        "message": message,
        "created_at": r.time()[0],
        "started_at": "",
        "finished_at": "",
        "result": "",
        "attempts": 0,
    }
    r.hset(key, mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in data.items()})
    # add to jobs zset for efficient listing
    try:
        r.zadd("jobs:zset", {job_id: float(data["created_at"])})
    except Exception:
        # fallback if zadd not available for some reason
        pass
    # push to queue for external workers
    r.rpush("reindex:queue", job_id)
    return get_job(job_id)


def set_job_started(job_id: str):
    r = _get_redis()
    key = _job_key(job_id)
    started = r.time()[0]
    r.hset(key, mapping={"status": "running", "started_at": started})
    # mark processing with score for requeue detection
    try:
        r.zadd("reindex:processing:zset", {job_id: float(started)})
    except Exception:
        pass


def set_job_result(job_id: str, result: Dict[str, Any], status: str = "success", message: Optional[str] = None):
    r = _get_redis()
    key = _job_key(job_id)
    r.hset(key, mapping={"status": status, "finished_at": r.time()[0], "result": json.dumps(result)})
    if message:
        r.hset(key, "message", message)
    # remove from processing set if present
    try:
        r.zrem("reindex:processing:zset", job_id)
    except Exception:
        pass


def set_job_failed(job_id: str, message: str):
    r = _get_redis()
    key = _job_key(job_id)
    r.hset(key, mapping={"status": "failed", "finished_at": r.time()[0], "message": message})
    try:
        r.zrem("reindex:processing:zset", job_id)
    except Exception:
        pass


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    r = _get_redis()
    key = _job_key(job_id)
    if not r.exists(key):
        return None
    raw = r.hgetall(key)
    job = {}
    for k, v in raw.items():
        try:
            s = v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
        except Exception:
            s = str(v)
        # attempt to parse json fields
        if s.startswith("{") or s.startswith("["):
            try:
                job[k] = json.loads(s)
                continue
            except Exception:
                pass
        # numeric fields stored as str
        try:
            job[k] = float(s) if "." in s else int(s)
        except Exception:
            job[k] = s
    return job


def list_jobs(limit: int | None = None) -> list[Dict[str, Any]]:
    """Return a list of job dicts stored in Redis, newest first.

    This scans keys matching `jobs:*`, loads each job via `get_job`,
    sorts by `created_at` (descending) and applies an optional limit.
    """
    r = _get_redis()
    # use sorted set for pagination (jobs:zset). newest first.
    try:
        # support offset/limit via python-level slicing if caller passes a tuple
        ids = r.zrevrange("jobs:zset", 0, -1)
    except Exception:
        # fallback to keys scan
        keys = r.keys("jobs:*")
        ids = []
        for k in keys:
            try:
                key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            except Exception:
                key = str(k)
            if key.startswith("jobs:"):
                ids.append(key.split(":", 1)[1])
    jobs = []
    for idb in ids:
        try:
            job_id = idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        except Exception:
            job_id = str(idb)
        job = get_job(job_id)
        if job:
            jobs.append(job)
    # sort by created_at desc
    # already returned in zrevrange order (newest first)
    return jobs


def list_jobs(limit: int | None = None, offset: int = 0) -> list[Dict[str, Any]]:
    r = _get_redis()
    try:
        if limit is None:
            ids = r.zrevrange("jobs:zset", offset, -1)
        else:
            ids = r.zrevrange("jobs:zset", offset, offset + limit - 1)
    except Exception:
        # fallback to keys scan and local slicing
        keys = r.keys("jobs:*")
        ids = []
        for k in keys:
            try:
                key = k.decode() if isinstance(k, (bytes, bytearray)) else str(k)
            except Exception:
                key = str(k)
            if key.startswith("jobs:"):
                ids.append(key.split(":", 1)[1])
        ids = ids[offset : None if limit is None else offset + limit]
    jobs = []
    for idb in ids:
        try:
            job_id = idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        except Exception:
            job_id = str(idb)
        job = get_job(job_id)
        if job:
            jobs.append(job)
    return jobs


def count_jobs() -> int:
    r = _get_redis()
    try:
        return int(r.zcard("jobs:zset"))
    except Exception:
        return len(r.keys("jobs:*"))


def cancel_all() -> int:
    r = _get_redis()
    try:
        ids = r.zrange("jobs:zset", 0, -1)
    except Exception:
        ids = []
    count = 0
    for idb in ids:
        try:
            job_id = idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        except Exception:
            job_id = str(idb)
        key = _job_key(job_id)
        if r.exists(key):
            r.hset(key, mapping={"status": "cancelled", "message": "cancelled by admin", "finished_at": r.time()[0]})
            count += 1
        try:
            r.lrem("reindex:queue", 0, job_id)
        except Exception:
            pass
        try:
            r.lrem("reindex:processing", 0, job_id)
        except Exception:
            pass
        try:
            r.zrem("reindex:processing:zset", job_id)
        except Exception:
            pass
    return count


def purge_jobs() -> int:
    r = _get_redis()
    try:
        ids = r.zrange("jobs:zset", 0, -1)
    except Exception:
        ids = []
    removed = 0
    for idb in ids:
        try:
            job_id = idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        except Exception:
            job_id = str(idb)
        key = _job_key(job_id)
        try:
            r.delete(key)
        except Exception:
            pass
        removed += 1
    # remove listings and queues
    try:
        r.delete("jobs:zset", "reindex:queue", "reindex:processing", "reindex:processing:zset")
    except Exception:
        pass
    return removed


def retry_job(job_id: str) -> bool:
    r = _get_redis()
    key = _job_key(job_id)
    if not r.exists(key):
        return False
    job = get_job(job_id) or {}
    status = job.get("status")
    if status == "running":
        return False
    # reset attempts and mark queued
    r.hset(key, mapping={"status": "queued", "attempts": 0, "message": "requeued by admin"})
    try:
        r.rpush("reindex:queue", job_id)
    except Exception:
        pass
    return True


def count_jobs() -> int:
    r = _get_redis()
    try:
        return int(r.zcard("jobs:zset"))
    except Exception:
        # fallback to keys
        return len(r.keys("jobs:*"))


def cancel_job(job_id: str) -> bool:
    r = _get_redis()
    key = _job_key(job_id)
    if not r.exists(key):
        return False
    # mark cancelled
    r.hset(key, mapping={"status": "cancelled", "message": "cancelled by user", "finished_at": r.time()[0]})
    # remove from queue and processing structures
    try:
        r.lrem("reindex:queue", 0, job_id)
    except Exception:
        pass
    try:
        r.lrem("reindex:processing", 0, job_id)
    except Exception:
        pass
    try:
        r.zrem("reindex:processing:zset", job_id)
    except Exception:
        pass
    return True


def pause_queue():
    r = _get_redis()
    r.set("reindex:paused", "1")


def resume_queue():
    r = _get_redis()
    r.delete("reindex:paused")


def is_paused() -> bool:
    r = _get_redis()
    return bool(r.exists("reindex:paused"))


def requeue_stuck_jobs(older_than_seconds: int = 300, max_retries: int = 5) -> int:
    """Move jobs from processing back to queue if they've been running too long.

    Returns the number of jobs requeued.
    """
    r = _get_redis()
    now = r.time()[0]
    cutoff = float(now - older_than_seconds)
    # find stuck job ids
    try:
        stuck = r.zrangebyscore("reindex:processing:zset", 0, cutoff)
    except Exception:
        return 0
    requeued = 0
    for idb in stuck:
        try:
            job_id = idb.decode() if isinstance(idb, (bytes, bytearray)) else str(idb)
        except Exception:
            job_id = str(idb)
        key = _job_key(job_id)
        job = get_job(job_id) or {}
        attempts = int(job.get("attempts", 0)) if job.get("attempts") is not None else 0
        if attempts >= max_retries:
            # mark failed
            set_job_failed(job_id, f"stuck and exceeded retries ({attempts})")
            # remove from processing set
            try:
                r.zrem("reindex:processing:zset", job_id)
            except Exception:
                pass
            continue
        # increment attempts and requeue
        attempts += 1
        r.hset(key, mapping={"attempts": attempts, "status": "queued", "message": "re-queued after worker interruption"})
        # push back to queue
        r.rpush("reindex:queue", job_id)
        # remove from processing set
        try:
            r.zrem("reindex:processing:zset", job_id)
        except Exception:
            pass
        requeued += 1
    return requeued
