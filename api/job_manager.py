"""Job manager facade that chooses between an in-memory manager and
an optional Postgres-backed persistent manager.

If the environment variable `POSTGRES_URL` is set, the Postgres manager
will be used. Otherwise the lightweight in-memory `reindex_manager`
is the default (keeps local dev/tests simple).

The Postgres implementation is imported lazily so the module does not
require `psycopg` unless `POSTGRES_URL` is configured.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from api import reindex_manager


_use_postgres = bool(os.environ.get("POSTGRES_URL"))
_use_redis = bool(os.environ.get("REDIS_URL"))


class PostgresNotConfiguredError(RuntimeError):
    pass


class PostgresJobManager:
    """Minimal Postgres-backed job manager using a `jobs` table.

    Table schema (created on first use):
      job_id TEXT PRIMARY KEY,
      status TEXT,
      message TEXT,
      created_at TIMESTAMP WITH TIME ZONE,
      started_at TIMESTAMP WITH TIME ZONE,
      finished_at TIMESTAMP WITH TIME ZONE,
      result JSONB
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self._conn = None

    def _get_conn(self):
        if self._conn is None:
            try:
                import psycopg
            except Exception as e:
                raise PostgresNotConfiguredError("psycopg not installed") from e
            self._psycopg = psycopg
            self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._ensure_table()
        return self._conn

    def _ensure_table(self):
        conn = self._conn
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                message TEXT,
                created_at TIMESTAMPTZ DEFAULT now(),
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                result JSONB
            )
            """
        )
        cur.close()

    def create_job(self, job_id: str, message: str = "queued") -> Dict[str, Any]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO jobs (job_id, status, message) VALUES (%s, %s, %s) ON CONFLICT (job_id) DO NOTHING",
            (job_id, "queued", message),
        )
        cur.close()
        return self.get_job(job_id)

    def set_job_started(self, job_id: str):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status=%s, started_at=now() WHERE job_id=%s",
            ("running", job_id),
        )
        cur.close()

    def set_job_result(self, job_id: str, result: Dict[str, Any], status: str = "success", message: Optional[str] = None):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status=%s, finished_at=now(), result=%s, message=COALESCE(%s, message) WHERE job_id=%s",
            (status, json.dumps(result), message, job_id),
        )
        cur.close()

    def set_job_failed(self, job_id: str, message: str):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status=%s, finished_at=now(), message=%s WHERE job_id=%s",
            ("failed", message, job_id),
        )
        cur.close()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT job_id, status, message, extract(epoch from created_at) as created_at, extract(epoch from started_at) as started_at, extract(epoch from finished_at) as finished_at, result FROM jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        job = {
            "job_id": row[0],
            "status": row[1],
            "message": row[2],
            "created_at": float(row[3]) if row[3] is not None else None,
            "started_at": float(row[4]) if row[4] is not None else None,
            "finished_at": float(row[5]) if row[5] is not None else None,
            "result": row[6],
        }
        return job

    def list_jobs(self, limit: int | None = None) -> list[Dict[str, Any]]:
        conn = self._get_conn()
        cur = conn.cursor()
        if limit is None:
            cur.execute("SELECT job_id, status, message, extract(epoch from created_at) as created_at, extract(epoch from started_at) as started_at, extract(epoch from finished_at) as finished_at, result FROM jobs ORDER BY created_at DESC")
        else:
            cur.execute("SELECT job_id, status, message, extract(epoch from created_at) as created_at, extract(epoch from started_at) as started_at, extract(epoch from finished_at) as finished_at, result FROM jobs ORDER BY created_at DESC LIMIT %s", (limit,))
        rows = cur.fetchall()
        cur.close()
        jobs = []
        for row in rows:
            jobs.append(
                {
                    "job_id": row[0],
                    "status": row[1],
                    "message": row[2],
                    "created_at": float(row[3]) if row[3] is not None else None,
                    "started_at": float(row[4]) if row[4] is not None else None,
                    "finished_at": float(row[5]) if row[5] is not None else None,
                    "result": row[6],
                }
            )
        return jobs


# Facade functions exposed to callers
_pg_manager: Optional[PostgresJobManager] = None
if _use_postgres:
    _pg_manager = PostgresJobManager(os.environ.get("POSTGRES_URL"))
_redis_manager = None
if _use_redis:
    try:
        from api.redis_job_manager import (
            create_job as _redis_create_job,
            set_job_started as _redis_set_job_started,
            set_job_result as _redis_set_job_result,
            set_job_failed as _redis_set_job_failed,
            get_job as _redis_get_job,
            list_jobs as _redis_list_jobs,
            pause_queue as _redis_pause,
            resume_queue as _redis_resume,
            is_paused as _redis_is_paused,
            requeue_stuck_jobs as _redis_requeue,
            count_jobs as _redis_count,
            cancel_all as _redis_cancel_all,
            purge_jobs as _redis_purge,
            retry_job as _redis_retry,
            cancel_job as _redis_cancel,
        )

        _redis_manager = {
            "create": _redis_create_job,
            "start": _redis_set_job_started,
            "result": _redis_set_job_result,
            "fail": _redis_set_job_failed,
            "get": _redis_get_job,
            "list": _redis_list_jobs,
            "pause": _redis_pause,
            "resume": _redis_resume,
            "is_paused": _redis_is_paused,
            "requeue": _redis_requeue,
            "count": _redis_count,
            "cancel_all": _redis_cancel_all,
            "purge": _redis_purge,
            "retry": _redis_retry,
            "cancel": _redis_cancel,
        }
    except Exception:
        _redis_manager = None


def create_job(job_id: str, message: str = "queued") -> Dict[str, Any]:
    if _use_redis and _redis_manager:
        return _redis_manager["create"](job_id, message=message)
    if _use_postgres and _pg_manager:
        return _pg_manager.create_job(job_id, message=message)
    return reindex_manager.create_job(job_id, message=message)


def set_job_started(job_id: str):
    if _use_redis and _redis_manager:
        return _redis_manager["start"](job_id)
    if _use_postgres and _pg_manager:
        return _pg_manager.set_job_started(job_id)
    return reindex_manager.set_job_started(job_id)


def set_job_result(job_id: str, result: Dict[str, Any], status: str = "success", message: Optional[str] = None):
    if _use_redis and _redis_manager:
        return _redis_manager["result"](job_id, result, status=status, message=message)
    if _use_postgres and _pg_manager:
        return _pg_manager.set_job_result(job_id, result=result, status=status, message=message)
    return reindex_manager.set_job_result(job_id, result, status=status, message=message)


def set_job_failed(job_id: str, message: str):
    if _use_redis and _redis_manager:
        return _redis_manager["fail"](job_id, message=message)
    if _use_postgres and _pg_manager:
        return _pg_manager.set_job_failed(job_id, message=message)
    return reindex_manager.set_job_failed(job_id, message)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    if _use_redis and _redis_manager:
        return _redis_manager["get"](job_id)
    if _use_postgres and _pg_manager:
        return _pg_manager.get_job(job_id)
    return reindex_manager.get_job(job_id)


def list_jobs(limit: int | None = None, offset: int = 0) -> list[Dict[str, Any]]:
    if _use_redis and _redis_manager:
        return _redis_manager["list"](limit=limit, offset=offset)
    if _use_postgres and _pg_manager:
        return _pg_manager.list_jobs(limit=limit)
    return reindex_manager.list_jobs(limit=limit, offset=offset)


def pause_queue():
    if _use_redis and _redis_manager:
        return _redis_manager["pause"]()
    if _use_postgres and _pg_manager:
        raise RuntimeError("Pause not implemented for Postgres manager")
    return reindex_manager.pause_queue()


def resume_queue():
    if _use_redis and _redis_manager:
        return _redis_manager["resume"]()
    if _use_postgres and _pg_manager:
        raise RuntimeError("Resume not implemented for Postgres manager")
    return reindex_manager.resume_queue()


def is_paused() -> bool:
    if _use_redis and _redis_manager:
        return _redis_manager["is_paused"]()
    if _use_postgres and _pg_manager:
        return False
    return reindex_manager.is_paused()


def requeue_stuck_jobs(older_than_seconds: int = 300, max_retries: int = 5) -> int:
    if _use_redis and _redis_manager:
        return _redis_manager["requeue"](older_than_seconds=older_than_seconds, max_retries=max_retries)
    if _use_postgres and _pg_manager:
        raise RuntimeError("requeue_stuck_jobs not implemented for Postgres manager")
    return reindex_manager.requeue_stuck_jobs(older_than_seconds=older_than_seconds, max_retries=max_retries)


def count_jobs() -> int:
    if _use_redis and _redis_manager:
        return _redis_manager["count"]()
    if _use_postgres and _pg_manager:
        # simple count via Postgres
        return _pg_manager.count_jobs()
    return reindex_manager.count_jobs()


def cancel_job(job_id: str) -> bool:
    if _use_redis and _redis_manager:
        return _redis_manager["cancel"](job_id)
    if _use_postgres and _pg_manager:
        raise RuntimeError("cancel_job not implemented for Postgres manager")
    return reindex_manager.cancel_job(job_id)


def cancel_all() -> int:
    if _use_redis and _redis_manager:
        return _redis_manager["cancel_all"]()
    if _use_postgres and _pg_manager:
        raise RuntimeError("cancel_all not implemented for Postgres manager")
    return reindex_manager.cancel_all()


def purge_jobs() -> int:
    if _use_redis and _redis_manager:
        return _redis_manager["purge"]()
    if _use_postgres and _pg_manager:
        raise RuntimeError("purge_jobs not implemented for Postgres manager")
    return reindex_manager.purge_jobs()


def retry_job(job_id: str) -> bool:
    if _use_redis and _redis_manager:
        return _redis_manager["retry"](job_id)
    if _use_postgres and _pg_manager:
        raise RuntimeError("retry_job not implemented for Postgres manager")
    return reindex_manager.retry_job(job_id)
