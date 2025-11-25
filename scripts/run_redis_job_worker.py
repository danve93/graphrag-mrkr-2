"""Simple Redis worker that processes reindex jobs from `reindex:queue`.

This script blocks on the Redis list and invokes `reindex_tasks.run_reindex_job`
for each job_id. Run it as a long-running process in production when
`REDIS_URL` is configured.
"""
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        import redis
    except Exception as e:
        logger.error("redis package not available: %s", e)
        raise

    r = redis.from_url(os.environ.get("REDIS_URL"))
    logger.info("Connected to Redis, listening for jobs on 'reindex:queue'")

    from api import reindex_tasks

    MAX_RETRIES = int(os.environ.get("REINDEX_MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.environ.get("REINDEX_RETRY_DELAY", "5"))

    from api import redis_job_manager as redis_mgr

    while True:
        try:
            # respect pause flag
            try:
                if redis_mgr.is_paused():
                    logger.info("Queue is paused; sleeping briefly")
                    import time

                    time.sleep(5)
                    continue
            except Exception:
                # if pause check fails, keep running
                pass

            # atomically move job from queue -> processing list
            item = r.brpoplpush("reindex:queue", "reindex:processing", timeout=5)
            if not item:
                # periodically attempt to recover stuck jobs
                try:
                    requeued = redis_mgr.requeue_stuck_jobs()
                    if requeued:
                        logger.info("Requeued %d stuck jobs", requeued)
                except Exception:
                    logger.exception("Failed to requeue stuck jobs")
                continue
            job_id = item.decode() if isinstance(item, (bytes, bytearray)) else item
            logger.info("Picked up job %s", job_id)
            # mark started (adds to processing zset)
            try:
                redis_mgr.set_job_started(job_id)
            except Exception:
                logger.exception("Failed to mark job started for %s", job_id)
            try:
                reindex_tasks.run_reindex_job(job_id)
                # success
                redis_mgr.set_job_result(job_id, {"message": "completed"}, status="success")
                # remove from processing list if present
                try:
                    r.lrem("reindex:processing", 0, job_id)
                except Exception:
                    pass
            except Exception as e:
                logger.exception("Failed to process job %s: %s", job_id, e)
                # Attempt retry logic: read attempts from job record and requeue if allowed
                try:
                    job = redis_mgr.get_job(job_id) or {}
                    attempts = int(job.get("attempts", 0)) if job.get("attempts") is not None else 0
                    if attempts < MAX_RETRIES:
                        attempts += 1
                        # update attempts and requeue
                        from api.redis_job_manager import _get_redis

                        rr = _get_redis()
                        key = f"jobs:{job_id}"
                        rr.hset(key, mapping={"attempts": attempts})
                        logger.info("Re-queueing job %s (attempt %d/%d)", job_id, attempts, MAX_RETRIES)
                        # small delay to avoid tight requeue loops
                        import time

                        time.sleep(RETRY_DELAY)
                        # remove from processing and push back to queue
                        try:
                            r.lrem("reindex:processing", 0, job_id)
                        except Exception:
                            pass
                        rr.rpush("reindex:queue", job_id)
                    else:
                        logger.error("Job %s exceeded max retries (%d)", job_id, MAX_RETRIES)
                        redis_mgr.set_job_failed(job_id, f"exceeded retries ({attempts})")
                        try:
                            r.lrem("reindex:processing", 0, job_id)
                        except Exception:
                            pass
                except Exception:
                    logger.exception("Failed to apply retry handling for job %s", job_id)

        except KeyboardInterrupt:
            logger.info("Worker shutting down")
            break
        except Exception as e:
            logger.exception("Worker loop error: %s", e)


if __name__ == "__main__":
    main()
