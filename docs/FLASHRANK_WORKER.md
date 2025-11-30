FlashRank prewarm worker

Purpose

Run the FlashRank prewarming out-of-process so the web process doesn't perform heavy CPU/IO work at startup.

What we added

- `scripts/flashrank_prewarm_worker.py` — a simple worker script that imports `prewarm_ranker()` and runs it.
- `config.settings` new flag: `FLASHRANK_PREWARM_IN_PROCESS` (environment variable) / `flashrank_prewarm_in_process` setting. When `False`, the web process will NOT prewarm; the worker should be started separately.
- `api/main.py` respects the new setting and only schedules in-process prewarm when `flashrank_prewarm_in_process` is true.

How to run the worker

Option A — Run inside the backend container (quick):

```bash
# copy script to container (already done in deployment steps) and run it
docker exec -it graphrag-backend python /app/scripts/flashrank_prewarm_worker.py
```

This runs the prewarm once and exits. It is out-of-process with respect to the web request threads.

Option B — Run as a separate one-shot container (recommended in production):

Add a lightweight service to your `docker-compose.yml` referencing the existing backend image (example snippet):

```yaml
  flashrank-worker:
    image: graphrag-backend:latest
    command: ["python", "/app/scripts/flashrank_prewarm_worker.py"]
    environment:
      - FLASHRANK_PREWARM_IN_PROCESS=0
    depends_on:
      - neo4j
```

Start the worker:

```bash
# start only the worker (compose project name may differ)
docker compose up --no-deps --build flashrank-worker
```

Option C — Run in systemd/cron/CI runner as a one-shot job

Run the same command in any environment that has access to the code and Python deps:

```bash
python scripts/flashrank_prewarm_worker.py
```

Notes & considerations

- Make sure the worker runs with the same Python environment and has access to the same configuration (e.g., `NEO4J_URI`, credentials and any model cache path). If the worker runs in a separate container, mount `flashrank_cache_dir` (if configured) so downloads and caches are shared.
- To disable in-process prewarm in the web process, set `FLASHRANK_PREWARM_IN_PROCESS=0` (env) or set `flashrank_prewarm_in_process=false` in your deployment config.
- If you want the worker to run automatically on deploy, add it as a service that runs once and exits; orchestrators like Kubernetes can run it as a `Job` or `InitContainer` if you need prewarm before traffic.
