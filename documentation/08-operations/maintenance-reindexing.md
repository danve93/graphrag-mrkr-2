# Maintenance & Reindexing

Operational tasks for keeping Amber healthy.

## Reindexing

Recompute similarities and communities after significant updates.

### API Trigger

```bash
curl -X POST http://localhost:8000/api/database/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "rebuild_embeddings": false,
    "rebuild_similarities": true,
    "run_clustering": true
  }'
```

### Script

```bash
python scripts/create_similarities.py
python scripts/build_leiden_projection.py
python scripts/run_clustering.py
```

## Cache Management

```python
from core.singletons import SingletonManager
manager = SingletonManager()

# Clear caches
manager.entity_label_cache.clear()
manager.embedding_cache.clear()
manager.retrieval_cache.clear()
```

## Database Cleanup (Dev Only)

```bash
# CAUTION: irreversible
curl -X DELETE "http://localhost:8000/api/database/clear?confirm=yes"
```

## Job Management

```bash
# List jobs
curl http://localhost:8000/api/jobs?limit=20

# Cancel job
curl -X POST http://localhost:8000/api/jobs/{job_id}/cancel

# Cleanup old jobs
curl -X POST http://localhost:8000/api/jobs/cleanup -H "Content-Type: application/json" -d '{"older_than_days": 30}'
```

## Health Checks

```bash
curl http://localhost:8000/api/database/health
curl http://localhost:8000/api/database/cache-stats
curl http://localhost:8000/api/database/stats
```

## Backup & Restore

- Use Neo4j export/import tools for backups
- Persist volumes in Docker (`docker compose down` keeps data)
- Avoid `down -v` unless you intend to wipe the database

## Recommended Maintenance Cadence

- Weekly: check cache stats, job failures
- Monthly: reindex similarities and run clustering
- Quarterly: review resource limits and presets
