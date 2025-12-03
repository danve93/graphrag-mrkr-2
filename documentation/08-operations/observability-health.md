# Observability & Health

Monitor system health, performance, and logs.

## Health Endpoints

```bash
# Service health
curl http://localhost:8000/api/database/health

# Cache metrics
curl http://localhost:8000/api/database/cache-stats

# Database stats
curl http://localhost:8000/api/database/stats
```

## Logs

```bash
# Backend logs (Docker)
docker compose logs -f backend

# Frontend logs
docker compose logs -f frontend

# Local runtime
tail -f logs/backend.log
```

## Metrics & Monitoring

- CacheMetrics API: `/api/database/cache-stats`
- SSE stage durations in chat stream
- Job progress via `/api/jobs/{job_id}` or WebSocket stream

## Alerts (Recommended)

- Neo4j down or slow queries
- OpenAI API quota/limit errors
- High cache miss rates (>70% misses)
- Job failures or long durations

## Performance Checks

```bash
# Retrieval latency (stage event)
curl -N -H "Content-Type: application/json" \
  -d '{"message":"test","session_id":"ops"}' \
  http://localhost:8000/api/chat | grep '"stage":"retrieval"'
```

## Capacity Planning

- Increase `NEO4J_MAX_CONNECTION_POOL_SIZE` for high concurrency
- Tune cache sizes to fit memory budget (~200MB default)
- Scale frontend/backend replicas (Compose or K8s)
