# Troubleshooting

Common issues and fixes.

## Backend Fails to Start

- Missing `.env`: create from `.env.example`
- `OPENAI_API_KEY` missing: set in `.env`
- Neo4j unreachable: check `NEO4J_URI`, container running
- Port conflicts: ensure `8000` (backend) and `3000` (frontend) free

## Neo4j Issues

- Connection refused: verify credentials and URI
- Slow queries: increase `NEO4J_MAX_CONNECTION_POOL_SIZE`
- Memory errors: adjust Neo4j heap size; reduce clustering scale

## OpenAI / LLM Issues

- Rate limits: tune `EMBEDDING_DELAY_*`, `LLM_DELAY_*`
- Quota exceeded: check billing or use Ollama locally
- Timeout: retry with backoff; verify network

## FlashRank Reranker

- Slow startup: prewarm via `scripts/flashrank_prewarm_worker.py`
- Memory usage: switch to TinyBERT
- Disabled unexpectedly: set `FLASHRANK_ENABLED=true`

## Caching Problems

- Low hit rate: increase sizes, TTLs
- Stale results: reduce TTL or disable response cache
- High memory: reduce cache sizes

## SSE Streaming

- No tokens: verify `/api/chat` returns `text/event-stream`
- Broken stream: check proxy/timeouts; use `curl -N`
- Frontend lag: confirm incremental rendering enabled

## Jobs & Ingestion

- Stuck job: cancel via `POST /api/jobs/{id}/cancel`
- Failed ingestion: inspect logs and entity extraction settings
- Large PDFs: enable Marker with OCR for accuracy

## Database Maintenance

- Reindex after large updates: `POST /api/database/reindex`
- Clear dev database: `DELETE /api/database/clear?confirm=yes` (CAUTION)

## Diagnostics Checklist

```bash
# Health
curl http://localhost:8000/api/database/health

# Cache stats
curl http://localhost:8000/api/database/cache-stats

# Recent jobs
curl http://localhost:8000/api/jobs?limit=10
```
