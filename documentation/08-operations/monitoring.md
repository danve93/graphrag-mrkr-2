# Operations Runbook: Monitoring & Troubleshooting

## Overview

This runbook provides operational guidance for monitoring, troubleshooting, and optimizing the Amber RAG system in production.

## Table of Contents

- [Health Checks](#health-checks)
- [Stage Timing Monitoring](#stage-timing-monitoring)
- [Cache Performance](#cache-performance)
- [Common Issues](#common-issues)
- [Performance Optimization](#performance-optimization)
- [Log Analysis](#log-analysis)
- [Alerting Thresholds](#alerting-thresholds)

## Health Checks

### System Health Endpoint

Check overall system health:

```bash
curl http://localhost:8000/api/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "neo4j": "connected",
  "flashrank": "enabled",
  "embedding_provider": "openai",
  "llm_provider": "openai"
}
```

**Failure Indicators:**
- `neo4j: "disconnected"` → Check Neo4j service status
- `flashrank: "disabled"` → Check FlashRank model availability
- Missing providers → Check API keys and environment variables

### Database Statistics

Query database health:

```bash
curl http://localhost:8000/api/database/stats
```

**Expected Response:**
```json
{
  "documents": 42,
  "chunks": 8534,
  "entities": 1234,
  "relationships": 5678,
  "communities": 89
}
```

**Anomaly Detection:**
- Zero chunks with documents > 0 → Ingestion failure
- Zero entities with chunks > 0 → Entity extraction disabled or failed
- Communities count discrepancy → Clustering may need re-run

### Cache Metrics

Monitor cache performance:

```bash
curl http://localhost:8000/api/database/cache-stats
```

**Expected Response:**
```json
{
  "entity_label_cache": {
    "size": 1234,
    "hits": 5678,
    "misses": 890,
    "hit_rate": 0.864
  },
  "embedding_cache": {
    "size": 4567,
    "hits": 12345,
    "misses": 2345,
    "hit_rate": 0.840
  },
  "retrieval_cache": {
    "size": 234,
    "hits": 567,
    "misses": 123,
    "hit_rate": 0.822
  }
}
```

**Performance Indicators:**
- **Entity label cache hit rate**: Target 70-80%
- **Embedding cache hit rate**: Target 40-60%
- **Retrieval cache hit rate**: Target 20-30%

## Stage Timing Monitoring

### Normal Stage Durations

Expected timing ranges for each pipeline stage:

| Stage | Normal Range | Warning Threshold | Critical Threshold |
|-------|--------------|-------------------|-------------------|
| Query Analysis | 100-300ms | >500ms | >1000ms |
| Retrieval | 300-800ms | >1500ms | >3000ms |
| Graph Reasoning | 150-500ms | >1000ms | >2000ms |
| Generation | 2000-10000ms | >15000ms | >30000ms |

### Monitoring Stage Timing

**Method 1: Frontend UI**

The LoadingIndicator displays real-time stage progress:
- Stage name with duration in tooltips
- Metadata (chunks retrieved, context items)
- Total duration after completion

**Method 2: Log Analysis**

Backend logs include stage timing:

```bash
docker logs graphrag-backend | grep "stage_duration"
```

**Method 3: API Response**

Non-streaming endpoint returns timing in response:

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "session_id": "monitor-123"}' \
  | jq '.stages[] | {name, duration_ms}'
```

### Stage Performance Analysis

**Query Analysis Slow (>500ms)**
- Likely causes: LLM provider latency, complex query parsing
- Check: OpenAI/Ollama API response times
- Fix: Consider query analysis caching or simplified parsing

**Retrieval Slow (>1500ms)**
- Likely causes: Large vector search, many documents, Neo4j query performance
- Check: Number of chunks in database, Neo4j query time
- Fix: Reduce top_k, enable caching, optimize Neo4j indexes

**Graph Reasoning Slow (>1000ms)**
- Likely causes: Deep graph traversal, large expansion
- Check: max_expansion_depth, max_expanded_chunks settings
- Fix: Reduce expansion parameters, optimize Neo4j relationship indexes

**Generation Slow (>15000ms)**
- Likely causes: Large context, slow LLM provider, long response
- Check: Context size, LLM provider status, temperature setting
- Fix: Reduce context chunks, use faster model, lower temperature

## Cache Performance

### Cache Hit Rate Analysis

**Entity Label Cache (Target: 70-80%)**

Low hit rate (<50%) indicates:
- Many unique entity queries (expected for diverse workload)
- TTL too short (increase from 300s)
- Cache size too small (increase from 5000)

**Embedding Cache (Target: 40-60%)**

Low hit rate (<30%) indicates:
- Diverse query patterns (expected)
- Many unique document chunks
- Consider: Pre-warming cache with common queries

**Retrieval Cache (Target: 20-30%)**

Low hit rate (<10%) indicates:
- Users changing retrieval parameters frequently
- Short TTL causing premature eviction (60s default)
- High diversity in queries (expected for exploratory use)

High hit rate (>50%) indicates:
- Repetitive queries
- Consider: Longer TTL or larger cache size

### Cache Tuning

**Increase Cache Sizes** (config/settings.py):
```python
ENTITY_LABEL_CACHE_SIZE=10000  # from 5000
EMBEDDING_CACHE_SIZE=20000     # from 10000
RETRIEVAL_CACHE_SIZE=2000      # from 1000
```

**Adjust TTL** (config/settings.py):
```python
ENTITY_LABEL_CACHE_TTL=600     # from 300 (5min → 10min)
RETRIEVAL_CACHE_TTL=120        # from 60 (1min → 2min)
```

**Disable Caching for Debugging**:
```bash
export ENABLE_CACHING=false
```

### Cache Warming

Pre-warm caches after deployment:

```python
import requests

# Warm retrieval cache with common queries
common_queries = [
    "What is Carbonio?",
    "How do I install Carbonio?",
    "What are backup options?",
]

for query in common_queries:
    requests.post("http://localhost:8000/api/chat/query", json={
        "message": query,
        "session_id": "warmup"
    })
```

## Common Issues

### Issue: Slow Query Performance

**Symptoms:**
- Total duration >15s consistently
- User complaints about response time

**Diagnosis:**
1. Check stage timing breakdown
2. Identify slowest stage
3. Review cache hit rates
4. Check database size

**Resolution:**

If **Retrieval** is slow:
```bash
# Check database size
curl http://localhost:8000/api/database/stats

# If >10k chunks, consider:
# - Reduce top_k from 10 to 5
# - Enable caching
# - Add Neo4j indexes
```

If **Generation** is slow:
```bash
# Switch to faster model
# In Chat Tuning UI: GPT-4 → GPT-3.5-turbo

# Or reduce context
# Set max_expanded_chunks=10 (from 15)
```

### Issue: Cache Not Working

**Symptoms:**
- Hit rate 0% across all caches
- Performance not improving on repeated queries

**Diagnosis:**
```bash
# Check if caching is enabled
docker logs graphrag-backend | grep "ENABLE_CACHING"

# Should show: ENABLE_CACHING=true
```

**Resolution:**
```bash
# If disabled, enable in .env
echo "ENABLE_CACHING=true" >> .env

# Restart backend
docker compose up -d --build backend
```

### Issue: Retrieval Cache Misses on Identical Queries

**Symptoms:**
- Same query with same parameters not hitting cache
- Retrieval cache hit rate very low despite repeated queries

**Diagnosis:**
- Check if retrieval parameters are changing between requests
- 14-parameter hash means ANY change invalidates cache

**Resolution:**
- Use consistent parameters for repeated queries
- Avoid changing top_k, weights, or expansion settings unnecessarily
- Consider session-level parameter locking in UI

### Issue: Conversation Context Lost

**Symptoms:**
- Follow-up questions don't reference previous messages
- "it" and "that" not resolved correctly

**Diagnosis:**
```bash
# Check if same session_id is being used
docker logs graphrag-backend | grep "session_id"
```

**Resolution:**
- Ensure frontend maintains session_id across follow-up queries
- Clear session and start new conversation if needed
- Check history endpoint: `GET /api/history/{session_id}`

### Issue: Stage Timing Not Displayed in UI

**Symptoms:**
- LoadingIndicator shows stages but no timing
- Tooltips missing duration information

**Diagnosis:**
1. Check backend response includes stages with duration_ms
2. Check frontend parsing of SSE events

**Resolution:**
```bash
# Test non-streaming endpoint
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "session_id": "test"}' \
  | jq '.stages'

# Should show duration_ms for each stage
# If missing, check backend version (M3+ required)

# Rebuild frontend if needed
cd frontend && npm run build
```

### Issue: High Memory Usage

**Symptoms:**
- Docker containers using >4GB memory
- OOM errors in logs

**Diagnosis:**
```bash
# Check container memory usage
docker stats

# Check cache sizes
curl http://localhost:8000/api/database/cache-stats | jq '.[] | .size'
```

**Resolution:**
```bash
# Reduce cache sizes in .env
ENTITY_LABEL_CACHE_SIZE=2500
EMBEDDING_CACHE_SIZE=5000
RETRIEVAL_CACHE_SIZE=500

# Restart backend
docker compose up -d --build backend
```

## Performance Optimization

### Query Optimization Checklist

1. **Enable Caching** (30-50% latency reduction)
   ```bash
   ENABLE_CACHING=true
   ```

2. **Tune Retrieval Parameters**
   - Reduce `top_k` if precision is sufficient (10 → 5)
   - Lower `max_expanded_chunks` if context is sufficient (15 → 10)
   - Reduce `max_expansion_depth` if shallow paths suffice (2 → 1)

3. **Enable RRF** (better fusion, minimal overhead)
   ```bash
   ENABLE_RRF=true
   RRF_K=60
   ```

4. **Optimize Reranking**
   - Use FlashRank for better ordering (5-10% accuracy gain)
   - Limit candidates: `flashrank_max_candidates=20` (from 50)
   - Adjust blend weight: `flashrank_blend_weight=0.7` (from 0.5)

5. **Use Faster LLM for Generation**
   - GPT-3.5-turbo: 2-3x faster than GPT-4
   - Lower temperature: 0.5 (from 0.7) for faster sampling

### Database Optimization

**Neo4j Index Creation:**

```cypher
// Chunk embeddings
CREATE INDEX chunk_embedding IF NOT EXISTS FOR (c:Chunk) ON (c.embedding);

// Entity labels
CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label);

// Relationship strengths
CREATE INDEX rel_strength IF NOT EXISTS FOR ()-[r:RELATED_TO]-() ON (r.strength);

// Document IDs
CREATE INDEX doc_id IF NOT EXISTS FOR (d:Document) ON (d.id);
```

**Check Index Usage:**
```cypher
// Show all indexes
SHOW INDEXES;

// Profile query to see index usage
PROFILE MATCH (c:Chunk) WHERE c.document_id = $doc_id RETURN c;
```

### Embedding Optimization

**Batch Embedding Generation:**

Use `embedding_concurrency` to parallelize:
```bash
EMBEDDING_CONCURRENCY=10  # from 5
```

**Reduce Embedding Model Size:**
```bash
# Faster embedding model
EMBEDDING_MODEL=text-embedding-3-small  # from text-embedding-3-large
```

## Log Analysis

### Key Log Patterns

**Successful Query:**
```
INFO: Query analysis complete: duration_ms=234
INFO: Retrieval complete: chunks_retrieved=10, duration_ms=456
INFO: Graph reasoning complete: context_items=8, duration_ms=189
INFO: Generation complete: response_length=342, duration_ms=3421
```

**Cache Hits:**
```
DEBUG: Retrieval cache HIT for query: <hash>
DEBUG: Entity label cache HIT for entity: Component
DEBUG: Embedding cache HIT for text: <hash>
```

**Error Patterns:**
```
ERROR: Neo4j connection failed: AuthError
ERROR: OpenAI API error: RateLimitError
ERROR: Entity extraction timeout after 30s
```

### Log Aggregation

**Filter by Stage:**
```bash
docker logs graphrag-backend | grep "Retrieval complete"
```

**Count Cache Hits:**
```bash
docker logs graphrag-backend | grep "cache HIT" | wc -l
```

**Find Slow Queries:**
```bash
docker logs graphrag-backend | grep "duration_ms" | awk '$NF > 5000'
```

## Alerting Thresholds

### Critical Alerts

1. **Health Check Failure**
   - Condition: `/api/health` returns non-200 status
   - Action: Check Neo4j connection, API keys, service availability

2. **Extreme Query Latency**
   - Condition: Total duration >30s
   - Action: Check stage breakdown, database performance, LLM provider status

3. **Cache Disabled**
   - Condition: All cache hit rates = 0%
   - Action: Verify `ENABLE_CACHING=true`, restart backend

4. **Database Disconnection**
   - Condition: Neo4j connection lost
   - Action: Check Neo4j service, network, credentials

### Warning Alerts

1. **High Stage Latency**
   - Condition: Any stage exceeds warning threshold
   - Action: Review stage-specific optimization guidance

2. **Low Cache Hit Rates**
   - Entity label: <50%
   - Embedding: <30%
   - Retrieval: <10%
   - Action: Consider cache tuning, workload analysis

3. **High Memory Usage**
   - Condition: Container memory >80% of limit
   - Action: Review cache sizes, consider scaling

4. **API Rate Limits**
   - Condition: LLM/embedding provider rate limit errors in logs
   - Action: Reduce concurrency, implement backoff, upgrade tier

## Performance Baseline

**Expected Performance (Single Query):**

| Metric | Value |
|--------|-------|
| Total Duration | 4-12s |
| Query Analysis | 200-300ms |
| Retrieval | 400-800ms |
| Graph Reasoning | 200-500ms |
| Generation | 3-10s |
| Cache Hit Rate (entity) | 70-80% |
| Cache Hit Rate (embedding) | 40-60% |
| Cache Hit Rate (retrieval) | 20-30% |
| Memory Usage | 1-2GB per container |

**Throughput:**
- Concurrent requests: 5-10 (limited by LLM provider)
- Requests per minute: 30-60 (with caching)

## Maintenance Tasks

### Daily
- Check health endpoint
- Review error logs
- Monitor cache hit rates

### Weekly
- Analyze stage timing trends
- Review slow query logs
- Check database statistics
- Validate cache effectiveness

### Monthly
- Review and tune retrieval parameters
- Optimize Neo4j indexes
- Update embedding/LLM models
- Performance load testing

## Related Documentation

- [Chat API Reference](06-api-reference/chat-api.md)
- [Deployment Checklist](deployment.md)
- [Rollback Procedures](rollback.md)
- [Configuration Reference](07-configuration/settings.md)
