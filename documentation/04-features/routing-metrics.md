# Routing Metrics

**Status**: Production-ready  
**Since**: Milestone 2.6  
**Feature Flag**: Enabled automatically with `ENABLE_QUERY_ROUTING`

## Overview

Routing metrics provide comprehensive observability into query routing performance, accuracy, and failure points. The metrics system tracks latency, cache efficiency, fallback rates, category usage, and seven distinct failure points across the RAG pipeline.

**Key Capabilities**:
- Real-time routing latency tracking (millisecond precision)
- Cache hit rate monitoring (30-40% typical)
- Category usage statistics and trends
- Failure point detection (FP1-FP7)
- User feedback accuracy tracking
- Multi-category query analysis

## Architecture

### Components

1. **RoutingMetrics Class** (`core/routing_metrics.py`)
   - Singleton metrics collector
   - In-memory statistics aggregation
   - Thread-safe counters and lists
   - Failure point categorization

2. **Metrics API** (`api/routers/database.py`)
   - RESTful endpoint for metrics retrieval
   - JSON response with aggregated stats
   - Real-time query

3. **UI Dashboard** (`frontend/src/pages/routing.tsx`)
   - Visual metrics display
   - Category distribution charts
   - Latency histograms
   - Cache hit rate gauges

### Tracked Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `total_queries` | Counter | Total queries routed |
| `avg_routing_latency_ms` | Average | Mean routing decision time |
| `routing_accuracy` | Average | User feedback accuracy (0.0-1.0) |
| `cache_hit_rate` | Ratio | Semantic cache hit rate |
| `fallback_rate` | Ratio | Queries requiring fallback expansion |
| `multi_category_rate` | Ratio | Queries matching 2+ categories |
| `top_categories` | Counter | Most common category assignments |
| `failure_point_rates` | Ratios | Rate of each failure point (FP1-FP7) |
| `failure_point_counts` | Counters | Absolute count of each failure point |

## Usage

### API Endpoint

```bash
# Get current routing metrics
curl http://localhost:8000/api/database/routing-metrics | python3 -m json.tool
```

**Response Example:**
```json
{
  "total_queries": 247,
  "avg_routing_latency_ms": 152.3,
  "routing_accuracy": 0.87,
  "cache_hit_rate": 0.41,
  "fallback_rate": 0.08,
  "multi_category_rate": 0.31,
  "top_categories": [
    ["configure", 68],
    ["installation", 52],
    ["troubleshooting", 41],
    ["api", 28],
    ["general", 22]
  ],
  "failure_point_rates": {
    "FP1": 0.02,
    "FP2": 0.03,
    "FP3": 0.01,
    "FP4": 0.00,
    "FP5": 0.05,
    "FP6": 0.02,
    "FP7": 0.00
  },
  "failure_point_counts": {
    "FP1": 5,
    "FP2": 7,
    "FP3": 3,
    "FP4": 0,
    "FP5": 12,
    "FP6": 5,
    "FP7": 1
  }
}
```

### UI Dashboard

Access the routing dashboard at `http://localhost:3000/routing`:

**Features:**
- Real-time metrics refresh (auto-refresh every 30s)
- Category distribution pie chart
- Latency trend line chart
- Cache performance gauge
- Failure point breakdown table
- Export metrics to CSV

### Programmatic Access

```python
from core.routing_metrics import routing_metrics

# Record a routing decision
routing_metrics.record_routing(
    categories=["installation", "configure"],
    confidence=0.88,
    latency_ms=145.2,
    used_cache=False,
    fallback_used=False
)

# Record user feedback
routing_metrics.record_user_feedback(correct=True)

# Record failure point
routing_metrics.record_failure_point(
    fp_type="FP2",
    details={"query": "...", "categories": [...], "reason": "insufficient_results"}
)

# Get statistics
stats = routing_metrics.get_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Average latency: {stats['avg_routing_latency_ms']:.0f}ms")
```

## Failure Points

Amber tracks seven distinct failure points across the RAG pipeline:

### FP1: Query Routing Failure
**Definition:** Router fails to classify query or returns low confidence

**Causes:**
- Ambiguous query phrasing
- New topic not in category taxonomy
- LLM provider error

**Detection:**
```python
if routing_result["confidence"] < threshold:
    routing_metrics.record_failure_point("FP1", {
        "query": query,
        "confidence": routing_result["confidence"],
        "reason": "low_confidence"
    })
```

**Mitigation:**
- Lower confidence threshold
- Expand category taxonomy
- Enable fallback (should be default)

### FP2: Insufficient Retrieval Results
**Definition:** Retrieval returns fewer than minimum required chunks

**Causes:**
- Category too narrow (over-filtering)
- Sparse document collection
- Query mismatch with indexed content

**Detection:**
```python
if len(chunks) < settings.fallback_min_results:
    routing_metrics.record_failure_point("FP2", {
        "query": query,
        "categories": categories,
        "chunk_count": len(chunks),
        "reason": "insufficient_chunks"
    })
```

**Mitigation:**
- Automatic fallback to all documents
- Lower fallback threshold
- Check document classification coverage

### FP3: Entity Extraction Failure
**Definition:** Entity extraction returns no entities or low-confidence entities

**Causes:**
- Query lacks clear entity references
- Entity linking threshold too high
- Incomplete entity graph

**Detection:**
```python
if not entities or max(e["confidence"] for e in entities) < threshold:
    routing_metrics.record_failure_point("FP3", {
        "query": query,
        "entities_found": len(entities),
        "reason": "no_entities_extracted"
    })
```

**Mitigation:**
- Fall back to pure semantic retrieval
- Lower entity linking threshold
- Improve entity extraction coverage

### FP4: Graph Expansion Timeout
**Definition:** Multi-hop reasoning exceeds time limit

**Causes:**
- Deep graph traversal (>3 hops)
- Large neighborhood size
- Slow database queries

**Detection:**
```python
if expansion_time > timeout_ms:
    routing_metrics.record_failure_point("FP4", {
        "query": query,
        "expansion_time_ms": expansion_time,
        "reason": "timeout"
    })
```

**Mitigation:**
- Reduce `max_expansion_depth`
- Increase timeout limit
- Optimize Neo4j indexes

### FP5: Reranking Failure
**Definition:** FlashRank reranker fails or times out

**Causes:**
- FlashRank model not loaded
- Batch size too large
- GPU memory exhausted

**Detection:**
```python
try:
    reranked = reranker.rerank(chunks)
except Exception as e:
    routing_metrics.record_failure_point("FP5", {
        "query": query,
        "chunk_count": len(chunks),
        "error": str(e),
        "reason": "reranker_error"
    })
```

**Mitigation:**
- Fall back to hybrid scores
- Reduce `flashrank_batch_size`
- Disable reranking temporarily

### FP6: LLM Generation Failure
**Definition:** LLM fails to generate response or returns empty output

**Causes:**
- LLM provider outage
- Rate limit exceeded
- Invalid prompt format

**Detection:**
```python
if not response_text or len(response_text) < 10:
    routing_metrics.record_failure_point("FP6", {
        "query": query,
        "context_length": len(context),
        "response_length": len(response_text),
        "reason": "empty_generation"
    })
```

**Mitigation:**
- Retry with backoff
- Fall back to alternative LLM provider
- Check API key and rate limits

### FP7: Smart Consolidation Deduplication Error
**Definition:** Semantic deduplication fails or produces empty results

**Causes:**
- Embedding generation failure
- Threshold too aggressive (>0.99)
- All chunks deduplicated

**Detection:**
```python
if len(deduplicated) == 0 and len(original) > 0:
    routing_metrics.record_failure_point("FP7", {
        "original_count": len(original),
        "threshold": settings.semantic_threshold,
        "reason": "full_deduplication"
    })
```

**Mitigation:**
- Fall back to original chunk list
- Lower semantic threshold
- Ensure embedding diversity

## Performance Benchmarks

### Typical Metrics (Production)

| Metric | Target | Typical | Alert Threshold |
|--------|--------|---------|-----------------|
| Cache hit rate | >30% | 35-45% | <20% |
| Avg latency | <200ms | 150-180ms | >500ms |
| Fallback rate | <10% | 5-8% | >20% |
| Routing accuracy | >85% | 87-92% | <75% |
| FP1 rate | <5% | 2-3% | >10% |
| FP2 rate | <10% | 5-8% | >15% |

### Latency Breakdown

Average routing decision latency:

| Component | Time (ms) | % of Total |
|-----------|-----------|------------|
| Cache lookup | 5 | 3% |
| Embedding generation | 25 | 16% |
| LLM classification | 120 | 79% |
| Neo4j category query | 3 | 2% |
| **Total** | **153** | **100%** |

### Cache Performance by Query Pattern

| Query Pattern | Hit Rate | Explanation |
|---------------|----------|-------------|
| Identical queries | 95%+ | Direct MD5 match |
| Paraphrased queries | 60-70% | Semantic similarity >0.92 |
| Topic-similar queries | 30-40% | Weak semantic overlap |
| Novel queries | 0% | No similar cache entries |

## Monitoring & Alerts

### Recommended Alerts

**High Latency Alert:**
```yaml
alert: RoutingLatencyHigh
expr: avg_routing_latency_ms > 500
for: 5m
severity: warning
message: "Routing latency exceeded 500ms (current: {{ $value }}ms)"
```

**Low Cache Hit Rate Alert:**
```yaml
alert: RoutingCacheLow
expr: cache_hit_rate < 0.20
for: 10m
severity: info
message: "Cache hit rate below 20% (current: {{ $value | printf \"%.1f%%\" }})"
```

**High Fallback Rate Alert:**
```yaml
alert: RoutingFallbackHigh
expr: fallback_rate > 0.20
for: 10m
severity: warning
message: "Fallback rate exceeded 20% (current: {{ $value | printf \"%.1f%%\" }})"
```

**Failure Point Spike Alert:**
```yaml
alert: RoutingFailureSpike
expr: failure_point_rates["FP1"] > 0.10
for: 5m
severity: critical
message: "FP1 (routing failure) rate exceeded 10%"
```

### Grafana Dashboard

Example Grafana panel JSON for routing metrics:

```json
{
  "title": "Query Routing Performance",
  "panels": [
    {
      "title": "Routing Latency",
      "type": "graph",
      "targets": [
        {
          "expr": "avg_routing_latency_ms",
          "legendFormat": "Avg Latency (ms)"
        }
      ]
    },
    {
      "title": "Cache Hit Rate",
      "type": "gauge",
      "targets": [
        {
          "expr": "cache_hit_rate * 100",
          "legendFormat": "Hit Rate (%)"
        }
      ],
      "min": 0,
      "max": 100,
      "thresholds": [
        {"value": 0, "color": "red"},
        {"value": 20, "color": "yellow"},
        {"value": 30, "color": "green"}
      ]
    },
    {
      "title": "Top Categories",
      "type": "piechart",
      "targets": [
        {
          "expr": "top_categories",
          "legendFormat": "{{ category }}"
        }
      ]
    }
  ]
}
```

## Troubleshooting

### Low Cache Hit Rate

**Diagnosis:**
```bash
# Check cache configuration
curl -s http://localhost:8000/api/database/routing-metrics | jq '.cache_hit_rate'

# Expected: 0.30-0.45
# If <0.20: investigate causes
```

**Common Causes:**
- High query diversity (expected for new users)
- Similarity threshold too strict (0.92 is high)
- Cache size too small (1000 entries)
- TTL too short (3600s may be insufficient)

**Solutions:**
```bash
# Relax similarity threshold
export ROUTING_CACHE_SIMILARITY_THRESHOLD=0.88

# Increase cache size and TTL
export ROUTING_CACHE_SIZE=2000
export ROUTING_CACHE_TTL=7200

# Restart backend
docker compose restart backend
```

### High Latency (>300ms)

**Diagnosis:**
```bash
# Get average latency
curl -s http://localhost:8000/api/database/routing-metrics | jq '.avg_routing_latency_ms'
```

**Common Causes:**
- Slow LLM provider (OpenAI API latency)
- Large category taxonomy (long prompt)
- Cache disabled

**Solutions:**
```bash
# Use faster model
export OPENAI_MODEL=gpt-4o-mini  # vs gpt-4

# Ensure cache enabled
export ENABLE_ROUTING_CACHE=true

# Consider local LLM for routing
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2:3b
```

### High Fallback Rate (>15%)

**Diagnosis:**
```bash
# Check fallback rate
curl -s http://localhost:8000/api/database/routing-metrics | jq '.fallback_rate'
```

**Common Causes:**
- Categories too narrow (over-filtering)
- Confidence threshold too low
- Sparse document collection

**Solutions:**
```bash
# Raise confidence threshold (more conservative)
export QUERY_ROUTING_CONFIDENCE_THRESHOLD=0.8

# Increase fallback minimum
export FALLBACK_MIN_RESULTS=5

# Verify document classification
python scripts/inspect_categories.py
```

### High FP2 Rate (Insufficient Results)

**Diagnosis:**
```bash
# Check FP2 rate
curl -s http://localhost:8000/api/database/routing-metrics | jq '.failure_point_rates.FP2'
```

**Common Causes:**
- Category filtering too aggressive
- Documents not classified
- Empty categories

**Solutions:**
```bash
# Reindex documents with classification
python scripts/reindex_classification.py

# Check category coverage
curl http://localhost:8000/api/database/category-stats
```

## Integration

### With Chat Interface

Routing metrics are displayed inline in the chat UI:

```typescript
// frontend/src/components/Chat/ChatInterface.tsx
{message.routing_info && (
  <RoutingBadge
    categories={message.routing_info.categories}
    confidence={message.routing_info.confidence}
    cacheHit={message.routing_info.cache_hit}
  />
)}
```

### With Logging

Metrics are automatically logged at INFO level:

```
[INFO] Routing cache hit: ['configure', 'installation'] (confidence 0.88, filter=True)
[INFO] Smart consolidation: 47 → 15 (repr) → 12 (dedup) → 10 (budget)
[WARNING] Failure Point FP2 detected: {'query': '...', 'chunk_count': 2, 'reason': 'insufficient_chunks'}
```

### With External Monitoring

Export metrics to Prometheus:

```python
from prometheus_client import Gauge, Counter

routing_latency_gauge = Gauge('routing_latency_ms', 'Routing decision latency')
cache_hit_counter = Counter('routing_cache_hits', 'Cache hit count')
cache_miss_counter = Counter('routing_cache_misses', 'Cache miss count')

# Update from routing_metrics
stats = routing_metrics.get_stats()
routing_latency_gauge.set(stats['avg_routing_latency_ms'])
cache_hit_counter.inc(routing_metrics.cache_hits)
```

## Related Documentation

- [Query Routing](04-features/query-routing.md) - Core routing architecture
- [Smart Consolidation](04-features/smart-consolidation.md) - Category-aware result ranking
- [Adaptive Routing](04-features/adaptive-routing.md) - Feedback-based learning
- [Performance Optimization](../../docs/OPTIMIZATION_FINDINGS.md) - System-wide performance tuning

## API Reference

### Get Metrics

```bash
GET /api/database/routing-metrics
```

**Response Schema:**
```typescript
interface RoutingMetrics {
  total_queries: number;
  avg_routing_latency_ms: number;
  routing_accuracy: number | null;
  cache_hit_rate: number;
  fallback_rate: number;
  multi_category_rate: number;
  top_categories: [string, number][];
  failure_point_rates: Record<string, number>;
  failure_point_counts: Record<string, number>;
}
```

### Record Feedback

```bash
POST /api/routing/feedback
Content-Type: application/json

{
  "message_id": "msg_123",
  "rating": 1,
  "categories": ["installation"],
  "confidence": 0.88
}
```

**Response:**
```json
{
  "success": true,
  "updated_accuracy": 0.89
}
```

### Reset Metrics

```bash
POST /api/database/routing-metrics/reset
```

**Response:**
```json
{
  "success": true,
  "message": "Routing metrics reset"
}
```
