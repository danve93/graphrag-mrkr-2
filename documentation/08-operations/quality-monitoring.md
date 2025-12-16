# Quality Monitoring Operations Guide

**Status:** ✅ Implemented
**Version:** 2.0.0
**Last Updated:** 2025-12-12

## Overview

The Quality Monitoring system provides real-time tracking and alerting for retrieval quality metrics. It helps detect performance degradation, quality drops, and errors early to maintain high RAG system reliability.

### Key Capabilities

- **Continuous Metrics Tracking**: Monitors all RAG queries with quality scores, latencies, and cache hit rates
- **Automatic Alerting**: Triggers alerts on quality drops (>30% below baseline), latency spikes (>2x baseline), and errors
- **Baseline Calculation**: Establishes baseline metrics from initial queries for comparison
- **REST API**: Provides endpoints for metrics visualization and monitoring integration
- **Zero Performance Impact**: Lightweight monitoring with <1ms overhead per query

### Metrics Tracked

| Metric Category | Metrics | Description |
|----------------|---------|-------------|
| **Quality** | avg, min, max, p50, p95, stddev | Answer quality scores (0-100) |
| **Latency** | avg, p50, p95, p99 | Retrieval and generation latencies (ms) |
| **Cache** | hit_rate | Percentage of queries served from cache |
| **Distribution** | query_type_distribution | Breakdown by chunk/entity/hybrid |
| **Errors** | error_rate | Percentage of failed queries |

---

## Configuration

### Settings ([config/settings.py](../../config/settings.py#L209-L218))

```python
# Quality Monitoring Configuration
enable_quality_monitoring: bool = True      # Enable/disable monitoring
quality_monitor_window_size: int = 1000     # Number of queries to track
quality_alert_threshold: float = 0.7        # Alert when quality drops below 70% of baseline
```

### Environment Variables

```bash
# Disable quality monitoring
ENABLE_QUALITY_MONITORING=false

# Adjust window size (default: 1000 queries)
QUALITY_MONITOR_WINDOW_SIZE=2000

# Adjust alert threshold (default: 0.7 = alert below 70% of baseline)
QUALITY_ALERT_THRESHOLD=0.6
```

### Configuration Guide

**`enable_quality_monitoring`**:
- **true** (default): Monitor all queries and track metrics
- **false**: Disable monitoring (reduces overhead to 0)
- **When to disable**: Only if overhead is critical and you don't need monitoring

**`quality_monitor_window_size`**:
- **100-500**: Small window, faster memory footprint, shorter history
- **1000** (default): Good balance for most deployments
- **2000-5000**: Large window, longer history, more memory usage
- **Recommendation**: Use 1000 for production, increase if you need longer trend analysis

**`quality_alert_threshold`**:
- **0.5**: Very sensitive - alerts on 50% quality drop
- **0.7** (default): Balanced - alerts on 30% quality drop
- **0.9**: Conservative - only alerts on 10% quality drop
- **Recommendation**: Start with 0.7, tune based on alert noise

---

## API Endpoints

All endpoints are under `/api/metrics/retrieval/` prefix.

### 1. Get Monitoring Summary

**Endpoint:** `GET /api/metrics/retrieval/summary`

Returns high-level monitoring status with recent alerts.

**Response:**
```json
{
  "enabled": true,
  "window_size": 1000,
  "queries_tracked": 523,
  "baseline": {
    "quality_score": 85.2,
    "latency_p95_ms": 287.5
  },
  "current_metrics": {
    "avg_quality_score": 83.1,
    "p95_quality_score": 91.2,
    "avg_total_latency_ms": 245.8,
    "p95_total_latency_ms": 312.4,
    "cache_hit_rate": 32.5,
    "error_rate": 0.2
  },
  "recent_alerts": [
    {
      "timestamp": "2025-12-12T15:30:45Z",
      "type": "quality_drop",
      "severity": "warning",
      "message": "Quality score dropped to 55.0 (65% of baseline 85.2)"
    }
  ]
}
```

**Use Cases:**
- Dashboard health overview
- Quick system status check
- Integration with monitoring tools (Datadog, Grafana)

### 2. Get Detailed Metrics

**Endpoint:** `GET /api/metrics/retrieval/detailed?window=100`

Returns detailed aggregated metrics over a time window.

**Parameters:**
- `window` (optional): Number of recent queries to aggregate (default: all tracked)

**Response:**
```json
{
  "window_size": 100,
  "time_range_start": 1702396845.123,
  "time_range_end": 1702397205.456,
  "avg_quality_score": 83.5,
  "min_quality_score": 45.0,
  "max_quality_score": 98.2,
  "p50_quality_score": 85.0,
  "p95_quality_score": 93.5,
  "quality_score_stddev": 12.3,
  "avg_total_latency_ms": 245.8,
  "p50_total_latency_ms": 210.5,
  "p95_total_latency_ms": 450.2,
  "p99_total_latency_ms": 620.8,
  "avg_retrieval_latency_ms": 120.3,
  "avg_generation_latency_ms": 125.5,
  "cache_hit_rate": 0.325,
  "query_type_distribution": {
    "chunk": 65,
    "entity": 20,
    "hybrid": 15
  },
  "error_rate": 0.02,
  "total_queries": 100
}
```

**Use Cases:**
- Detailed performance analysis
- Identifying performance trends
- Debugging quality issues
- Capacity planning

### 3. Get Recent Alerts

**Endpoint:** `GET /api/metrics/retrieval/alerts?limit=20`

Returns recent quality alerts sorted by timestamp (most recent first).

**Parameters:**
- `limit` (optional): Maximum alerts to return (1-100, default: 10)

**Response:**
```json
{
  "total_alerts": 3,
  "alerts": [
    {
      "timestamp": 1702397205.456,
      "type": "latency_spike",
      "severity": "warning",
      "message": "Latency spiked to 850.0ms (3.0x baseline p95 287.5ms)",
      "current_value": 850.0,
      "baseline_value": 287.5,
      "threshold": 2.0
    },
    {
      "timestamp": 1702396905.123,
      "type": "quality_drop",
      "severity": "critical",
      "message": "Quality score dropped to 35.0 (41% of baseline 85.2)",
      "current_value": 35.0,
      "baseline_value": 85.2,
      "threshold": 0.7
    },
    {
      "timestamp": 1702396605.789,
      "type": "error",
      "severity": "warning",
      "message": "Query failed: Database connection timeout",
      "current_value": 1.0,
      "baseline_value": 0.0,
      "threshold": 0.0
    }
  ]
}
```

**Use Cases:**
- Incident response
- Alert history review
- Trend analysis for recurring issues

### 4. Reset Monitor

**Endpoint:** `POST /api/metrics/retrieval/reset`

Resets the quality monitor (clears all metrics, alerts, and baseline).

**Response:**
```json
{
  "status": "success",
  "message": "Quality monitor reset successfully"
}
```

**Use Cases:**
- After major system changes (model upgrade, configuration change)
- Testing and development
- Clearing stale baseline after deployment

**⚠️ Warning:** Use with caution in production - clears all historical data.

---

## Alert Types and Response

### 1. Quality Drop Alerts

**Trigger:** Quality score falls below `quality_alert_threshold` (default 70%) of baseline

**Severity Levels:**
- **Critical**: Quality < 50% of baseline
- **Warning**: Quality 50-70% of baseline

**Example:**
```
Quality score dropped to 55.0 (65% of baseline 85.0)
```

**Possible Causes:**
- LLM model degradation or API issues
- Poor quality chunks in knowledge base
- Retrieval returning irrelevant context
- Context length issues (truncation)

**Response Actions:**
1. **Check recent ingestion**: Did new documents introduce low-quality content?
   ```bash
   # Check recent document ingestion logs
   grep "Document ingested" logs/backend.log | tail -20
   ```

2. **Verify LLM model status**: Is the LLM API healthy?
   ```bash
   # Check LLM response times
   curl /api/metrics/retrieval/detailed?window=50
   # Look at avg_generation_latency_ms
   ```

3. **Review retrieved chunks**: Are they relevant?
   ```bash
   # Enable DEBUG logging to see retrieved chunks
   export LOG_LEVEL=DEBUG
   ```

4. **Check context truncation**: Is context being cut off?
   - Review `max_context_length` setting
   - Check if chunks are too large

5. **Reindex if needed**: If quality persists, reindex the knowledge base

### 2. Latency Spike Alerts

**Trigger:** Total latency exceeds 2x baseline p95 latency

**Severity Levels:**
- **Critical**: Latency > 5x baseline
- **Warning**: Latency 2-5x baseline

**Example:**
```
Latency spiked to 1200.0ms (4.0x baseline p95 300.0ms)
```

**Possible Causes:**
- Database slow queries (Neo4j)
- LLM API slowdown
- Network issues
- High system load
- Large result sets

**Response Actions:**
1. **Identify bottleneck**: Check retrieval vs generation latency
   ```bash
   curl /api/metrics/retrieval/detailed
   # Compare avg_retrieval_latency_ms vs avg_generation_latency_ms
   ```

2. **If retrieval is slow**:
   - Check Neo4j performance: `CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Transactions")`
   - Review indexes: `SHOW INDEXES`
   - Check vector index status

3. **If generation is slow**:
   - Check LLM API status
   - Verify network connectivity to LLM provider
   - Review recent LLM model changes

4. **Check system resources**:
   ```bash
   # CPU and memory usage
   top

   # Neo4j memory
   docker stats neo4j
   ```

5. **Enable two-stage retrieval** if corpus is large:
   ```bash
   export ENABLE_TWO_STAGE_RETRIEVAL=true
   export TWO_STAGE_THRESHOLD_DOCS=5000
   ```

### 3. Error Alerts

**Trigger:** Query fails with an exception

**Severity:** Warning (all errors)

**Example:**
```
Query failed: Database connection timeout
```

**Possible Causes:**
- Database connectivity issues
- LLM API failures
- Malformed queries
- Resource exhaustion
- Configuration errors

**Response Actions:**
1. **Check error logs**:
   ```bash
   tail -f logs/backend.log | grep ERROR
   ```

2. **Verify service health**:
   ```bash
   # Check all services
   docker-compose ps

   # Check Neo4j health
   curl localhost:7474

   # Check backend health
   curl localhost:8000/api/health
   ```

3. **Review recent changes**: Did deployment or configuration change recently?

4. **Check error rate trend**:
   ```bash
   curl /api/metrics/retrieval/detailed
   # Review error_rate over time
   ```

5. **Restart services if needed**:
   ```bash
   docker-compose restart backend
   # or
   docker-compose restart neo4j
   ```

---

## Monitoring Integration

### Grafana Dashboard

Example Prometheus queries for Grafana:

```promql
# Average quality score
avg(quality_monitor_quality_score)

# P95 latency
histogram_quantile(0.95, quality_monitor_latency_ms)

# Cache hit rate
rate(quality_monitor_cache_hits[5m]) / rate(quality_monitor_total_queries[5m])

# Error rate
rate(quality_monitor_errors[5m])
```

### Datadog Integration

```python
# Custom integration example
import requests
from datadog import statsd

def sync_metrics_to_datadog():
    response = requests.get("http://localhost:8000/api/metrics/retrieval/summary")
    data = response.json()

    if "current_metrics" in data:
        metrics = data["current_metrics"]
        statsd.gauge("rag.quality_score", metrics["avg_quality_score"])
        statsd.gauge("rag.latency_p95", metrics["p95_total_latency_ms"])
        statsd.gauge("rag.cache_hit_rate", metrics["cache_hit_rate"])
        statsd.gauge("rag.error_rate", metrics["error_rate"])
```

### PagerDuty Alerting

```python
# Alert webhook integration
import requests

def check_alerts_and_notify():
    response = requests.get("http://localhost:8000/api/metrics/retrieval/alerts?limit=5")
    alerts = response.json()["alerts"]

    for alert in alerts:
        if alert["severity"] == "critical":
            # Trigger PagerDuty incident
            requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json={
                    "routing_key": "YOUR_INTEGRATION_KEY",
                    "event_action": "trigger",
                    "payload": {
                        "summary": alert["message"],
                        "severity": "critical",
                        "source": "RAG Quality Monitor",
                    }
                }
            )
```

---

## Troubleshooting

### Problem: No Baseline Established

**Symptoms:** Summary shows `baseline: null` even after many queries

**Causes:**
- Fewer than 100 queries (default threshold)
- Quality scoring disabled
- All queries are cache hits (no quality scores)

**Solutions:**
1. Wait for 100 queries to establish baseline
2. Check if quality scoring is enabled:
   ```bash
   grep enable_quality_scoring config/settings.py
   ```
3. Lower baseline threshold temporarily:
   ```python
   monitor.baseline_window_size = 50  # Lower from 100
   ```

### Problem: Too Many Alerts

**Symptoms:** Constant alert spam, alert fatigue

**Causes:**
- Alert threshold too sensitive
- High natural variability in queries
- Recent system changes not accounted for

**Solutions:**
1. Increase alert threshold:
   ```bash
   export QUALITY_ALERT_THRESHOLD=0.5  # More lenient (alert below 50%)
   ```

2. Reset baseline after major changes:
   ```bash
   curl -X POST http://localhost:8000/api/metrics/retrieval/reset
   ```

3. Review query patterns for outliers

### Problem: Monitoring Overhead Too High

**Symptoms:** Increased query latency, high memory usage

**Causes:**
- Very large window size
- Too many queries per second

**Solutions:**
1. Reduce window size:
   ```bash
   export QUALITY_MONITOR_WINDOW_SIZE=500  # Smaller window
   ```

2. Disable monitoring temporarily:
   ```bash
   export ENABLE_QUALITY_MONITORING=false
   ```

3. Sample queries instead of monitoring all (future enhancement)

### Problem: Metrics Not Updating

**Symptoms:** API returns stale metrics

**Causes:**
- Monitoring disabled
- No recent queries
- Cache serving all requests

**Solutions:**
1. Verify monitoring is enabled:
   ```bash
   curl http://localhost:8000/api/metrics/retrieval/summary | grep enabled
   ```

2. Check queries are being processed:
   ```bash
   tail -f logs/backend.log | grep "Processing query"
   ```

3. Send a test query to verify:
   ```bash
   curl -X POST http://localhost:8000/api/chat/query \
     -H "Content-Type: application/json" \
     -d '{"query": "test query"}'
   ```

---

## Best Practices

### 1. Establish Baseline in Production

Wait for 100+ real production queries before relying on alerts. Initial queries may not be representative.

### 2. Review Alerts Weekly

Set up a weekly review of alert patterns:
```bash
# Get last week's alerts
curl /api/metrics/retrieval/alerts?limit=100
```

Identify recurring issues and fix root causes rather than just responding to alerts.

### 3. Tune Thresholds Gradually

Start with default threshold (0.7) and adjust based on false positive rate:
- Too many false positives → Lower threshold (0.5-0.6)
- Missing real issues → Raise threshold (0.8-0.9)

### 4. Reset After Major Changes

After model upgrades, config changes, or reindexing:
```bash
curl -X POST http://localhost:8000/api/metrics/retrieval/reset
```

This prevents false alerts due to expected quality shifts.

### 5. Monitor Trends, Not Just Alerts

Use detailed metrics to spot gradual degradation:
```bash
# Compare last 100 vs last 500 queries
curl /api/metrics/retrieval/detailed?window=100 > recent.json
curl /api/metrics/retrieval/detailed?window=500 > historical.json
```

### 6. Integrate with Existing Monitoring

Don't create a monitoring island - integrate with your existing tools (Grafana, Datadog, PagerDuty) for unified observability.

---

## Related Documentation

- [Monitoring Guide](./monitoring.md) - General system monitoring
- [Troubleshooting Guide](./troubleshooting.md) - Common issues and solutions
- [Health Endpoints](./observability-health.md) - System health checks

---

**Last Updated:** 2025-12-12
**Feature Status:** ✅ Production Ready
