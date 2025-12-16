# TruLens Continuous Monitoring for Amber GraphRAG

Continuous real-time monitoring of RAG pipeline quality using TruLens feedback functions.

## Overview

TruLens provides **continuous monitoring** for the Amber GraphRAG pipeline, complementing the existing RAGAS system:

- **RAGAS**: Bi-annual batch benchmarks to find optimal configurations
- **TruLens**: Continuous real-time monitoring in production

### Key Features

- ✅ **Non-invasive**: Opt-in via environment variable, no core code modifications
- ✅ **Production-ready**: PostgreSQL storage with Prometheus metrics
- ✅ **Observable**: Grafana dashboards and Streamlit UI
- ✅ **Configurable**: Sampling rate, feedback functions, thresholds
- ✅ **Self-contained**: All files in `evals/trulens/`, easy to remove

## Architecture

```
TruLens Recording Pipeline:
  GraphRAG.query()
    → TruLens instrumentation
    → PostgreSQL storage
    → Feedback function evaluation (async)

Metrics Export Pipeline:
  PostgreSQL aggregations
    → /api/trulens/metrics endpoint (Prometheus format)
    → Prometheus scraping (every 15s)
    → Grafana visualization
```

## Installation

### 1. Install Dependencies

```bash
uv pip install -r evals/trulens/requirements-trulens.txt
```

### 2. Start PostgreSQL

```bash
# Start TruLens PostgreSQL database
docker compose --profile trulens up -d
```

This starts the `postgres-trulens` service on port 5433.

### 3. Configure

```bash
# Copy configuration template
cp evals/trulens/config.example.yaml evals/trulens/config.yaml

# Edit config.yaml
# - Set enabled: true
# - Configure database credentials
# - Adjust sampling_rate if needed (1.0 = 100%, 0.1 = 10%)
```

### 4. Enable Monitoring

Set environment variable:

```bash
export ENABLE_TRULENS_MONITORING=1
```

Or add to `.env`:

```
ENABLE_TRULENS_MONITORING=1
```

### 5. Start Backend

```bash
uvicorn api.main:app --reload
```

You should see:

```
✅ TruLens continuous monitoring initialized successfully
   Database: postgresql://postgres:***@localhost:5433/trulens
   Sampling rate: 100%
   Feedback functions: 5
```

## Usage

### View TruLens Dashboard

Launch the Streamlit dashboard:

```bash
uv run python evals/trulens/dashboard_launcher.py
```

Open http://localhost:8501 in your browser.

### Access Prometheus Metrics

```bash
curl http://localhost:8000/api/trulens/metrics
```

### View JSON Stats

```bash
curl http://localhost:8000/api/trulens/stats
```

### Export Reports

```python
from evals.trulens.utils.export_utils import TruLensExporter

exporter = TruLensExporter()

# Export to JSON
exporter.export_to_json('reports/trulens/records.json')

# Export to CSV
exporter.export_to_csv('reports/trulens/metrics.csv')

# Export summary to Markdown
exporter.export_summary_markdown('reports/trulens/summary.md')
```

## Feedback Functions

TruLens evaluates 5 feedback functions:

1. **Answer Relevance** (0-1): Does the answer address the question?
2. **Context Relevance** (0-1): Are retrieved chunks relevant to the query?
3. **Groundedness** (0-1): Is the answer faithful to the retrieved context?
4. **Graph Reasoning Quality** (0-1): Did graph enrichment add value?
5. **Latency Check** (0-1): Is response time acceptable (<5s)?

## Configuration

### Database Backends

**PostgreSQL (recommended for production):**

```yaml
trulens:
  database:
    backend: "postgresql"
    postgresql:
      host: "postgres-trulens"  # Docker Compose service name
      port: 5432
      database: "trulens"
      user: "postgres"
      password: "${TRULENS_DB_PASSWORD}"
```

**SQLite (development/CI):**

```yaml
trulens:
  database:
    backend: "sqlite"
    sqlite:
      path: "evals/trulens/trulens.db"
```

### Sampling Rate

Adjust to reduce LLM API costs in high-traffic environments:

```yaml
trulens:
  sampling_rate: 0.1  # Monitor 10% of requests
```

### Thresholds

Set quality thresholds for alerting:

```yaml
feedback:
  thresholds:
    answer_relevance_min: 0.7
    groundedness_min: 0.8
    latency_max_ms: 5000
```

## Prometheus Integration

### Metrics Endpoint

TruLens exposes metrics at `/api/trulens/metrics`:

- `trulens_answer_relevance_score` (gauge)
- `trulens_groundedness_score` (gauge)
- `trulens_context_relevance_score` (gauge)
- `trulens_query_latency_seconds` (histogram with p50, p95, p99)
- `trulens_queries_total` (counter)
- `trulens_errors_total` (counter)

### Prometheus Configuration

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'trulens'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/trulens/metrics'
    scrape_interval: 15s
```

## Grafana Dashboard

Import the pre-built dashboard:

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `evals/trulens/observability/grafana_dashboard.json`

The dashboard includes 6 panels:
- Answer Quality Over Time
- Query Latency Percentiles (p50, p95, p99)
- Queries Per Minute
- Error Rate
- Feedback Score Distribution
- Quality Score by Retrieval Mode

## Troubleshooting

### TruLens not installed

```
❌ TruLens not installed
Install with: uv pip install -r evals/trulens/requirements-trulens.txt
```

### PostgreSQL connection failed

```
Failed to initialize TruLens: could not connect to server
```

**Fix**: Ensure PostgreSQL is running:

```bash
docker compose --profile trulens up -d
docker compose ps  # Check postgres-trulens is running
```

### No metrics available

```
# TruLens monitoring is disabled
```

**Fix**: Check configuration:

1. `ENABLE_TRULENS_MONITORING=1` is set
2. `config.yaml` has `enabled: true`
3. Backend restarted after configuration changes

### Dashboard shows no data

**Fix**: Send some queries to populate data:

```bash
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Carbonio?", "retrieval_mode": "graph_enhanced"}'
```

## Relationship to Other Systems

### vs RAGAS

- **RAGAS**: Offline batch evaluation with gold standard datasets
- **TruLens**: Online continuous monitoring of live queries
- **Use both**: RAGAS for benchmarking, TruLens for production monitoring

### vs quality_monitor

- **quality_monitor**: Lightweight in-memory metrics, real-time alerts
- **TruLens**: Persistent detailed feedback analysis, historical trends
- **Complementary**: quality_monitor for alerts, TruLens for analysis

## Performance Impact

- **Instrumentation overhead**: <10% of query latency
- **Sampling**: Reduce overhead by lowering sampling_rate
- **Async feedback**: Feedback functions run asynchronously (non-blocking)

## Database Management

### Reset Database

```python
from evals.trulens.utils.db_manager import reset_database

reset_database("postgresql://postgres:password@localhost:5433/trulens")
```

### Backup Database

```python
from evals.trulens.utils.db_manager import backup_database

backup_database(
    "sqlite:///evals/trulens/trulens.db",
    "backups/trulens_backup_20251213.db"
)
```

## Contributing

TruLens monitoring is self-contained in `evals/trulens/`. To add features:

1. Add feedback functions in `feedback_functions.py`
2. Update metrics in `observability/prometheus_metrics.py`
3. Update dashboard in `observability/grafana_dashboard.json`

## License

Same as Amber GraphRAG project.
