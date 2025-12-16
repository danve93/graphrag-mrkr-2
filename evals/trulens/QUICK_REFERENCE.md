# TruLens Quick Reference

Quick start guide for TruLens continuous monitoring.

## Installation (One-Time Setup)

```bash
# 1. Install dependencies
uv pip install -r evals/trulens/requirements-trulens.txt

# 2. Start PostgreSQL
docker compose --profile trulens up -d

# 3. Configure
cp evals/trulens/config.example.yaml evals/trulens/config.yaml
# Edit config.yaml: set enabled: true

# 4. Set environment variable
export ENABLE_TRULENS_MONITORING=1
# Or add to .env file: ENABLE_TRULENS_MONITORING=1
```

## Daily Usage

### Start Monitoring

```bash
# Start backend with monitoring enabled
ENABLE_TRULENS_MONITORING=1 uvicorn api.main:app --reload

# Or with Docker Compose (add to .env):
ENABLE_TRULENS_MONITORING=1
docker compose up -d
```

### View Dashboard

```bash
# Launch TruLens Streamlit dashboard
uv run python evals/trulens/dashboard_launcher.py
# Open: http://localhost:8501
```

### Check Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/api/trulens/metrics

# JSON stats
curl http://localhost:8000/api/trulens/stats

# Health check
curl http://localhost:8000/api/trulens/health
```

### Export Reports

```bash
# Export summary to Markdown
uv run python -c "
from evals.trulens.utils.export_utils import TruLensExporter
exporter = TruLensExporter()
exporter.export_summary_markdown('reports/trulens/summary.md')
"

# Export to CSV
uv run python -c "
from evals.trulens.utils.export_utils import TruLensExporter
exporter = TruLensExporter()
exporter.export_to_csv('reports/trulens/metrics.csv')
"
```

## Common Tasks

### Change Sampling Rate

Edit `config.yaml`:

```yaml
trulens:
  sampling_rate: 0.1  # 10% of requests
```

Restart backend.

### Switch to SQLite (Development)

Edit `config.yaml`:

```yaml
trulens:
  database:
    backend: "sqlite"
```

No PostgreSQL needed.

### Disable Monitoring

```bash
# Method 1: Environment variable
unset ENABLE_TRULENS_MONITORING
# or
export ENABLE_TRULENS_MONITORING=0

# Method 2: Config file
# Edit config.yaml: enabled: false

# Restart backend
```

### Reset Database

```bash
uv run python -c "
from evals.trulens.utils.db_manager import reset_database
reset_database('postgresql://postgres:trulens_password@localhost:5433/trulens')
"
```

## Prometheus + Grafana Setup

### Add to Prometheus

Edit `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'trulens'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/trulens/metrics'
    scrape_interval: 15s
```

### Import Grafana Dashboard

1. Open Grafana → Dashboards → Import
2. Upload `evals/trulens/observability/grafana_dashboard.json`
3. Select Prometheus data source
4. Click Import

## Troubleshooting

### Monitoring not working

```bash
# Check if enabled
curl http://localhost:8000/api/trulens/health

# Check logs
docker compose logs backend | grep TruLens

# Verify PostgreSQL is running
docker compose ps postgres-trulens
```

### Dashboard has no data

```bash
# Send a test query
curl -X POST http://localhost:8000/api/chat/query \
  -H "Content-Type: application/json" \
  -d '{"message": "Test query", "retrieval_mode": "graph_enhanced"}'

# Refresh dashboard
```

### High overhead

Reduce sampling rate in `config.yaml`:

```yaml
trulens:
  sampling_rate: 0.1  # Monitor only 10%
```

## File Locations

- **Config**: `evals/trulens/config.yaml`
- **Database**: PostgreSQL on port 5433 (or `evals/trulens/trulens.db` for SQLite)
- **Metrics**: http://localhost:8000/api/trulens/metrics
- **Dashboard**: http://localhost:8501 (when launched)
- **Logs**: Check backend logs for "TruLens" entries

## Feedback Functions

Monitor these 5 metrics via dashboard or Prometheus:

1. **Answer Relevance**: 0-1 (target: >0.7)
2. **Groundedness**: 0-1 (target: >0.8)
3. **Context Relevance**: 0-1 (target: >0.6)
4. **Graph Reasoning Quality**: 0-1 (custom metric)
5. **Latency Check**: 0-1 (target: <5 seconds)

## Next Steps

- See full documentation: [README.md](README.md)
- Customize feedback functions: [feedback_functions.py](feedback_functions.py)
- OpenTelemetry integration: [observability/opentelemetry_notes.md](observability/opentelemetry_notes.md)
