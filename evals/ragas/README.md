# Ragas Evaluation Harness (Amber)

Self-contained evaluation utilities for Amber's GraphRAG using Ragas. This folder is optional and removable; it does not touch `settings.py` or core services.

**📚 Full Documentation:** [documentation/08-operations/evaluation-system.md](../../documentation/08-operations/evaluation-system.md)
**⚡ Quick Reference:** [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

---

## Install (optional deps)
```bash
uv pip install -r evals/ragas/requirements-ragas.txt
```

## Dataset expectations
- Location: `evals/ragas/testsets/`
- Template/gold files you’ll provide: `amber_carbonio_ragas_eval_template.{jsonl,csv}`, `amber_carbonio_ragas_gold_testset.{jsonl,csv}`
- Fields per row:
  - `user_input` (question)
  - `reference` (ground truth answer)
  - `retrieved_contexts` (optional list of strings)
  - `response` (optional prefilled answer; filled by runner when calling backend)
  - `metadata`: `intent` (admin|user), `source_doc`, `source_pages`, `qa_type`

## Config
Edit `evals/ragas/config.example.yaml` (copy to `config.yaml` if desired):
- `backend`: base URL, endpoints, optional API key env var (leave null if not needed), timeout
- `defaults`: model names, concurrency, retries (start with low concurrency and higher timeout to avoid timeouts)
- `variants`: per-run payload flags (router on/off, structured_kg on/off, rrf, flashrank, clustering)
- `paths`: dataset, output_dir, optional baseline file
- `reporting`: metrics to compute

## Commands (examples)
```bash
# run vector-only
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant vector_only \
  --out reports/ragas/vector_only.json \
  --progress-every 5 \
  --progress-interval 5

# run graph/hybrid
uv run python evals/ragas/ragas_runner.py \
  --config evals/ragas/config.example.yaml \
  --dataset evals/ragas/testsets/amber_carbonio_ragas_gold_testset.csv \
  --variant graph_hybrid \
  --out reports/ragas/graph_hybrid.json \
  --progress-every 5 \
  --progress-interval 5

# compare (regression check vs baseline if provided)
uv run python evals/ragas/ragas_report.py \
  --inputs reports/ragas/vector_only.json reports/ragas/graph_hybrid.json \
  --baseline reports/ragas/baseline.json \
  --out reports/ragas/summary.md
```

## What the scripts do (scaffold)
- `ragas_runner.py`: loads dataset, calls the live backend (`/api/chat/query`, non-stream) to collect the same answers/contexts users get, applies variant flags (e.g., retrieval_mode), computes Ragas metrics, writes raw JSON report.
- `ragas_report.py`: aggregates raw reports, checks >5% regressions vs baseline, computes cross-doc leakage from metadata, emits Markdown summary.

## Observability Integration

### Prometheus Metrics

RAGAS benchmark results are exposed as Prometheus metrics for monitoring and alerting.

**Access Metrics:**
```bash
# Prometheus exposition format
curl http://localhost:8000/api/ragas/metrics

# JSON statistics
curl http://localhost:8000/api/ragas/stats

# Variant comparison
curl http://localhost:8000/api/ragas/comparison
```

**Metrics Exposed:**
```
# Core RAGAS metrics (0-1 scale)
ragas_context_precision{variant="graph_hybrid"} 0.712
ragas_context_recall{variant="graph_hybrid"} 0.634
ragas_faithfulness{variant="graph_hybrid"} 0.891
ragas_answer_relevancy{variant="graph_hybrid"} 0.823

# Metadata
ragas_last_evaluation_timestamp{variant="graph_hybrid"} 1702472400
ragas_evaluations_total{variant="graph_hybrid"} 150
```

**Prometheus Configuration:**

Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'ragas'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/api/ragas/metrics'
    scrape_interval: 5m  # RAGAS benchmarks run less frequently
```

### Grafana Dashboard

Import the pre-built RAGAS dashboard to visualize benchmark results.

**Steps:**
1. Open Grafana UI → Dashboards → Import
2. Upload `evals/ragas/grafana_dashboard.json`
3. Select your Prometheus data source
4. Click Import

**Dashboard Panels:**
- **RAGAS Metrics Comparison**: Time series showing all 4 metrics across variants
- **Faithfulness Score**: Stat panel with color thresholds (red <0.8, yellow 0.8-0.85, green >0.85)
- **Context Precision/Recall/Answer Relevancy**: Individual stat panels with thresholds
- **Variant Performance Comparison**: Bar gauge showing average score per variant
- **Last Evaluation Timestamp**: When each variant was last evaluated
- **Total Evaluations**: Cumulative evaluation count
- **Regression Detection**: Percentage change from baseline variant

### Variant Comparison API

Compare variants and detect performance regressions programmatically.

**Example:**
```bash
curl http://localhost:8000/api/ragas/comparison
```

**Response:**
```json
{
  "best_overall_variant": "graph_hybrid",
  "metric_winners": {
    "context_precision": "graph_hybrid",
    "context_recall": "graph_enhanced",
    "faithfulness": "graph_hybrid",
    "answer_relevancy": "graph_hybrid"
  },
  "all_variants": {
    "graph_hybrid": {
      "context_precision": 0.712,
      "context_recall": 0.634,
      "faithfulness": 0.891,
      "answer_relevancy": 0.823,
      "average_score": 0.765
    },
    "vector_only": {
      "context_precision": 0.654,
      "context_recall": 0.589,
      "faithfulness": 0.812,
      "answer_relevancy": 0.765,
      "average_score": 0.705
    }
  }
}
```

**Use Cases:**
- CI/CD pipelines: Fail build if new variant regresses >5%
- Automated alerts: Trigger notification when faithfulness drops below 0.8
- A/B testing: Programmatically select best variant for production

### OpenTelemetry Integration (Future)

Distributed tracing support for RAGAS batch evaluations is planned for future implementation.

**Benefits:**
- Trace individual sample evaluations
- Track LLM API call latency per metric
- Analyze parallelization efficiency
- Debug slow or failing samples

**Documentation:** See [opentelemetry_notes.md](opentelemetry_notes.md) for implementation guidance.

## Removal
Delete `evals/ragas/` (including `testsets/`) and `reports/ragas/` to fully remove the harness. No core files are modified.

## Progress visibility
- `--progress-every N`: prints a line every N completed requests (includes error counts and a sample error if present).
- `--progress-interval N`: prints heartbeat every N seconds with started/completed/errors.
