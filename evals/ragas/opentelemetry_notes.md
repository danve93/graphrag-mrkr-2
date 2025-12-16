# OpenTelemetry Integration for RAGAS (Future Enhancement)

Notes and guidance for integrating OpenTelemetry (OTLP) distributed tracing with RAGAS batch evaluation.

## Overview

OpenTelemetry provides distributed tracing capabilities that can complement RAGAS benchmark evaluation:

- **RAGAS**: Evaluates quality metrics on test datasets (context precision, faithfulness, etc.)
- **OpenTelemetry**: Tracks execution flow, performance, parallelization efficiency

## Why OpenTelemetry for RAGAS?

Current RAGAS implementation tracks:
- Overall benchmark duration
- Final metric scores per variant
- Test case results (pass/fail)

OpenTelemetry would add:
- **Batch tracing**: Track entire benchmark run → variant evaluation → sample processing
- **Span timing**: See exactly which samples or metrics are slow
- **Parallelization analysis**: Identify bottlenecks in concurrent evaluation
- **LLM call tracking**: Monitor API latency, retries, token usage per metric
- **Correlation**: Link benchmark runs to code commits via trace metadata

## Architecture

```
Benchmark Run (root span)
  ↓
[SPAN: ragas.benchmark_run] ← Root span (variant: graph_hybrid)
  ↓
  ├─ [SPAN: ragas.dataset_load] ← Load test dataset
  ├─ [SPAN: ragas.variant_evaluation] ← Evaluate one variant
  │    ├─ [SPAN: ragas.sample_evaluation] ← Process single sample (parallel)
  │    │    ├─ [SPAN: ragas.metric.context_precision] ← Compute metric
  │    │    │    ├─ [SPAN: llm_api_call] ← OpenAI API call
  │    │    │    └─ [SPAN: llm_api_call] ← Retry/follow-up
  │    │    ├─ [SPAN: ragas.metric.faithfulness]
  │    │    │    ├─ [SPAN: llm_api_call]
  │    │    │    └─ [SPAN: llm_api_call]
  │    │    ├─ [SPAN: ragas.metric.context_recall]
  │    │    └─ [SPAN: ragas.metric.answer_relevancy]
  │    └─ [SPAN: ragas.aggregation] ← Compute final scores
  └─ [SPAN: ragas.report_generation] ← Generate JSON/Markdown reports
```

## Use Cases

### 1. Performance Profiling
Identify which metrics or samples are slowest:

```python
# Example trace analysis query (Jaeger/Tempo)
# "Find slowest RAGAS samples in last benchmark run"

SELECT sample_id, duration_ms
FROM spans
WHERE span_name = 'ragas.sample_evaluation'
  AND trace_id = '<benchmark_trace_id>'
ORDER BY duration_ms DESC
LIMIT 10;
```

### 2. LLM Cost Attribution
Track LLM API calls per metric to understand cost breakdown:

```python
# "How many LLM calls does faithfulness metric require per sample?"

SELECT
  parent_span_name AS metric,
  COUNT(*) AS llm_calls,
  AVG(duration_ms) AS avg_latency_ms
FROM spans
WHERE span_name = 'llm_api_call'
GROUP BY parent_span_name;
```

### 3. Parallelization Efficiency
Analyze whether concurrent evaluation is effective:

```python
# "Are samples being processed in parallel or sequentially?"

# Good: Overlapping sample_evaluation spans (parallel)
# Bad: Sequential sample_evaluation spans (bottleneck)
```

## Implementation Approach

### 1. Install OpenTelemetry SDK

Add to `requirements-ragas.txt`:

```txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation>=0.41b0
```

### 2. Initialize OTLP Exporter

```python
# evals/ragas/otlp_config.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def initialize_otlp(endpoint: str = "http://localhost:4317"):
    """Initialize OpenTelemetry tracing for RAGAS."""
    # Set up tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)

    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    return trace.get_tracer("amber.ragas")
```

### 3. Instrument Benchmark Runner

```python
# Modify evals/ragas/run_ragas_benchmarks.py

from opentelemetry import trace
from typing import Optional

class RagasBenchmarkRunner:
    def __init__(self, enable_otlp: bool = False):
        self.enable_otlp = enable_otlp
        self.tracer: Optional[trace.Tracer] = None

        if enable_otlp:
            from evals.ragas.otlp_config import initialize_otlp
            self.tracer = initialize_otlp()

    def run_benchmark(self, variant_name: str, dataset_path: str):
        """Run benchmark with optional tracing."""
        if self.tracer:
            with self.tracer.start_as_current_span(
                "ragas.benchmark_run",
                attributes={
                    "variant": variant_name,
                    "dataset": dataset_path,
                    "ragas_version": ragas.__version__,
                }
            ) as span:
                result = self._run_with_tracing(variant_name, dataset_path, span)
                return result
        else:
            # No tracing, execute normally
            return self._run_without_tracing(variant_name, dataset_path)

    def _run_with_tracing(self, variant_name, dataset_path, root_span):
        """Execute benchmark with span instrumentation."""
        # Load dataset (child span)
        with self.tracer.start_as_current_span("ragas.dataset_load") as span:
            dataset = self._load_dataset(dataset_path)
            span.set_attribute("sample_count", len(dataset))

        # Evaluate variant (child span)
        with self.tracer.start_as_current_span(
            "ragas.variant_evaluation",
            attributes={"variant": variant_name}
        ) as span:
            results = self._evaluate_variant(variant_name, dataset)
            span.set_attribute("success_rate", results.success_rate)

        # Generate report (child span)
        with self.tracer.start_as_current_span("ragas.report_generation"):
            self._generate_reports(variant_name, results)

        return results
```

### 4. Instrument Sample Evaluation (Parallel)

```python
# Instrument individual sample processing

def evaluate_sample(self, sample, metrics):
    """Evaluate single sample with tracing."""
    if self.tracer:
        with self.tracer.start_as_current_span(
            "ragas.sample_evaluation",
            attributes={
                "sample_id": sample.id,
                "question": sample.question[:100],  # Truncate for privacy
            }
        ) as span:
            metric_scores = {}

            # Evaluate each metric
            for metric_name in metrics:
                metric_scores[metric_name] = self._evaluate_metric(
                    sample, metric_name
                )

            # Add results to span
            span.set_attribute("avg_score", sum(metric_scores.values()) / len(metric_scores))

            return metric_scores
    else:
        # No tracing
        return self._evaluate_sample_without_tracing(sample, metrics)
```

### 5. Instrument LLM Calls

```python
# Track individual LLM API calls for cost attribution

def evaluate_metric(self, sample, metric_name):
    """Evaluate single metric with LLM call tracing."""
    if self.tracer:
        with self.tracer.start_as_current_span(
            f"ragas.metric.{metric_name}",
            attributes={"metric": metric_name}
        ) as metric_span:
            # Instrument LLM calls
            with self.tracer.start_as_current_span(
                "llm_api_call",
                attributes={
                    "provider": "openai",
                    "model": "gpt-4",
                    "metric": metric_name,
                }
            ) as llm_span:
                # Call LLM
                response = self._call_llm(sample, metric_name)

                # Add LLM metadata
                llm_span.set_attribute("prompt_tokens", response.usage.prompt_tokens)
                llm_span.set_attribute("completion_tokens", response.usage.completion_tokens)
                llm_span.set_attribute("total_tokens", response.usage.total_tokens)
                llm_span.set_attribute("latency_ms", response.latency_ms)

                return self._parse_metric_score(response)
```

### 6. Link Traces to Benchmark Reports

```python
# Correlate OTLP trace ID with RAGAS report

def run_benchmark(self, variant_name, dataset_path):
    """Run benchmark and save trace ID in report."""
    if self.tracer:
        with self.tracer.start_as_current_span("ragas.benchmark_run") as span:
            # Get trace ID
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')

            # Run benchmark
            results = self._execute_benchmark(variant_name, dataset_path)

            # Save trace ID in report metadata
            results.metadata["otlp_trace_id"] = trace_id
            results.metadata["trace_url"] = f"http://jaeger:16686/trace/{trace_id}"

            # Write report with trace link
            self._write_report(variant_name, results)

            return results
```

## Observability Stack Integration

### Jaeger (Open Source)

```yaml
# docker-compose.yml

services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4317:4317"    # OTLP gRPC receiver
      - "4318:4318"    # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

Access Jaeger UI at http://localhost:16686

**Query Examples:**
- Service: `amber.ragas`
- Operation: `ragas.benchmark_run`
- Tags: `variant=graph_hybrid`, `dataset=carbonio_qa`

### Grafana Tempo (Cloud Native)

```yaml
# docker-compose.yml

services:
  tempo:
    image: grafana/tempo:latest
    command: ["-config.file=/etc/tempo.yaml"]
    volumes:
      - ./tempo-config.yaml:/etc/tempo.yaml
    ports:
      - "4317:4317"    # OTLP gRPC
      - "3200:3200"    # Tempo HTTP
```

**Tempo Configuration:**

```yaml
# tempo-config.yaml

server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        grpc:
          endpoint: 0.0.0.0:4317

storage:
  trace:
    backend: local
    local:
      path: /tmp/tempo/traces
```

Link Tempo as data source in Grafana.

## Configuration

Add to `evals/ragas/config.yaml`:

```yaml
ragas:
  observability:
    opentelemetry:
      enabled: false
      endpoint: "http://localhost:4317"  # OTLP endpoint
      service_name: "amber-ragas"
      sample_rate: 1.0  # 100% of benchmark runs
      # Attributes to include in all spans
      attributes:
        environment: "benchmarking"
        version: "2.0.0"
        team: "ml-platform"
```

## Benchmark Trace Example

Once implemented, a benchmark run trace would show:

```
amber.ragas.benchmark_run (variant: graph_hybrid, total: 180s)
├─ ragas.dataset_load (2s)
│  └─ samples: 50
├─ ragas.variant_evaluation (175s)
│  ├─ ragas.sample_evaluation (sample_id: 1, 3.5s) ← Parallel
│  │  ├─ ragas.metric.context_precision (0.9s)
│  │  │  └─ llm_api_call (800ms, tokens: 1200)
│  │  ├─ ragas.metric.faithfulness (1.2s)
│  │  │  ├─ llm_api_call (600ms, tokens: 800)
│  │  │  └─ llm_api_call (500ms, tokens: 700)
│  │  ├─ ragas.metric.context_recall (0.7s)
│  │  └─ ragas.metric.answer_relevancy (0.7s)
│  ├─ ragas.sample_evaluation (sample_id: 2, 3.2s) ← Parallel
│  ├─ ... (48 more samples)
│  └─ ragas.aggregation (0.5s)
└─ ragas.report_generation (3s)
   ├─ write_json (0.5s)
   ├─ write_markdown (1.5s)
   └─ write_prometheus_metrics (1s)
```

## Benefits

1. **Performance optimization**: Identify slow samples or metrics
2. **Cost analysis**: Track LLM API usage per metric
3. **Parallelization tuning**: Optimize concurrent evaluation
4. **Debugging**: Trace failures to specific samples or LLM calls
5. **Benchmarking history**: Compare trace durations across benchmark runs

## Metrics vs Traces

**Prometheus Metrics (current):**
- Aggregated scores (context_precision: 0.712)
- Total evaluation count
- Last run timestamp

**OpenTelemetry Traces (future):**
- Per-sample timing breakdown
- LLM call latency distribution
- Parallelization efficiency
- Error attribution

**Complementary use**: Metrics for dashboards, traces for debugging

## Implementation Timeline

**Phase 1** (Current): Prometheus metrics + Grafana dashboards ✅
**Phase 2** (Future): Add OpenTelemetry tracing for benchmark runs
**Phase 3** (Advanced): Correlate traces with regression detection

## Usage Example

```bash
# Enable OTLP tracing for RAGAS benchmarks
export RAGAS_ENABLE_OTLP=1
export RAGAS_OTLP_ENDPOINT=http://localhost:4317

# Run benchmark with tracing
uv run python evals/ragas/run_ragas_benchmarks.py

# View trace in Jaeger
# Navigate to http://localhost:16686
# Search for service: "amber.ragas"
# Filter by tag: variant=graph_hybrid

# Analyze slow samples
# Click on benchmark_run trace
# Sort child spans by duration
# Identify bottleneck metrics or samples
```

## Span Attributes Reference

**Root span (ragas.benchmark_run):**
- `variant` - Variant name (e.g., "graph_hybrid")
- `dataset` - Dataset path
- `ragas_version` - RAGAS library version
- `sample_count` - Number of samples evaluated
- `otlp_trace_id` - Trace ID for correlation

**Sample span (ragas.sample_evaluation):**
- `sample_id` - Unique sample identifier
- `question` - User query (truncated)
- `avg_score` - Average metric score for sample

**Metric span (ragas.metric.<name>):**
- `metric` - Metric name (e.g., "faithfulness")
- `score` - Computed score (0-1)

**LLM span (llm_api_call):**
- `provider` - LLM provider (e.g., "openai")
- `model` - Model name (e.g., "gpt-4")
- `metric` - Parent metric name
- `prompt_tokens` - Input token count
- `completion_tokens` - Output token count
- `total_tokens` - Total tokens
- `latency_ms` - API call latency

## Resources

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [OTLP Specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/otlp.md)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Docs](https://grafana.com/docs/tempo/)
- [RAGAS Documentation](https://docs.ragas.io/)

## Questions?

For implementation details, see:
- RAGAS runner: `evals/ragas/run_ragas_benchmarks.py`
- Prometheus exporter: `evals/ragas/prometheus_exporter.py`
- Grafana dashboard: `evals/ragas/grafana_dashboard.json`
- TruLens OpenTelemetry notes: `evals/trulens/observability/opentelemetry_notes.md`
