# OpenTelemetry Integration for TruLens (Future Enhancement)

Notes and guidance for integrating OpenTelemetry (OTLP) distributed tracing with TruLens monitoring.

## Overview

OpenTelemetry provides distributed tracing capabilities that can complement TruLens feedback evaluation:

- **TruLens**: Evaluates quality (relevance, groundedness, etc.)
- **OpenTelemetry**: Tracks execution flow, performance, dependencies

## Why OpenTelemetry?

Current TruLens implementation tracks:
- Query input/output
- Feedback scores
- Latency (total)

OpenTelemetry would add:
- **Distributed tracing**: Track query → retrieval → generation → response
- **Span timing**: See exactly which step is slow
- **Context propagation**: Follow requests across services
- **Correlation**: Link traces to TruLens feedback records

## Architecture

```
User Query
  ↓
[SPAN: api.chat.query] ← Root span
  ↓
  ├─ [SPAN: graph_rag.query_analysis] ← Query understanding
  ├─ [SPAN: graph_rag.retrieval] ← Document retrieval
  │    ├─ [SPAN: vector_search]
  │    ├─ [SPAN: graph_expansion]
  │    └─ [SPAN: reranking]
  ├─ [SPAN: graph_rag.generation] ← LLM response generation
  │    └─ [SPAN: llm_api_call]
  └─ [SPAN: trulens.feedback_evaluation] ← Async feedback
       ├─ [SPAN: answer_relevance]
       ├─ [SPAN: groundedness]
       └─ [SPAN: context_relevance]
```

## Implementation Approach

### 1. Install OpenTelemetry SDK

Add to `requirements-trulens.txt`:

```txt
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
opentelemetry-instrumentation-fastapi>=0.41b0
```

### 2. Initialize OTLP Exporter

```python
# evals/trulens/observability/otlp_config.py

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def initialize_otlp(endpoint: str = "http://localhost:4317"):
    """Initialize OpenTelemetry tracing."""
    # Set up tracer provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)

    # Add span processor
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    return trace.get_tracer("amber.graphrag")
```

### 3. Instrument GraphRAG Pipeline

```python
# Modify evals/trulens/trulens_wrapper.py

from opentelemetry import trace

class TruLensMonitor:
    def __init__(self, ..., enable_otlp: bool = False):
        self.enable_otlp = enable_otlp
        self.tracer = None

        if enable_otlp:
            from evals.trulens.observability.otlp_config import initialize_otlp
            self.tracer = initialize_otlp()

    def instrument_graph_rag(self, graph_rag_instance):
        original_query = graph_rag_instance.query

        @wraps(original_query)
        def wrapped_query(*args, **kwargs):
            user_query = args[0] if args else kwargs.get("user_query", "")

            # Create root span
            if self.tracer:
                with self.tracer.start_as_current_span(
                    "graph_rag.query",
                    attributes={
                        "query": user_query[:100],  # Truncate for privacy
                        "retrieval_mode": kwargs.get("retrieval_mode"),
                    }
                ) as span:
                    # Execute with tracing
                    result = self._execute_with_tracing(
                        original_query, span, *args, **kwargs
                    )
                    return result
            else:
                # No tracing, execute normally
                return original_query(*args, **kwargs)
```

### 4. Add Span Instrumentation to RAG Nodes

```python
# Example: Instrument retrieval node

from opentelemetry import trace

tracer = trace.get_tracer("amber.graphrag")

def retrieve_documents_async(...):
    with tracer.start_as_current_span("retrieval.retrieve_documents") as span:
        # Set attributes
        span.set_attribute("top_k", top_k)
        span.set_attribute("retrieval_mode", retrieval_mode)

        # Execute retrieval
        chunks = await _do_retrieval(...)

        # Add result attributes
        span.set_attribute("chunks_retrieved", len(chunks))

        return chunks
```

### 5. Correlate with TruLens Records

```python
# Link OTLP trace ID to TruLens record

from opentelemetry import trace

def wrapped_query(*args, **kwargs):
    # Get current span context
    span_context = trace.get_current_span().get_span_context()
    trace_id = span_context.trace_id

    # Execute TruLens recording
    with tru_app as recording:
        result = app_instance.query(user_query, **kwargs)

        # Add trace ID to TruLens metadata
        recording.record_metadata({
            "otlp_trace_id": trace_id,  # Link to OTLP trace
            "retrieval_mode": retrieval_mode,
            ...
        })

    return result
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
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

Access Jaeger UI at http://localhost:16686

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

Link Tempo as data source in Grafana.

## Configuration

Add to `evals/trulens/config.example.yaml`:

```yaml
observability:
  opentelemetry:
    enabled: false
    endpoint: "http://localhost:4317"  # OTLP endpoint
    service_name: "amber-graphrag"
    sample_rate: 1.0  # 100% of traces
    # Attributes to include
    attributes:
      environment: "production"
      version: "2.0.0"
```

## Query Trace Example

Once implemented, a query trace would show:

```
amber.graphrag.query (total: 3.2s)
├─ query_analysis (150ms)
├─ retrieval (1.8s)
│  ├─ vector_search (800ms)
│  ├─ graph_expansion (600ms)
│  └─ reranking (400ms)
├─ generation (1.1s)
│  └─ llm_api_call (1.0s)
└─ feedback_evaluation (async, 2.5s)
   ├─ answer_relevance (800ms)
   ├─ groundedness (900ms)
   └─ context_relevance (800ms)
```

## Benefits

1. **Performance debugging**: Identify slow components
2. **Error tracking**: See exactly where failures occur
3. **Dependency mapping**: Understand service interactions
4. **Capacity planning**: Analyze traffic patterns
5. **Quality correlation**: Link slow traces to low quality scores

## Implementation Timeline

**Phase 1** (Current): TruLens + Prometheus metrics ✅
**Phase 2** (Future): Add OpenTelemetry tracing
**Phase 3** (Advanced): Custom span processors for ML-specific metrics

## Resources

- [OpenTelemetry Python Docs](https://opentelemetry.io/docs/instrumentation/python/)
- [OTLP Specification](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/protocol/otlp.md)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Docs](https://grafana.com/docs/tempo/)

## Questions?

For implementation details, see:
- TruLens wrapper: `evals/trulens/trulens_wrapper.py`
- GraphRAG pipeline: `rag/graph_rag.py`
- Feedback functions: `evals/trulens/feedback_functions.py`
