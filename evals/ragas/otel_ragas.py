
import os
import contextlib
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure core module is importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.propagate import inject
    from core.otel_config import get_tracer, initialize_opentelemetry
except ImportError:
    # Dummy implementation if deps missing
    trace = None
    def get_tracer(name): return None
    def initialize_opentelemetry(): pass
    def inject(carrier): pass

@contextlib.contextmanager
def instrument_ragas_benchmark(
    run_id: str,
    variant: str,
    num_samples: int,
    config: Dict[str, Any]
):
    """
    Context manager to trace a RAGAS benchmark run.
     Initializes OTEL if ENABLE_OPENTELEMETRY is set.
    """
    # Initialize implementation if needed
    if os.getenv("ENABLE_OPENTELEMETRY") == "1":
        initialize_opentelemetry()
        
    tracer = get_tracer("amber.ragas")
    if not tracer:
        yield None
        return

    attributes = {
        "ragas.run_id": run_id,
        "ragas.variant": variant,
        "ragas.num_samples": num_samples,
    }

    with tracer.start_as_current_span("ragas.benchmark_run", attributes=attributes) as span:
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

def inject_trace_context(headers: Dict[str, str]):
    """Inject current trace context into HTTP headers."""
    if trace:
        inject(headers)

def get_trace_id(span) -> Optional[str]:
    """Get formatted trace ID from span."""
    if span and span.get_span_context().is_valid:
        return f"{span.get_span_context().trace_id:032x}"
    return None
