"""
OpenTelemetry Configuration Module.

Handles initialization of OpenTelemetry tracing with OTLP exporter,
configuring the TracerProvider, and managing lifecycle events.
"""

import logging
import os
from typing import Optional

# Import OpenTelemetry components safely to avoid crashes if dependencies are missing
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPHTTPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

logger = logging.getLogger(__name__)


def initialize_opentelemetry() -> bool:
    """
    Initialize OpenTelemetry tracing.

    Reads configuration from environment variables:
    - OTLP_ENDPOINT: Target endpoint for traces (default: http://localhost:4317)
    - OTLP_SERVICE_NAME: Service name for traces (default: amber-graphrag)
    - OTLP_SAMPLING_RATE: Sampling rate 0.0-1.0 (currently not applied at provider level)

    Returns:
        bool: True if initialization was successful
    """
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry packages not found. Skipping initialization.")
        return False

    try:
        endpoint = os.getenv("OTLP_ENDPOINT", "http://localhost:4317")
        service_name = os.getenv("OTLP_SERVICE_NAME", "amber-graphrag")
        
        # Create resource configuration
        resource = Resource.create(attributes={
            ResourceAttributes.SERVICE_NAME: service_name,
            "deployment.environment": os.getenv("NODE_ENV", "development"),
        })

        # Initialize TracerProvider
        provider = TracerProvider(resource=resource)
        
        # Configure OTLP Exporter
        # Check if endpoint implies HTTP or gRPC
        if "4318" in endpoint or endpoint.endswith("/v1/traces"):
            # Assume HTTP if port 4318 or standard HTTP path
            exporter = OTLPHTTPSpanExporter(endpoint=endpoint)
            logger.info(f"Initialized OTLP HTTP exporter pointing to {endpoint}")
        else:
            # Default to gRPC
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            logger.info(f"Initialized OTLP gRPC exporter pointing to {endpoint}")

        # Add Batch Span Processor
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)

        # Set global TracerProvider
        trace.set_tracer_provider(provider)
        
        logger.info(f"OpenTelemetry initialized for service: {service_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}", exc_info=True)
        return False


def get_tracer(name: str):
    """
    Get a tracer instance.
    
    If OpenTelemetry is not installed or configured, returns a proxy/no-op tracer.
    
    Args:
        name: Name of the component requesting the tracer (usually __name__)
    
    Returns:
        Tracer instance
    """
    if OTEL_AVAILABLE:
        return trace.get_tracer(name)
    
    # Return dummy tracer if OTEL not available
    class NoOpTracer:
        def start_as_current_span(self, name, **kwargs):
            return NoOpSpan()
            
    class NoOpSpan:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def set_attribute(self, *args): pass
        def add_event(self, *args): pass
        def set_status(self, *args): pass
        
    return NoOpTracer()
