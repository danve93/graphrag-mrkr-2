"""
TruLens instrumentation wrapper for GraphRAG pipeline.

Non-invasive wrapper that decorates GraphRAG.query() method to enable
continuous monitoring with TruLens feedback functions.

Updated for TruLens v2.5+ API - uses TruBasicApp for proper session connection.
"""

import logging
import random
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TruLensMonitor:
    """
    Non-invasive TruLens wrapper for GraphRAG pipeline monitoring.

    Design goals:
    - Self-contained and removable (lives in evals/trulens)
    - Works with PostgreSQL (primary) or SQLite (fallback)
    - No modifications to core RAG pipeline
    - Opt-in via environment variable or config flag

    Usage:
        monitor = TruLensMonitor(
            enabled=True,
            database_url="postgresql://user:pass@localhost:5432/trulens",
            sampling_rate=1.0,
            feedback_functions=[...]
        )
        monitor.instrument_graph_rag(graph_rag_instance)
    """

    def __init__(
        self,
        enabled: bool = False,
        database_url: str = "sqlite:///evals/trulens/trulens.db",
        sampling_rate: float = 1.0,
        feedback_functions: Optional[List] = None,
    ):
        """
        Initialize TruLens monitor.

        Args:
            enabled: Enable/disable monitoring
            database_url: SQLAlchemy database URL
                          e.g., "postgresql://user:pass@host:port/db"
                          or "sqlite:///path/to/trulens.db"
            sampling_rate: Fraction of requests to monitor (0.0 to 1.0)
            feedback_functions: List of TruLens Feedback objects
        """
        self.enabled = enabled
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))
        self.session = None
        self.feedback_functions = feedback_functions or []
        self.database_url = database_url

        if not enabled:
            logger.info("TruLens monitoring disabled")
            return

        try:
            from trulens.core import TruSession

            # Initialize TruLens session with specified database
            # TruSession is a singleton - this sets the global session for all TruBasicApp instances
            self.session = TruSession(database_url=database_url)

            logger.info(
                f"TruLens monitoring initialized "
                f"(database={database_url}, sampling_rate={sampling_rate})"
            )

        except ImportError as e:
            logger.error(
                f"TruLens not installed: {e}. "
                "Install with: pip install trulens trulens-providers-openai"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize TruLens: {e}")
            self.enabled = False

    def _should_monitor_request(self) -> bool:
        """
        Determine if this request should be monitored based on sampling rate.

        Returns:
            bool: True if request should be monitored
        """
        if not self.enabled or self.sampling_rate <= 0.0:
            return False
        if self.sampling_rate >= 1.0:
            return True
        return random.random() < self.sampling_rate

    def instrument_graph_rag(self, graph_rag_instance):
        """
        Instrument the GraphRAG.query() method with TruLens recording.

        Wraps the existing query method without modifying it. Records all
        query executions to the TruLens database for analysis.

        Args:
            graph_rag_instance: Instance of GraphRAG to instrument

        Returns:
            Instrumented GraphRAG instance (same object, modified in-place)
        """
        if not self.enabled:
            logger.debug("TruLens monitoring disabled; skipping instrumentation")
            return graph_rag_instance

        if self.session is None:
            logger.error("TruLens session not initialized; skipping instrumentation")
            return graph_rag_instance

        try:
            # Import TruBasicApp which properly connects to TruSession singleton
            from trulens.apps.basic import TruBasicApp

            # Save reference to original query method and monitor instance
            original_query = graph_rag_instance.query
            monitor = self

            @wraps(original_query)
            def wrapped_query(*args, **kwargs):
                """Wrapped query method with TruLens monitoring."""

                # Check if this request should be monitored (sampling)
                if not monitor._should_monitor_request():
                    # Skip monitoring, call original method
                    return original_query(*args, **kwargs)

                from core.otel_config import get_tracer
                tracer = get_tracer("amber.trulens")
                
                with tracer.start_as_current_span("trulens.monitor") as span:
                    # Extract query for logging
                    user_query = args[0] if args else kwargs.get("user_query", "")

                    # Extract metadata for TruLens record
                    retrieval_mode = kwargs.get("retrieval_mode", "hybrid")
                    session_id = kwargs.get("session_id", "unknown")
                    user_type = kwargs.get("user_type", "anonymous")

                    # Store kwargs for closure
                    query_kwargs = kwargs.copy()

                    try:
                        # Cache for storing the full result dict
                        cached_result = None
                        
                        # Remove user_query from kwargs if present to avoid duplicate argument
                        inner_kwargs = {k: v for k, v in query_kwargs.items() if k != "user_query"}

                        # Create a callable function for TruBasicApp
                        # TruBasicApp expects a function that takes a string and returns a string
                        def graphrag_query_fn(query_text: str) -> str:
                            """Execute GraphRAG query and return response string."""
                            nonlocal cached_result
                            cached_result = original_query(query_text, **inner_kwargs)
                            # TruBasicApp expects string output, extract the response
                            if isinstance(cached_result, dict):
                                return cached_result.get("response", str(cached_result))
                            return str(cached_result)

                        # Create TruBasicApp recorder with user_type in app name
                        # This separates external, admin, and user queries in the dashboard
                        app_id = f"amber_{user_type}_{retrieval_mode}"
                        
                        # Build metadata for debugging
                        record_metadata = {
                            "session_id": session_id,
                            "user_type": user_type,
                            "retrieval_mode": retrieval_mode,
                            "top_k": kwargs.get("top_k", 5),
                            "temperature": kwargs.get("temperature", 0.7),
                            "use_multi_hop": kwargs.get("use_multi_hop", False),
                            "llm_model": kwargs.get("llm_model", "default"),
                            "embedding_model": kwargs.get("embedding_model", "default"),
                        }

                        # Inject OpenTelemetry Trace ID
                        try:
                            if span.get_span_context().is_valid:
                                trace_id = f"{span.get_span_context().trace_id:032x}"
                                record_metadata["trace_id"] = trace_id
                                record_metadata["trace_link"] = (
                                    f"http://localhost:3200/explore?orgId=1&left="
                                    f"%5B%22now-1h%22,%22now%22,%22Tempo%22,%7B%22query%22:%22{trace_id}%22%7D%5D"
                                )
                        except Exception:
                            pass
                        
                        tru_recorder = TruBasicApp(
                            graphrag_query_fn,
                            app_name=app_id,
                            app_version="1.0",
                            feedbacks=monitor.feedback_functions,
                            metadata=record_metadata,  # Add metadata for debugging
                        )

                        # Execute query with TruLens recording
                        with tru_recorder as recording:
                            response_str = tru_recorder.app(user_query)

                        logger.debug(
                            f"TruLens recorded query: {user_query[:50]}... "
                            f"(mode={retrieval_mode}, session={session_id})"
                        )

                        # Return the cached full result dict from the query
                        return cached_result

                    except Exception as e:
                        # Log error but don't fail the query
                        logger.error(f"TruLens recording failed: {e}", exc_info=True)
                        # Fall back to original method
                        return original_query(*args, **kwargs)

            # Replace the query method with wrapped version
            graph_rag_instance.query = wrapped_query

            logger.info(
                f"GraphRAG.query() instrumented with TruLens "
                f"(sampling_rate={self.sampling_rate})"
            )

        except ImportError as e:
            logger.error(f"Failed to import TruLens modules: {e}")
        except Exception as e:
            logger.error(f"Failed to instrument GraphRAG: {e}", exc_info=True)

        return graph_rag_instance

    def stop(self):
        """Stop TruLens monitoring and close database connections."""
        if self.session is not None:
            try:
                logger.info("TruLens monitoring stopped")
            except Exception as e:
                logger.warning(f"Error stopping TruLens: {e}")


# Module-level singleton instance (set by initializer)
_monitor_instance: Optional[TruLensMonitor] = None


def get_monitor() -> Optional[TruLensMonitor]:
    """Get the global TruLens monitor instance."""
    return _monitor_instance


def set_monitor(monitor: TruLensMonitor):
    """Set the global TruLens monitor instance."""
    global _monitor_instance
    _monitor_instance = monitor
