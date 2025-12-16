"""
TruLens metrics API router.

Provides endpoints for:
- Prometheus metrics exposition
- JSON stats for internal monitoring
- TruLens dashboard integration
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trulens/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Get TruLens metrics in Prometheus exposition format.

    This endpoint is scraped by Prometheus for monitoring.

    Returns:
        Plain text in Prometheus format with metrics like:
        - trulens_answer_relevance_score
        - trulens_groundedness_score
        - trulens_context_relevance_score
        - trulens_query_latency_seconds
        - trulens_queries_total
        - trulens_errors_total
    """
    try:
        from evals.trulens.metrics_exporter import TruLensMetricsExporter

        exporter = TruLensMetricsExporter()
        metrics_text = exporter.get_prometheus_metrics()

        return metrics_text

    except ImportError:
        logger.warning("TruLens not installed; metrics unavailable")
        raise HTTPException(
            status_code=503,
            detail="TruLens monitoring not installed. Install with: uv pip install -r evals/trulens/requirements-trulens.txt"
        )
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate metrics: {str(e)}"
        )


@router.get("/trulens/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get TruLens statistics in JSON format.

    Returns aggregated statistics including:
    - Total queries monitored
    - Average feedback scores
    - Latency percentiles
    - Error rates

    Returns:
        dict: Statistics in JSON format
    """
    try:
        from evals.trulens.metrics_exporter import TruLensMetricsExporter

        exporter = TruLensMetricsExporter()
        stats = exporter.get_json_stats()

        return stats

    except ImportError:
        logger.warning("TruLens not installed; stats unavailable")
        raise HTTPException(
            status_code=503,
            detail="TruLens monitoring not installed"
        )
    except Exception as e:
        logger.error(f"Error generating stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate stats: {str(e)}"
        )


@router.get("/trulens/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for TruLens monitoring system.

    Returns:
        dict: Health status including database connectivity
    """
    try:
        from evals.trulens.trulens_wrapper import get_monitor

        monitor = get_monitor()

        if monitor is None or not monitor.enabled:
            return {
                "status": "disabled",
                "monitoring_enabled": False,
                "message": "TruLens monitoring is not enabled"
            }

        # Test database connection
        if monitor.session is not None:
            # Try a simple query to verify database is accessible
            try:
                # TruLens database check via session
                monitor.session.get_records_and_feedback(limit=1)
                db_status = "healthy"
            except Exception as e:
                db_status = f"unhealthy: {str(e)}"
        else:
            db_status = "not initialized"

        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "monitoring_enabled": True,
            "database_status": db_status,
            "sampling_rate": monitor.sampling_rate,
        }

    except ImportError:
        return {
            "status": "unavailable",
            "monitoring_enabled": False,
            "message": "TruLens not installed"
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}", exc_info=True)
        # Log directory contents to check if file update persisted
        return {
            "status": "error",
            "monitoring_enabled": False,
            "error": str(e)
        }
