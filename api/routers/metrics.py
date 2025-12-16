"""
API endpoints for retrieval quality monitoring and metrics.
"""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from rag.quality_monitor import quality_monitor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


class QualityMetricsSummary(BaseModel):
    """Summary of retrieval quality metrics."""
    enabled: bool = Field(..., description="Whether quality monitoring is enabled")
    window_size: int = Field(..., description="Maximum number of queries tracked")
    queries_tracked: int = Field(..., description="Number of queries currently tracked")
    baseline: Dict[str, Optional[float]] = Field(..., description="Baseline metrics for comparison")
    current_metrics: Optional[Dict[str, float]] = Field(None, description="Current aggregated metrics")
    recent_alerts: List[Dict[str, Any]] = Field(..., description="Recent quality alerts")


class Alert(BaseModel):
    """Quality alert information."""
    timestamp: str = Field(..., description="Alert timestamp (ISO 8601)")
    type: str = Field(..., description="Alert type (quality_drop, latency_spike, error)")
    severity: str = Field(..., description="Alert severity (warning, critical)")
    message: str = Field(..., description="Human-readable alert message")
    current_value: float = Field(..., description="Current metric value")
    baseline_value: float = Field(..., description="Baseline metric value")
    threshold: float = Field(..., description="Alert threshold")


class AggregatedMetricsResponse(BaseModel):
    """Detailed aggregated metrics over a time window."""
    window_size: int = Field(..., description="Number of queries in window")
    time_range_start: float = Field(..., description="Start timestamp (Unix)")
    time_range_end: float = Field(..., description="End timestamp (Unix)")

    # Quality metrics
    avg_quality_score: float = Field(..., description="Average quality score (0-100)")
    min_quality_score: float = Field(..., description="Minimum quality score")
    max_quality_score: float = Field(..., description="Maximum quality score")
    p50_quality_score: float = Field(..., description="50th percentile quality score")
    p95_quality_score: float = Field(..., description="95th percentile quality score")
    quality_score_stddev: float = Field(..., description="Quality score standard deviation")

    # Latency metrics
    avg_total_latency_ms: float = Field(..., description="Average total latency (ms)")
    p50_total_latency_ms: float = Field(..., description="50th percentile total latency")
    p95_total_latency_ms: float = Field(..., description="95th percentile total latency")
    p99_total_latency_ms: float = Field(..., description="99th percentile total latency")
    avg_retrieval_latency_ms: float = Field(..., description="Average retrieval latency")
    avg_generation_latency_ms: float = Field(..., description="Average generation latency")

    # Cache metrics
    cache_hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0)")

    # Query distribution
    query_type_distribution: Dict[str, int] = Field(..., description="Query type counts")

    # Error metrics
    error_rate: float = Field(..., description="Error rate (0.0-1.0)")
    total_queries: int = Field(..., description="Total queries in window")


@router.get("/metrics/retrieval/summary", response_model=QualityMetricsSummary)
async def get_retrieval_summary() -> QualityMetricsSummary:
    """
    Get a summary of retrieval quality monitoring status and metrics.

    Returns summary including:
    - Monitoring status (enabled/disabled)
    - Current tracked queries count
    - Baseline metrics
    - Recent aggregated metrics
    - Recent alerts (last 5)

    Returns:
        Quality metrics summary
    """
    try:
        summary = quality_monitor.get_summary()
        return QualityMetricsSummary(**summary)
    except Exception as e:
        logger.error(f"Failed to get quality metrics summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quality metrics: {str(e)}"
        )


@router.get("/metrics/retrieval/detailed", response_model=AggregatedMetricsResponse)
async def get_detailed_metrics(
    window: Optional[int] = Query(
        default=None,
        description="Number of recent queries to aggregate (default: all tracked)"
    )
) -> AggregatedMetricsResponse:
    """
    Get detailed aggregated metrics over a time window.

    Args:
        window: Optional number of recent queries to aggregate (default: all)

    Returns:
        Detailed aggregated metrics including quality scores, latencies, cache hit rate
    """
    try:
        if not quality_monitor.enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quality monitoring is disabled"
            )

        metrics = quality_monitor.get_aggregated_metrics(window=window)

        if metrics is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Insufficient data for metrics calculation"
            )

        return AggregatedMetricsResponse(
            window_size=metrics.window_size,
            time_range_start=metrics.time_range_start,
            time_range_end=metrics.time_range_end,
            avg_quality_score=metrics.avg_quality_score,
            min_quality_score=metrics.min_quality_score,
            max_quality_score=metrics.max_quality_score,
            p50_quality_score=metrics.p50_quality_score,
            p95_quality_score=metrics.p95_quality_score,
            quality_score_stddev=metrics.quality_score_stddev,
            avg_total_latency_ms=metrics.avg_total_latency_ms,
            p50_total_latency_ms=metrics.p50_total_latency_ms,
            p95_total_latency_ms=metrics.p95_total_latency_ms,
            p99_total_latency_ms=metrics.p99_total_latency_ms,
            avg_retrieval_latency_ms=metrics.avg_retrieval_latency_ms,
            avg_generation_latency_ms=metrics.avg_generation_latency_ms,
            cache_hit_rate=metrics.cache_hit_rate,
            query_type_distribution=metrics.query_type_distribution,
            error_rate=metrics.error_rate,
            total_queries=metrics.total_queries,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detailed metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve detailed metrics: {str(e)}"
        )


@router.get("/metrics/retrieval/alerts")
async def get_alerts(
    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of recent alerts to return"
    )
) -> Dict[str, Any]:
    """
    Get recent quality alerts.

    Args:
        limit: Maximum number of alerts to return (1-100, default 10)

    Returns:
        List of recent alerts sorted by timestamp (most recent first)
    """
    try:
        if not quality_monitor.enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quality monitoring is disabled"
            )

        alerts = quality_monitor.get_recent_alerts(limit=limit)

        # Convert to response format
        alert_list = [
            {
                "timestamp": alert.timestamp,
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "current_value": alert.current_value,
                "baseline_value": alert.baseline_value,
                "threshold": alert.threshold,
            }
            for alert in alerts
        ]

        return {
            "total_alerts": len(alert_list),
            "alerts": alert_list
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )


@router.post("/metrics/retrieval/reset")
async def reset_monitoring() -> Dict[str, str]:
    """
    Reset the quality monitor (clear all metrics and alerts).

    This endpoint clears:
    - All tracked query metrics
    - All alerts
    - Baseline metrics

    Use with caution in production.

    Returns:
        Status message
    """
    try:
        quality_monitor.reset()
        return {"status": "success", "message": "Quality monitor reset successfully"}
    except Exception as e:
        logger.error(f"Failed to reset quality monitor: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset quality monitor: {str(e)}"
        )
