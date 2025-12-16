"""
Retrieval quality monitoring system for continuous quality tracking and alerting.

This module provides real-time monitoring of retrieval quality metrics:
- Quality scores (answer quality from quality_scorer)
- Retrieval latency
- Cache hit rates
- Query type distribution

Features:
- Sliding window metrics aggregation
- Automatic alerting on quality degradation
- Percentile-based anomaly detection
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""

    timestamp: float
    query: str
    query_type: str  # "chunk", "entity", "hybrid"
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    num_chunks_retrieved: int
    quality_score: Optional[float] = None  # 0-100 if available
    quality_breakdown: Optional[Dict[str, float]] = None
    quality_confidence: Optional[str] = None  # "high", "medium", "low"
    cache_hit: bool = False
    error: Optional[str] = None


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""

    window_size: int
    time_range_start: float
    time_range_end: float

    # Quality metrics
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    p50_quality_score: float
    p95_quality_score: float
    quality_score_stddev: float

    # Latency metrics
    avg_total_latency_ms: float
    p50_total_latency_ms: float
    p95_total_latency_ms: float
    p99_total_latency_ms: float

    avg_retrieval_latency_ms: float
    avg_generation_latency_ms: float

    # Cache metrics
    cache_hit_rate: float

    # Query type distribution
    query_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Error metrics
    error_rate: float = 0.0
    total_queries: int = 0


@dataclass
class QualityAlert:
    """Alert triggered by quality degradation or anomaly."""

    timestamp: float
    alert_type: str  # "quality_drop", "latency_spike", "error_spike"
    severity: str  # "warning", "critical"
    message: str
    current_value: float
    baseline_value: float
    threshold: float


class RetrievalQualityMonitor:
    """
    Monitors retrieval quality metrics with alerting capabilities.

    Tracks metrics in a sliding window and detects:
    - Quality score drops (>30% below baseline)
    - Latency spikes (>2x p95)
    - Error rate increases
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        alert_threshold: Optional[float] = None,
    ):
        """
        Initialize the quality monitor.

        Args:
            window_size: Number of queries to track (default from settings)
            alert_threshold: Quality drop threshold 0-1 (default from settings)
        """
        self.enabled = getattr(settings, "enable_quality_monitoring", True)
        self.window_size = window_size or getattr(
            settings, "quality_monitor_window_size", 1000
        )
        self.alert_threshold = alert_threshold or getattr(
            settings, "quality_alert_threshold", 0.7
        )

        # Sliding window of query metrics
        self.metrics_window: Deque[QueryMetrics] = deque(maxlen=self.window_size)

        # Alert history
        self.alerts: List[QualityAlert] = []
        self.max_alerts = 100  # Keep last 100 alerts

        # Baseline metrics (calculated from first N queries)
        self.baseline_quality_score: Optional[float] = None
        self.baseline_latency_p95: Optional[float] = None
        self.baseline_window_size = 100  # Use first 100 queries for baseline

        logger.info(
            f"Quality monitor initialized: window_size={self.window_size}, "
            f"alert_threshold={self.alert_threshold}"
        )

    def record_query(
        self,
        query: str,
        query_type: str,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        num_chunks_retrieved: int,
        quality_score: Optional[float] = None,
        quality_breakdown: Optional[Dict[str, float]] = None,
        quality_confidence: Optional[str] = None,
        cache_hit: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Record metrics for a single query.

        Args:
            query: The user query
            query_type: Type of retrieval ("chunk", "entity", "hybrid")
            retrieval_latency_ms: Time spent retrieving chunks
            generation_latency_ms: Time spent generating answer
            num_chunks_retrieved: Number of chunks retrieved
            quality_score: Answer quality score 0-100 (if available)
            quality_breakdown: Quality component scores
            quality_confidence: Quality confidence level
            cache_hit: Whether cache was hit
            error: Error message if query failed
        """
        if not self.enabled:
            return

        metrics = QueryMetrics(
            timestamp=time.time(),
            query=query,
            query_type=query_type,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=retrieval_latency_ms + generation_latency_ms,
            num_chunks_retrieved=num_chunks_retrieved,
            quality_score=quality_score,
            quality_breakdown=quality_breakdown,
            quality_confidence=quality_confidence,
            cache_hit=cache_hit,
            error=error,
        )

        self.metrics_window.append(metrics)

        # Update baseline if we have enough data
        if (
            self.baseline_quality_score is None
            and len(self.metrics_window) >= self.baseline_window_size
        ):
            self._calculate_baseline()

        # Check for alerts if baseline is established
        if self.baseline_quality_score is not None:
            self._check_for_alerts(metrics)

    def _calculate_baseline(self) -> None:
        """Calculate baseline metrics from initial queries."""
        if len(self.metrics_window) < self.baseline_window_size:
            return

        # Use first baseline_window_size queries
        baseline_queries = list(self.metrics_window)[:self.baseline_window_size]

        # Calculate baseline quality score
        quality_scores = [
            m.quality_score for m in baseline_queries
            if m.quality_score is not None
        ]
        if quality_scores:
            self.baseline_quality_score = sum(quality_scores) / len(quality_scores)

        # Calculate baseline latency p95
        latencies = [m.total_latency_ms for m in baseline_queries]
        latencies_sorted = sorted(latencies)
        p95_index = int(len(latencies_sorted) * 0.95)
        self.baseline_latency_p95 = latencies_sorted[p95_index]

        logger.info(
            f"Baseline established: quality={self.baseline_quality_score:.1f}, "
            f"latency_p95={self.baseline_latency_p95:.1f}ms"
        )

    def _check_for_alerts(self, metrics: QueryMetrics) -> None:
        """
        Check if current metrics trigger any alerts.

        Args:
            metrics: Current query metrics
        """
        # Alert on quality drop
        if (
            metrics.quality_score is not None
            and self.baseline_quality_score is not None
        ):
            quality_ratio = metrics.quality_score / self.baseline_quality_score

            if quality_ratio < self.alert_threshold:
                severity = "critical" if quality_ratio < 0.5 else "warning"
                alert = QualityAlert(
                    timestamp=time.time(),
                    alert_type="quality_drop",
                    severity=severity,
                    message=f"Quality score dropped to {metrics.quality_score:.1f} "
                            f"({quality_ratio*100:.0f}% of baseline {self.baseline_quality_score:.1f})",
                    current_value=metrics.quality_score,
                    baseline_value=self.baseline_quality_score,
                    threshold=self.alert_threshold,
                )
                self._add_alert(alert)

        # Alert on latency spike
        if self.baseline_latency_p95 is not None:
            latency_ratio = metrics.total_latency_ms / self.baseline_latency_p95

            if latency_ratio > 2.0:
                severity = "critical" if latency_ratio > 5.0 else "warning"
                alert = QualityAlert(
                    timestamp=time.time(),
                    alert_type="latency_spike",
                    severity=severity,
                    message=f"Latency spiked to {metrics.total_latency_ms:.1f}ms "
                            f"({latency_ratio:.1f}x baseline p95 {self.baseline_latency_p95:.1f}ms)",
                    current_value=metrics.total_latency_ms,
                    baseline_value=self.baseline_latency_p95,
                    threshold=2.0,
                )
                self._add_alert(alert)

        # Alert on error
        if metrics.error is not None:
            alert = QualityAlert(
                timestamp=time.time(),
                alert_type="error",
                severity="warning",
                message=f"Query failed: {metrics.error}",
                current_value=1.0,
                baseline_value=0.0,
                threshold=0.0,
            )
            self._add_alert(alert)

    def _add_alert(self, alert: QualityAlert) -> None:
        """
        Add an alert to the history.

        Args:
            alert: Alert to add
        """
        self.alerts.append(alert)

        # Keep only last max_alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]

        # Log the alert
        log_func = logger.critical if alert.severity == "critical" else logger.warning
        log_func(f"ALERT [{alert.alert_type}]: {alert.message}")

    def get_aggregated_metrics(
        self, window: Optional[int] = None
    ) -> Optional[AggregatedMetrics]:
        """
        Get aggregated metrics over a time window.

        Args:
            window: Number of recent queries to aggregate (default: all in window)

        Returns:
            Aggregated metrics or None if insufficient data
        """
        if not self.enabled or len(self.metrics_window) == 0:
            return None

        # Get queries to aggregate
        queries = list(self.metrics_window)
        if window is not None and window < len(queries):
            queries = queries[-window:]

        if not queries:
            return None

        # Calculate quality metrics
        quality_scores = [
            m.quality_score for m in queries if m.quality_score is not None
        ]

        if quality_scores:
            quality_scores_sorted = sorted(quality_scores)
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            p50_quality = quality_scores_sorted[len(quality_scores_sorted) // 2]
            p95_quality = quality_scores_sorted[int(len(quality_scores_sorted) * 0.95)]

            # Standard deviation
            quality_variance = sum(
                (s - avg_quality) ** 2 for s in quality_scores
            ) / len(quality_scores)
            quality_stddev = quality_variance ** 0.5
        else:
            avg_quality = min_quality = max_quality = 0.0
            p50_quality = p95_quality = quality_stddev = 0.0

        # Calculate latency metrics
        total_latencies = [m.total_latency_ms for m in queries]
        total_latencies_sorted = sorted(total_latencies)

        avg_total_latency = sum(total_latencies) / len(total_latencies)
        p50_total_latency = total_latencies_sorted[len(total_latencies_sorted) // 2]
        p95_total_latency = total_latencies_sorted[int(len(total_latencies_sorted) * 0.95)]
        p99_total_latency = total_latencies_sorted[int(len(total_latencies_sorted) * 0.99)]

        retrieval_latencies = [m.retrieval_latency_ms for m in queries]
        generation_latencies = [m.generation_latency_ms for m in queries]
        avg_retrieval_latency = sum(retrieval_latencies) / len(retrieval_latencies)
        avg_generation_latency = sum(generation_latencies) / len(generation_latencies)

        # Calculate cache metrics
        cache_hits = sum(1 for m in queries if m.cache_hit)
        cache_hit_rate = cache_hits / len(queries)

        # Calculate query type distribution
        query_types: Dict[str, int] = {}
        for m in queries:
            query_types[m.query_type] = query_types.get(m.query_type, 0) + 1

        # Calculate error rate
        errors = sum(1 for m in queries if m.error is not None)
        error_rate = errors / len(queries)

        return AggregatedMetrics(
            window_size=len(queries),
            time_range_start=queries[0].timestamp,
            time_range_end=queries[-1].timestamp,
            avg_quality_score=avg_quality,
            min_quality_score=min_quality,
            max_quality_score=max_quality,
            p50_quality_score=p50_quality,
            p95_quality_score=p95_quality,
            quality_score_stddev=quality_stddev,
            avg_total_latency_ms=avg_total_latency,
            p50_total_latency_ms=p50_total_latency,
            p95_total_latency_ms=p95_total_latency,
            p99_total_latency_ms=p99_total_latency,
            avg_retrieval_latency_ms=avg_retrieval_latency,
            avg_generation_latency_ms=avg_generation_latency,
            cache_hit_rate=cache_hit_rate,
            query_type_distribution=query_types,
            error_rate=error_rate,
            total_queries=len(queries),
        )

    def get_recent_alerts(self, limit: int = 10) -> List[QualityAlert]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts (most recent first)
        """
        return list(reversed(self.alerts[-limit:]))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current monitoring state.

        Returns:
            Dictionary with monitoring summary
        """
        if not self.enabled:
            return {"enabled": False}

        metrics = self.get_aggregated_metrics()
        recent_alerts = self.get_recent_alerts(limit=5)

        summary = {
            "enabled": True,
            "window_size": self.window_size,
            "queries_tracked": len(self.metrics_window),
            "baseline": {
                "quality_score": self.baseline_quality_score,
                "latency_p95_ms": self.baseline_latency_p95,
            },
        }

        if metrics:
            summary["current_metrics"] = {
                "avg_quality_score": round(metrics.avg_quality_score, 1),
                "p95_quality_score": round(metrics.p95_quality_score, 1),
                "avg_total_latency_ms": round(metrics.avg_total_latency_ms, 1),
                "p95_total_latency_ms": round(metrics.p95_total_latency_ms, 1),
                "cache_hit_rate": round(metrics.cache_hit_rate * 100, 1),
                "error_rate": round(metrics.error_rate * 100, 2),
            }

        summary["recent_alerts"] = [
            {
                "timestamp": datetime.fromtimestamp(
                    alert.timestamp, tz=timezone.utc
                ).isoformat(),
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
            }
            for alert in recent_alerts
        ]

        return summary

    def reset(self) -> None:
        """Reset the monitor (clear all metrics and alerts)."""
        self.metrics_window.clear()
        self.alerts.clear()
        self.baseline_quality_score = None
        self.baseline_latency_p95 = None
        logger.info("Quality monitor reset")


# Global instance
quality_monitor = RetrievalQualityMonitor()
