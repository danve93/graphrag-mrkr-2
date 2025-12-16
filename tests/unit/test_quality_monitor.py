"""
Unit tests for retrieval quality monitoring functionality.

Tests cover:
- Metrics recording and aggregation
- Baseline calculation
- Alert triggering (quality drops, latency spikes, errors)
- Configuration integration
- API endpoint functionality
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from rag.quality_monitor import RetrievalQualityMonitor, QueryMetrics, QualityAlert


class TestQueryMetricsRecording:
    """Tests for recording query metrics."""

    def test_record_single_query(self):
        """Test recording a single query's metrics."""
        monitor = RetrievalQualityMonitor(window_size=100)

        monitor.record_query(
            query="test query",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=85.0,
            quality_breakdown={"context_relevance": 90.0, "coherence": 80.0},
            quality_confidence="high",
            cache_hit=False,
        )

        assert len(monitor.metrics_window) == 1
        metrics = monitor.metrics_window[0]
        assert metrics.query == "test query"
        assert metrics.query_type == "chunk"
        assert metrics.retrieval_latency_ms == 50.0
        assert metrics.generation_latency_ms == 100.0
        assert metrics.total_latency_ms == 150.0
        assert metrics.num_chunks_retrieved == 5
        assert metrics.quality_score == 85.0
        assert metrics.cache_hit is False

    def test_record_cache_hit_query(self):
        """Test recording a cache hit query."""
        monitor = RetrievalQualityMonitor(window_size=100)

        monitor.record_query(
            query="cached query",
            query_type="hybrid",
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            num_chunks_retrieved=5,
            quality_score=90.0,
            cache_hit=True,
        )

        assert len(monitor.metrics_window) == 1
        metrics = monitor.metrics_window[0]
        assert metrics.cache_hit is True
        assert metrics.total_latency_ms == 0.0

    def test_sliding_window_behavior(self):
        """Test that metrics window maintains size limit."""
        monitor = RetrievalQualityMonitor(window_size=10)

        # Record 20 queries
        for i in range(20):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
            )

        # Window should only contain last 10
        assert len(monitor.metrics_window) == 10
        assert monitor.metrics_window[-1].query == "query 19"
        assert monitor.metrics_window[0].query == "query 10"

    def test_disabled_monitor_no_recording(self):
        """Test that disabled monitor doesn't record metrics."""
        with patch("rag.quality_monitor.settings") as mock_settings:
            mock_settings.enable_quality_monitoring = False
            mock_settings.quality_monitor_window_size = 100
            mock_settings.quality_alert_threshold = 0.7

            monitor = RetrievalQualityMonitor()
            monitor.enabled = False  # Explicitly set to False

            monitor.record_query(
                query="test",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
            )

            # Should not record when disabled
            assert len(monitor.metrics_window) == 0


class TestBaselineCalculation:
    """Tests for baseline metrics calculation."""

    def test_baseline_calculation_after_threshold(self):
        """Test that baseline is calculated after reaching threshold queries."""
        monitor = RetrievalQualityMonitor(window_size=200)
        monitor.baseline_window_size = 10  # Use small threshold for testing

        # Record 10 queries with consistent metrics
        for i in range(10):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
            )

        # Baseline should be established
        assert monitor.baseline_quality_score is not None
        assert monitor.baseline_latency_p95 is not None
        assert 79.0 <= monitor.baseline_quality_score <= 81.0  # Should be ~80.0

    def test_no_baseline_before_threshold(self):
        """Test that baseline is not calculated before threshold."""
        monitor = RetrievalQualityMonitor(window_size=200)
        monitor.baseline_window_size = 100

        # Record only 10 queries (less than threshold)
        for i in range(10):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
            )

        # Baseline should not be established yet
        assert monitor.baseline_quality_score is None
        assert monitor.baseline_latency_p95 is None


class TestAlertTriggering:
    """Tests for alert triggering logic."""

    def test_quality_drop_alert(self):
        """Test that quality drop triggers an alert."""
        monitor = RetrievalQualityMonitor(window_size=200, alert_threshold=0.7)
        monitor.baseline_quality_score = 90.0  # Set baseline manually

        # Record a query with significantly lower quality
        monitor.record_query(
            query="low quality query",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=50.0,  # 55% of baseline (below 70% threshold)
        )

        # Should have triggered an alert
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[-1]
        assert alert.alert_type == "quality_drop"
        assert alert.current_value == 50.0
        assert alert.baseline_value == 90.0

    def test_quality_drop_severity_levels(self):
        """Test different severity levels for quality drops."""
        monitor = RetrievalQualityMonitor(window_size=200, alert_threshold=0.7)
        monitor.baseline_quality_score = 100.0

        # Record query at 40% of baseline (critical)
        monitor.record_query(
            query="critical drop",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=40.0,
        )

        assert monitor.alerts[-1].severity == "critical"

        # Record query at 60% of baseline (warning)
        monitor.record_query(
            query="warning drop",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=60.0,
        )

        assert monitor.alerts[-1].severity == "warning"

    def test_latency_spike_alert(self):
        """Test that latency spike triggers an alert."""
        monitor = RetrievalQualityMonitor(window_size=200)
        # Set both baselines to enable alert checking
        monitor.baseline_quality_score = 80.0
        monitor.baseline_latency_p95 = 100.0

        # Record a query with 3x baseline latency
        monitor.record_query(
            query="slow query",
            query_type="chunk",
            retrieval_latency_ms=200.0,
            generation_latency_ms=100.0,  # Total: 300ms (3x baseline)
            num_chunks_retrieved=5,
            quality_score=80.0,
        )

        # Should have triggered an alert
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[-1]
        assert alert.alert_type == "latency_spike"
        assert alert.current_value == 300.0
        assert alert.baseline_value == 100.0

    def test_error_alert(self):
        """Test that errors trigger alerts."""
        monitor = RetrievalQualityMonitor(window_size=200)
        # Set baseline to enable alert checking
        monitor.baseline_quality_score = 80.0

        # Record a failed query
        monitor.record_query(
            query="failed query",
            query_type="chunk",
            retrieval_latency_ms=0.0,
            generation_latency_ms=0.0,
            num_chunks_retrieved=0,
            error="Database connection failed",
        )

        # Should have triggered an error alert
        assert len(monitor.alerts) > 0
        alert = monitor.alerts[-1]
        assert alert.alert_type == "error"
        assert alert.severity == "warning"

    def test_no_alert_when_quality_above_threshold(self):
        """Test that no alert is triggered when quality is acceptable."""
        monitor = RetrievalQualityMonitor(window_size=200, alert_threshold=0.7)
        monitor.baseline_quality_score = 90.0

        # Record query at 80% of baseline (above threshold)
        monitor.record_query(
            query="acceptable quality",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=72.0,  # 80% of baseline
        )

        # Should not trigger alert
        assert len(monitor.alerts) == 0


class TestAggregatedMetrics:
    """Tests for aggregated metrics calculation."""

    def test_aggregated_metrics_calculation(self):
        """Test calculation of aggregated metrics."""
        monitor = RetrievalQualityMonitor(window_size=100)

        # Record 10 queries with varying metrics
        quality_scores = [70, 75, 80, 85, 90, 95, 80, 85, 90, 75]
        for i, score in enumerate(quality_scores):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=float(score),
            )

        metrics = monitor.get_aggregated_metrics()

        assert metrics is not None
        assert metrics.window_size == 10
        assert metrics.total_queries == 10

        # Check quality metrics
        assert metrics.min_quality_score == 70.0
        assert metrics.max_quality_score == 95.0
        assert 82.0 <= metrics.avg_quality_score <= 83.0  # Average ~82.5

        # Check latency metrics
        assert metrics.avg_total_latency_ms == 150.0  # 50 + 100
        assert metrics.avg_retrieval_latency_ms == 50.0
        assert metrics.avg_generation_latency_ms == 100.0

        # Check cache metrics
        assert metrics.cache_hit_rate == 0.0  # No cache hits

        # Check query type distribution
        assert metrics.query_type_distribution == {"chunk": 10}

        # Check error rate
        assert metrics.error_rate == 0.0

    def test_aggregated_metrics_with_cache_hits(self):
        """Test aggregated metrics with cache hits."""
        monitor = RetrievalQualityMonitor(window_size=100)

        # Record 5 cache hits and 5 misses
        for i in range(10):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=0.0 if i < 5 else 50.0,
                generation_latency_ms=0.0 if i < 5 else 100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
                cache_hit=(i < 5),
            )

        metrics = monitor.get_aggregated_metrics()

        assert metrics.cache_hit_rate == 0.5  # 50% cache hit rate

    def test_aggregated_metrics_with_mixed_query_types(self):
        """Test aggregated metrics with different query types."""
        monitor = RetrievalQualityMonitor(window_size=100)

        # Record queries of different types
        query_types = ["chunk", "chunk", "entity", "hybrid", "chunk"]
        for i, qtype in enumerate(query_types):
            monitor.record_query(
                query=f"query {i}",
                query_type=qtype,
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
            )

        metrics = monitor.get_aggregated_metrics()

        assert metrics.query_type_distribution == {
            "chunk": 3,
            "entity": 1,
            "hybrid": 1
        }

    def test_aggregated_metrics_with_window_parameter(self):
        """Test aggregated metrics with window parameter."""
        monitor = RetrievalQualityMonitor(window_size=100)

        # Record 20 queries
        for i in range(20):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=float(80 + i),  # Varying scores
            )

        # Get metrics for last 5 queries only
        metrics = monitor.get_aggregated_metrics(window=5)

        assert metrics.window_size == 5
        assert metrics.min_quality_score == 95.0  # Last 5: 95-99
        assert metrics.max_quality_score == 99.0

    def test_aggregated_metrics_returns_none_when_empty(self):
        """Test that aggregated metrics returns None when no data."""
        monitor = RetrievalQualityMonitor(window_size=100)

        metrics = monitor.get_aggregated_metrics()

        assert metrics is None


class TestConfiguration:
    """Tests for quality monitoring configuration."""

    def test_default_configuration(self):
        """Test that quality monitoring has correct default settings."""
        from config.settings import settings

        # Verify settings exist
        assert hasattr(settings, 'enable_quality_monitoring')
        assert hasattr(settings, 'quality_monitor_window_size')
        assert hasattr(settings, 'quality_alert_threshold')

        # Verify types
        assert isinstance(settings.enable_quality_monitoring, bool)
        assert isinstance(settings.quality_monitor_window_size, int)
        assert isinstance(settings.quality_alert_threshold, float)

        # Verify reasonable defaults
        assert settings.quality_monitor_window_size > 0
        assert 0.0 <= settings.quality_alert_threshold <= 1.0

    def test_monitor_respects_config(self):
        """Test that monitor uses configuration values."""
        with patch("rag.quality_monitor.settings") as mock_settings:
            mock_settings.enable_quality_monitoring = True
            mock_settings.quality_monitor_window_size = 500
            mock_settings.quality_alert_threshold = 0.8

            monitor = RetrievalQualityMonitor()

            assert monitor.enabled is True
            assert monitor.window_size == 500
            assert monitor.alert_threshold == 0.8


class TestMonitorSummary:
    """Tests for quality monitor summary functionality."""

    def test_get_summary_when_disabled(self):
        """Test summary when monitoring is disabled."""
        with patch("rag.quality_monitor.settings") as mock_settings:
            mock_settings.enable_quality_monitoring = False
            mock_settings.quality_monitor_window_size = 100
            mock_settings.quality_alert_threshold = 0.7

            monitor = RetrievalQualityMonitor()
            monitor.enabled = False
            summary = monitor.get_summary()

            assert summary["enabled"] is False

    def test_get_summary_with_data(self):
        """Test summary with recorded metrics."""
        monitor = RetrievalQualityMonitor(window_size=100)
        monitor.baseline_quality_score = 85.0
        monitor.baseline_latency_p95 = 150.0

        # Record some queries
        for i in range(10):
            monitor.record_query(
                query=f"query {i}",
                query_type="chunk",
                retrieval_latency_ms=50.0,
                generation_latency_ms=100.0,
                num_chunks_retrieved=5,
                quality_score=80.0,
            )

        summary = monitor.get_summary()

        assert summary["enabled"] is True
        assert summary["window_size"] == 100
        assert summary["queries_tracked"] == 10
        assert summary["baseline"]["quality_score"] == 85.0
        assert summary["baseline"]["latency_p95_ms"] == 150.0
        assert "current_metrics" in summary
        assert "recent_alerts" in summary


class TestMonitorReset:
    """Tests for resetting the monitor."""

    def test_reset_clears_all_data(self):
        """Test that reset clears all metrics and alerts."""
        monitor = RetrievalQualityMonitor(window_size=100)

        # Record some data
        monitor.record_query(
            query="test",
            query_type="chunk",
            retrieval_latency_ms=50.0,
            generation_latency_ms=100.0,
            num_chunks_retrieved=5,
            quality_score=80.0,
        )

        monitor.baseline_quality_score = 85.0
        monitor.alerts.append(QualityAlert(
            timestamp=1234567890.0,
            alert_type="test",
            severity="warning",
            message="test alert",
            current_value=50.0,
            baseline_value=85.0,
            threshold=0.7,
        ))

        # Reset
        monitor.reset()

        # Verify everything is cleared
        assert len(monitor.metrics_window) == 0
        assert len(monitor.alerts) == 0
        assert monitor.baseline_quality_score is None
        assert monitor.baseline_latency_p95 is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
