"""
Prometheus metrics exporter for TruLens.

Queries TruLens database and formats metrics in Prometheus exposition format.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TruLensMetricsExporter:
    """
    Export TruLens monitoring data as Prometheus metrics.

    Queries the TruLens database for recent feedback evaluations and
    formats them as Prometheus metrics for scraping.
    """

    def __init__(self, aggregation_window_minutes: int = 5):
        """
        Initialize metrics exporter.

        Args:
            aggregation_window_minutes: Window for aggregating metrics (default 5min)
        """
        self.aggregation_window = aggregation_window_minutes

    def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus exposition format.

        Returns:
            str: Metrics in Prometheus text format
        """
        try:
            from evals.trulens.trulens_wrapper import get_monitor

            monitor = get_monitor()

            if monitor is None or not monitor.enabled or monitor.tru is None:
                # Return empty metrics if monitoring is disabled
                return self._empty_metrics()

            # Query TruLens database for recent records
            records = self._get_recent_records(monitor.tru)

            # Aggregate metrics
            metrics = self._aggregate_metrics(records)

            # Format as Prometheus exposition format
            return self._format_prometheus(metrics)

        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}", exc_info=True)
            return self._error_metrics(str(e))

    def get_json_stats(self) -> Dict[str, Any]:
        """
        Get statistics in JSON format for API endpoint.

        Returns:
            dict: Statistics with aggregated metrics
        """
        try:
            from evals.trulens.trulens_wrapper import get_monitor

            monitor = get_monitor()

            if monitor is None or not monitor.enabled or monitor.tru is None:
                return {
                    "status": "disabled",
                    "monitoring_enabled": False,
                }

            # Query TruLens database
            records = self._get_recent_records(monitor.tru)

            # Aggregate metrics
            metrics = self._aggregate_metrics(records)

            return {
                "status": "active",
                "monitoring_enabled": True,
                "aggregation_window_minutes": self.aggregation_window,
                "total_records": len(records),
                "metrics": metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating JSON stats: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    def _get_recent_records(self, tru) -> List[Dict]:
        """
        Get recent TruLens records from database.

        Args:
            tru: TruLens instance

        Returns:
            list: Recent records with feedback scores
        """
        try:
            # Calculate time window
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.aggregation_window)

            # Query TruLens for recent records
            # Note: TruLens API may vary by version
            records_df = tru.get_records_and_feedback()

            if records_df is None or len(records_df) == 0:
                return []

            # Filter to recent records
            # Convert timestamp column to datetime if needed
            if 'ts' in records_df.columns:
                records_df['ts'] = pd.to_datetime(records_df['ts'])
                recent_records = records_df[records_df['ts'] >= cutoff_time]
            else:
                # Fallback: use all records if timestamp filtering fails
                recent_records = records_df

            return recent_records.to_dict('records')

        except Exception as e:
            logger.warning(f"Error querying TruLens records: {e}")
            return []

    def _aggregate_metrics(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics from TruLens records.

        Args:
            records: List of TruLens record dicts

        Returns:
            dict: Aggregated metrics
        """
        if not records:
            return {
                "query_count": 0,
                "error_count": 0,
                "avg_answer_relevance": 0.0,
                "avg_groundedness": 0.0,
                "avg_context_relevance": 0.0,
                "avg_latency_seconds": 0.0,
            }

        # Initialize accumulators
        answer_relevance_scores = []
        groundedness_scores = []
        context_relevance_scores = []
        latencies = []
        error_count = 0

        for record in records:
            # Extract feedback scores
            # Note: TruLens stores feedback in specific columns/fields
            try:
                # Answer Relevance
                if 'Answer Relevance' in record:
                    score = record['Answer Relevance']
                    if score is not None:
                        answer_relevance_scores.append(float(score))

                # Groundedness
                if 'Groundedness' in record:
                    score = record['Groundedness']
                    if score is not None:
                        groundedness_scores.append(float(score))

                # Context Relevance
                if 'Context Relevance' in record:
                    score = record['Context Relevance']
                    if score is not None:
                        context_relevance_scores.append(float(score))

                # Latency (if tracked in metadata)
                if 'latency' in record:
                    latencies.append(float(record['latency']))

                # Track errors
                if record.get('error') or record.get('status') == 'error':
                    error_count += 1

            except Exception as e:
                logger.warning(f"Error processing record: {e}")
                error_count += 1

        # Calculate averages
        return {
            "query_count": len(records),
            "error_count": error_count,
            "avg_answer_relevance": sum(answer_relevance_scores) / len(answer_relevance_scores) if answer_relevance_scores else 0.0,
            "avg_groundedness": sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0.0,
            "avg_context_relevance": sum(context_relevance_scores) / len(context_relevance_scores) if context_relevance_scores else 0.0,
            "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
            "p95_latency_seconds": self._percentile(latencies, 95) if latencies else 0.0,
            "p99_latency_seconds": self._percentile(latencies, 99) if latencies else 0.0,
        }

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * (percentile / 100.0))
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _format_prometheus(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrics in Prometheus exposition format.

        Args:
            metrics: Aggregated metrics dict

        Returns:
            str: Prometheus text format
        """
        lines = []

        # Total queries counter
        lines.append("# HELP trulens_queries_total Total number of queries monitored by TruLens")
        lines.append("# TYPE trulens_queries_total counter")
        lines.append(f"trulens_queries_total {metrics['query_count']}")
        lines.append("")

        # Error counter
        lines.append("# HELP trulens_errors_total Total number of errors during monitoring")
        lines.append("# TYPE trulens_errors_total counter")
        lines.append(f"trulens_errors_total {metrics['error_count']}")
        lines.append("")

        # Answer relevance gauge
        lines.append("# HELP trulens_answer_relevance_score Average answer relevance score (0-1)")
        lines.append("# TYPE trulens_answer_relevance_score gauge")
        lines.append(f"trulens_answer_relevance_score {metrics['avg_answer_relevance']:.4f}")
        lines.append("")

        # Groundedness gauge
        lines.append("# HELP trulens_groundedness_score Average groundedness score (0-1)")
        lines.append("# TYPE trulens_groundedness_score gauge")
        lines.append(f"trulens_groundedness_score {metrics['avg_groundedness']:.4f}")
        lines.append("")

        # Context relevance gauge
        lines.append("# HELP trulens_context_relevance_score Average context relevance score (0-1)")
        lines.append("# TYPE trulens_context_relevance_score gauge")
        lines.append(f"trulens_context_relevance_score {metrics['avg_context_relevance']:.4f}")
        lines.append("")

        # Latency metrics
        lines.append("# HELP trulens_query_latency_seconds Query latency in seconds")
        lines.append("# TYPE trulens_query_latency_seconds summary")
        lines.append(f"trulens_query_latency_seconds_sum {metrics['avg_latency_seconds'] * metrics['query_count']:.4f}")
        lines.append(f"trulens_query_latency_seconds_count {metrics['query_count']}")
        lines.append(f"trulens_query_latency_seconds{{quantile=\"0.95\"}} {metrics['p95_latency_seconds']:.4f}")
        lines.append(f"trulens_query_latency_seconds{{quantile=\"0.99\"}} {metrics['p99_latency_seconds']:.4f}")
        lines.append("")

        return "\n".join(lines)

    def _empty_metrics(self) -> str:
        """Return empty metrics when monitoring is disabled."""
        return """# TruLens monitoring is disabled
# Enable with ENABLE_TRULENS_MONITORING=1
"""

    def _error_metrics(self, error_msg: str) -> str:
        """Return error metrics."""
        return f"""# TruLens metrics error
# {error_msg}
trulens_errors_total 1
"""
