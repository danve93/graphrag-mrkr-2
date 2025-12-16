"""
Prometheus metrics exporter for RAGAS evaluation results.

Reads RAGAS JSON reports and exports metrics in Prometheus format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class RagasPrometheusExporter:
    """
    Export RAGAS benchmark results as Prometheus metrics.

    Reads JSON result files from reports/ragas/ and formats them
    as Prometheus metrics for tracking quality trends over time.
    """

    def __init__(self, reports_dir: str = "reports/ragas"):
        """
        Initialize RAGAS Prometheus exporter.

        Args:
            reports_dir: Directory containing RAGAS JSON reports
        """
        self.reports_dir = Path(reports_dir)

    def get_prometheus_metrics(self) -> str:
        """
        Get RAGAS metrics in Prometheus exposition format.

        Returns:
            str: Metrics in Prometheus text format
        """
        try:
            # Load latest results for each variant
            variant_results = self._load_latest_results()

            if not variant_results:
                return self._empty_metrics()

            # Format as Prometheus metrics
            return self._format_prometheus(variant_results)

        except Exception as e:
            logger.error(f"Error exporting RAGAS Prometheus metrics: {e}", exc_info=True)
            return self._error_metrics(str(e))

    def get_json_stats(self) -> Dict[str, Any]:
        """
        Get RAGAS statistics in JSON format.

        Returns:
            dict: Statistics with benchmark results
        """
        try:
            variant_results = self._load_latest_results()

            if not variant_results:
                return {
                    "status": "no_data",
                    "message": "No RAGAS results found in reports directory"
                }

            # Extract summary statistics
            stats = {
                "status": "active",
                "reports_directory": str(self.reports_dir),
                "total_variants": len(variant_results),
                "variants": {},
                "timestamp": datetime.utcnow().isoformat(),
            }

            for variant_name, result in variant_results.items():
                metrics = result.get("metrics", {}).get("aggregate", {})
                stats["variants"][variant_name] = {
                    "run_id": result.get("run_id"),
                    "timestamp": result.get("timestamp_utc"),
                    "evaluated_count": result.get("counts", {}).get("evaluated", 0),
                    "metrics": metrics,
                }

            return stats

        except Exception as e:
            logger.error(f"Error generating RAGAS JSON stats: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def _load_latest_results(self) -> Dict[str, Dict]:
        """
        Load latest RAGAS results for each variant.

        Returns:
            dict: Mapping of variant_name -> result_dict
        """
        if not self.reports_dir.exists():
            logger.warning(f"RAGAS reports directory not found: {self.reports_dir}")
            return {}

        variant_results = {}

        # Find all JSON result files
        for json_file in self.reports_dir.glob("*.json"):
            try:
                with json_file.open() as f:
                    result = json.load(f)

                variant = result.get("variant", json_file.stem)

                # Store latest result for each variant
                # (could be enhanced to track history)
                variant_results[variant] = result

            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue

        return variant_results

    def _format_prometheus(self, variant_results: Dict[str, Dict]) -> str:
        """
        Format RAGAS results in Prometheus exposition format.

        Args:
            variant_results: Mapping of variant -> result dict

        Returns:
            str: Prometheus text format
        """
        lines = []

        # Total evaluations counter
        lines.append("# HELP ragas_evaluations_total Total number of samples evaluated by RAGAS")
        lines.append("# TYPE ragas_evaluations_total counter")
        for variant, result in variant_results.items():
            count = result.get("counts", {}).get("evaluated", 0)
            lines.append(f'ragas_evaluations_total{{variant="{variant}"}} {count}')
        lines.append("")

        # Context precision gauge
        lines.append("# HELP ragas_context_precision Context precision score (0-1)")
        lines.append("# TYPE ragas_context_precision gauge")
        for variant, result in variant_results.items():
            score = result.get("metrics", {}).get("aggregate", {}).get("context_precision", 0.0)
            lines.append(f'ragas_context_precision{{variant="{variant}"}} {score:.4f}')
        lines.append("")

        # Context recall gauge
        lines.append("# HELP ragas_context_recall Context recall score (0-1)")
        lines.append("# TYPE ragas_context_recall gauge")
        for variant, result in variant_results.items():
            score = result.get("metrics", {}).get("aggregate", {}).get("context_recall", 0.0)
            lines.append(f'ragas_context_recall{{variant="{variant}"}} {score:.4f}')
        lines.append("")

        # Faithfulness gauge
        lines.append("# HELP ragas_faithfulness Faithfulness score (0-1)")
        lines.append("# TYPE ragas_faithfulness gauge")
        for variant, result in variant_results.items():
            score = result.get("metrics", {}).get("aggregate", {}).get("faithfulness", 0.0)
            lines.append(f'ragas_faithfulness{{variant="{variant}"}} {score:.4f}')
        lines.append("")

        # Answer relevancy gauge
        lines.append("# HELP ragas_answer_relevancy Answer relevancy score (0-1)")
        lines.append("# TYPE ragas_answer_relevancy gauge")
        for variant, result in variant_results.items():
            score = result.get("metrics", {}).get("aggregate", {}).get("answer_relevancy", 0.0)
            lines.append(f'ragas_answer_relevancy{{variant="{variant}"}} {score:.4f}')
        lines.append("")

        # Evaluation timestamp
        lines.append("# HELP ragas_last_evaluation_timestamp Unix timestamp of last evaluation")
        lines.append("# TYPE ragas_last_evaluation_timestamp gauge")
        for variant, result in variant_results.items():
            timestamp_str = result.get("timestamp_utc", "")
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp = dt.timestamp()
            except:
                timestamp = 0
            lines.append(f'ragas_last_evaluation_timestamp{{variant="{variant}"}} {timestamp}')
        lines.append("")

        # Error count
        lines.append("# HELP ragas_errors_total Total number of errors during evaluation")
        lines.append("# TYPE ragas_errors_total counter")
        for variant, result in variant_results.items():
            errors = result.get("counts", {}).get("errors", 0)
            lines.append(f'ragas_errors_total{{variant="{variant}"}} {errors}')
        lines.append("")

        return "\n".join(lines)

    def _empty_metrics(self) -> str:
        """Return empty metrics when no data available."""
        return """# RAGAS benchmarks not found
# Run RAGAS evaluations first: uv run python evals/ragas/ragas_runner.py
"""

    def _error_metrics(self, error_msg: str) -> str:
        """Return error metrics."""
        return f"""# RAGAS metrics error
# {error_msg}
ragas_errors_total 1
"""


def export_latest_results_to_prometheus(output_file: str = "reports/ragas/metrics.prom"):
    """
    Export latest RAGAS results to Prometheus text file.

    Args:
        output_file: Output file path for Prometheus metrics
    """
    exporter = RagasPrometheusExporter()
    metrics_text = exporter.get_prometheus_metrics()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w') as f:
        f.write(metrics_text)

    logger.info(f"Exported RAGAS metrics to {output_path}")


if __name__ == "__main__":
    # Standalone script to export metrics
    export_latest_results_to_prometheus()
    print("✅ RAGAS metrics exported to reports/ragas/metrics.prom")
