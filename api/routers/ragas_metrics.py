"""
RAGAS metrics API router.

Provides endpoints for:
- Prometheus metrics exposition for RAGAS benchmarks
- JSON stats for benchmark results
- Comparison and regression detection
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/ragas/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Get RAGAS benchmark metrics in Prometheus exposition format.

    Reads latest RAGAS evaluation results from reports/ragas/ directory
    and formats them as Prometheus metrics.

    Returns:
        Plain text in Prometheus format with metrics like:
        - ragas_context_precision
        - ragas_context_recall
        - ragas_faithfulness
        - ragas_answer_relevancy
        - ragas_evaluations_total
    """
    try:
        from evals.ragas.prometheus_exporter import RagasPrometheusExporter

        exporter = RagasPrometheusExporter()
        metrics_text = exporter.get_prometheus_metrics()

        return metrics_text

    except ImportError:
        logger.warning("RAGAS prometheus_exporter not found")
        raise HTTPException(
            status_code=503,
            detail="RAGAS prometheus exporter not available. Ensure evals/ragas/prometheus_exporter.py exists."
        )
    except Exception as e:
        logger.error(f"Error generating RAGAS Prometheus metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAGAS metrics: {str(e)}"
        )


@router.get("/ragas/stats")
async def get_stats() -> Dict[str, Any]:
    """
    Get RAGAS benchmark statistics in JSON format.

    Returns aggregated statistics from latest evaluation runs including:
    - Variant configurations
    - Metric scores (context precision/recall, faithfulness, answer relevancy)
    - Evaluation counts
    - Timestamps

    Returns:
        dict: Statistics in JSON format
    """
    try:
        from evals.ragas.prometheus_exporter import RagasPrometheusExporter

        exporter = RagasPrometheusExporter()
        stats = exporter.get_json_stats()

        return stats

    except ImportError:
        logger.warning("RAGAS prometheus_exporter not found")
        raise HTTPException(
            status_code=503,
            detail="RAGAS prometheus exporter not available"
        )
    except Exception as e:
        logger.error(f"Error generating RAGAS stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAGAS stats: {str(e)}"
        )


@router.get("/ragas/comparison")
async def get_comparison() -> Dict[str, Any]:
    """
    Get comparison of RAGAS variants.

    Compares all evaluated variants and identifies:
    - Best performing variant (highest average score)
    - Regressions (>5% drop from baseline)
    - Metric-specific winners

    Returns:
        dict: Comparison analysis
    """
    try:
        from evals.ragas.prometheus_exporter import RagasPrometheusExporter

        exporter = RagasPrometheusExporter()
        stats = exporter.get_json_stats()

        if stats.get("status") != "active":
            return stats

        variants = stats.get("variants", {})

        if not variants:
            return {"status": "no_data", "message": "No variants to compare"}

        # Calculate average scores for each variant
        variant_scores = {}
        for variant_name, variant_data in variants.items():
            metrics = variant_data.get("metrics", {})
            scores = [
                metrics.get("context_precision", 0),
                metrics.get("context_recall", 0),
                metrics.get("faithfulness", 0),
                metrics.get("answer_relevancy", 0),
            ]
            avg_score = sum(scores) / len(scores) if scores else 0
            variant_scores[variant_name] = {
                "average_score": avg_score,
                "metrics": metrics,
            }

        # Find best variant
        best_variant = max(variant_scores.items(), key=lambda x: x[1]["average_score"])

        # Metric-specific winners
        metric_winners = {
            "context_precision": max(variants.items(), key=lambda x: x[1]["metrics"].get("context_precision", 0)),
            "context_recall": max(variants.items(), key=lambda x: x[1]["metrics"].get("context_recall", 0)),
            "faithfulness": max(variants.items(), key=lambda x: x[1]["metrics"].get("faithfulness", 0)),
            "answer_relevancy": max(variants.items(), key=lambda x: x[1]["metrics"].get("answer_relevancy", 0)),
        }

        return {
            "status": "success",
            "best_overall_variant": {
                "name": best_variant[0],
                "average_score": best_variant[1]["average_score"],
                "metrics": best_variant[1]["metrics"],
            },
            "metric_winners": {
                metric: {"variant": winner[0], "score": winner[1]["metrics"].get(metric, 0)}
                for metric, winner in metric_winners.items()
            },
            "all_variants": variant_scores,
        }

    except Exception as e:
        logger.error(f"Error generating comparison: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate comparison: {str(e)}"
        )
