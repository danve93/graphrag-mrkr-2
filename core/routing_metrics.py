"""
Metrics tracking for query routing and RAG failure points.
"""

import logging
from typing import Dict, Any, List
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class RoutingMetrics:
    """Track routing performance and failure point occurrences."""

    def __init__(self):
        self.routing_latency: List[float] = []  # seconds
        self.routing_accuracy: List[float] = []  # user feedback (0.0-1.0)
        self.category_usage: Counter = Counter()
        self.fallback_triggered: int = 0
        self.multi_category_queries: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # Failure point tracking (FP1-7)
        self.failure_points: defaultdict[str, int] = defaultdict(int)

        self.total_queries: int = 0
        self._metrics_file = "data/routing_metrics.json"
        
        # Load persisted metrics if available
        self.load_metrics()

    def load_metrics(self):
        """Load metrics from JSON file."""
        import json
        import os
        
        if not os.path.exists(self._metrics_file):
            return
            
        try:
            with open(self._metrics_file, 'r') as f:
                data = json.load(f)
                
            self.total_queries = data.get("total_queries", 0)
            self.routing_latency = data.get("routing_latency", [])
            self.routing_accuracy = data.get("routing_accuracy", [])
            self.category_usage = Counter(data.get("category_usage", {}))
            self.fallback_triggered = data.get("fallback_triggered", 0)
            self.multi_category_queries = data.get("multi_category_queries", 0)
            self.cache_hits = data.get("cache_hits", 0)
            self.cache_misses = data.get("cache_misses", 0)
            self.failure_points = defaultdict(int, data.get("failure_points", {}))
            
            logger.info(f"Loaded routing metrics from {self._metrics_file}")
        except Exception as e:
            logger.error(f"Failed to load routing metrics: {e}")

    def save_metrics(self):
        """Save metrics to JSON file."""
        import json
        import os
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._metrics_file), exist_ok=True)
            
            data = {
                "total_queries": self.total_queries,
                "routing_latency": self.routing_latency,
                "routing_accuracy": self.routing_accuracy,
                "category_usage": dict(self.category_usage),
                "fallback_triggered": self.fallback_triggered,
                "multi_category_queries": self.multi_category_queries,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "failure_points": dict(self.failure_points)
            }
            
            with open(self._metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save routing metrics: {e}")

    def clear_metrics(self):
        """Reset all metrics and clear persistence file."""
        import os
        
        self.routing_latency = []
        self.routing_accuracy = []
        self.category_usage = Counter()
        self.fallback_triggered = 0
        self.multi_category_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.failure_points = defaultdict(int)
        self.total_queries = 0
        
        try:
            if os.path.exists(self._metrics_file):
                os.remove(self._metrics_file)
            logger.info("Cleared routing metrics")
        except Exception as e:
            logger.error(f"Failed to clear routing metrics file: {e}")

    def record_routing(
        self,
        categories: List[str],
        confidence: float,
        latency_ms: float,
        used_cache: bool = False,
        fallback_used: bool = False,
    ):
        """Record a routing decision."""
        self.total_queries += 1
        # store seconds
        self.routing_latency.append(latency_ms / 1000.0)

        for category in categories:
            self.category_usage[category] += 1

        if len(categories) > 2:
            self.multi_category_queries += 1

        if used_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if fallback_used:
            self.fallback_triggered += 1
            
        # Save after update
        self.save_metrics()

    def record_failure_point(self, fp_type: str, details: Dict[str, Any]):
        """
        Record occurrence of a failure point.

        fp_type: FP1, FP2, FP3, FP4, FP5, FP6, or FP7
        details: context about the failure
        """
        self.failure_points[fp_type] += 1
        self.save_metrics()
        logger.warning(f"Failure Point {fp_type} detected: {details}")

    def record_user_feedback(self, correct: bool):
        """Record user feedback on routing accuracy."""
        self.routing_accuracy.append(1.0 if correct else 0.0)
        self.save_metrics()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        cache_total = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / cache_total if cache_total > 0 else 0.0

        avg_latency = (
            sum(self.routing_latency) / len(self.routing_latency)
            if self.routing_latency
            else 0.0
        )

        avg_accuracy = (
            sum(self.routing_accuracy) / len(self.routing_accuracy)
            if self.routing_accuracy
            else None
        )

        fallback_rate = (
            self.fallback_triggered / self.total_queries if self.total_queries > 0 else 0.0
        )

        multi_category_rate = (
            self.multi_category_queries / self.total_queries if self.total_queries > 0 else 0.0
        )

        # Calculate failure point rates
        fp_rates = {
            fp: count / self.total_queries if self.total_queries > 0 else 0.0
            for fp, count in self.failure_points.items()
        }

        return {
            "total_queries": self.total_queries,
            "avg_routing_latency_ms": avg_latency * 1000,
            "routing_accuracy": avg_accuracy,
            "cache_hit_rate": cache_hit_rate,
            "fallback_rate": fallback_rate,
            "multi_category_rate": multi_category_rate,
            "top_categories": self.category_usage.most_common(10),
            "failure_point_rates": fp_rates,
            "failure_point_counts": dict(self.failure_points),
        }


# Singleton instance
routing_metrics = RoutingMetrics()
