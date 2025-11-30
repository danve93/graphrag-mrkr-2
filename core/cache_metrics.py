"""Cache metrics collection and reporting."""

import logging
from typing import Dict, Any
from core.singletons import (
    get_entity_label_cache,
    get_embedding_cache,
    get_retrieval_cache,
    get_response_cache,
)

logger = logging.getLogger(__name__)


class CacheMetrics:
    """Collect and report cache performance metrics."""
    
    def __init__(self):
        """Initialize metrics counters."""
        self.entity_label_hits = 0
        self.entity_label_misses = 0
        self.embedding_hits = 0
        self.embedding_misses = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        self.response_hits = 0
        self.response_misses = 0
        self.response_invalidations = 0
    
    def record_entity_label_hit(self):
        """Record an entity label cache hit."""
        self.entity_label_hits += 1
    
    def record_entity_label_miss(self):
        """Record an entity label cache miss."""
        self.entity_label_misses += 1
    
    def record_embedding_hit(self):
        """Record an embedding cache hit."""
        self.embedding_hits += 1
    
    def record_embedding_miss(self):
        """Record an embedding cache miss."""
        self.embedding_misses += 1
    
    def record_retrieval_hit(self):
        """Record a retrieval cache hit."""
        self.retrieval_hits += 1
    
    def record_retrieval_miss(self):
        """Record a retrieval cache miss."""
        self.retrieval_misses += 1

    def record_response_hit(self):
        """Record a response cache hit."""
        self.response_hits += 1

    def record_response_miss(self):
        """Record a response cache miss."""
        self.response_misses += 1

    def record_response_invalidation(self):
        """Record a response cache invalidation (explicit clear/delete)."""
        self.response_invalidations += 1
    
    def get_report(self) -> Dict[str, Any]:
        """
        Generate cache performance report.
        
        Returns:
            Dictionary containing cache statistics for all cache types
        """
        def calc_hit_rate(hits, misses):
            total = hits + misses
            return (hits / total * 100) if total > 0 else 0.0
        
        return {
            "entity_labels": {
                "hits": self.entity_label_hits,
                "misses": self.entity_label_misses,
                "hit_rate": calc_hit_rate(self.entity_label_hits, self.entity_label_misses),
                "cache_size": len(get_entity_label_cache()),
            },
            "embeddings": {
                "hits": self.embedding_hits,
                "misses": self.embedding_misses,
                "hit_rate": calc_hit_rate(self.embedding_hits, self.embedding_misses),
                "cache_size": len(get_embedding_cache()),
            },
            "retrieval": {
                "hits": self.retrieval_hits,
                "misses": self.retrieval_misses,
                "hit_rate": calc_hit_rate(self.retrieval_hits, self.retrieval_misses),
                "cache_size": len(get_retrieval_cache()),
            },
            "response": {
                "hits": self.response_hits,
                "misses": self.response_misses,
                "hit_rate": calc_hit_rate(self.response_hits, self.response_misses),
                "cache_size": len(get_response_cache()),
                "invalidations": self.response_invalidations,
            },
            "summary": {
                "total_hits": self.entity_label_hits + self.embedding_hits + self.retrieval_hits,
                "total_misses": self.entity_label_misses + self.embedding_misses + self.retrieval_misses + self.response_misses,
                "overall_hit_rate": calc_hit_rate(
                    self.entity_label_hits + self.embedding_hits + self.retrieval_hits + self.response_hits,
                    self.entity_label_misses + self.embedding_misses + self.retrieval_misses + self.response_misses
                ),
            },
        }
    
    def log_report(self):
        """Log cache performance report to logger."""
        report = self.get_report()
        logger.info("=== Cache Performance Report ===")
        
        for cache_type, stats in report.items():
            if cache_type == "summary":
                logger.info(
                    f"SUMMARY: "
                    f"total_hits={stats['total_hits']}, "
                    f"total_misses={stats['total_misses']}, "
                    f"overall_hit_rate={stats['overall_hit_rate']:.1f}%"
                )
            else:
                logger.info(
                    f"{cache_type}: "
                    f"hits={stats['hits']}, "
                    f"misses={stats['misses']}, "
                    f"hit_rate={stats['hit_rate']:.1f}%, "
                    f"size={stats['cache_size']}"
                )
    
    def reset(self):
        """Reset all metrics counters."""
        self.entity_label_hits = 0
        self.entity_label_misses = 0
        self.embedding_hits = 0
        self.embedding_misses = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        self.response_hits = 0
        self.response_misses = 0
        self.response_invalidations = 0
        logger.info("Cache metrics reset")


# Global metrics instance
cache_metrics = CacheMetrics()
