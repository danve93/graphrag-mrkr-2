"""
Semantic cache for routing decisions - uses embeddings for similarity matching.
"""

import logging
import hashlib
from typing import Optional, Dict, Any

from cachetools import TTLCache
import numpy as np

from core.embeddings import embedding_manager
from config.settings import settings

logger = logging.getLogger(__name__)


class RoutingCache:
    """Semantic cache for query routing decisions."""

    def __init__(self, maxsize: int = 1000, ttl: int = 3600, similarity_threshold: float = 0.92):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.similarity_threshold = similarity_threshold
        self.hit_count = 0
        self.miss_count = 0

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached routing decision using semantic similarity."""
        if not settings.enable_routing_cache:
            return None

        try:
            # Compute current query embedding (sync method, no await)
            query_embedding = embedding_manager.get_embedding(query)
            if query_embedding is None:
                self.miss_count += 1
                return None

            query_vec = np.array(query_embedding, dtype=float)

            # Compare against cached embeddings
            best_match_key = None
            best_sim = -1.0
            for key, emb in self.embeddings.items():
                if emb is None:
                    continue
                # cosine similarity
                denom = (np.linalg.norm(query_vec) * np.linalg.norm(emb))
                if denom == 0:
                    continue
                sim = float(np.dot(query_vec, emb) / denom)
                if sim > best_sim:
                    best_sim = sim
                    best_match_key = key

            if best_match_key and best_sim >= self.similarity_threshold:
                self.hit_count += 1
                logger.debug(f"Routing cache hit (sim={best_sim:.3f}) for key {best_match_key[:8]}...")
                return self.cache.get(best_match_key)

            self.miss_count += 1
            return None
        except Exception as e:
            logger.error(f"Routing cache lookup failed: {e}")
            return None

    async def set(self, query: str, result: Dict[str, Any]):
        """Cache routing decision with query embedding."""
        if not settings.enable_routing_cache:
            return

        try:
            key = hashlib.md5(query.lower().encode()).hexdigest()
            # get_embedding is synchronous, no await needed
            query_embedding = embedding_manager.get_embedding(query)
            if query_embedding is None:
                return
            self.cache[key] = result
            self.embeddings[key] = np.array(query_embedding, dtype=float)
            logger.debug(f"Cached routing decision for query (hash: {key[:8]}...)")
        except Exception as e:
            logger.error(f"Routing cache store failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total) if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "maxsize": self.cache.maxsize,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "ttl": self.cache.ttl,
        }


# Singleton instance
routing_cache = RoutingCache(
    maxsize=settings.routing_cache_size,
    ttl=settings.routing_cache_ttl,
    similarity_threshold=settings.routing_cache_similarity_threshold,
)
