"""
Adaptive routing with user feedback-based weight adjustment.

This module implements a feedback learning system that adjusts retrieval weights
based on user ratings (thumbs up/down) to improve routing decisions over time.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Represents a single feedback event."""
    query: str
    session_id: str
    message_id: str
    rating: int  # 1 for positive, -1 for negative
    routing_info: Dict[str, Any]
    timestamp: float
    weights_at_time: Dict[str, float]


@dataclass
class WeightState:
    """Tracks the state of adaptive weights."""
    chunk_weight: float = 0.5
    entity_weight: float = 0.3
    path_weight: float = 0.2
    last_updated: float = field(default_factory=time.time)
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0


class FeedbackLearner:
    """
    Learns optimal retrieval weights from user feedback.
    
    Uses exponential moving average to adjust weights based on positive/negative
    feedback signals. Implements conservative learning with minimum sample requirements
    and bounded weight adjustments.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        min_samples: int = 5,
        weight_min: float = 0.1,
        weight_max: float = 0.9,
        decay_factor: float = 0.95
    ):
        """
        Initialize feedback learner.
        
        Args:
            learning_rate: Step size for weight updates (0.0-1.0)
            min_samples: Minimum feedback samples before adjusting weights
            weight_min: Minimum allowed weight value
            weight_max: Maximum allowed weight value
            decay_factor: Decay for exponential moving average
        """
        self.learning_rate = learning_rate
        self.min_samples = min_samples
        self.weight_min = weight_min
        self.weight_max = weight_max
        self.decay_factor = decay_factor
        
        # Track feedback events
        self.feedback_history: List[FeedbackEvent] = []
        
        # Current weight state
        self.weights = WeightState()
        
        # Track performance by query type
        self.query_type_performance: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "total": 0}
        )
        
        # Track routing decisions
        self.routing_decisions: Dict[str, int] = defaultdict(int)
        
        logger.info(
            f"Initialized FeedbackLearner: lr={learning_rate}, "
            f"min_samples={min_samples}, decay={decay_factor}"
        )
    
    def record_feedback(
        self,
        query: str,
        session_id: str,
        message_id: str,
        rating: int,
        routing_info: Dict[str, Any]
    ) -> None:
        """
        Record a feedback event.
        
        Args:
            query: User query
            session_id: Session identifier
            message_id: Message identifier for the response
            rating: 1 for positive (thumbs up), -1 for negative (thumbs down)
            routing_info: Routing metadata (categories, confidence, etc.)
        """
        event = FeedbackEvent(
            query=query,
            session_id=session_id,
            message_id=message_id,
            rating=rating,
            routing_info=routing_info,
            timestamp=time.time(),
            weights_at_time={
                "chunk_weight": self.weights.chunk_weight,
                "entity_weight": self.weights.entity_weight,
                "path_weight": self.weights.path_weight
            }
        )
        
        self.feedback_history.append(event)
        
        # Update counters
        self.weights.total_feedback += 1
        if rating > 0:
            self.weights.positive_feedback += 1
        else:
            self.weights.negative_feedback += 1
        
        # Update query type performance
        query_type = routing_info.get("query_type", "unknown")
        self.query_type_performance[query_type]["total"] += 1
        if rating > 0:
            self.query_type_performance[query_type]["positive"] += 1
        else:
            self.query_type_performance[query_type]["negative"] += 1
        
        # Track routing decisions
        routed_to = routing_info.get("routed_to", "unknown")
        self.routing_decisions[routed_to] += 1
        
        logger.info(
            f"Recorded feedback: rating={rating}, query_type={query_type}, "
            f"total_feedback={self.weights.total_feedback}"
        )
        
        # Adjust weights if we have enough samples
        if self.weights.total_feedback >= self.min_samples:
            self._adjust_weights(event)
    
    def _adjust_weights(self, event: FeedbackEvent) -> None:
        """
        Adjust retrieval weights based on feedback.
        
        Strategy:
        - Positive feedback: slightly increase weights that were used
        - Negative feedback: slightly decrease weights that were used
        - Use exponential moving average for smooth adjustments
        - Normalize weights to sum to 1.0
        """
        rating = event.rating
        routing_info = event.routing_info
        
        # Determine which retrieval mode was used
        retrieval_mode = routing_info.get("retrieval_mode", "hybrid")
        query_type = routing_info.get("query_type", "unknown")
        
        # Calculate adjustment magnitude based on learning rate and rating
        adjustment = self.learning_rate * rating
        
        # Apply query-type specific adjustments
        if query_type == "entity_focused":
            # Entity-focused queries: adjust entity weight more
            self.weights.entity_weight += adjustment * 1.5
            self.weights.chunk_weight += adjustment * 0.5
        elif query_type == "keyword_focused":
            # Keyword-focused: adjust chunk weight more
            self.weights.chunk_weight += adjustment * 1.5
            self.weights.entity_weight += adjustment * 0.3
        elif query_type == "path_based":
            # Path-based: adjust path weight more
            self.weights.path_weight += adjustment * 1.5
            self.weights.entity_weight += adjustment * 0.5
        else:
            # Balanced: adjust all weights equally
            self.weights.chunk_weight += adjustment
            self.weights.entity_weight += adjustment
            self.weights.path_weight += adjustment
        
        # Clamp weights to valid range
        self.weights.chunk_weight = max(self.weight_min, min(self.weight_max, self.weights.chunk_weight))
        self.weights.entity_weight = max(self.weight_min, min(self.weight_max, self.weights.entity_weight))
        self.weights.path_weight = max(self.weight_min, min(self.weight_max, self.weights.path_weight))
        
        # Normalize weights to sum to 1.0
        total = self.weights.chunk_weight + self.weights.entity_weight + self.weights.path_weight
        if total > 0:
            self.weights.chunk_weight /= total
            self.weights.entity_weight /= total
            self.weights.path_weight /= total
        
        self.weights.last_updated = time.time()
        
        logger.info(
            f"Adjusted weights: chunk={self.weights.chunk_weight:.3f}, "
            f"entity={self.weights.entity_weight:.3f}, "
            f"path={self.weights.path_weight:.3f}"
        )
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current adaptive weights.
        
        Returns:
            Dictionary with chunk_weight, entity_weight, path_weight
        """
        return {
            "chunk_weight": self.weights.chunk_weight,
            "entity_weight": self.weights.entity_weight,
            "path_weight": self.weights.path_weight
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get feedback learning metrics.
        
        Returns:
            Dictionary with convergence metrics, accuracy, weight history
        """
        total = self.weights.total_feedback
        positive = self.weights.positive_feedback
        negative = self.weights.negative_feedback
        
        # Calculate accuracy (percentage of positive feedback)
        accuracy = (positive / total * 100) if total > 0 else 0.0
        
        # Calculate convergence (stability of weights over recent feedback)
        convergence = self._calculate_convergence()
        
        # Query type breakdown
        query_type_stats = {}
        for qtype, stats in self.query_type_performance.items():
            total_qt = stats["total"]
            if total_qt > 0:
                query_type_stats[qtype] = {
                    "total": total_qt,
                    "accuracy": (stats["positive"] / total_qt * 100)
                }
        
        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "accuracy": accuracy,
            "convergence": convergence,
            "current_weights": self.get_weights(),
            "last_updated": self.weights.last_updated,
            "query_type_performance": query_type_stats,
            "routing_decisions": dict(self.routing_decisions),
            "learning_active": total >= self.min_samples
        }
    
    def _calculate_convergence(self, window: int = 10) -> float:
        """
        Calculate weight convergence over recent feedback.
        
        Convergence is measured as the inverse of weight variance over the last N
        feedback events. Higher values indicate more stable weights (better convergence).
        
        Args:
            window: Number of recent feedback events to consider
            
        Returns:
            Convergence score (0.0-1.0), where 1.0 is perfect convergence
        """
        if len(self.feedback_history) < window:
            return 0.0
        
        # Get recent weight snapshots
        recent_events = self.feedback_history[-window:]
        
        # Calculate variance for each weight
        chunk_weights = [e.weights_at_time["chunk_weight"] for e in recent_events]
        entity_weights = [e.weights_at_time["entity_weight"] for e in recent_events]
        path_weights = [e.weights_at_time["path_weight"] for e in recent_events]
        
        def variance(values: List[float]) -> float:
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        
        # Average variance across all weights
        avg_variance = (
            variance(chunk_weights) +
            variance(entity_weights) +
            variance(path_weights)
        ) / 3.0
        
        # Convert to convergence score (inverse relationship)
        # Lower variance = higher convergence
        # Use exponential decay to map variance to [0, 1]
        convergence = max(0.0, min(1.0, 1.0 - avg_variance * 10.0))
        
        return convergence
    
    def reset(self) -> None:
        """Reset all feedback learning state."""
        self.feedback_history.clear()
        self.weights = WeightState()
        self.query_type_performance.clear()
        self.routing_decisions.clear()
        logger.info("Reset feedback learner state")
    
    def get_recommendation(
        self,
        query: str,
        query_type: str,
        default_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Get weight recommendation for a query.
        
        Args:
            query: User query
            query_type: Detected query type
            default_weights: Default weights to use if learning is not active
            
        Returns:
            Tuple of (recommended_weights, metadata)
        """
        # If not enough feedback, use defaults
        if self.weights.total_feedback < self.min_samples:
            weights = default_weights or {
                "chunk_weight": 0.5,
                "entity_weight": 0.3,
                "path_weight": 0.2
            }
            metadata = {
                "source": "default",
                "learning_active": False,
                "samples_needed": self.min_samples - self.weights.total_feedback
            }
        else:
            weights = self.get_weights()
            metadata = {
                "source": "adaptive",
                "learning_active": True,
                "accuracy": (self.weights.positive_feedback / self.weights.total_feedback * 100),
                "total_samples": self.weights.total_feedback
            }
        
        return weights, metadata


# Singleton instance
_feedback_learner: Optional[FeedbackLearner] = None


def get_feedback_learner() -> FeedbackLearner:
    """Get or create the singleton feedback learner instance."""
    global _feedback_learner
    if _feedback_learner is None:
        _feedback_learner = FeedbackLearner(
            learning_rate=getattr(settings, "adaptive_learning_rate", 0.1),
            min_samples=getattr(settings, "adaptive_min_samples", 5),
            weight_min=getattr(settings, "adaptive_weight_min", 0.1),
            weight_max=getattr(settings, "adaptive_weight_max", 0.9),
            decay_factor=getattr(settings, "adaptive_decay_factor", 0.95)
        )
    return _feedback_learner


def reset_feedback_learner() -> None:
    """Reset the singleton feedback learner."""
    global _feedback_learner
    if _feedback_learner is not None:
        _feedback_learner.reset()
