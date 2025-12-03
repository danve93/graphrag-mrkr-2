"""
Feedback metrics tracking and storage.

Stores user feedback in Neo4j and computes metrics for adaptive routing.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from core.graph_db import graph_db

logger = logging.getLogger(__name__)


class FeedbackMetrics:
    """Track and store user feedback for routing decisions."""
    
    def __init__(self):
        """Initialize feedback metrics tracker."""
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Ensure Neo4j schema for feedback storage."""
        try:
            with graph_db.driver.session() as session:
                # Create Feedback node constraint
                session.run("""
                    CREATE CONSTRAINT feedback_id_unique IF NOT EXISTS
                    FOR (f:Feedback) REQUIRE f.id IS UNIQUE
                """)
                
                # Create indexes
                session.run("""
                    CREATE INDEX feedback_timestamp IF NOT EXISTS
                    FOR (f:Feedback) ON (f.timestamp)
                """)
                
                session.run("""
                    CREATE INDEX feedback_session IF NOT EXISTS
                    FOR (f:Feedback) ON (f.session_id)
                """)
                
                logger.info("Feedback schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize feedback schema: {e}")
    
    def store_feedback(
        self,
        query: str,
        session_id: str,
        message_id: str,
        rating: int,
        routing_info: Dict[str, Any],
        weights_used: Dict[str, float]
    ) -> str:
        """
        Store feedback event in Neo4j.
        
        Args:
            query: User query
            session_id: Session identifier
            message_id: Message identifier
            rating: 1 for positive, -1 for negative
            routing_info: Routing metadata
            weights_used: Retrieval weights at time of query
            
        Returns:
            Feedback node ID
        """
        try:
            feedback_id = f"{session_id}_{message_id}_{int(time.time() * 1000)}"
            
            with graph_db.driver.session() as session:
                result = session.run("""
                    CREATE (f:Feedback {
                        id: $feedback_id,
                        query: $query,
                        session_id: $session_id,
                        message_id: $message_id,
                        rating: $rating,
                        timestamp: datetime($timestamp),
                        query_type: $query_type,
                        routed_to: $routed_to,
                        routing_confidence: $routing_confidence,
                        chunk_weight: $chunk_weight,
                        entity_weight: $entity_weight,
                        path_weight: $path_weight,
                        retrieval_mode: $retrieval_mode
                    })
                    RETURN f.id as id
                """,
                    feedback_id=feedback_id,
                    query=query,
                    session_id=session_id,
                    message_id=message_id,
                    rating=rating,
                    timestamp=datetime.utcnow().isoformat(),
                    query_type=routing_info.get("query_type", "unknown"),
                    routed_to=routing_info.get("routed_to", "unknown"),
                    routing_confidence=routing_info.get("confidence", 0.0),
                    chunk_weight=weights_used.get("chunk_weight", 0.5),
                    entity_weight=weights_used.get("entity_weight", 0.3),
                    path_weight=weights_used.get("path_weight", 0.2),
                    retrieval_mode=routing_info.get("retrieval_mode", "hybrid")
                )
                
                logger.info(f"Stored feedback: {feedback_id}, rating={rating}")
                return feedback_id
                
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return ""
    
    def get_recent_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent feedback events.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of feedback events
        """
        try:
            with graph_db.driver.session() as session:
                result = session.run("""
                    MATCH (f:Feedback)
                    RETURN f
                    ORDER BY f.timestamp DESC
                    LIMIT $limit
                """, limit=limit)
                
                feedback_list = []
                for record in result:
                    node = record["f"]
                    feedback_list.append(dict(node))
                
                return feedback_list
                
        except Exception as e:
            logger.error(f"Failed to get recent feedback: {e}")
            return []
    
    def get_metrics_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Get feedback metrics summary.
        
        Args:
            days: Number of days to include in summary
            
        Returns:
            Dictionary with metrics
        """
        try:
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            with graph_db.driver.session() as session:
                # Overall stats
                result = session.run("""
                    MATCH (f:Feedback)
                    WHERE f.timestamp >= datetime($cutoff)
                    RETURN 
                        count(f) as total,
                        sum(CASE WHEN f.rating > 0 THEN 1 ELSE 0 END) as positive,
                        sum(CASE WHEN f.rating < 0 THEN 1 ELSE 0 END) as negative,
                        avg(f.rating) as avg_rating
                """, cutoff=cutoff_date).single()
                
                total = result["total"] or 0
                positive = result["positive"] or 0
                negative = result["negative"] or 0
                avg_rating = result["avg_rating"] or 0.0
                
                # Query type breakdown
                query_type_result = session.run("""
                    MATCH (f:Feedback)
                    WHERE f.timestamp >= datetime($cutoff)
                    WITH f.query_type as query_type, 
                         count(f) as total,
                         sum(CASE WHEN f.rating > 0 THEN 1 ELSE 0 END) as positive
                    RETURN query_type, total, positive
                """, cutoff=cutoff_date)
                
                query_type_stats = {}
                for record in query_type_result:
                    qtype = record["query_type"]
                    qt_total = record["total"]
                    qt_positive = record["positive"]
                    query_type_stats[qtype] = {
                        "total": qt_total,
                        "accuracy": (qt_positive / qt_total * 100) if qt_total > 0 else 0.0
                    }
                
                # Routing decision breakdown
                routing_result = session.run("""
                    MATCH (f:Feedback)
                    WHERE f.timestamp >= datetime($cutoff)
                    RETURN f.routed_to as routed_to, count(f) as count
                """, cutoff=cutoff_date)
                
                routing_decisions = {}
                for record in routing_result:
                    routing_decisions[record["routed_to"]] = record["count"]
                
                # Weight trends (last 20 feedback events)
                weight_trends_result = session.run("""
                    MATCH (f:Feedback)
                    RETURN f.timestamp as timestamp,
                           f.chunk_weight as chunk_weight,
                           f.entity_weight as entity_weight,
                           f.path_weight as path_weight
                    ORDER BY f.timestamp DESC
                    LIMIT 20
                """)
                
                weight_trends = []
                for record in weight_trends_result:
                    weight_trends.append({
                        "timestamp": str(record["timestamp"]),
                        "chunk_weight": record["chunk_weight"],
                        "entity_weight": record["entity_weight"],
                        "path_weight": record["path_weight"]
                    })
                
                return {
                    "total_feedback": total,
                    "positive_feedback": positive,
                    "negative_feedback": negative,
                    "accuracy": (positive / total * 100) if total > 0 else 0.0,
                    "avg_rating": float(avg_rating),
                    "query_type_performance": query_type_stats,
                    "routing_decisions": routing_decisions,
                    "weight_trends": weight_trends,
                    "period_days": days
                }
                
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "accuracy": 0.0,
                "avg_rating": 0.0,
                "query_type_performance": {},
                "routing_decisions": {},
                "weight_trends": [],
                "period_days": days
            }
    
    def clear_feedback(self, days: Optional[int] = None) -> int:
        """
        Clear feedback data.
        
        Args:
            days: If provided, only clear feedback older than N days.
                  If None, clear all feedback.
        
        Returns:
            Number of feedback nodes deleted
        """
        try:
            with graph_db.driver.session() as session:
                if days is not None:
                    cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                    result = session.run("""
                        MATCH (f:Feedback)
                        WHERE f.timestamp < datetime($cutoff)
                        DELETE f
                        RETURN count(f) as deleted
                    """, cutoff=cutoff_date)
                else:
                    result = session.run("""
                        MATCH (f:Feedback)
                        DELETE f
                        RETURN count(f) as deleted
                    """)
                
                deleted = result.single()["deleted"]
                logger.info(f"Cleared {deleted} feedback nodes")
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to clear feedback: {e}")
            return 0


# Singleton instance
_feedback_metrics: Optional[FeedbackMetrics] = None


def get_feedback_metrics() -> FeedbackMetrics:
    """Get or create the singleton feedback metrics instance."""
    global _feedback_metrics
    if _feedback_metrics is None:
        _feedback_metrics = FeedbackMetrics()
    return _feedback_metrics
