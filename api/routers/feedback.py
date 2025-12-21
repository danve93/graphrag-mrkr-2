"""
API endpoints for user feedback and adaptive routing.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from rag.nodes.adaptive_router import get_feedback_learner, reset_feedback_learner
from core.feedback_metrics import get_feedback_metrics
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])


class FeedbackRequest(BaseModel):
    """Request for submitting user feedback."""
    query: str = Field(..., description="Original user query")
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Message identifier")
    rating: int = Field(..., description="1 for positive (thumbs up), -1 for negative (thumbs down)")
    routing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Routing metadata (query_type, routed_to, confidence, etc.)"
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback."""
    success: bool
    feedback_id: str
    weights_updated: bool
    current_weights: Dict[str, float]
    learning_active: bool


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Submit user feedback for a routing decision.
    
    Args:
        request: Feedback submission with rating and context
        
    Returns:
        Feedback response with updated weights
    """
    try:
        if not settings.enable_adaptive_routing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Adaptive routing is not enabled"
            )
        
        # Validate rating
        if request.rating not in [1, -1]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rating must be 1 (positive) or -1 (negative)"
            )
        
        # Get learner and metrics
        learner = get_feedback_learner()
        metrics = get_feedback_metrics()
        
        # Get current weights before recording feedback
        weights_before = learner.get_weights()
        
        # Record feedback in learner (adjusts weights if enough samples)
        learner.record_feedback(
            query=request.query,
            session_id=request.session_id,
            message_id=request.message_id,
            rating=request.rating,
            routing_info=request.routing_info
        )
        
        # Get updated weights
        weights_after = learner.get_weights()
        weights_updated = weights_before != weights_after
        
        # Store in Neo4j
        feedback_id = metrics.store_feedback(
            query=request.query,
            session_id=request.session_id,
            message_id=request.message_id,
            rating=request.rating,
            routing_info=request.routing_info,
            weights_used=weights_before
        )
        
        # Record feedback for Routing Accuracy metric
        try:
            from core.routing_metrics import routing_metrics
            routing_metrics.record_user_feedback(correct=(request.rating == 1))
        except Exception as e:
            logger.warning(f"Failed to record routing accuracy feedback: {e}")
        
        # Get learner metrics
        learner_metrics = learner.get_metrics()
        
        return FeedbackResponse(
            success=True,
            feedback_id=feedback_id,
            weights_updated=weights_updated,
            current_weights=weights_after,
            learning_active=learner_metrics["learning_active"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feedback submission failed: {str(e)}"
        )


@router.get("/feedback/metrics")
async def get_metrics(days: int = 7) -> Dict[str, Any]:
    """
    Get feedback and learning metrics.
    
    Args:
        days: Number of days to include in metrics (default: 7)
        
    Returns:
        Combined metrics from learner and storage
    """
    try:
        if not settings.enable_adaptive_routing:
            return {
                "enabled": False,
                "message": "Adaptive routing is not enabled"
            }
        
        # Get learner metrics (in-memory)
        learner = get_feedback_learner()
        learner_metrics = learner.get_metrics()
        
        # Get storage metrics (Neo4j)
        metrics = get_feedback_metrics()
        storage_metrics = metrics.get_metrics_summary(days=days)
        
        # Combine metrics
        return {
            "enabled": True,
            "learner": learner_metrics,
            "storage": storage_metrics,
            "settings": {
                "learning_rate": settings.adaptive_learning_rate,
                "min_samples": settings.adaptive_min_samples,
                "weight_min": settings.adaptive_weight_min,
                "weight_max": settings.adaptive_weight_max,
                "decay_factor": settings.adaptive_decay_factor
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics retrieval failed: {str(e)}"
        )


@router.get("/feedback/weights")
async def get_weights() -> Dict[str, Any]:
    """
    Get current adaptive weights.
    
    Returns:
        Current weights and metadata
    """
    try:
        if not settings.enable_adaptive_routing:
            return {
                "enabled": False,
                "weights": {
                    "chunk_weight": 0.5,
                    "entity_weight": 0.3,
                    "path_weight": 0.2
                },
                "source": "default"
            }
        
        learner = get_feedback_learner()
        weights = learner.get_weights()
        metrics = learner.get_metrics()
        
        return {
            "enabled": True,
            "weights": weights,
            "source": "adaptive" if metrics["learning_active"] else "default",
            "learning_active": metrics["learning_active"],
            "total_feedback": metrics["total_feedback"],
            "accuracy": metrics["accuracy"],
            "convergence": metrics["convergence"],
            "last_updated": metrics["last_updated"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get weights: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weight retrieval failed: {str(e)}"
        )


@router.post("/feedback/reset")
async def reset_feedback() -> Dict[str, str]:
    """
    Reset all feedback learning state.
    
    Returns:
        Success message
    """
    try:
        if not settings.enable_adaptive_routing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Adaptive routing is not enabled"
            )
        
        # Reset learner
        reset_feedback_learner()
        
        # Optionally clear Neo4j storage
        metrics = get_feedback_metrics()
        deleted = metrics.clear_feedback()
        
        return {
            "message": f"Feedback learning state reset successfully. Cleared {deleted} feedback records."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {str(e)}"
        )


@router.get("/feedback/recent")
async def get_recent_feedback(limit: int = 50) -> Dict[str, Any]:
    """
    Get recent feedback events.
    
    Args:
        limit: Maximum number of events to return (default: 50, max: 200)
        
    Returns:
        List of recent feedback events
    """
    try:
        if not settings.enable_adaptive_routing:
            return {
                "enabled": False,
                "feedback": []
            }
        
        # Clamp limit
        limit = max(1, min(200, limit))
        
        metrics = get_feedback_metrics()
        feedback_list = metrics.get_recent_feedback(limit=limit)
        
        return {
            "enabled": True,
            "feedback": feedback_list,
            "count": len(feedback_list)
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recent feedback retrieval failed: {str(e)}"
        )
