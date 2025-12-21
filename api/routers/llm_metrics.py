"""
LLM Metrics API endpoints.

Provides access to LLM token usage analytics.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-usage", tags=["llm-metrics"])


class UsageSummary(BaseModel):
    """Overall LLM usage summary."""

    total_calls: int = Field(..., description="Total number of LLM calls")
    total_input_tokens: int = Field(..., description="Total input tokens")
    total_output_tokens: int = Field(..., description="Total output tokens")
    total_tokens: int = Field(..., description="Combined input + output tokens")
    avg_latency_ms: float = Field(..., description="Average latency in milliseconds")


class OperationBreakdown(BaseModel):
    """Usage breakdown by operation."""

    by_operation: Dict[str, Dict[str, int]] = Field(
        ..., description="Token usage grouped by operation type"
    )


class ProviderBreakdown(BaseModel):
    """Usage breakdown by provider."""

    by_provider: Dict[str, Dict[str, int]] = Field(
        ..., description="Token usage grouped by provider"
    )


class ModelBreakdown(BaseModel):
    """Usage breakdown by model."""

    by_model: Dict[str, Dict[str, int]] = Field(
        ..., description="Token usage grouped by model"
    )


class DocumentUsage(BaseModel):
    """Usage for a specific document."""

    document_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    first_call: Optional[str]
    last_call: Optional[str]


@router.get("/summary", response_model=UsageSummary)
async def get_usage_summary():
    """
    Get overall LLM usage summary.

    Returns total calls, tokens, and average latency across all operations.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_summary()
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-operation", response_model=OperationBreakdown)
async def get_usage_by_operation():
    """
    Get LLM usage breakdown by operation type.

    Returns token counts grouped by operation (e.g., ingestion.entity_extraction, rag.generation).
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {"by_operation": usage_tracker.get_by_operation()}
    except Exception as e:
        logger.error(f"Failed to get usage by operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-provider", response_model=ProviderBreakdown)
async def get_usage_by_provider():
    """
    Get LLM usage breakdown by provider.

    Returns token counts grouped by provider (openai, anthropic, mistral, etc.).
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {"by_provider": usage_tracker.get_by_provider()}
    except Exception as e:
        logger.error(f"Failed to get usage by provider: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-model", response_model=ModelBreakdown)
async def get_usage_by_model():
    """
    Get LLM usage breakdown by model.

    Returns token counts grouped by model (gpt-4o-mini, claude-sonnet-4-5, etc.).
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {"by_model": usage_tracker.get_by_model()}
    except Exception as e:
        logger.error(f"Failed to get usage by model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-document/{document_id}", response_model=DocumentUsage)
async def get_usage_by_document(document_id: str):
    """
    Get LLM usage for a specific document.

    Returns total tokens used during ingestion of this document.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_by_document(document_id)
    except Exception as e:
        logger.error(f"Failed to get usage for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_usage(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of recent records")
):
    """
    Get recent LLM usage records.

    Returns the most recent LLM calls with full details.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {"records": usage_tracker.get_recent(limit)}
    except Exception as e:
        logger.error(f"Failed to get recent usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/full-report")
async def get_full_report():
    """
    Get a complete usage report.

    Combines summary, operation breakdown, provider breakdown, and model breakdown.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {
            "summary": usage_tracker.get_summary(),
            "by_operation": usage_tracker.get_by_operation(),
            "by_provider": usage_tracker.get_by_provider(),
            "by_model": usage_tracker.get_by_model(),
        }
    except Exception as e:
        logger.error(f"Failed to get full report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-estimate")
async def get_cost_estimate():
    """
    Get estimated costs based on token usage.

    Uses default pricing per 1M tokens for common models.
    Returns total cost and breakdown by model.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_cost_estimate()
    except Exception as e:
        logger.error(f"Failed to get cost estimate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-trends")
async def get_time_trends(
    days: int = Query(default=7, ge=1, le=90, description="Number of days to analyze")
):
    """
    Get usage trends over time.

    Returns daily aggregates for the specified period and hourly data for last 24h.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_time_trends(days)
    except Exception as e:
        logger.error(f"Failed to get time trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/success-rate")
async def get_success_rate():
    """
    Get success/error rate statistics.

    Returns total calls, success rate percentage, and recent errors.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_success_rate()
    except Exception as e:
        logger.error(f"Failed to get success rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/efficiency")
async def get_efficiency_metrics():
    """
    Get token efficiency metrics.

    Returns input/output ratios, averages per call, and per-operation efficiency.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return usage_tracker.get_efficiency_metrics()
    except Exception as e:
        logger.error(f"Failed to get efficiency metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-conversation")
async def get_usage_by_conversation(
    limit: int = Query(default=20, ge=1, le=100, description="Max conversations to return")
):
    """
    Get LLM usage breakdown by conversation.

    Returns top conversations ordered by total token usage with cost estimates in EUR.
    """
    try:
        from core.llm_usage_tracker import usage_tracker

        return {"conversations": usage_tracker.get_by_conversation(limit)}
    except Exception as e:
        logger.error(f"Failed to get usage by conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
