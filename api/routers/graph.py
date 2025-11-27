"""Graph visualization endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from neo4j.exceptions import ServiceUnavailable

from api.models import GraphResponse
from core.graph_db import graph_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/clustered", response_model=GraphResponse)
async def get_clustered_graph(
    community_id: Optional[int] = Query(
        default=None, description="Filter graph by a specific community id"
    ),
    node_type: Optional[str] = Query(
        default=None, description="Restrict nodes to a specific entity type"
    ),
    level: Optional[int] = Query(
        default=None, description="Community detection level for the selection"
    ),
    document_id: Optional[str] = Query(
        default=None, description="Filter graph to entities from a specific document"
    ),
    limit: int = Query(
        default=300,
        gt=0,
        le=1000,
        description="Maximum number of seed nodes to include before expansion",
    ),
) -> GraphResponse:
    """Return clustered graph JSON for the UI."""

    try:
        graph = graph_db.get_clustered_graph(
            community_id=community_id,
            node_type=node_type,
            level=level,
            limit=limit,
            document_id=document_id,
        )
        return GraphResponse(**graph)
    except Exception as exc:  # pragma: no cover - thin HTTP wrapper
        # Let ServiceUnavailable bubble to the global handler
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.exception("Failed to fetch clustered graph: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load graph data")


@router.get("/communities")
async def get_communities(
    level: Optional[int] = Query(default=None, description="Community detection level (required)"),
) -> list:
    """Return communities and their member entities for a given clustering level."""

    if level is None:
        raise HTTPException(status_code=400, detail="Query parameter 'level' is required")

    try:
        communities = graph_db.get_communities_for_level(level)
        return communities
    except Exception as exc:  # pragma: no cover - thin HTTP wrapper
        # Let ServiceUnavailable bubble to the global handler
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.exception("Failed to fetch communities for level %s: %s", level, exc)
        raise HTTPException(status_code=500, detail="Failed to load communities")
