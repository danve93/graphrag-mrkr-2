"""Graph visualization endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

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
        )
        return GraphResponse(**graph)
    except Exception as exc:  # pragma: no cover - thin HTTP wrapper
        logger.exception("Failed to fetch clustered graph: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load graph data")
