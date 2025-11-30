"""Graph visualization endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from neo4j.exceptions import ServiceUnavailable, TransientError

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
        default=100,
        gt=0,
        le=500,
        description="Maximum number of seed nodes to include before expansion",
    ),
) -> GraphResponse:
    """Return clustered graph JSON for the UI."""

    # For document-scoped graphs, avoid expensive pre-counts (which can
    # themselves trigger DB memory pressure) and conservatively cap the
    # number of seed nodes. This prevents the backend from constructing
    # very large transactions when the document contains thousands of
    # entities. Users can still view smaller subsets via the Communities
    # tab or by filtering by `community_id`.
    if document_id:
        # Conservative default for document-scoped visualizations
        limit = min(limit, 50)
    else:
        # For global graph queries, enforce a low default when unfiltered
        if community_id is None and node_type is None:
            limit = min(limit, 50)

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
        # If Neo4j transient memory errors occur, return a clear 503 so the UI can
        # surface a friendly message and avoid downstream parsing errors.
        if isinstance(exc, TransientError):
            logger.exception("Neo4j transient error while fetching clustered graph: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=(
                    "Graph database memory threshold reached while building the clustered graph. "
                    "Try reducing the 'limit' query parameter (e.g. ?limit=50) or increase Neo4j transaction memory."
                ),
            )

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
