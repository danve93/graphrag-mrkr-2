"""
Graph Editor API for manual curation and safety operations.
"""

from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from api.auth import require_admin
from core.graph_db import graph_db
from pydantic import BaseModel
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/snapshot")
async def get_graph_snapshot(current_user=Depends(require_admin)):
    """
    Export the entire graph as a JSON snapshot for backup.
    Returns a Cytoscape-compatible JSON {nodes: [], edges: []}.
    """
    try:
        # Use existing graph DB export functionality or implement new one
        snapshot = graph_db.export_graph_snapshot()
        return snapshot
    except Exception as e:
        logger.error(f"Snapshot failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restore")
async def restore_graph_snapshot(
    file: UploadFile = File(...),
    confirmation: str = Body(..., description="Must be 'DELETE' to confirm"),
    current_user=Depends(require_admin)
):
    """
    Disaster Recovery: Wipe the DB and restore from a JSON snapshot.
    REQUIRES 'confirmation' field to be 'DELETE'.
    """
    if confirmation != "DELETE":
        raise HTTPException(status_code=400, detail="Confirmation must be 'DELETE'")

    try:
        content = await file.read()
        data = json.loads(content)
        
        # Validation (Basic)
        if "nodes" not in data or "edges" not in data:
             raise HTTPException(status_code=400, detail="Invalid snapshot format. Missing 'nodes' or 'edges'.")

        # Perform the Restore
        graph_db.restore_graph_snapshot(data)
        
        return {"status": "success", "message": "Graph restored successfully from snapshot."}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")
    except Exception as e:
         logger.error(f"Restore failed: {e}")
         raise HTTPException(status_code=500, detail=str(e))
         logger.error(f"Restore failed: {e}")
         raise HTTPException(status_code=500, detail=str(e))

# Data Models for Phase 1
class EdgeCreateRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = {}

class EdgeDeleteRequest(BaseModel):
    source_id: str
    target_id: str
    relation_type: str

class NodeUpdateRequest(BaseModel):
    node_id: str
    properties: Dict[str, Any]

@router.post("/edge")
async def create_edge(
    request: EdgeCreateRequest,
    current_user=Depends(require_admin)
):
    """Create a new relationship (manual curation)."""
    try:
        success = graph_db.create_relationship(
            request.source_id, 
            request.target_id, 
            request.relation_type, 
            request.properties
        )
        if not success:
             raise HTTPException(status_code=404, detail="Source or Target node not found.")
        return {"status": "success", "message": "Edge created."}
    except Exception as e:
        logger.error(f"Create edge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/edge")
async def delete_edge(
    # Using Body for DELETE with JSON payload is standard in some REST designs but tricky in others.
    # FastAPI supports it if we declare it as Body, but fetch often strips it.
    # Safer to accept query params for simple deletes or use POST for 'actions'.
    # For now, let's try Body as standard client libs support it.
    request: EdgeDeleteRequest = Body(...),
    current_user=Depends(require_admin)
):
    """Remove a relationship (Prune)."""
    try:
        success = graph_db.delete_relationship(
            request.source_id,
            request.target_id,
            request.relation_type
        )
        if not success:
             raise HTTPException(status_code=404, detail="Edge not found.")
        return {"status": "success", "message": "Edge deleted."}
    except Exception as e:
        logger.error(f"Delete edge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/node")
async def update_node(
    request: NodeUpdateRequest,
    current_user=Depends(require_admin)
):
    """Update node properties (e.g. rename, change type)."""
    try:
        success = graph_db.update_node_properties(
            request.node_id,
            request.properties
        )
        if not success:
             raise HTTPException(status_code=404, detail="Node not found.")
        return {"status": "success", "message": "Node updated."}
    except Exception as e:
        logger.error(f"Update node failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class HealRequest(BaseModel):
    node_id: str

@router.post("/heal")
async def suggest_healing(
    request: HealRequest,
    current_user=Depends(require_admin)
):
    """
    AI Graph Healing: Suggest missing connections for a node using vector similarity.
    Returns value-added suggestions based on semantic similarity.
    """
    try:
        suggestions = graph_db.heal_node(request.node_id)
        return {"status": "success", "suggestions": suggestions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Healing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class MergeNodesRequest(BaseModel):
    target_id: str
    source_ids: List[str]

@router.post("/nodes/merge")
async def merge_nodes_endpoint(
    request: MergeNodesRequest,
    current_user=Depends(require_admin)
):
    """
    Merge multiple source nodes into a target node.
    Transfers relationships and deletes sources.
    """
    try:
        if not request.source_ids:
             raise HTTPException(status_code=400, detail="No source nodes provided.")
             
        if request.target_id in request.source_ids:
             raise HTTPException(status_code=400, detail="Target node cannot be in source list.")

        success = graph_db.merge_nodes(request.target_id, request.source_ids)
        if success:
            return {"status": "success", "message": "Nodes merged successfully."}
        else:
             raise HTTPException(status_code=500, detail="Merge operation returned false.")
             
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Merge nodes failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orphans")
async def get_orphan_nodes(current_user=Depends(require_admin)):
    """
    Detect 'orphan' nodes: entities not connected to the main graph component.
    Returns IDs of nodes that are either isolated or in small disconnected clusters.
    """
    try:
        orphans = graph_db.find_orphan_nodes()
        return {"status": "success", "orphan_ids": orphans}
    except Exception as e:
        logger.error(f"Orphan detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
