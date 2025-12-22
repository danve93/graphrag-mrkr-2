"""Pattern management API endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from core.chunk_pattern_store import get_pattern_store, ChunkPattern

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patterns", tags=["patterns"])


@router.get("")
async def list_patterns(
    enabled_only: bool = Query(False, description="Only return enabled patterns"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    include_builtin: bool = Query(True, description="Include built-in patterns"),
):
    """List all chunk patterns.
    
    Returns patterns sorted by built-in status, then usage count.
    """
    try:
        store = get_pattern_store()
        patterns = store.get_patterns(
            enabled_only=enabled_only,
            action=action,
            include_builtin=include_builtin,
        )
        return {
            "total": len(patterns),
            "patterns": [p.to_dict() for p in patterns],
        }
    except Exception as exc:
        logger.error("Failed to list patterns: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list patterns") from exc


@router.get("/{pattern_id}")
async def get_pattern(pattern_id: str):
    """Get a single pattern by ID."""
    try:
        store = get_pattern_store()
        pattern = store.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        return pattern.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get pattern %s: %s", pattern_id, exc)
        raise HTTPException(status_code=500, detail="Failed to get pattern") from exc


from api.models import CreatePatternRequest, UpdatePatternRequest

# ... imports ...


@router.post("")
async def create_pattern(request: CreatePatternRequest):
    """Create a new pattern.
    
    Request body validated by CreatePatternRequest model.
    """
    try:
        data = request.model_dump()
        store = get_pattern_store()
        pattern = ChunkPattern.from_dict(data)
        pattern.is_builtin = False  # User-created patterns are never built-in
        created = store.create_pattern(pattern)
        return {"status": "success", "pattern": created.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create pattern: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create pattern") from exc


@router.patch("/{pattern_id}")
async def update_pattern(pattern_id: str, request: UpdatePatternRequest):
    """Update an existing pattern.
    
    Partial updates supported via UpdatePatternRequest model.
    Built-in patterns can be disabled but not deleted.
    """
    try:
        data = request.model_dump(exclude_unset=True)
        store = get_pattern_store()
        updated = store.update_pattern(pattern_id, data)
        if not updated:
            raise HTTPException(status_code=404, detail="Pattern not found")
        return {"status": "success", "pattern": updated.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update pattern %s: %s", pattern_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update pattern") from exc


@router.delete("/{pattern_id}")
async def delete_pattern(pattern_id: str):
    """Delete a pattern.
    
    Built-in patterns cannot be deleted (disable them instead).
    """
    try:
        store = get_pattern_store()
        pattern = store.get_pattern(pattern_id)
        if not pattern:
            raise HTTPException(status_code=404, detail="Pattern not found")
        if pattern.is_builtin:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete built-in pattern. Disable it instead."
            )
        
        success = store.delete_pattern(pattern_id)
        if not success:
            raise HTTPException(status_code=500, detail="Delete failed")
        
        return {"status": "success", "deleted_id": pattern_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete pattern %s: %s", pattern_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete pattern") from exc


@router.post("/{pattern_id}/toggle")
async def toggle_pattern(pattern_id: str, enabled: bool = Query(...)):
    """Enable or disable a pattern."""
    try:
        store = get_pattern_store()
        updated = store.toggle_pattern(pattern_id, enabled)
        if not updated:
            raise HTTPException(status_code=404, detail="Pattern not found")
        return {"status": "success", "pattern": updated.to_dict()}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to toggle pattern %s: %s", pattern_id, exc)
        raise HTTPException(status_code=500, detail="Failed to toggle pattern") from exc


@router.get("/export/json")
async def export_patterns(
    include_builtin: bool = Query(False, description="Include built-in patterns in export"),
):
    """Export patterns as JSON for backup/sharing."""
    try:
        store = get_pattern_store()
        export_data = store.export_patterns(include_builtin=include_builtin)
        return JSONResponse(content=export_data)
    except Exception as exc:
        logger.error("Failed to export patterns: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to export patterns") from exc


@router.post("/import/json")
async def import_patterns(
    data: dict,
    overwrite: bool = Query(False, description="Overwrite existing patterns with same ID"),
):
    """Import patterns from JSON export.
    
    Request body should match export format.
    """
    try:
        store = get_pattern_store()
        results = store.import_patterns(data, overwrite=overwrite)
        return {
            "status": "success",
            "results": results,
        }
    except Exception as exc:
        logger.error("Failed to import patterns: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to import patterns") from exc
