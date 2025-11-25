"""
Classification configuration API endpoints.

Manages entity types, overrides, relationship suggestions, and Leiden parameters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from api.models import ReindexResult, ReindexJobResponse, ReindexJobStatus
from api import job_manager
from api import reindex_tasks
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "classification_config.json"


class ClassificationConfig(BaseModel):
    """Classification configuration schema."""
    entity_types: list[str]
    entity_type_overrides: Dict[str, str]
    relationship_suggestions: list[str]
    low_value_patterns: list[str]
    leiden_parameters: Dict[str, Any]


def load_config() -> Dict[str, Any]:
    """Load classification configuration from JSON file."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Classification config not found: {CONFIG_PATH}")
        raise HTTPException(status_code=500, detail="Classification configuration file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in classification config: {e}")
        raise HTTPException(status_code=500, detail="Invalid classification configuration file")


def save_config(config: Dict[str, Any]) -> None:
    """Save classification configuration to JSON file."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info("Classification configuration saved successfully")
    except Exception as e:
        logger.error(f"Failed to save classification config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@router.get("/config")
async def get_classification_config() -> ClassificationConfig:
    """
    Get current classification configuration.
    
    Returns entity types, overrides, relationship suggestions, 
    low-value patterns, and Leiden parameters.
    """
    config = load_config()
    return ClassificationConfig(**config)


@router.post("/config")
async def update_classification_config(config: ClassificationConfig) -> Dict[str, str]:
    """
    Update classification configuration.
    
    Updates the configuration file with new entity types, overrides, 
    relationships, and Leiden parameters.
    
    Note: Changes take effect on next document ingestion or after reindex.
    """
    # Use Pydantic v2 `model_dump` for serialization
    save_config(config.model_dump())
    return {"status": "success", "message": "Classification configuration updated successfully"}


@router.post("/reindex", response_model=ReindexJobResponse, status_code=202)
async def trigger_reindex(confirmation: Dict[str, bool], background_tasks: BackgroundTasks) -> ReindexJobResponse:
    """
    Trigger dangerous reindex operation.
    
    Re-runs entity extraction and clustering on all documents using
    the current classification configuration.
    
    **WARNING**: This is a destructive operation that will:
    - Clear existing entities and relationships
    - Re-extract entities from all documents
    - Re-run clustering algorithm
    - May take significant time depending on document count
    
    Requires explicit confirmation: {"confirmed": true}
    """
    if not confirmation.get("confirmed", False):
        raise HTTPException(
            status_code=400,
            detail="Reindex operation requires explicit confirmation"
        )
    
    from config.settings import settings
    from core.graph_db import graph_db
    from ingestion.document_processor import DocumentProcessor
    
    if not settings.enable_entity_extraction:
        raise HTTPException(
            status_code=400,
            detail="Entity extraction is disabled in settings"
        )
    
    # Create a job record and schedule the background work
    job_id = str(uuid.uuid4())
    job_manager.create_job(job_id, message="queued")

    def background_reindex(job_id: str):
        try:
            job_manager.set_job_started(job_id)
            logger.warning("Background reindex started")

            # Step 1: Get all document IDs
            with graph_db.driver.session() as session:  # type: ignore
                result = session.run("MATCH (d:Document) RETURN d.id as doc_id, d.filename as filename")
                documents = [{"doc_id": record["doc_id"], "filename": record["filename"]} for record in result]

            if not documents:
                res = {
                    "status": "success",
                    "message": "No documents found to reindex",
                    "documents_processed": 0,
                    "entities_cleared": 0,
                }
                job_manager.set_job_result(job_id, res, status="success")
                return

            total_cleared = 0
            for doc in documents:
                try:
                    result = graph_db.reset_document_entities(doc["doc_id"])
                    total_cleared += result.get("entities_removed", 0)
                except Exception as e:
                    logger.error(f"Failed to clear entities for {doc['doc_id']}: {e}")

            processor = DocumentProcessor()
            extraction_result = processor.extract_entities_for_all_documents()

            if extraction_result is None:
                res = {
                    "status": "partial",
                    "message": f"Cleared {total_cleared} entities, but entity extraction is disabled",
                    "documents_processed": 0,
                    "entities_cleared": total_cleared,
                }
                job_manager.set_job_result(job_id, res, status="partial")
                return

            try:
                graph_db.create_all_entity_similarities()
            except Exception as e:
                logger.warning(f"Failed to create entity similarities: {e}")

            clustering_result = None
            try:
                from core.graph_clustering import run_leiden_clustering
                clustering_result = run_leiden_clustering()
            except Exception as e:
                logger.warning(f"Clustering failed or not available: {e}")

            res = {
                "status": "success",
                "message": f"Reindex complete: cleared {total_cleared} entities, processed {extraction_result.get('processed_documents', 0)} documents",
                "documents_processed": extraction_result.get("processed_documents", 0),
                "entities_cleared": total_cleared,
                "extraction_result": extraction_result,
                "clustering_result": clustering_result,
            }
            job_manager.set_job_result(job_id, res, status="success")

        except Exception as e:
            logger.error(f"Background reindex failed: {e}", exc_info=True)
            job_manager.set_job_failed(job_id, str(e))

    # If Redis queueing is configured prefer enqueue-only behavior so external
    # workers process jobs. Otherwise fall back to in-process BackgroundTasks.
    import os

    if os.environ.get("REDIS_URL"):
        # `job_manager.create_job` already enqueues the job in Redis.
        logger.info("REDIS_URL detected: job %s enqueued for external worker", job_id)
    else:
        background_tasks.add_task(reindex_tasks.run_reindex_job, job_id)

    status_url = f"/api/classification/reindex/{job_id}"
    return ReindexJobResponse(job_id=job_id, status_url=status_url, status="queued")



@router.get("/reindex/{job_id}")
async def get_reindex_job_status(job_id: str) -> ReindexJobStatus:
    rec = job_manager.get_job(job_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Job not found")
    # Map the stored dict into typed model; extraction of result -> ReindexResult if present
    result = rec.get("result")
    reindex_result = None
    if result:
        # The stored result is a plain dict matching ReindexResult fields
        # Use `model_fields` (Pydantic v2) instead of deprecated `__fields__`.
        allowed = set(ReindexResult.model_fields.keys())
        reindex_result = ReindexResult(**{k: v for k, v in result.items() if k in allowed})
    return ReindexJobStatus(
        job_id=rec.get("job_id"),
        status=rec.get("status"),
        message=rec.get("message"),
        created_at=rec.get("created_at"),
        started_at=rec.get("started_at"),
        finished_at=rec.get("finished_at"),
        result=reindex_result,
    )
