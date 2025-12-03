"""
Classification configuration API endpoints.

Manages entity types, overrides, relationship suggestions, and Leiden parameters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

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
    # Invalidate response cache immediately at reindex request time to avoid stale responses
    try:
        from core.singletons import clear_response_cache
        clear_response_cache()
    except Exception:
        logger.warning("Failed to clear response cache at reindex trigger")
    
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
            # Best-effort: invalidate again inside background worker context
            try:
                from core.singletons import clear_response_cache
                clear_response_cache()
            except Exception:
                pass
            job_manager.set_job_started(job_id)
            logger.warning("Background reindex started")

            # Step 1: Get all document IDs
            try:
                with graph_db.session_scope() as session:
                    result = session.run("MATCH (d:Document) RETURN d.id as doc_id, d.filename as filename")
                    documents = [{"doc_id": record["doc_id"], "filename": record["filename"]} for record in result]
            except Exception as exc:
                # If Neo4j is unavailable, mark job failed and exit early
                try:
                    # avoid circular import if ServiceUnavailable not available here
                    from neo4j.exceptions import ServiceUnavailable
                    if isinstance(exc, ServiceUnavailable):
                        logger.error("Graph DB unavailable during reindex start: %s", exc)
                        job_manager.set_job_failed(job_id, "Graph database unavailable")
                        return
                except Exception:
                    pass
                # For other errors, surface and fail the job
                logger.error("Failed to list documents for reindex: %s", exc)
                job_manager.set_job_failed(job_id, str(exc))
                return

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

            # Invalidate response cache after reindex completes
            try:
                from core.singletons import clear_response_cache
                clear_response_cache()
            except Exception:
                pass

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


# ============================================================================
# Category Management Endpoints
# ============================================================================

class CategoryCreate(BaseModel):
    """Request model for creating a category."""
    name: str
    description: str
    keywords: list[str]
    patterns: list[str] | None = None
    parent_id: str | None = None


class CategoryResponse(BaseModel):
    """Response model for category data."""
    id: str
    name: str
    description: str
    keywords: list[str]
    patterns: list[str]
    approved: bool
    document_count: int
    children: list[str] | None = None
    created_at: str | None = None
    
class CategoryUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    keywords: list[str] | None = None
    patterns: list[str] | None = None
    parent_id: str | None = None



class QueryClassificationRequest(BaseModel):
    """Request model for query classification."""
    query: str


class QueryClassificationResponse(BaseModel):
    """Response model for query classification results."""
    classifications: list[Dict[str, Any]]


@router.post("/categories/generate")
async def generate_categories(
    max_categories: int = 10,
    sample_size: int = 100
) -> Dict[str, Any]:
    """
    Generate category taxonomy from corpus analysis using LLM.
    
    Args:
        max_categories: Maximum number of categories to generate (default: 10)
        sample_size: Number of documents to sample for analysis (default: 100)
        
    Returns:
        List of proposed categories with metadata (requires human approval)
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        categories = await manager.analyze_corpus_for_categories(max_categories, sample_size)
        return {"status": "success", "categories": categories}
    except Exception as e:
        logger.error(f"Category generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Category generation failed: {str(e)}")


@router.post("/categories")
async def create_category(category: CategoryCreate) -> CategoryResponse:
    """
    Create a new category.
    
    Categories start as unapproved and require explicit approval before use.
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        result = await manager.create_category(
            name=category.name,
            description=category.description,
            keywords=category.keywords,
            patterns=category.patterns,
            parent_id=category.parent_id,
            approved=False
        )
        return CategoryResponse(**result)
    except ValueError as ve:
        if str(ve) == "CATEGORY_NAME_CONFLICT":
            raise HTTPException(status_code=409, detail="Category name already exists")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Category creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Category creation failed: {str(e)}")


@router.get("/categories")
async def list_categories(approved_only: bool = True) -> list[CategoryResponse]:
    """
    List all categories.
    
    Args:
        approved_only: If true, only return approved categories (default: true)
        
    Returns:
        List of categories with metadata
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        categories = await manager.get_all_categories(approved_only=approved_only)
        formatted: list[CategoryResponse] = []
        for cat in categories:
            if isinstance(cat.get("created_at"), (dict,)):
                # defensive: if created_at serialized oddly
                cat["created_at"] = str(cat.get("created_at"))
            elif cat.get("created_at") is not None and not isinstance(cat.get("created_at"), str):
                cat["created_at"] = str(cat.get("created_at"))
            formatted.append(CategoryResponse(**cat))
        return formatted
    except Exception as e:
        logger.error(f"Failed to list categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")
    
@router.patch("/categories/{category_id}")
async def update_category(category_id: str, update: CategoryUpdate):
    try:
        from rag.nodes.category_manager import CategoryManager
        manager = CategoryManager()
        updated = await manager.update_category(
            category_id,
            {k: v for k, v in update.dict().items() if v is not None}
        )
        if not updated:
            raise HTTPException(status_code=404, detail="Category not found")
        return updated
    except ValueError as ve:
        if str(ve) == "CATEGORY_NAME_CONFLICT":
            raise HTTPException(status_code=409, detail="Category name already exists")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to update category {category_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update category: {e}")


@router.get("/categories/{category_id}")
async def get_category(category_id: str) -> CategoryResponse:
    """Get category by ID."""
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        category = await manager.get_category_by_id(category_id)
        if not category:
            raise HTTPException(status_code=404, detail="Category not found")
        if category.get("created_at") is not None and not isinstance(category.get("created_at"), str):
            category["created_at"] = str(category["created_at"])
        return CategoryResponse(**category)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get category: {str(e)}")


@router.post("/categories/{category_id}/approve")
async def approve_category(category_id: str) -> Dict[str, str]:
    """
    Approve a category for production use.
    
    Only approved categories are used for query routing and document filtering.
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        success = await manager.approve_category(category_id)
        if not success:
            raise HTTPException(status_code=404, detail="Category not found")
        return {"status": "success", "message": f"Category {category_id} approved"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve category: {str(e)}")


@router.delete("/categories/{category_id}")
async def delete_category(category_id: str) -> Dict[str, str]:
    """Delete a category and its relationships."""
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        await manager.delete_category(category_id)
        return {"status": "success", "message": f"Category {category_id} deleted"}
    except Exception as e:
        logger.error(f"Failed to delete category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete category: {str(e)}")


@router.post("/categories/classify")
async def classify_query(request: QueryClassificationRequest) -> QueryClassificationResponse:
    """
    Classify a user query to determine relevant categories.
    
    Returns list of categories with confidence scores for query routing.
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        classifications = await manager.classify_query(request.query)
        results = [
            {"category_id": cat_id, "confidence": conf}
            for cat_id, conf in classifications
        ]
        return QueryClassificationResponse(classifications=results)
    except Exception as e:
        logger.error(f"Query classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query classification failed: {str(e)}")


@router.post("/categories/auto-assign")
async def auto_categorize_documents(batch_size: int = 10) -> Dict[str, Any]:
    """
    Automatically categorize uncategorized documents.
    
    Args:
        batch_size: Number of documents to process (default: 10)
        
    Returns:
        Statistics about categorization results
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        stats = await manager.auto_categorize_documents(batch_size=batch_size)
        return {"status": "success", "statistics": stats}
    except Exception as e:
        logger.error(f"Auto-categorization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-categorization failed: {str(e)}")


@router.post("/documents/{document_id}/categories/{category_id}")
async def assign_document_to_category(
    document_id: str,
    category_id: str,
    confidence: float = 1.0
) -> Dict[str, str]:
    """
    Manually assign a document to a category.
    
    Args:
        document_id: Document ID
        category_id: Category ID
        confidence: Confidence score (0-1, default: 1.0)
        
    Returns:
        Success status
    """
    from rag.nodes.category_manager import CategoryManager
    
    manager = CategoryManager()
    try:
        success = await manager.assign_document_to_category(
            document_id=document_id,
            category_id=category_id,
            confidence=confidence,
            auto_assigned=False
        )
        if not success:
            raise HTTPException(status_code=404, detail="Document or category not found")
        return {"status": "success", "message": f"Document {document_id} assigned to category {category_id}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to assign document to category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to assign document: {str(e)}")
