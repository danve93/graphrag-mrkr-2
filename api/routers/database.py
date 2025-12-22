"""
Database router for managing documents and database operations.
"""

import asyncio
import hashlib
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile, Depends, Query

from api.models import (
    DatabaseStats,
    DocumentUploadResponse,
    StagedDocument,
    StageDocumentResponse,
    ProcessProgress,
    ProcessDocumentsRequest,
    ProcessingSummary,
    FolderListResponse,
    FolderSummary,
    CreateFolderRequest,
    RenameFolderRequest,
    MoveDocumentRequest,
    ReorderDocumentsRequest,
)
from config.settings import settings
from core.graph_db import graph_db
from neo4j.exceptions import ServiceUnavailable
from ingestion.document_processor import EntityExtractionState, document_processor
from core.singletons import get_response_cache, get_blocking_executor, SHUTTING_DOWN
from core.cache_metrics import cache_metrics
from api.auth import require_admin
from core.chunk_pattern_learner import get_pattern_learner

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for staged documents and processing progress
# In production, you'd use Redis or a database
_staged_documents: Dict[str, StagedDocument] = {}
_progress_percentage: Dict[str, ProcessProgress] = {}

_processing_queue: List[str] = []
_queue_lock = asyncio.Lock()
_processing_lock = asyncio.Lock()
_queue_event = asyncio.Event()
_processing_worker: Optional[asyncio.Task] = None  # type: ignore[var-annotated]
_global_processing_state: Dict[str, Any] = {
    "is_processing": False,
    "current_file_id": None,
    "current_document_id": None,
    "current_filename": None,
    "current_stage": None,
    "progress_percentage": 0.0,
    "queue_length": 0,
}
_active_document_jobs: Dict[str, str] = {}

_ENTITY_STAGE_FRACTIONS = {
    EntityExtractionState.STARTING: 0.05,
    EntityExtractionState.LLM_EXTRACTION: 0.25,
    EntityExtractionState.EMBEDDING_GENERATION: 0.45,
    EntityExtractionState.DATABASE_OPERATIONS: 0.70,
    EntityExtractionState.CLUSTERING: 0.80,
    EntityExtractionState.VALIDATION: 0.90,
    EntityExtractionState.COMPLETED: 1.0,
    EntityExtractionState.ERROR: 1.0,
}


def _calculate_overall_progress(progress: ProcessProgress) -> float:
    """
    Compute overall progress percentage blending chunk/entity phases.
    
    Progress Weights (Issue S3 - Refined):
    - Phase 1: Preparation (0-20%)
      - Classification: 0-4%
      - Content Filtering: 4-8%
      - Chunking: 8-14%
      - Summarization: 14-20%
    - Phase 2: Embedding (20-80%)
      - Driven by chunk_progress (0.0 to 1.0)
    - Phase 3: Entity Extraction (80-100%)
      - Driven by entity_progress (0.0 to 1.0)
    """
    stage = (progress.stage or "").lower()
    
    # Phase 0: Conversion (0-4%) - Document conversion/extraction stage
    if stage == "conversion":
        # Conversion: 0-2%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 0.0 + (sub_progress * 2.0) if sub_progress else 1.0
    
    if stage == "post_conversion":
        # Post-conversion: 2-4%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 2.0 + (sub_progress * 2.0) if sub_progress else 3.0
    
    # Phase 1: Preparation (4-20%) with refined sub-stage interpolation
    # Issue S3: Use fractional progress within sub-stages when available
    if stage == "classification":
        # Classification: 4-8%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 4.0 + (sub_progress * 4.0) if sub_progress else 6.0
    
    if stage == "content_filtering":
        # Content filtering: 8-12%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 8.0 + (sub_progress * 4.0) if sub_progress else 10.0
    
    if stage == "chunking":
        # Chunking: 12-16%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 12.0 + (sub_progress * 4.0) if sub_progress else 14.0
    
    if stage == "summarization":
        # Summarization: 16-20%
        sub_progress = getattr(progress, 'sub_progress', 0.0) or 0.0
        return 16.0 + (sub_progress * 4.0) if sub_progress else 18.0
        
    # Phase 2: Embedding (20-80%)
    # This phase is driven by chunk_progress (0.0 to 1.0)
    if stage == "embedding" or (not progress.entity_progress and progress.chunk_progress > 0):
        chunk_fraction = progress.chunk_progress or 0.0
        return 20.0 + (chunk_fraction * 60.0)

    # Phase 3: Entity Extraction (80-100%)
    # This phase is driven by entity_progress (0.0 to 1.0)
    # If we have entity progress, we assume embedding is done (100% of 60% = 60%, plus 20% base = 80%)
    if progress.entity_progress and progress.entity_progress > 0:
        entity_fraction = progress.entity_progress or 0.0
        return 80.0 + (entity_fraction * 20.0)

    # Fallback/Default
    return 0.0


def _invalidate_document_cache(document_id: str) -> None:
    if not getattr(settings, "enable_caching", True):
        return
    try:
        cache = get_response_cache()
        for key in (f"document_summary:{document_id}", f"document_metadata:{document_id}"):
            cache.delete(key)
        try:
            cache_metrics.record_response_invalidation()
        except Exception:
            pass
    except Exception:
        # Cache invalidation should never block folder operations.
        pass


def _update_queue_positions() -> None:
    """Update queue position hints for pending progress entries."""

    for idx, file_id in enumerate(_processing_queue):
        progress = _progress_percentage.get(file_id)
        if progress:
            progress.queue_position = idx

    _global_processing_state["queue_length"] = len(_processing_queue) + (
        1 if _global_processing_state.get("is_processing") else 0
    )


def _sync_progress_from_neo4j(progress: ProcessProgress) -> None:
    """Sync the processing stage from Neo4j document node to in-memory progress.
    
    This ensures early stages (conversion, post_conversion, classification, etc.)
    are visible in the UI even when the background worker hasn't called the callback yet.
    """
    if not progress.document_id:
        return
    
    try:
        # Query the document's processing stage directly from Neo4j
        doc_props = graph_db.get_document_metadata(progress.document_id)
        if not doc_props:
            return
            
        neo4j_stage = doc_props.get("processing_stage")
        
        if neo4j_stage:
            # Update the stage if Neo4j has a more specific/different stage
            # Prioritize Neo4j stage for early stages that the callback might miss
            early_stages = {"conversion", "post_conversion", "classification", "content_filtering"}
            if neo4j_stage in early_stages or progress.stage in {None, "Starting", "chunking"}:
                progress.stage = neo4j_stage
            
        # Recalculate overall progress with updated stage
        progress.progress_percentage = _calculate_overall_progress(progress)
        
    except Exception as e:
        # Don't let sync failures break progress reporting
        logger.debug(f"Failed to sync progress from Neo4j for {progress.document_id}: {e}")



def _should_update_document_status(doc_id: str, job_id: Optional[str]) -> bool:
    if not job_id:
        return True
    active_job = _active_document_jobs.get(doc_id)
    return active_job is None or active_job == job_id


def _mark_document_status(
    doc_id: str,
    status: str,
    stage: str,
    progress_value: float,
    job_id: Optional[str] = None,
) -> None:
    """Persist processing status directly on the document node."""

    if not _should_update_document_status(doc_id, job_id):
        logger.debug(
            "Skipping status update for %s from job %s (active=%s)",
            doc_id,
            job_id,
            _active_document_jobs.get(doc_id),
        )
        return

    try:
        graph_db.create_document_node(
            doc_id,
            {
                "processing_status": status,
                "processing_stage": stage,
                "progress_percentage": progress_value,
                "processing_updated_at": time.time(),
            },
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Failed to update processing metadata for %s: %s", doc_id, exc)


async def _cancel_existing_document_jobs(
    document_id: Optional[str],
    replacement_file_id: Optional[str] = None,
) -> None:
    if not document_id:
        return

    if replacement_file_id:
        _active_document_jobs[document_id] = replacement_file_id

    existing_file_ids = {
        fid
        for fid, staged in _staged_documents.items()
        if staged.document_id == document_id
    }
    existing_file_ids.update(
        fid
        for fid, progress in _progress_percentage.items()
        if progress.document_id == document_id
    )

    if not existing_file_ids:
        return

    for fid in existing_file_ids:
        progress = _progress_percentage.get(fid)
        if progress:
            progress.cancelled = True
            progress.status = "cancelled"
            progress.stage = "cancelled"
            progress.error = "superseded_by_new_upload"
        _staged_documents.pop(fid, None)
        _progress_percentage.pop(fid, None)

    async with _queue_lock:
        for fid in existing_file_ids:
            if fid in _processing_queue:
                _processing_queue.remove(fid)

    _update_queue_positions()

    try:
        _mark_document_status(
            document_id,
            "cancelled",
            "cancelled",
            0.0,
            job_id=replacement_file_id,
        )
    except Exception:
        pass


def restore_processing_state() -> int:
    """
    Restore processing state from Neo4j on startup (Issue #2).
    
    Reloads any documents that were staged or in-progress during a previous
    shutdown back into the in-memory queue for processing.
    
    Also recovers direct uploads by scanning data/staged_uploads for files.
    
    Returns:
        Number of documents restored to the queue.
    """
    try:
        staged_docs = graph_db.get_staged_documents()
        if not staged_docs:
            logger.info("No staged documents to restore from database")
            return 0

        # Build a map of files in staged_uploads for recovery
        staged_uploads_dir = Path("data/staged_uploads")
        staged_files_by_id = {}
        if staged_uploads_dir.exists():
            for f in staged_uploads_dir.iterdir():
                if f.is_file():
                    # Files are named: {file_id}_{filename}
                    # Try to extract doc_id from the path
                    parts = f.name.split("_", 1)
                    if len(parts) >= 1:
                        potential_id = parts[0]
                        staged_files_by_id[potential_id] = str(f)

        restored_count = 0
        for doc in staged_docs:
            doc_id = doc.get("id")
            if not doc_id:
                continue

            file_id = doc_id  # Use doc_id as file_id for consistency

            # Skip if already in memory
            if file_id in _staged_documents:
                continue

            # Get file path - try persisted path first, then scan staged_uploads
            file_path = doc.get("file_path") or ""
            if not file_path or not Path(file_path).exists():
                # Try to recover from staged_uploads
                recovered_path = staged_files_by_id.get(doc_id)
                if recovered_path and Path(recovered_path).exists():
                    logger.info(f"Recovered file path for {doc_id} from staged_uploads: {recovered_path}")
                    file_path = recovered_path
                    # Update the document node with corrected path
                    try:
                        graph_db.create_document_node(doc_id, {"file_path": file_path})
                    except Exception:
                        pass
                else:
                    # No file available - skip (will be handled by _cleanup_stale_jobs)
                    logger.debug(f"Skipping restore for {doc_id}: no file path available")
                    continue

            # Recreate StagedDocument from persisted data
            staged_doc = StagedDocument(
                file_id=file_id,
                filename=doc.get("original_filename") or doc.get("filename") or "unknown",
                file_path=file_path,
                file_size=0,  # Not stored in graph, will be recalculated
                mime_type=doc.get("mime_type") or "application/octet-stream",
                hash=doc_id,  # Use doc_id as hash since original hash isn't stored
                hashtags=doc.get("hashtags") or [],
            )
            _staged_documents[file_id] = staged_doc

            # Recreate ProcessProgress from persisted data
            status = doc.get("processing_status", "staged")
            stage = doc.get("processing_stage", "queued")
            progress = doc.get("progress_percentage", 0.0)

            _progress_percentage[file_id] = ProcessProgress(
                file_id=file_id,
                status=status if status in ["pending", "processing"] else "pending",
                stage=stage,
                progress_percentage=progress,
                message=f"Restored from previous session ({stage})",
            )

            # Add to processing queue if not completed
            if status not in ["complete", "completed", "failed"]:
                if file_id not in _processing_queue:
                    _processing_queue.append(file_id)

            restored_count += 1
            logger.info(f"Restored staged document: {file_id} (status={status}, stage={stage})")

        if restored_count > 0:
            _update_queue_positions()
            logger.info(f"Restored {restored_count} documents to processing queue")

        return restored_count

    except Exception as e:
        logger.error(f"Failed to restore processing state: {e}")
        return 0


async def _ensure_worker_running() -> None:
    """Guarantee that the background worker is alive to process the queue."""

    global _processing_worker

    if _processing_worker is None or _processing_worker.done():
        loop = asyncio.get_running_loop()
        _processing_worker = loop.create_task(_processing_worker_loop())


def cleanup_orphaned_staged_files() -> int:
    """
    Remove files in data/staged_uploads that are not referenced in the database.
    Returns the count of deleted files.
    """
    if not getattr(settings, "enable_stale_job_cleanup", True):
        logger.info("Stale job cleanup disabled by configuration.")
        return 0

    try:
        staging_dir = Path("data/staged_uploads")
        if not staging_dir.exists():
            return 0

        # Get all Document file paths from Neo4j
        # Use a list of paths directly from properties
        with graph_db.session_scope() as session:
            result = session.run("MATCH (d:Document) WHERE d.file_path IS NOT NULL RETURN d.file_path as path")
            registered_paths = {record["path"] for record in result}

        count = 0
        now = time.time()
        # Issue #13: 1 hour grace period to prevent deleting active uploads 
        # that haven't been committed to DB yet
        GRACE_PERIOD_SECONDS = 3600

        for file_path in staging_dir.iterdir():
            if file_path.is_file():
                # Normalize path for comparison (Neo4j paths are strings)
                if str(file_path) not in registered_paths:
                    # Check modification time
                    mtime = file_path.stat().st_mtime
                    if (now - mtime) < GRACE_PERIOD_SECONDS:
                        logger.debug(f"Skipping potential orphan {file_path.name}: too recent ({int(now-mtime)}s)")
                        continue

                    logger.warning("Deleting orphaned staged file: %s", file_path)
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception as e:
                        logger.error("Failed to delete orphaned file %s: %s", file_path, e)
        
        if count > 0:
            logger.info("Cleaned up %d orphaned staged files", count)
        return count
    except Exception as e:
        logger.error("Error during orphaned file cleanup: %s", e)
        return 0


async def _processing_worker_loop() -> None:
    """Continuously drain the processing queue in FIFO order.
    
    Design Note (Issue #5 - Race Condition Handling):
    This worker implements a single-worker, FIFO processing model. The design
    intentionally reads the queue head inside _queue_lock but processes the job
    outside of it (to allow new items to be queued during processing).
    
    The _processing_lock ensures only one job runs at a time. The single-worker
    assumption is critical: if multiple workers were started, they could race
    to process the same job. This is enforced by _ensure_worker_running().
    
    If multi-worker support is ever needed, convert to atomic pop-and-process
    or use a proper distributed queue (Redis, etc).
    """
    # Issue #5: Verify single-worker invariant
    global _processing_worker
    assert _processing_worker is not None, "Worker loop running without registration"

    while True:
        await _queue_event.wait()

        while True:
            async with _queue_lock:
                if not _processing_queue:
                    _queue_event.clear()
                    break
                file_id = _processing_queue[0]

            await _process_single_job(file_id)

        if not _queue_event.is_set():
            break

        if not _queue_event.is_set():
            break


def _estimate_total_chunks_for_path(file_path: Path) -> int:
    """Best-effort chunk count estimation to drive progress UI."""

    try:
        if not file_path.exists():
            return 0

        from core.chunking import document_chunker  # Local import to avoid cycles
        from ingestion.loaders.image_loader import ImageLoader
        from ingestion.loaders.pdf_loader import PDFLoader

        loader = document_processor.loaders.get(file_path.suffix.lower())
        if not loader:
            return 0

        if isinstance(loader, ImageLoader):
            result = loader.load_with_metadata(file_path)
            content = result.get("content") if result else ""
        elif isinstance(loader, PDFLoader):
            result = loader.load_with_metadata(file_path)
            content = result.get("content") if result else ""
        else:
            content = loader.load(file_path)

        if not content:
            return 0

        temp_chunks = document_chunker.chunk_text(
            content,
            f"preview_{file_path.stem}",
            source_metadata={"file_extension": file_path.suffix.lower()},
        )
        return len(temp_chunks)
    except Exception as exc:  # pragma: no cover - advisory logging
        logger.debug("Failed to estimate chunks for %s: %s", file_path, exc)
        return 0


async def _process_single_job(file_id: str) -> None:
    """Process a single staged job while enforcing sequential execution."""

    async with _processing_lock:
        staged_doc = _staged_documents.get(file_id)
        progress = _progress_percentage.get(file_id)

        if staged_doc is None:
            async with _queue_lock:
                if file_id in _processing_queue:
                    _processing_queue.remove(file_id)
                    _update_queue_positions()
            if progress:
                progress.status = "error"
                progress.error = "Staged document not found"
                progress.progress_percentage = 0.0
            return

        if progress is None:
            progress = ProcessProgress(
                file_id=file_id,
                document_id=staged_doc.document_id,
                filename=staged_doc.filename,
                status="processing",
                stage="conversion" if staged_doc.mode != "entities_only" else "entity_extraction",
                mode=staged_doc.mode,
                queue_position=0,
                chunks_processed=0,
                total_chunks=0,
                chunk_progress=0.0,
                entity_progress=0.0,
                progress_percentage=0.0,
                entity_state=None,
            )
            _progress_percentage[file_id] = progress

        async with _queue_lock:
            if _processing_queue and _processing_queue[0] == file_id:
                _processing_queue.pop(0)
            elif file_id in _processing_queue:
                _processing_queue.remove(file_id)
            _update_queue_positions()

        file_path = Path(staged_doc.file_path)
        doc_id = staged_doc.document_id
        if not doc_id and file_path:
            doc_id = document_processor.compute_document_id(file_path)
            staged_doc.document_id = doc_id
        progress.document_id = doc_id

        stage_name = "entity_extraction" if staged_doc.mode == "entities_only" else "conversion"
        progress.stage = stage_name
        progress.status = "processing"
        progress.error = None
        progress.queue_position = 0

        if doc_id:
            _active_document_jobs[doc_id] = file_id

        _global_processing_state.update(
            {
                "is_processing": True,
                "current_file_id": file_id,
                "current_document_id": doc_id,
                "current_filename": staged_doc.filename,
                "current_stage": stage_name,
                "progress_percentage": progress.progress_percentage,
                "queue_length": len(_processing_queue) + 1,
            }
        )

        if doc_id:
            _mark_document_status(
                doc_id,
                "processing",
                stage_name,
                progress.progress_percentage,
                job_id=file_id,
            )

        success = False
        try:
            if staged_doc.mode == "entities_only":
                success = await _run_entity_job(file_id, staged_doc, progress)
            else:
                success = await _run_chunk_job(file_id, staged_doc, progress)

                # Check for cancellation between phases
                if progress.cancelled:
                    logger.info(f"Job {file_id} cancelled between phases")
                    progress.status = "cancelled"
                    progress.stage = "cancelled"
                    success = False
                elif success and staged_doc.mode != "chunks_only":
                    progress.stage = "entity_extraction"
                    _global_processing_state["current_stage"] = "entity_extraction"
                    if doc_id:
                        _mark_document_status(
                            doc_id,
                            "processing",
                            "entity_extraction",
                            progress.progress_percentage,
                            job_id=file_id,
                        )
                    success = await _run_entity_job(file_id, staged_doc, progress)

        finally:
            if success:
                progress.status = "completed"
                progress.stage = "completed"
                if staged_doc.mode in {"full", "chunks_only"}:
                    progress.chunk_progress = 1.0
                if staged_doc.mode in {"full", "entities_only"}:
                    progress.entity_progress = 1.0 if progress.entity_progress not in (None, 0.0) else progress.entity_progress
                progress.progress_percentage = 100.0
                if doc_id:
                    _mark_document_status(
                        doc_id,
                        "completed",
                        "completed",
                        100.0,
                        job_id=file_id,
                    )
            else:
                if progress.status == "cancelled":
                    progress.progress_percentage = min(progress.progress_percentage, 99.0)
                else:
                    if progress.status != "error":
                        progress.status = "error"
                        if not progress.error:
                            progress.error = "Processing failed"
                    progress.progress_percentage = min(progress.progress_percentage, 99.0)
                    if doc_id:
                        _mark_document_status(
                            doc_id,
                            "error",
                            progress.stage or "error",
                            progress.progress_percentage,
                            job_id=file_id,
                        )

            if doc_id and _active_document_jobs.get(doc_id) == file_id:
                _active_document_jobs.pop(doc_id, None)

            if staged_doc.mode in {"full", "chunks_only", "entities_only"}:
                _staged_documents.pop(file_id, None)

            _global_processing_state.update(
                {
                    "is_processing": False,
                    "current_file_id": None,
                    "current_document_id": None,
                    "current_filename": None,
                    "current_stage": None,
                    "progress_percentage": 0.0,
                    "queue_length": len(_processing_queue),
                }
            )

            if progress.status in {"completed", "error", "cancelled"}:
                asyncio.create_task(_cleanup_progress_entry(file_id))


async def _run_chunk_job(
    file_id: str, staged_doc: StagedDocument, progress: ProcessProgress
) -> bool:
    """Process chunk extraction/embedding for a staged document."""

    # Check for cancellation before starting
    if progress.cancelled:
        logger.info(f"Chunk job cancelled for {file_id}")
        progress.status = "cancelled"
        progress.stage = "cancelled"
        return False

    file_path = Path(staged_doc.file_path)
    if not file_path.exists():
        progress.status = "error"
        progress.error = "File not found"
        progress.stage = "error"
        return False

    doc_id = staged_doc.document_id or document_processor.compute_document_id(file_path)
    staged_doc.document_id = doc_id
    progress.document_id = doc_id

    loop = asyncio.get_running_loop()
    try:
        executor = get_blocking_executor()
        estimated_chunks = await loop.run_in_executor(
            executor, partial(_estimate_total_chunks_for_path, file_path)
        )
    except RuntimeError as e:
        logger.debug(f"Blocking executor unavailable while estimating chunks: {e}.")
        if SHUTTING_DOWN:
            logger.info("Process shutting down; aborting chunk estimation")
            estimated_chunks = 0
        else:
            try:
                executor = get_blocking_executor()
                estimated_chunks = await loop.run_in_executor(
                    executor, partial(_estimate_total_chunks_for_path, file_path)
                )
            except Exception as e2:
                logger.error(f"Failed to schedule chunk estimation: {e2}")
                estimated_chunks = 0
    if estimated_chunks:
        progress.total_chunks = estimated_chunks

    def _chunk_progress_callback(
        chunks_processed: int, message: Optional[str] = None
    ) -> None:
        if message:
            progress.message = message
            msg_lower = message.lower()
            if "conversion" in msg_lower or "converting" in msg_lower:
                if "post" in msg_lower:
                    progress.stage = "post_conversion"
                else:
                    progress.stage = "conversion"
            elif "classification" in msg_lower:
                progress.stage = "classification"
            elif "summary" in msg_lower or "abstract" in msg_lower:
                progress.stage = "summarization"
            elif "content_filter" in msg_lower or "filtering" in msg_lower:
                progress.stage = "content_filtering"
            elif "embedding" in msg_lower:
                progress.stage = "embedding"
            elif "chunking" in msg_lower:
                progress.stage = "chunking"
        
        progress.chunks_processed = chunks_processed
        if progress.total_chunks:
            progress.chunk_progress = min(1.0, chunks_processed / progress.total_chunks)
        else:
            if chunks_processed == 0:
                progress.chunk_progress = 0.0
                if not message:
                    progress.message = "Starting"
                    progress.stage = "Starting"
            else:
                progress.chunk_progress = 0.9
        progress.progress_percentage = _calculate_overall_progress(progress)
        _global_processing_state["progress_percentage"] = progress.progress_percentage

    _chunk_progress_callback.is_cancelled = lambda: progress.cancelled

    process_fn = partial(
        document_processor.process_file_chunks_only,
        file_path,
        staged_doc.filename,
        _chunk_progress_callback,
        None,
        doc_id,
    )

    try:
        executor = get_blocking_executor()
        result = await loop.run_in_executor(executor, process_fn)
    except RuntimeError as e:
        logger.debug(f"Blocking executor unavailable while processing chunks: {e}.")
        if SHUTTING_DOWN:
            logger.info("Process shutting down; aborting chunk processing")
            result = {"status": "error", "error": "shutting_down"}
        else:
            try:
                executor = get_blocking_executor()
                result = await loop.run_in_executor(executor, process_fn)
            except Exception as e2:
                logger.error(f"Failed to schedule chunk processing: {e2}")
                result = {"status": "error", "error": str(e2)}

    if not result or result.get("status") != "success":
        progress.error = (
            result.get("error") if isinstance(result, dict) else "Failed to process document"
        )
        progress.status = "error"
        return False

    created_chunks = result.get("chunks_created", 0)
    progress.chunks_processed = created_chunks
    progress.total_chunks = created_chunks or progress.total_chunks
    progress.chunk_progress = 1.0 if created_chunks else progress.chunk_progress
    progress.progress_percentage = _calculate_overall_progress(progress)
    _global_processing_state["progress_percentage"] = progress.progress_percentage

    return True


async def _run_entity_job(
    file_id: str, staged_doc: StagedDocument, progress: ProcessProgress
) -> bool:
    """Run entity extraction for a document and update progress accordingly."""

    # Check for cancellation before starting
    if progress.cancelled:
        logger.info(f"Entity job cancelled for {file_id}")
        progress.status = "cancelled"
        progress.stage = "cancelled"
        return False

    doc_id = progress.document_id
    if not doc_id:
        progress.error = "Unknown document identifier for entity extraction"
        progress.status = "error"
        progress.stage = "error"
        return False

    loop = asyncio.get_running_loop()

    def _entity_progress_callback(
        state: EntityExtractionState, fraction: float, info: Optional[str]
    ) -> None:
        progress.entity_state = state.value
        progress.stage = state.value  # Update stage for granular UI feedback
        if fraction and fraction > 0:
            progress.entity_progress = min(1.0, max(0.0, fraction))
        else:
            progress.entity_progress = _ENTITY_STAGE_FRACTIONS.get(state, 0.0)
        if info:
            progress.message = info
            # Parse chunk counts from info message like "Extracting entities (123/456 chunks)"
            # or "Gleaning extraction (2 passes) - (123/456 chunks)"
            import re
            match = re.search(r'\((\d+)/(\d+)\s*(?:chunks)?\)', info)
            if match:
                progress.chunks_processed = int(match.group(1))
                progress.total_chunks = int(match.group(2))
        progress.progress_percentage = _calculate_overall_progress(progress)
        _global_processing_state["progress_percentage"] = progress.progress_percentage

    _entity_progress_callback.is_cancelled = lambda: progress.cancelled

    entity_fn = partial(
        document_processor.extract_entities_for_document,
        doc_id,
        staged_doc.filename,
        _entity_progress_callback,
    )

    try:
        executor = get_blocking_executor()
        result = await loop.run_in_executor(executor, entity_fn)
    except RuntimeError as e:
        logger.debug(f"Blocking executor unavailable while extracting entities: {e}.")
        if SHUTTING_DOWN:
            logger.info("Process shutting down; aborting entity extraction")
            result = {"status": "error", "error": "shutting_down"}
        else:
            try:
                executor = get_blocking_executor()
                result = await loop.run_in_executor(executor, entity_fn)
            except Exception as e2:
                logger.error(f"Failed to schedule entity extraction: {e2}")
                result = {"status": "error", "error": str(e2)}

    status = result.get("status") if isinstance(result, dict) else "error"

    if status not in {"success", "skipped"}:
        progress.status = "error"
        progress.error = result.get("error") if isinstance(result, dict) else "Entity extraction failed"
        return False

    if status in {"success", "skipped"}:
        progress.entity_progress = 1.0
    progress.progress_percentage = _calculate_overall_progress(progress)
    _global_processing_state["progress_percentage"] = progress.progress_percentage

    return True


async def _cleanup_progress_entry(file_id: str, delay: float = 5.0) -> None:
    """Remove progress records after a cooldown to keep UI responsive."""

    try:
        await asyncio.sleep(delay)
        _progress_percentage.pop(file_id, None)
    except Exception:  # pragma: no cover - defensive
        pass


async def _enqueue_processing_jobs(file_ids: List[str]) -> List[str]:
    """Normalize progress entries and add jobs to the processing queue."""

    enqueued: List[str] = []

    async with _queue_lock:
        for file_id in file_ids:
            staged_doc = _staged_documents[file_id]

            if file_id not in _progress_percentage:
                _progress_percentage[file_id] = ProcessProgress(
                    file_id=file_id,
                    document_id=staged_doc.document_id,
                    filename=staged_doc.filename,
                    status="queued",
                    stage="queued",
                    mode=staged_doc.mode,
                    queue_position=None,
                    chunks_processed=0,
                    total_chunks=0,
                    chunk_progress=0.0,
                    entity_progress=0.0,
                    progress_percentage=0.0,
                    entity_state=None,
                )
            else:
                progress = _progress_percentage[file_id]
                progress.status = "queued"
                progress.stage = "queued"
                progress.mode = staged_doc.mode
                progress.chunks_processed = 0
                progress.total_chunks = 0
                progress.chunk_progress = 0.0
                progress.entity_progress = 0.0
                progress.progress_percentage = 0.0
                progress.error = None
                progress.entity_state = None

            if (
                file_id not in _processing_queue
                and _global_processing_state.get("current_file_id") != file_id
            ):
                _processing_queue.append(file_id)
                enqueued.append(file_id)

            if staged_doc.document_id:
                _active_document_jobs[staged_doc.document_id] = file_id
                _mark_document_status(
                    staged_doc.document_id,
                    "queued",
                    "queued",
                    0.0,
                    job_id=file_id,
                )

        _update_queue_positions()
        if _processing_queue:
            _queue_event.set()

    if enqueued:
        await _ensure_worker_running()

    return enqueued


def _is_document_processing(document_id: Optional[str]) -> bool:
    """Check if a document already has an active or pending processing job."""

    if not document_id:
        return False

    for progress in _progress_percentage.values():
        if (
            progress.document_id == document_id
            and progress.status in {"queued", "processing"}
        ):
            return True

    return _global_processing_state.get("current_document_id") == document_id


@router.get("/cache-stats")
async def get_cache_stats():
    """
    Get cache performance statistics.
    
    Returns:
        Dictionary containing cache hit rates, sizes, and metrics
    """
    from core.cache_metrics import cache_metrics
    return cache_metrics.get_report()

@router.get("/routing-metrics")
async def get_routing_metrics():
    """Get query routing performance metrics."""
    try:
        from core.routing_metrics import routing_metrics
        return routing_metrics.get_stats()
    except Exception as e:
        logger.error(f"Failed to get routing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/routing-metrics")
async def clear_routing_metrics(
    current_user=Depends(require_admin),
):
    """
    Clear all routing metrics data.
    Requires admin privileges.
    """
    try:
        from core.routing_metrics import routing_metrics
        routing_metrics.clear_metrics()
        return {"status": "success", "message": "Routing metrics cleared"}
    except Exception as e:
        logger.error(f"Failed to clear routing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats(
    include_suggestion_status: bool = Query(False, description="Check if docs have suggestion"),
):
    """
    Get database statistics including document and chunk counts.

    Returns:
        Database statistics
    """
    try:
        stats = graph_db.get_database_stats()

        documents = stats.get("documents", [])
        
        if include_suggestion_status:
            learner = get_pattern_learner()
            for doc in documents:
                doc_id = doc.get("document_id")
                try:
                    doc["has_suggestions"] = learner.has_suggestions(doc_id)
                except Exception:
                    doc["has_suggestions"] = False

        # Enrich documents with live queue state
        for doc in documents:
            doc_id = doc.get("document_id")
            progress_match = None

            for progress in _progress_percentage.values():
                if progress.document_id == doc_id:
                    progress_match = progress
                    break

            if progress_match:
                doc["processing_status"] = progress_match.status
                doc["processing_stage"] = progress_match.stage
                doc["progress_percentage"] = progress_match.progress_percentage
                doc["queue_position"] = progress_match.queue_position
            else:
                # Check if Neo4j has live processing data (from background threads)
                # that isn't in the in-memory queue (e.g., from DocumentProcessor.update_document)
                neo4j_status = doc.get("processing_status")
                neo4j_stage = doc.get("processing_stage")
                neo4j_progress = doc.get("processing_progress")
                
                if neo4j_status == "processing" and neo4j_progress is not None:
                    # Use Neo4j values directly - background thread is updating the DB
                    doc["processing_status"] = neo4j_status
                    doc["processing_stage"] = neo4j_stage or "unknown"
                    doc["progress_percentage"] = neo4j_progress
                else:
                    doc.setdefault("processing_status", neo4j_status or "idle")
                    doc.setdefault("processing_stage", neo4j_stage or "idle")
                    doc.setdefault("progress_percentage", neo4j_progress or 0.0)

        active_progress = [
            progress
            for progress in _progress_percentage.values()
            if progress.status in {"queued", "processing"}
        ]

        processing_summary = ProcessingSummary(
            is_processing=_global_processing_state.get("is_processing", False),
            current_file_id=_global_processing_state.get("current_file_id"),
            current_document_id=_global_processing_state.get("current_document_id"),
            current_filename=_global_processing_state.get("current_filename"),
            current_stage=_global_processing_state.get("current_stage"),
            progress_percentage=_global_processing_state.get("progress_percentage"),
            queue_length=_global_processing_state.get("queue_length", 0),
            pending_documents=active_progress,
        )

        return DatabaseStats(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            total_entities=stats.get("total_entities", 0),
            total_relationships=stats.get("total_relationships", 0),
            documents=documents,
            processing=processing_summary,
            enable_delete_operations=settings.enable_delete_operations,
        )

    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Args:
        file: Uploaded file

    Returns:
        Upload response with processing results
    """
    filename = file.filename or "unknown"
    
    try:
        # Generate a unique file ID to avoid conflicts
        file_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()[:16]
        
        # Save uploaded file to staged_uploads directory (not /tmp) so it persists for preview
        staged_dir = Path("data/staged_uploads")
        staged_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the same naming pattern as stage endpoint for consistency
        safe_filename = filename.replace(" ", " ")  # Keep spaces for now
        file_path = staged_dir / f"{file_id}_{safe_filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        try:
            executor = get_blocking_executor()
            result = await loop.run_in_executor(
                executor,
                document_processor.process_file,
                file_path,
                filename,  # Pass original filename
            )
        except RuntimeError as e:
            logger.debug(f"Blocking executor unavailable while processing upload: {e}.")
            if SHUTTING_DOWN:
                logger.info("Process shutting down; aborting upload processing")
                result = {"status": "error", "error": "shutting_down"}
            else:
                try:
                    executor = get_blocking_executor()
                    result = await loop.run_in_executor(
                        executor,
                        document_processor.process_file,
                        file_path,
                        filename,
                    )
                except Exception as e2:
                    logger.error(f"Failed to schedule upload processing: {e2}")
                    result = {"status": "error", "error": str(e2)}

        # DO NOT delete the file - we need it for previews
        # The file path is stored in the database and used by the preview endpoint

        if result and result.get("status") == "success":
            # Invalidate response cache on successful upload to avoid stale responses
            try:
                get_response_cache().clear()
                try:
                    from core.cache_metrics import cache_metrics

                    cache_metrics.record_response_invalidation()
                except Exception:
                    pass
                logger.info("Cleared response cache after document upload: %s", result.get("document_id"))
            except Exception:
                logger.warning("Failed to clear response cache after upload")
            return DocumentUploadResponse(
                filename=filename,
                status="success",
                chunks_created=result.get("chunks_created", 0),
                document_id=result.get("document_id"),
                processing_status=result.get("processing_status"),
                processing_stage=result.get("processing_stage"),
            )
        else:
            error_msg = result.get("error", "Unknown error") if result else "Processing failed"
            return DocumentUploadResponse(
                filename=filename,
                status="error",
                chunks_created=0,
                processing_status=result.get("processing_status") if result else "error",
                processing_stage=result.get("processing_stage") if result else "error",
                error=error_msg,
            )

    except ServiceUnavailable as e:
        logger.error("Graph DB unavailable during document upload: %s", e)
        return DocumentUploadResponse(
            filename=filename,
            status="error",
            chunks_created=0,
            error="Graph database unavailable",
        )
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            # Let global handler handle DB unavailability for uniform 503 responses
            raise
        logger.error(f"Document upload failed: {e}")
        return DocumentUploadResponse(
            filename=filename,
            status="error",
            chunks_created=0,
            error=str(e),
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.

    Args:
        document_id: Document ID to delete

    Returns:
        Deletion result
    """
    if not settings.enable_delete_operations:
        raise HTTPException(
            status_code=403,
            detail="Delete operations are disabled in this configuration"
        )
    
    try:
        # Issue #6 Fix: Retrieve file_path before deletion to clean up physical file
        file_path_to_delete = None
        try:
            file_info = graph_db.get_document_file_info(document_id)
            file_path_to_delete = file_info.get("file_path")
        except Exception as e:
            logger.warning(f"Could not retrieve file_path for document {document_id}: {e}")

        graph_db.delete_document(document_id)

        # Delete physical file if it exists
        if file_path_to_delete:
            try:
                file_path = Path(file_path_to_delete)
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted physical file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete physical file {file_path_to_delete}: {e}")

        # CLEANUP IN-MEMORY STATE (Fix for UI persistence bug)
        # Find file_id associated with this document_id
        file_id_to_remove = None
        for fid, doc in _staged_documents.items():
            if doc.document_id == document_id:
                file_id_to_remove = fid
                break
        
        # If we didn't find it in staged docs, check progress map directly
        if not file_id_to_remove:
             for fid, prog in _progress_percentage.items():
                if prog.document_id == document_id:
                    file_id_to_remove = fid
                    break

        if file_id_to_remove:
            _staged_documents.pop(file_id_to_remove, None)
            _progress_percentage.pop(file_id_to_remove, None)
            
            # Remove from queue if present
            async with _queue_lock:
                if file_id_to_remove in _processing_queue:
                    _processing_queue.remove(file_id_to_remove)
            
            # Reset global state if this was the current file
            if _global_processing_state.get("current_file_id") == file_id_to_remove:
                 _global_processing_state.update({
                    "is_processing": False,
                    "current_file_id": None,
                    "current_document_id": None,
                    "current_filename": None,
                    "current_stage": None,
                    "progress_percentage": 0.0,
                })
            
            logger.info(f"Cleaned up in-memory state for file_id {file_id_to_remove}")

        try:
            get_response_cache().clear()
            try:
                from core.cache_metrics import cache_metrics

                cache_metrics.record_response_invalidation()
            except Exception:
                pass
            logger.info("Cleared response cache after document delete: %s", document_id)
        except Exception:
            logger.warning("Failed to clear response cache after document delete")
        return {"status": "success", "message": f"Document {document_id} deleted"}

    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup-orphans")
async def cleanup_orphans():
    """
    Manually trigger cleanup of orphaned chunks and entities.
    
    Orphaned chunks are chunks that are not connected to any Document node
    via a HAS_CHUNK relationship. This can happen if a document was deleted
    improperly or if there was a processing failure.
    
    Returns:
        Dict with counts: {"chunks_deleted": N, "entities_deleted": M}
    """
    try:
        # Use grace_period=0 for manual cleanup to delete all orphans immediately
        result = graph_db.cleanup_orphaned_chunks(grace_period_minutes=0)
        
        logger.info(
            f"Manual orphan cleanup: deleted {result.get('chunks_deleted', 0)} chunks "
            f"and {result.get('entities_deleted', 0)} entities"
        )
        
        return {
            "status": "success",
            "chunks_deleted": result.get("chunks_deleted", 0),
            "entities_deleted": result.get("entities_deleted", 0),
        }
    except Exception as e:
        logger.error(f"Failed to cleanup orphans: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel

class ClearDatabaseRequest(BaseModel):
    clearKnowledgeBase: bool = True
    clearConversations: bool = True

@router.post("/clear")
async def clear_database(request: ClearDatabaseRequest = ClearDatabaseRequest()):
    """
    Clear all data from the database.

    Returns:
        Clear operation result
    """
    if not settings.enable_delete_operations:
        raise HTTPException(
            status_code=403,
            detail="Delete operations are disabled in this configuration"
        )
    
    try:
        graph_db.clear_database(
            clear_knowledge_base=request.clearKnowledgeBase,
            clear_conversations=request.clearConversations
        )

        # Issue #6 Fix: Purge staged_uploads directory
        try:
            staged_uploads_dir = Path("data/staged_uploads")
            if staged_uploads_dir.exists():
                import shutil
                for item in staged_uploads_dir.iterdir():
                    try:
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    except Exception as e:
                        logger.warning(f"Failed to remove {item}: {e}")
                logger.info("Purged staged_uploads directory after clear_database")
        except Exception as e:
            logger.warning(f"Failed to purge staged_uploads: {e}")

        # CLEANUP IN-MEMORY STATE (Fix for UI persistence bug)
        if request.clearKnowledgeBase:
            _staged_documents.clear()
            _progress_percentage.clear()
            async with _queue_lock:
                _processing_queue.clear()
            
            _global_processing_state.update({
                "is_processing": False,
                "current_file_id": None,
                "current_document_id": None,
                "current_filename": None,
                "current_stage": None,
                "progress_percentage": 0.0,
                "queue_length": 0
            })
            logger.info("Cleared in-memory processing state")

        try:
            get_response_cache().clear()
            try:
                from core.cache_metrics import cache_metrics

                cache_metrics.record_response_invalidation()
            except Exception:
                pass
            logger.info("Cleared response cache after clear_database")
        except Exception:
            logger.warning("Failed to clear response cache after clear_database")
        return {"status": "success", "message": "Database cleared"}

    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    include_suggestion_status: bool = Query(False, description="Check if docs have suggestion"),
):
    """
    List all documents in the database.

    Returns:
        List of documents with metadata
    """
    try:
        documents = graph_db.list_documents()
        
        if include_suggestion_status:
            learner = get_pattern_learner()
            for doc in documents:
                try:
                    doc["has_suggestions"] = learner.has_suggestions(doc["id"])
                except Exception:
                    doc["has_suggestions"] = False
                    
        return {"documents": documents}

    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hashtags")
async def list_hashtags():
    """
    List all unique hashtags from all documents.

    Returns:
        List of hashtags
    """
    try:
        hashtags = graph_db.get_all_hashtags()
        return {"hashtags": hashtags}

    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to list hashtags: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/folders", response_model=FolderListResponse)
async def list_folders():
    """List all folders with document counts."""
    try:
        folders = graph_db.list_folders()
        return FolderListResponse(folders=[FolderSummary(**folder) for folder in folders])
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to list folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/folders", response_model=FolderSummary)
async def create_folder(request: CreateFolderRequest):
    """Create a new folder."""
    try:
        folder = graph_db.create_folder(request.name)
        return FolderSummary(**folder)
    except ValueError as exc:
        message = str(exc)
        if "exists" in message:
            raise HTTPException(status_code=409, detail=message)
        raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to create folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/folders/{folder_id}", response_model=FolderSummary)
async def rename_folder(folder_id: str, request: RenameFolderRequest):
    """Rename a folder."""
    try:
        folder = graph_db.rename_folder(folder_id, request.name)
        folder["document_count"] = next(
            (f.get("document_count", 0) for f in graph_db.list_folders() if f.get("id") == folder_id),
            0,
        )
        return FolderSummary(**folder)
    except ValueError as exc:
        message = str(exc)
        if "exists" in message:
            raise HTTPException(status_code=409, detail=message)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to rename folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/folders/{folder_id}")
async def delete_folder(
    folder_id: str,
    mode: Literal["move_to_root", "delete_documents"] = Query(..., description="Delete mode"),
):
    """Delete a folder, optionally deleting documents."""
    if mode == "delete_documents" and not settings.enable_delete_operations:
        raise HTTPException(
            status_code=403,
            detail="Delete operations are disabled in this configuration",
        )

    try:
        result = graph_db.delete_folder(folder_id, mode)
        for doc_id in result.get("document_ids", []):
            _invalidate_document_cache(doc_id)
        return {
            "status": "success",
            "folder_id": folder_id,
            "documents_deleted": result.get("documents_deleted", 0),
            "documents_moved": result.get("documents_moved", 0),
        }
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to delete folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/documents/{document_id}/folder")
async def move_document_to_folder(document_id: str, request: MoveDocumentRequest):
    """Move a document into a folder or back to root."""
    try:
        result = graph_db.assign_document_folder(document_id, request.folder_id)
        _invalidate_document_cache(document_id)
        return {"status": "success", "document_id": document_id, **result}
    except ValueError as exc:
        message = str(exc)
        if "not found" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message)
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to move document {document_id} to folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/order")
async def reorder_documents(request: ReorderDocumentsRequest):
    """Persist manual ordering for documents."""
    try:
        updated = graph_db.reorder_documents(request.document_ids, request.folder_id)
        for doc_id in request.document_ids:
            _invalidate_document_cache(doc_id)
        return {"status": "success", "updated": updated}
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to reorder documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage", response_model=StageDocumentResponse)
async def stage_document(file: UploadFile = File(...)):
    """
    Stage a document for later processing.

    Args:
        file: Uploaded file

    Returns:
        Staged document information
    """
    filename = file.filename or "unknown"

    try:
        # Generate unique file ID
        file_id = hashlib.md5(
            f"{filename}_{time.time()}".encode()
        ).hexdigest()[:16]

        # Save file to staging area
        staging_dir = Path("data/staged_uploads")
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = staging_dir / f"{file_id}_{filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        document_id = document_processor.compute_document_id(file_path)

        await _cancel_existing_document_jobs(
            document_id=document_id,
            replacement_file_id=file_id,
        )

        metadata = document_processor.build_metadata(file_path, filename)
        
        metadata.setdefault("uploaded_at", time.time())
        metadata["original_filename"] = filename
        metadata["processing_status"] = "staged"
        metadata["processing_stage"] = "staged"
        metadata["progress_percentage"] = 0.0
        metadata["processing_updated_at"] = time.time()

        try:
            graph_db.create_document_node(document_id, metadata)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning(
                "Failed to create placeholder document node for %s: %s",
                document_id,
                exc,
            )

        # Store staged document info
        staged_doc = StagedDocument(
            file_id=file_id,
            filename=filename,
            file_size=len(content),
            file_path=str(file_path),
            timestamp=time.time(),
            document_id=document_id,
            mode="full",
        )
        _staged_documents[file_id] = staged_doc

        # Automatically enqueue for processing
        enqueued = await _enqueue_processing_jobs([file_id])
        
        # Issue #5: Background task to drain queue if not running
        # if enqueued:
        #    _ensure_worker_running()
        
        return StageDocumentResponse(
            file_id=file_id,
            filename=filename,
            document_id=document_id,
            status="queued" if enqueued else "staged",
            processing_status="queued" if enqueued else "staged",
            processing_stage="queued" if enqueued else "staged",
        )

    except ServiceUnavailable as e:
        logger.error("Graph DB unavailable when staging document: %s", e)
        return StageDocumentResponse(
            file_id="",
            filename=filename,
            status="error",
            processing_status="error",
            processing_stage="error",
            error="Graph database unavailable",
        )
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to stage document: {e}")
        return StageDocumentResponse(
            file_id="",
            filename=filename,
            status="error",
            processing_status="error",
            processing_stage="error",
            error=str(e)
        )


@router.get("/staged")
async def list_staged_documents():
    """
    List all staged documents.

    Returns:
        List of staged documents
    """
    return {"documents": list(_staged_documents.values())}


@router.delete("/staged/{file_id}")
async def delete_staged_document(file_id: str):
    """
    Delete a staged document.

    Args:
        file_id: File ID to delete

    Returns:
        Deletion result
    """
    try:
        if file_id not in _staged_documents:
            raise HTTPException(status_code=404, detail="Document not found")

        staged_doc = _staged_documents[file_id]
        
        # Delete file
        file_path = Path(staged_doc.file_path)
        if file_path.exists():
            file_path.unlink()

        # Remove from staged list
        del _staged_documents[file_id]

        # Mark as cancelled if in progress, then remove from tracking
        if file_id in _progress_percentage:
            _progress_percentage[file_id].cancelled = True
            logger.info(f"Cancelling in-progress job for {file_id}")
            # Don't delete immediately - let the worker see the cancelled flag
            # The worker will clean up after it detects cancellation

        # Remove from processing queue if exists
        async with _queue_lock:
            if file_id in _processing_queue:
                _processing_queue.remove(file_id)
                _update_queue_positions()

        return {"status": "success", "message": f"Staged document {file_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to delete staged document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process")
async def process_documents(request: ProcessDocumentsRequest):
    """
    Process staged documents in the background.

    Args:
        request: List of file IDs to process
        background_tasks: FastAPI background tasks

    Returns:
        Processing status
    """
    try:
        # Validate file IDs
        invalid_ids = [fid for fid in request.file_ids if fid not in _staged_documents]
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file IDs: {invalid_ids}"
            )

        enqueued = await _enqueue_processing_jobs(request.file_ids)

        return {
            "status": "queued" if enqueued else "noop",
            "message": f"Queued {len(enqueued)} documents for processing",
            "file_ids": enqueued,
        }

    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, ServiceUnavailable):
            raise
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{file_id}")
async def get_progress_percentage(file_id: str):
    """
    Get processing progress for a specific file.

    Args:
        file_id: File ID

    Returns:
        Processing progress
    """
    if file_id not in _progress_percentage:
        raise HTTPException(status_code=404, detail="Progress not found")

    return _progress_percentage[file_id]


@router.get("/progress")
async def get_all_progress_percentage():
    """
    Get processing progress for all files.

    Returns:
        List of processing progress
    """
    progress_list = list(_progress_percentage.values())
    
    # Sync stage from Neo4j for documents in early stages
    # This ensures conversion, post_conversion, etc. are visible even if callback hasn't fired
    for progress in progress_list:
        if progress.status == "processing" and progress.document_id:
            _sync_progress_from_neo4j(progress)
    
    pending_progress = [
        progress
        for progress in progress_list
        if progress.status in {"queued", "processing"}
    ]

    return {
        "progress": progress_list,
        "global": {
            **_global_processing_state,
            "pending_documents": pending_progress,
        },
    }


@router.post("/documents/{document_id}/process/chunks")
async def reprocess_document_chunks(document_id: str):
    """Queue full document processing for an existing document lacking chunks."""

    if _is_document_processing(document_id):
        raise HTTPException(status_code=409, detail="Document is already processing")

    try:
        file_info = graph_db.get_document_file_info(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    file_path = file_info.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Document file path missing")

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document file not available on server")

    filename = file_info.get("file_name") or path.name
    file_size = path.stat().st_size if path.exists() else 0

    file_id = hashlib.md5(f"{document_id}_{time.time()}".encode()).hexdigest()[:16]

    staged_doc = StagedDocument(
        file_id=file_id,
        filename=filename,
        file_size=file_size,
        file_path=str(path),
        timestamp=time.time(),
        document_id=document_id,
        mode="full",
    )
    _staged_documents[file_id] = staged_doc

    _mark_document_status(document_id, "queued", "queued", 0.0, job_id=file_id)

    enqueued = await _enqueue_processing_jobs([file_id])

    return {
        "status": "queued" if enqueued else "noop",
        "file_id": file_id,
        "document_id": document_id,
    }


@router.post("/documents/{document_id}/process/entities")
async def reprocess_document_entities(document_id: str):
    """Queue entity extraction for an existing document that lacks entities."""

    if _is_document_processing(document_id):
        raise HTTPException(status_code=409, detail="Document is already processing")

    try:
        details = graph_db.get_document_details(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    if not details.get("chunks"):
        raise HTTPException(
            status_code=400,
            detail="Document has no chunks. Re-run chunk processing first.",
        )

    file_info = graph_db.get_document_file_info(document_id)
    file_path = file_info.get("file_path")
    path = Path(file_path) if file_path else None

    filename = details.get("file_name") or file_info.get("file_name") or (
        path.name if path else document_id
    )
    file_size = path.stat().st_size if path and path.exists() else 0

    file_id = hashlib.md5(f"entities_{document_id}_{time.time()}".encode()).hexdigest()[:16]

    staged_doc = StagedDocument(
        file_id=file_id,
        filename=filename,
        file_size=file_size,
        file_path=str(path) if path else "",
        timestamp=time.time(),
        document_id=document_id,
        mode="entities_only",
    )
    _staged_documents[file_id] = staged_doc

    _mark_document_status(
        document_id,
        "queued",
        "entity_extraction",
        0.0,
        job_id=file_id,
    )

    enqueued = await _enqueue_processing_jobs([file_id])

    return {
        "status": "queued" if enqueued else "noop",
        "file_id": file_id,
        "document_id": document_id,
    }


async def _process_documents_task(file_ids: List[str]):
    """
    Background task to process documents.

    Args:
        file_ids: List of file IDs to process
    """
    for file_id in file_ids:
        try:
            if file_id not in _staged_documents:
                logger.error(f"File ID {file_id} not found in staged documents")
                continue

            staged_doc = _staged_documents[file_id]
            file_path = Path(staged_doc.file_path)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                _progress_percentage[file_id].status = "error"
                _progress_percentage[file_id].error = "File not found"
                continue

            # Estimate total chunks first
            logger.info(f"Processing file: {file_path}")
            
            # Create a progress callback
            def progress_callback(chunks_processed: int):
                if file_id in _progress_percentage:
                    progress = _progress_percentage[file_id]
                    progress.chunks_processed = chunks_processed
                    if progress.total_chunks > 0:
                        progress.progress_percentage = (
                            chunks_processed / progress.total_chunks
                        ) * 100

            # Get file extension and load content to estimate chunks
            from core.chunking import document_chunker
            file_ext = file_path.suffix.lower()
            loader = document_processor.loaders.get(file_ext)
            
            if loader:
                # Load content to estimate chunks
                from ingestion.loaders.image_loader import ImageLoader
                from ingestion.loaders.pdf_loader import PDFLoader
                
                if isinstance(loader, ImageLoader):
                    result = loader.load_with_metadata(file_path)
                    content = result["content"] if result else ""
                elif isinstance(loader, PDFLoader):
                    result = loader.load_with_metadata(file_path)
                    content = result["content"] if result else ""
                else:
                    content = loader.load(file_path)
                
                if content:
                    # Estimate total chunks
                    temp_chunks = document_chunker.chunk_text(
                        content,
                        f"temp_{file_id}",
                        source_metadata={"file_extension": file_ext},
                    )
                    _progress_percentage[file_id].total_chunks = len(temp_chunks)
                    logger.info(
                        f"Estimated {len(temp_chunks)} chunks for {staged_doc.filename}"
                    )

            # Process the file
            loop = asyncio.get_event_loop()
            try:
                executor = get_blocking_executor()
                result = await loop.run_in_executor(
                    executor,
                    document_processor.process_file,
                    file_path,
                    staged_doc.filename,
                    progress_callback,
                )
            except RuntimeError as e:
                logger.debug(f"Blocking executor unavailable while processing file: {e}.")
                if SHUTTING_DOWN:
                    logger.info("Process shutting down; aborting file processing")
                    result = {"status": "error", "error": "shutting_down"}
                else:
                    try:
                        executor = get_blocking_executor()
                        result = await loop.run_in_executor(
                            executor,
                            document_processor.process_file,
                            file_path,
                            staged_doc.filename,
                            progress_callback,
                        )
                    except Exception as e2:
                        logger.error(f"Failed to schedule file processing: {e2}")
                        result = {"status": "error", "error": str(e2)}

            if result and result.get("status") == "success":
                _progress_percentage[file_id].status = "completed"
                _progress_percentage[file_id].chunks_processed = result.get(
                    "chunks_created", 0
                )
                _progress_percentage[file_id].total_chunks = result.get(
                    "chunks_created", 0
                )
                _progress_percentage[file_id].progress_percentage = 100.0

                # Remove from staged documents
                del _staged_documents[file_id]

                # DO NOT delete the file - we need it for previews
                # The file path is stored in the database and used by the preview endpoint
                # If disk space becomes an issue, implement a proper file retention policy
                logger.info(
                    f"Successfully processed {staged_doc.filename}: "
                    f"{result.get('chunks_created', 0)} chunks, file retained for preview"
                )
            else:
                error_msg = result.get("error", "Processing failed") if result else "Processing failed"
                _progress_percentage[file_id].status = "error"
                _progress_percentage[file_id].error = error_msg
                logger.error(f"Failed to process {staged_doc.filename}: {error_msg}")

        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
            if file_id in _progress_percentage:
                _progress_percentage[file_id].status = "error"
                _progress_percentage[file_id].error = str(e)

    # Clean up completed/error progress after a delay
    await asyncio.sleep(5)
    for file_id in list(_progress_percentage.keys()):
        if _progress_percentage[file_id].status in ["completed", "error"]:
            del _progress_percentage[file_id]
