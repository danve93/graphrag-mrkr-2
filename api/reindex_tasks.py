"""Reusable reindex job implementation.

This module contains `run_reindex_job(job_id)` which performs the same
pipeline previously in the classification router. It updates job status
via `api.job_manager` so that workers and the router share the same
status reporting.
"""
from __future__ import annotations

import logging

from api import job_manager

logger = logging.getLogger(__name__)


def run_reindex_job(job_id: str) -> None:
    """Perform reindex pipeline and record status to the job manager.

    This function is safe to call from a FastAPI BackgroundTasks worker
    or a separate process (worker script).
    """
    try:
        job_manager.set_job_started(job_id)
        logger.info("Reindex job started: %s", job_id)

        # Import heavy modules lazily to keep imports cheap for routers/tests
        from config.settings import settings
        from core.graph_db import graph_db
        from ingestion.document_processor import DocumentProcessor

        if not settings.enable_entity_extraction:
            job_manager.set_job_result(
                job_id,
                {
                    "status": "partial",
                    "message": "Entity extraction disabled in settings",
                    "documents_processed": 0,
                    "entities_cleared": 0,
                },
                status="partial",
            )
            return

        # Step 1: collect documents
        with graph_db.driver.session() as session:  # type: ignore
            result = session.run(
                "MATCH (d:Document) RETURN d.id as doc_id, d.filename as filename"
            )
            documents = [{"doc_id": r["doc_id"], "filename": r["filename"]} for r in result]

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
                logger.error("Failed to clear entities for %s: %s", doc.get("doc_id"), e)

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
            logger.warning("Failed to create entity similarities: %s", e)

        clustering_result = None
        try:
            from core.graph_clustering import run_leiden_clustering

            clustering_result = run_leiden_clustering()
        except Exception as e:
            logger.warning("Clustering failed or not available: %s", e)

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
        logger.exception("Reindex job failed: %s", e)
        job_manager.set_job_failed(job_id, str(e))
