"""
Multi-format document processor for the RAG pipeline.
"""

import asyncio
import nest_asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config.settings import settings
from core.chunking import document_chunker
from core.document_summarizer import document_summarizer
from core.embeddings import embedding_manager
from core.entity_extraction import EntityExtractor
from core.singletons import get_response_cache, get_blocking_executor, SHUTTING_DOWN
from core.entity_graph import EntityGraph
from core.graph_db import graph_db
from ingestion.converters import document_converter
from ingestion.content_filters import get_content_filter
# (get_response_cache imported above)

logger = logging.getLogger(__name__)


class EntityExtractionState(Enum):
    """States for entity extraction operations."""

    STARTING = "starting"
    LLM_EXTRACTION = "llm_extraction"
    EMBEDDING_GENERATION = "embedding_generation"
    DATABASE_OPERATIONS = "database_operations"
    CLUSTERING = "clustering"
    VALIDATION = "validation"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class EntityExtractionStatus:
    """Status of an entity extraction operation."""

    operation_id: str
    document_id: str
    document_name: str
    state: EntityExtractionState
    started_at: float
    last_updated: float
    progress_info: Optional[str] = None
    error_message: Optional[str] = None


class DocumentProcessor:
    """Processes documents of various formats and stores them in the graph database."""

    def __init__(self):
        """Initialize the document processor."""
        # Initialize loaders with intelligent OCR support (shared with converter)
        self.converter = document_converter
        self.loaders = dict(self.converter.loaders)

        # Smart OCR is always enabled and applied intelligently
        # No user configuration needed - OCR is applied only where necessary
        self.enable_quality_filtering = getattr(
            settings, "enable_quality_filtering", True
        )

        # Initialize entity extractor if enabled
        if settings.enable_entity_extraction:
            self.entity_extractor = EntityExtractor()
            logger.info("Entity extraction enabled")
        else:
            self.entity_extractor = None
            logger.info("Entity extraction disabled")

        # Track background entity extraction threads so the UI can detect ongoing work
        self._bg_entity_threads = []
        self._bg_lock = threading.Lock()

        # Track entity extraction operation states for more robust status checking
        self._entity_extraction_operations: Dict[str, EntityExtractionStatus] = {}
        self._operations_lock = threading.Lock()
        
        # Cleanup any stale jobs from previous runs
        if not SHUTTING_DOWN:
            self._cleanup_stale_jobs()
            self._cleanup_orphaned_chunks()

    def _cleanup_stale_jobs(self):
        """Mark documents that were extracting during a previous shutdown as failed."""
        if not settings.enable_stale_job_cleanup:
            logger.info("Stale job cleanup disabled via configuration.")
            return

        try:
            # Query for documents stuck in processing
            stuck_docs = graph_db.get_documents_by_processing_status("processing")
            if not stuck_docs:
                return
                
            count = 0
            for doc in stuck_docs:
                # We can't know for sure if it was *actually* running on this instance,
                # but if we are starting up and find docs in 'processing', it's likely they are stuck.
                # However, in a distributed setup with multiple replicas, this is risky.
                # But document_processor is currently designed as a singleton per backend instance?
                # For now, let's just log and mark them as failed to be safe for the single-instance case.
                
                # Double check progress to guess if it was interrupted
                # If last updated was long ago? We don't have timestamp easily here without query.
                
                doc_id = doc.get("id")
                if not doc_id: 
                    continue
                    
                logger.warning(f"Found stale document in processing state: {doc_id}. Marking as failed.")
                try:
                    graph_db.create_document_node(
                        doc_id, 
                        {
                            "processing_status": "failed",
                            "error": "Processing interrupted (e.g., service restart)",
                            "processing_stage": "failed"
                        }
                    )
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup stale document {doc_id}: {e}")
            
            if count > 0:
                logger.info(f"Cleaned up {count} stale processing jobs")
                
        except Exception as e:
            logger.error(f"Failed to cleanup stale jobs: {e}")

    def _cleanup_orphaned_chunks(self):
        """Clean up orphaned chunks (not connected to any Document) on startup."""
        if not settings.enable_orphan_cleanup_on_startup:
            logger.info("Orphan cleanup on startup disabled via configuration.")
            return
        
        try:
            grace_period = settings.orphan_cleanup_grace_period_minutes
            result = graph_db.cleanup_orphaned_chunks(grace_period_minutes=grace_period)
            
            chunks_deleted = result.get("chunks_deleted", 0)
            entities_deleted = result.get("entities_deleted", 0)
            
            if chunks_deleted > 0 or entities_deleted > 0:
                logger.info(
                    f"Startup orphan cleanup: deleted {chunks_deleted} orphaned chunks "
                    f"and {entities_deleted} orphaned entities (grace period: {grace_period} min)"
                )
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned chunks: {e}")

    def compute_document_id(self, file_path: Path) -> str:
        """Compute the deterministic document id for a given file path."""
        return self._generate_document_id(file_path)

    def build_metadata(
        self, file_path: Path, original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build document metadata without mutating the database."""
        return self._extract_metadata(file_path, original_filename)

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time."""
        try:
            mtime = file_path.stat().st_mtime
        except FileNotFoundError:
            # Fallback for tests/mocked files that do not exist on disk
            logger.warning("File not found when generating document id for %s; using current time fallback", file_path)
            mtime = time.time()
        content = f"{file_path}_{mtime}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate a consistent entity ID from entity name."""
        return hashlib.md5(entity_name.lower().encode()).hexdigest()[:16]

    def _generate_operation_id(
        self, doc_id: str, operation_type: str = "entity_extraction"
    ) -> str:
        """Generate a unique operation ID."""
        timestamp = str(time.time())
        content = f"{doc_id}_{operation_type}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_gleaning_config(self, document_type: Optional[str] = None) -> tuple:
        """
        Determine gleaning configuration based on settings and document type.
        
        Returns:
            (use_gleaning: bool, max_gleanings: int)
        """
        if not settings.enable_gleaning:
            return False, 0
        
        # Check document-type-specific override
        if document_type and document_type in settings.gleaning_by_doc_type:
            max_gleanings = settings.gleaning_by_doc_type[document_type]
            logger.info(
                f"Using type-specific gleaning for '{document_type}': {max_gleanings} passes"
            )
            return max_gleanings > 0, max_gleanings
        
        # Use default
        max_gleanings = settings.max_gleanings
        logger.info(f"Using default gleaning: {max_gleanings} passes")
        return max_gleanings > 0, max_gleanings

    def _prepare_chunks_for_extraction(
        self, doc_id: str, processed_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Prepare chunk payloads for entity extraction.

        Prefers freshly processed chunks (when available) to ensure we attach
        the exact chunk IDs that were embedded and stored, but will fall back
        to querying Neo4j if necessary.
        """

        if processed_chunks:
            return [
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": chunk.get("content", ""),
                    "document_id": doc_id,
                }
                for chunk in processed_chunks
                if chunk.get("chunk_id")
            ]

        chunks_from_db = graph_db.get_document_chunks(doc_id)
        for chunk in chunks_from_db:
            chunk["document_id"] = doc_id
        return chunks_from_db

    def _build_extraction_metrics(
        self,
        doc_id: str,
        chunks_for_extraction: List[Dict[str, Any]],
        entity_dict,
        relationship_dict,
        created_entities: int,
        created_relationships: int,
        parse_errors: int = 0,
    ) -> Dict[str, Any]:
        """Create a lightweight metrics payload for monitoring extraction quality.
        
        Args:
            doc_id: Document ID
            chunks_for_extraction: List of chunks processed
            entity_dict: Dictionary of extracted entities
            relationship_dict: Dictionary of extracted relationships
            created_entities: Count of entities created
            created_relationships: Count of relationships created
            parse_errors: Count of parse failures during extraction (Issue #4)
        """

        total_chunks = len(chunks_for_extraction)
        referenced_chunks = {
            cid
            for entity in entity_dict.values()
            for cid in (entity.source_text_units or entity.source_chunks or [])
            if cid
        }
        relationships_requested = sum(len(rels) for rels in relationship_dict.values())
        coverage = (
            len(referenced_chunks) / total_chunks
            if total_chunks > 0
            else 0.0
        )

        metrics = {
            "document_id": doc_id,
            "chunks_processed": total_chunks,
            "entities_created": created_entities,
            "relationships_created": created_relationships,
            "relationships_requested": relationships_requested,
            "chunk_coverage": round(coverage, 3),
            "entities_per_chunk": round(
                created_entities / total_chunks, 3
            )
            if total_chunks
            else 0.0,
            "unique_chunks_with_entities": len(referenced_chunks),
            "parse_errors": parse_errors,  # Issue #4: Track extraction failures
        }

        logger.info(
            "Entity extraction metrics for %s — chunks: %s, entities: %s, "
            "relationships: %s (requested %s), coverage: %.1f%%",
            doc_id,
            total_chunks,
            created_entities,
            created_relationships,
            relationships_requested,
            coverage * 100,
        )
        return metrics

    def _persist_extraction_results(
        self,
        doc_id: str,
        chunks_for_extraction: List[Dict[str, Any]],
        entity_dict,
        relationship_dict,
    ) -> tuple[int, int, Dict[str, Any]]:
        """Persist entities/relationships and return metrics.

        When settings.sync_entity_embeddings is True, use a fully synchronous path to avoid
        background event loop/thread issues (important for test determinism). Otherwise fall
        back to the existing async parallel creation implementation.
        """

        if getattr(settings, "sync_entity_embeddings", False):
            created_entities = 0
            created_relationships = 0

            # Synchronous entity creation with embeddings
            for entity in entity_dict.values():
                try:
                    entity_id = self._generate_entity_id(entity.name)
                    graph_db.create_entity_node(
                        entity_id,
                        entity.name,
                        entity.type,
                        entity.description,
                        entity.importance_score,
                        entity.source_text_units or entity.source_chunks or [],
                        entity.source_text_units or entity.source_chunks or [],
                    )
                    # Link source chunks
                    for chunk_id in (
                        entity.source_text_units or entity.source_chunks or []
                    ):
                        try:
                            graph_db.create_chunk_entity_relationship(
                                chunk_id, entity_id
                            )
                        except Exception:
                            pass
                    created_entities += 1
                except Exception as e:
                    logger.error(
                        "Synchronous entity persistence failed for %s in doc %s: %s",
                        entity.name,
                        doc_id,
                        e,
                    )

            # Relationships
            for relationships in relationship_dict.values():
                for relationship in relationships:
                    try:
                        source_id = self._generate_entity_id(
                            relationship.source_entity
                        )
                        target_id = self._generate_entity_id(
                            relationship.target_entity
                        )
                        graph_db.create_entity_relationship(
                            entity_id1=source_id,
                            entity_id2=target_id,
                            relationship_type=relationship.relationship_type
                            or "RELATED_TO",
                            description=relationship.description,
                            strength=relationship.strength,
                            source_chunks=(
                                relationship.source_text_units
                                or relationship.source_chunks
                                or []
                            ),
                        )
                        created_relationships += 1
                    except Exception:
                        pass
        else:
            def _run_async_creation():
                return self._create_entities_async(
                    entity_dict, relationship_dict, doc_id
                )

            try:
                created_entities, created_relationships = self._run_async(
                    _run_async_creation()
                )
            except RuntimeError:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    created_entities, created_relationships = loop.run_until_complete(
                        _run_async_creation()
                    )
                else:
                    created_entities, created_relationships = loop.run_until_complete(
                        _run_async_creation()
                    )

        metrics = self._build_extraction_metrics(
            doc_id,
            chunks_for_extraction,
            entity_dict,
            relationship_dict,
            created_entities,
            created_relationships,
        )
        import json
        # Convert metrics dict to JSON string for Neo4j storage (Neo4j only accepts primitives)
        metrics_json = json.dumps(metrics)
        graph_db.create_document_node(
            doc_id, {"entity_extraction_metrics": metrics_json}
        )
        return created_entities, created_relationships, metrics

    def _start_entity_operation(self, doc_id: str, doc_name: str) -> str:
        """Start tracking an entity extraction operation."""
        operation_id = self._generate_operation_id(doc_id)
        with self._operations_lock:
            self._entity_extraction_operations[operation_id] = EntityExtractionStatus(
                operation_id=operation_id,
                document_id=doc_id,
                document_name=doc_name,
                state=EntityExtractionState.STARTING,
                started_at=time.time(),
                last_updated=time.time(),
            )
        return operation_id

    def _update_entity_operation(
        self,
        operation_id: str,
        state: EntityExtractionState,
        progress_info: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Update the state of an entity extraction operation."""
        with self._operations_lock:
            if operation_id in self._entity_extraction_operations:
                status = self._entity_extraction_operations[operation_id]
                status.state = state
                status.last_updated = time.time()
                if progress_info:
                    status.progress_info = progress_info
                if error_message:
                    status.error_message = error_message

    def _complete_entity_operation(self, operation_id: str):
        """Mark an entity extraction operation as completed and remove it from tracking."""
        with self._operations_lock:
            if operation_id in self._entity_extraction_operations:
                del self._entity_extraction_operations[operation_id]

    def _cleanup_stale_operations(self, max_age_seconds: float = 3600):
        """Remove operations that are too old (likely stale due to crashes)."""
        current_time = time.time()
        with self._operations_lock:
            stale_ops = [
                op_id
                for op_id, status in self._entity_extraction_operations.items()
                if current_time - status.started_at > max_age_seconds
            ]
            for op_id in stale_ops:
                logger.warning(f"Removing stale entity extraction operation: {op_id}")
                del self._entity_extraction_operations[op_id]

    def _extract_metadata(
        self, file_path: Path, original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata from file."""
        import mimetypes
        
        # Use original filename if provided, otherwise use file path name
        filename = original_filename if original_filename else file_path.name
        # Replace spaces with underscores for cleaner database storage
        if original_filename and " " in filename:
            filename = filename.replace(" ", "_")

        # Detect MIME type based on file extension
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            # Fallback to generic binary type if detection fails
            mime_type = "application/octet-stream"

        try:
            stat_result = file_path.stat()
            file_size = stat_result.st_size
            created_at = stat_result.st_ctime
            modified_at = stat_result.st_mtime
        except FileNotFoundError:
            logger.warning("File not found when extracting metadata for %s; using defaults", file_path)
            file_size = 0
            created_at = modified_at = time.time()

        return {
            "filename": filename,
            "original_filename": original_filename if original_filename else file_path.name,
            "file_path": str(file_path),
            "file_size": file_size,
            "file_extension": file_path.suffix,
            "mime_type": mime_type,
            "created_at": created_at,
            "modified_at": modified_at,
            "link": "",  # Empty string instead of None to avoid Neo4j warnings
        }

    def _derive_content_primary_type(self, file_extension: Optional[str]) -> str:
        """Derive a simple primary content type from a file extension.

        Returns a small set of canonical types (pdf, image, text, word, presentation,
        spreadsheet, unknown) used to populate the `content_primary_type` property
        on Document nodes.
        """
        ext = (file_extension or "").lower()
        mapping = {
            ".pdf": "pdf",
            ".docx": "word",
            ".doc": "word",
            ".txt": "text",
            ".md": "text",
            ".py": "text",
            ".js": "text",
            ".html": "text",
            ".css": "text",
            ".csv": "spreadsheet",
            ".xlsx": "spreadsheet",
            ".xls": "spreadsheet",
            ".pptx": "presentation",
            ".ppt": "presentation",
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".tiff": "image",
            ".bmp": "image",
        }
        return mapping.get(ext, "unknown")

    def _load_category_config(self) -> Dict[str, Any]:
        """Load category configuration from `config/document_categories.json`."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "document_categories.json"
            with open(config_path, "r", encoding="utf-8") as f:
                import json as _json
                return _json.load(f)
        except Exception as e:
            logger.debug(f"Failed to load category config: {e}")
            return {"categories": {"general": {"title": "General", "keywords": []}}}

    def classify_document_categories(
        self,
        filename: str,
        content: str,
    ) -> Dict[str, Any]:
        """Classify document into categories using LLM with heuristic fallback.

        Returns dict with keys: categories (List[str]), confidence (float), keywords (List[str]), difficulty (str).
        """
        try:
            from core.llm import llm_manager
            config = self._load_category_config()
            cats = config.get("categories", {})
            cats_lines = []
            for cid, data in cats.items():
                title = data.get("title", cid)
                kws = ", ".join(data.get("keywords", [])[:8])
                cats_lines.append(f"- {cid}: {title} (keywords: {kws})")
            categories_desc = "\n".join(cats_lines)

            prompt = (
                "You are a document classification assistant. "
                "Classify the document into 1-2 categories and extract 3-6 keywords.\n\n"
                f"Filename: {filename}\n\n"
                "Content (truncated to first 1200 chars):\n" + content[:1200] + "\n\n"
                "Available Categories:\n" + categories_desc + "\n\n"
                "Respond strictly as JSON: {\n"
                "  \"categories\": [\"install\"],\n"
                "  \"confidence\": 0.8,\n"
                "  \"keywords\": [\"setup\", \"prerequisites\"],\n"
                "  \"difficulty\": \"beginner\"\n"
                "}"
            )

            resp = llm_manager.generate_response(
                prompt=prompt,
                model=getattr(settings, "classification_model", "gpt-4o-mini"),
                temperature=0.1,
                max_tokens=200,
            )
            text = (resp or "").strip()
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()
            import json as _json
            parsed = _json.loads(text)
            categories = parsed.get("categories") or [getattr(settings, "classification_default_category", "general")]
            confidence = float(parsed.get("confidence", 0.0))
            keywords = parsed.get("keywords") or []
            difficulty = parsed.get("difficulty") or "intermediate"
            return {
                "categories": categories,
                "confidence": confidence,
                "keywords": keywords,
                "difficulty": difficulty,
            }
        except Exception as e:
            logger.debug(f"Classification failed, using heuristic fallback: {e}")
            # Simple keyword heuristic
            lower = content.lower()
            categories = []
            if any(k in lower for k in ("install", "setup", "prerequisite")):
                categories.append("install")
            if any(k in lower for k in ("config", "setting", "parameter")):
                categories.append("configure")
            if any(k in lower for k in ("error", "fail", "diagnostic", "log")):
                categories.append("troubleshoot")
            if not categories:
                categories = [getattr(settings, "classification_default_category", "general")]
            keywords = []
            return {"categories": categories[:2], "confidence": 0.5, "keywords": keywords, "difficulty": "intermediate"}

    def _run_async(self, coro):
        """Helper to run a coroutine from a synchronous context safely."""
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            msg = str(e).lower()
            if "already running" in msg or "running event loop" in msg:
                try:
                    import nest_asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        nest_asyncio.apply(loop)
                        return loop.run_until_complete(coro)
                    return loop.run_until_complete(coro)
                except Exception:
                    raise e
            raise e

    async def process_file_async(
        self,
        chunks: List[Dict[str, Any]],
        doc_id: str,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously process file chunks: embedding generation and persistence.

        Args:
            chunks: list of chunk dicts
            doc_id: document id
            progress_callback: optional callback function to report progress (chunk_processed_count)

        Returns:
            List of processed chunk summaries
        """
        processed_chunks: List[Dict[str, Any]] = []
        processed_count = 0

        concurrency = getattr(settings, "embedding_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _embed_and_store(chunk):
            nonlocal processed_count
            content = chunk["content"]
            chunk_id = chunk["chunk_id"]
            metadata = chunk.get("metadata", {})

            async with sem:
                try:
                    logger.debug("Requesting embedding for chunk %s (len=%s)", chunk_id, len(content) if content else 0)
                    # Rate limiting is now handled in EmbeddingManager
                    embedding = await embedding_manager.aget_embedding(content)
                    logger.debug(
                        "Received embedding for chunk %s (len=%s)",
                        chunk_id,
                        len(embedding) if embedding is not None else 0,
                    )
                except Exception as e:
                    logger.error(f"Async embedding failed for {chunk_id}: {e}")
                    embedding = []

            # Persist to DB in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            try:
                logger.debug("Persisting chunk %s to DB (doc: %s)", chunk_id, doc_id)
                executor = get_blocking_executor()
                await loop.run_in_executor(
                    executor,
                    graph_db.create_chunk_node,
                    chunk_id,
                    doc_id,
                    content,
                    embedding,
                    metadata,
                )
                logger.debug("Persisted chunk %s to DB", chunk_id)
            except RuntimeError as e:
                logger.debug(
                    f"Blocking executor unavailable while persisting chunk {chunk_id}: {e}."
                )
                if SHUTTING_DOWN:
                    logger.info(
                        "Process shutting down; aborting persist for chunk %s",
                        chunk_id,
                    )
                else:
                    try:
                        executor = get_blocking_executor()
                        await loop.run_in_executor(
                            executor,
                            graph_db.create_chunk_node,
                            chunk_id,
                            doc_id,
                            content,
                            embedding,
                            metadata,
                        )
                        logger.debug("Persisted chunk %s to DB", chunk_id)
                    except Exception as e2:
                        logger.error("Failed to persist chunk %s to DB: %s", chunk_id, e2)

            # Report progress
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count)

            return {"chunk_id": chunk_id, "content": content, "metadata": metadata}

        tasks = [asyncio.create_task(_embed_and_store(c)) for c in chunks]

        for coro in asyncio.as_completed(tasks):
            # Check for cancellation
            if progress_callback and hasattr(progress_callback, 'is_cancelled') and progress_callback.is_cancelled():
                 logger.info("Cancellation detected in process_file_async")
                 for t in tasks:
                     t.cancel()
                 # Wait for tasks to cancel to avoid warnings
                 await asyncio.gather(*tasks, return_exceptions=True)
                 raise asyncio.CancelledError("Processing cancelled by user")

            try:
                res = await coro
                processed_chunks.append(res)
            except asyncio.CancelledError:
                 raise
            except Exception as e:
                logger.error(f"Error in embedding task: {e}")

        return processed_chunks

    async def _persist_with_entity_graph(
        self,
        entity_dict: Dict,
        relationship_dict: Dict,
        doc_id: str
    ) -> Dict[str, Any]:
        """
        Persist entities/relationships via NetworkX EntityGraph (Phase 2).
        
        Flow:
        1. Build EntityGraph from extracted entities
        2. Apply deduplication and accumulation
        3. Filter by importance/strength thresholds
        4. Convert to batch UNWIND queries
        5. Execute in single transaction
        6. Return metrics
        
        Returns:
            Dict with metrics (entities_unique, relationships_unique, etc.)
        """
        try:
            logger.info(f"Using Phase 2 (NetworkX) persistence for doc {doc_id}")
            
            # Build EntityGraph
            entity_graph = EntityGraph()
            
            # Add all entities (with importance filtering)
            entities_added = 0
            for entity in entity_dict.values():
                if entity.importance_score >= settings.importance_score_threshold:
                    entity_graph.add_entity(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        importance_score=entity.importance_score,
                        source_chunks=entity.source_chunks or []
                    )
                    entities_added += 1
            
            logger.debug(f"Added {entities_added} entities to graph (filtered by importance >= {settings.importance_score_threshold})")
            
            # Add all relationships (with strength filtering)
            relationships_added = 0
            for relationships in relationship_dict.values():
                for rel in relationships:
                    if rel.strength >= settings.strength_threshold:
                        entity_graph.add_relationship(
                            source=rel.source_entity,
                            target=rel.target_entity,
                            rel_type=rel.relationship_type or "RELATED_TO",
                            description=rel.description or "",
                            strength=rel.strength,
                            source_chunks=rel.source_chunks or []
                        )
                        relationships_added += 1
            
            logger.debug(f"Added {relationships_added} relationships to graph (filtered by strength >= {settings.strength_threshold})")
            
            # === PHASE 4: Summarize Descriptions (if enabled) ===
            summarization_stats = {"status": "disabled"}
            if settings.enable_description_summarization:
                logger.info("Phase 4: Summarizing entity/relationship descriptions")
                try:
                    summarization_stats = await entity_graph.summarize_descriptions()
                    logger.info(
                        f"Summarization complete: {summarization_stats.get('entities_summarized', 0)} entities, "
                        f"{summarization_stats.get('relationships_summarized', 0)} relationships, "
                        f"{summarization_stats.get('average_compression_ratio', 1.0):.1%} compression"
                    )
                except Exception as e:
                    logger.error(f"Phase 4 summarization failed: {e}", exc_info=True)
                    summarization_stats = {"status": "error", "error": str(e)}
            
            # Get graph stats before persistence
            stats = entity_graph.get_stats()
            
            # Convert to batch queries
            entity_query, entity_params, rel_query, rel_params = \
                entity_graph.to_neo4j_batch_queries(doc_id)
            
            # Execute batch insert
            batch_result = graph_db.execute_batch_unwind(
                entity_query, entity_params,
                rel_query, rel_params
            )
            
            logger.info(
                f"Phase 2 persistence complete: {stats['node_count']} unique entities, "
                f"{stats['edge_count']} unique relationships, "
                f"{stats['orphan_count']} orphans, "
                f"{batch_result['batches']} batches"
            )
            
            # Return metrics
            return {
                "status": "success",
                "phase": "phase2",
                "entities_extracted": len(entity_dict),
                "entities_unique": stats["node_count"],
                "entities_merged": len(entity_dict) - stats["node_count"],
                "relationships_extracted": sum(len(rels) for rels in relationship_dict.values()),
                "relationships_unique": stats["edge_count"],
                "orphan_entities": stats["orphan_count"],
                "batches": batch_result["batches"],
                "entities_created": batch_result.get("entities_created", stats["node_count"]),
                "relationships_created": batch_result.get("relationships_created", stats["edge_count"]),
                "summarization": summarization_stats  # Phase 4 stats
            }
        
        except Exception as e:
            logger.error(f"Phase 2 persistence failed for doc {doc_id}: {e}", exc_info=True)
            raise
    
    async def _create_entities_async(
        self, entity_dict, relationship_dict, doc_id_local: str
    ) -> tuple[int, int]:
        """
        Create entities and relationships - routes to Phase 1 or Phase 2.
        
        Phase 1 (default): Individual transactions per entity/relationship
        Phase 2 (opt-in): NetworkX intermediate graph + batch UNWIND
        """
        # Check if Phase 2 is enabled
        if settings.enable_phase2_networkx:
            try:
                # Issue #12 Fix: Removed nested asyncio.run and run_in_executor.
                # Since we are already in an async method, we should await directly.
                # CPU-intensive graph building is handled internally if needed,
                # but nested loops are dangerous and inefficient.
                metrics = await self._persist_with_entity_graph(
                    entity_dict, relationship_dict, doc_id_local
                )
                return (metrics["entities_unique"], metrics["relationships_unique"])
            except Exception as e:
                logger.error(f"Phase 2 failed, falling back to Phase 1: {e}")
                # Fall through to Phase 1
        
        # Phase 1 implementation (original)
        logger.info(f"Using Phase 1 (direct) persistence for doc {doc_id_local}")
        created_entities = 0
        created_relationships = 0

        # Process entities in parallel with controlled concurrency
        concurrency = getattr(settings, "embedding_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _create_single_entity(entity):
            nonlocal created_entities
            async with sem:
                try:
                    entity_id = self._generate_entity_id(entity.name)
                    # Rate limiting is now handled in EmbeddingManager
                    # Use async entity creation
                    await graph_db.acreate_entity_node(
                        entity_id,
                        entity.name,
                        entity.type,
                        entity.description,
                        entity.importance_score,
                        entity.source_text_units or entity.source_chunks or [],
                        entity.source_text_units or entity.source_chunks or [],
                    )
                    # Link chunks to entity (run in executor to avoid blocking)
                    loop = asyncio.get_running_loop()
                    for chunk_id in (
                        entity.source_text_units or entity.source_chunks or []
                    ):
                        try:
                            try:
                                executor = get_blocking_executor()
                                await loop.run_in_executor(
                                    executor,
                                    graph_db.create_chunk_entity_relationship,
                                    chunk_id,
                                    entity_id,
                                )
                            except RuntimeError as e:
                                logger.debug(
                                    f"Blocking executor unavailable while creating chunk-entity rel {chunk_id}->{entity_id}: {e}"
                                )
                                if SHUTTING_DOWN:
                                    logger.info(
                                        "Process shutting down; aborting chunk-entity rel %s->%s",
                                        chunk_id,
                                        entity_id,
                                    )
                                    continue
                                executor = get_blocking_executor()
                                await loop.run_in_executor(
                                    executor,
                                    graph_db.create_chunk_entity_relationship,
                                    chunk_id,
                                    entity_id,
                                )
                        except Exception as e:
                            logger.error(
                                f"Failed to create chunk-entity rel {chunk_id}->{entity_id}: {e}"
                            )
                    created_entities += 1
                    return entity_id
                except Exception as e:
                    logger.error(
                        f"Failed to persist entity {entity.name} for doc {doc_id_local}: {e}"
                    )
                    return None

        # Create all entities in parallel
        if entity_dict:
            tasks = [
                asyncio.create_task(_create_single_entity(entity))
                for entity in entity_dict.values()
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    logger.error(f"Error in entity creation task: {e}")

        # Create relationships after all entities are created
        # Store entity relationships
        for relationships in relationship_dict.values():
            for relationship in relationships:
                try:
                    source_id = self._generate_entity_id(relationship.source_entity)
                    target_id = self._generate_entity_id(relationship.target_entity)
                    graph_db.create_entity_relationship(
                        entity_id1=source_id,
                        entity_id2=target_id,
                        relationship_type=relationship.relationship_type
                        or "RELATED_TO",
                        description=relationship.description,
                        strength=relationship.strength,
                        source_chunks=(
                            relationship.source_text_units
                            or relationship.source_chunks
                            or []
                        ),
                    )
                    created_relationships += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to persist relationship for doc {doc_id_local}: {e}"
                    )

        return created_entities, created_relationships

    def process_file(
        self,
        file_path: Path,
        original_filename: Optional[str] = None,
        progress_callback=None,
        enable_quality_filtering: Optional[bool] = None,
        document_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file and store it in the graph database.

        Args:
            file_path: Path to the file to process
            original_filename: Optional original filename to preserve (useful for uploaded files)
            progress_callback: Optional callback function to report chunk processing progress
            enable_quality_filtering: Optional override for quality filtering (if None, uses global setting)

        Returns:
            Processing result dictionary or None if failed
        """
        try:
            # Apply RAG tuning config overrides (runtime sync from UI)
            from config.settings import apply_rag_tuning_overrides
            apply_rag_tuning_overrides(settings)
            
            use_quality_filtering = (
                enable_quality_filtering
                if enable_quality_filtering is not None
                else self.enable_quality_filtering
            )

            start_time = time.time()
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Convert the document to Markdown content using format-specific converters
            conversion_result = self.converter.convert(file_path, original_filename)
            if not conversion_result or not conversion_result.get("content"):
                logger.warning("No content extracted from %s", file_path)
                return None

            content = conversion_result.get("content", "")

            # Generate document ID and extract metadata
            doc_id = document_id or self._generate_document_id(file_path)
            metadata = self._extract_metadata(file_path, original_filename)
            metadata.update(conversion_result.get("metadata", {}))

            # Ensure content_primary_type is set (derive from file extension if missing)
            metadata["content_primary_type"] = metadata.get(
                "content_primary_type"
            ) or self._derive_content_primary_type(metadata.get("file_extension"))

            processing_state = {
                "processing_status": "processing",
                "processing_stage": "conversion",
                "processing_progress": 5.0,
                "processing_error": None,
                # Store chunking parameters for incremental update validation
                "chunk_size_used": settings.chunk_size,
                "chunk_overlap_used": settings.chunk_overlap,
            }

            # Create document node
            graph_db.create_document_node(doc_id, {**metadata, **processing_state})

            # Chunk the document with enhanced processing
            # Classification Stage
            classification_result = None
            logger.info(f"DEBUG: Checking classification setting: {getattr(settings, 'enable_document_classification', 'MISSING')}")
            if getattr(settings, "enable_document_classification", True):
                time.sleep(2)
                logger.info(f"!!! STARTING CLASSIFICATION FOR {doc_id} !!!")
                graph_db.create_document_node(
                    doc_id,
                    {"processing_stage": "classification", "processing_progress": 10.0},
                )
                classification_result = self.classify_document_categories(
                    filename=metadata.get("filename", ""),
                    content=content,
                )
                metadata.update(
                    {
                        "categories": classification_result.get("categories"),
                        "classification_confidence": classification_result.get("confidence"),
                        "keywords": classification_result.get("keywords"),
                        "difficulty": classification_result.get("difficulty"),
                    }
                )
                # Update node with classification data
                graph_db.create_document_node(doc_id, metadata)

            graph_db.create_document_node(
                doc_id,
                {"processing_stage": "chunking", "processing_progress": 25.0},
            )

            if use_quality_filtering:
                chunks = document_chunker.chunk_text(
                    content,
                    doc_id,
                    enable_quality_filtering=use_quality_filtering,
                    enable_ocr_enhancement=False,  # OCR already applied by smart loaders
                )
                logger.info(
                    f"Used enhanced chunking (Quality filtering: {use_quality_filtering}) for {doc_id}"
                )
            else:
                chunks = document_chunker.chunk_text(content, doc_id)
                logger.info(f"Used standard chunking for {doc_id}")

            # Annotate chunk-level positional metadata
            for idx, chunk in enumerate(chunks, start=1):
                chunk_metadata = chunk.get("metadata", {}) or {}
                chunk_metadata.update(
                    {
                        "document_id": doc_id,
                        "source_document": metadata.get("filename"),
                        "chunk_number": idx,
                    }
                )
                # Inherit section title if provided by converter/chunker
                section_title = chunk_metadata.get("section_title") or None
                if section_title:
                    chunk_metadata["semantic_group"] = section_title.lower().replace(" ", "_")
                chunk["metadata"] = chunk_metadata

            # Extract document summary, type, and hashtags after chunking
            if progress_callback:
                try:
                    progress_callback(0, message="Generating abstract")
                except TypeError:
                    progress_callback(0)
            logger.info(f"Extracting summary for document {doc_id}")
            summary_data = document_summarizer.extract_summary(chunks)
            graph_db.create_document_node(
                doc_id,
                {"processing_stage": "summarization", "processing_progress": 60.0},
            )

            # Update document node with summary information
            graph_db.update_document_summary(
                doc_id=doc_id,
                summary=summary_data.get("summary", ""),
                document_type=summary_data.get("document_type", "other"),
                hashtags=summary_data.get("hashtags", []),
            )
            # Document classification and metadata enrichment
            if getattr(settings, "enable_document_classification", False):
                    graph_db.create_document_node(
                        doc_id,
                        {"processing_stage": "metadata_enrichment", "processing_progress": 62.0},
                    )
                    
                    if classification_result:
                        cls = classification_result
                    else:
                        cls = self.classify_document_categories(
                            metadata.get("filename"), content
                        )
                    
                    confidence = cls.get("confidence", 0.0)
                    categories = cls.get("categories", [])
                    apply_cls = confidence >= getattr(
                        settings, "classification_confidence_threshold", 0.7
                    )
                    doc_category = (
                        categories[0]
                        if (categories and apply_cls)
                        else getattr(settings, "classification_default_category", "general")
                    )
                    enrich = {
                        "category": doc_category,
                        "categories": categories,
                        "classification_confidence": confidence,
                        "keywords": cls.get("keywords", []),
                        "difficulty": cls.get("difficulty", "intermediate"),
                    }
                    graph_db.create_document_node(doc_id, enrich)
                    # Propagate category to chunks in-memory so metadata is stored during embedding persist
                    for c in chunks:
                        md = c.get("metadata", {}) or {}
                        md["category"] = doc_category
                        c["metadata"] = md
                    logger.info(
                        f"Classified document {doc_id} → {doc_category} (confidence={confidence:.2f})"
                    )
            logger.info(
                f"Summary extracted for {doc_id}: type={summary_data.get('document_type')}, "
                f"hashtags={len(summary_data.get('hashtags', []))}"
            )

            # Apply content quality filtering before embedding (if enabled)
            if getattr(settings, "enable_content_filtering", False):
                graph_db.create_document_node(
                    doc_id,
                    {"processing_stage": "content_filtering", "processing_progress": 68.0},
                )
                logger.info(f"Applying content quality filtering to {len(chunks)} chunks")
                content_filter = get_content_filter()

                # Prepare chunks for filtering with enhanced metadata
                chunks_to_filter = []
                for chunk in chunks:
                    chunk_metadata = chunk.get("metadata", {})

                    # Add file type information for filter logic
                    chunk_metadata["file_type"] = metadata.get("file_extension", "")

                    # Mark conversation threads (if detected)
                    chunk_metadata["is_conversation"] = False  # Can be enhanced with detection logic

                    # Mark structured data
                    file_ext = metadata.get("file_extension", "").lower()
                    chunk_metadata["is_structured_data"] = file_ext in [".csv", ".xlsx", ".xls"]

                    # Mark code files
                    chunk_metadata["is_code"] = file_ext in [".py", ".js", ".java", ".cpp", ".html", ".css"]

                    chunks_to_filter.append({
                        "content": chunk.get("content", ""),
                        "metadata": chunk_metadata,
                        "original": chunk  # Keep reference to original
                    })

                # Filter chunks
                filtered_chunks_data, filter_metrics = content_filter.filter_chunks(chunks_to_filter)

                # Extract original chunks that passed filtering
                chunks = [item["original"] for item in filtered_chunks_data]

                # Log filtering metrics
                metrics_summary = filter_metrics.get_summary()
                logger.info(
                    f"Content filtering complete for {doc_id}: "
                    f"{metrics_summary['passed_chunks']}/{metrics_summary['total_chunks']} chunks passed "
                    f"({metrics_summary['pass_rate']:.1f}% pass rate, "
                    f"{metrics_summary['filter_rate']:.1f}% filtered)"
                )

                if metrics_summary['filter_reasons']:
                    logger.info(f"Filter reasons: {metrics_summary['filter_reasons']}")

                # Reset metrics for next document
                content_filter.reset_metrics()
            else:
                logger.debug("Content filtering disabled, processing all chunks")

            # Process chunks asynchronously with configurable concurrency (embeddings + storing)
            graph_db.create_document_node(
                doc_id,
                {"processing_stage": "embedding", "processing_progress": 75.0},
            )

            processed_chunks = self._run_async(
                self.process_file_async(chunks, doc_id, progress_callback)
            )

            # After chunk processing finishes, schedule entity extraction in background (if enabled)
            entity_count = 0
            relationship_count = 0

            should_extract_entities = settings.enable_entity_extraction
            if should_extract_entities and self.entity_extractor is None:
                # Initialize entity extractor if the global setting enabled it at startup
                logger.info(
                    "Initializing entity extractor for background processing..."
                )
                self.entity_extractor = EntityExtractor()

            if should_extract_entities and self.entity_extractor:
                extractor = self.entity_extractor
                chunks_for_extraction = self._prepare_chunks_for_extraction(
                    doc_id, processed_chunks
                )
                logger.info(
                    "Scheduling entity extraction with retry/backoff for %s chunks (doc: %s)",
                    len(chunks_for_extraction),
                    doc_id,
                )

                # Synchronous path (tests/deterministic runs) when sync_entity_embeddings enabled
                if getattr(settings, "sync_entity_embeddings", False):
                    # Determine gleaning configuration
                    use_gleaning, max_gleanings = self._get_gleaning_config(document_type=None)
                    
                    if use_gleaning:
                        logger.info(
                            "Running synchronous GLEANING extraction for %s chunks (max_gleanings=%s)...",
                            len(chunks_for_extraction),
                            max_gleanings
                        )
                    else:
                        logger.info(
                            "Running synchronous entity extraction (SYNC_ENTITY_EMBEDDINGS enabled) for %s chunks...",
                            len(chunks_for_extraction),
                        )
                    
                    try:
                        try:
                            if use_gleaning:
                                entity_dict, relationship_dict = self._run_async(
                                    extractor.extract_from_chunks_with_gleaning(
                                        chunks_for_extraction,
                                        max_gleanings=max_gleanings,
                                        document_type=None
                                    )
                                )
                            else:
                                entity_dict, relationship_dict = self._run_async(
                                    extractor.extract_from_chunks(chunks_for_extraction)
                                )
                        except RuntimeError:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                if use_gleaning:
                                    entity_dict, relationship_dict = loop.run_until_complete(
                                        extractor.extract_from_chunks_with_gleaning(
                                            chunks_for_extraction,
                                            max_gleanings=max_gleanings,
                                            document_type=None
                                        )
                                    )
                                else:
                                    entity_dict, relationship_dict = loop.run_until_complete(
                                        extractor.extract_from_chunks(chunks_for_extraction)
                                    )
                            else:
                                if use_gleaning:
                                    entity_dict, relationship_dict = loop.run_until_complete(
                                        extractor.extract_from_chunks_with_gleaning(
                                            chunks_for_extraction,
                                            max_gleanings=max_gleanings,
                                            document_type=None
                                        )
                                    )
                                else:
                                    entity_dict, relationship_dict = loop.run_until_complete(
                                        extractor.extract_from_chunks(chunks_for_extraction)
                                    )

                        created_entities, created_relationships, metrics = self._persist_extraction_results(
                            doc_id, chunks_for_extraction, entity_dict, relationship_dict
                        )
                        entity_count = created_entities
                        relationship_count = created_relationships

                        # Create entity similarities and validate embeddings
                        try:
                            graph_db.create_entity_similarities(doc_id)
                        except Exception as e:
                            logger.debug(
                                "Failed to create entity similarities for %s (sync path): %s",
                                doc_id,
                                e,
                            )
                        try:
                            graph_db.validate_entity_embeddings(doc_id)
                        except Exception as e:
                            logger.debug(
                                "Failed to validate entity embeddings for %s (sync path): %s",
                                doc_id,
                                e,
                            )
                        # Ensure any missing chunk->entity relationships for this document are repaired.
                        try:
                            repair_stats = graph_db.repair_contains_entity_relationships_for_document(doc_id)
                            logger.info(
                                "Post-ingest CONTAINS_ENTITY repair for %s: created=%s (before=%s after=%s)",
                                doc_id,
                                repair_stats.get("created"),
                                repair_stats.get("before"),
                                repair_stats.get("after"),
                            )
                        except Exception as e:
                            logger.debug(
                                "Failed to run document-scoped CONTAINS_ENTITY repair for %s: %s",
                                doc_id,
                                e,
                            )
                        # Update precomputed summary counts for fast document loads
                        graph_db.create_document_node(
                            doc_id,
                            {"processing_stage": "post_processing", "processing_progress": 95.0},
                        )
                        try:
                            if getattr(settings, "enable_document_summaries", True):
                                stats = graph_db.update_document_precomputed_summary(doc_id)
                                logger.info(
                                    "Updated precomputed document summary for %s: %s",
                                    doc_id,
                                    stats,
                                )
                                try:
                                    preview_stats = graph_db.update_document_preview(doc_id)
                                    logger.info(
                                        "Updated document preview for %s: %s",
                                        doc_id,
                                        preview_stats,
                                    )
                                except Exception as e:
                                    logger.debug("Failed to update document preview for %s: %s", doc_id, e)
                        except Exception as e:
                            logger.debug(
                                "Failed to update precomputed document summary for %s: %s",
                                doc_id,
                                e,
                            )
                        # Invalidate caches for this document so frontends see fresh values
                        try:
                            cache = get_response_cache()
                            for key in (f"document_summary:{doc_id}", f"document_metadata:{doc_id}"):
                                try:
                                    if key in cache:
                                        del cache[key]
                                except Exception:
                                    pass
                            try:
                                from core.cache_metrics import cache_metrics as _cache_metrics
                                _cache_metrics.record_response_invalidation()
                            except Exception:
                                pass
                        except Exception:
                            pass
                        logger.info(
                            "Synchronous entity extraction finished for %s: %s entities, %s relationships",
                            doc_id,
                            created_entities,
                            created_relationships,
                        )
                    except Exception as sync_e:
                        logger.error(
                            "Synchronous entity extraction failed for %s: %s",
                            doc_id,
                            sync_e,
                        )
                else:
                    # Start background entity extraction in a separate thread
                    self._start_background_extraction(doc_id, chunks_for_extraction, original_filename)

            # Create similarity relationships between chunks
            try:
                relationships_created = graph_db.create_chunk_similarities(doc_id)
                logger.info(
                    f"Created {relationships_created} similarity relationships for document {doc_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create similarity relationships for document {doc_id}: {e}"
                )
                relationships_created = 0

            result = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "chunks_created": len(processed_chunks),
                "entities_created": entity_count,
                "entity_relationships_created": relationship_count,
                "similarity_relationships_created": relationships_created,
                "metadata": metadata,
                "status": "success",
                "processing_status": "completed",
                "processing_stage": "completed",
            }

            graph_db.create_document_node(
                doc_id,
                {
                    "processing_status": "completed",
                    "processing_stage": "completed",
                    "processing_progress": 100.0,
                    "processing_error": None,
                },
            )

            # Create temporal nodes for time-based retrieval
            if settings.enable_temporal_filtering:
                try:
                    graph_db.create_temporal_nodes_for_document(doc_id)
                    logger.debug(f"Created temporal nodes for document {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to create temporal nodes for {doc_id}: {e}")

            logger.info(
                f"Successfully processed {file_path}: {len(processed_chunks)} chunks created"
            )
            # add processing duration
            duration = time.time() - start_time
            result["duration_seconds"] = duration

            # Invalidate response cache after successful processing to avoid stale answers
            try:
                cache = get_response_cache()
                cache.clear()
                logger.info("Cleared response cache after processing document: %s", doc_id)
            except Exception as e:
                logger.warning("Failed to clear response cache after processing: %s", e)

            # Print to stdout for quick feedback
            print(
                f"Processed {file_path} in {duration:.2f}s — {len(processed_chunks)} chunks"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
            try:
                doc_id = document_id or self._generate_document_id(file_path)
                graph_db.create_document_node(
                    doc_id,
                    {
                        "processing_status": "error",
                        "processing_stage": "error",
                        "processing_progress": 0.0,
                        "processing_error": str(e),
                    },
                )
            except Exception:
                pass

            return {
                "file_path": str(file_path),
                "status": "error",
                "error": str(e),
                "processing_status": "error",
                "processing_stage": "error",
            }

    def update_document(
        self,
        doc_id: str,
        file_path: Path,
        original_filename: Optional[str] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Update a document incrementally, preserving unchanged chunks.
        """
        try:
            start_time = time.time()
            
            if not file_path.exists():
                return {
                    "status": "error",
                    "error": f"File not found: {file_path}",
                    "document_id": doc_id,
                }
            
            # Update processing status immediately
            graph_db.create_document_node(
                doc_id,
                {"processing_status": "processing", "processing_stage": "conversion", "processing_progress": 5.0}
            )
            
            if progress_callback:
                try:
                    progress_callback(5, message="Starting incremental update...")
                except TypeError:
                    progress_callback(5)

            # Copy file to background-safe location immediately
            import shutil
            suffix = Path(file_path).suffix
            bg_safe_path = Path(file_path).parent / f"bg_full_upd_{doc_id}_{int(time.time())}{suffix}"
            shutil.copy(file_path, bg_safe_path)

            # Launch EVERYTHING in background: conversion, diffing, embedding, extraction
            t = threading.Thread(
                target=self._background_full_update_worker_impl,
                args=(doc_id, bg_safe_path, original_filename, start_time),
                daemon=True
            )
            with self._bg_lock:
                self._bg_entity_threads.append(t)
            t.start()
            
            # Return immediately to unblock the UI / API
            return {
                "status": "processing",
                "document_id": doc_id,
                "message": "Update initiated successfully and is running in the background."
            }

        except Exception as e:
            logger.error(f"Failed to initiate update for document {doc_id}: {e}", exc_info=True)
            graph_db.create_document_node(
                doc_id,
                {"processing_status": "error", "processing_error": str(e)}
            )
            return {"status": "error", "error": str(e)}

    def _background_full_update_worker_impl(
        self,
        doc_id: str,
        file_path: Path,
        original_filename: Optional[str],
        start_time: float
    ):
        """Complete background worker for handling the entire document update lifecycle."""
        try:
            # 1. Validate chunking parameters
            stored_params = graph_db.get_document_chunking_params(doc_id)
            if (
                stored_params and (
                    stored_params.get("chunk_size_used") != settings.chunk_size or
                    stored_params.get("chunk_overlap_used") != settings.chunk_overlap
                )
            ):
                raise ValueError("Chunking parameters have changed; incremental update not possible.")

            # 2. Conversion Phase
            graph_db.create_document_node(doc_id, {"processing_status": "processing", "processing_stage": "conversion", "processing_progress": 5.0})
            conversion_result = self.converter.convert(file_path, original_filename)
            if not conversion_result or not conversion_result.get("content"):
                raise ValueError("No content extracted from file")
            
            content = conversion_result.get("content", "")
            new_chunks = document_chunker.chunk_text(content, doc_id)

            # 3. Diffing Phase
            graph_db.create_document_node(doc_id, {"processing_status": "processing", "processing_stage": "chunking", "processing_progress": 15.0})
            existing_hashes = graph_db.get_chunk_hashes_for_document(doc_id)
            existing_hash_set = set(existing_hashes.keys())
            
            new_hash_map = {}
            for chunk in new_chunks:
                chunk_hash = chunk.get("metadata", {}).get("content_hash", "")
                if chunk_hash:
                    new_hash_map[chunk_hash] = chunk
            new_hash_set = set(new_hash_map.keys())
            
            unchanged_hashes = existing_hash_set & new_hash_set
            removed_hashes = existing_hash_set - new_hash_set
            added_hashes = new_hash_set - existing_hash_set

            # 4. Deletion Phase
            if removed_hashes:
                chunk_ids_to_delete = [existing_hashes[h] for h in removed_hashes]
                graph_db.delete_chunks_with_entity_cleanup(chunk_ids_to_delete)

            # 5. Embedding Phase
            chunks_to_add = [new_hash_map[h] for h in added_hashes]
            added_chunk_count = len(added_hashes)
            
            if chunks_to_add:
                for idx, chunk in enumerate(chunks_to_add, start=1):
                    chunk_metadata = chunk.get("metadata", {}) or {}
                    chunk_metadata.update({
                        "document_id": doc_id,
                        "chunk_number": idx,
                    })
                    chunk["metadata"] = chunk_metadata

                graph_db.create_document_node(doc_id, {"processing_status": "processing", "processing_stage": "embedding", "processing_progress": 20.0})

                def update_progress_callback(processed_count):
                    total = len(chunks_to_add)
                    if total > 0:
                        progress = 20.0 + (processed_count / total * 55.0)
                        graph_db.create_document_node(doc_id, {"processing_status": "processing", "processing_progress": progress, "processing_stage": "embedding"})

                processed_chunks = self._run_async(self.process_file_async(chunks_to_add, doc_id, update_progress_callback))
                
                # 6. Extraction Phase
                if added_chunk_count > 0 and getattr(settings, "enable_entity_extraction", True):
                    chunks_for_extraction = self._prepare_chunks_for_extraction(doc_id, processed_chunks)
                    graph_db.create_document_node(doc_id, {"processing_status": "processing", "processing_stage": "entity_extraction", "processing_progress": 75.0})
                    self._background_entity_worker_impl(doc_id, chunks_for_extraction, original_filename)
                else:
                    self._finalize_document_update(doc_id, file_path, original_filename, conversion_result.get("metadata", {}))
            else:
                self._finalize_document_update(doc_id, file_path, original_filename, conversion_result.get("metadata", {}))

            # 7. Similarities
            if added_chunk_count > 0 and getattr(settings, 'create_chunk_similarities', True):
                try:
                    graph_db.create_chunk_similarities(doc_id)
                except Exception: pass

            logger.info(f"Background update completed for {doc_id} in {time.time() - start_time:.2f}s")

        except Exception as e:
            logger.error(f"Background full update failed for {doc_id}: {e}", exc_info=True)
            graph_db.create_document_node(doc_id, {"processing_status": "error", "processing_error": str(e)})
        finally:
            if file_path.exists():
                try: file_path.unlink()
                except Exception: pass
            with self._bg_lock:
                curr = threading.current_thread()
                if curr in self._bg_entity_threads: self._bg_entity_threads.remove(curr)

    def _finalize_document_update(self, doc_id, file_path, original_filename, conversion_metadata):
        """Helper to mark document as completed."""
        metadata = self._extract_metadata(file_path, original_filename)
        metadata.update(conversion_metadata)
        metadata.update({
            "processing_status": "completed",
            "processing_stage": "completed",
            "processing_progress": 100.0,
            "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        })
        graph_db.create_document_node(doc_id, metadata)
        
        # Reset global processing state to stop frontend polling
        try:
            from api.routers.database import _global_processing_state
            if _global_processing_state.get("current_document_id") == doc_id:
                _global_processing_state["is_processing"] = False
                _global_processing_state["current_document_id"] = None
                _global_processing_state["current_filename"] = None
                _global_processing_state["current_stage"] = None
                _global_processing_state["progress_percentage"] = 0.0
        except ImportError:
            pass  # Gracefully handle if import fails

    def _background_update_worker_impl(
        self,
        doc_id: str,
        chunks_to_add: List[Dict[str, Any]],
        file_path: Path,
        original_filename: Optional[str],
        conversion_metadata: Dict[str, Any],
        unchanged_hashes: set,
    ):
        # NOTE: This method is now legacy and replaced by _background_full_update_worker_impl
        # but kept for backward compatibility if any other call site exists.
        pass

    def _start_background_extraction(self, doc_id, chunks_for_extraction, original_filename):
        """Start background entity extraction in a separate thread."""
        t = threading.Thread(
            target=self._background_entity_worker_impl,
            args=(doc_id, chunks_for_extraction, original_filename),
            daemon=True
        )
        with self._bg_lock:
            self._bg_entity_threads.append(t)
        t.start()

    def _background_entity_worker_impl(self, doc_id_local, chunks_local, original_filename: Optional[str] = None):
        """Internal helper for background entity extraction."""
        operation_id = None
        try:
            # Get extractor instance (initializing if needed)
            if self.entity_extractor is None:
                self.entity_extractor = EntityExtractor()
            extractor = self.entity_extractor

            # Start tracking
            filename = original_filename if original_filename else doc_id_local
            operation_id = self._start_entity_operation(doc_id_local, filename)
            
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.STARTING,
                "Starting background entity extraction"
            )

            use_gleaning, max_gleanings = self._get_gleaning_config()
            
            # Define granularity callback
            def extraction_progress_callback(processed, total):
                if total == 0: return
                msg = f"Extracting entities ({processed}/{total} chunks)"
                if use_gleaning:
                    msg = f"Gleaning extraction ({max_gleanings} passes) - ({processed}/{total} chunks)"
                
                self._update_entity_operation(operation_id, EntityExtractionState.LLM_EXTRACTION, msg)
                
                # Map 75-95%
                progress = 75.0 + ((processed / total) * 20.0)
                try:
                    graph_db.create_document_node(doc_id_local, {"processing_progress": min(progress, 95.0)})
                except Exception: pass

            # Run extraction
            try:
                if use_gleaning:
                    entity_dict, relationship_dict = self._run_async(
                        extractor.extract_from_chunks_with_gleaning(
                            chunks_local, max_gleanings=max_gleanings, progress_callback=extraction_progress_callback
                        )
                    )
                else:
                    entity_dict, relationship_dict = self._run_async(
                        extractor.extract_from_chunks(
                            chunks_local, progress_callback=extraction_progress_callback
                        )
                    )
            except Exception as e:
                logger.error(f"Extraction failed for {doc_id_local}: {e}")
                self._update_entity_operation(operation_id, EntityExtractionState.ERROR, str(e))
                return

            # Persist
            self._update_entity_operation(operation_id, EntityExtractionState.EMBEDDING_GENERATION, "Generating embeddings")
            created_ents, created_rels, _ = self._persist_extraction_results(
                doc_id_local, chunks_local, entity_dict, relationship_dict
            )
            
            # Validation & Repair
            self._update_entity_operation(operation_id, EntityExtractionState.VALIDATION, "Validating results")
            graph_db.create_entity_similarities(doc_id_local)
            graph_db.validate_entity_embeddings(doc_id_local)
            graph_db.repair_contains_entity_relationships_for_document(doc_id_local)
            
            # Finalize
            graph_db.create_document_node(doc_id_local, {
                "processing_status": "completed",
                "processing_stage": "completed",
                "processing_progress": 100.0
            })
            
            # Cache cleanup
            try:
                from core.singletons import get_response_cache
                cache = get_response_cache()
                for key in (f"document_summary:{doc_id_local}", f"document_metadata:{doc_id_local}"):
                    if key in cache: del cache[key]
            except Exception: pass
            
            logger.info(f"Background extraction finished for {doc_id_local}")

        except Exception as e:
            logger.error(f"Background worker failed for {doc_id_local}: {e}")
            if operation_id:
                self._update_entity_operation(operation_id, EntityExtractionState.ERROR, str(e))
        finally:
            if operation_id:
                self._complete_entity_operation(operation_id)
            with self._bg_lock:
                curr = threading.current_thread()
                if curr in self._bg_entity_threads:
                    self._bg_entity_threads.remove(curr)


    def is_entity_extraction_running(self) -> bool:
        """
        Return True if any background entity extraction operations are currently running.

        This function provides a comprehensive check that considers:
        1. Thread status (alive threads) - both new state-tracked and legacy threads
        2. Operation states (LLM extraction, embedding generation, database operations, validation)
        3. Thread names and activities to detect entity extraction work
        4. Cleanup of stale operations

        Returns:
            bool: True if any entity extraction is in progress, False otherwise
        """
        # First, clean up stale operations (older than 1 hour)
        self._cleanup_stale_operations()

        # Check thread status and operation states
        with self._bg_lock:
            # Clean up dead threads
            self._bg_entity_threads = [
                t for t in self._bg_entity_threads if t.is_alive()
            ]
            has_alive_threads = len(self._bg_entity_threads) > 0

            # Additional check: look for threads that might be doing entity work
            # but weren't tracked by the new system (legacy threads)
            import threading

            all_threads = threading.enumerate()
            entity_related_threads = []

            for thread in all_threads:
                # Check if thread is daemon and potentially doing entity extraction
                if thread.daemon and thread.is_alive():
                    try:
                        # Check thread name for entity-related work
                        thread_name = thread.name.lower()
                        if (
                            "entity" in thread_name
                            or "background" in thread_name
                            or "batch" in thread_name
                            or "global" in thread_name
                        ):
                            # Additional check: not the main thread or our current thread
                            if (
                                thread != threading.current_thread()
                                and thread != threading.main_thread()
                            ):
                                entity_related_threads.append(thread)
                    except (AttributeError, TypeError):
                        # Some threads might not have accessible name info
                        pass

            has_legacy_entity_threads = len(entity_related_threads) > 0

        with self._operations_lock:
            # Check if any operations are in non-completed states
            has_active_operations = any(
                status.state
                in [
                    EntityExtractionState.STARTING,
                    EntityExtractionState.LLM_EXTRACTION,
                    EntityExtractionState.EMBEDDING_GENERATION,
                    EntityExtractionState.DATABASE_OPERATIONS,
                    EntityExtractionState.VALIDATION,
                ]
                for status in self._entity_extraction_operations.values()
            )

        # Return True if any of these conditions are met
        return has_alive_threads or has_legacy_entity_threads or has_active_operations

    def get_entity_extraction_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about ongoing entity extraction operations.

        Returns:
            dict: Detailed status including active operations, their states, and progress
        """
        # Clean up stale operations first
        self._cleanup_stale_operations()

        with self._bg_lock:
            # Clean up dead threads
            self._bg_entity_threads = [
                t for t in self._bg_entity_threads if t.is_alive()
            ]
            thread_count = len(self._bg_entity_threads)

        with self._operations_lock:
            # Copy current operations to avoid locking issues
            operations_copy = dict(self._entity_extraction_operations)

        # Prepare detailed status information
        active_operations = []
        for op_id, status in operations_copy.items():
            runtime = time.time() - status.started_at
            active_operations.append(
                {
                    "operation_id": op_id,
                    "document_id": status.document_id,
                    "document_name": status.document_name,
                    "state": status.state.value,
                    "runtime_seconds": runtime,
                    "progress_info": status.progress_info,
                    "error_message": status.error_message,
                    "last_updated": status.last_updated,
                }
            )

        # Sort by start time (most recent first)
        active_operations.sort(key=lambda x: x["last_updated"], reverse=True)

        return {
            "is_running": len(active_operations) > 0 or thread_count > 0,
            "active_threads": thread_count,
            "active_operations": len(active_operations),
            "operations": active_operations,
            "states_summary": {
                state.value: sum(
                    1 for op in active_operations if op["state"] == state.value
                )
                for state in EntityExtractionState
            },
        }

    def process_directory(
        self, directory_path: Path, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories

        Returns:
            List of processing results
        """
        results = []

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found or not a directory: {directory_path}")
            return results

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                result = self.process_file(file_path)
                if result:
                    results.append(result)

        logger.info(f"Processed directory {directory_path}: {len(results)} files")
        return results

    def process_multiple_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple files.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of processing results
        """
        results = []

        for file_path in file_paths:
            result = self.process_file(file_path)
            if result:
                results.append(result)

        logger.info(f"Processed {len(file_paths)} files: {len(results)} successful")
        return results

    def process_file_chunks_only(
        self,
        file_path: Path,
        original_filename: Optional[str] = None,
        progress_callback=None,
        enable_quality_filtering: Optional[bool] = None,
        document_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a file for chunks only, without entity extraction.
        This is used for batch processing to avoid rate limiting.

        Args:
            file_path: Path to the file to process
            original_filename: Optional original filename to preserve
            progress_callback: Optional callback function to report chunk processing progress

            enable_quality_filtering: Optional override for quality filtering (if None, uses global setting)

        Returns:
            Processing result dictionary or None if failed
        """
        try:
            use_quality_filtering = (
                enable_quality_filtering
                if enable_quality_filtering is not None
                else self.enable_quality_filtering
            )

            start_time = time.time()
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Generate document ID and extract metadata (needed for all cases)
            doc_id = document_id or self._generate_document_id(file_path)
            metadata = self._extract_metadata(file_path, original_filename)

            logger.info(f"Processing file chunks only: {file_path}")

            conversion_result = self.converter.convert(file_path, original_filename)
            if not conversion_result or not conversion_result.get("content"):
                logger.warning("No content extracted from %s", file_path)
                return None

            content = conversion_result.get("content", "")
            metadata.update(conversion_result.get("metadata", {}))

            # Ensure content_primary_type is set (derive from file extension if missing)
            metadata["content_primary_type"] = metadata.get(
                "content_primary_type"
            ) or self._derive_content_primary_type(metadata.get("file_extension"))

            # Create document node
            graph_db.create_document_node(
                doc_id,
                {
                    **metadata,
                    "processing_status": "processing",
                    "processing_stage": "chunking",
                    "processing_progress": 15.0,
                },
            )

            # Classification Stage
            classification_result = None
            logger.info(f"DEBUG: Checking classification setting (chunks_only): {getattr(settings, 'enable_document_classification', 'MISSING')}")
            if getattr(settings, "enable_document_classification", True):
                # Notify progress callback about classification stage
                if progress_callback:
                    try:
                        progress_callback(0, message="Classifying document")
                    except TypeError:
                        progress_callback(0)
                
                time.sleep(2)
                logger.info(f"!!! STARTING CLASSIFICATION FOR {doc_id} !!!")
                graph_db.create_document_node(
                    doc_id,
                    {"processing_stage": "classification", "processing_progress": 10.0},
                )
                classification_result = self.classify_document_categories(
                    filename=metadata.get("filename", ""),
                    content=content,
                )
                metadata.update(
                    {
                        "categories": classification_result.get("categories"),
                        "classification_confidence": classification_result.get("confidence"),
                        "keywords": classification_result.get("keywords"),
                        "difficulty": classification_result.get("difficulty"),
                    }
                )
                # Update node with classification data
                enrich = {
                    "category": metadata.get("categories", [])[0] if metadata.get("categories") else "general",
                    "categories": metadata.get("categories"),
                    "classification_confidence": metadata.get("classification_confidence"),
                    "keywords": metadata.get("keywords"),
                    "difficulty": metadata.get("difficulty"),
                }
                graph_db.create_document_node(doc_id, enrich)
                logger.info(
                    f"Classified document {doc_id} → {enrich['category']} (confidence={enrich['classification_confidence']:.2f})"
                )

            # Chunk the document with enhanced processing
            if progress_callback:
                try:
                    progress_callback(0, message="Chunking content")
                except TypeError:
                    progress_callback(0)

            if use_quality_filtering:
                chunks = document_chunker.chunk_text(
                    content,
                    doc_id,
                    enable_quality_filtering=use_quality_filtering,
                    enable_ocr_enhancement=False,  # OCR already applied by smart loaders
                )
                logger.info(
                    f"Used enhanced chunking (Quality filtering: {use_quality_filtering}) for {doc_id}"
                )
            else:
                chunks = document_chunker.chunk_text(content, doc_id)
                logger.info(f"Used standard chunking for {doc_id}")

            for chunk in chunks:
                chunk_metadata = chunk.get("metadata", {}) or {}
                chunk_metadata.update(
                    {
                        "document_id": doc_id,
                        "source_document": metadata.get("filename"),
                    }
                )
                chunk["metadata"] = chunk_metadata

            # Extract document summary, type, and hashtags after chunking
            if progress_callback:
                try:
                    progress_callback(0, message="Generating abstract")
                except TypeError:
                    progress_callback(0)
            logger.info(f"Extracting summary for document {doc_id}")
            summary_data = document_summarizer.extract_summary(chunks)

            graph_db.create_document_node(
                doc_id,
                {"processing_stage": "summarization", "processing_progress": 60.0},
            )
            
            # Update document node with summary information
            graph_db.update_document_summary(
                doc_id=doc_id,
                summary=summary_data.get("summary", ""),
                document_type=summary_data.get("document_type", "other"),
                hashtags=summary_data.get("hashtags", [])
            )
            logger.info(
                f"Summary extracted for {doc_id}: type={summary_data.get('document_type')}, "
                f"hashtags={len(summary_data.get('hashtags', []))}"
            )

            # Process chunks asynchronously with configurable concurrency (embeddings + storing)
            graph_db.create_document_node(
                doc_id,
                {"processing_stage": "embedding", "processing_progress": 75.0},
            )
            
            if progress_callback:
                try:
                    progress_callback(0, message="Embedding chunks")
                except TypeError:
                    progress_callback(0)

            processed_chunks = self._run_async(
                self.process_file_async(chunks, doc_id, progress_callback)
            )
            # Create similarity relationships between chunks
            try:
                relationships_created = graph_db.create_chunk_similarities(doc_id)
                logger.info(
                    f"Created {relationships_created} similarity relationships for document {doc_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create similarity relationships for document {doc_id}: {e}"
                )
                relationships_created = 0

            # Validate chunk embeddings after processing
            try:
                validation_results = graph_db.validate_chunk_embeddings(doc_id)
                if not validation_results["validation_passed"]:
                    logger.warning(
                        f"Document {doc_id} has {validation_results['invalid_chunks']} invalid chunk embeddings"
                    )
                    # Optionally fix invalid embeddings
                    invalid_chunk_ids = [
                        chunk["chunk_id"]
                        for chunk in validation_results["invalid_chunk_details"]
                    ]
                    if invalid_chunk_ids:
                        logger.info(
                            f"Attempting to fix {len(invalid_chunk_ids)} invalid chunk embeddings..."
                        )
                        fix_results = graph_db.fix_invalid_embeddings(
                            chunk_ids=invalid_chunk_ids
                        )
                        logger.info(
                            f"Fixed {fix_results['chunks_fixed']} chunk embeddings"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to validate chunk embeddings for document {doc_id}: {e}"
                )

            result = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "chunks_created": len(processed_chunks),
                "entities_created": 0,  # No entities in chunks-only mode
                "entity_relationships_created": 0,  # No entity relationships in chunks-only mode
                "similarity_relationships_created": relationships_created,
                "metadata": metadata,
                "status": "success",
                "processing_status": "completed",
                "processing_stage": "completed",
            }

            graph_db.create_document_node(
                doc_id,
                {
                    "processing_status": "completed",
                    "processing_stage": "completed",
                    "processing_progress": 100.0,
                    "processing_error": None,
                },
            )

            # Create temporal nodes for time-based retrieval
            if settings.enable_temporal_filtering:
                try:
                    graph_db.create_temporal_nodes_for_document(doc_id)
                    logger.debug(f"Created temporal nodes for document {doc_id}")
                except Exception as e:
                    logger.warning(f"Failed to create temporal nodes for {doc_id}: {e}")

            logger.info(
                f"Successfully processed {file_path} (chunks only): {len(processed_chunks)} chunks created"
            )
            # add processing duration
            duration = time.time() - start_time
            result["duration_seconds"] = duration

            # Print to stdout for quick feedback
            print(
                f"Processed {file_path} in {duration:.2f}s — {len(processed_chunks)} chunks (chunks-only)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path} (chunks only): {e}")
            try:
                doc_id = document_id or self._generate_document_id(file_path)
                graph_db.create_document_node(
                    doc_id,
                    {
                        "processing_status": "error",
                        "processing_stage": "error",
                        "processing_progress": 0.0,
                        "processing_error": str(e),
                    },
                )
            except Exception:
                pass

            return {
                "file_path": str(file_path),
                "status": "error",
                "error": str(e),
                "processing_status": "error",
                "processing_stage": "error",
            }

    def extract_entities_for_document(
        self,
        doc_id: str,
        file_name: str,
        progress_callback: Optional[
            Callable[[EntityExtractionState, float, Optional[str]], None]
        ] = None,
    ) -> Dict[str, Any]:
        """Extract entities for an existing document synchronously."""

        def _emit(state: EntityExtractionState, fraction: float, info: Optional[str] = None):
            if progress_callback:
                try:
                    progress_callback(state, fraction, info)
                except Exception as callback_exc:  # pragma: no cover - defensive
                    logger.debug(
                        "Entity progress callback failed for %s: %s", doc_id, callback_exc
                    )

        if not settings.enable_entity_extraction:
            logger.info("Entity extraction disabled, skipping for %s", doc_id)
            return {
                "status": "skipped",
                "reason": "entity_extraction_disabled",
                "document_id": doc_id,
                "entities_created": 0,
                "relationships_created": 0,
            }

        if self.entity_extractor is None:
            logger.info("Initializing entity extractor on-demand...")
            self.entity_extractor = EntityExtractor()

        extractor = self.entity_extractor
        operation_id = self._start_entity_operation(doc_id, file_name)
        _emit(EntityExtractionState.STARTING, 0.0, "Preparing entity extraction")

        try:
            chunks_for_extraction = self._prepare_chunks_for_extraction(doc_id)
            if not chunks_for_extraction:
                raise ValueError("No chunks found for document")

            cleanup_stats = graph_db.reset_document_entities(doc_id)
            logger.info(
                "Cleared previous entities for %s before re-extraction: %s",
                doc_id,
                cleanup_stats,
            )

            # Phase 1: LLM extraction
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.LLM_EXTRACTION,
                "Running LLM entity extraction",
            )
            _emit(
                EntityExtractionState.LLM_EXTRACTION,
                0.2,
                "Processing chunks",
            )

            # Check for cancellation via callback
            is_cancelled = None
            if progress_callback and hasattr(progress_callback, 'is_cancelled'):
                is_cancelled = progress_callback.is_cancelled

            entity_dict, relationship_dict = self._run_async(
                extractor.extract_from_chunks(chunks_for_extraction, is_cancelled=is_cancelled)
            )

            # Phase 2: Embedding generation / database operations
            entity_count_estimate = len(entity_dict)
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.EMBEDDING_GENERATION,
                f"Generating embeddings for {entity_count_estimate} entities",
            )
            _emit(
                EntityExtractionState.EMBEDDING_GENERATION,
                0.45,
                f"Embedding {entity_count_estimate} entities",
            )

            (
                created_entities,
                created_relationships,
                metrics,
            ) = self._persist_extraction_results(
                doc_id, chunks_for_extraction, entity_dict, relationship_dict
            )

            # Phase 3: Relationship wiring
            expected_rels = metrics.get("relationships_requested", 0)
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.DATABASE_OPERATIONS,
                f"Creating {expected_rels} relationships",
            )
            _emit(
                EntityExtractionState.DATABASE_OPERATIONS,
                0.7,
                f"Linking {expected_rels} relationships",
            )

            # Optional similarity generation
            try:
                graph_db.create_entity_similarities(doc_id)
            except Exception as sim_exc:  # pragma: no cover
                logger.debug(
                    "Failed to create entity similarities for %s: %s", doc_id, sim_exc
                )

            # Auto-clustering: run after similarity generation to assign communities
            try:
                # Update tracked operation state and emit a progress callback for clustering
                try:
                    self._update_entity_operation(
                        operation_id,
                        EntityExtractionState.CLUSTERING,
                        "Detecting communities",
                    )
                except Exception:
                    # Ensure that failure to update internal tracking doesn't stop progress emission
                    logger.debug("Failed to update entity operation state for clustering", exc_info=True)

                # Emit a progress update to any provided callback
                _emit(EntityExtractionState.CLUSTERING, 0.0, "Detecting communities")

                from core.graph_clustering import run_auto_clustering
                clustering_result = run_auto_clustering(graph_db.driver)
                if clustering_result.get("status") == "success":
                    logger.info(
                        "Auto-clustering completed: %s communities, %s nodes updated",
                        clustering_result.get("communities_count", 0),
                        clustering_result.get("updated_nodes", 0),
                    )
                else:
                    logger.debug(
                        "Auto-clustering status: %s",
                        clustering_result.get("status", "unknown"),
                    )
            except Exception as cluster_exc:  # pragma: no cover
                logger.warning(
                    "Failed to run auto-clustering for %s: %s", doc_id, cluster_exc
                )

            # Phase 4: Validation
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.VALIDATION,
                "Validating entity embeddings",
            )
            _emit(
                EntityExtractionState.VALIDATION,
                0.85,
                "Validating entity embeddings",
            )

            try:
                validation_results = graph_db.validate_entity_embeddings(doc_id)
                if not validation_results["validation_passed"]:
                    logger.warning(
                        "Document %s has %s invalid entity embeddings",
                        doc_id,
                        validation_results["invalid_embeddings"],
                    )
            except Exception as validation_exc:  # pragma: no cover
                logger.warning(
                    "Failed to validate entity embeddings for %s: %s",
                    doc_id,
                    validation_exc,
                )

            self._update_entity_operation(
                operation_id,
                EntityExtractionState.COMPLETED,
                "Entity extraction completed",
            )
            _emit(EntityExtractionState.COMPLETED, 1.0, "Entity extraction completed")

            return {
                "status": "success",
                "document_id": doc_id,
                "entities_created": created_entities,
                "relationships_created": created_relationships,
            }

        except Exception as exc:  # pragma: no cover - high level error logging
            logger.error("Entity extraction failed for %s: %s", doc_id, exc)
            self._update_entity_operation(
                operation_id,
                EntityExtractionState.ERROR,
                error_message=str(exc),
            )
            _emit(EntityExtractionState.ERROR, 1.0, str(exc))
            return {
                "status": "error",
                "document_id": doc_id,
                "error": str(exc),
                "entities_created": 0,
                "relationships_created": 0,
            }
        finally:
            self._complete_entity_operation(operation_id)

    def process_batch_entities(
        self, processed_documents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process entity extraction for multiple documents in batch to avoid rate limiting.

        Args:
            processed_documents: List of document info dicts with document_id and file_name

        Returns:
            Dictionary with entity extraction statistics
        """
        if not settings.enable_entity_extraction:
            logger.info("Entity extraction disabled, skipping batch processing")
            return None

        # Initialize entity extractor if not already done
        if self.entity_extractor is None:
            logger.info("Initializing entity extractor for batch processing...")
            self.entity_extractor = EntityExtractor()

        logger.info(
            f"Starting batch entity extraction for {len(processed_documents)} documents"
        )

        def _batch_entity_worker(documents_list):
            """Background worker for batch entity extraction."""
            operation_ids = []
            try:
                # Get a reference to the entity extractor (should be initialized by now)
                extractor = self.entity_extractor
                if not extractor:
                    logger.error("Entity extractor not available in batch worker")
                    return None

                total_entities = 0
                total_relationships = 0
                by_document = {}

                for doc_info in documents_list:
                    doc_id = doc_info["document_id"]
                    file_name = doc_info["file_name"]
                    operation_id = None

                    try:
                        # Start tracking this operation
                        operation_id = self._start_entity_operation(doc_id, file_name)
                        operation_ids.append(operation_id)

                        logger.info(
                            f"Processing entities for document {doc_id} ({file_name}) - operation: {operation_id}"
                        )

                        # Get chunks for this document from the database
                        chunks_for_extraction = self._prepare_chunks_for_extraction(
                            doc_id
                        )

                        if not chunks_for_extraction:
                            logger.warning(f"No chunks found for document {doc_id}")
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message="No chunks found for document",
                            )
                            continue

                        # Phase 1: LLM extraction
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.LLM_EXTRACTION,
                            "Running LLM entity extraction",
                        )
                        entity_dict, relationship_dict = self._run_async(
                            extractor.extract_from_chunks(chunks_for_extraction)
                        )

                        # Phase 2: Embedding generation and database operations
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.EMBEDDING_GENERATION,
                            f"Generating embeddings for {len(entity_dict)} entities",
                        )

                        (
                            created_entities,
                            created_relationships,
                            metrics,
                        ) = self._persist_extraction_results(
                            doc_id, chunks_for_extraction, entity_dict, relationship_dict
                        )

                        # Phase 3: Database operations for relationships
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.DATABASE_OPERATIONS,
                            f"Creating {metrics.get('relationships_requested', 0)} relationships",
                        )

                        # Optionally create entity similarities for this document
                        try:
                            graph_db.create_entity_similarities(doc_id)
                        except Exception as e:
                            logger.debug(
                                f"Failed to create entity similarities for {doc_id}: {e}"
                            )

                        # Phase 4: Validation
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.VALIDATION,
                            "Validating entity embeddings",
                        )

                        # Validate entity embeddings after processing
                        try:
                            validation_results = graph_db.validate_entity_embeddings(
                                doc_id
                            )
                            if not validation_results["validation_passed"]:
                                logger.warning(
                                    f"Document {doc_id} has {validation_results['invalid_embeddings']} invalid entity embeddings"
                                )
                                # Optionally fix invalid embeddings
                                invalid_entity_ids = [
                                    entity["entity_id"]
                                    for entity in validation_results[
                                        "invalid_entity_details"
                                    ]
                                ]
                                if invalid_entity_ids:
                                    logger.info(
                                        f"Attempting to fix {len(invalid_entity_ids)} invalid entity embeddings..."
                                    )
                                    fix_results = graph_db.fix_invalid_embeddings(
                                        entity_ids=invalid_entity_ids
                                    )
                                    logger.info(
                                        f"Fixed {fix_results['entities_fixed']} entity embeddings"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Failed to validate entity embeddings for document {doc_id}: {e}"
                            )

                        logger.info(
                            f"Batch entity extraction finished for {doc_id}: {created_entities} entities, {created_relationships} relationships"
                        )

                        # Track stats
                        total_entities += created_entities
                        total_relationships += created_relationships
                        by_document[doc_id] = {
                            "entities": created_entities,
                            "relationships": created_relationships,
                        }

                    except Exception as e:
                        logger.error(
                            f"Failed to process entities for document {doc_id}: {e}"
                        )
                        if operation_id:
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message=f"Failed to process document: {str(e)}",
                            )
                    finally:
                        # Complete the operation tracking for this document
                        if operation_id:
                            self._complete_entity_operation(operation_id)

                return {
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "by_document": by_document,
                }

            except Exception as e:
                logger.error(f"Unhandled error in batch entity worker: {e}")
                return None
            finally:
                # Complete any remaining operations
                for op_id in operation_ids:
                    try:
                        self._complete_entity_operation(op_id)
                    except Exception:
                        pass

        # Start thread and track it so the UI can detect background work
        t = threading.Thread(
            target=_batch_entity_worker, args=(processed_documents,), daemon=True
        )
        with self._bg_lock:
            self._bg_entity_threads.append(t)
        t.start()

        # Return None for now since this is asynchronous - stats will be available later
        return None

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loaders.keys())

    def estimate_chunks_from_files(self, uploaded_files: List[Any]) -> int:
        """
        Estimate total number of chunks that will be created from uploaded files.

        Args:
            uploaded_files: List of uploaded file objects with .name and .getvalue() methods

        Returns:
            Estimated total number of chunks
        """
        import tempfile

        total_chunks = 0

        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                loader = self.loaders.get(file_ext)

                if not loader:
                    logger.warning(f"No loader available for file type: {file_ext}")
                    continue

                # Save file temporarily to load content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                try:
                    # Load content and estimate chunks
                    content = loader.load(tmp_path)
                    if content:
                        # Use the same chunking logic to get accurate count
                        chunks = document_chunker.chunk_text(
                            content, f"temp_{uploaded_file.name}"
                        )
                        total_chunks += len(chunks)
                        logger.debug(
                            f"Estimated {len(chunks)} chunks for {uploaded_file.name}"
                        )
                finally:
                    # Clean up temporary file
                    if tmp_path.exists():
                        tmp_path.unlink()

            except Exception as e:
                logger.error(f"Error estimating chunks for {uploaded_file.name}: {e}")
                # Fallback estimation based on file size (rough estimate)
                try:
                    file_size = len(uploaded_file.getvalue())
                    estimated_chunks = max(
                        1, file_size // (settings.chunk_size * 2)
                    )  # Conservative estimate
                    total_chunks += estimated_chunks
                    logger.debug(
                        f"Fallback estimated {estimated_chunks} chunks for {uploaded_file.name}"
                    )
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback estimation also failed for {uploaded_file.name}: {fallback_e}"
                    )
                    # Last resort: assume 1 chunk
                    total_chunks += 1

        logger.info(
            f"Estimated total of {total_chunks} chunks from {len(uploaded_files)} files"
        )
        return max(1, total_chunks)  # Ensure at least 1 chunk

    def extract_entities_for_all_documents(self) -> Optional[Dict[str, Any]]:
        """
        Extract entities for all documents that don't have entities yet.
        This is a global operation that processes all documents missing entity extraction.

        Returns:
            Dictionary with extraction statistics or None if extraction is disabled
        """
        if not settings.enable_entity_extraction:
            logger.info("Entity extraction disabled, skipping global extraction")
            return None

        # Initialize entity extractor if not already done
        if self.entity_extractor is None:
            logger.info("Initializing entity extractor for global extraction...")
            self.entity_extractor = EntityExtractor()

        # Get entity extraction status from database
        try:
            status = graph_db.get_entity_extraction_status()
            docs_without_entities = [
                doc
                for doc in status["documents"]
                if not doc["entities_extracted"] and doc["total_chunks"] > 0
            ]

            if not docs_without_entities:
                logger.info("All documents already have entities extracted")
                return {
                    "status": "no_action_needed",
                    "message": "All documents already have entities extracted",
                    "processed_documents": 0,
                }

            logger.info(
                f"Starting global entity extraction for {len(docs_without_entities)} documents"
            )

            # Prepare document list for batch processing
            documents_to_process = [
                {"document_id": doc["document_id"], "file_name": doc["filename"]}
                for doc in docs_without_entities
            ]

            # Start batch entity extraction in background
            def _global_entity_worker(documents_list):
                """Background worker for global entity extraction."""
                operation_ids = []
                try:
                    extractor = self.entity_extractor
                    if not extractor:
                        logger.error(
                            "Entity extractor not available in global extraction worker"
                        )
                        return None

                    total_entities = 0
                    total_relationships = 0
                    processed_docs = 0

                    for doc_info in documents_list:
                        doc_id = doc_info["document_id"]
                        file_name = doc_info["file_name"]
                        operation_id = None

                        try:
                            # Start tracking this operation
                            operation_id = self._start_entity_operation(
                                doc_id, file_name
                            )
                            operation_ids.append(operation_id)

                            logger.info(
                                f"Processing entities for document {doc_id} ({file_name}) - operation: {operation_id}"
                            )

                            # Get chunks for this document from the database
                            chunks_for_extraction = self._prepare_chunks_for_extraction(
                                doc_id
                            )

                            if not chunks_for_extraction:
                                logger.warning(f"No chunks found for document {doc_id}")
                                self._update_entity_operation(
                                    operation_id,
                                    EntityExtractionState.ERROR,
                                    error_message="No chunks found for document",
                                )
                                continue

                            # Phase 1: LLM extraction
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.LLM_EXTRACTION,
                                "Running LLM entity extraction",
                            )
                            entity_dict, relationship_dict = self._run_async(
                                extractor.extract_from_chunks(chunks_for_extraction)
                            )

                            # Phase 2: Embedding generation and database operations
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.EMBEDDING_GENERATION,
                                f"Generating embeddings for {len(entity_dict)} entities",
                            )

                            (
                                created_entities,
                                created_relationships,
                                metrics,
                            ) = self._persist_extraction_results(
                                doc_id, chunks_for_extraction, entity_dict, relationship_dict
                            )

                            # Phase 3: Database operations for relationships
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.DATABASE_OPERATIONS,
                                f"Creating {metrics.get('relationships_requested', 0)} relationships",
                            )

                            # Optionally create entity similarities for this document
                            try:
                                graph_db.create_entity_similarities(doc_id)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to create entity similarities for {doc_id}: {e}"
                                )

                            # Phase 4: Validation
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.VALIDATION,
                                "Validating entity embeddings",
                            )

                            # Validate entity embeddings after processing
                            try:
                                validation_results = (
                                    graph_db.validate_entity_embeddings(doc_id)
                                )
                                if not validation_results["validation_passed"]:
                                    logger.warning(
                                        f"Document {doc_id} has {validation_results['invalid_embeddings']} invalid entity embeddings"
                                    )
                                    # Optionally fix invalid embeddings
                                    invalid_entity_ids = [
                                        entity["entity_id"]
                                        for entity in validation_results[
                                            "invalid_entity_details"
                                        ]
                                    ]
                                    if invalid_entity_ids:
                                        logger.info(
                                            f"Attempting to fix {len(invalid_entity_ids)} invalid entity embeddings..."
                                        )
                                        fix_results = graph_db.fix_invalid_embeddings(
                                            entity_ids=invalid_entity_ids
                                        )
                                        logger.info(
                                            f"Fixed {fix_results['entities_fixed']} entity embeddings"
                                        )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to validate entity embeddings for document {doc_id}: {e}"
                                )

                            logger.info(
                                f"Global entity extraction finished for {doc_id}: {created_entities} entities, {created_relationships} relationships"
                            )

                            # Track stats
                            total_entities += created_entities
                            total_relationships += created_relationships
                            processed_docs += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to process entities for document {doc_id}: {e}"
                            )
                            if operation_id:
                                self._update_entity_operation(
                                    operation_id,
                                    EntityExtractionState.ERROR,
                                    error_message=f"Failed to process document: {str(e)}",
                                )
                        finally:
                            # Complete the operation tracking for this document
                            if operation_id:
                                self._complete_entity_operation(operation_id)

                    logger.info(
                        f"Global entity extraction completed: {processed_docs} documents, {total_entities} entities, {total_relationships} relationships"
                    )

                except Exception as e:
                    logger.error(
                        f"Unhandled error in global entity extraction worker: {e}"
                    )
                finally:
                    # Complete any remaining operations
                    for op_id in operation_ids:
                        try:
                            self._complete_entity_operation(op_id)
                        except Exception:
                            pass

                    # Remove this thread from the tracking list when finished
                    try:
                        with self._bg_lock:
                            current = threading.current_thread()
                            if current in self._bg_entity_threads:
                                self._bg_entity_threads.remove(current)
                    except Exception:
                        pass

            # Start thread and track it so the UI can detect background work
            t = threading.Thread(
                target=_global_entity_worker, args=(documents_to_process,), daemon=True
            )
            with self._bg_lock:
                self._bg_entity_threads.append(t)
            t.start()

            return {
                "status": "started",
                "message": f"Started entity extraction for {len(docs_without_entities)} documents",
                "processed_documents": len(docs_without_entities),
            }

        except Exception as e:
            logger.error(f"Failed to start global entity extraction: {e}")
            return {
                "status": "error",
                "message": f"Failed to start entity extraction: {str(e)}",
                "processed_documents": 0,
            }


# Global document processor instance
document_processor = DocumentProcessor()
