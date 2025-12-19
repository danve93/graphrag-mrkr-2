"""Document metadata and preview routes."""

import logging
import mimetypes
import time
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from neo4j.exceptions import ServiceUnavailable

from api.models import DocumentMetadataResponse, UpdateHashtagsRequest, UpdateMetadataRequest
from core.document_summarizer import document_summarizer
from core.graph_db import graph_db
from core.singletons import get_response_cache, ResponseKeyLock
from core.cache_metrics import cache_metrics
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Specific sub-path routes must come BEFORE the generic /{document_id} route
# to avoid FastAPI matching sub-paths as document IDs

@router.post("/{document_id}/generate-summary")
async def generate_document_summary(document_id: str):
    """Generate or regenerate summary for a document."""
    try:
        # Get document chunks
        chunks = graph_db.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document has no chunks. Please process chunks first."
            )

        # Extract summary
        summary_data = document_summarizer.extract_summary(chunks)

        # Update document with summary
        graph_db.update_document_summary(
            doc_id=document_id,
            summary=summary_data.get("summary", ""),
            document_type=summary_data.get("document_type", "other"),
            hashtags=summary_data.get("hashtags", [])
        )

        return {
            "document_id": document_id,
            "summary": summary_data.get("summary", ""),
            "document_type": summary_data.get("document_type", "other"),
            "hashtags": summary_data.get("hashtags", []),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to generate summary for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary"
        ) from exc


@router.get("/{document_id}/summary")
async def get_document_summary(document_id: str):
    """Get lightweight document overview without full entity/chunk data.
    
    Used for fast initial page load and navigation previews.
    """
    try:
        cache = get_response_cache()
        cache_key = f"document_summary:{document_id}"

        # Try the cache first when enabled
        if getattr(settings, "enable_caching", True) and getattr(settings, "enable_document_summaries", True):
            cached = cache.get(cache_key)
            if cached is not None:
                cache_metrics.record_document_summary_hit()
                return cached
            cache_metrics.record_document_summary_miss()

        # Singleflight to avoid stampede on cache miss
        with ResponseKeyLock(cache_key, timeout=3) as acquired:
            # If another thread populated the cache while we waited, return it
            if acquired:
                # proceed to compute and populate
                pass
            else:
                # recheck cache quickly
                cached = cache.get(cache_key)
                if cached is not None:
                    cache_metrics.record_document_summary_hit()
                    return cached

            # Prefer precomputed fields when available on Document node
            with graph_db.session_scope() as session:
                rec = session.run(
                    """
                    MATCH (d:Document {id: $document_id})
                    RETURN d.id AS id,
                           d.filename AS filename,
                           d.original_filename AS original_filename,
                           d.mime_type AS mime_type,
                           d.size_bytes AS size_bytes,
                           d.created_at AS created_at,
                           d.link AS link,
                           d.uploader AS uploader,
                           d.precomputed_chunk_count AS pre_chunks,
                           d.precomputed_entity_count AS pre_entities,
                           d.precomputed_community_count AS pre_communities,
                           d.precomputed_similarity_count AS pre_similarities,
                           d.precomputed_summary_updated_at AS pre_updated_at
                               , d.precomputed_top_communities_json AS top_comm_json,
                               d.precomputed_top_similarities_json AS top_sims_json
                    """,
                    document_id=document_id,
                ).single()

                if not rec:
                    raise HTTPException(status_code=404, detail="Document not found")

                # If precomputed values exist (not null), use them; otherwise compute missing ones
                pre_chunks = rec.get("pre_chunks")
                pre_entities = rec.get("pre_entities")
                pre_communities = rec.get("pre_communities")
                pre_similarities = rec.get("pre_similarities")
                pre_updated_at = rec.get("pre_updated_at")

                # Guard against legacy/stale precomputed fields that were never computed by
                # `update_document_precomputed_summary()` (they may exist without an updated_at).
                # In that case, treat them as missing and recompute from relationships.
                if pre_updated_at is None:
                    pre_chunks = None
                    pre_entities = None
                    pre_communities = None
                    pre_similarities = None

                # Initialize stats with precomputed values or None
                chunks_count = pre_chunks
                entities_count = pre_entities
                communities_count = pre_communities
                similarities_count = pre_similarities

                # Compute any missing stats individually
                needs_chunks = chunks_count is None
                needs_entities = entities_count is None
                needs_communities = communities_count is None
                needs_similarities = similarities_count is None

                if needs_chunks or needs_entities or needs_communities:
                    # Run aggregation query for missing chunk/entity/community stats
                    agg_result = session.run(
                        """
                        MATCH (d:Document {id: $document_id})
                        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                        OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                        RETURN 
                            count(DISTINCT c) AS chunk_count,
                            count(DISTINCT e) AS entity_count,
                            count(DISTINCT e.community_id) AS community_count
                        """,
                        document_id=document_id,
                    ).single()
                    
                    if agg_result:
                        if needs_chunks:
                            chunks_count = agg_result["chunk_count"] or 0
                        if needs_entities:
                            entities_count = agg_result["entity_count"] or 0
                        if needs_communities:
                            communities_count = agg_result["community_count"] or 0

                if needs_similarities:
                    # Run separate query for similarities (more expensive)
                    sim_result = session.run(
                        """
                        MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c1:Chunk)-[s:SIMILAR_TO]-(c2:Chunk)
                        WHERE (d)-[:HAS_CHUNK]->(c2) AND c1.id < c2.id
                        RETURN count(DISTINCT s) AS similarity_count
                        """,
                        document_id=document_id,
                    ).single()
                    
                    if sim_result:
                        similarities_count = sim_result["similarity_count"] or 0

                # Parse precomputed preview JSON fields when present
                top_communities = None
                top_similarities = None
                try:
                    top_communities = json.loads(rec.get("top_comm_json")) if rec.get("top_comm_json") else None
                except Exception:
                    top_communities = None
                try:
                    top_similarities = json.loads(rec.get("top_sims_json")) if rec.get("top_sims_json") else None
                except Exception:
                    top_similarities = None

                resp = {
                    "id": rec["id"],
                    "filename": rec["filename"],
                    "original_filename": rec["original_filename"],
                    "mime_type": rec["mime_type"],
                    "size_bytes": rec["size_bytes"],
                    "created_at": rec["created_at"],
                    "link": rec["link"],
                    "uploader": rec["uploader"],
                    "stats": {
                        "chunks": int(chunks_count or 0),
                        "entities": int(entities_count or 0),
                        "communities": int(communities_count or 0),
                        "similarities": int(similarities_count or 0),
                    },
                    "previews": {
                        "top_communities": top_communities,
                        "top_similarities": top_similarities,
                    },
                }


            # Populate cache for next callers
            try:
                if getattr(settings, "enable_caching", True) and getattr(settings, "enable_document_summaries", True):
                    cache_ttl = getattr(settings, "document_summary_ttl", 300)
                    cache.set(cache_key, resp, ttl=cache_ttl)
            except Exception:
                pass

            return resp
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to get document summary for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document summary") from exc


@router.get("/{document_id}/entities")
async def get_document_entities(
    document_id: str,
    community_id: Optional[int] = Query(default=None, description="Filter by community"),
    entity_type: Optional[str] = Query(default=None, description="Filter by entity type"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0)
):
    """Get entities for a document with pagination and filtering.
    
    Used for progressive loading of entity lists.
    """
    try:
        # Build dynamic WHERE clause
        filters = []
        params = {
            "doc_id": document_id,
            "limit": limit,
            "offset": offset
        }
        
        if community_id is not None:
            filters.append("e.community_id = $community_id")
            params["community_id"] = community_id
            
        if entity_type:
            filters.append("e.type = $entity_type")
            params["entity_type"] = entity_type
        
        where_clause = f"AND {' AND '.join(filters)}" if filters else ""
        
        with graph_db.session_scope() as session:
            # Get total count
            count_result = session.run(
                f"""
                MATCH (d:Document {{id: $doc_id}})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE TRUE {where_clause}
                RETURN count(DISTINCT e) as total
                """,
                **params
            ).single()
            total = count_result["total"] if count_result else 0
            
            # Get paginated entities
            entity_records = session.run(
                f"""
                MATCH (d:Document {{id: $doc_id}})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE TRUE {where_clause}
                WITH e, collect(DISTINCT c.chunk_index) as positions
                RETURN 
                    e.type as type,
                    e.name as text,
                    e.community_id as community_id,
                    e.level as level,
                    size(positions) as count,
                    positions
                ORDER BY type ASC, text ASC
                SKIP $offset
                LIMIT $limit
                """,
                **params
            )
            
            entities = [
                {
                    "type": record["type"],
                    "text": record["text"],
                    "community_id": record["community_id"],
                    "level": record["level"],
                    "count": record["count"],
                    "positions": [pos for pos in (record["positions"] or []) if pos is not None]
                }
                for record in entity_records
            ]
        
        return {
            "document_id": document_id,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total,
            "entities": entities
        }
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to get entities for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve entities") from exc


@router.get("/{document_id}/similarities")
async def get_document_chunk_similarities(
    document_id: str,
    limit: int = Query(default=50, ge=1, le=200, description="Number of similarities per page"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    min_score: float = Query(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    exact_count: bool = Query(default=False, description="Compute exact total (slower, ~13s)"),
    debug: bool = Query(default=False, description="Return timing debug info in response"),
):
    """Get chunk-to-chunk similarity relationships for a document.

    Returns only IDs and scores for efficient transfer. Use `/chunks/{chunk_id}` to fetch
    full chunk content on demand.

    By default this endpoint uses a fast estimate for the total count on the first page.
    Set `exact_count=true` to compute a precise total (may be slow for very large docs).
    """
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        timings = {}
        start_all = time.perf_counter()

        # Count chunks quickly to decide on short-circuit behavior
        with graph_db.session_scope() as session:
            t0 = time.perf_counter()
            chunk_count_rec = session.run(
                "MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) as cnt",
                document_id=document_id,
            ).single()
            t1 = time.perf_counter()
            timings["chunk_count_query_s"] = t1 - t0

            num_chunks = chunk_count_rec["cnt"] if chunk_count_rec else 0

            if num_chunks < 1:
                base_resp = {
                    "document_id": document_id,
                    "total": 0,
                    "estimated": False,
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                    "similarities": [],
                }
                if debug:
                    timings["total_elapsed_s"] = time.perf_counter() - start_all
                    base_resp["_timings"] = timings
                return base_resp

        # Main paginated query (scoped to the document to keep results compact)
        query = """
            MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
            WHERE (d)-[:HAS_CHUNK]->(c2)
              AND c1.id < c2.id
              AND coalesce(sim.score, 0) >= $min_score
            WITH c1.id AS chunk1_id,
                 c2.id AS chunk2_id,
                 coalesce(sim.score, 0) AS score,
                 sim.rank AS rank
            ORDER BY coalesce(rank, 999999), score DESC
            SKIP $offset
            LIMIT $limit
            RETURN chunk1_id, chunk2_id, score, rank
        """

        with graph_db.session_scope() as session:
            t2 = time.perf_counter()
            results = session.run(
                query,
                document_id=document_id,
                offset=offset,
                limit=limit,
                min_score=min_score,
            ).data()
            t3 = time.perf_counter()
            timings["main_query_s"] = t3 - t2

            # Determine count strategy: exact if requested or if paging beyond first page
            estimated = False
            total = None
            if exact_count or offset > 0:
                count_query = """
                    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)
                    WHERE (d)-[:HAS_CHUNK]->(c2)
                      AND c1.id < c2.id
                      AND coalesce(sim.score, 0) >= $min_score
                    RETURN count(*) as total
                """
                t4 = time.perf_counter()
                total_result = session.run(
                    count_query,
                    document_id=document_id,
                    min_score=min_score,
                ).single()
                t5 = time.perf_counter()
                timings["count_query_s"] = t5 - t4
                total = total_result["total"] if total_result else 0
            else:
                # Fast heuristic for first page: assume ~3.5 similarities per chunk
                total = min(10000, int(num_chunks * 3.5))
                estimated = True

        timings["total_elapsed_s"] = time.perf_counter() - start_all

        response = {
            "document_id": document_id,
            "total": total,
            "estimated": estimated,
            "limit": limit,
            "offset": offset,
            "has_more": len(results) == limit,
            "similarities": results,
        }
        if debug:
            response["_timings"] = timings
            logger.info("similarities timings for %s: %s", document_id, timings)

        return response
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to get chunk similarities for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve chunk similarities") from exc



@router.post("/{document_id}/hashtags")
async def update_document_hashtags(document_id: str, request: UpdateHashtagsRequest):
    """Update the hashtags for a document."""
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update hashtags
        graph_db.update_document_hashtags(doc_id=document_id, hashtags=request.hashtags)

        return {"document_id": document_id, "hashtags": request.hashtags, "status": "success"}
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to update hashtags for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update hashtags") from exc


@router.patch("/{document_id}/metadata")
async def update_document_metadata(document_id: str, request: UpdateMetadataRequest):
    """Update the metadata for a document."""
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update metadata
        graph_db.update_document_metadata(doc_id=document_id, metadata=request.metadata)

        return {"document_id": document_id, "metadata": request.metadata, "status": "success"}
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to update metadata for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update metadata") from exc





# PUT endpoint for incremental document updates
@router.put("/{document_id}")
async def update_document_content(
    document_id: str,
    file: UploadFile = File(...),
):
    """
    Update an existing document incrementally.
    
    This endpoint enables efficient document updates by:
    - Only processing chunks that have changed (based on content hash)
    - Preserving unchanged chunks and their entities
    - Cleaning up orphaned entities from removed chunks
    
    Returns a summary of what changed (unchanged/added/removed counts).
    
    Note: If chunking parameters (chunk_size, chunk_overlap) have changed since
    the document was first ingested, this will return an error. In that case,
    delete and re-upload the document or reindex the corpus first.
    """
    from pathlib import Path
    import tempfile
    import shutil
    from ingestion.document_processor import document_processor
    
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = Path(tmp.name)
        
        try:
            # Import global processing state to trigger frontend polling
            from api.routers.database import _global_processing_state
            
            # Set processing state to trigger frontend polling
            _global_processing_state["is_processing"] = True
            _global_processing_state["current_document_id"] = document_id
            _global_processing_state["current_filename"] = file.filename
            _global_processing_state["current_stage"] = "conversion"
            _global_processing_state["progress_percentage"] = 5.0
            
            # Call the incremental update method
            result = document_processor.update_document(
                doc_id=document_id,
                file_path=temp_path,
                original_filename=file.filename,
            )
            
            # Check for errors
            if result.get("status") == "error":
                error_type = result.get("error", "unknown")
                if error_type == "chunking_params_changed":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "chunking_params_changed",
                            "message": result.get("message"),
                            "stored_params": result.get("stored_params"),
                            "current_params": result.get("current_params"),
                            "options": [
                                "Delete and re-upload the document",
                                "Reindex the entire corpus with new parameters first"
                            ]
                        }
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=result.get("error", "Unknown error during update")
                    )
            
            return {
                "document_id": document_id,
                "status": result.get("status", "success"),
                "message": result.get("message", "Processing incremental update"),
                "changes": {
                    "unchanged_chunks": result.get("unchanged_chunks", 0),
                    "added_chunks": result.get("added_chunks", 0),
                    "removed_chunks": result.get("removed_chunks", 0),
                    "entities_removed": result.get("entities_removed", 0),
                    "relationships_cleaned": result.get("relationships_cleaned", 0),
                },
                "processing_time": result.get("processing_time"),
            }
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to update document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to update document") from exc


@router.get("/search-similar")
async def search_similar_documents(
    filename: str = Query(..., description="Filename to search for similar documents"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results")
):
    """
    Search for documents with similar filenames.
    
    Returns both exact matches and fuzzy matches for use in upload update dialog.
    """
    import re
    from difflib import SequenceMatcher
    
    try:
        # Get all documents
        all_docs = graph_db.get_all_documents()
        
        # Parse uploaded filename
        base_name = Path(filename).stem.lower()
        extension = Path(filename).suffix.lower()
        
        # Remove common version patterns for matching (e.g., _v1, _v2, -final, -draft)
        version_pattern = re.compile(r'[-_](v\d+|final|draft|new|old|updated|revised|\d{8}|\d{4}[-_]\d{2}[-_]\d{2})$', re.IGNORECASE)
        normalized_base = version_pattern.sub('', base_name)
        
        results = []
        for doc in all_docs:
            doc_filename = doc.get('original_filename') or doc.get('filename') or ''
            doc_base = Path(doc_filename).stem.lower()
            doc_ext = Path(doc_filename).suffix.lower()
            doc_normalized = version_pattern.sub('', doc_base)
            
            # Calculate match scores
            is_exact = doc_filename.lower() == filename.lower()
            is_normalized_match = doc_normalized == normalized_base and doc_ext == extension
            
            # Fuzzy similarity score
            similarity = SequenceMatcher(None, normalized_base, doc_normalized).ratio()
            
            # Only include if reasonably similar (> 0.5 similarity) or exact/normalized match
            if is_exact or is_normalized_match or similarity > 0.5:
                results.append({
                    "document_id": doc.get('id') or doc.get('document_id'),
                    "filename": doc_filename,
                    "is_exact_match": is_exact,
                    "is_normalized_match": is_normalized_match,
                    "similarity_score": round(similarity, 2),
                    "document_type": doc.get('document_type'),
                    "created_at": doc.get('created_at'),
                    "chunk_count": doc.get('chunk_count', 0),
                })
        
        # Sort: exact matches first, then normalized matches, then by similarity
        results.sort(key=lambda x: (
            -x['is_exact_match'],
            -x['is_normalized_match'],
            -x['similarity_score']
        ))
        
        return {
            "query_filename": filename,
            "matches": results[:limit],
            "total_found": len(results),
        }
        
    except Exception as exc:
        logger.error("Failed to search similar documents: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to search documents") from exc


# Generic document metadata route comes LAST to avoid catching sub-paths
@router.get("/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
    """Return document metadata and related analytics."""
    try:
        # Use the shared graph_db helper to retrieve document details (ensures
        # consistent behavior with other endpoints like /chunks and /entities).
        try:
            details = graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found") from None

        # Build a lightweight response using the details returned by the helper.
        resp = {
            "id": details.get("id") or document_id,
            "title": details.get("title"),
            "file_name": details.get("file_name"),
            "original_filename": details.get("original_filename"),
            "mime_type": details.get("mime_type"),
            "preview_url": details.get("preview_url"),
            "uploaded_at": details.get("uploaded_at"),
            "uploader": details.get("uploader"),
            "summary": details.get("summary"),
            "document_type": details.get("document_type"),
            "hashtags": details.get("hashtags") or [],
            # Avoid sending full chunks/entities on this route; frontend
            # should lazy-load them via the dedicated endpoints.
            "chunks": [],
            "entities": [],
            "quality_scores": details.get("quality_scores"),
            "related_documents": details.get("related_documents"),
            "metadata": details.get("metadata"),
        }

        return DocumentMetadataResponse(**resp)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except HTTPException:
        # Propagate HTTP exceptions (e.g., 404) so the correct status is returned
        raise
    except Exception as exc:  # pragma: no cover - defensive logging
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to load document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document") from exc


@router.get("/chunks/{chunk_id}")
async def get_chunk_details(chunk_id: str):
    """Get full details for a specific chunk.
    
    Used for on-demand loading of chunk content in similarity views.
    """
    try:
        with graph_db.session_scope() as session:
            result = session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                OPTIONAL MATCH (c)<-[:HAS_CHUNK]-(d:Document)
                RETURN 
                    c.id AS id,
                    c.content AS content,
                    c.chunk_index AS index,
                    coalesce(c.offset, 0) AS offset,
                    d.id AS document_id,
                    d.filename AS document_name
                """,
                chunk_id=chunk_id
            ).single()
            
            if not result:
                raise HTTPException(status_code=404, detail="Chunk not found")
            
            return {
                "id": result["id"],
                "content": result["content"],
                "index": result["index"],
                "offset": result["offset"],
                "document_id": result["document_id"],
                "document_name": result["document_name"]
            }
    except HTTPException:
        raise
    except Exception as exc:
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to get chunk details for %s: %s", chunk_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve chunk details") from exc
@router.get("/{document_id}/preview")
@router.head("/{document_id}/preview")
async def get_document_preview(
    document_id: str,
    chunk_index: Optional[int] = Query(None, description="Chunk index to preview"),
    chunk_id: Optional[str] = Query(None, description="Chunk id to preview"),
):
    """Stream the document file or return JSON chunk preview when requested.

    If `chunk_index` or `chunk_id` is supplied, return JSON with the chunk content
    and minimal metadata to allow the frontend to highlight the chunk. Otherwise,
    fall back to streaming the original file.
    """
    # If chunk-level preview requested, try to return chunk content
    if chunk_index is not None or chunk_id is not None:
        try:
            # Prefer scanning the document's stored chunk records for reliability
            chunks = graph_db.get_document_chunks(document_id) or []

            if chunk_id:
                # Find chunk by id in the document's chunk list
                found = None
                for c in chunks:
                    cid = c.get("chunk_id") or c.get("id")
                    if cid == chunk_id:
                        found = c
                        break

                if found:
                    logger.info(
                        "Serving chunk preview (chunk_id=%s, chunk_index=%s) for document %s",
                        found.get("chunk_id") or found.get("id"),
                        found.get("index") or found.get("chunk_index"),
                        document_id,
                    )
                    return JSONResponse(
                        {
                            "document_id": document_id,
                            "chunk_id": found.get("chunk_id") or found.get("id"),
                            "chunk_index": found.get("index") or found.get("chunk_index"),
                            "content": found.get("content") or found.get("text") or "",
                        }
                    )

                # Fallback: try the lower-level content getter (some stores keep chunks separately)
                content = graph_db._get_chunk_content_sync(chunk_id)
                if content:
                    logger.info(
                        "Serving chunk preview (chunk_id=%s) for document %s via fallback content getter",
                        chunk_id,
                        document_id,
                    )
                    return JSONResponse(
                        {"document_id": document_id, "chunk_id": chunk_id, "chunk_index": None, "content": content}
                    )

                raise ValueError("Chunk not found")

            else:
                # Use provided index to find matching chunk
                matching = []
                for pos, c in enumerate(chunks):
                    # Some chunk records might have 'index' key
                    idx = c.get("index") or c.get("chunk_index")
                    if idx is None:
                        # Use positional index as fallback if chunk_index missing
                        idx = pos
                    if int(idx) == int(chunk_index):
                        matching.append(c)
                        break

                if not matching:
                    raise ValueError("Chunk not found")

                chunk = matching[0]
                logger.info(
                    "Serving chunk preview (chunk_id=%s, chunk_index=%s) for document %s (index lookup)",
                    chunk.get("chunk_id") or chunk.get("id"),
                    chunk.get("index") or chunk.get("chunk_index"),
                    document_id,
                )
                return JSONResponse(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                        "chunk_index": chunk.get("index") or chunk.get("chunk_index"),
                        "content": chunk.get("content") or chunk.get("text") or "",
                    }
                )
        except ValueError:
            raise HTTPException(status_code=404, detail="Chunk not found")
        except Exception as exc:
            if isinstance(exc, ServiceUnavailable):
                raise
            logger.exception("Failed to get chunk preview for %s", document_id)
            raise HTTPException(status_code=500, detail="Failed to retrieve chunk preview") from exc

    # Fallback: return the stored file for download/streaming
    try:
        info = graph_db.get_document_file_info(document_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # pragma: no cover - defensive logging
        if isinstance(exc, ServiceUnavailable):
            raise
        logger.error("Failed to load preview info for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to prepare preview") from exc

    # Check if there's a preview_url to redirect to
    preview_url = info.get("preview_url")
    if preview_url:
        return RedirectResponse(url=preview_url, status_code=302)

    file_path = info.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Preview not available")

    path = Path(file_path)
    # If path is relative, make it absolute relative to the project root
    if not path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up from api/routers/ to project root
        path = project_root / path

    if not path.exists() or not path.is_file():
        logger.error(f"File not found at path: {path}")
        raise HTTPException(status_code=404, detail="Preview not available")

    # Determine a reliable media type: prefer stored value, otherwise guess from filename
    media_type = info.get("mime_type")
    if not media_type:
        guessed = mimetypes.guess_type(path.name)[0]
        media_type = guessed or "application/octet-stream"

    # Normalize markdown files to text/markdown when appropriate
    if path.suffix.lower() == ".md" and ("octet-stream" in media_type or media_type == "application/octet-stream"):
        media_type = "text/markdown"

    # For markdown and text files, serve inline (for preview) instead of as attachment (download)
    is_markdown = media_type and ("markdown" in media_type or "text/plain" in media_type or "text/x-markdown" in media_type or path.suffix.lower() == ".md")
    is_text = media_type and media_type.startswith("text/")
    
    if is_markdown or is_text:
        # Read the file content and return as plain text/markdown to trigger inline display
        content = path.read_text(encoding="utf-8", errors="replace")
        # If it's a plain text file, return raw text with correct media type
        from fastapi import Response

        if media_type == "text/plain":
            return Response(content, media_type="text/plain")

        # For markdown and other text-like types, return JSON with inline disposition
        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "content": content,
                "mime_type": media_type,
                "filename": info.get("file_name"),
            },
            headers={"Content-Disposition": "inline"},
        )
    
    return FileResponse(path, media_type=media_type, filename=info.get("file_name"))


@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str, limit: int = Query(default=500, ge=1, le=2000), offset: int = Query(default=0, ge=0)):
    """Return chunk list for a document with server-side pagination.

    - `limit` (default 500) controls page size.
    - `offset` controls pagination offset.

    Returns `total` count and `has_more` flag to help the frontend lazy-load large documents.
    """
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        with graph_db.session_scope() as session:
            # total count
            total_rec = session.run(
                "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk) RETURN count(c) AS total",
                doc_id=document_id,
            ).single()
            total = total_rec["total"] if total_rec else 0

            # paginated records
            records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id AS id, c.content AS content, c.chunk_index AS chunk_index, coalesce(c.offset, 0) AS offset
                ORDER BY c.chunk_index ASC
                SKIP $offset
                LIMIT $limit
                """,
                doc_id=document_id,
                offset=offset,
                limit=limit,
            ).data()

            chunks = [
                {
                    "id": r.get("id"),
                    "text": r.get("content") or r.get("text") or "",
                    "index": r.get("chunk_index"),
                    "offset": r.get("offset"),
                }
                for r in records
            ]

        return {
            "document_id": document_id,
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "has_more": (offset + len(chunks)) < int(total),
            "chunks": chunks,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load chunks for %s", document_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve chunks") from exc


@router.get("/{document_id}/entity-summary")
async def get_document_entity_summary(document_id: str):
    """Return aggregated counts of entities grouped by entity type for the document.

    Example response:
    {
      "document_id": "...",
      "total": 1234,
      "groups": [ {"type": "ACCOUNT", "count": 506}, {"type": "MAILING_LIST", "count": 30}, ... ]
    }
    """
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        with graph_db.session_scope() as session:
            # Aggregation by entity type and total
            agg_q = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH e.type AS type, count(DISTINCT e) AS cnt
                ORDER BY cnt DESC
                RETURN type, cnt
                """,
                doc_id=document_id,
            )

            groups = [
                {"type": rec["type"] or "<unknown>", "count": int(rec["cnt"])}
                for rec in agg_q
            ]

            total_rec = session.run(
                "MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(:Chunk)-[:CONTAINS_ENTITY]->(e:Entity) RETURN count(DISTINCT e) AS total",
                doc_id=document_id,
            ).single()
            total = int(total_rec["total"]) if total_rec else 0

        return {"document_id": document_id, "total": total, "groups": groups}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to load entity summary for %s", document_id)
        raise HTTPException(status_code=500, detail="Failed to retrieve entity summary") from exc
