"""Document metadata and preview routes."""

import logging
import mimetypes
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from neo4j.exceptions import ServiceUnavailable

from api.models import DocumentMetadataResponse, UpdateHashtagsRequest
from core.document_summarizer import document_summarizer
from core.graph_db import graph_db

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
        with graph_db.session_scope() as session:
            result = session.run(
                """
                MATCH (d:Document {id: $document_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH d, 
                     count(DISTINCT c) AS chunk_count,
                     count(DISTINCT e) AS entity_count,
                     count(DISTINCT e.community_id) AS community_count
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c2:Chunk)-[s:SIMILAR_TO]->()
                WHERE c2.id < id(startNode(s))
                RETURN 
                    d.id AS id,
                    d.filename AS filename,
                    d.original_filename AS original_filename,
                    d.mime_type AS mime_type,
                    d.size_bytes AS size_bytes,
                    d.created_at AS created_at,
                    d.link AS link,
                    d.uploader AS uploader,
                    chunk_count,
                    entity_count,
                    community_count,
                    count(DISTINCT s) AS similarity_count
                """,
                document_id=document_id
            ).single()
            
            if not result:
                raise HTTPException(status_code=404, detail="Document not found")
            
            return {
                "id": result["id"],
                "filename": result["filename"],
                "original_filename": result["original_filename"],
                "mime_type": result["mime_type"],
                "size_bytes": result["size_bytes"],
                "created_at": result["created_at"],
                "link": result["link"],
                "uploader": result["uploader"],
                "stats": {
                    "chunks": result["chunk_count"],
                    "entities": result["entity_count"],
                    "communities": result["community_count"],
                    "similarities": result["similarity_count"]
                }
            }
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
    debug: bool = Query(default=False, description="Return timing debug info in response")
):
    """Get chunk-to-chunk similarity relationships for a document.
    
    Returns only IDs and scores for efficient transfer.
    Use /chunks/{chunk_id} to fetch full chunk content on demand.
    
    Performance: By default, uses fast estimation for total count on first page.
    Set exact_count=true to compute precise total (adds ~13s for large documents).
    """
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        timings = {}
        start_all = time.perf_counter()

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
                    "similarities": []
                }
                if debug:
                    timings["total_elapsed_s"] = time.perf_counter() - start_all
                    base_resp["_timings"] = timings
                return base_resp

        # Optimized query - use precomputed rank when available for fast pagination
        # Scope the query to the document to avoid fetching chunk lists in Python
        query = """
            MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)-[:HAS_CHUNK]->(d)
            WHERE c1.id < c2.id
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

        # Fetch main results (fast). Ordering uses `rank` when present.
        with graph_db.session_scope() as session:
            t2 = time.perf_counter()
            results = session.run(query,
                document_id=document_id,
                offset=offset,
                limit=limit,
                min_score=min_score
            ).data()
            t3 = time.perf_counter()
            timings["main_query_s"] = t3 - t2

            # Determine count strategy for performance optimization
            estimated = False
            total = None
            if exact_count or offset > 0:
                # User requested exact count OR navigating past first page (needs accurate total)
                count_query = """
                    MATCH (d:Document {id: $document_id})-[:HAS_CHUNK]->(c1:Chunk)-[sim:SIMILAR_TO]-(c2:Chunk)-[:HAS_CHUNK]->(d)
                    WHERE c1.id < c2.id
                      AND coalesce(sim.score, 0) >= $min_score
                    RETURN count(*) as total
                """
                t4 = time.perf_counter()
                total_result = session.run(count_query,
                    document_id=document_id,
                    min_score=min_score
                ).single()
                t5 = time.perf_counter()
                timings["count_query_s"] = t5 - t4
                total = total_result["total"] if total_result else 0
            else:
                # First page with default settings: use fast estimate
                # Estimate based on typical similarity density (~3.5 similarities per chunk)
                # Use previously fetched num_chunks
                total = min(10000, int(num_chunks * 3.5))
                estimated = True

        timings["total_elapsed_s"] = time.perf_counter() - start_all
        
        response = {
            "document_id": document_id,
            "total": total,
            "estimated": estimated,
            "limit": limit,
            "offset": offset,
            "has_more": len(results) == limit,  # Simple check: if we got full page, likely more exist
            "similarities": results
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
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve chunk similarities"
        ) from exc



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


# Generic document metadata route comes LAST to avoid catching sub-paths
@router.get("/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
    """Return document metadata and related analytics."""
    try:
        details = graph_db.get_document_details(document_id)
        return DocumentMetadataResponse(**details)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
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
