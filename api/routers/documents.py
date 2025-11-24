"""Document metadata and preview routes."""

import logging
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from api.models import DocumentMetadataResponse, UpdateHashtagsRequest
from core.document_summarizer import document_summarizer
from core.graph_db import graph_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
    """Return document metadata and related analytics."""
    try:
        details = graph_db.get_document_details(document_id)
        return DocumentMetadataResponse(**details)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document") from exc


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
        logger.error("Failed to generate summary for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary"
        ) from exc


@router.patch("/{document_id}/hashtags")
async def update_document_hashtags(document_id: str, request: UpdateHashtagsRequest):
    """Update the hashtags for a document."""
    try:
        # Verify document exists
        try:
            graph_db.get_document_details(document_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="Document not found")

        # Update hashtags
        graph_db.update_document_hashtags(
            doc_id=document_id,
            hashtags=request.hashtags
        )

        return {
            "document_id": document_id,
            "hashtags": request.hashtags,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to update hashtags for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to update hashtags"
        ) from exc


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
            logger.exception("Failed to get chunk preview for %s", document_id)
            raise HTTPException(status_code=500, detail="Failed to retrieve chunk preview") from exc

    # Fallback: return the stored file for download/streaming
    try:
        info = graph_db.get_document_file_info(document_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # pragma: no cover - defensive logging
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
        # Read the file content and return as plain text to trigger inline display
        content = path.read_text(encoding="utf-8", errors="replace")
        return JSONResponse(
            status_code=200,
            content={
                "document_id": document_id,
                "content": content,
                "mime_type": media_type,
                "filename": info.get("file_name"),
            },
            headers={
                "Content-Disposition": "inline"
            }
        )
    
    return FileResponse(path, media_type=media_type, filename=info.get("file_name"))
