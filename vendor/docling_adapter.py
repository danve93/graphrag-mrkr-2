"""Adapter to use Docling conversion when available.

This module wraps optional imports of `docling` and performs conversion
returning a dict with `content`, `metadata`, and the raw `docling_document`
for downstream chunking. Returns None when Docling is unavailable or conversion fails.

For PDFs exceeding `docling_max_pages`, returns a special result indicating
the caller should use Marker fallback instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_pdf_page_count(file_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(file_path))
        count = len(pdf)
        pdf.close()
        return count
    except Exception as e:
        logger.warning("Failed to get PDF page count for %s: %s", file_path, e)
        return 0


def convert_document(
    file_path: Path, output_format: str = "markdown"
) -> Optional[Dict[str, Any]]:
    """Convert a document using Docling if installed.

    For PDFs exceeding the page limit, returns a special "use_marker_fallback" 
    result to signal the caller should use Marker instead.

    Args:
        file_path: Path to input document
        output_format: Export format; currently supports "markdown" only

    Returns:
        Dict with keys `content`, `metadata`, and `docling_document`,
        or special dict with `use_marker_fallback: True` for large PDFs,
        or None on failure/unavailable.
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except Exception as e:
        logger.info("Docling library not available: %s", e)
        return None

    if output_format != "markdown":
        logger.warning(
            "Unsupported Docling output_format '%s', defaulting to markdown", output_format
        )
        output_format = "markdown"

    # Load settings for configuration
    try:
        from config.settings import settings
        max_pages = getattr(settings, "docling_max_pages", 200)
        document_timeout = float(getattr(settings, "docling_document_timeout", 300))
    except Exception:
        max_pages = 200
        document_timeout = 300.0

    # Check page count for PDFs
    is_pdf = file_path.suffix.lower() == ".pdf"
    
    if is_pdf:
        page_count = _get_pdf_page_count(file_path)
        
        if page_count > max_pages:
            logger.warning(
                "PDF %s has %d pages, exceeds limit of %d. Signaling Marker fallback.",
                file_path.name, page_count, max_pages
            )
            return {
                "use_marker_fallback": True,
                "content": None,
                "metadata": {
                    "page_count": page_count,
                    "docling_max_pages": max_pages,
                    "conversion_warning": f"Document has {page_count} pages (limit: {max_pages}). Using Marker for stable processing.",
                },
                "docling_document": None,
            }
        
        logger.info(
            "PDF %s has %d pages (within limit of %d). Using docling.",
            file_path.name, page_count, max_pages
        )

    # Single-pass conversion (no batching - preserves docling_document for hybrid chunker)
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
        
        # Configure with timeout for safety
        pipeline_options = PdfPipelineOptions(
            document_timeout=document_timeout,
        )
        
        if is_pdf:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        else:
            converter = DocumentConverter()
        
        logger.info("Starting docling conversion for %s", file_path.name)
        result = converter.convert(str(file_path))
        doc = getattr(result, "document", None)
        
        if doc is None:
            logger.warning("Docling conversion returned no document for %s", file_path)
            return None

        content = doc.export_to_markdown()
        if not content:
            logger.warning("Docling produced empty output for %s", file_path)
            return None

        # Get page count for metadata if not already known
        if is_pdf and not page_count:
            page_count = _get_pdf_page_count(file_path)

        logger.info(
            "Docling conversion successful for %s: %d chars extracted",
            file_path.name, len(content)
        )

        return {
            "content": content,
            "metadata": {
                "conversion_pipeline": "docling",
                "docling_output_format": output_format,
                "total_pages": page_count if is_pdf else None,
            },
            "docling_document": doc,
        }
        
    except Exception as e:
        logger.error("Docling conversion failed for %s: %s", file_path, e)
        return None
