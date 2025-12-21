"""Adapter to use Docling conversion when available.

This module wraps optional imports of `docling` and performs conversion
returning a dict with `content`, `metadata`, and the raw `docling_document`
for downstream chunking. Returns None when Docling is unavailable or conversion fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def convert_document(
    file_path: Path, output_format: str = "markdown"
) -> Optional[Dict[str, Any]]:
    """Convert a document using Docling if installed.

    Args:
        file_path: Path to input document
        output_format: Export format; currently supports "markdown" only

    Returns:
        Dict with keys `content`, `metadata`, and `docling_document`,
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

    try:
        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        doc = getattr(result, "document", None)
        if doc is None:
            logger.warning("Docling conversion returned no document for %s", file_path)
            return None

        content = doc.export_to_markdown()
        if not content:
            logger.warning("Docling produced empty output for %s", file_path)
            return None

        return {
            "content": content,
            "metadata": {
                "conversion_pipeline": "docling",
                "docling_output_format": output_format,
            },
            "docling_document": doc,
        }
    except Exception as e:
        logger.error("Docling conversion failed for %s: %s", file_path, e)
        return None
