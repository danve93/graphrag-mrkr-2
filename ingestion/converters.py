"""Document conversion utilities for ingestion.

This module normalizes loader outputs into Markdown content so downstream
chunking, summarization, and tagging operate on a consistent representation.
PDFs can optionally be converted via Marker for higher fidelity. When disabled,
the smart OCR + PDF text extraction path is used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.pptx_loader import PPTXLoader
from ingestion.loaders.text_loader import TextLoader
from ingestion.loaders.xlsx_loader import XLSXLoader
from config.settings import settings
from vendor.marker_adapter import convert_pdf as marker_convert_pdf

logger = logging.getLogger(__name__)


class DocumentConverter:
    """Convert supported document formats into Markdown plus metadata."""

    def __init__(self) -> None:
        pdf_loader = PDFLoader()
        self.loaders = {
            ".pdf": pdf_loader,
            ".docx": DOCXLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".py": TextLoader(),
            ".js": TextLoader(),
            ".html": TextLoader(),
            ".css": TextLoader(),
            ".csv": CSVLoader(),
            ".pptx": PPTXLoader(),
            ".xlsx": XLSXLoader(),
            ".xls": XLSXLoader(),
            ".jpg": ImageLoader(),
            ".jpeg": ImageLoader(),
            ".png": ImageLoader(),
            ".tiff": ImageLoader(),
            ".bmp": ImageLoader(),
        }

    def _wrap_markdown(self, header: str, body: str) -> str:
        return f"# {header}\n\n{body.strip()}" if body else header

    def _convert_pdf_with_marker(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Convert PDF using Marker via adapter when available."""
        # API key MUST come from environment variables only (MARKER_LLM_API_KEY or OPENAI_API_KEY)
        # Never accept API keys from JSON config or user input
        marker_api_key = settings.marker_llm_api_key or settings.openai_api_key
        
        marker_config: Dict[str, Any] = {
            "output_format": settings.marker_output_format,
            "use_llm": settings.marker_use_llm,
            "paginate_output": settings.marker_paginate_output,
            "force_ocr": settings.marker_force_ocr,
            "strip_existing_ocr": settings.marker_strip_existing_ocr,
            "pdftext_workers": settings.marker_pdftext_workers,
            "llm_service": settings.marker_llm_service,
            "llm_model": settings.marker_llm_model,
            "llm_api_key": marker_api_key,
        }
        return marker_convert_pdf(file_path, marker_config)

    def _convert_pdf(self, file_path: Path) -> Optional[Dict[str, Any]]:
        # Prefer Marker path when enabled; fallback to smart OCR loader
        if getattr(settings, "use_marker_for_pdf", False):
            marker_result = self._convert_pdf_with_marker(file_path)
            if marker_result:
                return marker_result
            else:
                logger.info("Falling back to smart OCR PDF loader for %s", file_path)

        loader: PDFLoader = self.loaders[".pdf"]  # type: ignore[assignment]
        result = loader.load_with_metadata(file_path)
        if not result:
            return None

        content = result.get("content", "")
        ocr_metadata = result.get("metadata", {})

        # Preserve page structure with headings to aid provenance
        pages = [page.strip() for page in content.split("--- Page") if page.strip()]
        markdown_pages = []
        for idx, page in enumerate(pages, start=1):
            markdown_pages.append(f"## Page {idx}\n\n{page}")
        markdown = "\n\n".join(markdown_pages) if markdown_pages else content

        return {
            "content": markdown,
            "metadata": {
                **ocr_metadata,
                "conversion_pipeline": "smart_ocr_markdown",
            },
        }

    def _convert_simple_loader(self, loader: Any, file_path: Path, title: str) -> Optional[Dict[str, Any]]:
        content = loader.load(file_path)
        if not content:
            return None
        return {
            "content": self._wrap_markdown(title, content),
            "metadata": {"conversion_pipeline": "plain_markdown"},
        }

    def convert(self, file_path: Path, original_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Convert a document to Markdown content and metadata."""

        file_ext = file_path.suffix.lower()
        loader = self.loaders.get(file_ext)
        if not loader:
            logger.warning("No converter found for extension %s", file_ext)
            return None

        name = original_filename or file_path.name
        if file_ext == ".pdf":
            return self._convert_pdf(file_path)

        if isinstance(loader, ImageLoader):
            result = loader.load_with_metadata(file_path)
            if not result:
                return None
            return {
                "content": self._wrap_markdown(name, result.get("content", "")),
                "metadata": {
                    **result.get("metadata", {}),
                    "conversion_pipeline": "ocr_image_markdown",
                },
            }

        if hasattr(loader, "load_with_metadata"):
            result = loader.load_with_metadata(file_path)  # type: ignore[attr-defined]
            if result:
                return {
                    "content": self._wrap_markdown(name, result.get("content", "")),
                    "metadata": {
                        **result.get("metadata", {}),
                        "conversion_pipeline": "structured_markdown",
                    },
                }

        return self._convert_simple_loader(loader, file_path, name)


document_converter = DocumentConverter()
