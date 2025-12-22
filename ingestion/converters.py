"""Document conversion utilities for ingestion.

This module normalizes loader outputs into Markdown (or raw HTML for heading-aware
chunking) so downstream chunking, summarization, and tagging operate on a
consistent representation.
PDFs can optionally be converted via Marker for higher fidelity. When disabled,
the smart OCR + PDF text extraction path is used. Docling can be selected to
convert all supported formats into Markdown for ingestion.
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
from vendor.docling_adapter import convert_document as docling_convert_document
from vendor.marker_adapter import convert_pdf as marker_convert_pdf

logger = logging.getLogger(__name__)

DOCLING_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".md",
    ".markdown",
    ".adoc",
    ".asciidoc",
    ".html",
    ".htm",
    ".xhtml",
    ".xht",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".vtt",
    ".mp3",
    ".wav",
}
HTML_EXTENSIONS = {".html", ".htm", ".xhtml", ".xht"}


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
            ".htm": TextLoader(),
            ".xhtml": TextLoader(),
            ".xht": TextLoader(),
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
        self.docling_extensions = set(DOCLING_SUPPORTED_EXTENSIONS)

    def _wrap_markdown(self, header: str, body: str) -> str:
        return f"# {header}\n\n{body.strip()}" if body else header

    def _resolve_conversion_provider(self) -> str:
        provider = (getattr(settings, "document_conversion_provider", None) or "auto").strip().lower()
        if provider not in {"auto", "native", "marker", "docling"}:
            logger.warning("Invalid document_conversion_provider '%s', defaulting to auto", provider)
            provider = "auto"
        if provider == "auto":
            return "marker" if getattr(settings, "use_marker_for_pdf", False) else "native"
        return provider

    def _should_use_docling(self, provider: str, file_ext: str) -> bool:
        if provider != "docling":
            return False
        if file_ext not in self.docling_extensions:
            return False
        if file_ext in HTML_EXTENSIONS and getattr(settings, "chunker_strategy_html", "html_heading") == "html_heading":
            return False
        return True

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

    def _convert_pdf(self, file_path: Path, use_marker: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        # Prefer Marker path when enabled; fallback to smart OCR loader
        use_marker = getattr(settings, "use_marker_for_pdf", False) if use_marker is None else use_marker
        if use_marker:
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

    def _convert_with_docling(self, file_path: Path) -> Optional[Dict[str, Any]]:
        return docling_convert_document(file_path, output_format="markdown")

    def supports_extension(self, file_ext: str) -> bool:
        provider = self._resolve_conversion_provider()
        if provider == "docling" and file_ext in self.docling_extensions:
            return True
        return file_ext in self.loaders

    def get_supported_extensions(self) -> list[str]:
        provider = self._resolve_conversion_provider()
        extensions = set(self.loaders.keys())
        if provider == "docling":
            extensions.update(self.docling_extensions)
        return sorted(extensions)

    def _convert_simple_loader(self, loader: Any, file_path: Path, title: str) -> Optional[Dict[str, Any]]:
        content = loader.load(file_path)
        if not content:
            return None
        return {
            "content": self._wrap_markdown(title, content),
            "metadata": {"conversion_pipeline": "plain_markdown"},
        }

    def _convert_html(self, loader: Any, file_path: Path) -> Optional[Dict[str, Any]]:
        content = loader.load(file_path)
        if not content:
            return None
        return {
            "content": content,
            "metadata": {"conversion_pipeline": "html_raw"},
        }

    def _get_pdf_page_count(self, file_path: Path) -> int:
        """Get the number of pages in a PDF file efficiently."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            return len(reader.pages)
        except Exception as e:
            logger.warning("Failed to get PDF page count for %s: %s", file_path, e)
            return 0

    def convert(self, file_path: Path, original_filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Convert a document to Markdown content and metadata."""

        file_ext = file_path.suffix.lower()
        provider = self._resolve_conversion_provider()
        
        # Explicitly check PDF page count for Docling/Marker decision
        if file_ext == ".pdf":
            page_count = self._get_pdf_page_count(file_path)
            docling_limit = getattr(settings, "docling_max_pages", 50)
            
            # User Feedback: Check page count BEFORE invoking Docling
            if provider == "docling" and page_count > docling_limit:
                logger.info(
                    "PDF %s has %d pages (limit: %d). Switching from Docling to Marker.",
                    file_path.name,
                    page_count,
                    docling_limit
                )
                provider = "marker"
                # Add warning about the switch for the UI
                # Note: We can't easily pass this warning out unless we attach it to the result later.
                # Since we switch provider, we'll try marker below.

        if self._should_use_docling(provider, file_ext):
            docling_result = self._convert_with_docling(file_path)
            
            # Additional fallback check in case docling fails internally or returns fallback signal
            if docling_result and docling_result.get("use_marker_fallback"):
                logger.warning(
                    "Docling signaled Marker fallback for %s: %s",
                    file_path.name,
                    docling_result.get("metadata", {}).get("conversion_warning", "Large PDF")
                )
                # Fall through to try marker
                provider = "marker"
            
            elif docling_result and docling_result.get("content"):
                return docling_result
            elif file_ext not in self.loaders:
                logger.warning("Docling conversion failed for %s and no fallback loader exists", file_path)
                return None
            else:
                 logger.warning("Docling conversion failed for %s; falling back to native loader", file_path)
                 # Fall through to native loader if docling failed completely

        loader = self.loaders.get(file_ext)
        if not loader:
            logger.warning("No converter found for extension %s", file_ext)
            return None

        name = original_filename or file_path.name
        if file_ext == ".pdf":
            if provider == "marker":
                return self._convert_pdf(file_path)
            if provider == "native":
                return self._convert_pdf(file_path, use_marker=False)
            return self._convert_pdf(file_path)

        if file_ext in HTML_EXTENSIONS:
            return self._convert_html(loader, file_path)

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
