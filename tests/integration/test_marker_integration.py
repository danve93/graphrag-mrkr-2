"""
Integration test for Marker-backed PDF conversion in the ingestion pipeline.

This test uses a real PDF generated via reportlab and verifies that the
DocumentConverter converts it to markdown and attaches appropriate metadata.
It also validates fallback behavior when Marker is not available.
"""
import logging
from pathlib import Path

import pytest

from config.settings import settings
from ingestion.converters import document_converter
from core.chunking import DocumentChunker

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory):
    """Create a small multi-page real PDF using reportlab."""
    pdf_path = tmp_path_factory.mktemp("marker_pdf") / "sample.pdf"
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception as e:
        pytest.skip(f"reportlab not available: {e}")

    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    # Page 1 content
    c.drawString(72, 720, "Amber Integration Test Document")
    c.drawString(72, 700, "Components and Architecture")
    c.drawString(72, 680, "MTA (Mail Transfer Agent) routes emails to Mailstore.")
    c.drawString(72, 660, "Proxy provides secure access; Docs & Editor enable collaboration.")
    c.showPage()
    # Page 2 content
    c.drawString(72, 720, "Security & Certificates")
    c.drawString(72, 700, "TLS certificates encrypt communications; DOS filter protects services.")
    c.drawString(72, 680, "S/MIME enables signing; OTP adds strong authentication.")
    c.showPage()
    c.save()
    return pdf_path


def test_pdf_conversion_with_marker_toggle(sample_pdf, monkeypatch):
    """Verify PDF conversion with Marker toggle and fallback behavior."""
    # Enable marker toggle
    monkeypatch.setattr(settings, "use_marker_for_pdf", True)
    # Default config values already set; leave `marker_llm_service` None

    result = document_converter.convert(Path(sample_pdf))
    assert result is not None, "Conversion returned no result"

    content = result.get("content", "")
    metadata = result.get("metadata", {})

    assert isinstance(content, str) and len(content) > 0
    assert isinstance(metadata, dict)

    pipeline = metadata.get("conversion_pipeline")
    assert pipeline in {"marker", "smart_ocr_markdown"}

    # Content should include page headers for provenance
    assert "## Page 1" in content or "{1}" in content

    # Chunking should work downstream
    chunker = DocumentChunker()
    chunks = chunker.chunk_text(content, "test_marker_doc")
    assert len(chunks) > 0
    logger.info(f"✓ Converted PDF via {pipeline}; {len(chunks)} chunks created")


def test_pdf_conversion_without_marker_toggle(sample_pdf, monkeypatch):
    """Verify baseline conversion when Marker is disabled."""
    monkeypatch.setattr(settings, "use_marker_for_pdf", False)

    result = document_converter.convert(Path(sample_pdf))
    assert result is not None
    metadata = result.get("metadata", {})
    assert metadata.get("conversion_pipeline") == "smart_ocr_markdown"

    content = result.get("content", "")
    assert isinstance(content, str) and len(content) > 0
    assert "## Page 1" in content
