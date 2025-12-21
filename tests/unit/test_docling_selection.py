from pathlib import Path

from config.settings import settings
from ingestion.converters import DocumentConverter


def test_docling_selected_prefers_docling(monkeypatch, tmp_path: Path) -> None:
    converter = DocumentConverter()
    monkeypatch.setattr(settings, "document_conversion_provider", "docling")

    file_path = tmp_path / "sample.md"
    file_path.write_text("Hello Docling", encoding="utf-8")

    called = {"docling": False}

    def _fake_docling(path: Path):
        called["docling"] = True
        return {
            "content": "# Docling\n\nHello",
            "metadata": {"conversion_pipeline": "docling"},
        }

    monkeypatch.setattr(converter, "_convert_with_docling", _fake_docling)

    result = converter.convert(file_path)
    assert called["docling"] is True
    assert result is not None
    assert result["metadata"]["conversion_pipeline"] == "docling"


def test_docling_fallbacks_to_native_when_unavailable(monkeypatch, tmp_path: Path) -> None:
    converter = DocumentConverter()
    monkeypatch.setattr(settings, "document_conversion_provider", "docling")

    file_path = tmp_path / "sample.md"
    file_path.write_text("Fallback path", encoding="utf-8")

    monkeypatch.setattr(converter, "_convert_with_docling", lambda _: None)

    result = converter.convert(file_path)
    assert result is not None
    assert result["metadata"]["conversion_pipeline"] in {"plain_markdown", "structured_markdown"}
