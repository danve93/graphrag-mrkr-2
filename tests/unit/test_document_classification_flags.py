from ingestion.document_processor import DocumentProcessor
from config.settings import settings


def test_classify_document_categories_parses_llm_response(monkeypatch):
    dp = DocumentProcessor.__new__(DocumentProcessor)  # bypass heavy __init__
    monkeypatch.setattr(dp, "_load_category_config", lambda: {"categories": {}})
    monkeypatch.setattr(
        "core.llm.llm_manager.generate_response",
        lambda **kwargs: '{"categories": ["install"], "confidence": 0.91, "keywords": ["setup"], "difficulty": "beginner"}',
    )

    result = dp.classify_document_categories("guide.txt", "Install steps here")

    assert result["categories"] == ["install"]
    assert result["confidence"] == 0.91
    assert result["keywords"] == ["setup"]
    assert result["difficulty"] == "beginner"


def test_classify_document_categories_falls_back_to_default(monkeypatch):
    dp = DocumentProcessor.__new__(DocumentProcessor)
    monkeypatch.setattr(dp, "_load_category_config", lambda: {"categories": {}})
    monkeypatch.setattr("core.llm.llm_manager.generate_response", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("LLM unavailable")))
    monkeypatch.setattr(settings, "classification_default_category", "general-test", raising=True)

    result = dp.classify_document_categories("notes.txt", "Random unrelated content")

    assert result["categories"] == ["general-test"]
    assert result["confidence"] == 0.5
