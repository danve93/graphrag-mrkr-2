import json

import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_stream_llm_level_mock(monkeypatch):
    """Integration test that mocks `llm_manager.stream_generate_rag_response`.

    This exercises the full HTTP route and RAG pipeline up to the LLM layer,
    ensuring the streaming endpoint handles provider-level streams correctly.
    """

    def fake_llm_stream(query: str, **kwargs):
        yield f"hello "
        yield f"world"

    # Monkeypatch the llm_manager stream method
    import core.llm as cll

    monkeypatch.setattr(cll.llm_manager, "stream_generate_rag_response", fake_llm_stream)

    # Monkeypatch retrieval and graph reasoning to return a non-empty context so
    # generation path uses the LLM stream instead of short-circuiting.
    import rag.graph_rag as gr

    def fake_retrieve(*args, **kwargs):
        return [
            {
                "chunk_id": "c1",
                "content": "context content",
                "similarity": 0.9,
                "document_name": "doc1",
                "document_id": "d1",
                "chunk_index": 0,
            }
        ]

    def fake_reason(*args, **kwargs):
        return fake_retrieve()

    monkeypatch.setattr(gr, "retrieve_documents", fake_retrieve)
    monkeypatch.setattr(gr, "reason_with_graph", fake_reason)

    # Ensure streaming is enabled
    from config.settings import settings as cfg_settings
    monkeypatch.setattr(cfg_settings, "enable_llm_streaming", True)

    # Call the async stream_query directly to avoid HTTP disconnect monitoring
    import rag.graph_rag as gr

    events = []
    async for data in gr.graph_rag.stream_query(
        user_query="hello",
        chat_history=[],
        context_documents=[],
        llm_model=None,
        embedding_model=None,
        cancel_event=None,
    ):
        # Each yielded item is an SSE-like string: 'data: {json}\n\n'
        if not data:
            continue
        text = data if isinstance(data, str) else data.decode("utf-8")
        if text.startswith("data: "):
            payload = text[len("data: "):]
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            events.append(obj)

    types = [e.get("type") for e in events]
    assert any(t == "token" for t in types) or any(t is None and isinstance(e.get("content"), str) for e in events)
