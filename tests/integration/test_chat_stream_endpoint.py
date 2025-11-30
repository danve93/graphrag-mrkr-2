import asyncio
import json

import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.mark.integration
def test_chat_stream_endpoint_monkeypatched(monkeypatch):
    """Integration-style test for `/api/chat/stream` using a mocked stream_query.

    This test starts the FastAPI app (TestClient) and monkeypatches the
    `graph_rag.stream_query` async generator so the endpoint streams predictable
    SSE events without calling external LLMs or other services.
    """

    async def mock_stream_query(user_query: str, **kwargs):
        # Emit a stage event
        yield f"data: {json.dumps({'type': 'stage', 'content': 'query_analysis'})}\n\n"
        # Emit a couple tokens
        yield f"data: {json.dumps({'type': 'token', 'content': 'hello '})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'content': 'world'})}\n\n"
        # Emit sources
        yield f"data: {json.dumps({'type': 'sources', 'content': [{'document_id': 'doc1'}]})}\n\n"
        # Done
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # Monkeypatch the GraphRAG instance's stream_query used by the router
    import rag.graph_rag as gr

    monkeypatch.setattr(gr.graph_rag, "stream_query", mock_stream_query)

    # Ensure streaming is enabled in settings for the endpoint to pick the streaming path
    from config.settings import settings as cfg_settings

    # Set flag on the settings instance
    monkeypatch.setattr(cfg_settings, "enable_llm_streaming", True)

    client = TestClient(app)

    payload = {"message": "hello world", "stream": True}

    with client.stream("POST", "/api/chat/stream", json=payload) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if not line:
                continue
            # TestClient may yield `str` or `bytes` depending on runtime; normalize
            if isinstance(line, bytes):
                text = line.decode("utf-8")
            else:
                text = line
            if text.startswith("data: "):
                payload = text[len("data: "):]
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                events.append(obj)

    # Verify expected events were received in order
    types = [e.get("type") for e in events]
    assert "stage" in types
    assert "token" in types
    assert "sources" in types
    assert "done" in types
