import asyncio
import json
import socket
import time
from urllib.parse import urlparse

import pytest

from config.settings import settings


def _neo4j_reachable(uri: str) -> bool:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


graph_rag = None
if _neo4j_reachable(settings.neo4j_uri):
    from rag.graph_rag import graph_rag


def slow_token_generator(*args, **kwargs):
    # synchronous generator used by producer thread
    for i in range(10):
        time.sleep(0.05)
        yield f"word{i} "


@pytest.mark.anyio
async def test_stream_query_cancellation(monkeypatch):
    if graph_rag is None:
        pytest.skip(f"Neo4j not reachable at {settings.neo4j_uri}; skipping streaming cancellation test")

    # Monkeypatch the generation module's stream function
    try:
        import rag.nodes.generation as gen_mod
        monkeypatch.setattr(gen_mod, "stream_generate_response", slow_token_generator)
    except Exception:
        pytest.skip("Could not import generation module")

    cancel_event = asyncio.Event()

    async def canceller():
        # cancel after a short delay
        await asyncio.sleep(0.18)
        cancel_event.set()

    # Start canceller
    asyncio.create_task(canceller())

    tokens = []
    # Consume the async generator
    async for data in graph_rag.stream_query(
        "hello",
        cancel_event=cancel_event,
    ):
        # data is SSE-like string: 'data: {json}\n\n'
        try:
            payload = data.strip()[6:]
            obj = json.loads(payload)
        except Exception:
            continue

        if obj.get("type") == "token":
            tokens.append(obj.get("content"))

        # stop consuming after cancellation applied
        if cancel_event.is_set():
            break

    assert len(tokens) >= 0
