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


def fast_producer(*args, **kwargs):
    # produce many tokens rapidly
    for i in range(200):
        yield f"tok{i} "


@pytest.mark.anyio
async def test_stream_backpressure(monkeypatch):
    if graph_rag is None:
        pytest.skip(f"Neo4j not reachable at {settings.neo4j_uri}; skipping stream backpressure test")

    # monkeypatch generation to be very fast producer
    try:
        import rag.nodes.generation as gen_mod
        monkeypatch.setattr(gen_mod, "stream_generate_response", fast_producer)
    except Exception:
        pytest.skip("generation module not available")

    tokens = []

    # Consume slowly to encourage queue fills/drops
    async for data in graph_rag.stream_query("backpressure test"):
        try:
            payload = data.strip()[6:]
            obj = json.loads(payload)
        except Exception:
            continue

        if obj.get("type") == "token":
            tokens.append(obj.get("content"))
            # artificially slow consumer
            await asyncio.sleep(0.005)

    # Ensure we received tokens but potentially less than produced due to bounded queue
    assert len(tokens) > 0
    assert len(tokens) <= 200
