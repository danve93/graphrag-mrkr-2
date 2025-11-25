import asyncio
import sys
import os

# Ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import settings as cfg_settings
from rag.retriever import document_retriever


def _make_candidate(cid, text, sim):
    return {"chunk_id": cid, "content": text, "similarity": sim}


def test_hybrid_retrieval_uses_reranker(monkeypatch):
    # Enable flashrank in settings
    cfg_settings.settings.flashrank_enabled = True
    cfg_settings.settings.flashrank_max_candidates = 10

    # Provide deterministic chunk results
    chunks = [
        _make_candidate("c1", "First passage", 0.9),
        _make_candidate("c2", "Second passage", 0.85),
        _make_candidate("c3", "Third passage", 0.8),
    ]

    async def fake_chunk_retrieval(query, top_k, allowed_document_ids=None, query_embedding=None):
        return chunks

    async def fake_entity_retrieval(query, top_k, allowed_document_ids=None):
        return []

    monkeypatch.setattr(document_retriever, "chunk_based_retrieval", fake_chunk_retrieval)
    monkeypatch.setattr(document_retriever, "entity_based_retrieval", fake_entity_retrieval)

    # Mock reranker to simply reverse the candidate order
    def fake_rerank(query, candidates, max_candidates=None):
        out = list(reversed(candidates))
        for i, c in enumerate(out):
            c["rerank_score"] = float(len(out) - i)
        return out

    import rag.rerankers.flashrank_reranker as fr

    monkeypatch.setattr(fr, "rerank_with_flashrank", fake_rerank)

    # Run hybrid retrieval
    results = asyncio.run(document_retriever.hybrid_retrieval("test query", top_k=3, chunk_weight=1.0, entity_weight=0.0))

    # Ensure the order is reversed by the reranker
    assert [r["chunk_id"] for r in results] == ["c3", "c2", "c1"]

    # Cleanup toggle
    cfg_settings.settings.flashrank_enabled = False


def test_reranker_fail_open(monkeypatch):
    cfg_settings.settings.flashrank_enabled = True

    chunks = [
        _make_candidate("x1", "X one", 0.9),
        _make_candidate("x2", "X two", 0.8),
    ]

    async def fake_chunk_retrieval(query, top_k, allowed_document_ids=None, query_embedding=None):
        return chunks

    async def fake_entity_retrieval(query, top_k, allowed_document_ids=None):
        return []

    monkeypatch.setattr(document_retriever, "chunk_based_retrieval", fake_chunk_retrieval)
    monkeypatch.setattr(document_retriever, "entity_based_retrieval", fake_entity_retrieval)

    def broken_rerank(query, candidates, max_candidates=None):
        raise RuntimeError("simulated failure")

    import rag.rerankers.flashrank_reranker as fr

    monkeypatch.setattr(fr, "rerank_with_flashrank", broken_rerank)

    results = asyncio.run(document_retriever.hybrid_retrieval("q", top_k=2, chunk_weight=1.0, entity_weight=0.0))

    # On failure, original order should be preserved (sorted by hybrid_score -> chunk similarity)
    assert [r["chunk_id"] for r in results] == ["x1", "x2"]

    cfg_settings.settings.flashrank_enabled = False
