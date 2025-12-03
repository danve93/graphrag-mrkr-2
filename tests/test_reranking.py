"""Tests for reranking behavior: blend and dynamic candidate cap."""

import pytest
from unittest.mock import patch, MagicMock


def _fake_ranker_with_scores(score_map):
    class _FakeRanker:
        def rerank(self, req):
            return [{"id": pid["id"], "score": score_map.get(pid["id"], 0.0)} for pid in req.passages]
    return _FakeRanker()


def _make_candidates(n=10):
    return [
        {
            "chunk_id": f"c{i}",
            "content": f"content {i}",
            "hybrid_score": (n - i) / n,  # descending hybrid
        }
        for i in range(n)
    ]


def test_flashrank_blend_extremes(monkeypatch):
    from rag.rerankers import flashrank_reranker as fr
    from config.settings import settings

    # Prepare fake ranker with reverse scores (so reranker order opposes hybrid)
    score_map = {f"c{i}": i / 10.0 for i in range(10)}  # ascending rerank
    monkeypatch.setattr(fr, "_available", True)
    monkeypatch.setattr(fr, "_init_ranker", lambda: _fake_ranker_with_scores(score_map))

    candidates = _make_candidates(10)

    # Pure reranker ordering
    settings.flashrank_enabled = True
    settings.flashrank_blend_weight = 0.0
    out = fr.rerank_with_flashrank("q", candidates[:])
    # Expect first item to have highest rerank score (c9)
    assert out[0]["chunk_id"] == "c9"

    # Blended strongly toward hybrid (low rerank weight -> prefer hybrid)
    settings.flashrank_blend_weight = 0.05
    out2 = fr.rerank_with_flashrank("q", candidates[:])
    # Expect first item to follow hybrid order (c0)
    assert out2[0]["chunk_id"] == "c0"


@pytest.mark.asyncio
async def test_dynamic_candidate_cap(monkeypatch):
    from rag.retriever import DocumentRetriever
    from config.settings import settings

    # Build retriever and patch retrieval stages to produce many items
    retriever = DocumentRetriever()
    many = _make_candidates(120)

    async def fake_chunk(*args, **kwargs):
        return many[:]

    async def fake_entity(*args, **kwargs):
        return []

    # Capture cap passed to reranker
    called = {"cap": None}

    def fake_rerank(query, candidates, max_candidates=None):
        called["cap"] = max_candidates
        return candidates

    settings.flashrank_enabled = True
    settings.flashrank_max_candidates = 50

    with patch.object(retriever, "chunk_based_retrieval", side_effect=fake_chunk):
        with patch.object(retriever, "entity_based_retrieval", side_effect=fake_entity):
            with patch("core.graph_db.graph_db.chunk_keyword_search", return_value=[]):
                with patch("rag.nodes.query_analysis.analyze_query", return_value={"multi_hop_recommended": False, "query_type": "factual"}):
                    with patch("rag.rerankers.flashrank_reranker.rerank_with_flashrank", side_effect=fake_rerank):
                        out = await retriever._hybrid_retrieval_direct(
                            query="q",
                            top_k=5,
                            chunk_weight=0.8,
                            entity_weight=0.2,
                        )

    # Dynamic base = max(5*4, 32) = 32; settings cap=50; len=120 => expect 32
    assert called["cap"] == 32
    assert len(out) >= 5
