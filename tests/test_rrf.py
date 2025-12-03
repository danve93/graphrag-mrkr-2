"""Unit tests for Reciprocal Rank Fusion (RRF) in retriever."""

import pytest
from unittest.mock import patch


def test_apply_rrf_basic():
    """RRF should rank items appearing in multiple lists higher."""
    from rag.retriever import DocumentRetriever

    r = DocumentRetriever()
    list_a = [
        {"chunk_id": "A"},
        {"chunk_id": "B"},
        {"chunk_id": "C"},
    ]
    list_b = [
        {"chunk_id": "C"},
        {"chunk_id": "A"},
        {"chunk_id": "D"},
    ]
    scores = r._apply_rrf([list_a, list_b], k=60)

    # A appears in both lists with good ranks; should outrank singletons
    assert scores["A"] > scores["B"]
    assert scores["A"] > scores["D"]
    # C is rank1 in list_b so should be competitive
    assert scores["C"] >= scores["B"]


@pytest.mark.asyncio
async def test_rrf_integration_flag():
    """When enable_rrf is True, hybrid retrieval uses RRF ordering."""
    from rag.retriever import DocumentRetriever
    from config.settings import settings

    retriever = DocumentRetriever()

    # Prepare synthetic modality lists
    chunk_results = [
        {"chunk_id": "X", "similarity": 0.9, "content": "X"},
        {"chunk_id": "Y", "similarity": 0.8, "content": "Y"},
    ]
    entity_results = [
        {"chunk_id": "Y", "similarity": 0.95, "content": "Y via entity"},
        {"chunk_id": "Z", "similarity": 0.7, "content": "Z via entity"},
    ]
    keyword_results = [
        {"chunk_id": "W", "keyword_score": 3.0, "content": "W via keyword"},
        {"chunk_id": "X", "keyword_score": 2.0, "content": "X via keyword"},
    ]

    # Enable RRF
    orig_rrf = settings.enable_rrf
    orig_kw = settings.enable_chunk_fulltext
    try:
        settings.enable_rrf = True
        settings.enable_chunk_fulltext = True

        with patch.object(retriever, 'chunk_based_retrieval', return_value=chunk_results):
            with patch.object(retriever, 'entity_based_retrieval', return_value=entity_results):
                with patch('core.graph_db.graph_db.chunk_keyword_search', return_value=keyword_results):
                    results = await retriever._hybrid_retrieval_direct(
                        query="q",
                        top_k=5,
                        chunk_weight=0.5,
                        entity_weight=0.5,
                        restrict_to_context=False,
                    )

        # Expectations: Y appears in chunk+entity, X in chunk+keyword; both should be at the top
        top_ids = [r["chunk_id"] for r in results[:2]]
        assert set(top_ids) == {"X", "Y"}
        # Ensure rrf_score is present
        assert all("rrf_score" in r for r in results[:2])
    finally:
        settings.enable_rrf = orig_rrf
        settings.enable_chunk_fulltext = orig_kw
