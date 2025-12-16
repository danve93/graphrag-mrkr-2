import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from rag.nodes.smart_consolidation import SmartConsolidator


def _make_chunks():
    return [
        {"id": 1, "category": "install", "text": "Install guide", "score": 0.9},
        {"id": 2, "document_category": "configure", "text": "Configure guide", "score": 0.8},
        {"id": 3, "text": "Uncategorized chunk", "score": 0.7},
    ]


def test_group_by_category_falls_back_to_uncategorized():
    consolidator = SmartConsolidator()
    groups = consolidator._group_by_category(_make_chunks())

    assert set(groups.keys()) == {"install", "configure", "uncategorized"}
    assert groups["install"][0]["id"] == 1
    assert groups["configure"][0]["id"] == 2
    assert groups["uncategorized"][0]["id"] == 3


def test_ensure_representation_reserves_slots_per_category():
    consolidator = SmartConsolidator(min_chunks_per_category=1)
    groups = {
        "install": [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.7},
        ],
        "configure": [
            {"id": 3, "score": 0.85},
        ],
        "troubleshooting": [
            {"id": 4, "score": 0.95},
        ],
    }

    selected = consolidator._ensure_representation(groups, ["install", "configure"], top_k=3)

    # At least one per target category, remaining slot goes to best overall
    ids = [c["id"] for c in selected]
    assert 1 in ids and 3 in ids
    assert len(selected) == 3
    # Highest remaining chunk (id 4) should fill the last slot
    assert 4 in ids


def test_deduplicate_semantic_keeps_highest_score():
    consolidator = SmartConsolidator(semantic_threshold=0.9)
    chunks = [
        {"id": 1, "text": "alpha text", "score": 0.9},
        {"id": 2, "text": "alpha duplicate", "score": 0.8},
        {"id": 3, "text": "beta text", "score": 0.7},
    ]

    with patch(
        "rag.nodes.smart_consolidation.embedding_manager.get_embedding",
        new=AsyncMock(side_effect=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    ):
        deduped = asyncio.run(consolidator._deduplicate_semantic(chunks))

    ids = [c["id"] for c in deduped]
    assert ids == [1, 3], "Duplicate with lower score should be removed"


def test_enforce_token_budget_trims_lowest_ranked_when_over_budget():
    consolidator = SmartConsolidator(max_tokens=10)
    chunks = [
        {"id": 1, "text": "a" * 20, "score": 1.0},  # ~5 tokens
        {"id": 2, "text": "b" * 20, "score": 0.9},  # ~5 tokens
        {"id": 3, "text": "c" * 20, "score": 0.8},  # would exceed budget
    ]

    result = consolidator._enforce_token_budget(chunks)

    assert [c["id"] for c in result] == [1, 2]


def test_consolidate_applies_representation_dedup_and_budget():
    consolidator = SmartConsolidator(
        max_tokens=1000,
        semantic_threshold=0.9,
        ensure_category_representation=True,
        min_chunks_per_category=1,
    )
    chunks = [
        {"id": 1, "category": "install", "text": "Install alpha", "score": 0.9},
        {"id": 2, "category": "install", "text": "Install alpha copy", "score": 0.85},
        {"id": 3, "category": "configure", "text": "Configure tips", "score": 0.8},
        {"id": 4, "category": "troubleshooting", "text": "Other chunk", "score": 0.5},
    ]

    with patch(
        "rag.nodes.smart_consolidation.embedding_manager.get_embedding",
        new=AsyncMock(side_effect=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    ):
        result = asyncio.run(consolidator.consolidate(chunks, categories=["install", "configure"], top_k=3))

    ids = [c["id"] for c in result]
    # One per requested category and deduplication removes duplicate install chunk
    assert 1 in ids and 3 in ids
    assert 2 not in ids
    assert len(result) == 2
