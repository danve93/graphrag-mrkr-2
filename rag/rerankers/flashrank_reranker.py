"""
Lightweight FlashRank wrapper for optional reranking.

This module provides a simple, lazy wrapper around the `flashrank` library.
If `flashrank` is not installed, the functions will behave as no-ops (returning
the original candidate ordering) so the pipeline remains functional and tests
can mock the reranker easily.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Lazy-loaded ranker singleton and init lock
_ranker = None
_init_lock = threading.Lock()
_available = True

try:
    from flashrank import Ranker, RerankRequest  # type: ignore
except Exception:
    # flashrank is optional — if unavailable, we will no-op
    _available = False


def _init_ranker() -> Optional[object]:
    global _ranker, _available
    if not _available:
        return None

    if _ranker is not None:
        return _ranker

    with _init_lock:
        if _ranker is not None:
            return _ranker

        try:
            cache_dir = getattr(settings, "flashrank_cache_dir", None)
            model_name = getattr(settings, "flashrank_model_name", None)
            _ranker = Ranker(model_name=model_name, cache_dir=cache_dir, max_length=getattr(settings, "flashrank_max_length", 128))
            logger.info("FlashRank ranker initialized: %s", model_name)
            return _ranker
        except Exception as e:
            logger.exception("Failed to initialize FlashRank ranker: %s", e)
            _available = False
            return None


def prewarm_ranker() -> Optional[object]:
    """Public helper to pre-warm / initialize the FlashRank ranker.

    This is safe to call multiple times; the underlying initializer is guarded
    with a lock and will no-op if FlashRank is not installed or initialization
    previously failed.
    """
    try:
        ranker = _init_ranker()
        if ranker is not None:
            logger.info("FlashRank pre-warm completed: %s", getattr(settings, "flashrank_model_name", None))
        else:
            logger.info("FlashRank pre-warm skipped or failed; ranker unavailable")
        return ranker
    except Exception as e:
        logger.exception("FlashRank pre-warm raised an unexpected error: %s", e)
        return None


def rerank_with_flashrank(query: str, candidates: List[Dict], max_candidates: Optional[int] = None) -> List[Dict]:
    """Rerank a list of candidate dicts using FlashRank.

    Args:
        query: User query string
        candidates: List of candidate dicts containing at least `chunk_id`/`id` and `content`/`text`.
        max_candidates: Optional cap on how many candidates to send to reranker.

    Returns:
        Reordered list of candidates (same items, possibly reordered). If FlashRank isn't
        available or an error occurs, the original list is returned unchanged.
    """
    if not getattr(settings, "flashrank_enabled", False):
        return candidates

    if not _available:
        logger.warning("FlashRank not available; returning original candidate order")
        return candidates

    ranker = _init_ranker()
    if ranker is None:
        return candidates

    try:
        cap = max_candidates or getattr(settings, "flashrank_max_candidates", 100)
        to_rank = candidates[:cap]

        passages = []
        for c in to_rank:
            cid = c.get("chunk_id") or c.get("id")
            text = c.get("content") or c.get("text") or ""
            meta = {k: v for k, v in c.items() if k not in ("content", "text")}
            passages.append({"id": cid, "text": text, "meta": meta})

        rerank_req = RerankRequest(query=query, passages=passages)
        t0 = time.perf_counter()
        results = ranker.rerank(rerank_req)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        logger.info("FlashRank rerank completed: candidates=%d elapsed_ms=%.1f model=%s", len(to_rank), elapsed_ms, getattr(settings, "flashrank_model_name", None))

        # results is a list of dicts with id and score — create mapping
        score_map = {r["id"]: r.get("score", 0.0) for r in results}

        # Attach rerank_score and produce new order
        for c in to_rank:
            cid = c.get("chunk_id") or c.get("id")
            c["rerank_score"] = score_map.get(cid, 0.0)

        # Combine scoring if requested
        blend = float(getattr(settings, "flashrank_blend_weight", 0.0) or 0.0)
        if blend > 0.0:
            for c in to_rank:
                hybrid = c.get("hybrid_score", c.get("similarity", 0.0))
                rer = c.get("rerank_score", 0.0)
                c["combined_score"] = blend * rer + (1.0 - blend) * hybrid
            to_rank.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
        else:
            to_rank.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # Reconstruct final list: reranked top portion + the rest
        reordered = to_rank + candidates[cap:]
        return reordered

    except Exception as e:
        logger.exception("FlashRank reranking failed: %s", e)
        return candidates
