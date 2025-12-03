#!/usr/bin/env python3
"""
Batch document classification and metadata update.

Usage:
  python scripts/reindex_classification.py --limit 100
  python scripts/reindex_classification.py --doc-id <id>

Runs LLM-assisted classification (if enabled) and updates Document + Chunk metadata.
Safe to run incrementally.
"""

import argparse
import logging
from typing import List, Dict, Any

from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=getattr(settings, "log_level", "INFO"))


def get_document_ids(limit: int | None = None) -> List[str]:
    try:
        return graph_db.list_document_ids(limit=limit)
    except Exception as e:
        logger.error(f"Failed to list document ids: {e}")
        return []


def classify_and_update(doc_id: str) -> Dict[str, Any]:
    try:
        doc = graph_db.get_document_metadata(doc_id)
        content = graph_db.get_document_text(doc_id)
        if not content:
            return {"status": "skipped", "reason": "no_content", "document_id": doc_id}

        dp = DocumentProcessor()
        if not getattr(settings, "enable_document_classification", False):
            logger.info("Classification disabled; set ENABLE_DOCUMENT_CLASSIFICATION=1 to enable.")
            return {"status": "skipped", "reason": "disabled", "document_id": doc_id}

        cls = dp.classify_document_categories(doc.get("filename", doc_id), content)
        confidence = float(cls.get("confidence", 0.0))
        apply_cls = confidence >= getattr(settings, "classification_confidence_threshold", 0.7)
        category = (cls.get("categories", []) or [settings.classification_default_category])[0]
        if not apply_cls:
            category = settings.classification_default_category
        enrich = {
            "category": category,
            "categories": cls.get("categories", []),
            "classification_confidence": confidence,
            "keywords": cls.get("keywords", []),
            "difficulty": cls.get("difficulty", "intermediate"),
        }
        graph_db.create_document_node(doc_id, enrich)

        # Update chunk metadata with category
        try:
            chunks = graph_db.get_document_chunks(doc_id)
            updated = 0
            for c in chunks:
                chunk_id = c.get("chunk_id")
                md = (c.get("metadata") or {})
                md["category"] = category
                graph_db.update_chunk_metadata(chunk_id, md)
                updated += 1
            logger.info(f"Updated {updated} chunks with category={category} for {doc_id}")
        except Exception as e:
            logger.debug(f"Failed to update chunk metadata for {doc_id}: {e}")

        return {"status": "success", "document_id": doc_id, "category": category, "confidence": confidence}
    except Exception as e:
        logger.error(f"Classification failed for {doc_id}: {e}")
        return {"status": "error", "document_id": doc_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch classification reindexer")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    parser.add_argument("--doc-id", type=str, default=None, help="Process a single document id")
    args = parser.parse_args()

    if args.doc_id:
        res = classify_and_update(args.doc_id)
        print(res)
        return

    ids = get_document_ids(limit=args.limit)
    ok = 0
    for i, doc_id in enumerate(ids, start=1):
        res = classify_and_update(doc_id)
        if res.get("status") == "success":
            ok += 1
        print(f"[{i}/{len(ids)}] {res}")
    print({"processed": len(ids), "success": ok})


if __name__ == "__main__":
    main()
