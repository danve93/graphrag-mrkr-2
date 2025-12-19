#!/usr/bin/env python3
"""
Repair missing (:Chunk)-[:CONTAINS_ENTITY]->(:Entity) relationships for one or more documents.

Why this exists:
- Some ingestion paths can create Entity nodes with `source_chunks` populated but fail to
  persist chunk->entity relationships reliably.
- The single-document view (and multiple API endpoints) count entities by traversing
  Document->Chunk->CONTAINS_ENTITY->Entity. Missing relationships make counts show as 0
  even when extraction ran.

This script is idempotent: it uses MERGE-style repairs under the hood.

Typical usage (Docker):
  docker compose exec backend python scripts/repair_entity_links.py --list
  docker compose exec backend python scripts/repair_entity_links.py --apply --update-summary
  docker compose exec backend python scripts/repair_entity_links.py --doc-id <doc_id> --apply --update-summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when running as a script (sys.path[0] becomes /app/scripts).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _find_docs_needing_repair(limit: int) -> List[Dict[str, Any]]:
    from core.graph_db import graph_db

    with graph_db.session_scope() as session:
        # Identify docs where entities reference the doc's chunks via `source_chunks`,
        # but those entities are not (fully) reachable through CONTAINS_ENTITY edges.
        rows = session.run(
            """
            MATCH (d:Document)
            WHERE d.entity_extraction_metrics IS NOT NULL
            MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WITH d, collect(c.id) AS doc_chunk_ids
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[:CONTAINS_ENTITY]->(linked:Entity)
            WITH d, doc_chunk_ids, count(DISTINCT linked) AS linked_entities
            MATCH (e:Entity)
            WHERE any(cid IN coalesce(e.source_chunks, []) WHERE cid IN doc_chunk_ids)
            WITH d, linked_entities, count(DISTINCT e) AS entities_with_sources
            WHERE entities_with_sources > linked_entities
            RETURN d.id AS id,
                   d.filename AS filename,
                   entities_with_sources,
                   linked_entities
            ORDER BY (entities_with_sources - linked_entities) DESC, entities_with_sources DESC
            LIMIT $limit
            """,
            limit=limit,
        ).data()

    return [
        {
            "id": r.get("id"),
            "filename": r.get("filename"),
            "entities_with_sources": int(r.get("entities_with_sources") or 0),
            "linked_entities": int(r.get("linked_entities") or 0),
        }
        for r in rows
    ]


def _repair_doc(doc_id: str, update_summary: bool) -> Dict[str, Any]:
    from core.graph_db import graph_db

    repair_stats = graph_db.repair_contains_entity_relationships_for_document(doc_id)

    summary_stats = None
    if update_summary:
        try:
            summary_stats = graph_db.update_document_precomputed_summary(doc_id)
            try:
                graph_db.update_document_preview(doc_id)
            except Exception:
                pass
        except Exception:
            summary_stats = None

    return {
        "doc_id": doc_id,
        "repair": repair_stats,
        "summary": summary_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair missing chunk->entity links for documents.")
    parser.add_argument("--doc-id", help="Repair a single document by id.")
    parser.add_argument("--list", action="store_true", help="List documents likely needing repair (no changes).")
    parser.add_argument("--limit", type=int, default=200, help="Max documents to list/repair (default: 200).")
    parser.add_argument("--apply", action="store_true", help="Apply repairs (otherwise dry-run/list only).")
    parser.add_argument(
        "--update-summary",
        action="store_true",
        help="After repair, refresh `precomputed_*` counts and previews on the Document node.",
    )
    args = parser.parse_args()

    if args.doc_id:
        if not args.apply:
            print("--doc-id requires --apply to make changes.")
            return 2
        result = _repair_doc(args.doc_id, update_summary=args.update_summary)
        print(result)
        return 0

    if args.list and not args.apply:
        docs = _find_docs_needing_repair(limit=args.limit)
        if not docs:
            print("No documents found needing repair.")
            return 0
        for d in docs:
            delta = d["entities_with_sources"] - d["linked_entities"]
            print(
                f'{d["id"]}  {d.get("filename") or ""}  linked={d["linked_entities"]}  '
                f'with_sources={d["entities_with_sources"]}  missing≈{delta}'
            )
        return 0

    if args.apply:
        docs = _find_docs_needing_repair(limit=args.limit)
        if not docs:
            print("No documents found needing repair.")
            return 0

        repaired = 0
        for d in docs:
            doc_id = d["id"]
            out = _repair_doc(doc_id, update_summary=args.update_summary)
            repaired += 1
            print(out)

        print(f"Repaired {repaired} document(s).")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
