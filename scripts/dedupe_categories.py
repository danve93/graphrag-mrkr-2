#!/usr/bin/env python3
"""
Detect and optionally fix duplicate Category names in Neo4j (case-insensitive).

Usage:
  python scripts/dedupe_categories.py --list
  python scripts/dedupe_categories.py --apply --strategy delete-duplicates
  python scripts/dedupe_categories.py --apply --strategy merge-keep-first

Strategies:
- delete-duplicates: keep one arbitrary (first by created_at), delete others and their relationships
- merge-keep-first: re-link BELONGS_TO edges to the first category, delete duplicates; preserves children via CHILD_OF

Safety:
- Default is a dry-run; use --apply to perform changes.
- Prints a summary of actions.
"""
import argparse
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

from core.graph_db import graph_db


def find_duplicates():
    query = """
    MATCH (c:Category)
    WITH toLower(c.name) AS lname, collect(c) AS cats
    WHERE size(cats) > 1
    RETURN lname AS name_key, cats
    ORDER BY name_key
    """
    with graph_db.session_scope() as session:
        result = session.run(query)
        dups = []
        for rec in result:
            cats = rec["cats"]
            items = [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "created_at": str(c.get("created_at")) if c.get("created_at") else None,
                    "approved": c.get("approved", False),
                    "document_count": c.get("document_count", 0),
                }
                for c in cats
            ]
            dups.append({"key": rec["name_key"], "categories": items})
        return dups


def delete_duplicates(dups, apply=False):
    """Delete all but the first category in each duplicate group."""
    total_deleted = 0
    with graph_db.session_scope() as session:
        for group in dups:
            cats = group["categories"]
            # Keep the earliest by created_at if available, else first
            keep = sorted(
                cats,
                key=lambda x: (x.get("created_at") or "9999-12-31T00:00:00Z")
            )[0]
            to_delete = [c for c in cats if c["id"] != keep["id"]]
            if not to_delete:
                continue
            print(f"Duplicate key '{group['key']}' — keeping {keep['id']} ({keep['name']}) and deleting {len(to_delete)} others")
            if apply:
                for c in to_delete:
                    session.run(
                        """
                        MATCH (x:Category {id: $id})
                        OPTIONAL MATCH (x)-[r]-()
                        DELETE r, x
                        """,
                        id=c["id"],
                    )
                    total_deleted += 1
    return {"deleted": total_deleted}


def merge_keep_first(dups, apply=False):
    """Merge duplicates by keeping the first and re-linking relationships."""
    total_merged = 0
    with graph_db.session_scope() as session:
        for group in dups:
            cats = group["categories"]
            keep = sorted(
                cats,
                key=lambda x: (x.get("created_at") or "9999-12-31T00:00:00Z")
            )[0]
            to_merge = [c for c in cats if c["id"] != keep["id"]]
            if not to_merge:
                continue
            print(f"Duplicate key '{group['key']}' — keeping {keep['id']} and merging {len(to_merge)} others")
            if apply:
                for c in to_merge:
                    # Re-link BELONGS_TO from documents
                    session.run(
                        """
                        MATCH (d:Document)-[r:BELONGS_TO]->(c:Category {id: $dup_id})
                        MATCH (keep:Category {id: $keep_id})
                        MERGE (d)-[nr:BELONGS_TO]->(keep)
                        SET nr.confidence = coalesce(r.confidence, 1.0),
                            nr.auto_assigned = coalesce(r.auto_assigned, false),
                            nr.assigned_at = timestamp()
                        DELETE r
                        """,
                        dup_id=c["id"],
                        keep_id=keep["id"],
                    )
                    # Re-link children (CHILD_OF)
                    session.run(
                        """
                        MATCH (child:Category)-[rel:CHILD_OF]->(c:Category {id: $dup_id})
                        MATCH (keep:Category {id: $keep_id})
                        MERGE (child)-[:CHILD_OF]->(keep)
                        DELETE rel
                        """,
                        dup_id=c["id"],
                        keep_id=keep["id"],
                    )
                    # Delete duplicate category node
                    session.run(
                        """
                        MATCH (x:Category {id: $dup_id})
                        OPTIONAL MATCH (x)-[r]-()
                        DELETE r, x
                        """,
                        dup_id=c["id"],
                    )
                    total_merged += 1
    return {"merged": total_merged}


def main():
    parser = argparse.ArgumentParser(description="Detect and fix duplicate Category names (case-insensitive)")
    parser.add_argument("--list", action="store_true", help="List duplicate groups")
    parser.add_argument("--apply", action="store_true", help="Apply changes (otherwise dry-run)")
    parser.add_argument(
        "--strategy",
        choices=["delete-duplicates", "merge-keep-first"],
        default="delete-duplicates",
        help="Fix strategy to apply when --apply is set",
    )
    args = parser.parse_args()

    dups = find_duplicates()

    if not dups:
        print("No duplicate category names found.")
        return

    print(f"Found {len(dups)} duplicate name groups:")
    for g in dups:
        print(f"- key='{g['key']}'")
        for c in g["categories"]:
            print(f"  • id={c['id']} name={c['name']} created_at={c['created_at']} approved={c['approved']} doc_count={c['document_count']}")

    if args.list and not args.apply:
        return

    if args.apply:
        if args.strategy == "delete-duplicates":
            res = delete_duplicates(dups, apply=True)
            print(f"Deleted {res['deleted']} duplicate categories")
        elif args.strategy == "merge-keep-first":
            res = merge_keep_first(dups, apply=True)
            print(f"Merged {res['merged']} duplicate categories")
    else:
        print("Dry-run: no changes applied. Re-run with --apply to fix duplicates.")


if __name__ == "__main__":
    main()
