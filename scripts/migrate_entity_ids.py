#!/usr/bin/env python3
"""Migration: set `id` property on Entity nodes that are missing it.

This script previews nodes without `id`, then sets `id = elementId(e)` for them
and reports counts before/after.

Run: PYTHONPATH=. python3 scripts/migrate_entity_ids.py
"""
import json
from core.graph_db import graph_db


def main():
    with graph_db.session_scope() as session:
        # Count before
        total = session.run("MATCH (e:Entity) RETURN count(e) AS total").single()["total"]
        missing = session.run("MATCH (e:Entity) WHERE e.id IS NULL RETURN count(e) AS missing").single()["missing"]
        print(f"Total Entity nodes: {total}")
        print(f"Entity nodes missing id: {missing}")

        if missing == 0:
            print("No migration needed. Exiting.")
            return

        # Preview a few nodes
        preview = session.run(
            "MATCH (e:Entity) WHERE e.id IS NULL RETURN elementId(e) AS internal_id, e.name AS name, keys(e) AS keys LIMIT 20"
        )
        samples = [r.data() for r in preview]
        print("Sample nodes to be updated (up to 20):")
        print(json.dumps(samples, indent=2))

        # Apply migration: set id = elementId(e)
        print("Applying migration: setting e.id = elementId(e) for nodes where e.id IS NULL...")
        result = session.run(
            "MATCH (e:Entity) WHERE e.id IS NULL SET e.id = elementId(e) RETURN count(e) AS updated"
        )
        updated = result.single()["updated"]
        print(f"Updated nodes: {updated}")

        # Verify
        missing_after = session.run("MATCH (e:Entity) WHERE e.id IS NULL RETURN count(e) AS missing").single()["missing"]
        print(f"Entity nodes missing id after migration: {missing_after}")


if __name__ == '__main__':
    main()
