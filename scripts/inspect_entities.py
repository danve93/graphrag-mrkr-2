#!/usr/bin/env python3
"""Inspect Entity nodes in Neo4j for missing `id` and print samples.

This script uses the project's `core.singletons.get_graph_db_driver` to connect
with the same configuration the app uses.

Run: python scripts/inspect_entities.py
"""
import json
from neo4j import Driver
from core.graph_db import graph_db

if __name__ == '__main__':
    try:
        # Use graph_db.session_scope() to ensure driver lifecycle and retries
        pass
    except Exception as e:
        print(f"Failed to initialize graph_db: {e}")
        raise

    with graph_db.session_scope() as session:
        # Count total Entity nodes
        result = session.run("MATCH (e:Entity) RETURN count(e) AS total")
        total = result.single()["total"]
        print(f"Total Entity nodes: {total}")

        # Count entities missing id
        result = session.run("MATCH (e:Entity) WHERE e.id IS NULL RETURN count(e) AS missing")
        missing = result.single()["missing"]
        print(f"Entity nodes missing `id`: {missing}")

        # Show sample properties for nodes missing id
        result = session.run(
            "MATCH (e:Entity) WHERE e.id IS NULL RETURN elementId(e) AS internal_id, e AS props LIMIT 50"
        )
        samples = [record.data() for record in result]
        print("Samples of nodes missing id (up to 50):")
        print(json.dumps(samples, default=str, indent=2))

        # Check for legacy-like properties on all entities
        legacy_props = ["entity_id", "entityId", "uuid", "node_id"]
        legacy_counts = {}
        for prop in legacy_props:
            q = f"MATCH (e:Entity) WHERE e.{prop} IS NOT NULL RETURN count(e) AS c"
            try:
                r = session.run(q)
                legacy_counts[prop] = r.single()["c"]
            except Exception:
                legacy_counts[prop] = "error"
        print("Counts for potential legacy properties:")
        print(json.dumps(legacy_counts, indent=2))

    print("Inspection complete.")
