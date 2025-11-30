#!/usr/bin/env python3
"""Build a Leiden-friendly projection from the entity graph.

This script extracts Entity nodes and their undirected relationships (``SIMILAR_TO``
by default, optionally ``RELATED_TO``) into serialized CSV files that can be
consumed by Leiden clustering or downstream tooling. It also records the
projection parameters (edge types, minimum weight filter, resolution hint) in a
metadata file so experiments can be reproduced without code changes.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config.settings import settings  # noqa: E402
from core.graph_analysis.leiden_utils import (  # noqa: E402
    ENTITY_EDGE_LABELS,
    build_entity_leiden_projection_cypher,
)
from core.graph_db import graph_db  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Entity graph projection for Leiden clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/leiden"),
        help="Directory where nodes.csv, edges.csv, and metadata.json will be written.",
    )
    parser.add_argument(
        "--edge-types",
        nargs="+",
        default=list(ENTITY_EDGE_LABELS),
        help="Undirected relationship types to include (e.g., SIMILAR_TO RELATED_TO).",
    )
    parser.add_argument(
        "--directional-edge-types",
        nargs="+",
        default=None,
        help=(
            "Optional directional relationship labels to mirror for undirected Leiden runs. "
            "Omit unless you need to include directional edges."
        ),
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop edges whose normalized weight falls below this threshold.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help=(
            "Resolution parameter to record alongside the projection; helps reproduce "
            "Leiden runs without editing this script."
        ),
    )
    parser.add_argument(
        "--weight-field",
        type=str,
        default="weight",
        help="Name of the weight field to emit for edges.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_entity_nodes() -> list[Mapping[str, object]]:
    query = """
    MATCH (e:Entity)
    RETURN id(e) AS internal_id, e.id AS entity_id, e.name AS name, labels(e) AS labels
    """
    with graph_db.session_scope() as session:
        result = session.run(query)
        nodes = [record.data() for record in result]

    for node in nodes:
        if node.get("entity_id") is None:
            raise ValueError("Entity node missing required `entity_id` property")
        if not node.get("labels"):
            raise ValueError(f"Entity {node['entity_id']} is missing labels")

    return nodes


def load_entity_edges(
    relationship_labels: Sequence[str],
    directional_labels: Sequence[str] | None,
    weight_field: str,
) -> list[Mapping[str, object]]:
    cypher = build_entity_leiden_projection_cypher(
        weight_field=weight_field,
        relationship_labels=relationship_labels,
        directional_labels=directional_labels,
    )
    with graph_db.session_scope() as session:
        result = session.run(cypher)
        edges = [record.data() for record in result]

    for edge in edges:
        if weight_field not in edge:
            raise ValueError(f"Edge projection missing `{weight_field}` field")
        weight = edge[weight_field]
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Edge weight must be numeric; got {weight!r}")
        if edge.get("source") is None or edge.get("target") is None:
            raise ValueError("Edge projection missing `source` or `target`")

    return edges


def filter_edges_by_weight(
    edges: Iterable[Mapping[str, object]],
    min_weight: float,
    weight_field: str,
) -> list[Mapping[str, object]]:
    filtered: list[Mapping[str, object]] = []
    for edge in edges:
        weight = float(edge[weight_field])
        if weight >= min_weight:
            filtered.append(edge)
    return filtered


def map_edge_node_ids(
    edges: Iterable[Mapping[str, object]],
    node_id_lookup: Mapping[int, str],
    weight_field: str,
) -> list[Mapping[str, object]]:
    mapped: list[Mapping[str, object]] = []
    for edge in edges:
        source_internal = edge["source"]
        target_internal = edge["target"]
        if source_internal not in node_id_lookup or target_internal not in node_id_lookup:
            raise ValueError(
                "Edge references an Entity missing an `entity_id` mapping; ensure nodes are loaded first."
            )

        mapped.append(
            {
                "source": node_id_lookup[source_internal],
                "target": node_id_lookup[target_internal],
                weight_field: float(edge[weight_field]),
                "undirected": True,
            }
        )
    return mapped


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metadata(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    configure_logging()

    logging.info("Loading Entity nodes...")
    nodes = load_entity_nodes()

    node_lookup = {int(node["internal_id"]): str(node["entity_id"]) for node in nodes}

    logging.info("Projecting Entity edges for Leiden...")
    raw_edges = load_entity_edges(
        relationship_labels=args.edge_types,
        directional_labels=args.directional_edge_types,
        weight_field=args.weight_field,
    )

    edges = filter_edges_by_weight(raw_edges, min_weight=args.min_weight, weight_field=args.weight_field)
    mapped_edges = map_edge_node_ids(edges, node_lookup, weight_field=args.weight_field)

    node_rows = [
        {
            "node_id": node_lookup[int(node["internal_id"])],
            "name": node.get("name", ""),
            "labels": ";".join(sorted(set(node.get("labels", [])))),
        }
        for node in nodes
    ]

    output_dir: Path = args.output_dir
    write_csv(output_dir / "nodes.csv", ["node_id", "name", "labels"], node_rows)
    write_csv(
        output_dir / "edges.csv",
        ["source", "target", args.weight_field, "undirected"],
        mapped_edges,
    )

    metadata = {
        "edge_types": args.edge_types,
        "directional_edge_types": args.directional_edge_types or [],
        "min_weight": args.min_weight,
        "weight_field": args.weight_field,
        "resolution": args.resolution,
        "node_count": len(node_rows),
        "edge_count": len(mapped_edges),
        "undirected": True,
        "notes": "Adjust --edge-types or --resolution at runtime to rerun experiments without editing code.",
    }
    write_metadata(output_dir / "metadata.json", metadata)

    logging.info(
        "Projection complete: %s nodes, %s edges written to %s",
        len(node_rows),
        len(mapped_edges),
        output_dir,
    )


if __name__ == "__main__":
    main()
