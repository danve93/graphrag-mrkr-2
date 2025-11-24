#!/usr/bin/env python3
"""Run Leiden clustering on the entity graph and persist community labels."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config.settings import settings  # noqa: E402
from core.graph_clustering import (  # noqa: E402
    fetch_entity_projection,
    normalize_edge_weights,
    run_leiden_clustering,
    to_igraph,
    write_communities_to_neo4j,
)
from core.graph_db import graph_db  # noqa: E402


DEFAULT_WEIGHT_FIELDS: Sequence[str] = (
    "weight",
    "similarity",
    "score",
    "strength",
    "frequency",
    "count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project the Entity graph, run Leiden, and write community IDs back to Neo4j.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--relationship-types",
        nargs="+",
        default=list(settings.clustering_relationship_types),
        help="Relationship labels to include in the clustering projection.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=settings.clustering_resolution,
        help="Leiden resolution parameter",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        default=settings.clustering_min_edge_weight,
        help="Drop edges with normalized weight below this value",
    )
    parser.add_argument(
        "--weight-field",
        type=str,
        default="weight",
        help="Name for the normalized weight attribute",
    )
    parser.add_argument(
        "--weight-fields",
        nargs="+",
        default=list(DEFAULT_WEIGHT_FIELDS),
        help="Edge properties to consider when deriving unit-level weights",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=settings.clustering_level,
        help="Level indicator to attach alongside community IDs",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def filter_edges(edges_df: pd.DataFrame, weight_field: str, min_weight: float) -> pd.DataFrame:
    if edges_df.empty:
        return edges_df
    return edges_df[edges_df[weight_field] >= min_weight].reset_index(drop=True)


def main() -> None:
    if not settings.enable_clustering or not settings.enable_graph_clustering:
        print(
            "Graph clustering disabled via settings; set ENABLE_CLUSTERING and "
            "ENABLE_GRAPH_CLUSTERING to true to re-enable."
        )
        return

    args = parse_args()
    configure_logging()

    logging.info("Loading entity graph projection with relationships: %s", args.relationship_types)
    nodes_df, edges_df = fetch_entity_projection(
        graph_db.driver, relationship_types=args.relationship_types
    )

    logging.info("Normalizing edge weights with fields: %s", args.weight_fields)
    edges_df = normalize_edge_weights(
        edges_df, preferred_fields=args.weight_fields, weight_field=args.weight_field
    )
    edges_df = filter_edges(edges_df, weight_field=args.weight_field, min_weight=args.min_edge_weight)

    logging.info("Building igraph projection (%s nodes, %s edges)", len(nodes_df), len(edges_df))
    graph = to_igraph(nodes_df, edges_df, weight_field=args.weight_field)

    if graph.vcount() == 0:
        logging.warning("No Entity nodes available for clustering; exiting early.")
        return
    if graph.ecount() == 0:
        logging.warning("Graph has no edges after filtering; skipping clustering.")
        return

    logging.info("Running Leiden clustering (resolution=%s)...", args.resolution)
    membership, modularity = run_leiden_clustering(
        graph, resolution=args.resolution, weight_field=args.weight_field
    )
    logging.info("Leiden finished: %s communities, modularity=%.4f", len(set(membership.values())), modularity)

    updated = write_communities_to_neo4j(graph_db.driver, membership, level=args.level)
    logging.info("Community labels persisted to Neo4j (updated=%s)", updated)


if __name__ == "__main__":
    main()
