"""Utilities for exporting the Neo4j entity graph and running community detection."""

import logging
from typing import Iterable, Mapping, Sequence, Tuple

import igraph as ig
import pandas as pd
from neo4j import Driver

from config.settings import settings
from core.graph_analysis.leiden_utils import ENTITY_EDGE_LABELS, resolve_weight

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_FIELDS: Tuple[str, ...] = (
    "weight",
    "similarity",
    "score",
    "strength",
    "frequency",
    "count",
)


def fetch_entity_projection(
    driver: Driver,
    relationship_types: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return nodes and edges for the entity graph as pandas DataFrames."""

    if driver is None:
        raise ValueError("Neo4j driver is not initialized")

    edge_labels = relationship_types or list(ENTITY_EDGE_LABELS)
    if not edge_labels:
        raise ValueError("relationship_types must include at least one label")
    label_union = "|".join(edge_labels)

    node_query = """
    MATCH (e:Entity)
    RETURN elementId(e) AS internal_id, e.id AS entity_id, coalesce(e.name, "") AS name
    """

    edge_query = f"""
    MATCH (e1:Entity)-[r:{label_union}]-(e2:Entity)
    WITH e1, e2, r, elementId(e1) AS source_internal, elementId(e2) AS target_internal
    WITH CASE WHEN source_internal < target_internal THEN e1 ELSE e2 END AS source_node,
         CASE WHEN source_internal < target_internal THEN e2 ELSE e1 END AS target_node,
         type(r) AS relationship_type, properties(r) AS properties
    RETURN DISTINCT
        elementId(source_node) AS source_internal,
        elementId(target_node) AS target_internal,
        source_node.id AS source_id,
        target_node.id AS target_id,
        relationship_type,
        properties
    """

    with driver.session() as session:  # type: ignore
        node_records = session.run(node_query)
        nodes = [record.data() for record in node_records]

        edge_records = session.run(edge_query)
        edges = [record.data() for record in edge_records]

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)

    if not nodes_df.empty and nodes_df["entity_id"].isnull().any():
        raise ValueError("Entity nodes must include an `id` property")

    if not edges_df.empty and edges_df[["source_id", "target_id"]].isnull().any().any():
        raise ValueError("Edge projection missing entity identifiers; check Entity nodes")

    return nodes_df, edges_df


def normalize_edge_weights(
    edges_df: pd.DataFrame,
    preferred_fields: Iterable[str] | None = None,
    weight_field: str = "weight",
) -> pd.DataFrame:
    """Normalize edge weights into a single numeric column."""

    if edges_df.empty:
        edges_df[weight_field] = []
        return edges_df

    fields: list[str] = list(preferred_fields) if preferred_fields else list(DEFAULT_WEIGHT_FIELDS)

    def _resolve(row: Mapping[str, object]) -> float:
        properties = row.get("properties") or {}
        relationship_type = row.get("relationship_type")
        return resolve_weight(
            properties if isinstance(properties, Mapping) else {},
            relationship_type=relationship_type if isinstance(relationship_type, str) else None,
            preferred_fields=fields,
            default=settings.default_edge_weight,
        )

    normalized = edges_df.copy()
    normalized[weight_field] = normalized.apply(_resolve, axis=1)
    return normalized


def to_igraph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    weight_field: str = "weight",
) -> ig.Graph:
    """Convert node and edge frames into an igraph.Graph.
    
    Creates an igraph where vertices are identified by their index positions.
    Entity IDs are stored as vertex attributes for reference.
    """

    graph = ig.Graph()
    node_ids = nodes_df["entity_id"].tolist()
    
    # Add vertices
    graph.add_vertices(len(node_ids))
    graph.vs["entity_id"] = node_ids
    graph.vs["name"] = nodes_df["name"].tolist()

    if not edges_df.empty:
        # Create a mapping from entity_id to vertex index
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        # Convert edges from entity IDs to vertex indices
        edges_as_indices = []
        node_set = set(node_ids)
        
        for _, row in edges_df.iterrows():
            src_id = row["source_id"]
            tgt_id = row["target_id"]
            
            # Only add edges where both source and target exist
            if src_id in node_set and tgt_id in node_set:
                src_idx = node_to_idx[src_id]
                tgt_idx = node_to_idx[tgt_id]
                edges_as_indices.append((src_idx, tgt_idx))
        
        if edges_as_indices:
            graph.add_edges(edges_as_indices)
            # Extract weights for valid edges
            valid_edges = edges_df[
                (edges_df["source_id"].isin(node_set)) & (edges_df["target_id"].isin(node_set))
            ]
            graph.es[weight_field] = valid_edges[weight_field].astype(float).tolist()
        else:
            graph.es[weight_field] = []
    else:
        graph.es[weight_field] = []

    return graph


def run_leiden_clustering(
    graph: ig.Graph,
    resolution: float = 1.0,
    weight_field: str = "weight",
) -> tuple[dict[str, int], float]:
    """Run Leiden on the provided graph and return membership and modularity."""

    partition = graph.community_leiden(
        objective_function="modularity",
        weights=graph.es[weight_field] if weight_field in graph.es.attributes() else None,
        resolution=resolution,
    )

    membership = {
        graph.vs[idx]["entity_id"]: int(community)
        for idx, community in enumerate(partition.membership)
    }
    modularity = float(partition.modularity)
    return membership, modularity


def write_communities_to_neo4j(
    driver: Driver, membership: Mapping[str, int], level: int = 0
) -> int:
    """Persist community assignments to Neo4j Entity nodes."""

    if driver is None:
        raise ValueError("Neo4j driver is not initialized")

    rows = [
        {"entity_id": entity_id, "community_id": community_id, "level": level}
        for entity_id, community_id in membership.items()
    ]

    if not rows:
        logger.warning("No community assignments to write.")
        return 0

    query = """
    UNWIND $rows AS row
    MATCH (e:Entity {id: row.entity_id})
    SET e.community_id = row.community_id,
        e.level = row.level
    RETURN count(e) AS updated
    """

    with driver.session() as session:  # type: ignore
        result = session.run(query, rows=rows)
        record = result.single()
        updated = record["updated"] if record else 0

    logger.info("Updated %s Entity nodes with community assignments", updated)
    return int(updated)


def run_auto_clustering(
    driver: Driver,
    relationship_types: Sequence[str] | None = None,
    resolution: float | None = None,
    min_edge_weight: float | None = None,
    level: int = 0,
) -> dict[str, int | float]:
    """
    Run full clustering pipeline (fetch → normalize → cluster → persist).
    
    Args:
        driver: Neo4j driver instance
        relationship_types: Edge types to include (defaults to settings.clustering_relationship_types)
        resolution: Leiden resolution (defaults to settings.clustering_resolution)
        min_edge_weight: Drop edges below this weight (defaults to settings.clustering_min_edge_weight)
        level: Community level indicator
    
    Returns:
        Dict with status, communities_count, modularity, updated_nodes
    """
    if not settings.enable_clustering or not settings.enable_graph_clustering:
        logger.info("Graph clustering disabled in settings; skipping auto-clustering.")
        return {"status": "disabled", "communities_count": 0, "modularity": 0.0, "updated_nodes": 0}

    try:
        rel_types = relationship_types or list(settings.clustering_relationship_types)
        res = resolution if resolution is not None else settings.clustering_resolution
        min_weight = min_edge_weight if min_edge_weight is not None else settings.clustering_min_edge_weight

        logger.info("Auto-clustering: loading entity projection with relationships=%s", rel_types)
        nodes_df, edges_df = fetch_entity_projection(driver, relationship_types=rel_types)

        if nodes_df.empty:
            logger.info("No entities available for clustering.")
            return {"status": "no_entities", "communities_count": 0, "modularity": 0.0, "updated_nodes": 0}

        logger.debug("Normalizing edge weights (min_weight=%s)", min_weight)
        edges_df = normalize_edge_weights(edges_df, preferred_fields=DEFAULT_WEIGHT_FIELDS, weight_field="weight")
        
        if not edges_df.empty:
            edges_df = edges_df[edges_df["weight"] >= min_weight].reset_index(drop=True)

        logger.debug("Building igraph (%s nodes, %s edges)", len(nodes_df), len(edges_df))
        graph = to_igraph(nodes_df, edges_df, weight_field="weight")

        if graph.vcount() == 0:
            logger.warning("No nodes in graph projection; skipping clustering.")
            return {"status": "no_nodes", "communities_count": 0, "modularity": 0.0, "updated_nodes": 0}

        if graph.ecount() == 0:
            logger.warning("No edges after filtering; skipping clustering.")
            return {"status": "no_edges", "communities_count": 0, "modularity": 0.0, "updated_nodes": 0}

        logger.info("Running Leiden clustering (resolution=%s)...", res)
        membership, modularity = run_leiden_clustering(graph, resolution=res, weight_field="weight")
        communities_count = len(set(membership.values()))

        logger.info("Leiden finished: %s communities, modularity=%.4f", communities_count, modularity)
        updated = write_communities_to_neo4j(driver, membership, level=level)

        return {
            "status": "success",
            "communities_count": communities_count,
            "modularity": float(modularity),
            "updated_nodes": updated,
        }

    except Exception as e:
        logger.error("Auto-clustering failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e), "communities_count": 0, "modularity": 0.0, "updated_nodes": 0}
