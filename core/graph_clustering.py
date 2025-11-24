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
    RETURN id(e) AS internal_id, e.id AS entity_id, coalesce(e.name, "") AS name
    """

    edge_query = f"""
    MATCH (e1:Entity)-[r:{label_union}]-(e2:Entity)
    WITH e1, e2, r, id(e1) AS source_internal, id(e2) AS target_internal
    WITH CASE WHEN source_internal < target_internal THEN e1 ELSE e2 END AS source_node,
         CASE WHEN source_internal < target_internal THEN e2 ELSE e1 END AS target_node,
         type(r) AS relationship_type, properties(r) AS properties
    RETURN DISTINCT
        id(source_node) AS source_internal,
        id(target_node) AS target_internal,
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
    """Convert node and edge frames into an igraph.Graph."""

    graph = ig.Graph()
    graph.add_vertices(nodes_df["entity_id"].tolist())
    graph.vs["entity_id"] = nodes_df["entity_id"].tolist()
    graph.vs["name"] = nodes_df["name"].tolist()

    if not edges_df.empty:
        graph.add_edges(list(zip(edges_df["source_id"], edges_df["target_id"])))
        graph.es[weight_field] = edges_df[weight_field].astype(float).tolist()
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
        resolution_parameter=resolution,
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
