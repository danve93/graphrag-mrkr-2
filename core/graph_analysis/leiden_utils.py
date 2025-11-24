"""Utilities for running Leiden clustering over the entity graph.

Relationship creation patterns:
* ``SIMILAR_TO`` edges are inserted with ``MERGE (a)-[r:SIMILAR_TO]-(b)`` in
  ``core.graph_db`` for both chunk-level and entity-level similarities. These are
  undirected and already safe for undirected community detection.
* ``RELATED_TO`` edges between entities follow the same undirected merge pattern
  and expose a ``strength`` property.
* Directional edges (for example ``CONTAINS_ENTITY`` from chunks to entities) are
  intentionally excluded from entity clustering because they are not symmetric and
  would need to be mirrored before Leiden can consume them.

Leiden expects a single numeric weight property. The helpers below normalize the
available properties into a unified ``weight`` attribute for projections, and they
fall back to ``1.0`` when a weight is missing so the algorithm can still run at
resolution values provided by callers. Default projections are undirected and
weighted; directional edges should either be dropped or symmetrized before use.
"""

from typing import Any, Iterable, Mapping, MutableMapping, Sequence

DEFAULT_WEIGHT: float = 1.0
ENTITY_EDGE_LABELS: Sequence[str] = ("SIMILAR_TO", "RELATED_TO")
EDGE_WEIGHT_FIELDS: Mapping[str, Sequence[str]] = {
    "SIMILAR_TO": ("similarity", "score"),
    "RELATED_TO": ("strength",),
}


def get_entity_edge_weight_fields() -> Mapping[str, Sequence[str]]:
    """Return available weight-like properties for entity relationships.

    This documents the numeric properties exposed by entity edges so callers can
    decide whether to precompute a unified ``weight`` field or filter down to a
    specific relationship label before projection.
    """

    return EDGE_WEIGHT_FIELDS


def resolve_weight(
    properties: Mapping[str, Any],
    relationship_type: str | None = None,
    preferred_fields: Iterable[str] | None = None,
    default: float = DEFAULT_WEIGHT,
) -> float:
    """Pick a numeric weight from a set of known property names.

    The search order honors custom ``preferred_fields`` first, then uses defaults for
    the relationship type (``similarity``/``score`` for ``SIMILAR_TO`` edges and
    ``strength`` for ``RELATED_TO`` edges). Missing or non-numeric values fall back to
    ``default`` so Leiden can still run using an unweighted interpretation.
    """

    search_order: list[str] = []

    if preferred_fields:
        search_order.extend(preferred_fields)

    if relationship_type and relationship_type in EDGE_WEIGHT_FIELDS:
        search_order.extend(EDGE_WEIGHT_FIELDS[relationship_type])

    for field in search_order:
        value = properties.get(field)
        if isinstance(value, (int, float)):
            return float(value)

    for fallback_field in ("weight", "similarity", "score", "strength"):
        value = properties.get(fallback_field)
        if isinstance(value, (int, float)):
            return float(value)

    return default


def normalize_weight_property(
    properties: Mapping[str, Any],
    relationship_type: str | None = None,
    preferred_fields: Iterable[str] | None = None,
    target_field: str = "weight",
    default: float = DEFAULT_WEIGHT,
) -> MutableMapping[str, Any]:
    """Return a copy of ``properties`` with a normalized weight field.

    This is useful when building Cypher projections for Leiden: it ensures every edge
    exposes the ``target_field`` expected by the algorithm, defaulting to ``1.0`` when
    a suitable weight is absent.
    """

    normalized: MutableMapping[str, Any] = dict(properties)
    normalized[target_field] = resolve_weight(
        properties,
        relationship_type=relationship_type,
        preferred_fields=preferred_fields,
        default=default,
    )
    return normalized


def build_leiden_parameters(
    weight_property: str = "weight", resolution: float = 1.0
) -> Mapping[str, float | str]:
    """Constructs the parameter dict expected by GDS Leiden calls.

    ``weight_property`` should match the key produced by :func:`normalize_weight_property`.
    ``resolution`` controls cluster granularity; callers can raise it for more
    communities or lower it for fewer. If edges lack explicit weights, the helper's
    default handling ensures the value ``1.0`` is supplied so the algorithm treats the
    graph as unweighted rather than failing.
    """

    return {"weightProperty": weight_property, "resolution": resolution}


def _coalesce_weight(relationship_alias: str = "r", default: float = DEFAULT_WEIGHT) -> str:
    """Return a Cypher expression that picks a numeric weight for an edge."""

    fields: Sequence[str] = ("weight", "similarity", "score", "strength")
    field_chain = ", ".join(f"{relationship_alias}.{field}" for field in fields)
    return f"coalesce({field_chain}, {default})"


def build_entity_leiden_projection_cypher(
    weight_field: str = "weight",
    relationship_labels: Sequence[str] | None = None,
    directional_labels: Sequence[str] | None = None,
    symmetrize_directional: bool = True,
) -> str:
    """Build a Cypher projection that returns an undirected edge list for Leiden.

    The default projection includes only entity-to-entity relationships created by
    ingestion (``SIMILAR_TO`` and ``RELATED_TO``), which are already stored as
    undirected edges. Directional relationships are ignored unless explicitly
    provided via ``directional_labels``; when supplied, they are mirrored so Leiden
    can still operate in undirected weighted mode.

    Parameters
    ----------
    weight_field:
        Name for the normalized weight property emitted in the projection.
    relationship_labels:
        Entity relationship types to include. Defaults to the standard undirected
        labels: ``SIMILAR_TO`` and ``RELATED_TO``.
    directional_labels:
        Optional directional relationship types to mirror into undirected pairs.
        Use this only when absolutely necessary; most Leiden runs should drop
        directional edges instead of mixing them with undirected entity links.
    symmetrize_directional:
        When ``True`` (default), directional edges are doubled (forward + reverse)
        to make them safe for undirected community detection. When ``False``,
        directional edges are excluded from the projection.
    """

    labels = relationship_labels or ENTITY_EDGE_LABELS
    label_union = "|".join(labels)
    weight_expr = _coalesce_weight(default=DEFAULT_WEIGHT)

    edge_projection = (
        f"MATCH (e1:Entity)-[r:{label_union}]-(e2:Entity)\n"
        f"WITH id(e1) AS source, id(e2) AS target, {weight_expr} AS {weight_field}\n"
        f"WITH CASE WHEN source < target THEN source ELSE target END AS source,\n"
        f"     CASE WHEN source < target THEN target ELSE source END AS target,\n"
        f"     {weight_field}\n"
        f"RETURN DISTINCT source, target, {weight_field}"
    ).strip()

    if directional_labels and symmetrize_directional:
        directional_union = "|".join(directional_labels)
        mirrored = (
            f"MATCH (src)-[r:{directional_union}]->(dst)\n"
            f"UNWIND [[id(src), id(dst)], [id(dst), id(src)]] AS pair\n"
            f"WITH pair[0] AS source, pair[1] AS target, {weight_expr} AS {weight_field}\n"
            f"RETURN DISTINCT source, target, {weight_field}"
        ).strip()

        edge_projection = "\nUNION\n".join([edge_projection, mirrored])

    return edge_projection
