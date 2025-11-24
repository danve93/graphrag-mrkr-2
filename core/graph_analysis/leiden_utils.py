"""Utilities for running Leiden clustering over the entity graph.

The entity graph exposes two relationship types:
- ``SIMILAR_TO`` edges between entities created from embedding similarity. These edges
  carry ``similarity`` weights (or ``score`` when coming from chunk similarity projections).
- ``RELATED_TO`` edges created from extracted relationships. These edges expose
  ``strength`` values.

Leiden expects a single numeric weight property. The helpers below normalize the
available properties into a unified ``weight`` attribute for projections, and they
fall back to ``1.0`` when a weight is missing so the algorithm can still run at
resolution values provided by callers.
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
