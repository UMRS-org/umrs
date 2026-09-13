"""First-class ontology for knowledge graphs (minimal loop).

A Knowledge Graph binds to an Ontology (``ontology_id`` / ``ontology_version``
on the registry row, migration 0069); the ontology defines the *valid* entity
types and relation types, and the canonical write path (``PgGraphSink``)
validates every compiled payload against it **before** touching durable
storage.

Resolution order in :func:`ontology_for_graph`:

1. no graph / no ``ontology_id``            → :data:`DEFAULT_ONTOLOGY`
2. ``ontology_id == "r4-default"``          → :data:`DEFAULT_ONTOLOGY`
   (``ontology_version`` must match)
3. inline serialized ontology in the graph meta (``graph["ontology"]`` or
   ``graph["provenance"]["ontology"]``) with a matching id → that ontology
4. otherwise → :class:`ValueError` (unknown ontology reference — fail loud)

Documented next architectural step (NOT implemented): a first-class
``knowledge_ontologies`` registry table. That is a new PG schema —
review-required per mutation-safety. Until then, custom ontologies travel
inline in graph provenance and this module is the single binding site;
query semantics (graph-visible types) derive from the same objects.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server`` (foundation module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .store import KnowledgeStore

# Canonical relation vocabulary produced by the knowledge compiler
# (pg_graph_sink extraction). Keep in sync with the extraction edge loops.
CANONICAL_RELATION_TYPES: frozenset[str] = frozenset(
    {"depends_on", "cites", "implements"}
)


@dataclass(frozen=True)
class Ontology:
    """Versioned type vocabulary bound to a knowledge graph."""

    ontology_id: str
    version: str
    entity_types: frozenset[str]
    relation_types: frozenset[str]
    description: str = ""
    # ADR-049 Leg C: relations the closure inference legs MAY traverse.
    # Additive in v1 — absent from serialized payloads means the declaring
    # ontology's own default (here: depends_on only, the dependency
    # relation whose closure scientists actually ask for).
    transitive_relations: frozenset[str] = frozenset({"depends_on"})
    # ADR-049 Leg D: (extracted-content key, canonical relation) pairs the
    # graph sink additionally EMITS edges for, on top of the base
    # dependencies/citations/algorithms loops. Empty means base extraction
    # only (serialized-absence tolerant, same as transitive_relations).
    extraction_fields: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for graph provenance / API payloads."""
        out = {
            "ontology_id": self.ontology_id,
            "version": self.version,
            "entity_types": sorted(self.entity_types),
            "relation_types": sorted(self.relation_types),
            "transitive_relations": sorted(self.transitive_relations),
            "description": self.description,
        }
        if self.extraction_fields:
            out["extraction_fields"] = {
                field: relation for field, relation in sorted(self.extraction_fields)
            }
        return out


def ontology_from_dict(raw: Mapping[str, Any]) -> Ontology:
    """Build an Ontology from a serialized mapping (strict shape)."""
    try:
        transitive = raw.get("transitive_relations")
        extraction = raw.get("extraction_fields") or {}
        return Ontology(
            ontology_id=str(raw["ontology_id"]),
            version=str(raw["version"]),
            entity_types=frozenset(str(t) for t in raw["entity_types"]),
            relation_types=frozenset(str(t) for t in raw["relation_types"]),
            description=str(raw.get("description", "")),
            transitive_relations=(
                frozenset(str(t) for t in transitive) if transitive is not None
                else frozenset({"depends_on"})
            ),
            extraction_fields=tuple(
                sorted((str(field), str(relation)) for field, relation in extraction.items())
            ) if isinstance(extraction, Mapping) else (),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid serialized ontology: missing {exc}") from exc


DEFAULT_ONTOLOGY = Ontology(
    ontology_id="r4-default",
    version="1",
    entity_types=KnowledgeStore.ALLOWED_NODE_TYPES,
    relation_types=CANONICAL_RELATION_TYPES,
    description=(
        "Default R4 ontology — the 7 knowledge node categories enforced by the "
        "knowledge_nodes DB CHECK constraint plus the canonical compiler relations."
    ),
)


class OntologyViolationError(ValueError):
    """Compiled payload references types outside the graph's ontology."""

    def __init__(
        self,
        ontology: Ontology,
        *,
        unknown_entity_types: list[str],
        unknown_relation_types: list[str],
    ) -> None:
        self.ontology = ontology
        self.unknown_entity_types = sorted(set(unknown_entity_types))
        self.unknown_relation_types = sorted(set(unknown_relation_types))
        parts = []
        if self.unknown_entity_types:
            parts.append(f"entity types {self.unknown_entity_types}")
        if self.unknown_relation_types:
            parts.append(f"relation types {self.unknown_relation_types}")
        super().__init__(
            f"payload violates ontology '{ontology.ontology_id}' v{ontology.version}: "
            + "; ".join(parts)
        )


def ontology_for_graph(
    graph: Optional[Mapping[str, Any]],
    *,
    registry: Any | None = None,
) -> Ontology:
    """Resolve the ontology bound to a graph registry row (or meta mapping).

    Resolution order (ADR-049 Leg A):

    1. no graph / no ``ontology_id``        → :data:`DEFAULT_ONTOLOGY`
    2. ``ontology_id == "r4-default"``      → :data:`DEFAULT_ONTOLOGY`
       (``ontology_version`` must match)
    3. **registry-backed live record** (0091, when a registry store is
       handed in) at the pinned (id, version) → that ontology
    4. inline serialized ontology in the graph meta (LEGACY path), strict
       consistency (id + pinned version must match the payload)
    5. otherwise → :class:`ValueError` (unknown ontology reference — fail
       loud, never smudged to the default).
    """
    if not graph:
        return DEFAULT_ONTOLOGY
    ontology_id = graph.get("ontology_id")
    version = graph.get("ontology_version")
    if not ontology_id:
        return DEFAULT_ONTOLOGY
    if ontology_id == DEFAULT_ONTOLOGY.ontology_id:
        if version not in (None, DEFAULT_ONTOLOGY.version):
            raise ValueError(
                f"ontology 'r4-default' has no version {version!r} "
                f"(current: {DEFAULT_ONTOLOGY.version!r})"
            )
        return DEFAULT_ONTOLOGY
    if registry is not None and version is not None:
        record = registry.live_coordinate(str(ontology_id), str(version))
        if record is not None:
            return record.ontology
    custom = _inline_ontology(graph, ontology_id, version)
    if custom is not None:
        return custom
    raise ValueError(
        f"unknown ontology {ontology_id!r} referenced by graph "
        f"{graph.get('graph_id', '?')!r} — register it in the ontology "
        "registry (ADR-049 Leg A), keep it as inline provenance (legacy), "
        "or use 'r4-default'"
    )


def _inline_ontology(
    graph: Mapping[str, Any],
    ontology_id: Any,
    version: Any,
) -> Ontology | None:
    """Resolve an inline-registered ontology with strict consistency checks."""
    provenance = graph.get("provenance")
    inline = graph.get("ontology") or (
        provenance.get("ontology") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(inline, Mapping):
        return None
    custom = ontology_from_dict(inline)
    if custom.ontology_id != ontology_id:
        raise ValueError(
            f"graph references ontology {ontology_id!r} but the inline "
            f"ontology is {custom.ontology_id!r}"
        )
    if version is not None and custom.version != str(version):
        raise ValueError(
            f"graph pins ontology {ontology_id!r} v{version!r} but the "
            f"inline ontology is v{custom.version!r}"
        )
    return custom


def validate_graph_payload(
    ontology: Ontology,
    *,
    node_types: Any = (),
    relation_types: Any = (),
) -> None:
    """Raise :class:`OntologyViolationError` for any type outside the ontology."""
    unknown_nodes = [t for t in node_types if t not in ontology.entity_types]
    unknown_relations = [t for t in relation_types if t not in ontology.relation_types]
    if unknown_nodes or unknown_relations:
        raise OntologyViolationError(
            ontology,
            unknown_entity_types=unknown_nodes,
            unknown_relation_types=unknown_relations,
        )


__all__ = [
    "CANONICAL_RELATION_TYPES",
    "DEFAULT_ONTOLOGY",
    "Ontology",
    "OntologyViolationError",
    "ontology_for_graph",
    "ontology_from_dict",
    "validate_graph_payload",
]


# ── sat_kg → canonical relation mapping (ADR-026 exit path) ─────────────
#
# Per ADR-026, migrating a consumer from the deprecated ``sat_kg_*`` tables
# must NOT adopt sat's 25-relation enum wholesale; it maps onto the
# canonical vocabulary (+ ontology extension). This table is the single
# source for that projection.
#
# Shape: sat RelationType enum name → (canonical_relation_type, inverse).
# ``inverse=True`` means the canonical store keeps the edge in the opposite
# direction (semantics-preserving). Rows whose canonical target collapses
# two sat relations must keep the original in ``content["sat_relation"]``.

SAT_RELATION_MAP: dict[str, tuple[str, bool]] = {
    # pass-through (canonical base vocabulary)
    "CITES": ("cites", False),
    "IMPLEMENTS": ("implements", False),
    "DEPENDS_ON": ("depends_on", False),
    # evidence — ontology extension group 1
    "SUPPORTS": ("supports", False),
    "CONTRADICTS": ("contradicts", False),
    "EVIDENCE_FOR": ("supports", False),  # collapse; keep sat_relation
    "EVIDENCE_AGAINST": ("contradicts", False),  # collapse; keep sat_relation
    "REFUTES": ("refutes", False),
    # research flow — ontology extension group 2
    "VERIFIES": ("verifies", False),
    "ASSUMES": ("assumes", False),
    "ANSWERS": ("answers", False),
    "ANSWERED_BY": ("answers", True),
    # derivation — ontology extension group 3
    "PRODUCES": ("produces", False),
    "PRODUCED_BY": ("produces", True),
    "DERIVES_FROM": ("derives_from", False),
    # citation/docs — ontology extension group 4
    "REFERENCES": ("references", False),
    "REFERENCED_BY": ("references", True),
    "DESCRIBES": ("describes", False),
    "DESCRIBED_BY": ("describes", True),
    "IMPLEMENTED_BY": ("implements", True),
    # composition — ontology extension group 5
    "PART_OF": ("part_of", False),
    "CONTAINS": ("contains", False),
    "USES": ("uses", False),
    # generic fallback
    "RELATED_TO": ("related_to", False),
}

# Canonical types added by the sat migration (ADR-026 exit-path vocab).
# Consumers may bind graphs to an ontology whose relation_types include
# CANONICAL_RELATION_TYPES ∪ this set.
SAT_CANONICAL_RELATION_TYPES: frozenset[str] = frozenset(
    rel for _name, (rel, _inv) in SAT_RELATION_MAP.items()
)

# Collapse set: sat relations whose canonical target is shared with
# another sat relation and therefore must record `content["sat_relation"]`.
_SAT_COLLAPSE_SOURCES: frozenset[str] = frozenset({"EVIDENCE_FOR", "EVIDENCE_AGAINST"})


def canonical_relation(sat_relation_name: str) -> tuple[str, bool]:
    """Project a sat relation name onto its canonical (relation, inverse).

    Unknown names fall back to ``related_to`` (non-inverse), mirroring the
    correction controller's existing RELATED_TO fallback behavior.
    """
    return SAT_RELATION_MAP.get(sat_relation_name, ("related_to", False))


def sat_relation_preserves_original(sat_relation_name: str) -> bool:
    """True when the original sat name must be kept in content['sat_relation']."""
    return sat_relation_name in _SAT_COLLAPSE_SOURCES


def canonical_to_sat(relation_type: str, sat_relation: str | None) -> str:
    """Project canonical relation back to the sat enum name for the view."""
    if sat_relation:
        return sat_relation.upper()
    for name, (canonical, _inv) in SAT_RELATION_MAP.items():
        if canonical == relation_type:
            return name
    return "RELATED_TO"
