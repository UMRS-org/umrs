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

    def to_dict(self) -> dict[str, Any]:
        """Serialize for graph provenance / API payloads."""
        return {
            "ontology_id": self.ontology_id,
            "version": self.version,
            "entity_types": sorted(self.entity_types),
            "relation_types": sorted(self.relation_types),
            "description": self.description,
        }


def ontology_from_dict(raw: Mapping[str, Any]) -> Ontology:
    """Build an Ontology from a serialized mapping (strict shape)."""
    try:
        return Ontology(
            ontology_id=str(raw["ontology_id"]),
            version=str(raw["version"]),
            entity_types=frozenset(str(t) for t in raw["entity_types"]),
            relation_types=frozenset(str(t) for t in raw["relation_types"]),
            description=str(raw.get("description", "")),
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


def ontology_for_graph(graph: Optional[Mapping[str, Any]]) -> Ontology:
    """Resolve the ontology bound to a graph registry row (or meta mapping).

    See module docstring for the resolution order. Raises ``ValueError`` for
    version mismatches and unknown ontology references (fail loud).
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
    provenance = graph.get("provenance")
    inline = graph.get("ontology") or (
        provenance.get("ontology") if isinstance(provenance, Mapping) else None
    )
    if isinstance(inline, Mapping):
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
    raise ValueError(
        f"unknown ontology {ontology_id!r} referenced by graph "
        f"{graph.get('graph_id', '?')!r} — register it inline in graph provenance "
        "or use 'r4-default'"
    )


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
