"""First-class ontology for knowledge graphs.

A Knowledge Graph binds to an Ontology (``ontology_id`` / ``ontology_version``
on the registry row, migration 0069); the ontology defines the *valid* entity
types and relation types, and the canonical write path (``PgGraphSink``)
validates every compiled payload against it **before** touching durable
storage.

Resolution order in :func:`ontology_for_graph`:

1. no graph / no ``ontology_id``            → :data:`DEFAULT_ONTOLOGY`
2. a built-in id (``r4-default``, ``r4-science``) → that ontology
   (``ontology_version`` must match when given)
3. the ontology registry (``knowledge_ontologies``, migrations 0091 and
   0139) when a registry store is handed in; an id without a version
   resolves the latest live version
4. inline serialized ontology in the graph meta (``graph["ontology"]`` or
   ``graph["provenance"]["ontology"]``) with a matching id (legacy path)
5. otherwise → :class:`ValueError` (unknown ontology reference — fail loud)

ADR-116 D-4 adds optional fields (aliases, descriptions, relation domain and
range, class buckets, abstract classes, prefixes, source format). They are
absent from older serialized ontologies and default to empty.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server`` (foundation module).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .store import KnowledgeStore

# Canonical relation vocabulary produced by the knowledge compiler
# (pg_graph_sink extraction). Keep in sync with the extraction edge loops.
CANONICAL_RELATION_TYPES: frozenset[str] = frozenset(
    {"depends_on", "cites", "implements"}
)

# The 7 node_type values the knowledge_nodes CHECK constraint admits. Every
# ontology class maps onto one of them (ADR-116 D-6, ``class_buckets``).
NODE_TYPE_BUCKETS: frozenset[str] = KnowledgeStore.ALLOWED_NODE_TYPES

R4_SCIENCE_ID = "r4-science"


@dataclass(frozen=True)
class Ontology:
    """Versioned type vocabulary bound to a knowledge graph."""

    ontology_id: str
    version: str
    entity_types: frozenset[str]
    relation_types: frozenset[str]
    description: str = ""
    # ADR-049 Leg C: relations the closure inference legs MAY traverse.
    # Serialized absence defaults to depends_on only when the ontology
    # declares depends_on (ontology_from_dict).
    transitive_relations: frozenset[str] = frozenset({"depends_on"})
    # ADR-049 Leg D: (extracted-content key, canonical relation) pairs the
    # graph sink additionally EMITS edges for, on top of the base
    # dependencies/citations/algorithms loops. Empty means base extraction
    # only (serialized-absence tolerant, same as transitive_relations).
    extraction_fields: tuple[tuple[str, str], ...] = ()
    # G-P1-6 follow-up: (child, parent) pairs declaring the entity subtype
    # lattice; queries at a parent type include descendants transitively.
    # Members MUST reference declared entity_types — checked in from_dict /
    # declared-types validation, never silently widened.
    subtype_of: tuple[tuple[str, str], ...] = ()
    # ADR-116 D-4: optional vocabulary detail (absent in older payloads).
    # term (class or relation) -> alias strings
    aliases: tuple[tuple[str, tuple[str, ...]], ...] = ()
    # term -> description
    descriptions: tuple[tuple[str, str], ...] = ()
    # (relation, domain class, range class); "" means any class
    relation_signatures: tuple[tuple[str, str, str], ...] = ()
    # class -> one of NODE_TYPE_BUCKETS
    class_buckets: tuple[tuple[str, str], ...] = ()
    # classes allowed in queries but never assigned to an extracted node
    abstract_types: frozenset[str] = frozenset()
    # prefix -> IRI base
    prefixes: tuple[tuple[str, str], ...] = ()
    # "linkml" | "json" | "builtin" | ""
    source_format: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for graph provenance / API payloads."""
        out: dict[str, Any] = {
            "ontology_id": self.ontology_id,
            "version": self.version,
            "entity_types": sorted(self.entity_types),
            "relation_types": sorted(self.relation_types),
            "transitive_relations": sorted(self.transitive_relations),
            "description": self.description,
        }
        if self.extraction_fields:
            out["extraction_fields"] = dict(sorted(self.extraction_fields))
        if self.subtype_of:
            out["subtype_of"] = dict(sorted(self.subtype_of))
        out.update(self._optional_dict())
        return out

    def _optional_dict(self) -> dict[str, Any]:
        """The ADR-116 optional fields, each only when set."""
        optional: dict[str, Any] = {
            "aliases": {term: list(names) for term, names in sorted(self.aliases)},
            "descriptions": dict(sorted(self.descriptions)),
            "relation_signatures": [
                {"relation": rel, "domain": dom, "range": rng}
                for rel, dom, rng in sorted(self.relation_signatures)
            ],
            "class_buckets": dict(sorted(self.class_buckets)),
            "abstract_types": sorted(self.abstract_types),
            "prefixes": dict(sorted(self.prefixes)),
            "source_format": self.source_format,
        }
        return {key: value for key, value in optional.items() if value}


def normalize_term(raw: Any) -> str:
    """Canonical lowercase snake_case form of a class or relation name.

    ``MachineLearningModel`` → ``machine_learning_model``; ``data-set`` →
    ``data_set``; ``  Neural Network `` → ``neural_network``.
    """
    text = _CAMEL_UPPER_RUN.sub(r"\1_\2", str(raw).strip())
    text = _CAMEL_LOWER_UPPER.sub(r"\1_\2", text)
    return _NON_WORD.sub("_", text.lower()).strip("_")


_CAMEL_UPPER_RUN = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_LOWER_UPPER = re.compile(r"([a-z0-9])([A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _str_pairs(value: Any) -> tuple[tuple[str, str], ...]:
    """Sorted (key, value) string pairs from a mapping; () otherwise."""
    if not isinstance(value, Mapping):
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in value.items()))


def _alias_pairs(value: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Sorted (term, aliases) pairs from ``{term: [alias, ...]}``."""
    if not isinstance(value, Mapping):
        return ()
    return tuple(
        sorted(
            (str(term), tuple(str(a) for a in names)) for term, names in value.items()
        )
    )


def _signature_triples(value: Any) -> tuple[tuple[str, str, str], ...]:
    """Sorted (relation, domain, range) triples from the serialized list."""
    if not isinstance(value, list):
        return ()
    triples = []
    for item in value:
        if isinstance(item, Mapping):
            triples.append(
                (
                    str(item["relation"]),
                    str(item.get("domain") or ""),
                    str(item.get("range") or ""),
                )
            )
        else:
            rel, dom, rng = item
            triples.append((str(rel), str(dom), str(rng)))
    return tuple(sorted(triples))


def _require_declared(
    field_name: str, members: Iterable[str], declared: frozenset[str]
) -> None:
    unknown = sorted({m for m in members if m not in declared})
    if unknown:
        raise ValueError(
            f"{field_name} references undeclared entity types "
            f"{unknown} — declare them in entity_types first"
        )


def _check_signatures(
    triples: tuple[tuple[str, str, str], ...],
    entity_types: frozenset[str],
    relation_types: frozenset[str],
) -> None:
    unknown_relations = sorted(
        {rel for rel, _d, _r in triples if rel not in relation_types}
    )
    if unknown_relations:
        raise ValueError(
            f"relation_signatures name undeclared relation types {unknown_relations}"
        )
    classes = [c for _rel, dom, rng in triples for c in (dom, rng) if c]
    _require_declared("relation_signatures", classes, entity_types)


def _check_buckets(
    pairs: tuple[tuple[str, str], ...], entity_types: frozenset[str]
) -> None:
    _require_declared("class_buckets", (cls for cls, _b in pairs), entity_types)
    bad = sorted({bucket for _cls, bucket in pairs if bucket not in NODE_TYPE_BUCKETS})
    if bad:
        raise ValueError(
            f"class_buckets use unknown buckets {bad} (allowed: {sorted(NODE_TYPE_BUCKETS)})"
        )


def _transitive_from(value: Any, relation_types: frozenset[str]) -> frozenset[str]:
    """Serialized transitive set; absence means depends_on only when declared."""
    if value is None:
        return frozenset({"depends_on"}) & relation_types
    return frozenset(str(t) for t in value)


def ontology_from_dict(raw: Mapping[str, Any]) -> Ontology:
    """Build an Ontology from a serialized mapping (strict shape)."""
    try:
        return _ontology_from_mapping(raw)
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid serialized ontology: missing {exc}") from exc


def _ontology_from_mapping(raw: Mapping[str, Any]) -> Ontology:
    entity_types = frozenset(str(t) for t in raw["entity_types"])
    relation_types = frozenset(str(t) for t in raw["relation_types"])
    subtype_pairs = _str_pairs(raw.get("subtype_of"))
    _require_declared(
        "subtype_of", (m for pair in subtype_pairs for m in pair), entity_types
    )
    signatures = _signature_triples(raw.get("relation_signatures"))
    _check_signatures(signatures, entity_types, relation_types)
    buckets = _str_pairs(raw.get("class_buckets"))
    _check_buckets(buckets, entity_types)
    abstract = frozenset(str(t) for t in raw.get("abstract_types") or ())
    _require_declared("abstract_types", abstract, entity_types)
    return Ontology(
        ontology_id=str(raw["ontology_id"]),
        version=str(raw["version"]),
        entity_types=entity_types,
        relation_types=relation_types,
        description=str(raw.get("description", "")),
        transitive_relations=_transitive_from(
            raw.get("transitive_relations"), relation_types
        ),
        extraction_fields=_str_pairs(raw.get("extraction_fields")),
        subtype_of=subtype_pairs,
        aliases=_alias_pairs(raw.get("aliases")),
        descriptions=_str_pairs(raw.get("descriptions")),
        relation_signatures=signatures,
        class_buckets=buckets,
        abstract_types=abstract,
        prefixes=_str_pairs(raw.get("prefixes")),
        source_format=str(raw.get("source_format") or ""),
    )


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

BUILTIN_ONTOLOGY_IDS: tuple[str, ...] = (DEFAULT_ONTOLOGY.ontology_id, R4_SCIENCE_ID)


def r4_science_ontology() -> Ontology:
    """The built-in ``r4-science@1`` ontology (ADR-116 D-4)."""
    from .r4_science_schema import r4_science_ontology as _build

    return _build()


def builtin_ontology(ontology_id: Any, version: Any = None) -> Ontology | None:
    """The built-in ontology with this id, or None when the id is not built in.

    Raises :class:`ValueError` when the id is built in but the version is not.
    """
    if ontology_id == DEFAULT_ONTOLOGY.ontology_id:
        ontology = DEFAULT_ONTOLOGY
    elif ontology_id == R4_SCIENCE_ID:
        ontology = r4_science_ontology()
    else:
        return None
    if version not in (None, ontology.version):
        raise ValueError(
            f"ontology {ontology_id!r} has no version {version!r} "
            f"(current: {ontology.version!r})"
        )
    return ontology


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
    graph: Mapping[str, Any] | None,
    *,
    registry: Any | None = None,
) -> Ontology:
    """Resolve the ontology bound to a graph registry row (or meta mapping).

    Resolution order (ADR-049 Leg A, ADR-116 D-5):

    1. no graph / no ``ontology_id``        → :data:`DEFAULT_ONTOLOGY`
    2. a built-in id                        → that ontology
       (``ontology_version`` must match when given)
    3. **registry-backed record** (0091, when a registry store is handed
       in) of the graph's project or global, at the pinned (id, version) —
       retired versions included, since retiring blocks new bindings only —
       or at the latest live version when the graph pins none → that ontology
    4. inline serialized ontology in the graph meta (LEGACY path), strict
       consistency (id + pinned version must match the payload)
    5. otherwise → :class:`ValueError` (unknown ontology reference — fail
       loud, never smudged to the default).
    """
    if not graph or not graph.get("ontology_id"):
        return DEFAULT_ONTOLOGY
    ontology_id = graph.get("ontology_id")
    version = graph.get("ontology_version")
    builtin = builtin_ontology(ontology_id, version)
    if builtin is not None:
        return builtin
    record = _registry_record(registry, ontology_id, version, graph.get("project_id"))
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


def _registry_record(
    registry: Any | None, ontology_id: Any, version: Any, project_id: Any
) -> Any | None:
    """Registry record for (id, version); latest live version when None.

    Only a global record or one of the graph's own project counts, the
    project's own first (a graph with no project reads global records
    only): another project's record would put its class descriptions into
    this graph's extraction prompt. A pinned version that was retired after
    the graph was bound keeps resolving (retiring blocks new bindings only,
    ``pinned_ontology_version``).
    """
    if registry is None:
        return None
    scope = project_id or None
    if version is None:
        latest = getattr(registry, "latest_live_version", None)
        version = latest(str(ontology_id), scope) if latest is not None else None
        if version is None:
            return None
    record = _coordinate_record(registry, str(ontology_id), str(version), scope)
    if record is None or getattr(record, "project_id", None) in (None, scope):
        return record
    return None


def _coordinate_record(
    registry: Any, ontology_id: str, version: str, scope: str | None
) -> Any | None:
    """``resolve_coordinate`` (live or retired, scoped) when the registry has
    it; a bare ``live_coordinate`` otherwise (the result is scope-checked)."""
    resolve = getattr(registry, "resolve_coordinate", None)
    if resolve is not None:
        return resolve(ontology_id, version, scope)
    return registry.live_coordinate(ontology_id, version)


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
    "BUILTIN_ONTOLOGY_IDS",
    "CANONICAL_RELATION_TYPES",
    "DEFAULT_ONTOLOGY",
    "NODE_TYPE_BUCKETS",
    "R4_SCIENCE_ID",
    "Ontology",
    "OntologyViolationError",
    "builtin_ontology",
    "normalize_term",
    "ontology_for_graph",
    "ontology_from_dict",
    "r4_science_ontology",
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
