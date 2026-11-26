"""Shared types of the typed knowledge spine (ADR-081 W2a, slice KN-S7).

This module is the seam between the pure pieces of the term registry:

- ``umrs.knowledge.ontologies`` parses the bundled YAML documents into
  :class:`OntologyDoc` / :class:`TermSpec`;
- ``umrs.knowledge.term_closure`` computes the reflexive subsumption closure
  and each ontology's admitted term set;
- ``umrs.knowledge.ontologies.seed_sql`` renders the seed migration;
- ``umrs.knowledge.term_registry`` reads a :class:`RegistrySnapshot` for one
  graph from Postgres;
- ``umrs.knowledge.typing`` (``prepare_row``) and
  ``umrs.knowledge.typing_relations`` (``prepare_edge``) apply the D-4 rules
  to one row against a :class:`TypingContext`.

Nothing here touches a database or the filesystem. Vocabulary constants
mirror the CHECK constraints of the ``knowledge_terms`` migration.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: ``knowledge_terms.kind``.
TERM_KINDS: tuple[str, ...] = ("class", "relation", "attribute")
#: Kinds that take part in the subsumption closure (attributes do not).
CLOSURE_KINDS: frozenset[str] = frozenset({"class", "relation"})
#: The seven ``knowledge_nodes.node_type`` facet values (0055 CHECK); a class
#: carries one of them as its bucket.
BUCKETS: frozenset[str] = frozenset(
    {"class", "function", "file", "module", "concept", "entity", "relationship"}
)
#: ``knowledge_terms.datatype`` (an attribute's value type).
DATATYPES: tuple[str, ...] = (
    "number",
    "integer",
    "string",
    "boolean",
    "datetime",
    "date",
    "interval",
    "json",
    "node",
)
#: ``knowledge_terms.origin``.
ORIGINS: tuple[str, ...] = ("standard", "imported", "minted")
#: ``knowledge_graphs.ontology_mode`` (ADR-081 D-5).
ONTOLOGY_MODES: tuple[str, ...] = ("warn", "enforce")
#: ``knowledge_write_batches.origin_actor``: the closed actor vocabulary of
#: ``umrs.knowledge.actors`` plus ``system`` for server-internal writers.
ORIGIN_ACTORS: tuple[str, ...] = ("user", "agent", "system")

#: Class a warn-mode write falls back to when its class is missing or not
#: admitted. Its node_type is the writer's facet, not a bucket of its own.
UNCLASSIFIED = "r4:Unclassified"
#: Relation a warn-mode write falls back to for an unknown relation.
RELATED_TO = "r4:relatedTo"
#: The core profile a graph without an ontology resolves to (ADR-081 D-3).
CORE_REF = "r4-core@2"
#: The standard layer the core profile imports.
STD_REF = "std@1"
#: ``knowledge_ontologies.record_id`` prefix of a bundled (package data) record.
BUNDLED_RECORD_PREFIX = "bundled:"

#: Content key that receives attrs a warn-mode write could not keep.
UNDECLARED_KEY = "undeclared"
#: Content key that keeps a requested class that was not admitted.
CLASS_RAW_KEY = "class_raw"
#: Content key that keeps the writer's raw relation when it fell back.
REL_RAW_KEY = "rel_raw"


def bundled_record_id(ref: str) -> str:
    """``knowledge_ontologies.record_id`` of the bundled ontology ``ref``."""
    return f"{BUNDLED_RECORD_PREFIX}{ref}"


class OntologyViolationError(ValueError):
    """An enforce-mode batch broke a D-4 rule; nothing of it was written."""

    def __init__(self, violations: Iterable[Violation]) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations)
        codes = ", ".join(sorted({v.code for v in self.violations}))
        super().__init__(f"ontology violation ({len(self.violations)}): {codes}")


@dataclass(frozen=True)
class TermSpec:
    """One ``knowledge_terms`` row (ADR-081 D-2). Terms are immutable."""

    term_id: str  # CURIE, e.g. "r4:Document"
    iri: str
    kind: str  # one of TERM_KINDS
    label: str
    parents: tuple[str, ...] = ()
    bucket: str | None = None  # classes only; one of BUCKETS
    domain_classes: tuple[str, ...] = ()  # relation/attribute subject classes
    range_classes: tuple[str, ...] = ()  # relation object classes
    datatype: str | None = None  # attributes only; one of DATATYPES
    unit: str | None = None
    transitive: bool = False
    functional: bool = False
    inverse_of: str | None = None
    aliases: tuple[str, ...] = ()  # legacy slugs, e.g. "part_of"
    synonyms: tuple[str, ...] = ()
    definition: str = ""
    required_on: tuple[str, ...] = ()  # attributes: classes that require it
    origin: str = "minted"  # one of ORIGINS
    verified: bool = True
    source: str = ""
    status: str = "active"  # active | deprecated
    replaced_by: str | None = None
    mappings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyDoc:
    """One parsed ontology document (bundled package data or registered)."""

    ontology_id: str
    version: str
    iri_base: str
    prefixes: Mapping[str, str]
    imports: tuple[str, ...] = ()  # refs "<id>@<version>"
    description: str = ""
    terms: tuple[TermSpec, ...] = ()
    profile: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ref(self) -> str:
        """``<ontology_id>@<version>``."""
        return f"{self.ontology_id}@{self.version}"


@dataclass(frozen=True)
class RegistrySnapshot:
    """What ``term_registry.fetch_snapshot`` read for one graph.

    ``terms`` is the admitted set of the graph's ontology record (imports
    included); ``closure`` holds ``(ancestor, descendant, depth)`` rows for
    every admitted class and relation, reflexive (depth 0 = self).
    """

    ontology_ref: str
    mode: str  # one of ONTOLOGY_MODES
    terms: tuple[TermSpec, ...]
    closure: tuple[tuple[str, str, int], ...]
    record_id: str = ""


@dataclass(frozen=True)
class Gap:
    """One vocabulary miss for ``knowledge_vocab_gaps`` (kind, raw value)."""

    kind: str  # class | relation | attribute
    value: str
    sample_id: str = ""


@dataclass(frozen=True)
class Violation:
    """One D-4 rule a row broke. ``code`` values:

    ``class_missing``, ``class_unknown``, ``bucket_invalid``,
    ``attr_undeclared``, ``attr_domain``, ``attr_datatype``,
    ``relation_unknown``, ``relation_domain``, ``relation_range``.
    """

    code: str
    subject_id: str
    detail: str = ""

    def as_json(self) -> dict[str, str]:
        """The ``knowledge_write_batches.violations`` element."""
        return {"code": self.code, "subject_id": self.subject_id, "detail": self.detail}


@dataclass(frozen=True)
class NodeDraft:
    """A node as a writer hands it to the typing rules."""

    node_id: str
    node_type: str  # writer's facet; used as-is only for UNCLASSIFIED
    node_class: str | None = None  # CURIE or alias; None = untyped writer
    attrs: Mapping[str, Any] = field(default_factory=dict)
    content: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedNode:
    """``prepare_row``'s result: the columns to write plus what it found."""

    node_class: str
    node_type: str
    attrs: Mapping[str, Any]
    content: Mapping[str, Any]
    gaps: tuple[Gap, ...] = ()
    violations: tuple[Violation, ...] = ()
    missing_required: tuple[str, ...] = ()


@dataclass(frozen=True)
class EdgeDraft:
    """An edge as a writer hands it to the typing rules."""

    edge_id: str
    relation_type: str  # writer's raw relation (kept in the legacy column)
    relation_class: str | None = None  # CURIE or alias; overrides relation_type
    content: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedEdge:
    """``prepare_edge``'s result."""

    relation_class: str
    content: Mapping[str, Any]
    gaps: tuple[Gap, ...] = ()
    violations: tuple[Violation, ...] = ()


def _alias_index(terms: Iterable[TermSpec]) -> dict[tuple[str, str], str]:
    """(kind, casefolded alias) -> term_id; the first term wins. Labels are
    not indexed: two vocabularies share labels ("Person", "Concept")."""
    index: dict[tuple[str, str], str] = {}
    for term in terms:
        for name in term.aliases:
            index.setdefault((term.kind, name.casefold()), term.term_id)
    return index


def _ancestor_map(
    closure: Iterable[tuple[str, str, int]],
) -> dict[str, frozenset[str]]:
    found: dict[str, set[str]] = {}
    for ancestor, descendant, _depth in closure:
        found.setdefault(descendant, {descendant}).add(ancestor)
    return {key: frozenset(value) for key, value in found.items()}


@dataclass(frozen=True)
class TypingContext:
    """The admitted vocabulary of one graph, resolved once (ADR-081 D-4).

    Build it with :meth:`from_snapshot`. Lookups are exact on the CURIE and
    case-insensitive on aliases, within one term kind.
    """

    ontology_ref: str
    mode: str
    terms: Mapping[str, TermSpec]
    ancestors: Mapping[str, frozenset[str]]
    aliases: Mapping[tuple[str, str], str]

    @classmethod
    def from_snapshot(cls, snapshot: RegistrySnapshot) -> TypingContext:
        """Index a snapshot. Raises ValueError on an unknown mode."""
        if snapshot.mode not in ONTOLOGY_MODES:
            raise ValueError(f"unknown ontology mode {snapshot.mode!r}")
        terms = {t.term_id: t for t in snapshot.terms}
        return cls(
            ontology_ref=snapshot.ontology_ref,
            mode=snapshot.mode,
            terms=terms,
            ancestors=_ancestor_map(snapshot.closure),
            aliases=_alias_index(snapshot.terms),
        )

    @property
    def enforcing(self) -> bool:
        """True when a violation must fail the whole batch."""
        return self.mode == "enforce"

    def resolve(self, raw: str | None, kind: str) -> TermSpec | None:
        """The admitted active term of ``kind`` named by ``raw`` (CURIE, or an
        alias compared case-insensitively), else None."""
        if not raw:
            return None
        term = self.terms.get(raw)
        if term is None:
            term_id = self.aliases.get((kind, raw.casefold()))
            term = self.terms.get(term_id) if term_id else None
        if term is None or term.kind != kind or term.status != "active":
            return None
        return term

    def is_a(self, term_id: str, ancestor: str) -> bool:
        """``term_id`` is ``ancestor`` or under it in the closure."""
        return ancestor in self.ancestors.get(term_id, frozenset({term_id}))

    def under_any(self, term_id: str, ancestors: Iterable[str]) -> bool:
        """``term_id`` sits under at least one of ``ancestors``."""
        return any(self.is_a(term_id, a) for a in ancestors)

    def required_attributes(self, class_id: str) -> tuple[str, ...]:
        """Attribute CURIEs whose ``required_on`` admits ``class_id``, sorted."""
        return tuple(
            sorted(
                t.term_id
                for t in self.terms.values()
                if t.kind == "attribute" and self.under_any(class_id, t.required_on)
            )
        )
