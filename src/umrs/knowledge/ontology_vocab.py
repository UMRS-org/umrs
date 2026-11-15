"""Extraction vocabulary taken from a graph's ontology (ADR-116 D-3, D-6).

The understand stage lists the vocabulary's classes and relations in the
extraction prompt, maps the model's replies back to class and relation
names (name and alias matching, case, space, hyphen and plural tolerant),
and picks the 7-value ``node_type`` bucket for each class. GLiNER pre-tagging
takes its labels from the same vocabulary.

:func:`resolve_graph_vocabulary` is the one call the ingest path makes: it
never raises; when anything fails it returns the built-in ``r4-science``
vocabulary and a note naming why.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server`` (foundation module).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from .ontology import (
    DEFAULT_ONTOLOGY,
    NODE_TYPE_BUCKETS,
    R4_SCIENCE_ID,
    Ontology,
    normalize_term,
    ontology_for_graph,
    r4_science_ontology,
)

logger = logging.getLogger(__name__)

UNCLASSIFIED = "unclassified"
DEFAULT_BUCKET = "entity"
MAX_GLINER_LABELS = 25
# The vocabulary goes into every window's system prompt: a term's detail and
# the whole listing are capped so a large ontology cannot crowd out the text.
PROMPT_DETAIL_MAX = 200
PROMPT_ALIASES_MAX = 8
PROMPT_BLOCK_MAX = 24_000
_AUXILIARY_PREFIXES = ("is_", "are_", "was_", "were_", "has_been_", "have_been_")


def _plural_candidates(name: str) -> Iterator[str]:
    """``name`` itself, then simple singular forms of its last word."""
    yield name
    if name.endswith("ies") and len(name) > 4:
        yield name[:-3] + "y"
    if name.endswith("es") and len(name) > 3:
        yield name[:-2]
    if name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        yield name[:-1]


def _relation_candidates(name: str) -> Iterator[str]:
    """``name``, without a leading auxiliary verb, and in third person."""
    yield name
    for prefix in _AUXILIARY_PREFIXES:
        if name.startswith(prefix) and len(name) > len(prefix):
            yield name[len(prefix) :]
    yield name + "s"
    if name.endswith("s") and not name.endswith("ss"):
        yield name[:-1]


def _lookup_table(
    names: list[str], aliases: dict[str, tuple[str, ...]]
) -> dict[str, str]:
    """normalized name or alias -> canonical term; names win over aliases."""
    table: dict[str, str] = {}
    for name in names:
        for alias in aliases.get(name, ()):
            table.setdefault(normalize_term(alias), name)
    table.update({normalize_term(name): name for name in names})
    table.pop("", None)
    return table


class ExtractionVocabulary:
    """Classes and relations one ontology offers to extraction."""

    def __init__(self, ontology: Ontology) -> None:
        self.ontology = ontology
        self.ontology_id = ontology.ontology_id
        self.version = ontology.version
        aliases = dict(ontology.aliases)
        self._descriptions = dict(ontology.descriptions)
        self._aliases = aliases
        self._buckets = dict(ontology.class_buckets)
        self._classes = sorted(ontology.entity_types - ontology.abstract_types)
        self._relations = sorted(ontology.relation_types)
        self._class_table = _lookup_table(self._classes, aliases)
        self._relation_table = _lookup_table(self._relations, aliases)

    def class_names(self) -> list[str]:
        """Classes extraction may assign (abstract classes left out)."""
        return list(self._classes)

    def relation_names(self) -> list[str]:
        return list(self._relations)

    def match_class(self, raw: Any) -> str | None:
        """Canonical class for a raw type string, or None when nothing matches."""
        name = normalize_term(raw or "")
        for candidate in _plural_candidates(name):
            if candidate in self._class_table:
                return self._class_table[candidate]
        return None

    def match_relation(self, raw: Any) -> str | None:
        """Canonical relation for a raw relation phrase, or None."""
        name = normalize_term(raw or "")
        if not name:
            return None
        for candidate in _relation_candidates(name):
            if candidate in self._relation_table:
                return self._relation_table[candidate]
        return None

    def bucket_for(self, cls: str) -> str:
        """The knowledge_nodes ``node_type`` value for a class."""
        if cls in self._buckets:
            return self._buckets[cls]
        if cls == UNCLASSIFIED:
            return "concept"
        if cls in NODE_TYPE_BUCKETS:
            return cls
        return DEFAULT_BUCKET

    def gliner_labels(self) -> list[str]:
        """Readable class labels for zero-shot NER (at most 25)."""
        return [name.replace("_", " ") for name in self._classes[:MAX_GLINER_LABELS]]

    def prompt_block(self) -> str:
        """Plain-text class and relation listing for the extraction prompt,
        at most ``PROMPT_BLOCK_MAX`` characters (see :attr:`prompt_truncated`)."""
        lines = self._prompt_lines()
        kept: list[str] = []
        size = 0
        for index, line in enumerate(lines):
            size += len(line) + 1
            if size > PROMPT_BLOCK_MAX:
                kept.append(
                    f"({len(lines) - index} more lines of the vocabulary not listed)"
                )
                break
            kept.append(line)
        return "\n".join(kept)

    @property
    def prompt_truncated(self) -> bool:
        """Whether :meth:`prompt_block` leaves terms out."""
        return sum(len(line) + 1 for line in self._prompt_lines()) > PROMPT_BLOCK_MAX

    def _prompt_lines(self) -> list[str]:
        lines = ['Entity types. Set "type" to exactly one of these names:']
        lines.extend(self._class_line(name) for name in self._classes)
        if self._relations:
            lines.append(
                'Relation types. Set "relation" to one of these names '
                "(source type -> target type):"
            )
            lines.extend(self._relation_line(name) for name in self._relations)
        return lines

    def _class_line(self, name: str) -> str:
        return f"- {name}{self._detail(name)}"

    def _relation_line(self, name: str) -> str:
        return f"- {name} ({self._signature_text(name)}){self._detail(name)}"

    def _detail(self, name: str) -> str:
        text = ""
        if self._descriptions.get(name):
            text += f": {self._descriptions[name][:PROMPT_DETAIL_MAX]}"
        if self._aliases.get(name):
            shown = self._aliases[name][:PROMPT_ALIASES_MAX]
            text += f" Also called: {', '.join(shown)}."
        return text

    def _signature_text(self, relation: str) -> str:
        pairs = sorted(
            (self._shown_class(dom), self._shown_class(rng))
            for rel, dom, rng in self.ontology.relation_signatures
            if rel == relation
        )
        if not pairs:
            return "any -> any"
        return "; ".join(f"{dom} -> {rng}" for dom, rng in dict.fromkeys(pairs))

    def _shown_class(self, cls: str) -> str:
        """A class as the prompt names it: an abstract root reads as ``any``."""
        if not cls:
            return "any"
        if cls in self.ontology.abstract_types and cls not in dict(
            self.ontology.subtype_of
        ):
            return "any"
        return cls


def vocabulary_for(ontology: Ontology) -> ExtractionVocabulary:
    """The extraction vocabulary of one ontology.

    The 7-bucket ``r4-default`` ontology is a storage vocabulary, not an
    extraction one, so it yields :func:`default_vocabulary`.
    """
    if ontology.ontology_id == DEFAULT_ONTOLOGY.ontology_id:
        return default_vocabulary()
    return ExtractionVocabulary(ontology)


@lru_cache(maxsize=1)
def default_vocabulary() -> ExtractionVocabulary:
    """The built-in ``r4-science@1`` vocabulary."""
    return ExtractionVocabulary(r4_science_ontology())


def resolve_graph_vocabulary(
    project_id: str, graph_id: str
) -> tuple[ExtractionVocabulary, list[str]]:
    """Vocabulary for the graph an understand job writes into.

    Never raises. Graphs bound to no ontology, ``r4-default`` or
    ``r4-science`` get :func:`default_vocabulary` with no note; any failure
    gets it with a note ``ontology_fallback:<reason>``.

    The default graph name ``''`` (what the workflow-output auto-ingest
    sends) means the project's default graph, so an ontology bound to that
    graph applies to it as it does to uploads; a default graph with no
    registry row or no database is the built-in vocabulary, with no note.
    """
    quiet = not graph_id
    if quiet:
        if not project_id:
            return default_vocabulary(), []
        from .graph_registry import default_graph_id

        graph_id = default_graph_id(project_id)
    try:
        vocab, notes = _resolve_from_registries(project_id, graph_id)
    except Exception as exc:  # noqa: BLE001 — extraction must not fail on vocabulary
        logger.warning("ontology vocabulary fallback for graph %s: %s", graph_id, exc)
        return default_vocabulary(), [f"ontology_fallback:{_reason(exc)}"]
    notes = [n for n in notes if not (quiet and n in _QUIET_FOR_DEFAULT)]
    if vocab.prompt_truncated:
        notes.append("vocabulary_truncated")
    return vocab, notes


# The default graph needs no registry row and no database to be read.
_QUIET_FOR_DEFAULT = frozenset(
    {"ontology_fallback:no_database", "ontology_fallback:graph_not_found"}
)


def _reason(exc: Exception) -> str:
    if isinstance(exc, ValueError):
        return "unresolved_ontology"
    return type(exc).__name__


def _resolve_from_registries(
    project_id: str, graph_id: str
) -> tuple[ExtractionVocabulary, list[str]]:
    from ..shared.pg import get_default_sync_pool, pg_configured_from_env
    from .graph_registry import GraphRegistryStore

    if not pg_configured_from_env():
        return default_vocabulary(), ["ontology_fallback:no_database"]
    pool = get_default_sync_pool()
    meta = GraphRegistryStore(pool).get_graph(graph_id)
    if meta is None:
        return default_vocabulary(), ["ontology_fallback:graph_not_found"]
    graph = meta.to_dict()
    graph["project_id"] = graph.get("project_id") or project_id
    if graph.get("ontology_id") in (
        None,
        "",
        DEFAULT_ONTOLOGY.ontology_id,
        R4_SCIENCE_ID,
    ):
        return default_vocabulary(), []
    return vocabulary_for(_registry_ontology(pool, graph)), []


def _registry_ontology(pool: Any, graph: dict[str, Any]) -> Ontology:
    from .ontology_registry import OntologyRegistryStore

    conn = pool.getconn()
    try:
        return ontology_for_graph(graph, registry=OntologyRegistryStore(conn))
    finally:
        pool.putconn(conn)


__all__ = [
    "UNCLASSIFIED",
    "ExtractionVocabulary",
    "default_vocabulary",
    "resolve_graph_vocabulary",
    "vocabulary_for",
]
