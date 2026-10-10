"""The graph bridge lane, for sites whose extraction model is large (ADR-093 D-14, C20).

At ingest a site LLM extracts entities from prose
(``jobs/knowledge_adapters._semantic_nodes_edges``): one semantic node per
entity per document, ``node-sem-{scope}{slug}-{artifact}``, with
``content.semantic_type`` and ``content.artifact_id``, and a ``mentions`` edge
from the document's file node to it. The same entity in two documents is two
nodes with one label. Extraction is reliable from about 30B (the LightRAG
floor), so the lane is gated: ``UMRS_RAG_GRAPH_LANE=off|auto|on``. The api
layer resolves the gate (this layer never reads the registry) and hands the
spec ``graph_lane`` ``on`` or ``off``:

- ``on`` runs the lane (an operator's or an eval's explicit choice);
- ``auto`` (unset) runs it only when the lane has passed its multi-hop gate
  (``UMRS_RAG_GRAPH_LANE_MEASURED=1``, which the maintainer sets once the
  probe passes; D-14: "no default flips without a multi-hop probe") AND the
  site's extraction model is large under the ONE size rule
  (``r4_contracts.models.model_size_class``: the EFFECTIVE size, so a
  30B-A3B extracts like a ~9.5B model and gets no lane; lead decision,
  2026-10-10). Until the probe passes, ``auto`` is off everywhere (review of
  ADR-093, 2026-10-10: every other ADR-093 stage defaults off until its gate
  passes);
- ``off``, and any value that is not one of the three (``false``, ``0``,
  ``disabled``: an operator's attempt to switch it off), is off. A bad value
  fails closed and is logged once.

When on, after the cut (never displacing a kept passage):

1. **link** — the query's content terms, under the lexical lane's rules
   (``to_tsvector('english', ...)``: its stemming and stop words), against
   the labels (and ``content.aliases`` when present) of the semantic nodes in
   the spec's scope (the dense lane's graph predicate, exclusions and ADR-050
   visibility). An entity is one label (its stemmed terms) over every
   document; the statement groups the nodes by label BEFORE its limit, so a
   label a thousand documents mention is one row, never the whole budget.
   Its score is the share of its label's terms the query holds, at least
   ``LINK_MIN_FRACTION``; at most ``MAX_ENTITIES`` are kept.
2. **route** — label variants of one named concept are one entity (a label
   whose terms hold another linked label's terms: "ZSM-5 zeolite adsorption"
   is "ZSM-5 zeolite"). Fire only when two linked entities sit in different
   documents (neither one's documents hold the other's), toward the
   documents that hold one of such a pair but not the other and that none
   of the kept passages came from.
3. **hop** — those documents' best chunk by cosine to the query vector, in
   ONE statement under the dense lane's scope, at most ``MAX_BRIDGES``,
   appended after the kept passages with ``lanes=('graph',)``; the entity
   that routed each one is reported as ``bridge_entity``.

The label match is written for an expression GIN index over
``to_tsvector('english', coalesce(label, ''))`` on the semantic rows
(:data:`LABEL_INDEX`). Without it the statement computes every label's
lexemes, so it runs only while the scope holds :data:`LABEL_SCAN_MAX_ROWS`
semantic nodes or fewer; above that the lane reports ``unavailable`` at once
instead of spending the statement budget on every retrieve.

Lane report: ``off`` | ``unavailable`` (no semantic graph in scope, a scope
too large to scan without the label index, no query vector, or a failed
statement by its exception type) | ``ok`` with counts. No chat-model call
happens here. User text reaches SQL only as a bound parameter.
Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from r4_contracts.models import model_size_class

from .dense import (
    _exclude_clause,
    as_content,
    gate_clause,
    scope_clause,
    vector_literal,
)
from .types import Candidate, LaneResult, RetrieveSpec

logger = logging.getLogger(__name__)

GRAPH_LANE = "graph"
GRAPH_ENV = "UMRS_RAG_GRAPH_LANE"
#: Set (``1``/``true``/``yes``/``on``) once the lane passed its multi-hop
#: gate; until then ``auto`` is off (D-14).
MEASURED_ENV = "UMRS_RAG_GRAPH_LANE_MEASURED"
GRAPH_MODES: tuple[str, ...] = ("off", "auto", "on")
_TRUE = frozenset({"1", "true", "yes", "on"})
MAX_ENTITIES = 8
MAX_BRIDGES = 2
#: A label is linked when the query holds at least this share of its terms:
#: a one- or two-term label in full, three of four, three of five.
LINK_MIN_FRACTION = 0.6
#: Labels (entities, not nodes) read per link, the strongest label match first.
LINK_GROUP_LIMIT = 64
#: Documents the hop reads chunks of (the most strongly linked first).
HOP_ARTIFACT_LIMIT = 16
#: Each statement's budget.
STATEMENT_TIMEOUT_MS = 1500
#: The expression index the label match is written for (a migration builds
#: it): ``gin (to_tsvector('english', coalesce(label, ''))) WHERE node_id
#: LIKE 'node-sem-%'``.
LABEL_INDEX = "idx_knowledge_nodes_semantic_label_fts"
#: Without the label index the link statement computes every label's lexemes
#: (about 20 us each): it runs only up to this many semantic nodes in scope.
LABEL_SCAN_MAX_ROWS = 20_000
SEMANTIC_PREFIX = "node-sem-"

# The semantic nodes of the scope: the dense lane's gate and exclusions,
# entity/concept rows (idx_node_project) with the writer's id prefix.
_SEMANTIC_WHERE = (
    "{gate} AND n.node_type IN ('entity', 'concept') AND n.node_id LIKE %s "
    "AND n.content->>'semantic_type' IS NOT NULL{exclude}"
)
# Whether the label index exists, and the scope's semantic nodes up to a cap.
_SCOPE_SQL = (
    "/* graph:scope */ SELECT to_regclass(%s) IS NOT NULL, (SELECT count(*) FROM "
    "(SELECT 1 FROM knowledge_nodes n WHERE {where} LIMIT %s) AS c)"
)
# The label's lexemes: the expression LABEL_INDEX holds.
_LABEL_TSV = "to_tsvector('english', coalesce(n.label, ''))"
_ALIAS_VALUES = (
    "jsonb_array_elements_text(CASE WHEN jsonb_typeof(n.content->'aliases') = "
    "'array' THEN n.content->'aliases' ELSE '[]'::jsonb END) AS a(v)"
)
_ALIASES = (
    "ARRAY(SELECT array_to_string(tsvector_to_array(to_tsvector('english', a.v)), ' ') "
    f"FROM {_ALIAS_VALUES})"
)
_TSQ = "(SELECT q.tsq FROM q)"
_ALIAS_HIT = (
    "jsonb_typeof(n.content->'aliases') = 'array' AND EXISTS (SELECT 1 FROM "
    f"{_ALIAS_VALUES} WHERE to_tsvector('english', a.v) @@ {_TSQ})"
)
_NODE_COLUMNS = (
    f"n.node_id, n.label, n.content, tsvector_to_array({_LABEL_TSV}) AS label_lex, "
    f"{_ALIASES} AS alias_lex"
)
_MENTIONED_BY = (
    "ARRAY(SELECT e.content->>'artifact_id' FROM knowledge_edges e "
    "WHERE e.target_id = m.node_id AND e.relation_type = 'mentions' "
    "AND e.content->>'artifact_id' IS NOT NULL)"
)
_MATCHED = (
    "cardinality(ARRAY(SELECT unnest(d.label_lex) INTERSECT "
    "SELECT unnest((SELECT q.lex FROM q))))"
)
# One row per label (an entity): its documents (own artifact and every
# 'mentions' edge) and its aliases aggregated BEFORE the limit, so a label
# many documents mention cannot crowd the query's other entities out. The
# label branch is index-ready; the alias branch reads only rows with aliases
# (a node both branches return is counted once: every aggregate is DISTINCT).
_LINK_SQL = (
    "/* graph:link */ WITH q AS MATERIALIZED ("
    " SELECT coalesce(array_agg(DISTINCT lex), '{{}}'::text[]) AS lex,"
    " string_agg(DISTINCT quote_literal(lex), ' | ')::tsquery AS tsq"
    " FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS lex"
    " WHERE strpos(lex, chr(92)) = 0"
    f"), m AS (SELECT {_NODE_COLUMNS}"
    " FROM knowledge_nodes n WHERE {where}"
    f" AND {_LABEL_TSV} @@ {_TSQ}"
    f" UNION ALL SELECT {_NODE_COLUMNS}"
    " FROM knowledge_nodes n WHERE {where}"
    f" AND {_ALIAS_HIT}"
    "), d AS (SELECT m.node_id, m.label, m.label_lex, al.v AS alias, doc.a AS artifact"
    f" FROM m CROSS JOIN LATERAL unnest(array_append({_MENTIONED_BY},"
    " m.content->>'artifact_id')) AS doc(a)"
    " LEFT JOIN LATERAL unnest(m.alias_lex) AS al(v) ON true"
    ") SELECT min(d.node_id), min(d.label), '{{}}', d.label_lex,"
    " coalesce(array_agg(DISTINCT d.alias) FILTER (WHERE d.alias IS NOT NULL), '{{}}'),"
    " coalesce(array_agg(DISTINCT d.artifact) FILTER (WHERE d.artifact <> ''), '{{}}'),"
    " (SELECT q.lex FROM q) FROM d GROUP BY d.label_lex"
    f" ORDER BY {_MATCHED}::float8 / greatest(cardinality(d.label_lex), 1) DESC,"
    f" {_MATCHED} DESC, cardinality(d.label_lex), d.label_lex LIMIT %s"
)
_HOP_SQL = (
    "/* graph:hop */ SELECT DISTINCT ON (n.content->>'artifact_id') e.node_id,"
    " n.content::text, (e.embedding <=> %s::vector) AS distance"
    " FROM knowledge_node_embeddings e JOIN knowledge_nodes n ON n.node_id = e.node_id"
    " WHERE {where} AND n.content->>'artifact_id' = ANY(%s)"
    " ORDER BY n.content->>'artifact_id', distance ASC, e.node_id"
)
_warned_modes: set[str] = set()


# ── the gate ────────────────────────────────────────────────────────────────


def _setting() -> str:
    return os.environ.get(GRAPH_ENV, "").strip().lower() or "auto"


def graph_lane_mode() -> str:
    """``UMRS_RAG_GRAPH_LANE``: ``off``, ``auto`` (unset) or ``on``. Any other
    value is ``off``: an operator who writes ``false`` or ``0`` meant off,
    and the lane is unmeasured (fail closed, logged once)."""
    setting = _setting()
    if setting in GRAPH_MODES:
        return setting
    if setting not in _warned_modes:
        _warned_modes.add(setting)
        logger.warning("%s=%r is not off|auto|on: the lane is off", GRAPH_ENV, setting)
    return "off"


def lane_measured() -> bool:
    """``UMRS_RAG_GRAPH_LANE_MEASURED``: the lane passed its multi-hop gate."""
    return os.environ.get(MEASURED_ENV, "").strip().lower() in _TRUE


def _explicit(mode: str) -> str:
    setting = _setting()
    if setting == mode:
        return f"{GRAPH_ENV}={mode}"
    return f"{GRAPH_ENV}={setting!r} is not off|auto|on: off"


def resolve_gate(extraction_model: Callable[[], str | None]) -> tuple[str, str]:
    """``(on|off, how)``. ``extraction_model`` is asked only under ``auto``
    (the api layer reads it from the registry); ``auto`` is on only for a
    large extraction model once the lane is measured."""
    mode = graph_lane_mode()
    if mode != "auto":
        return mode, _explicit(mode)
    model = extraction_model() or ""
    size = model_size_class(model)
    shown = model or "none"
    if size != "large":
        return "off", f"auto: extraction model {shown} is {size}, not large"
    if not lane_measured():
        return "off", (
            f"auto: extraction model {shown} is large, but the lane has not "
            f"passed its multi-hop gate ({MEASURED_ENV} unset)"
        )
    return "on", f"auto: extraction model {shown} is large"


# ── link and route (pure) ───────────────────────────────────────────────────


@dataclass
class LinkedEntity:
    """One label over every document that mentions it."""

    key: tuple[str, ...]  # the label's stemmed terms
    label: str
    score: float  # the share of the label's terms the query holds
    matched: int
    artifacts: set[str] = field(default_factory=set)


def _strings(value: Any) -> list[str]:
    return [str(v) for v in value or () if v]


def _fraction(terms: Sequence[str], query: set[str]) -> tuple[float, int]:
    unique = set(terms)
    if not unique:
        return 0.0, 0
    hit = len(unique & query)
    return hit / len(unique), hit


def _best_name(row: Sequence[Any], query: set[str]) -> tuple[float, int]:
    """The best match over the label and its aliases."""
    names = [_strings(row[3]), *(alias.split() for alias in _strings(row[4]))]
    return max(_fraction(terms, query) for terms in names)


def _row_artifacts(row: Sequence[Any]) -> set[str]:
    own = str(as_content(row[2]).get("artifact_id") or "")
    return {a for a in (own, *_strings(row[5])) if a}


def link(rows: Iterable[Sequence[Any]]) -> list[LinkedEntity]:
    """Entities from the link statement's rows ``(node_id, label, content,
    label_lex, alias_lex, mentioned_by, query_lex)``: grouped by label,
    scored, at least ``LINK_MIN_FRACTION``, the best ``MAX_ENTITIES``."""
    found: dict[tuple[str, ...], LinkedEntity] = {}
    for row in rows:
        score, matched = _best_name(row, set(_strings(row[6])))
        artifacts = _row_artifacts(row)
        if score < LINK_MIN_FRACTION or not artifacts:
            continue
        key = tuple(sorted(set(_strings(row[3])))) or (str(row[1]).lower(),)
        entity = found.setdefault(key, LinkedEntity(key, str(row[1]), score, matched))
        entity.score, entity.matched = max(
            (entity.score, entity.matched), (score, matched)
        )
        entity.artifacts |= artifacts
    ranked = sorted(found.values(), key=lambda e: (-e.score, -e.matched, e.key))
    return ranked[:MAX_ENTITIES]


def _merge(concept: LinkedEntity, variant: LinkedEntity) -> None:
    concept.score, concept.matched = max(
        (concept.score, concept.matched), (variant.score, variant.matched)
    )
    concept.artifacts |= variant.artifacts


def concepts(entities: Sequence[LinkedEntity]) -> list[LinkedEntity]:
    """The named concepts among linked labels: a label whose terms hold
    another linked label's terms is a variant of it ("ZSM-5 zeolite
    adsorption" of "ZSM-5 zeolite") and joins it, documents included.
    Copies; ranked like :func:`link`."""
    found: list[LinkedEntity] = []
    for entity in sorted(entities, key=lambda e: (len(e.key), e.key)):
        general = next((c for c in found if set(c.key) <= set(entity.key)), None)
        if general is None:
            found.append(dataclasses.replace(entity, artifacts=set(entity.artifacts)))
        else:
            _merge(general, entity)
    return sorted(found, key=lambda e: (-e.score, -e.matched, e.key))


def _apart(a: LinkedEntity, b: LinkedEntity) -> bool:
    """Each sits in a document the other does not (neither set holds the other)."""
    return not (a.artifacts <= b.artifacts or b.artifacts <= a.artifacts)


def _bridging(named: Sequence[LinkedEntity]) -> set[str]:
    """The documents that hold one entity of an apart pair but not the other."""
    return {
        artifact
        for i, a in enumerate(named)
        for b in named[i + 1 :]
        if _apart(a, b)
        for artifact in a.artifacts ^ b.artifacts
    }


def route(
    entities: Sequence[LinkedEntity], kept_artifacts: set[str]
) -> tuple[list[str], str]:
    """``(documents to hop to, why not)``: fire only when two linked
    concepts sit in different documents (ADR-093 D-14), toward the
    documents that hold one of such a pair and that no kept passage came
    from, most strongly linked first. Label variants of one concept, and
    entities that always sit in the same documents, never fire."""
    named = concepts(entities)
    if len(named) < 2:
        return [], "fewer than two linked entities"
    if len(set().union(*(e.artifacts for e in named))) < 2:
        return [], "the linked entities sit in one document"
    bridging = _bridging(named)
    if not bridging:
        return [], "no two linked entities sit in different documents"
    weight = _weights(named)
    fresh = sorted(bridging - kept_artifacts, key=lambda a: (-weight[a], a))
    if not fresh:
        return [], "every linked document is already kept"
    return fresh[:HOP_ARTIFACT_LIMIT], ""


def _weights(entities: Sequence[LinkedEntity]) -> dict[str, float]:
    """Per document: the summed scores of the linked entities it mentions."""
    weight: dict[str, float] = {}
    for entity in entities:
        for artifact in entity.artifacts:
            weight[artifact] = weight.get(artifact, 0.0) + entity.score
    return weight


def _bridge_entity(artifact: str, entities: Sequence[LinkedEntity]) -> str:
    """The best linked entity the document mentions (entities are ranked)."""
    return next((e.label for e in entities if artifact in e.artifacts), "")


def bridges(
    rows: Iterable[Sequence[Any]], entities: Sequence[LinkedEntity]
) -> list[Candidate]:
    """At most ``MAX_BRIDGES`` candidates from the hop's rows ``(node_id,
    content, distance)``: the most strongly linked documents first, then
    the closer chunk."""
    weight = _weights(entities)
    found = []
    for node_id, content_text, distance in rows:
        content = as_content(content_text)
        artifact = str(content.get("artifact_id") or "")
        index = content.get("chunk_index")
        found.append(
            Candidate(
                node_id=str(node_id),
                artifact_id=artifact,
                chunk_index=index if isinstance(index, int) else 0,
                content=content,
                cosine=1.0 - float(distance),
                lane_scores={GRAPH_LANE: round(weight.get(artifact, 0.0), 6)},
            )
        )
    found.sort(
        key=lambda c: (-c.lane_scores[GRAPH_LANE], -(c.cosine or 0.0), c.node_id)
    )
    for rank, candidate in enumerate(found[:MAX_BRIDGES], start=1):
        candidate.lane_ranks[GRAPH_LANE] = rank
    return found[:MAX_BRIDGES]


# ── database access ─────────────────────────────────────────────────────────


@contextmanager
def _cursor(pool: Any) -> Iterator[Any]:
    """One read transaction; each statement bounded by ``STATEMENT_TIMEOUT_MS``."""
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('statement_timeout', %s, true)",
                    (str(STATEMENT_TIMEOUT_MS),),
                )
                yield cur
    finally:
        pool.putconn(conn)


def _semantic_where(spec: RetrieveSpec) -> tuple[str, list[Any]]:
    gate, params = gate_clause(spec)
    exclude, exclude_params = _exclude_clause(spec)
    where = _SEMANTIC_WHERE.format(gate=gate, exclude=exclude)
    return where, [*params, f"{SEMANTIC_PREFIX}%", *exclude_params]


def _scope_blocked(cur: Any, spec: RetrieveSpec) -> str:
    """Why the link cannot run, or ``""``: no semantic node in scope, or more
    than :data:`LABEL_SCAN_MAX_ROWS` of them and no label index to read."""
    where, params = _semantic_where(spec)
    cur.execute(
        _SCOPE_SQL.format(where=where),
        [LABEL_INDEX, *params, LABEL_SCAN_MAX_ROWS + 1],
    )
    indexed, count = cur.fetchone() or (False, 0)
    if not count:
        return "no semantic graph in scope"
    if count > LABEL_SCAN_MAX_ROWS and not indexed:
        return (
            f"no label index ({LABEL_INDEX}) and more than {LABEL_SCAN_MAX_ROWS} "
            "semantic nodes in scope (a scan runs only up to that)"
        )
    return ""


def _linked(cur: Any, spec: RetrieveSpec) -> list[LinkedEntity]:
    where, params = _semantic_where(spec)
    cur.execute(
        _LINK_SQL.format(where=where),
        [spec.query, *params, *params, LINK_GROUP_LIMIT],
    )
    return link(cur.fetchall())


def _hop(
    cur: Any,
    spec: RetrieveSpec,
    vector: Sequence[float],
    model: str,
    documents: list[str],
) -> list[tuple]:
    where, params = scope_clause(spec, model)
    cur.execute(
        _HOP_SQL.format(where=where), [vector_literal(vector), *params, documents]
    )
    return list(cur.fetchall())


# ── the lane ────────────────────────────────────────────────────────────────


def _result(candidates: list[Candidate], **info: Any) -> LaneResult:
    reason = str(info.pop("reason", ""))
    return LaneResult(name=GRAPH_LANE, candidates=candidates, reason=reason, info=info)


def _run(
    pool: Any, spec: RetrieveSpec, vector: Sequence[float], model: str, kept: set[str]
) -> LaneResult:
    with _cursor(pool) as cur:
        blocked = _scope_blocked(cur, spec)
        if blocked:
            return LaneResult(name=GRAPH_LANE, status="unavailable", reason=blocked)
        entities = _linked(cur, spec)
        named = concepts(entities)
        documents, why_not = route(named, kept)
        rows = _hop(cur, spec, vector, model, documents) if documents else []
    found = bridges(rows, named)
    return _result(
        found,
        reason=why_not,
        linked=len(entities),
        concepts=len(named),
        documents=len(set().union(*(e.artifacts for e in named))),
        fired=bool(documents),
        bridges=[
            {
                "source_id": c.node_id,
                "bridge_entity": _bridge_entity(c.artifact_id, named),
            }
            for c in found
        ],
    )


def bridge_lane(
    pool: Any,
    spec: RetrieveSpec,
    *,
    vector: Sequence[float] | None,
    model: str,
    kept: Sequence[Candidate],
) -> LaneResult:
    """Lane ``graph``: at most ``MAX_BRIDGES`` bridge chunks for ``spec``
    (unhydrated; the pipeline hydrates and appends them after ``kept``).
    A failure is reported by its exception type; the message is logged."""
    if not vector:
        return LaneResult(
            name=GRAPH_LANE, status="unavailable", reason="no query vector"
        )
    started = time.perf_counter()
    kept_artifacts = {c.artifact_id for c in kept if c.artifact_id}
    try:
        result = _run(pool, spec, vector, model, kept_artifacts)
    except Exception as exc:  # noqa: BLE001 — a lane reports, it never raises (ADR-083 D-4)
        logger.warning("graph lane failed: %s", exc)
        result = LaneResult(
            name=GRAPH_LANE,
            status="unavailable",
            reason=f"graph lane failed ({type(exc).__name__})",
        )
    result.ms = round((time.perf_counter() - started) * 1000.0, 2)
    return result


__all__ = [
    "GRAPH_ENV",
    "GRAPH_LANE",
    "LABEL_INDEX",
    "LABEL_SCAN_MAX_ROWS",
    "LINK_MIN_FRACTION",
    "MAX_BRIDGES",
    "MAX_ENTITIES",
    "MEASURED_ENV",
    "LinkedEntity",
    "bridge_lane",
    "bridges",
    "concepts",
    "graph_lane_mode",
    "lane_measured",
    "link",
    "resolve_gate",
    "route",
]
