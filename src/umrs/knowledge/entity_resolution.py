"""Entity-resolution proposals (ADR-116 D-7): ``same_as`` edges, never merges.

Semantic node ids carry their artifact (``node-sem-{scope}{slug}-{artifact}``),
so one entity named in two documents is two nodes. At the end of an
understand job, :func:`propose_same_as` compares each semantic node the job
wrote with the extracted semantic nodes of the same project and graph and
writes a **proposed** ``same_as`` edge for each pair it believes name one
entity. A person establishes or ignores it through the existing review
routes; establishing copies each partner's label into the other node's
``content.aliases_reviewed`` (``KnowledgeStore.establish_edge``), which the
graph retrieval lane matches. Nodes are never merged or rewritten here.

Candidates: ``node-sem-*`` nodes with ``content.semantic_type`` that are not
chunks, in the same project and graph; never two nodes of one artifact, and
never a pair a ``same_as`` edge already links (either tier). Types must be
compatible: equal ``semantic_type``, or either side ``unclassified`` or
``concept``.

Two methods:

- ``label``: equal normalized label or alias (:func:`normalize_label`),
  score 1.0. Keys shorter than ``MIN_KEY_CHARS`` (single letters, math
  symbols such as ``R`` or ``ψ``) never match.
- ``embedding`` (opt-in: ``UMRS_KNOWLEDGE_ER_EMBEDDING=1`` and the caller
  passes ``embed_labels``; measured 2026-11-12 on 10 papers with e5-base-v2,
  about 17% of pairs at cosine >= 0.95 named one entity, against about 94%
  for the label method, so it is off by default): the job's
  nodes are embedded as ``"<semantic_type>: <label>"`` and stored under the
  model name ``<embedder_name>#label``. The dense retrieval lane reads only
  its own model name or the same basename staged under another path
  (``dense.same_space_pattern``), so it never reads these vectors. Each new
  node's nearest label vectors in the same graph are accepted at cosine
  similarity ``UMRS_KNOWLEDGE_ER_THRESHOLD`` (default 0.97) or above. Only
  nodes an earlier resolution run embedded have label vectors; older nodes
  are reached by the label method only.

Edge: ``edge-sameas-{scope}{a}-{b}`` (``a < b``), source ``a``, target ``b``,
relation ``same_as``, status ``proposed``, confidence = score, content
``{score, method, job_id, origin_actor: "extracted"}`` (the review queue
reads ``origin_actor`` as the row's origin and ``job_id`` as its run). At
most ``UMRS_KNOWLEDGE_ER_MAX_PER_JOB`` (default 200) edges per call, the
highest scores first, with a note when more were found.
``UMRS_KNOWLEDGE_ENTITY_RESOLUTION=0`` (or ``false``) turns it off.

:func:`propose_same_as` never raises: a failure becomes a note in the
result. Nothing here imports ``umrs.api`` or ``umrs.jobs``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .graph_registry import default_graph_id
from .store import SAME_AS_RELATION, KnowledgeEdge, SemanticNodeRow

logger = logging.getLogger(__name__)

ENABLE_ENV = "UMRS_KNOWLEDGE_ENTITY_RESOLUTION"
THRESHOLD_ENV = "UMRS_KNOWLEDGE_ER_THRESHOLD"
MAX_PER_JOB_ENV = "UMRS_KNOWLEDGE_ER_MAX_PER_JOB"
EMBED_MAX_ENV = "UMRS_KNOWLEDGE_ER_EMBED_MAX"
EMBEDDING_ENV = "UMRS_KNOWLEDGE_ER_EMBEDDING"
DEFAULT_THRESHOLD = 0.97
#: Shortest normalized label key the label method matches on.
MIN_KEY_CHARS = 2
DEFAULT_MAX_PER_JOB = 200
#: Most new nodes the embedding method searches for per job: each search is
#: an exact scan of the graph's label vectors (no ANN index on the untyped
#: 0075 column), so the cost is new nodes x graph size. The label method
#: still covers every node; a note names the cap when it is reached.
DEFAULT_EMBED_MAX = 500
#: Neighbour searches sent in one statement.
SEARCH_BATCH = 100
#: Suffix of the model name label vectors are stored under.
LABEL_MODEL_SUFFIX = "#label"
#: Most semantic nodes of one graph read as candidates (a note when reached).
SCAN_MAX = 50_000
#: Nearest label vectors read per new node.
NEIGHBOURS = 10
#: Types compatible with any other type.
LOOSE_TYPES = frozenset({"unclassified", "concept"})
ORIGIN_ACTOR = "extracted"
_OFF = frozenset({"0", "false", "off", "no"})
_ON = frozenset({"1", "true", "on", "yes"})
_METHOD_ORDER = {"label": 0, "embedding": 1}
_NON_WORD = re.compile(r"[\W_]+", re.UNICODE)


@dataclass
class ResolutionResult:
    """``proposed`` edges written, ``candidates`` pairs found (before the
    cap), and notes naming anything skipped or failed."""

    proposed: int = 0
    candidates: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Match:
    a: str  # the smaller node id
    b: str
    score: float
    method: str


# ── settings ─────────────────────────────────────────────────────────────────


def enabled() -> bool:
    """``UMRS_KNOWLEDGE_ENTITY_RESOLUTION``: on unless ``0``/``false``."""
    return os.environ.get(ENABLE_ENV, "").strip().lower() not in _OFF


def embedding_enabled() -> bool:
    """``UMRS_KNOWLEDGE_ER_EMBEDDING``: the embedding method, off unless ``1``/``true``."""
    return os.environ.get(EMBEDDING_ENV, "").strip().lower() in _ON


def threshold() -> float:
    """Cosine similarity an embedding match needs (bad values: the default)."""
    try:
        value = float(os.environ.get(THRESHOLD_ENV, "") or DEFAULT_THRESHOLD)
    except ValueError:
        return DEFAULT_THRESHOLD
    return value if 0.0 < value <= 1.0 else DEFAULT_THRESHOLD


def _env_count(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value >= 0 else default


def max_per_job() -> int:
    """Most edges one call writes (bad values: the default)."""
    return _env_count(MAX_PER_JOB_ENV, DEFAULT_MAX_PER_JOB)


def embed_max() -> int:
    """Most new nodes the embedding method searches for (bad values: the default)."""
    return _env_count(EMBED_MAX_ENV, DEFAULT_EMBED_MAX)


# ── the label method (pure) ──────────────────────────────────────────────────


def _singular(token: str) -> str:
    """A plain English plural strip (no dictionary): ``catalysts`` ->
    ``catalyst``, ``properties`` -> ``property``, ``boxes`` -> ``box``;
    ``gas``, ``analysis`` and ``class`` stay."""
    if len(token) <= 3 or not token.endswith("s"):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "ches", "shes", "zes")):
        return token[:-2]
    if token.endswith(("ss", "us", "is")):
        return token
    return token[:-1]


def normalize_label(text: str) -> str:
    """The key two names must share to match by label.

    The graph lane compares labels through Postgres' ``english`` text
    search configuration in SQL; there is no Python form of it, so this is
    a documented equivalent: Unicode NFKC, lowercase, every run of
    punctuation, underscores or spaces becomes one space, and each word
    loses a regular English plural ending (:func:`_singular`)."""
    folded = unicodedata.normalize("NFKC", str(text or "")).lower()
    words = _NON_WORD.sub(" ", folded).split()
    return " ".join(_singular(word) for word in words)


def types_compatible(a: str, b: str) -> bool:
    """Equal semantic types, or either is ``unclassified`` or ``concept``."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    return a == b or a in LOOSE_TYPES or b in LOOSE_TYPES


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairable(
    node: SemanticNodeRow, other: SemanticNodeRow, existing: set[tuple[str, str]]
) -> bool:
    """Two different nodes of different artifacts, compatible types, not
    already linked by a same_as edge."""
    if node.node_id == other.node_id or node.artifact_id == other.artifact_id:
        return False
    if not types_compatible(node.semantic_type, other.semantic_type):
        return False
    return _pair(node.node_id, other.node_id) not in existing


def _keys(node: SemanticNodeRow) -> set[str]:
    names = (node.label, *node.aliases)
    keys = (normalize_label(name) for name in names)
    return {key for key in keys if len(key) >= MIN_KEY_CHARS}


def _label_index(nodes: Iterable[SemanticNodeRow]) -> dict[str, list[SemanticNodeRow]]:
    index: dict[str, list[SemanticNodeRow]] = {}
    for node in nodes:
        for key in _keys(node):
            index.setdefault(key, []).append(node)
    return index


def label_matches(
    new_nodes: Sequence[SemanticNodeRow],
    graph_nodes: Sequence[SemanticNodeRow],
    existing: set[tuple[str, str]],
) -> list[_Match]:
    """Pairs whose normalized label or alias is equal (score 1.0)."""
    index = _label_index([*graph_nodes, *new_nodes])
    found: dict[tuple[str, str], _Match] = {}
    for node in new_nodes:
        for key in _keys(node):
            for other in index.get(key, ()):
                if _pairable(node, other, existing):
                    a, b = _pair(node.node_id, other.node_id)
                    found[(a, b)] = _Match(a, b, 1.0, "label")
    return list(found.values())


# ── the embedding method ─────────────────────────────────────────────────────


def label_text(node: SemanticNodeRow) -> str:
    """The string a node's label vector embeds."""
    return f"{node.semantic_type}: {node.label}"


def label_model_name(embedder_name: str) -> str:
    """The model name label vectors are stored under (never a dense space)."""
    return f"{embedder_name}{LABEL_MODEL_SUFFIX}"


async def _embed(
    embed_labels: Callable[[list[str]], list[list[float]]], texts: list[str]
) -> list[list[float]]:
    vectors = await asyncio.to_thread(embed_labels, texts)
    vectors = [list(map(float, v)) for v in vectors or ()]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(texts)} labels"
        )
    return vectors


async def _store_vectors(
    store: Any, nodes: Sequence[SemanticNodeRow], vectors: list[list[float]], model: str
) -> None:
    for node, vector in zip(nodes, vectors, strict=True):
        digest = hashlib.sha256(label_text(node).encode()).hexdigest()
        await store.upsert_node_embedding(
            node.node_id, model, vector, content_hash=digest
        )


def _near_matches(
    node: SemanticNodeRow,
    neighbours: list[tuple[str, float]],
    by_id: dict[str, SemanticNodeRow],
    existing: set[tuple[str, str]],
    cutoff: float,
) -> list[_Match]:
    found = []
    for other_id, distance in neighbours:
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        other = by_id.get(other_id)
        if other is not None and score >= cutoff and _pairable(node, other, existing):
            a, b = _pair(node.node_id, other_id)
            found.append(_Match(a, b, round(score, 4), "embedding"))
    return found


@dataclass(frozen=True)
class _Scope:
    project_id: str
    graph_id: str
    job_id: str


async def embedding_matches(
    store: Any,
    scope: _Scope,
    new_nodes: Sequence[SemanticNodeRow],
    by_id: dict[str, SemanticNodeRow],
    existing: set[tuple[str, str]],
    *,
    embed_labels: Callable[[list[str]], list[list[float]]],
    embedder_name: str,
) -> list[_Match]:
    """Embed the new nodes' labels, store them under ``<embedder>#label``,
    and pair each with its nearest label vectors in the same graph at or
    above :func:`threshold`. The search leaves out the node itself and its
    own artifact's nodes before taking the ``NEIGHBOURS`` nearest, and runs
    ``SEARCH_BATCH`` nodes per statement."""
    model = label_model_name(embedder_name)
    vectors = await _embed(embed_labels, [label_text(n) for n in new_nodes])
    await _store_vectors(store, new_nodes, vectors, model)
    queries = [
        (node.node_id, node.artifact_id, vector)
        for node, vector in zip(new_nodes, vectors, strict=True)
    ]
    cutoff, found = threshold(), []
    for start in range(0, len(queries), SEARCH_BATCH):
        neighbours = await store.nearest_label_neighbours(
            queries[start : start + SEARCH_BATCH],
            project_id=scope.project_id,
            model_name=model,
            limit=NEIGHBOURS,
            graph_id=scope.graph_id,
        )
        for node in new_nodes[start : start + SEARCH_BATCH]:
            near = neighbours.get(node.node_id, [])
            found.extend(_near_matches(node, near, by_id, existing, cutoff))
    return found


# ── writing ──────────────────────────────────────────────────────────────────


def node_id_scope(project_id: str, graph_id: str) -> str:
    """The id prefix of a named graph's rows; the default graph has none
    (the same rule as the understand job's semantic node ids)."""
    if not graph_id or graph_id == default_graph_id(project_id):
        return ""
    return f"{graph_id}-"


def same_as_edge(match: _Match, scope: _Scope) -> KnowledgeEdge:
    """The proposed edge for one match."""
    prefix = node_id_scope(scope.project_id, scope.graph_id)
    return KnowledgeEdge(
        edge_id=f"edge-sameas-{prefix}{match.a}-{match.b}",
        project_id=scope.project_id,
        source_id=match.a,
        target_id=match.b,
        relation_type=SAME_AS_RELATION,
        content={
            "score": match.score,
            "method": match.method,
            "job_id": scope.job_id,
            "origin_actor": ORIGIN_ACTOR,
        },
        status="proposed",
        confidence=match.score,
        graph_id=scope.graph_id,
    )


def _ranked(matches: Iterable[_Match]) -> list[_Match]:
    """One match per pair (the label method first), highest score first."""
    best: dict[tuple[str, str], _Match] = {}
    ordered = sorted(matches, key=lambda m: (_METHOD_ORDER[m.method], -m.score))
    for match in ordered:
        best.setdefault((match.a, match.b), match)
    return sorted(
        best.values(), key=lambda m: (-m.score, _METHOD_ORDER[m.method], m.a, m.b)
    )


async def _write(
    store: Any, matches: list[_Match], scope: _Scope, result: ResolutionResult
) -> None:
    cap = max_per_job()
    if len(matches) > cap:
        result.notes.append(f"entity_resolution_capped:{len(matches)}>{cap}")
    failed = 0
    for match in matches[:cap]:
        try:
            await store.upsert_edge(same_as_edge(match, scope))
            result.proposed += 1
        except Exception as exc:  # noqa: BLE001 — one bad pair never stops the rest
            failed += 1
            logger.warning("same_as edge %s-%s not written: %s", match.a, match.b, exc)
    if failed:
        result.notes.append(f"entity_resolution_write_failed:{failed}")


# ── the entry point ──────────────────────────────────────────────────────────


async def _read(
    store: Any, scope: _Scope, node_ids: list[str], result: ResolutionResult
) -> tuple[list[SemanticNodeRow], list[SemanticNodeRow], set[tuple[str, str]]]:
    kwargs = {"graph_id": scope.graph_id}
    new_nodes = await store.list_semantic_nodes(
        scope.project_id, node_ids=node_ids, **kwargs
    )
    if not new_nodes:
        return [], [], set()
    graph_nodes = await store.list_semantic_nodes(
        scope.project_id, limit=SCAN_MAX, **kwargs
    )
    if len(graph_nodes) >= SCAN_MAX:
        result.notes.append(f"entity_resolution_scan_capped:{SCAN_MAX}")
    existing = await store.same_as_pairs([n.node_id for n in new_nodes])
    return new_nodes, graph_nodes, existing


async def _embedding_or_note(
    store: Any,
    scope: _Scope,
    nodes: tuple[list[SemanticNodeRow], list[SemanticNodeRow], set[tuple[str, str]]],
    embed_labels: Callable[[list[str]], list[list[float]]] | None,
    embedder_name: str,
    result: ResolutionResult,
) -> list[_Match]:
    """The embedding method's matches; a failure is a note, and the label
    matches are still written."""
    if embed_labels is None or not embedding_enabled():
        return []
    if not embedder_name:
        result.notes.append("entity_resolution_embedding_skipped:no_embedder_name")
        return []
    new_nodes, graph_nodes, existing = nodes
    by_id = {n.node_id: n for n in (*graph_nodes, *new_nodes)}
    cap = embed_max()
    if len(new_nodes) > cap:
        result.notes.append(
            f"entity_resolution_embedding_capped:{cap}/{len(new_nodes)}"
        )
    try:
        return await embedding_matches(
            store,
            scope,
            new_nodes[:cap],
            by_id,
            existing,
            embed_labels=embed_labels,
            embedder_name=embedder_name,
        )
    except Exception as exc:  # noqa: BLE001 — reported as a note, never raised
        logger.warning("entity resolution embedding method failed: %s", exc)
        result.notes.append(f"entity_resolution_embedding_failed:{type(exc).__name__}")
        return []


async def _resolve(
    store: Any,
    scope: _Scope,
    node_ids: list[str],
    embed_labels: Callable[[list[str]], list[list[float]]] | None,
    embedder_name: str,
    result: ResolutionResult,
) -> None:
    nodes = await _read(store, scope, node_ids, result)
    new_nodes, graph_nodes, existing = nodes
    if not new_nodes:
        return
    matches = label_matches(new_nodes, graph_nodes, existing)
    matches += await _embedding_or_note(
        store, scope, nodes, embed_labels, embedder_name, result
    )
    ranked = _ranked(matches)
    result.candidates = len(ranked)
    await _write(store, ranked, scope, result)


async def propose_same_as(
    store: Any,
    *,
    project_id: str,
    graph_id: str,
    node_ids: Sequence[str],
    job_id: str,
    embed_labels: Callable[[list[str]], list[list[float]]] | None = None,
    embedder_name: str = "",
) -> ResolutionResult:
    """Write proposed ``same_as`` edges for the semantic nodes ``node_ids``
    (the ones a job just wrote). ``embed_labels`` is a sync callable (the
    site's chunk embedder); it runs in a worker thread. Never raises."""
    result = ResolutionResult()
    if not enabled():
        result.notes.append("entity_resolution:disabled")
        return result
    ids = sorted({str(n) for n in node_ids if n})
    if not ids:
        return result
    scope = _Scope(project_id=project_id, graph_id=graph_id or "", job_id=job_id)
    try:
        await _resolve(store, scope, ids, embed_labels, embedder_name, result)
    except Exception as exc:  # noqa: BLE001 — reported as a note, never raised
        logger.warning("entity resolution failed for job %s: %s", job_id, exc)
        result.notes.append(f"entity_resolution_failed:{type(exc).__name__}")
    return result
