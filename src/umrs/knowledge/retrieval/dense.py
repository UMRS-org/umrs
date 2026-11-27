"""The dense lane of the retrieval core, and the query encoder it reads (ADR-093 D-2).

Cosine (``<=>``) over the 0075 node vectors, in ONE registered embedding
space (ADR-083 D-1/D-2, slice KN-S6): the query model resolves through the
space registry, the read carries the space's partial-index predicate and
runs as a filtered ANN read on ``hnsw_<space_id>`` re-sorted by exact
distance (``ann.py``), chunk nodes only, under the ADR-050 visibility
predicate, plus a graph predicate when the caller selected graphs. A pool of
``limit`` candidates per query variant — larger than the ``k`` the cut keeps
(ADR-083 D-2's over-fetch shape).

Legacy mode (``spec.threshold`` set) runs the K3 similarity statement itself
— no chunk-kind predicate, no window, ``LIMIT k`` — so the ids and their
order are exactly those ``rag_spine.retrieve_passages`` returned before
ADR-093 (acceptance receipt 4).

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import httpx

from umrs.knowledge.access_scope import artifact_alive_predicate, visibility_predicate
from umrs.knowledge.embedding_spaces import EmbeddingSpace
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.text_embedder import embedding_space

from . import ann
from .types import BatchEncoder, Candidate, LaneResult, RetrieveSpec

logger = logging.getLogger(__name__)

LANE = "dense"

_FROM = (
    "FROM knowledge_node_embeddings e JOIN knowledge_nodes n ON n.node_id = e.node_id "
)
#: ``graph_registry.default_graph_id(n.project_id)`` computed in Postgres (the
#: sha256 rule; pinned against the Python function by a unit test).
DEFAULT_GRAPH_SQL = (
    "('g-' || left(encode(sha256(convert_to('project:' || n.project_id, 'UTF8')), "
    "'hex'), 16))"
)
#: A selected graph's own rows, plus the legacy ``graph_id = ''`` rows of every
#: project whose DEFAULT graph is selected — K3's ``_GraphScope`` rule.
_GRAPH_OWNED = (
    f"(n.graph_id = ANY(%s) OR (n.graph_id = '' AND {DEFAULT_GRAPH_SQL} = ANY(%s)))"
)
#: The pre-ADR-093 K3 similarity statement (api/knowledge_query.py), from FROM on.
_LEGACY_TAIL = (
    "FROM knowledge_node_embeddings e "
    "JOIN knowledge_nodes n ON n.node_id = e.node_id "
    "WHERE {gate} "
    "AND (e.model_name = %s OR e.model_name LIKE %s) "
    "ORDER BY distance ASC LIMIT %s"
)


# ── database access shared by the core ──────────────────────────────────────


def query_rows(pool: Any, sql: str, params: Sequence[Any]) -> list[tuple]:
    """Run one read; the transaction ends before the connection goes back.

    psycopg2's pool closes a returned connection whenever it already holds
    ``minconn`` idle ones, so ``with conn`` must sit inside ``try`` (the
    2026-10-09 "connection already closed" regression).
    """
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return list(cur.fetchall())
    finally:
        pool.putconn(conn)


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def same_space_pattern(model: str) -> str:
    """``LIKE`` pattern for the same model staged under another path."""
    space = embedding_space(model).replace("\\", "\\\\")
    return "%/" + space.replace("%", "\\%").replace("_", "\\_")


# ── the scope predicate ─────────────────────────────────────────────────────


def gate_clause(spec: RetrieveSpec) -> tuple[str, list[Any]]:
    """The ownership + ADR-050 visibility half of the scope, and its params.

    With graphs selected, legacy ``graph_id = ''`` rows belong to their
    project's DEFAULT graph only (KG-05 review, 2026-10-09): they are read
    for every project whose default graph is among the selected (resolved,
    readable) ids — the caller's or a shared one of another project — exactly
    as K3's ``_GraphScope.of`` derives it from graph metadata (review core-10,
    2026-10-10). A shared non-default graph shows only its graph-bound rows.
    """
    if spec.graph_ids is None:
        gate = visibility_predicate("n", include_ambient=spec.include_ambient)
        return gate, [spec.project_id]
    gate = visibility_predicate(
        "n", owned_sql=_GRAPH_OWNED, include_ambient=spec.include_ambient
    )
    return gate, [list(spec.graph_ids), list(spec.graph_ids)]


def _exclude_clause(spec: RetrieveSpec) -> tuple[str, list[Any]]:
    if not spec.exclude_graph_ids:
        return "", []
    sql = " AND n.graph_id <> ALL(%s)"
    if default_graph_id(spec.project_id) in spec.exclude_graph_ids:
        sql += " AND n.graph_id <> ''"
    return sql, [list(spec.exclude_graph_ids)]


def scope_clause(spec: RetrieveSpec, model: str) -> tuple[str, list[Any]]:
    """``WHERE`` body over ``n``/``e``: visibility, graphs, one space, chunks
    whose artifact is alive (wave F; legacy mode keeps its old statement).

    The exact reads (the calibrated null, the graph hop) use this name-based
    space rule; the ANN read uses :func:`chunk_scope_clause` plus the
    registry space filter (``ann.ann_query``)."""
    gate, params = gate_clause(spec)
    exclude, exclude_params = _exclude_clause(spec)
    sql = (
        f"{gate} AND (e.model_name = %s OR e.model_name LIKE %s) "
        f"AND n.content->>'kind' = 'chunk' AND {artifact_alive_predicate('n')}{exclude}"
    )
    return sql, [*params, model, same_space_pattern(model), *exclude_params]


def chunk_scope_clause(spec: RetrieveSpec) -> tuple[str, list[Any]]:
    """:func:`scope_clause` without its model filter: the ANN read adds the
    space's index predicate instead (ADR-083 D-2, slice KN-S6)."""
    gate, params = gate_clause(spec)
    exclude, exclude_params = _exclude_clause(spec)
    sql = (
        f"{gate} AND n.content->>'kind' = 'chunk' "
        f"AND {artifact_alive_predicate('n')}{exclude}"
    )
    return sql, [*params, *exclude_params]


# ── the lane ────────────────────────────────────────────────────────────────


def as_content(raw: Any) -> dict[str, Any]:
    """A JSONB value (text or already decoded) as a dict."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _candidate(node_id: str, content_text: Any, distance: Any) -> Candidate:
    content = as_content(content_text)
    index = content.get("chunk_index")
    return Candidate(
        node_id=str(node_id),
        artifact_id=str(content.get("artifact_id") or ""),
        chunk_index=index if isinstance(index, int) else 0,
        content=content,
        cosine=1.0 - float(distance),
    )


def _legacy_sql(spec: RetrieveSpec) -> str:
    gate = visibility_predicate("n", include_ambient=spec.include_ambient)
    tail = _LEGACY_TAIL.format(gate=gate)
    return (
        "SELECT e.node_id, n.content::text, (e.embedding <=> %s::vector) AS distance "
        + tail
    )


def _legacy_mode(spec: RetrieveSpec) -> bool:
    return (
        spec.threshold is not None
        and spec.graph_ids is None
        and not spec.exclude_graph_ids
    )


def _legacy_rows(
    pool: Any, vector: Sequence[float], model: str, spec: RetrieveSpec, limit: int
) -> list[tuple]:
    """The pre-ADR-093 K3 statement, unchanged (acceptance receipt 4)."""
    params = [
        vector_literal(vector),
        spec.project_id,
        model,
        same_space_pattern(model),
        limit,
    ]
    return query_rows(pool, _legacy_sql(spec), params)


#: The dense lane's ANN columns: ``(node_id, content, distance)`` rows.
_DENSE_COLUMNS = (("e.node_id", "node_id"), ("n.content::text", "content"))


def _ann_rows(
    pool: Any,
    vector: Sequence[float],
    space: EmbeddingSpace,
    spec: RetrieveSpec,
    limit: int,
) -> list[tuple]:
    """``(node_id, content, distance)`` rows for one query vector: the space's
    HNSW pool re-sorted by exact distance (ADR-083 D-2, ``ann.ann_query``)."""
    where, params = chunk_scope_clause(spec)
    sql, args = ann.ann_query(
        space,
        columns=_DENSE_COLUMNS,
        where=where,
        where_params=params,
        vector=vector,
        k=limit,
    )
    return query_rows(pool, sql, args)


def _scope_count(
    pool: Any, space: EmbeddingSpace, spec: RetrieveSpec
) -> tuple[int, str | None]:
    """The scope's vector count in ``space`` and its latest ``updated_at``
    (any re-embed moves it: the calibrated null's cache key)."""
    where, params = chunk_scope_clause(spec)
    sql, args = ann.count_query(space, where=where, where_params=params)
    rows = query_rows(pool, sql, args)
    if not rows:
        return 0, None
    stamp = rows[0][1] if len(rows[0]) > 1 else None
    return int(rows[0][0] or 0), None if stamp is None else str(stamp)


def _merge_variant(
    found: dict[str, Candidate], rows: list[tuple], variant: int, multi: bool
) -> None:
    for rank, row in enumerate(rows, start=1):
        fresh = _candidate(row[0], row[1], row[2])
        candidate = found.setdefault(fresh.node_id, fresh)
        candidate.cosine = max(candidate.cosine or -1.0, fresh.cosine or -1.0)
        candidate.lane_scores[LANE] = candidate.cosine
        candidate.lane_ranks[LANE] = min(candidate.lane_ranks.get(LANE, rank), rank)
        if multi:
            candidate.lane_scores[f"{LANE}#{variant}"] = fresh.cosine or 0.0
            candidate.lane_ranks[f"{LANE}#{variant}"] = rank


class _Scan:
    """What one dense scan found, over every query variant."""

    def __init__(self) -> None:
        self.found: dict[str, Candidate] = {}
        self.in_scope: int | None = None
        self.stamp: str | None = None
        self.space: EmbeddingSpace | None = None

    def merge(self, rows: list[tuple], variant: int, multi: bool) -> None:
        _merge_variant(self.found, rows, variant, multi)


def _scan(
    pool: Any, vectors: list[list[float]], model: str, spec: RetrieveSpec, limit: int
) -> _Scan:
    scan, multi = _Scan(), len(vectors) > 1
    if _legacy_mode(spec):
        for variant, vector in enumerate(vectors):
            rows = _legacy_rows(pool, vector, model, spec, limit)
            if variant == 0 and not rows:
                scan.in_scope = 0  # the old statement's "nothing in scope"
            scan.merge(rows, variant, multi)
        return scan
    scan.space = ann.query_space(pool, model)
    scan.in_scope, scan.stamp = _scope_count(pool, scan.space, spec)
    for variant, vector in enumerate(vectors):
        scan.merge(_ann_rows(pool, vector, scan.space, spec, limit), variant, multi)
    return scan


def _failure_reason(exc: Exception) -> str:
    """A named refusal keeps its code (``knowledge.embedding_space_*``)."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code.startswith("knowledge."):
        return f"dense lane refused ({code})"
    if str(exc).startswith(ann.CODE_DIMENSION):
        return f"dense lane refused ({ann.CODE_DIMENSION})"
    return f"dense lane failed ({type(exc).__name__})"


def _scan_info(scan: _Scan, model: str, variants: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "in_scope": scan.in_scope,
        "space": embedding_space(model),
        "variants": variants,
        "stamp": scan.stamp,
    }
    if scan.space is not None:
        info["space_id"] = scan.space.space_id
        info["index"] = scan.space.index_name
    return info


def dense_lane(
    pool: Any,
    vectors: list[list[float]],
    model: str,
    spec: RetrieveSpec,
    *,
    limit: int,
) -> LaneResult:
    """One lane over every query variant, best cosine first.

    A candidate's ``cosine`` is its best over the variants; ``lane_ranks``
    carries ``dense#i`` per variant when there are several, so the fusion
    stage can RRF the variant rankings. ``info["in_scope"]`` is the number of
    vectors in scope in the query's space (``None`` in legacy mode, which
    keeps the old statement exactly).

    Outside legacy mode the query model must resolve to one registered space
    (``ann.query_space``) and every variant is one ANN read on that space's
    ``hnsw_<space_id>`` (ADR-083 D-2); an unregistered model or a vector of
    the wrong dimension is an ``unavailable`` lane with the refusal's code.
    """
    started = time.perf_counter()
    try:
        scan = _scan(pool, vectors, model, spec, limit)
    except Exception as exc:  # noqa: BLE001 — a lane reports, it never raises
        logger.warning("dense lane failed: %s", exc)
        return LaneResult(
            name=LANE,
            status="unavailable",
            reason=_failure_reason(exc),
            ms=(time.perf_counter() - started) * 1000.0,
        )
    ordered = sorted(
        scan.found.values(), key=lambda c: (c.lane_ranks[LANE], -(c.cosine or 0.0))
    )
    return LaneResult(
        name=LANE,
        candidates=ordered,
        ms=(time.perf_counter() - started) * 1000.0,
        info=_scan_info(scan, model, len(vectors)),
    )


def cosines_for(
    pool: Any, node_ids: Sequence[str], vectors: list[list[float]], model: str
) -> dict[str, float]:
    """Cosines of named nodes (lexical-only finds) in ONE statement: the best
    over the query variants, in the same space as the dense lane."""
    if not node_ids or not vectors:
        return {}
    nearest = ", ".join(["e.embedding <=> %s::vector"] * len(vectors))
    sql = (
        f"SELECT e.node_id, LEAST({nearest}) FROM knowledge_node_embeddings e "
        "WHERE e.node_id = ANY(%s) AND (e.model_name = %s OR e.model_name LIKE %s)"
    )
    params = [
        *(vector_literal(v) for v in vectors),
        list(node_ids),
        model,
        same_space_pattern(model),
    ]
    return {
        str(node): 1.0 - float(dist) for node, dist in query_rows(pool, sql, params)
    }


# ── the query encoder (the rag service's POST /rag/embed) ────────────────────


def _embed_batch(
    base: str, texts: list[str], timeout: float
) -> tuple[str, list[list[float]]] | None:
    """``(model, vectors)`` from the batch form, or ``None`` on a 422 (an
    older service that knows only ``{"text"}``)."""
    response = httpx.post(f"{base}/rag/embed", json={"texts": texts}, timeout=timeout)
    if response.status_code == 422:
        return None
    response.raise_for_status()
    body = response.json()
    return str(body["model"]), [[float(v) for v in vec] for vec in body["vectors"]]


def _embed_each(
    base: str, texts: list[str], timeout: float
) -> tuple[str, list[list[float]]]:
    model, vectors = "", []
    for text in texts:
        response = httpx.post(f"{base}/rag/embed", json={"text": text}, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        model = str(body["model"])
        vectors.append([float(v) for v in body["vector"]])
    return model, vectors


def http_batch_encoder(base_url: str, timeout: float = 5.0) -> BatchEncoder:
    """A :data:`BatchEncoder` over the rag service (R6.1: the service owns the
    model and its query prefix; the core only ranks vectors). One request for
    all texts; one request per text when the service predates ``texts``."""
    base = (base_url or "").strip().rstrip("/")

    def encode(texts: Sequence[str]) -> tuple[str, list[list[float]] | None, str]:
        if not base:
            return "", None, "no query encoder configured"
        batch = [str(t) for t in texts]
        try:
            got = _embed_batch(base, batch, timeout) or _embed_each(
                base, batch, timeout
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("query encoder at %s unavailable: %s", base, exc)
            return "", None, f"query encoder unavailable ({type(exc).__name__})"
        model, vectors = got
        if len(vectors) != len(batch):
            return (
                model,
                None,
                f"query encoder returned {len(vectors)} vectors for {len(batch)} texts",
            )
        return model, vectors, ""

    return encode
