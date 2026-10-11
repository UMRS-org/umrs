"""The dense lane of the retrieval core, and the query encoder it reads (ADR-093 D-2).

Exact cosine (``<=>``) over the 0075 node vectors, in ONE embedding space
(the basename rule of ``knowledge_query._similarity_hits``, ADR-083 D-2),
chunk nodes only, under the ADR-050 visibility predicate, plus a graph
predicate when the caller selected graphs. A pool of ``limit`` candidates per
query variant — larger than the ``k`` the cut keeps (ADR-083 D-2's
over-fetch shape).

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
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.text_embedder import embedding_space

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
    whose artifact is alive (wave F; legacy mode keeps its old statement)."""
    gate, params = gate_clause(spec)
    exclude, exclude_params = _exclude_clause(spec)
    sql = (
        f"{gate} AND (e.model_name = %s OR e.model_name LIKE %s) "
        f"AND n.content->>'kind' = 'chunk' AND {artifact_alive_predicate('n')}{exclude}"
    )
    return sql, [*params, model, same_space_pattern(model), *exclude_params]


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


def _variant_rows(
    pool: Any, vector: Sequence[float], model: str, spec: RetrieveSpec, limit: int
) -> list[tuple]:
    """``(node_id, content, distance, in_scope, stamp)`` rows for one query
    vector; ``stamp`` is the scope's latest ``updated_at`` (the null key)."""
    if (
        spec.threshold is not None
        and spec.graph_ids is None
        and not spec.exclude_graph_ids
    ):
        params = [
            vector_literal(vector),
            spec.project_id,
            model,
            same_space_pattern(model),
            limit,
        ]
        return [(*row, None) for row in query_rows(pool, _legacy_sql(spec), params)]
    where, params = scope_clause(spec, model)
    sql = (
        "SELECT e.node_id, n.content::text, (e.embedding <=> %s::vector) AS distance, "
        "count(*) OVER () AS in_scope, max(e.updated_at) OVER () AS stamp "
        f"{_FROM}WHERE {where} "
        "ORDER BY distance ASC, e.node_id LIMIT %s"
    )
    return query_rows(pool, sql, [vector_literal(vector), *params, limit])


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


def _in_scope(rows: list[tuple]) -> int | None:
    if not rows:
        return 0
    value = rows[0][3]
    return int(value) if value is not None else None


def _stamp(rows: list[tuple]) -> str | None:
    """The scope's latest ``updated_at`` (any re-embed moves it), as text."""
    value = rows[0][4] if rows and len(rows[0]) > 4 else None
    return None if value is None else str(value)


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
    vectors the scan ranked (``None`` in legacy mode, which keeps the old
    statement exactly).
    """
    started = time.perf_counter()
    found: dict[str, Candidate] = {}
    in_scope: int | None = 0
    stamp: str | None = None
    try:
        for variant, vector in enumerate(vectors):
            rows = _variant_rows(pool, vector, model, spec, limit)
            if variant == 0:
                in_scope, stamp = _in_scope(rows), _stamp(rows)
            _merge_variant(found, rows, variant, len(vectors) > 1)
    except Exception as exc:  # noqa: BLE001 — a lane reports, it never raises
        logger.warning("dense lane failed: %s", exc)
        return LaneResult(
            name=LANE,
            status="unavailable",
            reason=f"dense lane failed ({type(exc).__name__})",
            ms=(time.perf_counter() - started) * 1000.0,
        )
    ordered = sorted(
        found.values(), key=lambda c: (c.lane_ranks[LANE], -(c.cosine or 0.0))
    )
    return LaneResult(
        name=LANE,
        candidates=ordered,
        ms=(time.perf_counter() - started) * 1000.0,
        info={
            "in_scope": in_scope,
            "space": embedding_space(model),
            "variants": len(vectors),
            "stamp": stamp,
        },
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
