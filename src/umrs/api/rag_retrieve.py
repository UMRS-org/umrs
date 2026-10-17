"""The retrieve verb: ``GET /umrs/rag/retrieve`` and its fetch handle (ADR-093 D-2).

Reached as ``/gw/rag/retrieve`` and ``/gw/rag/passages/{node_id}`` through
the gateway's existing rewrite. Both are GET so they work over the D-42
read-only rail a sandbox reaches UMRS through.

This module is the api half of the retrieval core: it authenticates
(``_require_rag_scope(write=False)``), resolves graph selection under the
KG-05 rules (``knowledge_query._resolve_selected_graphs`` with
``_access.graph_meta_readable``), builds the encoder and reranker the core
calls, and hands ``umrs.knowledge.retrieval.pipeline.retrieve`` a scoped
:class:`RetrieveSpec`. The JSON is ``RetrieveResult.to_dict()``.

Sized by the reading model (ADR-093 D-14): ``profile`` and
``reader_model`` (R4 sends its session model) pick the profile that supplies
the defaults for ``k_max``, ``pool``, ``merge`` and the partial cap
(``knowledge.retrieval.profiles``); explicit parameters still win. The graph
bridge lane's ``auto`` gate is resolved here from the site's extraction model
(the registry), because the knowledge layer never reads the registry.

No chat model is called on either route (acceptance receipt 3).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from fastapi import APIRouter, Query, Request

from umrs.knowledge.access_scope import VISIBILITY_AMBIENT, VISIBILITY_SYSTEM
from umrs.knowledge.retrieval import calibrate, graph_lane, pipeline, profiles
from umrs.knowledge.retrieval.dense import http_batch_encoder, query_rows
from umrs.knowledge.retrieval.expand import (
    chunk_range,
    is_full,
    neighbour_id,
    union_text,
)
from umrs.knowledge.retrieval.hydrate import hydrate
from umrs.knowledge.retrieval.types import (
    BatchEncoder,
    Candidate,
    Reranker,
    RetrieveSpec,
)

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

_STAGE = "^(auto|on|off)$"
_CUTOFF = "^(adaptive|fixed)$"
_PROFILE = "^(" + "|".join(profiles.PROFILE_CHOICES) + ")$"
READER_MODEL_CHARS = 200
QUERY_CHARS_MAX = 2048
VARIANTS_MAX = 3
LEGACY_TOP_K = 5  # the pre-ADR-093 /rag/query default, for threshold mode

_NODE_SQL = (
    "SELECT n.project_id, n.graph_id, n.visibility, n.content->>'chunk_index', "
    "n.content->>'artifact_id' "
    "FROM knowledge_nodes n WHERE n.node_id = %s AND n.content->>'kind' = 'chunk'"
)
# A neighbour id is string arithmetic on the centre's id, and any caller can
# import a node under a chosen id: only chunks of the centre's own project,
# graph, tier and artifact may be stitched to it (review of ADR-093, 2026-10-10).
_NEIGHBOURS_SQL = (
    "SELECT n.node_id FROM knowledge_nodes n "
    "WHERE n.node_id = ANY(%s) AND n.content->>'kind' = 'chunk' "
    "AND n.project_id IS NOT DISTINCT FROM %s "
    "AND n.graph_id IS NOT DISTINCT FROM %s "
    "AND n.visibility IS NOT DISTINCT FROM %s "
    "AND n.content->>'artifact_id' IS NOT DISTINCT FROM %s"
)
# Every graph holding rows of one project (the whole-project scope's graphs).
_PROJECT_GRAPHS_SQL = (
    "SELECT DISTINCT n.graph_id FROM knowledge_nodes n "
    "WHERE n.project_id = %s AND n.graph_id <> ''"
)


# ── what the core is handed ─────────────────────────────────────────────────


def batch_encoder() -> BatchEncoder:
    """The rag service's ``POST /rag/embed`` (R6.1: it owns the model), one
    per request. Each batch also carries the null's encoding probe, so the
    null cache sees a changed query encoding (``UMRS_RAG_QUERY_PREFIX``, a
    re-staged encoder) with no extra call (``calibrate.FingerprintedEncoder``)."""
    from .knowledge_query import _query_encoder_timeout, _query_encoder_url

    return calibrate.FingerprintedEncoder(
        http_batch_encoder(_query_encoder_url(), _query_encoder_timeout())
    )


def batch_encode(texts: Sequence[str]) -> tuple[str, list[list[float]] | None, str]:
    """One call of :func:`batch_encoder` (K3's null calibration reads this)."""
    return batch_encoder()(texts)


def site_reranker() -> Reranker | None:
    """The rag service's ``POST /rag/rerank`` when configured, else ``None``
    (the lane then reports ``unavailable`` if asked to run)."""
    try:
        from umrs.knowledge.retrieval.rerank import reranker_from_env
    except ImportError:
        return None
    return reranker_from_env()


def _env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        value = default
    return min(high, max(low, value))


def site_k_max() -> int:
    return _env_int("UMRS_RAG_FINAL_K_MAX", 4, 1, 12)


def site_pool() -> int:
    return _env_int("UMRS_RAG_POOL", 50, 10, 200)


def site_per_artifact() -> int:
    return _env_int("UMRS_RAG_PER_ARTIFACT", 2, 1, 4)


def is_admin(request: Request) -> bool:
    """The signed admin flag only (H-1, 2026-10-18): a forged tuple is a 401
    in strict mode, never an admin (it gates ambient rows)."""
    from . import rag as rag_api

    return rag_api._verified_admin(request)


# ── request → spec ──────────────────────────────────────────────────────────


def clean_queries(raw: list[str]) -> tuple[str, ...]:
    """1-3 non-empty variants, at most 2,048 characters in total."""
    queries = tuple(q.strip() for q in raw if q and q.strip())
    if not 1 <= len(queries) <= VARIANTS_MAX:
        raise_typed(
            PlatformError.rag(
                "bad_request", f"q takes 1-{VARIANTS_MAX} non-empty variants"
            )
        )
    if sum(len(q) for q in queries) > QUERY_CHARS_MAX:
        raise_typed(
            PlatformError.rag(
                "bad_request", f"q is limited to {QUERY_CHARS_MAX} characters in total"
            )
        )
    return queries


def _k_max(k_max: int | None, threshold: float | None, sized: int) -> int:
    if k_max is not None:
        return k_max
    return LEGACY_TOP_K if threshold is not None else sized


def profile_fields(
    reader_model: str | None, params: Mapping[str, Any]
) -> dict[str, Any]:
    """The spec fields the reading model's profile sizes (D-14).

    ``params`` may carry the request's ``profile``, ``k_max``, ``threshold``,
    ``pool`` and ``merge``; each explicit value wins. The small profile keeps
    the site's env defaults; a larger one never goes below them.
    """
    reader = (reader_model or "").strip()
    name, profile = profiles.resolve_profile(reader or None, params.get("profile"))
    sized_k = profiles.sized(name, site_k_max(), profile.k_max)
    return {
        "k_max": _k_max(params.get("k_max"), params.get("threshold"), sized_k),
        "pool": params.get("pool") or profiles.sized(name, site_pool(), profile.pool),
        "merge": profiles.merge_mode(params.get("merge"), profile),
        # A named reader's profile decides lexical inclusion; a request naming
        # no reader (an older client) keeps the site's UMRS_RAG_LEXICAL.
        "lexical": profiles.lexical_mode(params.get("lexical"), profile)
        if reader
        else (params.get("lexical") or "auto"),
        "partial_k": profile.partial_k,
        "profile": name,
        "reader_model": reader,
    }


def extraction_model(project_id: str) -> str | None:
    """The site's extraction model: the registry's chat model for the
    project (``_extract_chat_endpoint``, as ingest resolves it)."""
    from umrs.jobs.knowledge_adapters import _extract_chat_endpoint

    try:
        target = _extract_chat_endpoint(project_id)
    except Exception as exc:  # noqa: BLE001 — no model means the gate stays off
        logger.warning("extraction model lookup failed: %s", exc)
        return None
    return target[1] if target else None


def graph_fields(
    project_id: str, model: Callable[[], str | None] | None = None
) -> dict[str, str]:
    """``graph_lane``/``graph_gate`` for a spec: ``UMRS_RAG_GRAPH_LANE``,
    whose ``auto`` reads the extraction model (only then is it looked up)."""
    lane, gate = graph_lane.resolve_gate(
        model or (lambda: extraction_model(project_id))
    )
    return {"graph_lane": lane, "graph_gate": gate}


def build_spec(
    queries: tuple[str, ...], project_id: str, params: dict[str, Any]
) -> RetrieveSpec:
    """The spec before graph resolution (graphs are resolved off the loop)."""
    return RetrieveSpec(
        queries=queries,
        project_id=project_id,
        include_ambient=bool(params.get("include_ambient")),
        per_artifact=params.get("per_artifact") or site_per_artifact(),
        rerank=params.get("rerank") or "auto",
        cutoff=params.get("cutoff") or "adaptive",
        threshold=params.get("threshold"),
        trace=bool(params.get("trace")),
        **profile_fields(params.get("reader_model"), params),
    )


def graph_hidden(request: Request, db: Any, graph_id: str, project_id: str) -> bool:
    """A registered graph of the project's rows that the caller may not read.

    Rows of the caller's project are visible under the project rule — except
    those of a graph the caller may not read (another member's personal
    graph: ``scope_readable`` makes it owner-only). Legacy (``''``) and
    default-graph rows, and graphs the registry does not know, keep the
    project rule.
    """
    from umrs.knowledge.graph_registry import GraphRegistryStore, graph_id_is_default

    from ._access import graph_meta_readable
    from ._scope_identity import request_caller_scope

    if not graph_id or graph_id_is_default(graph_id, project_id):
        return False
    graph = GraphRegistryStore(db).get_graph(graph_id)
    return graph is not None and not graph_meta_readable(
        request, graph, request_caller_scope(request)
    )


def hidden_graphs(request: Request, db: Any, project_id: str) -> tuple[str, ...]:
    """The graphs a whole-project read leaves out for this caller (KG-05).

    Naming such a graph is refused (``_resolve_selected_graphs``); leaving
    ``graphs`` out must not read it either (review of ADR-093, 2026-10-10).
    Registry reads only — never the default-graph write.
    """
    rows = query_rows(db, _PROJECT_GRAPHS_SQL, (project_id,))
    graph_ids = sorted({str(row[0]) for row in rows if row and row[0]})
    return tuple(g for g in graph_ids if graph_hidden(request, db, g, project_id))


def resolve_graphs(
    request: Request,
    db: Any,
    spec: RetrieveSpec,
    graphs: str | None,
    exclude_graphs: str | None,
) -> RetrieveSpec:
    """KG-05: named graphs the caller may read; exclusions win. No ``graphs``
    means the whole project (the legacy scope, and no default-graph write on
    a read path) minus the exclusions and the graphs the caller may not read."""
    from umrs.knowledge.graph_registry import GraphRegistryStore

    from ._access import graph_meta_readable
    from ._scope_identity import request_caller_scope
    from .knowledge_query import (
        _resolve_exclusions,
        _resolve_selected_graphs,
        _split_csv,
    )

    excluded: set[str] = set()
    if _split_csv(exclude_graphs):
        excluded, _names = _resolve_exclusions(GraphRegistryStore(db), exclude_graphs)
    if not _split_csv(graphs):
        excluded |= set(hidden_graphs(request, db, spec.project_id))
        if not excluded:
            return spec
        return dataclasses.replace(spec, exclude_graph_ids=tuple(sorted(excluded)))
    scope = request_caller_scope(request)
    selected, _excluded = _resolve_selected_graphs(
        GraphRegistryStore(db),
        spec.project_id,
        graphs,
        exclude_graphs,
        lambda graph: graph_meta_readable(request, graph, scope),
    )
    return dataclasses.replace(
        spec,
        graph_ids=tuple(g.graph_id for g in selected),
        exclude_graph_ids=tuple(sorted(excluded)),
    )


def run_retrieve(
    request: Request,
    db: Any,
    spec: RetrieveSpec,
    graphs: str | None,
    exclude_graphs: str | None,
) -> dict[str, Any]:
    """Synchronous (DB, HTTP to the rag service): the route runs it in a thread."""
    from .rag_spine import _artifact_fetcher

    scoped = dataclasses.replace(
        resolve_graphs(request, db, spec, graphs, exclude_graphs),
        **graph_fields(spec.project_id),
    )
    result = pipeline.retrieve(
        db,
        scoped,
        encode=batch_encoder(),
        rerank=site_reranker(),
        fetch_bytes=_artifact_fetcher(),
    )
    return result.to_dict()


def _db(request: Request) -> Any:
    db = getattr(request.app.state, "umrs_db_pool", None)
    if db is None:
        raise_typed(PlatformError.rag("unavailable", "knowledge database unavailable"))
    return db


async def retrieve_passages(
    request: Request,
    q: list[str] = Query(..., description="The question; repeat for up to 3 variants"),
    project_id: str | None = Query(None),
    graphs: str | None = Query(None, description="Comma-separated graph ids/names"),
    exclude_graphs: str | None = Query(None),
    k_max: int | None = Query(None, ge=1, le=12),
    pool_size: int | None = Query(None, alias="pool", ge=10, le=200),
    per_artifact: int | None = Query(None, ge=1, le=4),
    lexical: str = Query("auto", pattern=_STAGE),
    rerank: str = Query("auto", pattern=_STAGE),
    merge: str = Query("auto", pattern=_STAGE),
    cutoff: str = Query("adaptive", pattern=_CUTOFF),
    threshold: float | None = Query(
        None, ge=0.0, le=1.0, description="Legacy absolute mode"
    ),
    trace: int = Query(0, ge=0, le=1),
    profile: str = Query(
        "auto", pattern=_PROFILE, description="Retrieval profile (ADR-093 D-14)"
    ),
    reader_model: str | None = Query(
        None,
        max_length=READER_MODEL_CHARS,
        description="The model that will read the passages; sizes the profile",
    ),
) -> dict[str, Any]:
    """Score, cut and judge passages for a question — no model generation.

    ``threshold`` set is the legacy absolute mode (top ``k``, stages off).
    ``profile``/``reader_model`` size the defaults (D-14); explicit knobs win.
    # auth: verified
    """
    from . import rag as rag_api

    _user, verified = rag_api._require_rag_scope(request, project_id, write=False)
    queries = clean_queries(q)
    db = _db(request)
    params = {
        "include_ambient": is_admin(request),
        "k_max": k_max,
        "pool": pool_size,
        "per_artifact": per_artifact,
        "lexical": lexical,
        "rerank": rerank,
        "merge": merge,
        "cutoff": cutoff,
        "threshold": threshold,
        "trace": trace,
        "profile": profile,
        "reader_model": reader_model,
    }
    spec = build_spec(queries, project_id or verified, params)
    return await asyncio.to_thread(
        run_retrieve, request, db, spec, graphs, exclude_graphs
    )


# ── the fetch handle ────────────────────────────────────────────────────────


def _graph_readable(request: Request, db: Any, graph_id: str) -> bool:
    from umrs.knowledge.graph_registry import GraphRegistryStore

    from ._access import graph_meta_readable
    from ._scope_identity import request_caller_scope

    graph = GraphRegistryStore(db).get_graph(graph_id) if graph_id else None
    return graph is not None and graph_meta_readable(
        request, graph, request_caller_scope(request)
    )


def row_graph_id(owner: str, graph_id: str) -> str:
    """The graph a row belongs to: a legacy ``graph_id = ''`` row is its
    project's DEFAULT graph. The Python twin of the retrieve verb's SQL rule
    (``dense._GRAPH_OWNED`` with ``DEFAULT_GRAPH_SQL``, pinned to
    ``default_graph_id`` by a unit test), so the handle serves exactly the
    legacy rows the verb returns (review of ADR-093, 2026-10-10)."""
    from umrs.knowledge.graph_registry import default_graph_id

    if graph_id or not owner:
        return graph_id
    return default_graph_id(owner)


def node_visible(request: Request, db: Any, project_id: str, row: tuple) -> bool:
    """The retrieve verb's predicate for one node: the caller's project (no
    ambient row unless admin, none of a graph the caller may not read), a
    system row, or a graph the caller may read — for another project's
    legacy row, that project's default graph (:func:`row_graph_id`)."""
    owner, graph_id, visibility = (
        str(row[0] or ""),
        str(row[1] or ""),
        str(row[2] or ""),
    )
    if visibility == VISIBILITY_AMBIENT and not is_admin(request):
        return False
    if visibility == VISIBILITY_SYSTEM:
        return True
    if owner == project_id:
        return not graph_hidden(request, db, graph_id, project_id)
    return _graph_readable(request, db, row_graph_id(owner, graph_id))


def _window(center: Candidate, window: int) -> list[Candidate]:
    ids = [
        (i, neighbour_id(center, i))
        for i in range(center.chunk_index - window, center.chunk_index + window + 1)
    ]
    return [
        center if i == center.chunk_index else Candidate(node_id=nid, chunk_index=i)
        for i, nid in ids
        if nid
    ]


def _scoped(
    db: Any, row: tuple, center: Candidate, pieces: list[Candidate]
) -> list[Candidate]:
    """The centre plus the neighbours that share its project, graph, tier and
    artifact — the predicate the centre passed, nothing wider."""
    others = [c.node_id for c in pieces if c is not center]
    if not others:
        return [center]
    owner, graph_id, visibility, _index, artifact_id = row[:5]
    params = (others, owner, graph_id, visibility, artifact_id)
    allowed = {str(r[0]) for r in query_rows(db, _NEIGHBOURS_SQL, params)}
    return [c for c in pieces if c is center or c.node_id in allowed]


def _passage_body(
    center: Candidate, pieces: list[Candidate]
) -> tuple[str, tuple[int, int] | None, list[int]]:
    """Text over the window when every piece is verified, else the centre alone."""
    present = [c for c in pieces if c.text]
    if len(present) > 1 and all(is_full(c) for c in present):
        span, text = union_text(present)
        return (
            text,
            span,
            [min(c.chunk_index for c in present), max(c.chunk_index for c in present)],
        )
    return center.text, center.span, list(chunk_range(center))


def fetch_passage(
    request: Request,
    db: Any,
    project_id: str,
    node_id: str,
    window: int,
    max_chars: int,
) -> dict[str, Any] | None:
    """One chunk (and up to ``window`` neighbours each side), or ``None`` when
    the caller may not see it — existence is not disclosed."""
    from .rag_spine import _artifact_fetcher

    rows = query_rows(db, _NODE_SQL, (node_id,))
    if not rows or not node_visible(request, db, project_id, rows[0]):
        return None
    index = rows[0][3]
    center = Candidate(
        node_id=node_id, chunk_index=int(index) if str(index or "").isdigit() else 0
    )
    pieces = _scoped(db, rows[0], center, _window(center, window))
    hydrate(db, pieces, _artifact_fetcher())
    text, span, covered = _passage_body(center, pieces)
    return {
        "source_id": node_id,
        "artifact_id": center.artifact_id,
        "source_name": center.source_name,
        "text": text[:max_chars],
        "span": list(span) if span else None,
        "window": window,
        "chunks": covered,
        "text_source": center.text_source,
    }


async def get_passage(
    request: Request,
    node_id: str,
    project_id: str | None = Query(None),
    window: int = Query(0, ge=0, le=2),
    max_chars: int = Query(4000, ge=1, le=4000),
) -> dict[str, Any]:
    """The full text of one retrieved passage (the retrieve verb's handle).
    # auth: verified
    """
    from . import rag as rag_api

    _user, verified = rag_api._require_rag_scope(request, project_id, write=False)
    db = _db(request)
    found = await asyncio.to_thread(
        fetch_passage, request, db, project_id or verified, node_id, window, max_chars
    )
    if found is None:
        raise_typed(PlatformError.rag("not_found", "passage not found"))
    return found


def register(router: APIRouter) -> None:
    """Mount both routes on the rag proxy router (``/umrs/rag`` prefix), as
    its own routes so the route inventory's auth audit sees them."""
    router.add_api_route("/retrieve", retrieve_passages, methods=["GET"])
    router.add_api_route("/passages/{node_id}", get_passage, methods=["GET"])
