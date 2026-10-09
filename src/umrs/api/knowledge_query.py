"""K3: unified knowledge query facade (read-side, additive).

Route: ``GET /umrs/knowledge/query``

One search entry across UMRS knowledge backends, replacing per-route
fan-out by agents/R4 (catalog / knowledge / provenance / artifact /
graph). Backends are composed via the ``search_api`` UnifiedSearch
port protocols. v1 wires the catalog metadata search; backends that
are not wired yet (knowledge graph, embeddings, RAG) are reported in
``unavailable_backends`` with reasons instead of silently returning
empty results — the response is honest about coverage.

WP-15 (2026-09-07, ADR-027 P1b facade leg): the ``rag`` scope is now
wired over the NODE-KEYED central corpus (knowledge_nodes + 0075
embeddings) — never the endpoint-coupled mini-RiVault corpus. Hits
carry ``metadata.corpus_venue: central-pg`` so callers see the venue.

Existing subsystem routes stay untouched (aliases per K3 guardrail).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Query, Request

from umrs.knowledge.access_scope import visibility_predicate
from umrs.knowledge.graph_registry import (
    GraphRegistryStore,
    KnowledgeGraphMeta,
    graph_id_is_default,
)
from umrs.search_api import SearchScope, UnifiedSearch

from ._access import graph_meta_readable
from ._scope_identity import request_caller_scope
from .catalog_store import get_optional_catalog_meta_store
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-query"])

_VALID_SCOPES = {scope.value for scope in SearchScope}

_UNWIRED_BACKENDS: dict[str, str] = {
    "embeddings": "query-embedding generation requires the embedding service",
    "rag": "central knowledge corpus unavailable (knowledge nodes not present)",
}


def _hit_to_dict(hit: Any, *, project_id: str) -> dict[str, Any]:
    """Map a CatalogHit to the ``search_api`` conversion dict shape."""
    meta = hit.meta
    return {
        "title": meta.s3_key.rsplit("/", 1)[-1],
        "content": meta.summary or meta.deterministic_summary or "",
        "score": float(hit.score),
        "source_id": meta.s3_key,
        "source_type": meta.detected_modality or "object",
        "metadata": {
            "owner": meta.owner,
            "tags": list(meta.tags),
            "acl_scope": meta.acl_scope,
            "project_id": project_id,
        },
    }


class _CatalogSearchBackend:
    """``search_api`` CatalogSearchable port over the live catalog store."""

    def __init__(self, meta_store: Any, *, include_ambient: bool = False) -> None:
        self._store = meta_store
        self._include_ambient = include_ambient

    def search_catalog(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if not project_id:
            # Catalog ACL is project-scoped; the facade requires it anyway.
            return []
        from umrs.data_catalog.query import search_catalog

        hits = search_catalog(
            self._store,
            query=query,
            project_id=project_id,
            limit=limit,
            include_ambient=self._include_ambient,
        )
        return [_hit_to_dict(hit, project_id=project_id) for hit in hits]


class _GraphSearchBackend:
    """``search_api`` KnowledgeGraphSearchable port over the canonical store.

    Searches ``knowledge_nodes`` scoped to the selected graphs. Legacy rows
    (``graph_id = ''``) are visible through their project's default graph —
    the same fallback ``KnowledgeStore.list_nodes`` uses.
    """

    def __init__(
        self,
        pool: Any,
        selected: list[KnowledgeGraphMeta],
        *,
        include_ambient: bool = False,
    ) -> None:
        self._include_ambient = include_ambient
        self._pool = pool
        self._selected = selected
        self._by_id = {g.graph_id: g for g in selected}
        # Legacy rows (graph_id '') belong to their project's DEFAULT graph
        # only — the rule graph_content and list_nodes use. Mapping every
        # project-bound graph let a shared non-default graph expose its
        # owner's legacy rows (review of KG-05, 2026-10-09).
        self._default_by_project = {
            g.project_id: g
            for g in selected
            if g.project_id and graph_id_is_default(g.graph_id, g.project_id)
        }

    def search_knowledge_graph(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        graph_ids = [g.graph_id for g in self._selected]
        default_projects = [
            pid
            for pid, meta in self._default_by_project.items()
            if meta.graph_id in graph_ids
        ]
        pattern = f"%{query}%"
        # ADR-050 tier gate rendered from the single predicate; ownership
        # half is the graph selection (explicit graphs + legacy '' rows).
        gate = visibility_predicate(
            owned_sql="(graph_id = ANY(%s) OR (graph_id = '' AND project_id = ANY(%s)))",
            include_ambient=self._include_ambient,
        )
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT node_id, project_id, label, node_type, content, graph_id "
                        "FROM knowledge_nodes "
                        f"WHERE {gate} "
                        "AND (label ILIKE %s OR content::text ILIKE %s) "
                        "ORDER BY (label ILIKE %s) DESC, node_id LIMIT %s",
                        (graph_ids, default_projects, pattern, pattern, pattern, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(query, row) for row in rows]

    def _hit(self, query: str, row: tuple) -> dict[str, Any]:
        node_id, project_id, label, node_type, content, graph_id = row
        meta = self._by_id.get(graph_id) or self._default_by_project.get(project_id)
        title = label or node_id
        return {
            "title": title,
            "content": str(content)[:200] if content else "",
            "score": 0.9 if query.lower() in title.lower() else 0.6,
            "source_id": node_id,
            "source_type": "knowledge_graph_node",
            "metadata": {
                "graph_id": graph_id or (meta.graph_id if meta else ""),
                "graph_name": meta.name if meta else "",
                "node_type": node_type,
                "project_id": project_id,
            },
        }


def _resolve_exclusions(
    registry: GraphRegistryStore, exclude_csv: str | None
) -> tuple[set[str], list[str]]:
    """Map exclude tokens to graph ids (unknown tokens only recorded by name)."""
    excluded_ids: set[str] = set()
    excluded_names: list[str] = []
    for token in _split_csv(exclude_csv):
        # Every graph the token can name: includes pick the first readable
        # candidate, so excluding only one would let a name slip through and
        # reveal whether another project holds a graph of that name.
        excluded_ids.update(g.graph_id for g in registry.graph_candidates(token))
        excluded_names.append(token)
    return excluded_ids, excluded_names


GraphReadable = Callable[[KnowledgeGraphMeta], bool]


def _resolve_includes(
    registry: GraphRegistryStore,
    tokens: list[str],
    excluded_ids: set[str],
    readable: GraphReadable,
) -> list[KnowledgeGraphMeta]:
    """Resolve include tokens, dropping excluded, duplicate and unreadable graphs.

    KG-05: ``resolve_graph`` matches an id, or a name across every project
    (a default graph is named after its project), so a caller could read
    another project's nodes by naming its graph. A graph the caller may not
    read is skipped exactly like an unknown token: existence is not disclosed.
    """
    selected: list[KnowledgeGraphMeta] = []
    for token in tokens:
        graph = next((g for g in registry.graph_candidates(token) if readable(g)), None)
        if graph is None or graph.graph_id in excluded_ids:
            continue
        if graph.graph_id in {g.graph_id for g in selected}:
            continue
        selected.append(graph)
    return selected


def _resolve_selected_graphs(
    registry: GraphRegistryStore,
    project_id: str,
    graphs_csv: str | None,
    exclude_csv: str | None,
    readable: GraphReadable | None = None,
) -> tuple[list[KnowledgeGraphMeta], list[str]]:
    """Resolve explicit include/exclude tokens into selected graph metas.

    Explicit exclusion beats inclusion. Without explicit inclusion the
    project's default graph is the single candidate (registry discovery /
    candidate ranking lives client-side in the selection policy).
    """
    excluded_ids, excluded_names = _resolve_exclusions(registry, exclude_csv)

    tokens = _split_csv(graphs_csv)
    if tokens:
        # No predicate, no named graphs: fail closed.
        allow = readable or (lambda _graph: False)
        return _resolve_includes(registry, tokens, excluded_ids, allow), excluded_names

    default = registry.ensure_default_graph(project_id)
    selected = [default] if default.graph_id not in excluded_ids else []
    return selected, excluded_names


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


# Query-time text encoder (R6.1): the RAG service owns it, this facade only
# asks. Unset → UMRS_RAG_DIRECT_URL (the same service), else no encoder.
_QUERY_ENCODER_URL_ENV = "UMRS_KNOWLEDGE_QUERY_ENCODER_URL"
_QUERY_ENCODER_TIMEOUT_ENV = "UMRS_KNOWLEDGE_QUERY_ENCODER_TIMEOUT"

# (model_name, vector or None, reason when None)
QueryEncoder = Callable[[str], tuple[str, list[float] | None, str]]


def _query_encoder_url() -> str:
    url = os.environ.get(_QUERY_ENCODER_URL_ENV) or os.environ.get(
        "UMRS_RAG_DIRECT_URL"
    )
    return (url or "").strip().rstrip("/")


def _query_encoder_timeout() -> float:
    try:
        return max(0.5, float(os.environ.get(_QUERY_ENCODER_TIMEOUT_ENV, "5")))
    except ValueError:
        return 5.0


def _encode_query(query: str) -> tuple[str, list[float] | None, str]:
    """Embed a query through the RAG service's ``POST /rag/embed``.

    The service applies the model's query prefix (e5 ``query:``) and names
    the model, so the caller can rank only vectors from that space (ADR-083).
    """
    base = _query_encoder_url()
    if not base:
        return "", None, "no query encoder configured"
    try:
        response = httpx.post(
            f"{base}/rag/embed", json={"text": query}, timeout=_query_encoder_timeout()
        )
        response.raise_for_status()
        body = response.json()
        return str(body["model"]), [float(v) for v in body["vector"]], ""
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        logger.warning("query encoder at %s unavailable: %s", base, exc)
        return "", None, f"query encoder unavailable ({type(exc).__name__})"


class _EmbeddingsSearchBackend:
    """``search_api`` EmbeddingsSearchable port over migration-0075 rows.

    Two honest lanes (hits carry ``metadata.lane``):

    - ``table-reader`` (default): exposes the canonical node-vector rows
      as honest hits — model/dim/content hash + the owning node — filtered
      by the query against node identity and model name, recency-ranked.
    - ``similarity`` (P3, 2026-08-23): real cosine ANN ranking when the
      caller supplies a PRE-COMPUTED ``query_embedding``. Query-time text
      embedding stays owned by
      ``umrs.api.rag_service.PrefixedSentenceTransformerEmbeddings``
      (R6.1 boundary) — this facade never loads a model; it only ranks
      vectors. Scale note: exact scan (`<=>`) is fine well past the
      current corpus size; the typed-shadow-column + HNSW migration is
      the documented follow-up when rows cross ~100k.

    No mock/fake fallback is ever synthesized: without a query vector the
    lane stays table-reader.
    """

    def __init__(
        self,
        pool: Any,
        *,
        include_ambient: bool = False,
        encoder: QueryEncoder | None = None,
    ) -> None:
        self._pool = pool
        self._include_ambient = include_ambient
        self._encoder = encoder
        # Which lane the last search ran and why — reported to the caller.
        self.lane_report: dict[str, str] | None = None

    def search_embeddings(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
        query_embedding: list[float] | None = None,
        model_name: str | None = None,
    ) -> list[dict[str, Any]]:
        if query_embedding:
            return self._similarity_hits(
                query_embedding,
                model_name=model_name,
                project_id=project_id,
                limit=limit,
            )
        if self._encoder is None:
            self.lane_report = {"lane": "table-reader", "reason": "no query encoder"}
            return self._table_reader_hits(query, project_id=project_id, limit=limit)
        model, vector, reason = self._encoder(query)
        if not vector:
            # ADR-083 D-4: an encoder outage is reported, never answered by
            # a lexical match labelled "embeddings".
            self.lane_report = {"lane": "unavailable", "reason": reason}
            return []
        hits = self._similarity_hits(
            vector, model_name=model, project_id=project_id, limit=limit
        )
        self.lane_report = {"lane": "similarity", "model": model}
        if not hits:
            # ADR-083 D-5: a space with no vectors here is no evidence.
            self.lane_report["reason"] = (
                f"no vectors in embedding space {model!r} for this project"
            )
        return hits

    def _table_reader_hits(
        self, query: str, *, project_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        # ADR-050 tier gate on the owning node (R-02 wave 2: this lane was
        # previously ungated — project-only, ambient rows still served).
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT e.node_id, e.model_name, e.dimension, e.created_at, "
                        "n.label, n.project_id "
                        "FROM knowledge_node_embeddings e "
                        "JOIN knowledge_nodes n ON n.node_id = e.node_id "
                        f"WHERE {visibility_predicate('n', include_ambient=self._include_ambient)} "
                        "AND (e.model_name ILIKE %s OR n.label ILIKE %s OR e.node_id ILIKE %s) "
                        "ORDER BY e.created_at DESC LIMIT %s",
                        (project_id, pattern, pattern, pattern, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(row) for row in rows]

    def _similarity_hits(
        self,
        query_embedding: list[float],
        *,
        model_name: str | None,
        project_id: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Cosine-distance ANN over 0075 rows using a pre-computed vector.

        Ranks one embedding space only (ADR-083 D-2): vectors from another
        model are not comparable, and a same-dimension model would mix
        silently instead of failing. A space is the model's basename
        (text_embedder.embedding_space), so the bundled model and a copy staged
        at another path rank together.
        """
        if not model_name:
            raise ValueError("a query vector needs the model_name of its space")
        from umrs.knowledge.text_embedder import embedding_space

        qv = "[" + ",".join(repr(float(v)) for v in query_embedding) + "]"
        # Same space = same model name, or the same model staged elsewhere.
        space = embedding_space(model_name).replace("\\", "\\\\")
        same_space = "%/" + space.replace("%", "\\%").replace("_", "\\_")
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT e.node_id, e.model_name, e.dimension, e.created_at, "
                        "n.label, n.project_id, (e.embedding <=> %s::vector) AS distance, "
                        "n.content::text "
                        "FROM knowledge_node_embeddings e "
                        "JOIN knowledge_nodes n ON n.node_id = e.node_id "
                        f"WHERE {visibility_predicate('n', include_ambient=self._include_ambient)} "
                        "AND (e.model_name = %s OR e.model_name LIKE %s) "
                        "ORDER BY distance ASC LIMIT %s",
                        (qv, project_id, model_name, same_space, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(row) for row in rows]

    @staticmethod
    def _hit(row: tuple) -> dict[str, Any]:
        node_id, model_name, dimension, _created, label, project_id = row[:6]
        content = f"vector row for {node_id} ({model_name}, {dimension}d)"
        if len(row) > 6:  # similarity lane: cosine distance in position 7
            distance = float(row[6])
            score, lane = max(0.0, 1.0 - distance), "similarity"
            extra = {"distance": distance}
            # The matched text, so a semantic hit is readable on its own.
            content = _first_text_field(row[7] if len(row) > 7 else "") or content
        else:
            score, lane, extra = 0.5, "table-reader", {}
        return {
            "title": label or node_id,
            "content": content,
            "score": score,
            "source_id": node_id,
            "source_type": "knowledge_node_embedding",
            "metadata": {
                "model_name": model_name,
                "dimension": dimension,
                "project_id": project_id,
                "lane": lane,
                **extra,
            },
        }


def _embeddings_backend_for(
    pool: Any, *, include_ambient: bool = False, encoder: QueryEncoder | None = None
) -> tuple[_EmbeddingsSearchBackend | None, str | None]:
    """Wire the 0075 backend iff the table exists; honesty reason otherwise."""
    if pool is None:
        return None, "embeddings search requires the database"
    try:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('knowledge_node_embeddings')")
                    found = cur.fetchone()
                present = bool(found and found[0])
        finally:
            pool.putconn(conn)
    except Exception as exc:  # noqa: BLE001 — honesty over hard failure
        logger.warning("0075 probe failed, embeddings backend unwired: %s", exc)
        return None, "embeddings search unavailable (table probe failed)"
    if not present:
        return None, "embeddings search unavailable (migration 0075 not applied)"
    return (
        _EmbeddingsSearchBackend(
            pool, include_ambient=include_ambient, encoder=encoder
        ),
        None,
    )


def _first_text_field(content_text: str) -> str | None:
    """First non-empty text-ish value of a JSONB payload, or None."""
    try:
        import json as _json

        parsed = _json.loads(content_text) if content_text else {}
    except (ValueError, TypeError):
        parsed = {}
    if not isinstance(parsed, dict):
        return None
    for key in ("text", "preview", "chunk", "excerpt", "summary"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:240]
    return None


def _node_snippet(content_text: str, content_hash: str) -> str:
    """Snippet for a corpus hit: first text-ish JSONB value, else a ref line.

    Never fabricates content — when the parsed payload has no text field,
    the hit keeps an honest pointer to the content hash.
    """
    text = _first_text_field(content_text)
    if text:
        return text
    if content_text and not content_text.startswith("{"):
        return content_text[:240]
    return f"(content ref: {content_hash[:16]}…)" if content_hash else "(no snippet)"


class _RAGSearchBackend:
    """``search_api`` RAGSearchable port over the node-keyed central corpus.

    ADR-027 P1b facade leg (WP-15): the ``rag`` scope answers from
    ``knowledge_nodes`` in central Postgres — the corpus the upload/intake
    paths mint into — instead of requiring a deployed mini-RiVault
    endpoint. The mini-RiVault corpus stays owned by ``rag_service``
    (endpoint lifecycle); this facade never proxies there and never
    fabricates its semantics: hits are corpus nodes, recency-ranked,
    fixed honest score (no relevance model — same convention as the
    embeddings table-reader lane), with ``metadata.corpus_venue`` naming
    the venue.
    """

    def __init__(self, pool: Any, *, include_ambient: bool = False) -> None:
        self._pool = pool
        self._include_ambient = include_ambient

    def search_rag(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        pattern = f"%{query}%"
        conn = self._pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    # ADR-050: visibility families. Project rows + system rows
                    # are visible; ambient rows are read-gated (kept, never
                    # served to regular callers). `visibility` added by
                    # migration 0092 with default 'project'; predicate rendered
                    # by knowledge.access_scope.visibility_predicate (one place).
                    cur.execute(
                        "SELECT n.node_id, n.label, n.node_type, n.content::text, "
                        "n.content_hash, n.updated_at "
                        "FROM knowledge_nodes n "
                        f"WHERE {visibility_predicate('n', include_ambient=self._include_ambient)} "
                        "AND (n.label ILIKE %s OR n.node_type ILIKE %s "
                        "OR n.content::text ILIKE %s) "
                        "ORDER BY n.updated_at DESC LIMIT %s",
                        (project_id, pattern, pattern, pattern, limit),
                    )
                    rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [self._hit(row) for row in rows]

    @staticmethod
    def _hit(row: tuple) -> dict[str, Any]:
        node_id, label, node_type, content_text, content_hash, _updated = row[:6]
        return {
            "title": label or node_id,
            "content": _node_snippet(content_text or "", content_hash or ""),
            "score": 0.5,
            "source_id": node_id,
            "source_type": "knowledge_corpus_node",
            "metadata": {
                "node_type": node_type,
                "lane": "central-corpus",
                "corpus_venue": "central-pg",
                "content_hash": content_hash,
            },
        }


def _rag_backend_for(
    pool: Any, *, include_ambient: bool = False
) -> tuple[_RAGSearchBackend | None, str | None]:
    """Wire the central-corpus RAG backend iff knowledge_nodes exists."""
    if pool is None:
        return None, "RAG search requires the database"
    try:
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('knowledge_nodes')")
                    found = cur.fetchone()
                present = bool(found and found[0])
        finally:
            pool.putconn(conn)
    except Exception as exc:  # noqa: BLE001 — honesty over hard failure
        logger.warning("corpus probe failed, RAG backend unwired: %s", exc)
        return None, "RAG search unavailable (corpus probe failed)"
    if not present:
        return (
            None,
            "central knowledge corpus unavailable (knowledge nodes not present)",
        )
    return _RAGSearchBackend(pool, include_ambient=include_ambient), None


def _build_unified_search(
    meta_store: Any,
    graph_backend: _GraphSearchBackend | None,
    graph_unavailable_reason: str | None,
    embeddings_backend: _EmbeddingsSearchBackend | None = None,
    embeddings_unavailable_reason: str | None = None,
    rag_backend: _RAGSearchBackend | None = None,
    rag_unavailable_reason: str | None = None,
    include_ambient: bool = False,
) -> tuple[UnifiedSearch, list[str], dict[str, str]]:
    """Compose pluggable backends.

    Returns ``(search, wired_backend_names, unwired_backend_reasons)``.
    """
    wired = ["catalog"]
    unwired = dict(_UNWIRED_BACKENDS)
    kwargs: dict[str, Any] = {
        "catalog": _CatalogSearchBackend(meta_store, include_ambient=include_ambient)
    }
    if graph_backend is not None:
        kwargs["knowledge_graph"] = graph_backend
        wired.append("knowledge_graph")
    else:
        unwired["knowledge_graph"] = graph_unavailable_reason or "unavailable"
    if embeddings_backend is not None:
        kwargs["embeddings"] = embeddings_backend
        unwired.pop("embeddings", None)
        wired.append("embeddings")
    elif embeddings_unavailable_reason is not None:
        unwired["embeddings"] = embeddings_unavailable_reason
    if rag_backend is not None:
        kwargs["rag"] = rag_backend
        unwired.pop("rag", None)
        wired.append("rag")
    elif rag_unavailable_reason is not None:
        unwired["rag"] = rag_unavailable_reason
    return UnifiedSearch(**kwargs), wired, unwired


def _parse_scope(raw: str) -> SearchScope:
    """Validate the scope query parameter against ``SearchScope`` names."""
    if raw in _VALID_SCOPES:
        return SearchScope(raw)
    raise_typed(
        PlatformError.knowledgequery(
            "bad_request",
            f"Unknown scope '{raw}'. Valid values: {sorted(_VALID_SCOPES)}",
        ),
    )


def _resolve_project_id(identity: CallerIdentity, project_id: str | None) -> str:
    """Verified project context wins; fall back to the explicit parameter."""
    resolved = identity.project_id or (project_id or "")
    if not resolved:
        raise_typed(
            PlatformError.knowledgequery(
                "bad_request", "project_id is required (or a verified project header)"
            ),
        )
    return resolved


@router.get("/query")
def query_knowledge(
    request: Request,
    q: str = Query(..., min_length=1, max_length=512, description="Free-text query"),
    project_id: str = Query(default=None, description="Project scope"),
    scope: str = Query(
        default="all",
        description="catalog | knowledge_graph | embeddings | rag | all",
    ),
    graphs: str = Query(
        default=None,
        description="Comma-separated graph ids/names to query (default: project graph)",
    ),
    exclude_graphs: str = Query(
        default=None,
        description="Comma-separated graph ids/names to exclude (wins over includes)",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """K3 facade: one search entry across UMRS knowledge backends.

    The knowledge-graph backend is wired when the durable pool exists and
    the graph selection is non-empty; every hit carries graph provenance
    (``metadata.graph_id`` / ``metadata.graph_name``).
    """
    meta_store = get_optional_catalog_meta_store()
    if meta_store is None:
        raise_typed(
            PlatformError.knowledgequery(
                "unavailable", "Catalog metadata store is unavailable"
            ),
        )
    resolved_project = _resolve_project_id(identity, project_id)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    # ADR-050 wave-2.1: the ambient read lane is admin-only; regular callers
    # never see ambient rows on any lane.
    include_ambient = bool(identity.is_admin)
    scope_for_graphs = request_caller_scope(request)
    backend, reason, selected, excluded = _graph_backend_for(
        pool,
        resolved_project,
        graphs,
        exclude_graphs,
        include_ambient=include_ambient,
        readable=lambda graph: graph_meta_readable(request, graph, scope_for_graphs),
    )
    emb_backend, emb_reason = _embeddings_backend_for(
        pool,
        include_ambient=include_ambient,
        encoder=_encode_query if _query_encoder_url() else None,
    )
    rag_backend, rag_reason = _rag_backend_for(pool, include_ambient=include_ambient)
    search, wired, unwired = _build_unified_search(
        meta_store,
        backend,
        reason,
        emb_backend,
        emb_reason,
        rag_backend,
        rag_reason,
        include_ambient=include_ambient,
    )
    results = search.search(
        q,
        scope=_parse_scope(scope),
        project_id=resolved_project,
        limit=limit,
    )
    payload = results.to_dict()
    payload["wired_backends"] = wired
    payload["unavailable_backends"] = unwired
    if emb_backend is not None and emb_backend.lane_report is not None:
        payload["embeddings_lane"] = emb_backend.lane_report
    payload["selected_graphs"] = [g.graph_id for g in selected]
    payload["excluded_graphs"] = excluded
    payload["ontologies"] = _ontology_blocks_for(
        selected, getattr(request.app.state, "umrs_db_pool", None)
    )
    return {"status": "ok", "query": payload}


def _graph_backend_for(
    pool: Any,
    project_id: str,
    graphs_csv: str | None,
    exclude_csv: str | None,
    *,
    include_ambient: bool = False,
    readable: GraphReadable | None = None,
) -> tuple[_GraphSearchBackend | None, str | None, list, list]:
    """Build the graph backend or an honesty reason, plus selection truth."""
    if pool is None:
        return (
            None,
            "knowledge graph search requires the database",
            [],
            _split_csv(exclude_csv),
        )
    registry = GraphRegistryStore(pool)
    selected, excluded = _resolve_selected_graphs(
        registry, project_id, graphs_csv, exclude_csv, readable
    )
    if not selected:
        return None, "no graphs selected (empty or fully excluded)", selected, excluded
    return (
        _GraphSearchBackend(pool, selected, include_ambient=include_ambient),
        None,
        selected,
        excluded,
    )


def _ontology_registry_for(pool: Any) -> tuple[Any | None, Any | None]:
    """Open a registry sync conn branched off a psycopg-shaped pool.

    Returns (registry, conn); both None when the registry is unavailable —
    callers then resolve via inline provenance only.
    """
    from umrs.knowledge.ontology_registry import OntologyRegistryStore

    if pool is None or not hasattr(pool, "getconn"):
        return None, None
    try:
        conn = pool.getconn()
    except Exception:  # noqa: BLE001 — registry degrades to inline only
        return None, None
    return OntologyRegistryStore(conn), conn


def _ontology_blocks_for(selected: list, pool: Any) -> list[dict[str, Any]]:
    """ADR-049 Leg B/A: the bound ontology per selected graph, registry
    (0091) resolving before inline provenance; unresolvable references are
    named via ontology_error, never smudged."""
    from umrs.knowledge.ontology import ontology_for_graph

    registry, conn = _ontology_registry_for(pool)
    blocks: list[dict[str, Any]] = []
    try:
        for graph in selected:
            meta = (
                graph
                if isinstance(graph, dict)
                else (graph.to_dict() if hasattr(graph, "to_dict") else {})
            )
            block = {"graph_id": meta.get("graph_id", "?")}
            try:
                blocks.append(
                    {
                        **block,
                        "ontology": ontology_for_graph(
                            meta, registry=registry
                        ).to_dict(),
                    }
                )
            except ValueError as exc:
                blocks.append({**block, "ontology_error": str(exc)})
    finally:
        if conn is not None and hasattr(pool, "putconn"):
            try:
                pool.putconn(conn)
            except Exception:  # noqa: BLE001 — pool return must not mask the response
                pass
    return blocks
