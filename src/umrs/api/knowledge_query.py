"""K3: unified knowledge query facade (read-side, additive).

Route: ``GET /umrs/knowledge/query``

One search entry across UMRS knowledge backends, replacing per-route
fan-out by agents/R4 (catalog / knowledge / provenance / artifact /
graph). Backends are composed via the ``search_api`` UnifiedSearch
port protocols. v1 wires the catalog metadata search; backends that
are not wired yet (knowledge graph, embeddings, RAG) are reported in
``unavailable_backends`` with reasons instead of silently returning
empty results — the response is honest about coverage.

Existing subsystem routes stay untouched (aliases per K3 guardrail).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from umrs.knowledge.graph_registry import GraphRegistryStore, KnowledgeGraphMeta
from umrs.search_api import SearchScope, UnifiedSearch

from .catalog_store import get_optional_catalog_meta_store
from .jobs import CallerIdentity, get_caller_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-query"])

_VALID_SCOPES = {scope.value for scope in SearchScope}

_UNWIRED_BACKENDS: dict[str, str] = {
    "embeddings": "query-embedding generation requires the embedding service",
    "rag": "RAG search requires a deployed mini-rivault instance",
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

    def __init__(self, meta_store: Any) -> None:
        self._store = meta_store

    def search_catalog(
        self, query: str, *, project_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        if not project_id:
            # Catalog ACL is project-scoped; the facade requires it anyway.
            return []
        from umrs.data_catalog.query import search_catalog

        hits = search_catalog(
            self._store, query=query, project_id=project_id, limit=limit
        )
        return [_hit_to_dict(hit, project_id=project_id) for hit in hits]


class _GraphSearchBackend:
    """``search_api`` KnowledgeGraphSearchable port over the canonical store.

    Searches ``knowledge_nodes`` scoped to the selected graphs. Legacy rows
    (``graph_id = ''``) are visible through their project's default graph —
    the same fallback ``KnowledgeStore.list_nodes`` uses.
    """

    def __init__(self, pool: Any, selected: list[KnowledgeGraphMeta]) -> None:
        self._pool = pool
        self._selected = selected
        self._by_id = {g.graph_id: g for g in selected}
        self._default_by_project = {
            g.project_id: g for g in selected if g.project_id
        }

    def search_knowledge_graph(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        graph_ids = [g.graph_id for g in self._selected]
        default_projects = [
            pid for pid, meta in self._default_by_project.items()
            if meta.graph_id in graph_ids
        ]
        pattern = f"%{query}%"
        with self._pool.getconn() as conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT node_id, project_id, label, node_type, content, graph_id "
                        "FROM knowledge_nodes "
                        "WHERE (graph_id = ANY(%s) "
                        "   OR (graph_id = '' AND project_id = ANY(%s))) "
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
    registry: GraphRegistryStore, exclude_csv: Optional[str]
) -> tuple[set[str], list[str]]:
    """Map exclude tokens to graph ids (unknown tokens only recorded by name)."""
    excluded_ids: set[str] = set()
    excluded_names: list[str] = []
    for token in _split_csv(exclude_csv):
        graph = registry.resolve_graph(token)
        if graph is not None:
            excluded_ids.add(graph.graph_id)
        excluded_names.append(token)
    return excluded_ids, excluded_names


def _resolve_includes(
    registry: GraphRegistryStore,
    tokens: list[str],
    excluded_ids: set[str],
) -> list[KnowledgeGraphMeta]:
    """Resolve include tokens, dropping excluded and duplicate graphs."""
    selected: list[KnowledgeGraphMeta] = []
    for token in tokens:
        graph = registry.resolve_graph(token)
        if graph is None or graph.graph_id in excluded_ids:
            continue
        if graph.graph_id in {g.graph_id for g in selected}:
            continue
        selected.append(graph)
    return selected


def _resolve_selected_graphs(
    registry: GraphRegistryStore,
    project_id: str,
    graphs_csv: Optional[str],
    exclude_csv: Optional[str],
) -> tuple[list[KnowledgeGraphMeta], list[str]]:
    """Resolve explicit include/exclude tokens into selected graph metas.

    Explicit exclusion beats inclusion. Without explicit inclusion the
    project's default graph is the single candidate (registry discovery /
    candidate ranking lives client-side in the selection policy).
    """
    excluded_ids, excluded_names = _resolve_exclusions(registry, exclude_csv)

    tokens = _split_csv(graphs_csv)
    if tokens:
        return _resolve_includes(registry, tokens, excluded_ids), excluded_names

    default = registry.ensure_default_graph(project_id)
    selected = [default] if default.graph_id not in excluded_ids else []
    return selected, excluded_names


def _split_csv(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _build_unified_search(
    meta_store: Any,
    graph_backend: Optional[_GraphSearchBackend],
    graph_unavailable_reason: Optional[str],
) -> tuple[UnifiedSearch, list[str], dict[str, str]]:
    """Compose pluggable backends.

    Returns ``(search, wired_backend_names, unwired_backend_reasons)``.
    """
    wired = ["catalog"]
    unwired = dict(_UNWIRED_BACKENDS)
    kwargs: dict[str, Any] = {"catalog": _CatalogSearchBackend(meta_store)}
    if graph_backend is not None:
        kwargs["knowledge_graph"] = graph_backend
        wired.append("knowledge_graph")
    else:
        unwired["knowledge_graph"] = graph_unavailable_reason or "unavailable"
    return UnifiedSearch(**kwargs), wired, unwired


def _parse_scope(raw: str) -> SearchScope:
    """Validate the scope query parameter against ``SearchScope`` names."""
    if raw in _VALID_SCOPES:
        return SearchScope(raw)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unknown scope '{raw}'. Valid values: {sorted(_VALID_SCOPES)}",
    )


def _resolve_project_id(identity: CallerIdentity, project_id: str | None) -> str:
    """Verified project context wins; fall back to the explicit parameter."""
    resolved = identity.project_id or (project_id or "")
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="project_id is required (or a verified project header)",
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog metadata store is unavailable",
        )
    resolved_project = _resolve_project_id(identity, project_id)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    backend, reason, selected, excluded = _graph_backend_for(
        pool, resolved_project, graphs, exclude_graphs
    )
    search, wired, unwired = _build_unified_search(meta_store, backend, reason)
    results = search.search(
        q,
        scope=_parse_scope(scope),
        project_id=resolved_project,
        limit=limit,
    )
    payload = results.to_dict()
    payload["wired_backends"] = wired
    payload["unavailable_backends"] = unwired
    payload["selected_graphs"] = [g.graph_id for g in selected]
    payload["excluded_graphs"] = excluded
    return {"status": "ok", "query": payload}


def _graph_backend_for(
    pool: Any,
    project_id: str,
    graphs_csv: Optional[str],
    exclude_csv: Optional[str],
) -> tuple[Optional[_GraphSearchBackend], Optional[str], list, list]:
    """Build the graph backend or an honesty reason, plus selection truth."""
    if pool is None:
        return None, "knowledge graph search requires the database", [], _split_csv(exclude_csv)
    registry = GraphRegistryStore(pool)
    selected, excluded = _resolve_selected_graphs(
        registry, project_id, graphs_csv, exclude_csv
    )
    if not selected:
        return None, "no graphs selected (empty or fully excluded)", selected, excluded
    return _GraphSearchBackend(pool, selected), None, selected, excluded
