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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from umrs.search_api import SearchScope, UnifiedSearch

from .catalog_store import get_optional_catalog_meta_store
from .jobs import CallerIdentity, get_caller_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge-query"])

_VALID_SCOPES = {scope.value for scope in SearchScope}

_UNWIRED_BACKENDS: dict[str, str] = {
    "knowledge_graph": "graph query backend over the PG sink not wired yet",
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


def _build_unified_search(
    meta_store: Any,
) -> tuple[UnifiedSearch, list[str], dict[str, str]]:
    """Compose pluggable backends.

    Returns ``(search, wired_backend_names, unwired_backend_reasons)``.
    """
    search = UnifiedSearch(catalog=_CatalogSearchBackend(meta_store))
    return search, ["catalog"], dict(_UNWIRED_BACKENDS)


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
    q: str = Query(..., min_length=1, max_length=512, description="Free-text query"),
    project_id: str = Query(default=None, description="Project scope"),
    scope: str = Query(
        default="all",
        description="catalog | knowledge_graph | embeddings | rag | all",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """K3 facade: one search entry across UMRS knowledge backends."""
    meta_store = get_optional_catalog_meta_store()
    if meta_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Catalog metadata store is unavailable",
        )
    resolved_project = _resolve_project_id(identity, project_id)
    search, wired, unwired = _build_unified_search(meta_store)
    results = search.search(
        q,
        scope=_parse_scope(scope),
        project_id=resolved_project,
        limit=limit,
    )
    payload = results.to_dict()
    payload["wired_backends"] = wired
    payload["unavailable_backends"] = unwired
    return {"status": "ok", "query": payload}
