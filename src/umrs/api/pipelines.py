"""
Pipeline and RAG Profile REST API (X-001, X-002): retired; component registry.

Retired (410 Gone; the caller is resolved first, strict mode 401 without the
signed tuple):

- ``/umrs/rag-profiles`` CRUD (X-002), owner decision 2026-08-22.
- ``/umrs/pipelines`` CRUD (X-001), 2026-10-18 (hardening plan H-10.1
  follow-up). Definitions lived in a module-global in-memory store
  (``_pipeline_store``) that held every project's rows and lost them at
  restart, and nothing at ingest or search time read them: the RAG service
  chunks with ``UMRS_RAG_CHUNK_MAX_TOKENS`` and embeds with
  ``UMRS_RAG_EMBEDDING_MODEL`` (``api/rag_service.py``). A table would have
  stored definitions that still change nothing. Decision and evidence:
  ``docs/development/dormant-modules.md``, "Module-global per-project state".

Served:

- GET /umrs/components/chunkers - List available chunkers
- GET /umrs/components/embedders - List available embedders
- GET /umrs/components/rerankers - List available rerankers
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..rag.registry import (
    get_chunker_registry,
    get_embedder_registry,
    get_reranker_registry,
)

router = APIRouter(prefix="/umrs", tags=["pipelines"])


def verified_user_project(request: Request) -> tuple[str, str]:
    """(user, project) of the HMAC-verified caller.

    The routes read the raw ``X-User-ID`` / ``X-Project-ID`` headers, so the
    project was whatever the client sent (H-1, 2026-10-17). Strict mode now
    needs the verified identity (401 otherwise); dev (mvp) mode keeps the old
    ``anonymous`` / ``default`` fallback.
    """
    from ._scope_identity import _strict_mode, request_caller_scope

    caller = request_caller_scope(request)
    # mvp's legacy tier keeps the raw project but not the user; dev clients
    # still name themselves with the raw header (never in strict mode).
    raw_user = "" if _strict_mode() else request.headers.get("x-user-id", "").strip()
    return caller.user_id or raw_user or "anonymous", caller.project_id or "default"


# =============================================================================
# Pipeline Endpoints (X-001) — RETIRED (H-10.1 follow-up, 2026-10-18)
# =============================================================================

_PIPELINES_GONE = (
    "Pipeline definitions are retired: they were kept in process memory "
    "(lost at restart) and no ingest or search ever read them. Ingestion "
    "chunks with the site setting UMRS_RAG_CHUNK_MAX_TOKENS and embeds with "
    "UMRS_RAG_EMBEDDING_MODEL; the available components are listed at "
    "GET /umrs/components/{chunkers,embedders,rerankers}."
)


def _pipelines_gone() -> None:
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_PIPELINES_GONE)


@router.post("/pipelines", status_code=status.HTTP_410_GONE)
async def create_pipeline(
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: pipeline definitions were never consulted."""
    _pipelines_gone()


@router.get("/pipelines", status_code=status.HTTP_410_GONE)
async def list_pipelines(
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: pipeline definitions were never consulted."""
    _pipelines_gone()


@router.get("/pipelines/{pipeline_id}", status_code=status.HTTP_410_GONE)
async def get_pipeline(
    pipeline_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: pipeline definitions were never consulted."""
    del pipeline_id
    _pipelines_gone()


@router.patch("/pipelines/{pipeline_id}", status_code=status.HTTP_410_GONE)
async def update_pipeline(
    pipeline_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: pipeline definitions were never consulted."""
    del pipeline_id
    _pipelines_gone()


@router.delete("/pipelines/{pipeline_id}", status_code=status.HTTP_410_GONE)
async def delete_pipeline(
    pipeline_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: pipeline definitions were never consulted."""
    del pipeline_id
    _pipelines_gone()


# =============================================================================
# RAG Profile Endpoints (X-002) — RETIRED (R3, 2026-08-22)
# =============================================================================

_RAG_PROFILES_GONE = (
    "rag-profiles CRUD is retired: profiles were ephemeral "
    "(in-memory), never consulted at ingest/search time, and are not "
    "part of the runtime path. Use site config for retrieval tuning; "
    "this router family is removed in the next release."
)


def _rag_profiles_gone() -> None:
    """rag-profiles CRUD retired (owner decision 2026-08-22).

    Profiles persisted to an ephemeral in-memory store, were never read at
    runtime, and raged nothing — retirement returns 410 Gone instead of a
    deceptive 2xx on a dead store.
    """
    raise HTTPException(status_code=status.HTTP_410_GONE, detail=_RAG_PROFILES_GONE)


@router.post("/rag-profiles", status_code=status.HTTP_410_GONE)
async def create_rag_profile(
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: RAG profiles were never consulted."""
    _rag_profiles_gone()


@router.get("/rag-profiles", status_code=status.HTTP_410_GONE)
async def list_rag_profiles(
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: RAG profiles were never consulted."""
    _rag_profiles_gone()


@router.get("/rag-profiles/{profile_id}", status_code=status.HTTP_410_GONE)
async def get_rag_profile(
    profile_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: RAG profiles were never consulted."""
    del profile_id
    _rag_profiles_gone()


@router.patch("/rag-profiles/{profile_id}", status_code=status.HTTP_410_GONE)
async def update_rag_profile(
    profile_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: RAG profiles were never consulted."""
    del profile_id
    _rag_profiles_gone()


@router.delete("/rag-profiles/{profile_id}", status_code=status.HTTP_410_GONE)
async def delete_rag_profile(
    profile_id: str,
    _caller: tuple[str, str] = Depends(verified_user_project),
) -> None:
    """Retired: RAG profiles were never consulted."""
    del profile_id
    _rag_profiles_gone()


# =============================================================================
# Component Registry Endpoints (X-003, X-004, X-005)
# =============================================================================


@router.get("/components/chunkers")
async def list_chunkers() -> list[dict[str, Any]]:
    """List available chunking methods (X-003)."""
    registry = get_chunker_registry()
    return [entry.to_dict() for entry in registry.list_all()]


@router.get("/components/embedders")
async def list_embedders(
    language: str | None = Query(None, description="Filter by language"),
    gpu_only: bool | None = Query(None, description="Filter by GPU requirement"),
) -> list[dict[str, Any]]:
    """List available embedding models (X-004)."""
    registry = get_embedder_registry()

    embedders = registry.list_all()
    if language:
        embedders = [
            e
            for e in embedders
            if language in e.languages or "multilingual" in e.languages
        ]
    if gpu_only is not None:
        embedders = [e for e in embedders if e.requires_gpu == gpu_only]

    return [entry.to_dict() for entry in embedders]


@router.get("/components/rerankers")
async def list_rerankers() -> list[dict[str, Any]]:
    """List available reranking methods (X-005)."""
    registry = get_reranker_registry()
    return [entry.to_dict() for entry in registry.list_all()]
