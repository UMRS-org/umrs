"""
RAG component registry REST API (X-003..X-005).

Retired and deleted:

- ``/umrs/rag-profiles`` CRUD (X-002): owner decision 2026-08-22 (410 from
  then), deleted 2026-10-18 (hardening plan H-10.1 rag-profiles wave) once
  its last client (``r4 rag retriever`` CRUD; the TUI ``/pipeline
  rag-profiles`` already printed a notice) stopped calling. Profiles were
  ephemeral (in-memory) and never read at ingest/search time; retrieval
  tuning lives in site config. The ``rag-profiles`` lane family and the
  ``/gw/rag-profiles`` edge locations went with the routes.
- ``/umrs/pipelines`` CRUD (X-001), 2026-10-18: definitions lived in a
  module-global in-memory store that held every project's rows and lost
  them at restart, and nothing at ingest or search time read them: the
  RAG service chunks with ``UMRS_RAG_CHUNK_MAX_TOKENS`` and embeds with
  ``UMRS_RAG_EMBEDDING_MODEL`` (``api/rag_service.py``). The routes
  answered 410 (umrs d4a8ad85) until their clients (``r4 rag ingest``,
  TUI ``/pipeline``) were retired; then the routes, the ``pipelines``
  lane family and the ``/gw/pipelines`` edge locations went (umrs
  197cf29c). Decision and evidence: ``docs/development/dormant-modules.md``,
  "Module-global per-project state".

Served:

- GET /umrs/components/chunkers - List available chunkers
- GET /umrs/components/embedders - List available embedders
- GET /umrs/components/rerankers - List available rerankers
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from ..rag.registry import (
    get_chunker_registry,
    get_embedder_registry,
    get_reranker_registry,
)

router = APIRouter(prefix="/umrs", tags=["pipelines"])


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
