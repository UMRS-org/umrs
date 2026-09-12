"""
Data catalog API — agent-facing search and raw read.

Endpoints:
- GET /umrs/catalog/search
- GET /umrs/catalog/read
- GET /umrs/catalog/stats
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from umrs.api.catalog_store import CatalogStoreUnavailable, get_catalog_meta_store
from umrs.api.files import get_project_id
from umrs.data_catalog.links import hydrate_links
from umrs.data_catalog.plane_counts import compute_plane_counts
from umrs.data_catalog.query import read_catalog_object, search_catalog
from umrs.knowledge.access_scope import CallerScope, project_readable
from umrs.shared.storage.s3_config import S3Config, get_s3_config

from ._scope_identity import get_caller_scope
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/catalog", tags=["catalog"])


def _enforce_catalog_scope(caller: CallerScope, project_id: str) -> None:
    """Three-tier boundary: system (all) / project (members) / user (owner)."""
    if project_readable(project_id, caller):
        return
    raise_typed(
        PlatformError.catalog(
            "forbidden",
            { "error": "scope_denied", "message": "Project is outside the caller's access tier.", }
        ),
    )


def _require_catalog_meta_store():
    try:
        return get_catalog_meta_store()
    except CatalogStoreUnavailable as exc:
        raise_typed(
            PlatformError.catalog(
                "unavailable",
                str(exc)
            ),
            cause=exc,
        )


class CatalogLinksResponse(BaseModel):
    """ADR-031 plane reads, hydrated server-side per catalogue row."""

    document_id: str
    link_method: str  # "s3_object" | "filename"
    chunk_nodes: int = 0
    vector_rows: list[dict[str, Any]] = Field(default_factory=list)


class CatalogEntryResponse(BaseModel):
    s3_key: str
    state: str
    summary: str
    deterministic_summary: str = ""
    perception_summary: str | None = None
    perception_model: str | None = None
    detected_modality: str
    tags: list[str] = Field(default_factory=list)
    filename_signals: list[str] = Field(default_factory=list)
    extracted_text_ref: str | None = None
    relevance_score: float = 0.0
    links: CatalogLinksResponse | None = None


class CatalogSearchResponse(BaseModel):
    query: str
    project_id: str
    results: list[CatalogEntryResponse]


class CatalogReadResponse(BaseModel):
    s3_key: str
    offset: int
    length: int
    total_size: int
    truncated: bool
    text: str | None = None
    read_source: str = "raw"
    extracted_text_ref: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    links: CatalogLinksResponse | None = None


class CatalogStatsResponse(BaseModel):
    project_id: str
    object_count: int
    states: dict[str, int] = Field(default_factory=dict)
    modalities: dict[str, int] = Field(default_factory=dict)
    # ADR-031 §Consequences 6: downstream-plane row counts. ``None`` = not
    # computable in this process (no shared pool / query failure) — honest
    # omission, never a fabricated zero.
    canonical_documents: int | None = None
    chunk_nodes: int | None = None
    vector_rows: int | None = None


@router.get(
    "/search",
    response_model=CatalogSearchResponse,
    summary="Search the data catalog",
)
async def catalog_search(
    request: Request,
    q: str = Query("", description="Search query (optional when filters are set)"),
    limit: int = Query(20, ge=1, le=100),
    modality: str | None = Query(None, description="Filter by detected_modality"),
    state: str | None = Query(None, description="Filter by catalog state"),
    tag: str | None = Query(None, description="Filter rows containing this tag"),
    prefix: str | None = Query(None, description="Filter s3_key prefix within project"),
    project_id: str = Depends(get_project_id),
    caller: CallerScope = Depends(get_caller_scope),
) -> CatalogSearchResponse:
    """Grep catalog metadata scoped to the caller's project."""
    _enforce_catalog_scope(caller, project_id)
    if not q.strip() and not any([modality, state, tag, prefix]):
        raise_typed(
            PlatformError.catalog(
                "unprocessable",
                "Provide q and/or at least one filter (modality, state, tag, prefix)"
            ),
        )
    meta_store = _require_catalog_meta_store()
    hits = search_catalog(
        meta_store,
        query=q,
        project_id=project_id,
        limit=limit,
        modality=modality,
        state=state,
        tag=tag,
        prefix=prefix,
    )
    pool = getattr(request.app.state, "umrs_db_pool", None)
    return CatalogSearchResponse(
        query=q,
        project_id=project_id,
        results=[
            CatalogEntryResponse(
                s3_key=hit.meta.s3_key,
                state=hit.meta.state.value,
                summary=hit.meta.summary,
                deterministic_summary=hit.meta.deterministic_summary,
                perception_summary=hit.meta.perception_summary,
                perception_model=hit.meta.perception_model,
                detected_modality=hit.meta.detected_modality,
                tags=hit.meta.tags,
                filename_signals=hit.meta.filename_signals,
                extracted_text_ref=hit.meta.extracted_text_ref,
                relevance_score=hit.score,
                links=hydrate_links(pool, project_id=project_id, s3_key=hit.meta.s3_key),
            )
            for hit in hits
        ],
    )


@router.get(
    "/read",
    response_model=CatalogReadResponse,
    summary="Read a catalog object (byte-bounded)",
)
async def catalog_read(
    request: Request,
    s3_key: str = Query(..., description="S3 object key"),
    offset: int = Query(0, ge=0),
    length: int = Query(1_048_576, ge=1, le=10_485_760),
    source: str = Query(
        "auto",
        description="auto | raw | extracted — auto uses extracted text for documents",
    ),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
    caller: CallerScope = Depends(get_caller_scope),
) -> CatalogReadResponse:
    """Read catalog object bytes or extracted document text with project ACL."""
    _enforce_catalog_scope(caller, project_id)
    if source not in {"auto", "raw", "extracted"}:
        raise_typed(
            PlatformError.catalog(
                "unprocessable",
                "source must be auto, raw, or extracted"
            ),
        )
    meta_store = _require_catalog_meta_store()
    s3_client = config.get_client()
    try:
        payload = read_catalog_object(
            s3_client,
            config.bucket,
            meta_store,
            s3_key=s3_key,
            project_id=project_id,
            offset=offset,
            length=length,
            source=source,  # type: ignore[arg-type]
        )
    except PermissionError as exc:
        raise_typed(
            PlatformError.catalog(
                "forbidden",
                str(exc)
            ),
            cause=exc,
        )
    except ValueError as exc:
        raise_typed(
            PlatformError.catalog(
                "not_found",
                str(exc)
            ),
            cause=exc,
        )
    except Exception as exc:
        logger.exception("catalog read failed for %s", s3_key)
        raise_typed(
            PlatformError.catalog("unavailable", "Failed to read catalog object"),
            cause=exc,
        )

    pool = getattr(request.app.state, "umrs_db_pool", None)
    payload["links"] = hydrate_links(pool, project_id=project_id, s3_key=s3_key)
    return CatalogReadResponse(**payload)


@router.get(
    "/stats",
    response_model=CatalogStatsResponse,
    summary="Catalog object counts by state and modality",
)
async def catalog_stats(
    request: Request,
    project_id: str = Depends(get_project_id),
    caller: CallerScope = Depends(get_caller_scope),
) -> CatalogStatsResponse:
    """Aggregate catalog stats + downstream plane counts (ADR-031 §Conseq. 6).

    Plane counts are a read-side projection of the canonical/vector planes;
    consistency classification itself stays with the plane reconciler.
    """
    _enforce_catalog_scope(caller, project_id)
    meta_store = _require_catalog_meta_store()
    stats = meta_store.project_stats(project_id)
    plane_counts = None
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is not None:
        try:
            plane_counts = compute_plane_counts(pool, project_id)
        except Exception:
            logger.warning(
                "plane counts unavailable for %s — omitting from stats",
                project_id,
                exc_info=True,
            )
    return CatalogStatsResponse(
        project_id=stats["project_id"],
        object_count=stats["object_count"],
        states=stats["states"],
        modalities=stats["modalities"],
        canonical_documents=plane_counts.canonical_documents if plane_counts else None,
        chunk_nodes=plane_counts.chunk_nodes if plane_counts else None,
        vector_rows=plane_counts.vector_rows if plane_counts else None,
    )
