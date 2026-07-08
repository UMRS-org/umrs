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

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from umrs.api.catalog_store import CatalogStoreUnavailable, get_catalog_meta_store
from umrs.api.files import S3Config, get_project_id, get_s3_config
from umrs.data_catalog.query import read_catalog_object, search_catalog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/catalog", tags=["catalog"])


def _require_catalog_meta_store():
    try:
        return get_catalog_meta_store()
    except CatalogStoreUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


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


class CatalogStatsResponse(BaseModel):
    project_id: str
    object_count: int
    states: dict[str, int] = Field(default_factory=dict)
    modalities: dict[str, int] = Field(default_factory=dict)


@router.get(
    "/search",
    response_model=CatalogSearchResponse,
    summary="Search the data catalog",
)
async def catalog_search(
    q: str = Query("", description="Search query (optional when filters are set)"),
    limit: int = Query(20, ge=1, le=100),
    modality: str | None = Query(None, description="Filter by detected_modality"),
    state: str | None = Query(None, description="Filter by catalog state"),
    tag: str | None = Query(None, description="Filter rows containing this tag"),
    prefix: str | None = Query(None, description="Filter s3_key prefix within project"),
    project_id: str = Depends(get_project_id),
) -> CatalogSearchResponse:
    """Grep catalog metadata scoped to the caller's project."""
    if not q.strip() and not any([modality, state, tag, prefix]):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide q and/or at least one filter (modality, state, tag, prefix)",
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
    s3_key: str = Query(..., description="S3 object key"),
    offset: int = Query(0, ge=0),
    length: int = Query(1_048_576, ge=1, le=10_485_760),
    source: str = Query(
        "auto",
        description="auto | raw | extracted — auto uses extracted text for documents",
    ),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> CatalogReadResponse:
    """Read catalog object bytes or extracted document text with project ACL."""
    if source not in {"auto", "raw", "extracted"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="source must be auto, raw, or extracted",
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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("catalog read failed for %s", s3_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read catalog object",
        ) from exc

    return CatalogReadResponse(**payload)


@router.get(
    "/stats",
    response_model=CatalogStatsResponse,
    summary="Catalog object counts by state and modality",
)
async def catalog_stats(
    project_id: str = Depends(get_project_id),
) -> CatalogStatsResponse:
    """Return aggregate catalog stats for a project (replaces S3 manifest)."""
    meta_store = _require_catalog_meta_store()
    stats = meta_store.project_stats(project_id)
    return CatalogStatsResponse(
        project_id=stats["project_id"],
        object_count=stats["object_count"],
        states=stats["states"],
        modalities=stats["modalities"],
    )
