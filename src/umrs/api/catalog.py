"""
Data catalog API — agent-facing search and raw read.

Endpoints:
- GET /umrs/catalog/search
- GET /umrs/catalog/read
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from umrs.api.files import S3Config, get_project_id, get_s3_config
from umrs.data_catalog.query import read_catalog_object, search_catalog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/catalog", tags=["catalog"])


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
    meta: dict[str, Any] = Field(default_factory=dict)


@router.get(
    "/search",
    response_model=CatalogSearchResponse,
    summary="Search the data catalog",
)
async def catalog_search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> CatalogSearchResponse:
    """Grep catalog metadata scoped to the caller's project."""
    s3_client = config.get_client()
    hits = search_catalog(
        s3_client,
        config.bucket,
        query=q,
        project_id=project_id,
        limit=limit,
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
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> CatalogReadResponse:
    """Read raw object bytes with JWT/project ACL enforcement."""
    s3_client = config.get_client()
    try:
        payload = read_catalog_object(
            s3_client,
            config.bucket,
            s3_key=s3_key,
            project_id=project_id,
            offset=offset,
            length=length,
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("catalog read failed for %s", s3_key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read catalog object",
        ) from exc

    return CatalogReadResponse(**payload)
