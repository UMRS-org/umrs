"""UMRS API routes for the knowledge model (Phase 2, Step 3.1).

Exposes the knowledge_model subsystem as HTTP API endpoints.

Routes:
  POST   /umrs/knowledge-model/objects       — Create a knowledge object
  GET    /umrs/knowledge-model/objects/{id}    — Get a knowledge object
  PUT    /umrs/knowledge-model/objects/{id}    — Update a knowledge object
  DELETE /umrs/knowledge-model/objects/{id}    — Delete a knowledge object
  GET    /umrs/knowledge-model/objects         — List knowledge objects
  GET    /umrs/knowledge-model/search          — Search knowledge objects
  GET    /umrs/knowledge-model/objects/{id}/history — Get object version history
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from .typed_errors import PlatformError, raise_typed
from pydantic import BaseModel, Field

from umrs.knowledge_model import (
    InMemoryKnowledgeModelStore,
    KnowledgeObject,
    KnowledgeObjectNotFoundError,
    KnowledgeObjectStatus,
    KnowledgeObjectType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge-model", tags=["knowledge-model"])

# Store instance — injected at startup
_store: InMemoryKnowledgeModelStore | None = None


def set_store(store: InMemoryKnowledgeModelStore) -> None:
    global _store
    _store = store


def _get_store() -> InMemoryKnowledgeModelStore:
    if _store is None:
        raise_typed(
            PlatformError.knowledgemodel(
                "unavailable",
                "Knowledge model store not initialized",
            )
        )
    return _store


# ── Request/Response Models ──────────────────────────────────────────────


class CreateObjectRequest(BaseModel):
    project_id: str = Field(..., description="Project ID")
    object_type: str = Field(..., description="Object type (paper, code, hypothesis, etc.)")
    content: dict[str, Any] = Field(default_factory=dict, description="Object content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_immutable: bool = Field(False, description="Whether this object can be updated")
    created_by: str = Field("", description="Who created the object")


class UpdateObjectRequest(BaseModel):
    content: dict[str, Any] | None = Field(None, description="New content")
    metadata: dict[str, Any] | None = Field(None, description="New metadata")


class ObjectResponse(BaseModel):
    object_id: str
    project_id: str
    object_type: str
    content: dict[str, Any]
    metadata: dict[str, Any]
    version: int
    status: str
    is_immutable: bool
    created_at: str
    updated_at: str
    created_by: str


class ObjectListResponse(BaseModel):
    objects: list[ObjectResponse]
    total: int


class ObjectHistoryResponse(BaseModel):
    object_id: str
    versions: list[dict[str, Any]]


class SearchResponse(BaseModel):
    query: str
    results: list[ObjectResponse]
    total: int


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/objects", response_model=ObjectResponse)
async def create_object(req: CreateObjectRequest) -> ObjectResponse:
    """Create a new knowledge object."""
    store = _get_store()
    try:
        object_type = KnowledgeObjectType(req.object_type)
    except ValueError:
        raise_typed(
            PlatformError.knowledgemodel(
                "bad_request",
                f"Invalid object type: {req.object_type}",
            )
        )

    object_id = store.create_object(
        project_id=req.project_id,
        object_type=object_type,
        content=req.content,
        metadata=req.metadata,
        is_immutable=req.is_immutable,
        created_by=req.created_by,
    )
    obj = store.get_object(object_id)
    return ObjectResponse(**obj.to_dict())


@router.get("/objects/{object_id}", response_model=ObjectResponse)
async def get_object(object_id: str) -> ObjectResponse:
    """Get a knowledge object by ID."""
    store = _get_store()
    obj = store.get_object(object_id)
    if obj is None:
        raise_typed(
            PlatformError.knowledgemodel(
                "not_found",
                f"Object {object_id} not found",
            )
        )
    return ObjectResponse(**obj.to_dict())


@router.put("/objects/{object_id}", response_model=ObjectResponse)
async def update_object(
    object_id: str,
    req: UpdateObjectRequest,
) -> ObjectResponse:
    """Update a knowledge object's content and/or metadata."""
    store = _get_store()
    result = store.update_object(
        object_id,
        content=req.content,
        metadata=req.metadata,
    )
    if not result:
        raise_typed(
            PlatformError.knowledgemodel(
                "not_found",
                f"Object {object_id} not found",
            )
        )
    obj = store.get_object(object_id)
    return ObjectResponse(**obj.to_dict())


@router.delete("/objects/{object_id}")
async def delete_object(
    object_id: str,
    deleted_by: str = Query("", description="Who deleted the object"),
    deletion_reason: str = Query("", description="Why the object was deleted"),
) -> dict[str, Any]:
    """Delete a knowledge object (soft delete via tombstone)."""
    store = _get_store()
    result = store.delete_object(
        object_id,
        deleted_by=deleted_by,
        deletion_reason=deletion_reason,
    )
    if not result:
        raise_typed(
            PlatformError.knowledgemodel(
                "not_found",
                f"Object {object_id} not found",
            )
        )
    return {"deleted": True, "object_id": object_id}


@router.get("/objects", response_model=ObjectListResponse)
async def list_objects(
    project_id: str | None = Query(None, description="Filter by project ID"),
    object_type: str | None = Query(None, description="Filter by object type"),
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
) -> ObjectListResponse:
    """List knowledge objects with optional filters."""
    store = _get_store()
    # Parse filters
    obj_type: KnowledgeObjectType | None = None
    if object_type:
        try:
            obj_type = KnowledgeObjectType(object_type)
        except ValueError:
            raise_typed(PlatformError.knowledgemodel("bad_request", f"Invalid object type: {object_type}"))

    obj_status: KnowledgeObjectStatus | None = None
    if status:
        try:
            obj_status = KnowledgeObjectStatus(status)
        except ValueError:
                        raise_typed(PlatformError.knowledgemodel("bad_request", f"Invalid status: {status}"))

    objects = store.list_objects(
        project_id=project_id,
        object_type=obj_type,
        status=obj_status,
        limit=limit,
    )
    return ObjectListResponse(
        objects=[ObjectResponse(**obj.to_dict()) for obj in objects],
        total=len(objects),
    )


@router.get("/search", response_model=SearchResponse)
async def search_objects(
    query: str = Query(..., description="Search query"),
    project_id: str | None = Query(None, description="Filter by project ID"),
    object_type: str | None = Query(None, description="Filter by object type"),
    limit: int = Query(20, ge=1, le=100),
) -> SearchResponse:
    """Search knowledge objects by content, metadata, or object_id."""
    store = _get_store()
    obj_type: KnowledgeObjectType | None = None
    if object_type:
        try:
            obj_type = KnowledgeObjectType(object_type)
        except ValueError:
            raise_typed(PlatformError.knowledgemodel("bad_request", f"Invalid object type: {object_type}"))

    objects = store.search_objects(
        query,
        project_id=project_id,
        object_type=obj_type,
        limit=limit,
    )
    return SearchResponse(
        query=query,
        results=[ObjectResponse(**obj.to_dict()) for obj in objects],
        total=len(objects),
    )


@router.get("/objects/{object_id}/history", response_model=ObjectHistoryResponse)
async def get_object_history(object_id: str) -> ObjectHistoryResponse:
    """Get the version history of a knowledge object."""
    store = _get_store()
    try:
        versions = store.get_object_history(object_id)
    except KnowledgeObjectNotFoundError:
        raise_typed(
            PlatformError.knowledgemodel(
                "not_found",
                f"Object {object_id} not found",
            )
        )
    return ObjectHistoryResponse(
        object_id=object_id,
        versions=[v.to_dict() for v in versions],
    )
