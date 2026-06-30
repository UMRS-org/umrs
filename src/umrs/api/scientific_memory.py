"""UMRS API routes for scientific memory (Phase 2, Step 3.2).

Exposes the scientific_memory subsystem as HTTP API endpoints.

Routes:
  POST   /umrs/scientific-memory/entries       — Add a memory entry
  GET    /umrs/scientific-memory/entries/{id}    — Get a memory entry
  GET    /umrs/scientific-memory/entries         — List memory entries
  GET    /umrs/scientific-memory/search           — Search memory entries
  PUT    /umrs/scientific-memory/entries/{id}/status — Update entry status
  GET    /umrs/scientific-memory/count            — Count memory entries
  GET    /umrs/scientific-memory/timeline         — Get project timeline
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.scientific_memory import (
    InMemoryScientificMemoryStore,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/scientific-memory", tags=["scientific-memory"])

# Store instance — injected at startup
_store: InMemoryScientificMemoryStore | None = None


def set_store(store: InMemoryScientificMemoryStore) -> None:
    global _store
    _store = store


def _get_store() -> InMemoryScientificMemoryStore:
    if _store is None:
        raise HTTPException(
            status_code=503,
            detail="Scientific memory store not initialized",
        )
    return _store


# ── Request/Response Models ──────────────────────────────────────────────


class AddEntryRequest(BaseModel):
    project_id: str = Field(..., description="Project ID")
    entry_type: str = Field("note", description="Entry type (generated_code, experiment_log, etc.)")
    source: str = Field("system", description="Who created the entry (user, agent, system)")
    content: str = Field("", description="Entry content")
    content_uri: str = Field("", description="S3 URI if content is large")
    summary: str = Field("", description="Short summary")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    confidence: float = Field(0.0, description="Confidence score (0.0–1.0)")
    references: list[str] = Field(default_factory=list, description="Other entry_ids this entry references")
    created_by: str = Field("", description="Who created the entry")


class EntryResponse(BaseModel):
    entry_id: str
    project_id: str
    experiment_id: str
    entry_type: str
    source: str
    content: str
    content_uri: str
    summary: str
    metadata: dict[str, Any]
    tags: list[str]
    status: str
    confidence: float
    references: list[str]
    created_at: str
    updated_at: str
    created_by: str


class EntryListResponse(BaseModel):
    entries: list[EntryResponse]
    total: int


class SearchResponse(BaseModel):
    query: str
    results: list[EntryResponse]
    total: int


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description="New status (active, archived, deprecated, retracted)")


class CountResponse(BaseModel):
    count: int


class TimelineResponse(BaseModel):
    project_id: str
    entries: list[EntryResponse]
    total: int


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/entries", response_model=EntryResponse)
async def add_entry(req: AddEntryRequest) -> EntryResponse:
    """Add a new memory entry."""
    store = _get_store()
    try:
        entry_type = MemoryEntryType(req.entry_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid entry type: {req.entry_type}",
        )
    entry = MemoryEntry(
        project_id=req.project_id,
        entry_type=entry_type,
        source=req.source,
        content=req.content,
        content_uri=req.content_uri,
        summary=req.summary,
        metadata=req.metadata,
        tags=req.tags,
        confidence=req.confidence,
        references=req.references,
        created_by=req.created_by,
    )
    entry_id = store.add(entry)
    entry = store.get(entry_id)
    return EntryResponse(**entry.to_dict())


@router.get("/entries/{entry_id}", response_model=EntryResponse)
async def get_entry(entry_id: str) -> EntryResponse:
    """Get a memory entry by ID."""
    store = _get_store()
    entry = store.get(entry_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"Entry {entry_id} not found",
        )
    return EntryResponse(**entry.to_dict())


@router.get("/entries", response_model=EntryListResponse)
async def list_entries(
    project_id: str | None = Query(None, description="Filter by project ID"),
    entry_type: str | None = Query(None, description="Filter by entry type"),
    status: str | None = Query(None, description="Filter by status"),
    experiment_id: str | None = Query(None, description="Filter by experiment ID"),
    limit: int = Query(100, ge=1, le=1000),
) -> EntryListResponse:
    """List memory entries with optional filters."""
    store = _get_store()
    # Parse filters
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entry type: {entry_type}",
            )

    estatus: MemoryEntryStatus | None = None
    if status:
        try:
            estatus = MemoryEntryStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}",
            )

    entries = store.list_entries(
        project_id=project_id,
        entry_type=etype,
        status=estatus,
        experiment_id=experiment_id,
        limit=limit,
    )
    return EntryListResponse(
        entries=[EntryResponse(**entry.to_dict()) for entry in entries],
        total=len(entries),
    )


@router.get("/search", response_model=SearchResponse)
async def search_entries(
    query: str = Query(..., description="Search query"),
    project_id: str | None = Query(None, description="Filter by project ID"),
    entry_type: str | None = Query(None, description="Filter by entry type"),
    limit: int = Query(20, ge=1, le=100),
) -> SearchResponse:
    """Search memory entries by content, summary, or tags."""
    store = _get_store()
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entry type: {entry_type}",
            )

    entries = store.search(
        query,
        project_id=project_id,
        entry_type=etype,
        limit=limit,
    )
    return SearchResponse(
        query=query,
        results=[EntryResponse(**entry.to_dict()) for entry in entries],
        total=len(entries),
    )


@router.put("/entries/{entry_id}/status")
async def update_status(
    entry_id: str,
    req: UpdateStatusRequest,
) -> dict[str, Any]:
    """Update the status of a memory entry."""
    store = _get_store()
    try:
        status = MemoryEntryStatus(req.status)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: {req.status}",
        )
    result = store.update_status(entry_id, status)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Entry {entry_id} not found",
        )
    return {"updated": True, "entry_id": entry_id, "status": req.status}


@router.get("/count", response_model=CountResponse)
async def count_entries(
    project_id: str | None = Query(None, description="Filter by project ID"),
    entry_type: str | None = Query(None, description="Filter by entry type"),
    status: str | None = Query(None, description="Filter by status"),
) -> CountResponse:
    """Count memory entries with optional filters."""
    store = _get_store()
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid entry type: {entry_type}",
            )

    estatus: MemoryEntryStatus | None = None
    if status:
        try:
            estatus = MemoryEntryStatus(status)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}",
            )

    count = store.count(
        project_id=project_id,
        entry_type=etype,
        status=estatus,
    )
    return CountResponse(count=count)


@router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(
    project_id: str = Query(..., description="Project ID"),
) -> TimelineResponse:
    """Get a timeline of all entries for a project, sorted by created_at."""
    store = _get_store()
    entries = store.get_timeline(project_id)
    return TimelineResponse(
        project_id=project_id,
        entries=[EntryResponse(**entry.to_dict()) for entry in entries],
        total=len(entries),
    )
