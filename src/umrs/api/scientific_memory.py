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

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from umrs.scientific_memory import (
    InMemoryScientificMemoryStore,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
)

from ._verified_headers import CallerIdentity, get_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/scientific-memory", tags=["scientific-memory"])

# Store instance — injected at startup
_store: InMemoryScientificMemoryStore | None = None


def set_store(store: InMemoryScientificMemoryStore) -> None:
    global _store
    _store = store


def _get_store() -> InMemoryScientificMemoryStore:
    if _store is None:
        raise_typed(
            PlatformError.scimem(
                "unavailable", "Scientific memory store not initialized"
            )
        )
    return _store


# ── Request/Response Models ──────────────────────────────────────────────


class AddEntryRequest(BaseModel):
    project_id: str = Field(
        "", description="Project ID (defaults to caller's verified project)"
    )
    entry_type: str = Field(
        "note", description="Entry type (generated_code, experiment_log, etc.)"
    )
    source: str = Field(
        "system", description="Who created the entry (user, agent, system)"
    )
    content: str = Field("", description="Entry content")
    content_uri: str = Field("", description="S3 URI if content is large")
    summary: str = Field("", description="Short summary")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    confidence: float = Field(0.0, description="Confidence score (0.0–1.0)")
    references: list[str] = Field(
        default_factory=list, description="Other entry_ids this entry references"
    )
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
    status: str = Field(
        ..., description="New status (active, archived, deprecated, retracted)"
    )


class CountResponse(BaseModel):
    count: int


class TimelineResponse(BaseModel):
    project_id: str
    entries: list[EntryResponse]
    total: int


# ── Routes ────────────────────────────────────────────────────────────────


@router.post("/entries", response_model=EntryResponse)
async def add_entry(
    req: AddEntryRequest,
    caller: CallerIdentity | None = Depends(get_caller),
) -> EntryResponse:
    """Add a new memory entry.

    Wave-6 F4: project + actor are pinned to the verified caller —
    self-declared `created_by` forging another user is impossible when an
    identity context is present.
    """
    store = _get_store()
    try:
        entry_type = MemoryEntryType(req.entry_type)
    except ValueError:
        raise_typed(
            PlatformError.scimem("bad_request", f"Invalid entry type: {req.entry_type}")
        )
    project_id = req.project_id
    created_by = req.created_by
    if caller is not None:
        project_id = caller.bind_for_write(req.project_id)
        created_by = caller.user_id
    entry = MemoryEntry(
        project_id=project_id,
        entry_type=entry_type,
        source=req.source,
        content=req.content,
        content_uri=req.content_uri,
        summary=req.summary,
        metadata=req.metadata,
        tags=req.tags,
        confidence=req.confidence,
        references=req.references,
        created_by=created_by,
    )
    entry_id = store.add(entry)
    entry = store.get(entry_id)
    return EntryResponse(**entry.to_dict())


@router.get("/entries/{entry_id}", response_model=EntryResponse)
async def get_entry(
    entry_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
) -> EntryResponse:
    """Get a memory entry by ID."""
    store = _get_store()
    entry = store.get(entry_id)
    if entry is None:
        raise_typed(PlatformError.scimem("not_found", f"Entry {entry_id} not found"))
    if caller is not None:
        caller.assert_can_access(entry.project_id)
    return EntryResponse(**entry.to_dict())


@router.get("/entries", response_model=EntryListResponse)
async def list_entries(
    project_id: str | None = Query(None, description="Filter by project ID"),
    entry_type: str | None = Query(None, description="Filter by entry type"),
    status: str | None = Query(None, description="Filter by status"),
    experiment_id: str | None = Query(None, description="Filter by experiment ID"),
    limit: int = Query(100, ge=1, le=1000),
    caller: CallerIdentity | None = Depends(get_caller),
) -> EntryListResponse:
    """List memory entries with optional filters."""
    store = _get_store()
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    # Parse filters
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise_typed(
                PlatformError.scimem("bad_request", f"Invalid entry type: {entry_type}")
            )

    estatus: MemoryEntryStatus | None = None
    if status:
        try:
            estatus = MemoryEntryStatus(status)
        except ValueError:
            raise_typed(
                PlatformError.scimem("bad_request", f"Invalid status: {status}")
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
    caller: CallerIdentity | None = Depends(get_caller),
) -> SearchResponse:
    """Search memory entries by content, summary, or tags."""
    store = _get_store()
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise_typed(
                PlatformError.scimem("bad_request", f"Invalid entry type: {entry_type}")
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
    caller: CallerIdentity | None = Depends(get_caller),
) -> dict[str, Any]:
    """Update the status of a memory entry."""
    store = _get_store()
    if caller is not None:
        existing = store.get(entry_id)
        if existing is None:
            raise_typed(
                PlatformError.scimem("not_found", f"Entry {entry_id} not found")
            )
        caller.assert_can_access(existing.project_id)
    try:
        status = MemoryEntryStatus(req.status)
    except ValueError:
        raise_typed(
            PlatformError.scimem("bad_request", f"Invalid status: {req.status}")
        )
    result = store.update_status(entry_id, status)
    if not result:
        raise_typed(PlatformError.scimem("not_found", f"Entry {entry_id} not found"))
    return {"updated": True, "entry_id": entry_id, "status": req.status}


@router.get("/count", response_model=CountResponse)
async def count_entries(
    project_id: str | None = Query(None, description="Filter by project ID"),
    entry_type: str | None = Query(None, description="Filter by entry type"),
    status: str | None = Query(None, description="Filter by status"),
    caller: CallerIdentity | None = Depends(get_caller),
) -> CountResponse:
    """Count memory entries with optional filters."""
    store = _get_store()
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    etype: MemoryEntryType | None = None
    if entry_type:
        try:
            etype = MemoryEntryType(entry_type)
        except ValueError:
            raise_typed(
                PlatformError.scimem("bad_request", f"Invalid entry type: {entry_type}")
            )

    estatus: MemoryEntryStatus | None = None
    if status:
        try:
            estatus = MemoryEntryStatus(status)
        except ValueError:
            raise_typed(
                PlatformError.scimem("bad_request", f"Invalid status: {status}")
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
    caller: CallerIdentity | None = Depends(get_caller),
) -> TimelineResponse:
    """Get a timeline of all entries for a project, sorted by created_at."""
    store = _get_store()
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    entries = store.get_timeline(project_id)
    return TimelineResponse(
        project_id=project_id,
        entries=[EntryResponse(**entry.to_dict()) for entry in entries],
        total=len(entries),
    )
