"""UMRS API routes for evidence tracking (aligned with the K1 satellite).

Durable store auto-resolves via ``umrs.evidence.build_default_store()`` —
Postgres when a database is configured, otherwise the in-memory reference.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.evidence import (
    Evidence,
    EvidenceSourceType,
    EvidenceStatus,
)

from ._verified_headers import CallerIdentity, get_caller

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/evidence", tags=["evidence"])

_store: Any = None


def set_store(store) -> None:
    global _store
    _store = store


def _get_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Evidence store not initialized")
    return _store


class CreateEvidenceRequest(BaseModel):
    project_id: str
    claim_id: str
    source_type: str = Field("manual")
    source_id: str = ""
    content: str = ""
    content_uri: str = ""
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    citation: str = ""


class UpdateStatusRequest(BaseModel):
    status: str


def _to_response(e: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": e.evidence_id,
        "project_id": e.project_id,
        "claim_id": e.claim_id,
        "source_type": e.source_type.value,
        "source_id": e.source_id,
        "content": e.content,
        "content_uri": e.content_uri,
        "relevance_score": e.relevance_score,
        "citation": e.citation,
        "status": e.status.value,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
    }


@router.post("", summary="Register evidence against a claim")
def create_evidence(
    req: CreateEvidenceRequest,
    caller: CallerIdentity | None = Depends(get_caller),
):
    project_id = req.project_id
    if caller is not None:
        project_id = caller.bind_for_write(req.project_id)
    evidence = Evidence(
        project_id=project_id,
        claim_id=req.claim_id,
        source_type=EvidenceSourceType(req.source_type),
        source_id=req.source_id,
        content=req.content,
        content_uri=req.content_uri,
        relevance_score=req.relevance_score,
        citation=req.citation,
    )
    evidence_id = _get_store().add_evidence(evidence)
    return {"evidence_id": evidence_id}


@router.get("/count", summary="Count evidence with optional scope filter")
def count_evidence(
    project_id: str | None = Query(None),
    source_type: str | None = Query(None),
    status: str | None = Query(None),
    caller: CallerIdentity | None = Depends(get_caller),
):
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    s_type = EvidenceSourceType(source_type) if source_type else None
    s_status = EvidenceStatus(status) if status else None
    return {
        "count": _get_store().count(
            project_id=project_id, source_type=s_type, status=s_status
        )
    }


@router.get("/search", summary="Full-text search over evidence content/citation")
def search_evidence(
    query: str,
    project_id: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    caller: CallerIdentity | None = Depends(get_caller),
):
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    entries = _get_store().search_evidence(query, project_id=project_id, limit=limit)
    return {"evidence": [_to_response(e) for e in entries]}


@router.get("/claims/{claim_id}", summary="All evidence for a claim")
def evidence_for_claim(
    claim_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
):
    entries = _get_store().get_evidence_for_claim(claim_id)
    if caller is not None and not caller.is_admin:
        entries = [e for e in entries if e.project_id == caller.project_id]
    return {"evidence": [_to_response(e) for e in entries]}


@router.get("/{evidence_id}", summary="View single evidence item")
def get_evidence(
    evidence_id: str,
    caller: CallerIdentity | None = Depends(get_caller),
):
    entry = _get_store().get_evidence(evidence_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Evidence {evidence_id} not found")
    if caller is not None:
        caller.assert_can_access(entry.project_id)
    return _to_response(entry)


@router.get("", summary="List evidence with filters")
def list_evidence(
    project_id: str | None = Query(None),
    source_type: str | None = Query(None),
    status: str | None = Query(None),
    claim_id: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    caller: CallerIdentity | None = Depends(get_caller),
):
    if caller is not None:
        project_id = caller.scope_filter(project_id)
    s_type = EvidenceSourceType(source_type) if source_type else None
    s_status = EvidenceStatus(status) if status else None
    entries = _get_store().list_evidence(
        project_id=project_id,
        source_type=s_type,
        status=s_status,
        claim_id=claim_id,
        limit=limit,
    )
    return {"evidence": [_to_response(e) for e in entries]}


@router.put("/{evidence_id}/status", summary="Retract or change evidence status")
def update_status(
    evidence_id: str,
    req: UpdateStatusRequest,
    caller: CallerIdentity | None = Depends(get_caller),
):
    if caller is not None:
        existing = _get_store().get_evidence(evidence_id)
        if existing is None:
            raise HTTPException(
                status_code=404, detail=f"Evidence {evidence_id} not found"
            )
        caller.assert_can_access(existing.project_id)
    found = _get_store().update_status(evidence_id, EvidenceStatus(req.status.lower()))
    if not found:
        raise HTTPException(status_code=404, detail=f"Evidence {evidence_id} not found")
    entry = _get_store().get_evidence(evidence_id)
    return _to_response(entry)
