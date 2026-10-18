"""UMRS API routes for M3.5 verification findings.

TASK-046: Persistence endpoints added so that chat-resolved findings
survive UMRS restarts.

Endpoints:
  POST /umrs/verification/chat/resolve  — Persist a resolved finding
  GET  /umrs/verification/findings      — List findings with filters
  PATCH /umrs/verification/findings/<key> — Transition finding status

Retired 2026-10-18 (hardening plan H-10.1): ``POST /umrs/verification/chat``
and ``GET /umrs/verification/sessions[/<id>]``. The chat reasoned over the
module-global M3.5 graph (project "default", shared with the retired
knowledge explorer) whatever project the caller had, so a project B caller
read project "default"'s nodes and graph paths. Its controller had no
session store, so the sessions routes always answered ``[]`` / ``null``.
No client calls them (the TUI computes its verification answers locally and
only posts ``/chat/resolve``) and the edge has no ``/gw/verification``
location. Per-project findings: ``GET /umrs/knowledge/findings``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from umrs.knowledge.verification_store import (
    TransitionError,
    VerificationStore,
)

from ._verified_headers import CallerIdentity, get_caller, get_verified_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/verification", tags=["verification"])

# Verification store — injected at startup if available (TASK-046)
_store: VerificationStore | None = None


def set_store(store: VerificationStore) -> None:
    """Inject the Postgres-backed verification store."""
    global _store
    _store = store


class ResolveFindingRequest(BaseModel):
    finding_key: str = Field(..., description="Unique key for the finding")
    report_id: str = Field("manual", description="Report ID")
    category: str = Field("verification_chat", description="Finding category")
    severity: str = Field("warning", description="Severity level")
    status: str = Field("resolved", description="Target status")


class TransitionFindingRequest(BaseModel):
    from_status: str = Field(..., description="Current status")
    to_status: str = Field(..., description="Target status")


# ── TASK-046: Persistence endpoints ──────────────────────────────────────


@router.post("/chat/resolve")
async def resolve_finding(
    req: ResolveFindingRequest,
    caller: CallerIdentity | None = Depends(get_caller),
) -> dict[str, str]:
    """Persist a chat-resolved finding to the knowledge store.

    The finding is upserted so subsequent transitions (open →
    acknowledged → resolved) can be applied via the findings API.
    Wave-6 F3: findings carry no project linkage, so when a verified
    caller context exists the write is admin-only.
    """
    if caller is not None and not caller.is_admin:
        raise_typed(
            PlatformError.verification(
                "forbidden", "verification finding writes are admin-scoped"
            ),
        )
    if _store is None:
        raise_typed(
            PlatformError.verification(
                "unavailable",
                "Verification store not initialized — cannot persist findings",
            ),
        )

    await _store.upsert_finding(
        finding_key=req.finding_key,
        report_id=req.report_id,
        category=req.category,
        severity=req.severity,
        title=f"Chat-resolved: {req.finding_key}",
        description=f"Status: {req.status}",
        evidence=req.finding_key,
        status=req.status,
    )

    return {"status": "persisted", "finding_key": req.finding_key}


@router.get("/findings")
async def list_findings(
    report_id: str | None = Query(None),
    status: str | None = Query(None, description="Filter by status"),
    severity: str | None = Query(None, description="Filter by severity"),
    limit: int = Query(200, ge=1, le=1000),
    caller: CallerIdentity | None = Depends(get_caller),
) -> list[dict[str, Any]]:
    """List verification findings with optional filters.

    Wave-6 F3: the findings store has no project dimension, so the
    dump is admin-scoped whenever a verified caller is present.
    """
    if caller is not None and not caller.is_admin:
        raise_typed(
            PlatformError.verification(
                "forbidden", "verification findings listing is admin-scoped"
            ),
        )
    if _store is None:
        raise_typed(
            PlatformError.verification(
                "unavailable",
                "Verification store not initialized — cannot list findings",
            ),
        )
    findings = await _store.list_findings(
        report_id=report_id,
        status=status,
        severity=severity,
        limit=limit,
    )
    return [
        {
            "finding_key": f.finding_key,
            "category": f.category,
            "severity": f.severity,
            "status": f.status,
            "title": f.title,
            "description": f.description,
            "evidence": f.evidence,
            "artifact_id": f.artifact_id,
        }
        for f in findings
    ]


@router.patch("/findings/{finding_key}")
async def transition_finding(
    finding_key: str,
    req: TransitionFindingRequest,
    caller: CallerIdentity = Depends(get_verified_caller),
) -> dict[str, Any]:
    """Transition a finding between states (open → acknowledged → resolved).

    Wave-6 F3: same admin scoping as the other findings endpoints.

    H-1 (2026-10-18): the transition ran whenever ``get_caller`` returned
    ``None``. Strict mode never does (a missing or forged tuple is 401), but
    an anonymous dev (mvp) caller could transition any finding (the store
    has no project dimension). A mutation now needs a verified admin in every
    mode; the admin flag is covered by the gateway HMAC.
    """
    if not caller.is_admin:
        raise_typed(
            PlatformError.verification(
                "forbidden", "verification finding transitions are admin-scoped"
            ),
        )
    if _store is None:
        raise_typed(
            PlatformError.verification(
                "unavailable",
                "Verification store not initialized — cannot transition findings",
            ),
        )
    try:
        finding = await _store.transition_finding(
            finding_key, req.from_status, req.to_status
        )
        return {
            "finding_key": finding.finding_key,
            "status": finding.status,
            "title": finding.title,
        }
    except TransitionError as exc:
        raise_typed(PlatformError.verification("bad_request", str(exc)))
