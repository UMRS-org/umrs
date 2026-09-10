"""Workload identity REST API (SEC-P3a).

Provides:
- POST  /umrs/workload-identity              Issue a workload identity
- POST  /umrs/workload-identity/validate     Validate a token
- GET   /umrs/workload-identity/{id}         Get identity details
- GET   /umrs/workload-identity              List identities
- POST  /umrs/workload-identity/{id}/revoke  Revoke an identity
- POST  /umrs/workload-identity/cleanup      Cleanup expired identities
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.workload_identity import WorkloadIdentityIssuer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workload-identity", tags=["workload-identity"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_issuer: WorkloadIdentityIssuer | None = None


def get_issuer() -> WorkloadIdentityIssuer:
    global _issuer
    if _issuer is None:
        _issuer = WorkloadIdentityIssuer()
    return _issuer


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IssueRequest(BaseModel):
    job_id: str
    step_name: str
    scope: str
    ttl_seconds: int = Field(default=900, gt=0)
    tier: str = ""
    owner_id: str = ""
    metadata: dict[str, Any] | None = None


class IssueResponse(BaseModel):
    identity_id: str
    job_id: str
    step_name: str
    scope: str
    token: str  # plaintext — returned only once
    issued_at: str | None = None
    expires_at: str | None = None
    tier: str = ""
    owner_id: str = ""


class ValidateRequest(BaseModel):
    token: str


class IdentityResponse(BaseModel):
    identity_id: str
    job_id: str
    step_name: str
    scope: str
    token_hash: str
    issued_at: str | None = None
    expires_at: str | None = None
    tier: str = ""
    owner_id: str = ""
    is_revoked: bool = False
    is_valid: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=IssueResponse, status_code=status.HTTP_201_CREATED)
def issue_identity(req: IssueRequest):
    issuer = get_issuer()
    identity, token = issuer.issue(
        job_id=req.job_id,
        step_name=req.step_name,
        scope=req.scope,
        ttl_seconds=req.ttl_seconds,
        tier=req.tier,
        owner_id=req.owner_id,
        metadata=req.metadata,
    )
    d = identity.to_dict()
    return IssueResponse(
        identity_id=d["identity_id"],
        job_id=d["job_id"],
        step_name=d["step_name"],
        scope=d["scope"],
        token=token,
        issued_at=d["issued_at"],
        expires_at=d["expires_at"],
        tier=d["tier"],
        owner_id=d["owner_id"],
    )


@router.post("/validate", response_model=IdentityResponse)
def validate_token(req: ValidateRequest):
    issuer = get_issuer()
    identity = issuer.validate(req.token)
    if not identity:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return IdentityResponse(**identity.to_dict())


@router.get("", response_model=list[IdentityResponse])
def list_identities(
    job_id: str | None = Query(None),
    scope: str | None = Query(None),
    active_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
):
    issuer = get_issuer()
    results = issuer.list_identities(
        job_id=job_id,
        scope=scope,
        active_only=active_only,
        limit=limit,
    )
    return [IdentityResponse(**i.to_dict()) for i in results]


@router.get("/{identity_id}", response_model=IdentityResponse)
def get_identity(identity_id: str):
    issuer = get_issuer()
    identity = issuer.get_identity(identity_id)
    if not identity:
        raise HTTPException(status_code=404, detail="Identity not found")
    return IdentityResponse(**identity.to_dict())


@router.post("/{identity_id}/revoke")
def revoke_identity(identity_id: str):
    issuer = get_issuer()
    ok = issuer.revoke(identity_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Identity not found")
    return {"status": "revoked", "identity_id": identity_id}


@router.post("/cleanup")
def cleanup_expired():
    issuer = get_issuer()
    count = issuer.cleanup_expired()
    return {"status": "ok", "expired_removed": count}
