"""Secret broker REST API (SEC-P3b).

Provides:
- POST  /umrs/secrets/lease                  Issue a secret lease
- POST  /umrs/secrets/validate               Validate a secret
- GET   /umrs/secrets/lease/{lease_id}       Get lease details
- GET   /umrs/secrets/lease                  List leases
- POST  /umrs/secrets/lease/{id}/revoke      Revoke a lease
- POST  /umrs/secrets/lease/{id}/renew       Renew a lease
- POST  /umrs/secrets/job/{job_id}/revoke    Revoke all job leases
- GET   /umrs/secrets/audit                  Get audit trail
- POST  /umrs/secrets/cleanup                Cleanup expired leases
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.secret_broker import SecretBroker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/secrets", tags=["secrets"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_broker: SecretBroker | None = None


def get_broker() -> SecretBroker:
    global _broker
    if _broker is None:
        _broker = SecretBroker()
    return _broker


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IssueLeaseRequest(BaseModel):
    job_id: str
    secret_name: str
    scope: str
    lease_seconds: int = 1800
    bound_to_step: str = ""
    tier: str = ""
    owner_id: str = ""


class IssueLeaseResponse(BaseModel):
    lease_id: str
    job_id: str
    secret_name: str
    scope: str
    secret_value: str  # plaintext — returned only once
    issued_at: str | None = None
    expires_at: str | None = None
    bound_to_step: str = ""
    tier: str = ""
    owner_id: str = ""


class ValidateSecretRequest(BaseModel):
    secret_value: str


class LeaseResponse(BaseModel):
    lease_id: str
    job_id: str
    secret_name: str
    scope: str
    issued_at: str | None = None
    expires_at: str | None = None
    is_revoked: bool = False
    revoked_at: str | None = None
    bound_to_step: str = ""
    tier: str = ""
    owner_id: str = ""
    secret_hash: str = ""
    access_count: int = 0
    last_accessed_at: str | None = None
    is_valid: bool = True


class RenewRequest(BaseModel):
    extend_seconds: int
    actor: str = ""


class AuditEntryResponse(BaseModel):
    entry_id: str
    lease_id: str
    operation: str
    timestamp: str | None = None
    actor: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/lease", response_model=IssueLeaseResponse, status_code=status.HTTP_201_CREATED
)
def issue_lease(req: IssueLeaseRequest):
    broker = get_broker()
    lease, secret_value = broker.issue_lease(
        job_id=req.job_id,
        secret_name=req.secret_name,
        scope=req.scope,
        lease_seconds=req.lease_seconds,
        bound_to_step=req.bound_to_step,
        tier=req.tier,
        owner_id=req.owner_id,
    )
    d = lease.to_dict()
    return IssueLeaseResponse(
        lease_id=d["lease_id"],
        job_id=d["job_id"],
        secret_name=d["secret_name"],
        scope=d["scope"],
        secret_value=secret_value,
        issued_at=d["issued_at"],
        expires_at=d["expires_at"],
        bound_to_step=d["bound_to_step"],
        tier=d["tier"],
        owner_id=d["owner_id"],
    )


@router.post("/validate", response_model=LeaseResponse)
def validate_secret(req: ValidateSecretRequest):
    broker = get_broker()
    lease = broker.validate_secret(req.secret_value)
    if not lease:
        raise HTTPException(status_code=401, detail="Invalid or expired secret")
    return LeaseResponse(**lease.to_dict())


@router.get("/lease", response_model=list[LeaseResponse])
def list_leases(
    job_id: str | None = Query(None),
    scope: str | None = Query(None),
    active_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
):
    broker = get_broker()
    results = broker.list_leases(
        job_id=job_id,
        scope=scope,
        active_only=active_only,
        limit=limit,
    )
    return [LeaseResponse(**l.to_dict()) for l in results]


@router.get("/lease/{lease_id}", response_model=LeaseResponse)
def get_lease(lease_id: str):
    broker = get_broker()
    lease = broker.get_lease(lease_id)
    if not lease:
        raise HTTPException(status_code=404, detail="Lease not found")
    return LeaseResponse(**lease.to_dict())


@router.post("/lease/{lease_id}/revoke")
def revoke_lease(lease_id: str):
    broker = get_broker()
    ok = broker.revoke_lease(lease_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Lease not found")
    return {"status": "revoked", "lease_id": lease_id}


@router.post("/lease/{lease_id}/renew")
def renew_lease(lease_id: str, req: RenewRequest):
    broker = get_broker()
    ok = broker.renew_lease(lease_id, req.extend_seconds, req.actor)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Lease not found, expired, or renewal exceeds max TTL",
        )
    return {"status": "renewed", "lease_id": lease_id}


@router.post("/job/{job_id}/revoke")
def revoke_job_leases(job_id: str):
    broker = get_broker()
    count = broker.revoke_job_leases(job_id)
    return {"status": "revoked", "job_id": job_id, "leases_revoked": count}


@router.get("/audit", response_model=list[AuditEntryResponse])
def get_audit_trail(
    lease_id: str | None = Query(None),
    operation: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    broker = get_broker()
    entries = broker.get_audit_trail(
        lease_id=lease_id,
        operation=operation,
        limit=limit,
    )
    return [AuditEntryResponse(**e.to_dict()) for e in entries]


@router.post("/cleanup")
def cleanup_expired():
    broker = get_broker()
    count = broker.cleanup_expired()
    return {"status": "ok", "expired_removed": count}
