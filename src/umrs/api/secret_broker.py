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

from fastapi import APIRouter, Depends, HTTPException, Query, status

from .typed_errors import PlatformError, raise_typed
from pydantic import BaseModel

from umrs.knowledge.access_scope import CallerScope

from ..jobs.secret_broker import SecretBroker, SecretLease
from ._scope_identity import get_caller_scope

logger = logging.getLogger(__name__)

# G-6 (R0-1): every /umrs/secrets/* route requires a verified caller identity
# (401 without verified headers in strict mode — no anonymous envelope).
router = APIRouter(
    prefix="/umrs/secrets",
    tags=["secrets"],
    dependencies=[Depends(get_caller_scope)],
)


def _check_lease_visible(
    lease: SecretLease | None, caller: CallerScope, lease_id: str
) -> SecretLease:
    """404 without disclosing: admin sees all, users see their own leases only."""
    if lease is None:
        raise_typed(PlatformError.secretbroker("not_found", "Lease not found"))
    if not caller.is_admin and lease.owner_id != caller.user_id:
        raise_typed(PlatformError.secretbroker("not_found", "Lease not found"))
    return lease


def _visible_leases(
    leases: list[SecretLease], caller: CallerScope
) -> list[SecretLease]:
    if caller.is_admin:
        return leases
    return [lease for lease in leases if lease.owner_id == caller.user_id]


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_broker: SecretBroker | None = None


def get_broker() -> SecretBroker:
    """Process-global broker; R0-2 durability via env-driven factory.

    ``UMRS_SECRET_BROKER_BACKEND=pg|memory|auto`` (default auto — Postgres
    when the shared pool resolves, else in-memory).
    """
    global _broker
    if _broker is None:
        from ..jobs.secret_broker import build_default_broker

        _broker = build_default_broker()
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
    # inference-scope bindings (R0-2/R1; optional — absent on legacy calls)
    session_id: str = ""
    source_ids: list[str] = []
    default_source_id: str = ""
    egress_ip: str = ""
    project: str = ""
    max_lifetime_seconds: int | None = None


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
def issue_lease(
    req: IssueLeaseRequest, caller: CallerScope = Depends(get_caller_scope)
):
    # Owner pinning (R0-1): non-admin callers mint leases for themselves only;
    # admin may mint on behalf (jobs plane). The body field never wins across owners.
    owner_id = req.owner_id if caller.is_admin else (caller.user_id or req.owner_id)
    broker = get_broker()
    lease, secret_value = broker.issue_lease(
        job_id=req.job_id,
        secret_name=req.secret_name,
        scope=req.scope,
        lease_seconds=req.lease_seconds,
        bound_to_step=req.bound_to_step,
        tier=req.tier,
        owner_id=owner_id,
        session_id=req.session_id,
        source_ids=tuple(req.source_ids),
        default_source_id=req.default_source_id,
        egress_ip=req.egress_ip,
        project=req.project,
        max_lifetime_seconds=req.max_lifetime_seconds,
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
        raise_typed(PlatformError.secretbroker("unauthorized", "Invalid or expired secret"))
    return LeaseResponse(**lease.to_dict())


@router.get("/lease", response_model=list[LeaseResponse])
def list_leases(
    job_id: str | None = Query(None),
    scope: str | None = Query(None),
    active_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    caller: CallerScope = Depends(get_caller_scope),
):
    broker = get_broker()
    results = broker.list_leases(
        job_id=job_id,
        scope=scope,
        active_only=active_only,
        limit=limit,
    )
    return [
        LeaseResponse(**lease.to_dict()) for lease in _visible_leases(results, caller)
    ]


@router.get("/lease/{lease_id}", response_model=LeaseResponse)
def get_lease(lease_id: str, caller: CallerScope = Depends(get_caller_scope)):
    broker = get_broker()
    lease = _check_lease_visible(broker.get_lease(lease_id), caller, lease_id)
    return LeaseResponse(**lease.to_dict())


@router.post("/lease/{lease_id}/revoke")
def revoke_lease(lease_id: str, caller: CallerScope = Depends(get_caller_scope)):
    broker = get_broker()
    _check_lease_visible(broker.get_lease(lease_id), caller, lease_id)
    ok = broker.revoke_lease(lease_id)
    if not ok:
        raise_typed(PlatformError.secretbroker("not_found", "Lease not found"))
    return {"status": "revoked", "lease_id": lease_id}


@router.post("/lease/{lease_id}/renew")
def renew_lease(
    lease_id: str, req: RenewRequest, caller: CallerScope = Depends(get_caller_scope)
):
    broker = get_broker()
    _check_lease_visible(broker.get_lease(lease_id), caller, lease_id)
    ok = broker.renew_lease(lease_id, req.extend_seconds, req.actor)
    if not ok:
        raise_typed(
            PlatformError.secretbroker(
                "bad_request",
                "Lease not found, expired, or renewal exceeds max TTL"
            ),
        )
    return {"status": "renewed", "lease_id": lease_id}


@router.post("/job/{job_id}/revoke")
def revoke_job_leases(job_id: str, caller: CallerScope = Depends(get_caller_scope)):
    # Operator-class action: revoking every lease of a job is admin-only
    # (per-lease revoke above is the owner path).
    if not caller.is_admin:
        raise_typed(PlatformError.secretbroker("forbidden", "admin required"))
    broker = get_broker()
    count = broker.revoke_job_leases(job_id)
    return {"status": "revoked", "job_id": job_id, "leases_revoked": count}


@router.get("/audit", response_model=list[AuditEntryResponse])
def get_audit_trail(
    lease_id: str | None = Query(None),
    operation: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    caller: CallerScope = Depends(get_caller_scope),
):
    broker = get_broker()
    entries = broker.get_audit_trail(
        lease_id=lease_id,
        operation=operation,
        limit=limit,
    )
    if caller.is_admin:
        return [AuditEntryResponse(**e.to_dict()) for e in entries]
    visible_ids = {
        lease.lease_id
        for lease in _visible_leases(broker.list_leases(limit=5000), caller)
    }
    return [
        AuditEntryResponse(**e.to_dict()) for e in entries if e.lease_id in visible_ids
    ]


@router.post("/cleanup")
def cleanup_expired(caller: CallerScope = Depends(get_caller_scope)):
    if not caller.is_admin:
        raise_typed(PlatformError.secretbroker("forbidden", "admin required"))
    broker = get_broker()
    count = broker.cleanup_expired()
    return {"status": "ok", "expired_removed": count}
