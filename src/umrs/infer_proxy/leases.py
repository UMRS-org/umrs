"""Lease extraction + authorization for the inference proxy (R1-2, D-6).

Every request: session token → SHA-256 → live lease read (optionally via
asyncpg when the server runs with one; the sync fallback maps the R0-2
broker's pool). No lease caching: revocation must deny the NEXT request.

Checks (design §2.3): exists · not expired · not revoked ·
``session_id == <sid>`` · client IP == ``lease.egress_ip`` · source ∈
``lease.source_ids`` (default source when no override header).

Lease kinds (ADR-062 amendment 2, S-D37-3): a sandbox lease reaches the
proxy as ``/s/{sid}/...`` from its session's egress sidecar; a
``kind=workstation`` lease (``tier == "workstation"``, minted by
``POST /umrs/infer/leases``) reaches it as ``/w/{sid}/...`` through the edge
route ``/gw/infer/``. Each route admits its own kind only, so an edge caller
can never present a sandbox token and a workstation token never rides the
sandbox route. A workstation lease must name its owner and project.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from umrs.jobs.secret_broker import SecretLease


@dataclass(frozen=True)
class LeaseVerdict:
    ok: bool
    # token_invalid | lease_expired | lease_revoked | lease_kind_mismatch |
    # lease_unbound | sid_mismatch | ip_mismatch | source_not_in_lease
    reason: str = ""
    lease: SecretLease | None = None
    source_id: str = ""

    def deny(self) -> bool:
        return not self.ok


def extract_session_token(headers: Any) -> str | None:
    """First of Authorization:Bearer | x-api-key | api-key | x-goog-api-key."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:
            return token
    for name in ("x-api-key", "api-key", "x-goog-api-key"):
        value = headers.get(name)
        if value:
            return value.strip() or None
    return None


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _expired(lease: SecretLease) -> bool:
    from umrs.jobs.secret_broker import _now

    return _now() >= lease.expires_at


def check_lease(
    lease: SecretLease | None,
    *,
    sid: str,
    client_ip: str,
    source_override: str | None,
) -> LeaseVerdict:
    """Pure verdict — no I/O. ``lease`` None = unknown token."""
    if lease is None:
        return LeaseVerdict(ok=False, reason="token_invalid")
    if lease.is_revoked:
        return LeaseVerdict(ok=False, reason="lease_revoked", lease=lease)
    if _expired(lease):
        return LeaseVerdict(ok=False, reason="lease_expired", lease=lease)
    if lease.session_id != sid:
        return LeaseVerdict(ok=False, reason="sid_mismatch", lease=lease)
    if lease.egress_ip and client_ip != lease.egress_ip:
        return LeaseVerdict(ok=False, reason="ip_mismatch", lease=lease)
    source_id = source_override or lease.default_source_id
    if source_id not in lease.source_ids:
        return LeaseVerdict(ok=False, reason="source_not_in_lease", lease=lease)
    return LeaseVerdict(ok=True, lease=lease, source_id=source_id)


# The workstation lease kind (S-D37-3). Stored in ``secret_leases.tier``
# (no schema change); ``secret_name`` names it for the audit trail.
WORKSTATION_KIND = "workstation"
WORKSTATION_SECRET_NAME = "workstation-inference"


def is_workstation_lease(lease: SecretLease | None) -> bool:
    return lease is not None and lease.tier == WORKSTATION_KIND


def check_route_lease(
    lease: SecretLease | None,
    *,
    workstation: bool,
    sid: str,
    client_ip: str,
    source_override: str | None,
) -> LeaseVerdict:
    """``check_lease`` plus the route's lease kind (``/w/`` vs ``/s/``).

    A workstation lease must also carry its owner and project: the class
    gate and the source-ownership check read them on every request.
    """
    verdict = check_lease(
        lease, sid=sid, client_ip=client_ip, source_override=source_override
    )
    if verdict.deny():
        return verdict
    if is_workstation_lease(verdict.lease) != workstation:
        return LeaseVerdict(ok=False, reason="lease_kind_mismatch", lease=lease)
    if workstation and not _names_owner_and_project(verdict.lease):
        return LeaseVerdict(ok=False, reason="lease_unbound", lease=lease)
    return verdict


def _names_owner_and_project(lease: SecretLease | None) -> bool:
    return bool(lease and lease.owner_id and lease.project)


def lease_by_token(pool: Any, token: str) -> SecretLease | None:
    """Read the live lease directly from Postgres (no cache — D-6)."""
    from umrs.jobs.secret_broker import SecretBroker

    if pool is None:
        return None
    return SecretBroker(db_pool=pool).lease_by_plaintext_token(token)
