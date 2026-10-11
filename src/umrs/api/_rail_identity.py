"""Lease-token principal for the frontend-venue worker rail (design §1.1).

The sandbox worker holds exactly one credential: its session token (a
broker lease, hash-only server-side). The rail routes authenticate it here:

    Authorization: Bearer <token>
      → SecretBroker.lease_by_plaintext_token
      → valid · scope == "inference" · lease.session_id == {session_id} in path
      → (optional) request.client.host == lease.egress_ip

The owner is ``lease.owner_id``; bodies never carry identity. User JWTs
never reach these routes (the sidecar strips the gateway identity headers
and the rail never reads them), and lease tokens are rejected by the user
routes (they carry no gateway identity) — three principals, none
interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RailAuthError(Exception):
    """Named refusal reason (token_invalid | lease_revoked | lease_expired |
    scope_mismatch | sid_mismatch | ip_mismatch)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RailPrincipal:
    owner: str
    session_id: str
    lease_id: str
    lease: Any


def bearer_token(headers: Any) -> str | None:
    """``Authorization: Bearer <token>`` only — never api-key shapes here."""
    raw = headers.get("authorization") or headers.get("Authorization") or ""
    if not raw:
        return None
    parts = raw.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        return None
    return parts[1].strip()


def _lease_refusal(lease: Any, session_id: str, client_ip: str, ip_check: bool) -> str:
    """The named reason a lease may not ride this rail ("" = accepted),
    checked in the historical order."""
    checks = (
        (lease.is_revoked, "lease_revoked"),
        (lease.is_expired, "lease_expired"),
        (lease.scope != "inference", "scope_mismatch"),
        (lease.session_id != session_id, "sid_mismatch"),
        (_ip_mismatch(lease, client_ip, ip_check), "ip_mismatch"),
        (not lease.owner_id, "token_invalid"),
    )
    return next((reason for failed, reason in checks if failed), "")


def _ip_mismatch(lease: Any, client_ip: str, ip_check: bool) -> bool:
    return bool(
        ip_check and lease.egress_ip and client_ip and client_ip != lease.egress_ip
    )


def resolve_rail_principal(
    *,
    token: str | None,
    session_id: str,
    client_ip: str,
    broker: Any,
    ip_check: bool = True,
) -> RailPrincipal:
    """Pure resolver (no HTTP): raises RailAuthError with the named reason."""
    lease = broker.lease_by_plaintext_token(token) if token else None
    if lease is None:
        raise RailAuthError("token_invalid")
    reason = _lease_refusal(lease, session_id, client_ip, ip_check)
    if reason:
        raise RailAuthError(reason)
    return RailPrincipal(
        owner=str(lease.owner_id),
        session_id=session_id,
        lease_id=str(lease.lease_id),
        lease=lease,
    )


# ── rail attestation (ADR-092 slice 8) ─────────────────────────────────────
#
# The rail gw re-dispatches a sandbox's platform verb in-process with the
# lease owner's minted identity, which a route cannot tell from the owner's
# own client. A team plan run hosted on the server venue must be written by
# its worker only, so for that family the rail also registers a one-time,
# unguessable attestation naming the lease's session, valid only while the
# re-dispatched call is in flight, and sends its handle in one header. The
# trace routes resolve it from the same app's state; a client cannot forge
# one (the rail strips the header from what the sandbox sends, and a handle
# exists only for the duration of the rail's own in-process call).

RAIL_ATTESTATION_HEADER = "x-umrs-rail-attestation"
_ATTESTATIONS_ATTR = "rail_attestations"


def _registry(state: Any) -> dict[str, str]:
    registry = getattr(state, _ATTESTATIONS_ATTR, None)
    if registry is None:
        registry = {}
        setattr(state, _ATTESTATIONS_ATTR, registry)
    return registry


def attest(state: Any, session_id: str) -> str:
    """Register an attestation for ``session_id``; returns its handle."""
    import secrets

    handle = secrets.token_urlsafe(32)
    _registry(state)[handle] = session_id
    return handle


def revoke_attestation(state: Any, handle: str) -> None:
    _registry(state).pop(handle, None)


def attested_session(state: Any, headers: Any) -> str | None:
    """The rail session a request was attested for, or None (any other caller)."""
    handle = headers.get(RAIL_ATTESTATION_HEADER) if headers is not None else None
    if not handle:
        return None
    return _registry(state).get(str(handle))
