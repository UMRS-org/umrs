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


def resolve_rail_principal(
    *,
    token: str | None,
    session_id: str,
    client_ip: str,
    broker: Any,
    ip_check: bool = True,
) -> RailPrincipal:
    """Pure resolver (no HTTP): raises RailAuthError with the named reason."""
    if not token:
        raise RailAuthError("token_invalid")
    lease = broker.lease_by_plaintext_token(token)
    if lease is None:
        raise RailAuthError("token_invalid")
    if lease.is_revoked:
        raise RailAuthError("lease_revoked")
    if lease.is_expired:
        raise RailAuthError("lease_expired")
    if lease.scope != "inference":
        raise RailAuthError("scope_mismatch")
    if lease.session_id != session_id:
        raise RailAuthError("sid_mismatch")
    if ip_check and lease.egress_ip and client_ip and client_ip != lease.egress_ip:
        raise RailAuthError("ip_mismatch")
    if not lease.owner_id:
        raise RailAuthError("token_invalid")
    return RailPrincipal(
        owner=str(lease.owner_id),
        session_id=session_id,
        lease_id=str(lease.lease_id),
        lease=lease,
    )
