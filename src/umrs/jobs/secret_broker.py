"""Vault-style secret broker (SEC-P3b).

Provides short-lived, scoped secret leases for workloads:
- Secrets are issued with TTL and revoked on job completion
- No persistent plaintext storage — only hashed references
- Audit trail for all secret operations

This module lives in ``umrs.jobs`` because it manages durable state tied
to job secrets and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LEASE_SECONDS = 1800  # 30 minutes
MAX_LEASE_SECONDS = 7200  # 2 hours


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class SecretLease:
    """A time-bound secret lease for a workload."""

    lease_id: str
    job_id: str
    secret_name: str  # logical name (e.g., "s3_credentials", "litellm_key")
    scope: str  # inference | download | rag | mcp

    # Lifecycle
    issued_at: datetime = field(default_factory=_now)
    expires_at: datetime = field(default_factory=_now)
    is_revoked: bool = False
    revoked_at: datetime | None = None

    # Binding
    bound_to_step: str = ""
    tier: str = ""
    owner_id: str = ""

    # Audit
    secret_hash: str = ""  # SHA-256 of issued secret value
    access_count: int = 0
    last_accessed_at: datetime | None = None

    @property
    def is_expired(self) -> bool:
        return _now() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.is_revoked

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "job_id": self.job_id,
            "secret_name": self.secret_name,
            "scope": self.scope,
            "issued_at": _iso(self.issued_at),
            "expires_at": _iso(self.expires_at),
            "is_revoked": self.is_revoked,
            "revoked_at": _iso(self.revoked_at),
            "bound_to_step": self.bound_to_step,
            "tier": self.tier,
            "owner_id": self.owner_id,
            "secret_hash": self.secret_hash,
            "access_count": self.access_count,
            "last_accessed_at": _iso(self.last_accessed_at),
            "is_valid": self.is_valid,
        }


@dataclass
class AuditEntry:
    """Audit record for secret operations."""

    entry_id: str
    lease_id: str
    operation: str  # issue | access | revoke | expire | renew
    timestamp: datetime = field(default_factory=_now)
    actor: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "lease_id": self.lease_id,
            "operation": self.operation,
            "timestamp": _iso(self.timestamp),
            "actor": self.actor,
            "detail": self.detail,
        }


VALID_SCOPES = {"inference", "download", "rag", "mcp", "general"}
VALID_OPERATIONS = {"issue", "access", "revoke", "expire", "renew"}


class SecretBroker:
    """Vault-style secret broker with lease-based access.

    Issues time-bound secret leases with automatic expiry.
    All operations are logged in an audit trail.
    Plaintext secrets are returned only at issue time.
    """

    def __init__(self) -> None:
        self._leases: dict[str, SecretLease] = {}
        self._secret_index: dict[str, str] = {}  # secret_hash → lease_id
        self._audit: list[AuditEntry] = []
        self._counter = 0

    def issue_lease(
        self,
        job_id: str,
        secret_name: str,
        scope: str,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        bound_to_step: str = "",
        tier: str = "",
        owner_id: str = "",
    ) -> tuple[SecretLease, str]:
        """Issue a new secret lease.

        Args:
            job_id: The job this secret is bound to.
            secret_name: Logical name of the secret.
            scope: Access scope (inference, download, rag, mcp, general).
            lease_seconds: Lease duration in seconds (max 7200).
            bound_to_step: Step this secret is bound to.
            tier: Deployment tier.
            owner_id: Owner of the workload.

        Returns:
            Tuple of (SecretLease, plaintext_secret).
            The plaintext secret is returned only once.

        Raises:
            ValueError: If scope is invalid or lease exceeds maximum.
        """
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid scope: {scope!r}; must be one of {VALID_SCOPES}")
        if lease_seconds <= 0 or lease_seconds > MAX_LEASE_SECONDS:
            raise ValueError(
                f"Lease must be 1-{MAX_LEASE_SECONDS} seconds, got {lease_seconds}"
            )

        secret_value = secrets.token_urlsafe(48)
        secret_hash = hashlib.sha256(secret_value.encode()).hexdigest()

        now = _now()
        lease = SecretLease(
            lease_id=f"lease-{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            secret_name=secret_name,
            scope=scope,
            issued_at=now,
            expires_at=now + timedelta(seconds=lease_seconds),
            bound_to_step=bound_to_step,
            tier=tier,
            owner_id=owner_id,
            secret_hash=secret_hash,
        )

        self._leases[lease.lease_id] = lease
        self._secret_index[secret_hash] = lease.lease_id
        self._audit_log(
            lease.lease_id,
            "issue",
            owner_id,
            f"scope={scope} ttl={lease_seconds}s name={secret_name}",
        )

        logger.info(
            "Secret lease issued: %s (job=%s name=%s scope=%s ttl=%ds)",
            lease.lease_id,
            job_id,
            secret_name,
            scope,
            lease_seconds,
        )
        return lease, secret_value

    def validate_secret(self, secret_value: str) -> SecretLease | None:
        """Validate a secret value and return its lease if valid.

        Also increments the access count and logs the access.
        Returns None if the secret is unknown, expired, or revoked.
        """
        secret_hash = hashlib.sha256(secret_value.encode()).hexdigest()
        lease_id = self._secret_index.get(secret_hash)
        if lease_id is None:
            return None

        lease = self._leases.get(lease_id)
        if lease is None:
            return None

        if not lease.is_valid:
            return None

        lease.access_count += 1
        lease.last_accessed_at = _now()
        self._audit_log(lease_id, "access", "", f"count={lease.access_count}")
        return lease

    def revoke_lease(self, lease_id: str, actor: str = "") -> bool:
        """Revoke a secret lease before expiry."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return False
        lease.is_revoked = True
        lease.revoked_at = _now()
        self._audit_log(lease_id, "revoke", actor, "manual revocation")
        logger.info("Secret lease revoked: %s by %s", lease_id, actor or "system")
        return True

    def revoke_job_leases(self, job_id: str, actor: str = "") -> int:
        """Revoke all leases for a job (on job completion/failure)."""
        count = 0
        for lease in self._leases.values():
            if lease.job_id == job_id and lease.is_valid:
                lease.is_revoked = True
                lease.revoked_at = _now()
                self._audit_log(
                    lease.lease_id,
                    "revoke",
                    actor,
                    f"job completion: {job_id}",
                )
                count += 1
        if count:
            logger.info("Revoked %d leases for job %s", count, job_id)
        return count

    def renew_lease(
        self,
        lease_id: str,
        extend_seconds: int,
        actor: str = "",
    ) -> bool:
        """Extend a lease's expiry time.

        Cannot exceed MAX_LEASE_SECONDS from the original issue time.
        """
        lease = self._leases.get(lease_id)
        if lease is None or not lease.is_valid:
            return False

        new_expiry = lease.expires_at + timedelta(seconds=extend_seconds)
        max_expiry = lease.issued_at + timedelta(seconds=MAX_LEASE_SECONDS)
        if new_expiry > max_expiry:
            return False

        lease.expires_at = new_expiry
        self._audit_log(lease_id, "renew", actor, f"extended by {extend_seconds}s")
        return True

    def get_lease(self, lease_id: str) -> SecretLease | None:
        return self._leases.get(lease_id)

    def list_leases(
        self,
        job_id: str | None = None,
        scope: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[SecretLease]:
        """List secret leases with optional filters."""
        results = list(self._leases.values())
        if job_id:
            results = [l for l in results if l.job_id == job_id]
        if scope:
            results = [l for l in results if l.scope == scope]
        if active_only:
            results = [l for l in results if l.is_valid]
        results.sort(key=lambda l: l.issued_at, reverse=True)
        return results[:limit]

    def get_audit_trail(
        self,
        lease_id: str | None = None,
        operation: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit trail with optional filters."""
        results = list(self._audit)
        if lease_id:
            results = [e for e in results if e.lease_id == lease_id]
        if operation:
            results = [e for e in results if e.operation == operation]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    def cleanup_expired(self) -> int:
        """Remove expired leases from memory. Returns count removed."""
        expired = [lid for lid, lease in self._leases.items() if lease.is_expired]
        for lid in expired:
            lease = self._leases.pop(lid)
            self._secret_index.pop(lease.secret_hash, None)
            self._audit_log(lid, "expire", "system", "ttl expired")
        if expired:
            logger.info("Cleaned up %d expired secret leases", len(expired))
        return len(expired)

    # ── Internal ────────────────────────────────────────────

    def _audit_log(
        self,
        lease_id: str,
        operation: str,
        actor: str,
        detail: str,
    ) -> None:
        self._counter += 1
        self._audit.append(
            AuditEntry(
                entry_id=f"aud-{self._counter:06d}",
                lease_id=lease_id,
                operation=operation,
                actor=actor,
                detail=detail,
            )
        )
