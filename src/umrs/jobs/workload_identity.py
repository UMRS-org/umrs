"""Short-lived workload identity (SEC-P3a).

Issues ephemeral identity tokens for container workloads with:
- Short TTL (default 15 minutes, configurable)
- Scope bound to a specific job/step
- Automatic expiry — no revocation needed

This module lives in ``umrs.jobs`` because it manages durable state tied
to job identities and does not import from ``umrs.api`` or ``umrs.server``.
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

DEFAULT_TTL_SECONDS = 900  # 15 minutes
MAX_TTL_SECONDS = 3600  # 1 hour hard cap


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class WorkloadIdentity:
    """An ephemeral identity token for a container workload."""

    identity_id: str
    job_id: str
    step_name: str
    scope: str  # inference | download | rag | mcp
    token_hash: str  # SHA-256 of the bearer token (token itself not stored)
    issued_at: datetime = field(default_factory=_now)
    expires_at: datetime = field(default_factory=_now)
    tier: str = ""  # system | project | user
    owner_id: str = ""
    is_revoked: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return _now() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        return not self.is_expired and not self.is_revoked

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "job_id": self.job_id,
            "step_name": self.step_name,
            "scope": self.scope,
            "token_hash": self.token_hash,
            "issued_at": _iso(self.issued_at),
            "expires_at": _iso(self.expires_at),
            "tier": self.tier,
            "owner_id": self.owner_id,
            "is_revoked": self.is_revoked,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


VALID_SCOPES = {"inference", "download", "rag", "mcp", "general"}


class WorkloadIdentityIssuer:
    """Issues and validates short-lived workload identity tokens.

    Tokens are cryptographically random, stored as SHA-256 hashes.
    The plaintext token is returned only once at issue time.
    """

    def __init__(self) -> None:
        self._identities: dict[str, WorkloadIdentity] = {}
        self._token_index: dict[str, str] = {}  # token_hash → identity_id

    def issue(
        self,
        job_id: str,
        step_name: str,
        scope: str,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        tier: str = "",
        owner_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[WorkloadIdentity, str]:
        """Issue a new workload identity token.

        Args:
            job_id: The job this identity is bound to.
            step_name: The step this identity is bound to.
            scope: The access scope (inference, download, rag, mcp, general).
            ttl_seconds: Token lifetime in seconds (max 3600).
            tier: Deployment tier (system, project, user).
            owner_id: Owner of the workload.
            metadata: Additional metadata to attach.

        Returns:
            Tuple of (WorkloadIdentity, plaintext_token).
            The plaintext token is returned only once.

        Raises:
            ValueError: If scope is invalid or TTL exceeds maximum.
        """
        if scope not in VALID_SCOPES:
            raise ValueError(
                f"Invalid scope: {scope!r}; must be one of {VALID_SCOPES}"
            )
        if ttl_seconds <= 0 or ttl_seconds > MAX_TTL_SECONDS:
            raise ValueError(
                f"TTL must be 1-{MAX_TTL_SECONDS} seconds, got {ttl_seconds}"
            )

        # Generate cryptographic token
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        now = _now()
        identity = WorkloadIdentity(
            identity_id=f"wid-{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            step_name=step_name,
            scope=scope,
            token_hash=token_hash,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            tier=tier,
            owner_id=owner_id,
            metadata=metadata or {},
        )

        self._identities[identity.identity_id] = identity
        self._token_index[token_hash] = identity.identity_id

        logger.info(
            "Workload identity issued: %s (job=%s step=%s scope=%s ttl=%ds)",
            identity.identity_id, job_id, step_name, scope, ttl_seconds,
        )
        return identity, token

    def validate(self, token: str) -> WorkloadIdentity | None:
        """Validate a bearer token and return its identity if valid.

        Returns None if the token is unknown, expired, or revoked.
        """
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        identity_id = self._token_index.get(token_hash)
        if identity_id is None:
            return None

        identity = self._identities.get(identity_id)
        if identity is None:
            return None

        if not identity.is_valid:
            return None

        return identity

    def revoke(self, identity_id: str) -> bool:
        """Revoke a workload identity before expiry."""
        identity = self._identities.get(identity_id)
        if identity is None:
            return False
        identity.is_revoked = True
        logger.info("Workload identity revoked: %s", identity_id)
        return True

    def get_identity(self, identity_id: str) -> WorkloadIdentity | None:
        return self._identities.get(identity_id)

    def list_identities(
        self,
        job_id: str | None = None,
        scope: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[WorkloadIdentity]:
        """List workload identities with optional filters."""
        results = list(self._identities.values())
        if job_id:
            results = [i for i in results if i.job_id == job_id]
        if scope:
            results = [i for i in results if i.scope == scope]
        if active_only:
            results = [i for i in results if i.is_valid]
        results.sort(key=lambda i: i.issued_at, reverse=True)
        return results[:limit]

    def cleanup_expired(self) -> int:
        """Remove expired identities from memory. Returns count removed."""
        expired = [
            iid for iid, identity in self._identities.items()
            if identity.is_expired
        ]
        for iid in expired:
            identity = self._identities.pop(iid)
            self._token_index.pop(identity.token_hash, None)
        if expired:
            logger.info("Cleaned up %d expired workload identities", len(expired))
        return len(expired)
