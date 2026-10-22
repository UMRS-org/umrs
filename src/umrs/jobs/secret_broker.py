"""Vault-style secret broker (SEC-P3b, R0-2 durable leases).

Provides short-lived, scoped secret leases for workloads:
- Secrets are issued with TTL and revoked on job completion
- No persistent plaintext storage — only hashed references
- Audit trail for all secret operations
- Since 0085: leases + audit persist to Postgres when a pool is wired
  (in-memory stays the default/test mode; restart semantics verified by
  the dual-backend test suite)

This module lives in ``umrs.jobs`` because it manages durable state tied
to job secrets and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover - psycopg2 is a runtime dep
    RealDictCursor = None  # type: ignore[assignment]

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
    scope: str  # inference | download | rag | mcp | general

    # Lifecycle
    issued_at: datetime = field(default_factory=_now)
    expires_at: datetime = field(default_factory=_now)
    is_revoked: bool = False
    revoked_at: datetime | None = None

    # Binding
    bound_to_step: str = ""
    tier: str = ""
    owner_id: str = ""

    # Inference-scope bindings (R0-2)
    session_id: str = ""
    source_ids: tuple[str, ...] = ()
    default_source_id: str = ""
    egress_ip: str = ""
    project: str = ""
    max_lifetime_at: datetime | None = None  # hard cap across renewals
    # The issuing site (0124): sites share umrs_db, accounts are per site.
    venue: str = ""

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
            "session_id": self.session_id,
            "source_ids": list(self.source_ids),
            "default_source_id": self.default_source_id,
            "egress_ip": self.egress_ip,
            "project": self.project,
            "max_lifetime_at": _iso(self.max_lifetime_at),
            "venue": self.venue,
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


def _to_uuid(value: str) -> str | None:
    """UUID string or None — source ids are UUIDs in the sealed store."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return None


class SecretBroker:
    """Vault-style secret broker with lease-based access.

    Issues time-bound secret leases with automatic expiry.
    All operations are logged in an audit trail.
    Plaintext secrets are returned only at issue time.
    """

    def __init__(self, db_pool: Any | None = None, *, venue: str | None = None) -> None:
        self._leases: dict[str, SecretLease] = {}
        self._secret_index: dict[str, str] = {}  # secret_hash → lease_id
        self._audit: list[AuditEntry] = []
        self._counter = 0
        self._pool = db_pool  # psycopg2-style pool; None = in-memory mode
        self._venue = venue  # None = this process's site name, read on first use

    @property
    def venue(self) -> str:
        """The site this broker issues for and revokes on (0124)."""
        if self._venue is None:
            from .store import configured_site_name

            self._venue = configured_site_name()
        return self._venue

    # ── Issue ──────────────────────────────────────────────────────────

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
        session_id: str = "",
        source_ids: tuple[str, ...] | list[str] = (),
        default_source_id: str = "",
        egress_ip: str = "",
        project: str = "",
        max_lifetime_seconds: int | None = None,
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
            session_id: Binding session (inference scope).
            source_ids: Sealed-source UUIDs the lease may reach (inference).
            default_source_id: Fallback source when X-R4-Source is absent.
            egress_ip: Sidecar egress IP the inference proxy binds to.
            project: Accounting project/group.
            max_lifetime_seconds: Hard cap across renewals (None = MAX_TTL).

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
        lifetime_cap = (
            now + timedelta(seconds=max_lifetime_seconds)
            if max_lifetime_seconds
            else None
        )
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
            session_id=session_id,
            source_ids=tuple(str(s) for s in source_ids),
            default_source_id=default_source_id,
            egress_ip=egress_ip,
            project=project,
            max_lifetime_at=lifetime_cap,
            venue=self.venue,
            secret_hash=secret_hash,
        )

        self._leases[lease.lease_id] = lease
        self._secret_index[secret_hash] = lease.lease_id
        self._persist_lease(lease)
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

    # ── Validate ────────────────────────────────────────────────────────

    def validate_secret(self, secret_value: str) -> SecretLease | None:
        """Validate a secret value and return its lease if valid.

        Also increments the access count and logs the access.
        Returns None if the secret is unknown, expired, or revoked.
        """
        secret_hash = hashlib.sha256(secret_value.encode()).hexdigest()
        lease = self._lease_by_hash(secret_hash)
        if lease is None or not lease.is_valid:
            return None

        lease.access_count += 1
        lease.last_accessed_at = _now()
        self._persist_access(lease)
        self._audit_log(lease.lease_id, "access", "", f"count={lease.access_count}")
        return lease

    # ── Revoke ──────────────────────────────────────────────────────────

    def revoke_lease(self, lease_id: str, actor: str = "") -> bool:
        """Revoke a secret lease before expiry."""
        lease = self.get_lease(lease_id)
        if lease is None:
            return False
        self._revoke_row(lease, actor, "manual revocation")
        return True

    def revoke_job_leases(self, job_id: str, actor: str = "") -> int:
        """Revoke all leases for a job (on job completion/failure)."""
        count = 0
        for lease in self.list_leases(limit=100000):
            if lease.job_id == job_id and lease.is_valid:
                self._revoke_row(lease, actor, f"job completion: {job_id}")
                count += 1
        if count:
            logger.info("Revoked %d leases for job %s", count, job_id)
        return count

    def revoke_by_session(self, session_id: str, actor: str = "") -> int:
        """Revoke cascade for a session end/orphan sweep."""
        count = 0
        for lease in self.list_leases(limit=100000):
            if lease.session_id == session_id and lease.is_valid:
                self._revoke_row(lease, actor, f"session end: {session_id}")
                count += 1
        if count:
            logger.info("Revoked %d leases for session %s", count, session_id)
        return count

    def revoke_by_source(self, source_id: str, actor: str = "") -> int:
        """Revoke cascade for unpush (every lease referencing the source)."""
        count = 0
        for lease in self.list_leases(limit=100000):
            if source_id in lease.source_ids and lease.is_valid:
                self._revoke_row(lease, actor, f"source unpushed: {source_id}")
                count += 1
        if count:
            logger.info("Revoked %d leases referencing source %s", count, source_id)
        return count

    def revoke_by_owner(
        self,
        owner_id: str,
        *,
        project: str | None = None,
        actor: str = "",
        reason: str = "",
    ) -> list[SecretLease]:
        """Revoke every live lease ``owner_id`` holds on this site (RS-0 S-2).

        ``project`` narrows it to that project's leases (remove-from-project).
        Leases of other sites are left alone: accounts are per site (0113) and
        the sites share one database. A lease issued before 0124 has no site
        and counts as this one's. Returns the leases revoked; the inference
        proxy and the rail read leases live, so they refuse them at once.
        """
        if not owner_id:
            return []
        detail = f"owner cascade: {owner_id}" + (
            f" project={project}" if project else ""
        )
        if reason:
            detail += f" ({reason})"
        revoked: list[SecretLease] = []
        for lease_id in self._owned_live_lease_ids(owner_id, project):
            lease = self.get_lease(lease_id)
            if lease is None or not lease.is_valid:
                continue
            self._revoke_row(lease, actor, detail)
            revoked.append(lease)
        if revoked:
            logger.info("Revoked %d leases owned by %s", len(revoked), owner_id)
        return revoked

    def active_owner_projects(self) -> set[tuple[str, str]]:
        """``(owner_id, project)`` of every live lease on this site (the
        standing sweep's work list)."""
        if self._pool is None:
            return {
                (lease.owner_id, lease.project)
                for lease in self._leases.values()
                if lease.owner_id and lease.is_valid and self._is_ours(lease.venue)
            }
        rows = self._pool_fetch(
            "SELECT DISTINCT owner_id, project FROM secret_leases"
            " WHERE NOT is_revoked AND expires_at > now() AND owner_id <> ''"
            " AND venue IN (%s, '')",
            (self.venue,),
        )
        return {(str(r["owner_id"]), str(r["project"])) for r in rows}

    def _is_ours(self, venue: str) -> bool:
        return venue in (self.venue, "")

    def _owned_live_lease_ids(self, owner_id: str, project: str | None) -> list[str]:
        if self._pool is None:
            return [
                lease.lease_id
                for lease in self._leases.values()
                if self._owned_live(lease, owner_id, project)
            ]
        sql = (
            "SELECT lease_id FROM secret_leases WHERE owner_id = %s"
            " AND NOT is_revoked AND expires_at > now() AND venue IN (%s, '')"
        )
        params: tuple[Any, ...] = (owner_id, self.venue)
        if project is not None:
            sql += " AND project = %s"
            params += (project,)
        return [str(r["lease_id"]) for r in self._pool_fetch(sql, params)]

    def _owned_live(
        self, lease: SecretLease, owner_id: str, project: str | None
    ) -> bool:
        if lease.owner_id != owner_id or not lease.is_valid:
            return False
        return self._is_ours(lease.venue) and project in (None, lease.project)

    def _pool_fetch(self, sql: str, params: tuple[Any, ...]) -> list[Any]:
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
            conn.commit()
        finally:
            self._pool_putconn(conn)
        return list(rows)

    def _revoke_row(self, lease: SecretLease, actor: str, detail: str) -> None:
        lease.is_revoked = True
        lease.revoked_at = _now()
        self._leases[lease.lease_id] = lease
        self._persist_revoke(lease)
        self._audit_log(lease.lease_id, "revoke", actor, detail)
        logger.info("Secret lease revoked: %s by %s", lease.lease_id, actor or "system")

    # ── Renew ───────────────────────────────────────────────────────────

    def renew_lease(
        self,
        lease_id: str,
        extend_seconds: int,
        actor: str = "",
    ) -> bool:
        """Extend a lease's expiry time.

        Cannot exceed MAX_LEASE_SECONDS from issue unless an explicit
        ``max_lifetime_at`` hard cap was set at issue time (then the cap wins).
        """
        lease = self.get_lease(lease_id)
        if lease is None or not lease.is_valid:
            return False

        new_expiry = lease.expires_at + timedelta(seconds=extend_seconds)
        cap = lease.max_lifetime_at or (
            lease.issued_at + timedelta(seconds=MAX_LEASE_SECONDS)
        )
        if new_expiry > cap:
            return False

        lease.expires_at = new_expiry
        lease.last_accessed_at = _now()  # heartbeat recency for the idle sweep
        self._persist_expiry(lease)
        self._audit_log(lease_id, "renew", actor, f"extended by {extend_seconds}s")
        return True

    # ── Bind (provisioner, after the sidecar is up) ─────────────────────

    def bind_egress_ip(self, lease_id: str, egress_ip: str, actor: str = "") -> bool:
        """Pin the sidecar's egress IP onto a lease (proxy re-checks per request)."""
        lease = self.get_lease(lease_id)
        if lease is None:
            return False
        lease.egress_ip = egress_ip
        self._leases[lease_id] = lease
        if self._pool is not None:
            conn = self._pool_conn()
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "UPDATE secret_leases SET egress_ip = %s::inet WHERE lease_id = %s",
                        (egress_ip, lease_id),
                    )
                conn.commit()
            finally:
                self._pool_putconn(conn)
        self._audit_log(lease_id, "bind", actor, f"egress_ip bound: {egress_ip}")
        return True

    # ── Reads ───────────────────────────────────────────────────────────

    def lease_by_plaintext_token(self, token: str) -> SecretLease | None:
        """Hash+read the live lease for a plaintext session token (R1-2 proxy)."""
        return self._lease_by_hash(hashlib.sha256(token.encode()).hexdigest())

    def get_lease(self, lease_id: str) -> SecretLease | None:
        if self._pool is not None:
            return self._pool_get_lease(lease_id)
        return self._leases.get(lease_id)

    @staticmethod
    def _lease_matches(
        lease: SecretLease,
        job_id: str | None,
        scope: str | None,
        active_only: bool,
    ) -> bool:
        if job_id and lease.job_id != job_id:
            return False
        if scope and lease.scope != scope:
            return False
        return not active_only or lease.is_valid

    def list_leases(
        self,
        job_id: str | None = None,
        scope: str | None = None,
        active_only: bool = False,
        limit: int = 50,
    ) -> list[SecretLease]:
        """List secret leases with optional filters."""
        results = (
            self._pool_list_leases()
            if self._pool is not None
            else list(self._leases.values())
        )
        matched = [
            lease
            for lease in results
            if self._lease_matches(lease, job_id, scope, active_only)
        ]
        matched.sort(key=lambda lease: lease.issued_at, reverse=True)
        return matched[:limit]

    def get_audit_trail(
        self,
        lease_id: str | None = None,
        operation: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit trail with optional filters."""
        results = self._pool_audit() if self._pool is not None else list(self._audit)
        if lease_id:
            results = [entry for entry in results if entry.lease_id == lease_id]
        if operation:
            results = [entry for entry in results if entry.operation == operation]
        results.sort(key=lambda entry: entry.timestamp, reverse=True)
        return results[:limit]

    def cleanup_expired(self) -> int:
        """Remove expired leases. Returns count removed."""
        expired = [lid for lid, lease in self._leases.items() if lease.is_expired]
        for lid in expired:
            lease = self._leases.pop(lid)
            self._secret_index.pop(lease.secret_hash, None)
            self._audit_log(lid, "expire", "system", "ttl expired")
        pool_removed = self._pool_cleanup_expired()
        count = len(expired) + pool_removed
        if count:
            logger.info("Cleaned up %d expired secret leases", count)
        return count

    # ── Internal ────────────────────────────────────────────

    def _audit_log(
        self,
        lease_id: str,
        operation: str,
        actor: str,
        detail: str,
    ) -> None:
        self._counter += 1
        entry = AuditEntry(
            entry_id=f"aud-{self._counter:06d}-{uuid.uuid4().hex[:6]}",
            lease_id=lease_id,
            operation=operation,
            actor=actor,
            detail=detail,
        )
        self._audit.append(entry)
        self._persist_audit(entry)

    # ── Persistence (0085; no-ops in memory mode) ───────────────────────

    def _pool_conn(self):
        if self._pool is None:
            return None
        return self._pool.getconn()

    def _pool_putconn(self, conn: Any) -> None:
        if self._pool is not None and conn is not None:
            self._pool.putconn(conn)

    def _persist_lease(self, lease: SecretLease) -> None:
        if self._pool is None:
            return
        sql = """
            INSERT INTO secret_leases (
                lease_id, token_hash, job_id, secret_name, scope, owner_id,
                bound_to_step, tier, session_id, source_ids, default_source_id,
                egress_ip, project, issued_at, expires_at, max_lifetime_at,
                is_revoked, revoked_at, access_count, last_accessed_at, venue
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s::uuid[], %s::uuid,
                %s::inet, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            ) ON CONFLICT (lease_id) DO NOTHING
        """
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    sql,
                    (
                        lease.lease_id,
                        lease.secret_hash,
                        lease.job_id,
                        lease.secret_name,
                        lease.scope,
                        lease.owner_id,
                        lease.bound_to_step,
                        lease.tier,
                        lease.session_id,
                        [_to_uuid(s) for s in lease.source_ids],
                        _to_uuid(lease.default_source_id),
                        lease.egress_ip or None,
                        lease.project,
                        lease.issued_at,
                        lease.expires_at,
                        lease.max_lifetime_at,
                        lease.is_revoked,
                        lease.revoked_at,
                        lease.access_count,
                        lease.last_accessed_at,
                        lease.venue,
                    ),
                )
            conn.commit()
        finally:
            self._pool_putconn(conn)

    def _persist_access(self, lease: SecretLease) -> None:
        if self._pool is None:
            return
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE secret_leases SET access_count = %s, last_accessed_at = %s "
                    "WHERE lease_id = %s",
                    (lease.access_count, lease.last_accessed_at, lease.lease_id),
                )
            conn.commit()
        finally:
            self._pool_putconn(conn)

    def _persist_revoke(self, lease: SecretLease) -> None:
        if self._pool is None:
            return
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE secret_leases SET is_revoked = TRUE, revoked_at = %s "
                    "WHERE lease_id = %s",
                    (lease.revoked_at, lease.lease_id),
                )
            conn.commit()
        finally:
            self._pool_putconn(conn)

    def _persist_expiry(self, lease: SecretLease) -> None:
        if self._pool is None:
            return
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE secret_leases SET expires_at = %s, last_accessed_at = %s "
                    "WHERE lease_id = %s",
                    (lease.expires_at, lease.last_accessed_at, lease.lease_id),
                )
            conn.commit()
        finally:
            self._pool_putconn(conn)

    def _persist_audit(self, entry: AuditEntry) -> None:
        if self._pool is None:
            return
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "INSERT INTO secret_audit (entry_id, lease_id, operation, actor, detail, "
                    "created_at) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        entry.entry_id,
                        entry.lease_id,
                        entry.operation,
                        entry.actor,
                        entry.detail,
                        entry.timestamp,
                    ),
                )
            conn.commit()
        finally:
            self._pool_putconn(conn)

    def _lease_by_hash(self, secret_hash: str) -> SecretLease | None:
        if self._pool is not None:
            return self._pool_lease_by_hash(secret_hash)
        lease_id = self._secret_index.get(secret_hash)
        return self._leases.get(lease_id) if lease_id else None

    @staticmethod
    def _source_ids_from_row(value: Any) -> tuple[str, ...]:
        """Postgres uuid[] may come back as a list of UUID-ish or a literal '{a,b}'."""
        if value is None:
            return ()
        if isinstance(value, str):
            inner = value.strip("{}")
            return (
                tuple(s.strip() for s in inner.split(",") if s.strip()) if inner else ()
            )
        return tuple(str(v) for v in value)

    def _row_to_lease(self, row: Any) -> SecretLease:
        return SecretLease(
            lease_id=str(row["lease_id"]),
            job_id=str(row["job_id"]),
            secret_name=str(row["secret_name"]),
            scope=str(row["scope"]),
            owner_id=str(row["owner_id"]),
            bound_to_step=str(row["bound_to_step"]),
            tier=str(row["tier"]),
            session_id=str(row["session_id"]),
            source_ids=self._source_ids_from_row(row["source_ids"]),
            default_source_id=str(row["default_source_id"] or ""),
            egress_ip=str(row["egress_ip"] or ""),
            project=str(row["project"]),
            issued_at=row["issued_at"],
            expires_at=row["expires_at"],
            max_lifetime_at=row["max_lifetime_at"],
            is_revoked=bool(row["is_revoked"]),
            revoked_at=row["revoked_at"],
            access_count=int(row["access_count"]),
            last_accessed_at=row["last_accessed_at"],
            secret_hash=str(row["token_hash"]),
        )

    _SELECT = """
        SELECT lease_id, token_hash, job_id, secret_name, scope, owner_id,
               bound_to_step, tier, session_id, source_ids, default_source_id,
               egress_ip, project, issued_at, expires_at, max_lifetime_at,
               is_revoked, revoked_at, access_count, last_accessed_at
          FROM secret_leases
    """

    def _pool_get_lease(self, lease_id: str) -> SecretLease | None:
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._SELECT + " WHERE lease_id = %s", (lease_id,))
                row = cur.fetchone()
        finally:
            self._pool_putconn(conn)
        return self._row_to_lease(row) if row else None

    def _pool_lease_by_hash(self, secret_hash: str) -> SecretLease | None:
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._SELECT + " WHERE token_hash = %s", (secret_hash,))
                row = cur.fetchone()
        finally:
            self._pool_putconn(conn)
        return self._row_to_lease(row) if row else None

    def _pool_list_leases(self) -> list[SecretLease]:
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(self._SELECT)
                rows = cur.fetchall()
        finally:
            self._pool_putconn(conn)
        return [self._row_to_lease(r) for r in rows]

    def _pool_audit(self) -> list[AuditEntry]:
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT entry_id, lease_id, operation, actor, detail, created_at "
                    "FROM secret_audit"
                )
                rows = cur.fetchall()
        finally:
            self._pool_putconn(conn)
        return [
            AuditEntry(
                entry_id=str(r["entry_id"]),
                lease_id=str(r["lease_id"]),
                operation=str(r["operation"]),
                actor=str(r["actor"]),
                detail=str(r["detail"]),
                timestamp=r["created_at"],
            )
            for r in rows
        ]

    def _pool_cleanup_expired(self) -> int:
        if self._pool is None:
            return 0
        conn = self._pool_conn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "DELETE FROM secret_leases WHERE expires_at <= now() RETURNING lease_id"
                )
                removed = cur.fetchall()
            conn.commit()
        finally:
            self._pool_putconn(conn)
        return len(removed)


def build_default_broker(backend: str | None = None) -> SecretBroker:
    """Env-driven factory (R0-2), mirroring the satellite-store pattern.

    ``UMRS_SECRET_BROKER_BACKEND`` / explicit arg:
      - ``pg``      → require the shared sync pool (error when unavailable)
      - ``memory``  → force in-memory (tests/dev)
      - ``auto``    → Postgres when a pool resolves from env, else in-memory
    """
    mode = (
        (backend or os.environ.get("UMRS_SECRET_BROKER_BACKEND") or "auto")
        .strip()
        .lower()
    )
    pool = None
    if mode != "memory":
        try:
            from umrs.shared.pg import get_default_sync_pool

            pool = get_default_sync_pool()
        except Exception:  # noqa: BLE001
            pool = None
        if mode == "pg" and pool is None:
            raise RuntimeError(
                "UMRS_SECRET_BROKER_BACKEND=pg but no Postgres pool is resolvable from env"
            )
    return SecretBroker(db_pool=pool)
