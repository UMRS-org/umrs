"""Verification store with report and finding lifecycle.

Immutable versioned reports. Finding upsert by ``finding_key`` with transition
rules (open → acknowledged → resolved → waived). Auto-resolve on non-detection.
Waiver persistence with evidence-hash guard.

Transition rules (enforced at store level):
  - open → acknowledged, open → resolved, open → waived
  - acknowledged → resolved, acknowledged → waived
  - resolved → immutable (no further transitions)
  - waived → waiver sticky with expiry; auto-resolve when waived expires

Schema: migration 0055_knowledge_store.sql (verification_reports,
verification_findings tables).
"""

from __future__ import annotations

import enum
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── time helpers ──────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ── exceptions / enums ───────────────────────────────────────────────────────


class TransitionError(ValueError):
    """Raised when a finding transition is invalid."""

    def __init__(self, finding_key: str, from_status: str, to_status: str) -> None:
        self.finding_key = finding_key
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Invalid transition for finding '{finding_key}': "
            f"{from_status} → {to_status}. "
            f"Valid: open→{['acknowledged', 'resolved', 'waived']}, "
            f"acknowledged→{['resolved', 'waived']}, resolved→[none]"
        )


class FindingStatus(str, enum.Enum):
    """Finding lifecycle states.

    * ``open`` – newly detected, needs attention.
    * ``acknowledged`` – reviewer has seen it but not fixed.
    * ``resolved`` – fixed / verified closed.
    * ``waived`` – approved exception with optional expiry.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    WAIVED = "waived"


# allowed transitions: from_status → list of allowed to_status
_VALID_TRANSITIONS: dict[str, list[str]] = {
    "open": ["acknowledged", "resolved", "waived"],
    "acknowledged": ["resolved", "waived"],
    "resolved": [],  # immutable
    "waived": ["waived"],  # re-waive allowed
}


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VerificationReport:
    """Immutable versioned verification report."""

    report_id: str
    project_id: str
    version: int
    spec_name: str
    confidence: float  # 0.0 – 1.0
    total_checks: int
    passed_checks: int
    failed_checks: int
    skipped_checks: int
    degraded: bool = False
    degraded_reason: str | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class VerificationFinding:
    """A verification finding with lifecycle transitions."""

    finding_key: str
    report_id: str
    category: str
    severity: str  # blocking, warning, info
    status: str  # open, acknowledged, resolved, waived
    title: str
    description: str = ""
    evidence: str = ""
    artifact_id: str | None = None
    waived_evidence: str | None = None
    waiver_expires: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


# ── store ─────────────────────────────────────────────────────────────────────


class VerificationStore:
    """Verification store with report and finding lifecycle.

    Backed by migration 0055 tables: ``verification_reports``,
    ``verification_findings``.
    """

    # -- construction -----------------------------------------------------------

    def __init__(self, pool) -> None:  # type: ignore[misc]
        """Initialize store with an asyncpg connection pool."""
        self._pool = pool

    # -- reports (immutable versioned) ------------------------------------------

    async def create_report(
        self,
        report_id: str,
        project_id: str,
        spec_name: str,
        version: int,
        confidence: float,
        total_checks: int,
        passed_checks: int,
        failed_checks: int,
        skipped_checks: int = 0,
        degraded: bool = False,
        degraded_reason: str | None = None,
    ) -> VerificationReport:
        """Create a new versioned verification report (immutable).

        Raises ``ValueError`` if a report with same
        ``(project_id, spec_name, version)`` already exists.
        """
        async with self._pool.acquire() as conn:
            try:
                ts = _utcnow()
                await conn.execute(
                    """INSERT INTO verification_reports
                       (report_id, project_id, version, spec_name, confidence,
                        total_checks, passed_checks, failed_checks, skipped_checks,
                        degraded, degraded_reason, created_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)""",
                    report_id,
                    project_id,
                    version,
                    spec_name,
                    confidence,
                    total_checks,
                    passed_checks,
                    failed_checks,
                    skipped_checks,
                    degraded,
                    degraded_reason,
                    ts,
                )
                report = VerificationReport(
                    report_id=report_id,
                    project_id=project_id,
                    version=version,
                    spec_name=spec_name,
                    confidence=confidence,
                    total_checks=total_checks,
                    passed_checks=passed_checks,
                    failed_checks=failed_checks,
                    skipped_checks=skipped_checks,
                    degraded=degraded,
                    degraded_reason=degraded_reason,
                    created_at=ts,
                )
                logger.info("Verification report created: %s v%d", spec_name, version)
                return report
            except Exception as exc:
                if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                    raise ValueError(
                        f"Report already exists: {project_id}/{spec_name}/v{version}"
                    ) from exc
                raise

    async def get_report(self, report_id: str) -> VerificationReport | None:
        """Fetch a single report by id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM verification_reports WHERE report_id = $1",
                report_id,
            )
            return VerificationReport(**row) if row else None

    async def list_reports(
        self,
        *,
        project_id: str | None = None,
        spec_name: str | None = None,
        limit: int = 100,
    ) -> list[VerificationReport]:
        """List reports with optional filters, newest first."""
        async with self._pool.acquire() as conn:
            sql = "SELECT * FROM verification_reports WHERE 1=1"
            params: list[Any] = []
            idx = 1

            if project_id:
                sql += f" AND project_id = ${idx}"
                params.append(project_id)
                idx += 1
            if spec_name:
                sql += f" AND spec_name = ${idx}"
                params.append(spec_name)
                idx += 1

            sql += f" ORDER BY created_at DESC LIMIT {limit}"
            rows = await conn.fetch(sql, *params)
            return [VerificationReport(**r) for r in rows]

    # -- findings (upsert + transitions) ----------------------------------------

    async def upsert_finding(
        self,
        finding_key: str,
        report_id: str,
        category: str,
        severity: str,
        title: str,
        *,
        status: str = "open",
        description: str = "",
        evidence: str = "",
        artifact_id: str | None = None,
    ) -> VerificationFinding:
        """Upsert a finding by *finding_key*.

        * If the finding does **not** exist → insert with the given ``status``.
        * If it **does** exist → update metadata fields (severity, title,
          description, evidence).  The status is **not** silently reset to
          ``open`` – transitions must go through :meth:`transition_finding`.
        """
        async with self._pool.acquire() as conn:
            ts = _utcnow()

            # 1. Check existing status
            existing = await conn.fetchrow(
                """SELECT status, severity, title FROM verification_findings
                   WHERE finding_key = $1""",
                finding_key,
            )

            if existing:
                # Already exists – update metadata only (preserve status)
                await conn.execute(
                    """UPDATE verification_findings
                       SET severity = $1, title = $2, description = $3,
                           evidence = $4, artifact_id = $5, updated_at = $6
                       WHERE finding_key = $7""",
                    severity,
                    title,
                    description,
                    evidence,
                    artifact_id,
                    ts,
                    finding_key,
                )
            else:
                # New finding – insert with provided status (must be valid)
                if status not in _VALID_TRANSITIONS:
                    raise TransitionError(finding_key, "(new)", status)
                await conn.execute(
                    """INSERT INTO verification_findings
                       (finding_key, report_id, category, severity, status,
                        title, description, evidence, artifact_id,
                        created_at, updated_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                    finding_key,
                    report_id,
                    category,
                    severity,
                    status,
                    title,
                    description,
                    evidence,
                    artifact_id,
                    ts,
                    ts,
                )

            return await self._find_by_key(conn, finding_key)

    async def transition_finding(
        self,
        finding_key: str,
        from_status: str,
        to_status: str,
    ) -> VerificationFinding:
        """Transition a finding between states.

        Enforces :data:`_VALID_TRANSITIONS`.
        """
        if to_status not in _VALID_TRANSITIONS.get(from_status, []):
            raise TransitionError(finding_key, from_status, to_status)

        async with self._pool.acquire() as conn:
            ts = _utcnow()
            result = await conn.execute(
                """UPDATE verification_findings
                   SET status = $1, updated_at = $2
                   WHERE finding_key = $3 AND status = $4""",
                to_status,
                ts,
                finding_key,
                from_status,
            )
            if result.split()[1] == "0":
                raise ValueError(
                    f"Finding '{finding_key}' not in status '{from_status}' "
                    f"(current status unknown)"
                )
            logger.debug(
                "Finding transition: %s %s → %s",
                finding_key,
                from_status,
                to_status,
            )
            return await self._find_by_key(conn, finding_key)

    async def waive_finding(
        self,
        finding_key: str,
        evidence: str,
        *,
        expires_after: timedelta | None = None,
    ) -> VerificationFinding:
        """Waive a finding with evidence-hash guard.

        Waiver is sticky: once waived, only re-waiving is allowed.
        If *expires_after* is given the waiver auto-resolves on
        :meth:`auto_resolve_expired` after that duration.
        """
        async with self._pool.acquire() as conn:
            ts = _utcnow()

            # Hash evidence for integrity guard
            evidence_hash = hashlib.sha256(evidence.encode()).hexdigest()
            logger.debug("Waiver evidence hash: %s", evidence_hash[:16])

            expires_at: datetime | None = None
            if expires_after is not None:
                expires_at = ts + expires_after

            result = await conn.execute(
                """UPDATE verification_findings
                   SET status = 'waived',
                       waived_evidence = $1,
                       waiver_expires = $2,
                       updated_at = $3
                   WHERE finding_key = $4""",
                evidence,
                expires_at,
                ts,
                finding_key,
            )
            if result.split()[1] == "0":
                raise ValueError(f"Finding '{finding_key}' not found for waiver")
            logger.info("Finding waived: %s (expires=%s)", finding_key, expires_at)
            return await self._find_by_key(conn, finding_key)

    # -- querying ---------------------------------------------------------------

    async def get_finding(self, finding_key: str) -> VerificationFinding | None:
        """Fetch a single finding by key."""
        async with self._pool.acquire() as conn:
            return await self._find_by_key(conn, finding_key)

    async def list_findings(
        self,
        *,
        report_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 200,
    ) -> list[VerificationFinding]:
        """List findings with optional filters."""
        async with self._pool.acquire() as conn:
            sql = "SELECT * FROM verification_findings WHERE 1=1"
            params: list[Any] = []
            idx = 1

            if report_id:
                sql += f" AND report_id = ${idx}"
                params.append(report_id)
                idx += 1
            if status:
                sql += f" AND status = ${idx}"
                params.append(status)
                idx += 1
            if severity:
                sql += f" AND severity = ${idx}"
                params.append(severity)
                idx += 1

            sql += f" ORDER BY updated_at DESC LIMIT {limit}"
            rows = await conn.fetch(sql, *params)
            return [VerificationFinding(**r) for r in rows]

    async def list_open_findings(
        self, report_id: str | None = None
    ) -> list[VerificationFinding]:
        """List all open or acknowledged findings."""
        return await self.list_findings(
            report_id=report_id, status="open"
        ) + await self.list_findings(report_id=report_id, status="acknowledged")

    async def list_blocking_findings(self, report_id: str) -> list[VerificationFinding]:
        """List blocking findings that prevent approval."""
        return await self.list_findings(
            report_id=report_id,
            status=None,  # all statuses
            severity="blocking",
        )

    # -- housekeeping -----------------------------------------------------------

    async def auto_resolve_expired(self) -> int:
        """Resolve findings whose waiver has expired.

        Returns the number of findings auto-resolved.
        """
        async with self._pool.acquire() as conn:
            ts = _utcnow()
            result = await conn.execute(
                """UPDATE verification_findings
                   SET status = 'resolved', updated_at = $1
                   WHERE status = 'waived'
                     AND waiver_expires IS NOT NULL
                     AND waiver_expires < now()""",
                ts,
            )
            count = int(result.split()[1])
            if count:
                logger.info("Auto-resolved %d expired waivers", count)
            return count

    # -- internals --------------------------------------------------------------

    @staticmethod
    async def _find_by_key(conn: Any, finding_key: str) -> VerificationFinding:
        """Helper: fetch a single finding row and return a dataclass."""
        row = await conn.fetchrow(
            "SELECT * FROM verification_findings WHERE finding_key = $1",
            finding_key,
        )
        if row is None:
            raise ValueError(f"Finding '{finding_key}' not found")
        return VerificationFinding(**row)

    # -- verification sessions (TASK-046) ---------------------------------------

    async def save_verification_session(
        self,
        session_id: str,
        project_id: str,
        question: str,
        evidence_ids: list[str],
        graph_traversal: list[str],
        confidence: float,
    ) -> None:
        """Persist a verification-chat session (evidence + traversal).

        The ephemeral LLM answer is NOT stored — it can be regenerated
        from the evidence IDs and graph traversal on demand.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO verification_sessions
                   (session_id, project_id, question, evidence_ids,
                    graph_traversal, confidence, timestamp)
                   VALUES ($1, $2, $3, $4, $5, $6, now())
                   ON CONFLICT (session_id) DO UPDATE SET
                       question = EXCLUDED.question,
                       evidence_ids = EXCLUDED.evidence_ids,
                       graph_traversal = EXCLUDED.graph_traversal,
                       confidence = EXCLUDED.confidence,
                       timestamp = now()
                """,
                session_id,
                project_id,
                question,
                evidence_ids,
                graph_traversal,
                confidence,
            )

    async def get_verification_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a single verification session by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM verification_sessions WHERE session_id = $1",
                session_id,
            )
            return dict(row) if row else None

    async def list_verification_sessions(
        self,
        *,
        project_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List verification sessions with optional project filter."""
        async with self._pool.acquire() as conn:
            if project_id:
                sql = (
                    "SELECT * FROM verification_sessions"
                    " WHERE project_id = $1"
                    " ORDER BY timestamp DESC LIMIT $2"
                )
                rows = await conn.fetch(sql, project_id, limit)
            else:
                sql = (
                    "SELECT * FROM verification_sessions"
                    " ORDER BY timestamp DESC LIMIT $1"
                )
                rows = await conn.fetch(sql, limit)
            return [dict(r) for r in rows]

    async def delete_verification_session(self, session_id: str) -> bool:
        """Delete a verification session. Returns True if found and deleted."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM verification_sessions WHERE session_id = $1",
                session_id,
            )
            return result.split()[1] == "1"
