"""Per-project readiness approvals (table ``readiness_approvals``, migration 0055).

A researcher approves one project's knowledge base for autonomous operation.
Each approval is one row: the project, the verification confidence at the
time of approval (``threshold``), who approved, why, and when it expires.
The latest unexpired row is the project's approval state; it holds only
while the latest verification report is clean and at or above its threshold.

Known limit: there is no revoke.  An approval lapses by expiry (7 days) or
when readiness drops below its threshold; ``latest_active`` picks the newest
unexpired row, so a later row cannot withdraw an earlier one.

It replaces the module-global explorer controller, which held ONE approval
for the whole server (TUI audit 2026-10-17, round 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

DEFAULT_TTL = timedelta(days=7)

_COLUMNS = (
    "approval_id, project_id, threshold, approved_by, approved_at, expires_at, reason"
)


@dataclass(frozen=True)
class ReadinessApproval:
    """One readiness approval row."""

    approval_id: str
    project_id: str
    threshold: float
    approved_by: str
    approved_at: datetime | None
    expires_at: datetime | None
    reason: str = ""

    @classmethod
    def from_row(cls, row: Any) -> ReadinessApproval:
        return cls(
            approval_id=row["approval_id"],
            project_id=row["project_id"],
            threshold=float(row["threshold"]),
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            expires_at=row["expires_at"],
            reason=row["reason"] or "",
        )

    def holds_for(self, latest: Any | None) -> bool:
        """Whether the approval still holds against the latest report.

        It lapses when there is no report, the report ran degraded, or its
        confidence (at the stored NUMERIC(3, 2) precision) fell below the
        threshold approved.
        """
        if latest is None or getattr(latest, "degraded", False):
            return False
        return _threshold(latest.confidence) >= self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "project_id": self.project_id,
            "threshold": self.threshold,
            "approved_by": self.approved_by,
            "approved_at": _iso(self.approved_at),
            "expires_at": _iso(self.expires_at),
            "reason": self.reason,
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _threshold(confidence: float) -> float:
    """NUMERIC(3, 2) in [0, 1]."""
    return round(min(max(float(confidence), 0.0), 1.0), 2)


class ReadinessApprovalStore:
    """asyncpg-backed writer/reader for ``readiness_approvals``."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def record(
        self,
        *,
        project_id: str,
        confidence: float,
        approved_by: str,
        reason: str = "",
        ttl: timedelta = DEFAULT_TTL,
        now: datetime | None = None,
    ) -> ReadinessApproval:
        """Insert one approval; ``confidence`` becomes its threshold."""
        approved_at = now or datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "INSERT INTO readiness_approvals "
                "(approval_id, project_id, threshold, approved_by, approved_at, "
                "expires_at, reason, audited) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE) "
                f"RETURNING {_COLUMNS}",
                f"rap-{uuid4().hex[:16]}",
                project_id,
                _threshold(confidence),
                approved_by,
                approved_at,
                approved_at + ttl,
                reason,
            )
        return ReadinessApproval.from_row(row)

    async def latest_active(
        self, project_id: str, *, now: datetime | None = None
    ) -> ReadinessApproval | None:
        """The project's newest approval that has not expired, if any."""
        at = now or datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLUMNS} FROM readiness_approvals "
                "WHERE project_id = $1 AND (expires_at IS NULL OR expires_at > $2) "
                "ORDER BY approved_at DESC NULLS LAST LIMIT 1",
                project_id,
                at,
            )
        return ReadinessApproval.from_row(row) if row is not None else None
