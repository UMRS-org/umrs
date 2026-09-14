"""SessionGcBackend (session GC ops job type, 2026-09-12).

Runs ``session_gc`` durable jobs: archives every non-archived companion
research-session row older than spec.max_days through the ADR-035 item-8
cascade (one transaction per session; never a straight DELETE — everything
with a lifecycle goes through its cascade, the tombstones are the legacy).

Spec contract (strict):

    max_days: int > 0       (required; refuse all others)

Side effects: only what archive_with_trust_cascade already does — status =
archived, archived_at+updated_at stamps, session-scope trust revoke in the
same txn, companion_session_archives tombstone. Attempt metrics carry the
archived ids list (bounded at 1000) + duration.

This is ARCHIVE-ONLY semantics (never DELETE), chosen deliberately: the
mirror-aware delete surface is a callable-personal-store flow (r4 local
deletes with failed-mirror tombstones replaying next reconcile), while the
server-side GC only archives because deleting a client-visible item here
would break pull-from-anywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .models import Job, JobAttempt
from .monitor import AttemptCheckResult, BackendStatusChecker
from ..research_sessions.store import SessionStore

logger = logging.getLogger(__name__)


class SessionGcBackend:
    """Executor for session_gc jobs (archive lane, durable com-loop).

    Per-attempt report lodging: get_report(attempt_id) is the honest face;
    the global singleton never collapses concurrent jobs onto each other.
    """

    def __init__(
        self,
        session_store: SessionStore | None = None,
        model: str = "session_gc",
    ) -> None:
        self.session_store = session_store
        self._model_tag = model
        self._reports: dict[str, dict[str, Any]] = {}

    @property
    def backend_name(self) -> str:
        return self._model_tag

    def report_for(self, attempt_id: str) -> dict[str, Any]:
        return dict(self._reports.get(attempt_id) or {})

    def forget(self, attempt_id: str) -> None:
        self._reports.pop(attempt_id, None)

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        """Strict spec chain → inline sweep (durable state outside the task)."""
        spec = job.spec or {}
        if "max_days" not in spec:
            raise ValueError("session_gc requires spec.max_days (>0, days)")
        try:
            max_days = int(spec.get("max_days") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"spec.max_days must be an integer, got: {spec.get('max_days')!r}"
            ) from exc
        if max_days <= 0:
            raise ValueError(f"spec.max_days must be > 0, got: {max_days}")

        if self.session_store is None:
            raise RuntimeError("session_gc lacks a shared umrs_db session store")

        from umrs.shared.trust_store_access import get_trust_store  # archive cascade's trust ledger

        trust_store = get_trust_store()
        started = asyncio.get_running_loop().time()
        archived_ids = await asyncio.to_thread(
            self.session_store.archive_aged_sessions,
            max_days=max_days,
            trust_store=trust_store,
        )
        duration_s = asyncio.get_running_loop().time() - started
        self._reports[job.job_id] = {
            "attempt_id": attempt.attempt_id,
            "max_days": max_days,
            "archived": len(archived_ids),
            "archived_ids_sample": archived_ids[:1000],
            "duration_s": round(duration_s, 3),
            "lane": "archive-only-by-design (no DELETE)",
        }
        logger.info(
            "session_gc job=%s attempt=%s archived=%d (max_days=%d)",
            job.job_id,
            attempt.attempt_id,
            len(archived_ids),
            max_days,
        )
        return attempt.attempt_id


class SessionGcStatusChecker(BackendStatusChecker):
    """Reads the backend's per-attempt report; inline sweeps map succeeded."""

    def __init__(self, backend: SessionGcBackend | None = None) -> None:
        self._backend = backend

    @property
    def job_type(self) -> "JobType":
        from .models import JobType

        return JobType.SESSION_GC

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        metrics = self._backend.report_for(job.job_id) if self._backend is not None else {}
        return AttemptCheckResult.succeeded(metrics=metrics)


def create_session_gc_backend(**kwargs: Any) -> SessionGcBackend:
    """Factory hook for the durable runtime."""
    return SessionGcBackend(**kwargs)


def create_session_gc_status_checker() -> SessionGcStatusChecker:
    """Factory hook for the composite status checker."""
    return SessionGcStatusChecker()
