"""Vision-call durable wrapping (G-P4-2, WP-09, ADR-044).

Every ``/vision/infer/submit`` execution mints one durable job
(``VISION_INFER``) with one attempt — the request-driven durability tier
of the ADR-044 taxonomy: no executor, no scheduler requeue
(``RestartPolicy.NEVER``, ``max_attempts=1``); the HTTP handlers drive the
transitions (admission → terminal-on-status/cancel). On UMRS restart the
in-flight attempt is marked LOST and the job resolves FAILED instead of
zombie-RUNNING — truthful for a request-scoped call.

All wrapping is additive and best-effort: when the durable pool is absent
(dev/test UIs), calls run exactly as before (job_id fields resolve None);
any transition error is logged and swallowed — the vision path never
breaks because its attribution layer faulted.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.jobs.models import JobType, RestartPolicy
from umrs.jobs.store import InvalidTransitionError, JobStore

logger = logging.getLogger(__name__)


def store_for(request: Any) -> JobStore | None:
    """Return the durable store when the app has a pool, else None."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return None
    return JobStore(pool)


def admit(
    store: JobStore | None,
    *,
    project_id: str,
    user_id: str,
    spec: dict[str, Any],
) -> tuple[str | None, str | None]:
    """Create job + first RUNNING attempt. Returns (job_id, attempt_id)."""
    if store is None:
        return None, None
    try:
        job_id = store.create_job(
            project_id,
            user_id,
            JobType.VISION_INFER,
            spec,
            max_attempts=1,
            restart_policy=RestartPolicy.NEVER,
        )
        attempt_id = store.create_attempt(job_id, backend="vision_endpoint")
        store.mark_attempt_running(attempt_id, status_detail="admitted")
        return job_id, attempt_id
    except Exception as exc:  # noqa: BLE001 — durability never breaks the call
        logger.warning("vision durable admit failed (call continues): %s", exc)
        return None, None


def terminal(
    store: JobStore | None,
    attempt_id: str | None,
    outcome: str,
    *,
    reason: str = "",
) -> None:
    """Best-effort terminal transition; idempotent across repeated polls."""
    if store is None or attempt_id is None:
        return
    try:
        if outcome == "succeeded":
            store.mark_attempt_succeeded(attempt_id, commit=True)
        elif outcome == "failed":
            store.mark_attempt_failed(attempt_id, exit_reason=reason or "downstream_failed")
    except InvalidTransitionError:
        # Already terminal from an earlier poll — quiet no-op.
        logger.debug("vision attempt %s already terminal", attempt_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision terminal transition failed (attempt %s): %s", attempt_id, exc)


def cancel(store: JobStore | None, job_id: str | None, *, reason: str) -> None:
    """Cancel the durable row (advisory-locked; does not kill the backend task)."""
    if store is None or job_id is None:
        return
    try:
        store.cancel_job(job_id, reason=reason)
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision cancel_job failed (job %s): %s", job_id, exc)


__all__ = ["admit", "cancel", "store_for", "terminal"]
