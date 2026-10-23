"""Cap on repeated failing re-provisioning (2026-10-23).

bdrt, 2026-10-20..22: job 8218 held dgx-spark-0's GPU behind a dead LiteLLM.
Every new request for the same model submitted a fresh Slurm job, which
landed on the same node, could not start vLLM (8.4 of 119.7 GiB free), used
up its respawn budget and failed: 49 FAILED jobs, each accepted as if the
last one had never happened.

Policy: when the last ``max_failures`` provision attempts of one
(project, model, variant) on this venue all failed inside ``window_sec``, a
new request is refused with ``endpoints.provision_capped`` (HTTP 429 and
Retry-After) instead of submitting another job. A success in between ends
the run. The refusal names the failed jobs and the time the oldest of them
leaves the window. An admin may pass ``override_failure_cap``.

Store: the audit trail (the persistent ``audit_events`` table when the
server has one, else the in-process ring buffer), so there is no new table
and no migration. The trail belongs to the api layer, which the orchestrator
may not import (import boundary): ``umrs.api.endpoints`` hands its emit and
query functions in through :func:`use_trail` when it is imported. Without a
trail nothing is recorded and nothing is refused. Each finished attempt is one ``endpoint.provision_attempt``
event, outcome ``success`` or ``error``, whose details carry the model,
variant, venue and job id. Every venue's server writes the same table, so
the venue is part of the key.

What counts (review F5 and its re-review, 2026-10-23): a failure the request
caused. A job the backend reported FAILED counts, also before it published a
location (the wrapper writes that only at the end of bring-up, so a missing
model, an unpullable image or a compose failure end the job before it); so
does a job that left the queue and then outran the provision budget, and a
STARTING endpoint the janitor reaped as stuck (:func:`record_reaped_stuck`).
A job still queued at the deadline, a job that stopped (cancelled), and an
endpoint being deprovisioned (row marked, gone, TERMINATED or EXPIRED) do not.

Clocks (re-review round 3, 2026-10-23): queue time never counts against the
model. The location wait gives a job that leaves the queue its own bring-up
window (``service._LocationClock``), the janitor ages a STARTING row from its
``starting_at`` stamp, and a row the janitor marked as reaped stuck is the
janitor's to count, not the provision path's (:func:`provision_path_counts`).

Tunables: ``UMRS_PROVISION_FAILURE_CAP`` (failures, default 3; 0 disables)
and ``UMRS_PROVISION_FAILURE_WINDOW_SEC`` (default 1800).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .endpoint_registry import EndpointStatus, RegisteredEndpoint

logger = logging.getLogger(__name__)

ATTEMPT_EVENT = "endpoint.provision_attempt"
OVERRIDE_EVENT = "endpoint.provision_cap_overridden"
_QUERY_LIMIT = 200
_ERROR_CHARS = 300

# emit_audit / query_audit_events of umrs.api.audit_events, set by use_trail.
_TRAIL: dict[str, Callable[..., Any]] = {}


def use_trail(
    emit: Callable[..., None], query: Callable[..., list[dict[str, Any]]]
) -> None:
    """Hand in the audit trail the attempts are written to and read from."""
    _TRAIL.update(emit=emit, query=query)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


@dataclass(frozen=True)
class CapPolicy:
    """How many failures inside which window pause a model's provisioning."""

    max_failures: int = 3
    window_sec: int = 1800

    @property
    def enabled(self) -> bool:
        return self.max_failures > 0 and self.window_sec > 0

    @classmethod
    def from_env(cls) -> CapPolicy:
        return cls(
            max_failures=_env_int("UMRS_PROVISION_FAILURE_CAP", cls.max_failures),
            window_sec=_env_int("UMRS_PROVISION_FAILURE_WINDOW_SEC", cls.window_sec),
        )


@dataclass(frozen=True)
class AttemptKey:
    """What one run of failures is counted per."""

    project_id: str
    model: str
    variant: str | None
    venue: str

    def details(self) -> dict[str, Any]:
        return {"model": self.model, "variant": self.variant or "", "venue": self.venue}

    def matches(self, details: dict[str, Any]) -> bool:
        return all(details.get(k, "") == v for k, v in self.details().items())


@dataclass(frozen=True)
class FailedAttempt:
    job_id: str
    ts: float
    error: str


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CapRefusal:
    """A refused provision: the failures that caused it and when to retry."""

    key: AttemptKey
    failures: tuple[FailedAttempt, ...]  # newest first
    window_sec: int
    retry_at: float

    def retry_after_sec(self, now: float) -> int:
        return max(1, math.ceil(self.retry_at - now))

    def body(self, now: float) -> dict[str, Any]:
        """The 429 response body: the message and its parts as fields."""
        return {
            "message": self.message(now),
            "model": self.key.model,
            "variant": self.key.variant,
            "venue": self.key.venue,
            "failed_jobs": [f.job_id for f in self.failures],
            "retry_at": _utc(self.retry_at),
            "retry_after_sec": self.retry_after_sec(now),
        }

    def message(self, now: float) -> str:
        variant = f" (variant {self.key.variant})" if self.key.variant else ""
        jobs = "; ".join(
            f"job {f.job_id} at {_utc(f.ts)}: {f.error[:120] or 'no error recorded'}"
            for f in self.failures
        )
        # The wait and the way out come first: clients cut long messages
        # (r4 keeps 400 characters), and the job list is the longest part.
        return (
            f"Provisioning '{self.key.model}'{variant} on {self.key.venue} for project "
            f"'{self.key.project_id}' is paused until {_utc(self.retry_at)} (in "
            f"{self.retry_after_sec(now)} s): the last {len(self.failures)} attempts "
            f"failed within {self.window_sec // 60} min. Fix the cause and retry then; "
            f"an admin may provision now with override_failure_cap=true. Failed: {jobs}."
        )


def current_venue() -> str:
    """This server's venue (env, then site config name)."""
    env = os.environ.get("UMRS_CLUSTER", "").strip()
    if env:
        return env
    try:
        from .site_config import load_site_config

        return str(load_site_config().name)
    except Exception:  # noqa: BLE001 - provenance only; never fail the caller
        return "unknown"


def record_attempt(
    key: AttemptKey, *, job_id: str, ok: bool, error: str | None = None, actor: str = ""
) -> None:
    """One finished provision attempt into the audit trail (never raises)."""
    emit = _TRAIL.get("emit")
    if emit is None:
        logger.debug("no audit trail; provision attempt %s not recorded", job_id)
        return
    details = key.details()
    details.update({"job_id": job_id, "error": (error or "")[:_ERROR_CHARS]})
    try:
        emit(
            ATTEMPT_EVENT,
            actor=actor,
            project_id=key.project_id,
            outcome="success" if ok else "error",
            scope="project",
            details=details,
        )
    except Exception:
        logger.warning("could not record provision attempt %s", job_id, exc_info=True)


# Registry marks of an endpoint someone is taking down (review F5).
DEPROVISION_MARKS = ("deprovision_requested_at", "cleanup_status")
# The janitor's mark on a STARTING endpoint it reaps as stuck (re-review N2b).
REAPED_STUCK_MARK = "reaped_stuck_at"


def failure_is_the_models(endpoint: RegisteredEndpoint | None) -> bool:
    """Whether a failed attempt of this registry row counts toward the cap.

    Not an endpoint that was being deprovisioned (the cancel ends the job),
    nor one whose row is gone or already expired/terminated (deprovisioned or
    a stale registry). The caller decides whether the job got far enough: a
    queue or assignment timeout says nothing about the model.
    """
    if endpoint is None:
        return False
    if endpoint.status in (EndpointStatus.TERMINATED, EndpointStatus.EXPIRED):
        return False
    return not any(mark in endpoint.metadata for mark in DEPROVISION_MARKS)


def provision_path_counts(endpoint: RegisteredEndpoint | None) -> bool:
    """Whether the provision path may record this row's failure.

    Not a row the janitor marked as reaped stuck (re-review N6, 2026-10-23):
    the janitor decides whether that one counts (:func:`record_reaped_stuck`),
    and the provision path seeing the row before its removal must not count
    it a second time, nor count one the janitor chose not to.
    """
    if endpoint is None or not failure_is_the_models(endpoint):
        return False
    return REAPED_STUCK_MARK not in endpoint.metadata


def _starting_for(endpoint: RegisteredEndpoint, now: datetime) -> float | None:
    """Seconds the row has been STARTING, or None if it did not record when."""
    starting_at = endpoint.starting_at
    if starting_at is None:
        return None
    return (now - starting_at).total_seconds()


def record_reaped_stuck(
    endpoint: RegisteredEndpoint,
    error: str,
    *,
    min_starting_sec: float,
    now: datetime | None = None,
) -> bool:
    """A STARTING endpoint the janitor reaps as stuck, as a failed attempt.

    Its job reached a node and the engine never became ready inside the
    janitor's stuck timeout. Without this the row was removed unmarked, the
    provision still waiting on it saw no row and did not count the failure,
    and a hanging engine was re-provisioned every ~10 min, uncapped.

    Counted only when the row has been STARTING for ``min_starting_sec``
    (the stuck timeout) by its own ``starting_at`` stamp (re-review N4,
    2026-10-23): ``created_at`` includes the queue wait, so a job that
    queued past the timeout and had just turned STARTING was counted. A row
    without the stamp is not counted. Returns whether it was recorded.
    """
    if endpoint.status != EndpointStatus.STARTING:
        return False
    if not failure_is_the_models(endpoint):
        return False
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    starting_for = _starting_for(endpoint, now)
    if starting_for is None or starting_for < min_starting_sec:
        logger.info(
            "stuck reap of %s not counted: STARTING for %s s (needs %s s)",
            endpoint.job_id,
            "an unknown time" if starting_for is None else int(starting_for),
            int(min_starting_sec),
        )
        return False
    key = AttemptKey(
        endpoint.project_id, endpoint.model, endpoint.variant, current_venue()
    )
    actor = str(endpoint.metadata.get("requesting_user") or "")
    record_attempt(key, job_id=endpoint.job_id, ok=False, error=error, actor=actor)
    return True


def recent_attempts(key: AttemptKey, *, since: float) -> list[dict[str, Any]]:
    """This key's attempt events since ``since``, any order."""
    query = _TRAIL.get("query")
    if query is None:
        return []
    rows = query(
        event=ATTEMPT_EVENT, project_id=key.project_id, since=since, limit=_QUERY_LIMIT
    )
    return [r for r in rows if key.matches(r.get("details") or {})]


def _as_attempt(row: dict[str, Any]) -> tuple[FailedAttempt, bool]:
    details = row.get("details") or {}
    attempt = FailedAttempt(
        job_id=str(details.get("job_id", "")),
        ts=float(row.get("ts", 0.0)),
        error=str(details.get("error", "")),
    )
    return attempt, row.get("outcome") == "success"


def leading_failures(
    rows: list[dict[str, Any]], *, since: float
) -> list[FailedAttempt]:
    """The failures since ``since`` newer than the last success, newest first.

    A job recorded twice counts once.
    """
    attempts = sorted(
        (_as_attempt(r) for r in rows), key=lambda a: a[0].ts, reverse=True
    )
    failures: list[FailedAttempt] = []
    seen: set[str] = set()
    for attempt, ok in attempts:
        if ok or attempt.ts < since:
            break
        if attempt.job_id not in seen:
            seen.add(attempt.job_id)
            failures.append(attempt)
    return failures


def evaluate(
    rows: list[dict[str, Any]], key: AttemptKey, policy: CapPolicy, now: float
) -> CapRefusal | None:
    """The refusal these attempt rows call for, or None."""
    failures = leading_failures(rows, since=now - policy.window_sec)
    if len(failures) < policy.max_failures:
        return None
    counted = tuple(failures[: policy.max_failures])
    return CapRefusal(
        key=key,
        failures=counted,
        window_sec=policy.window_sec,
        retry_at=counted[-1].ts + policy.window_sec,
    )


def check(
    key: AttemptKey, policy: CapPolicy | None = None, now: float | None = None
) -> CapRefusal | None:
    """Refusal for a new provision of ``key`` now, or None to go ahead."""
    policy = policy or CapPolicy.from_env()
    if not policy.enabled:
        return None
    now = time.time() if now is None else now
    return evaluate(
        recent_attempts(key, since=now - policy.window_sec), key, policy, now
    )
