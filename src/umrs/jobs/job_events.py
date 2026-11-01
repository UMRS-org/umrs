"""Pushed job events (ADR-105 D-5): no second job store.

A JobStore terminal listener writes one ``platform_notifications`` row
(migration 0109) per terminal transition: kind ``job.succeeded``,
``job.failed``, ``job.cancelled`` or ``job.lost``, owned by the job's user in
the job's project. The row's BIGSERIAL id is the consumer cursor
(``GET /umrs/jobs/events?after=<id>``); ``read_at`` is never used, so a TUI,
VS Code and a venue can follow one session.

The emission is best-effort (``platform_notifications.notify``): it never
blocks or fails the transition that already committed.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

JOB_EVENT_KIND_PREFIX = "job."

#: ``x-r4-session-id`` shape (the companion session id; same rule as
#: ``api/loop.py``). Client-asserted: it only tags the caller's own rows.
_SESSION_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

_TITLE_MAX = 200

_unwired_logged = False


def valid_session_id(value: Any) -> str | None:
    """The stripped session id when well-formed, else None (ignored)."""
    if not isinstance(value, str):
        return None
    sid = value.strip()
    return sid if _SESSION_ID.match(sid) else None


def stamp_origin_session(spec: dict[str, Any], session_id: Any) -> None:
    """Set ``spec.origin.session_id`` from a well-formed id (other origin keys
    kept); a missing or malformed id leaves ``spec`` untouched."""
    sid = valid_session_id(session_id)
    if sid is None:
        return
    origin = spec.get("origin")
    merged = dict(origin) if isinstance(origin, dict) else {}
    merged["session_id"] = sid
    spec["origin"] = merged


def _job_type_value(job: Any) -> str:
    job_type = getattr(job, "job_type", "")
    return str(getattr(job_type, "value", job_type) or "")


def _spec(job: Any) -> dict[str, Any]:
    spec = getattr(job, "spec", None)
    return spec if isinstance(spec, dict) else {}


def _origin_session(spec: dict[str, Any]) -> str | None:
    origin = spec.get("origin")
    if not isinstance(origin, dict):
        return None
    return valid_session_id(origin.get("session_id"))


def _execution_id(job_type: str, spec: dict[str, Any]) -> str | None:
    if job_type != "workflow_run":
        return None
    value = spec.get("execution_id")
    return str(value) if value else None


def _finished_at(job: Any) -> str:
    when = getattr(job, "last_state_change_at", None)
    if not isinstance(when, datetime):
        when = datetime.now(timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.isoformat()


def job_event_title(job: Any, state: str) -> str:
    """One short line: ``workflow relax finished: succeeded``."""
    job_type = _job_type_value(job)
    spec = _spec(job)
    name = spec.get("workflow_name") if job_type == "workflow_run" else None
    if name:
        title = f"workflow {name} finished: {state}"
    else:
        title = f"{job_type or 'job'} {getattr(job, 'job_id', '')} finished: {state}"
    return title[:_TITLE_MAX]


def job_event_detail(job: Any, state: str, exit_reason: str | None) -> dict[str, Any]:
    """The event payload a cursor consumer reads (the row's ``detail``)."""
    job_type = _job_type_value(job)
    spec = _spec(job)
    return {
        "job_id": str(getattr(job, "job_id", "")),
        "job_type": job_type,
        "status": state,
        "execution_id": _execution_id(job_type, spec),
        "session_id": _origin_session(spec),
        "exit_reason": exit_reason or None,
        "finished_at": _finished_at(job),
    }


def _emitter_ready() -> bool:
    """False (logged once) when this process has no notification pool."""
    global _unwired_logged
    from umrs.platform_notifications import emitter_wired

    if emitter_wired():
        return True
    if not _unwired_logged:
        _unwired_logged = True
        logger.warning(
            "job events not recorded: the notification emitter is unwired "
            "in this process (no umrs_db DSN)"
        )
    return False


def emit_job_terminal_event(
    job: Any, state: str, attempt: Any = None, exit_reason: str | None = ""
) -> bool:
    """The JobStore terminal listener: one ``job.<state>`` row, best-effort.

    Returns True when a row landed.
    """
    del attempt  # the job carries everything the event names
    if not _emitter_ready():
        return False
    from umrs.platform_notifications import notify

    return notify(
        kind=f"{JOB_EVENT_KIND_PREFIX}{state}",
        title=job_event_title(job, state),
        detail=job_event_detail(job, state, exit_reason),
        user_id=str(getattr(job, "user_id", "") or ""),
        project_id=str(getattr(job, "project_id", "") or ""),
    )


def install_job_event_listener() -> None:
    """Register the emitter process-wide (idempotent): every JobStore in
    this process — scheduler, HTTP sidecar, ZMQ durable backend — fires it.
    Best-effort: a wiring fault only logs."""
    try:
        from .store import register_process_job_terminal_listener

        register_process_job_terminal_listener(emit_job_terminal_event)
    except Exception as exc:  # noqa: BLE001 — boot must survive a wiring fault
        logger.warning("Job event emitter registration skipped: %s", exc)
