"""Venue lifecycle audit events (S-7, remote-session plan wave RS-0).

One helper writes every venue lifecycle transition to ``audit_events``
through the existing emitter (``umrs.api.audit_events.emit_audit``: the
structured log line, the ring buffer and the persistent ``AuditStore`` the
sidecar registers at boot). Nothing else: no second store, no framework.

Events (``event`` column; ``actor`` is who caused it):

    venue.session.start    owner asked for a sandbox (job submitted)
    venue.session.stop     a session ended; ``details.cause`` says why
    venue.session.pause    owner paused the worker between turns
    venue.session.resume   owner resumed it
    venue.lease.issue      the session lease was minted (provisioning)
    venue.lease.revoke     a session lease was revoked (teardown, idle)
    venue.admin.list       an admin listed sessions across users
    venue.admin.show       an admin read one user's session
    venue.admin.denied     a non-admin called an admin session route

``details`` always carries ``session_id`` and ``owner`` (the session's user)
when known; never a token, a secret or a source credential.

The emitter lives in ``umrs.api``; the import is lazy and function-local so
``umrs.jobs`` callers (the session backend) never import ``umrs.api`` at
module load (ADR-053). An audit failure is logged and never fails the
lifecycle transition it describes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Why a session ended (``details.cause`` of ``venue.session.stop``).
STOP_CAUSES = frozenset(
    {
        "owner",  # the owner's stop route
        "admin",  # POST /umrs/admin/sessions/{sid}/stop
        "idle",  # no worker activity beyond UMRS_VENUE_IDLE_SECONDS
        "lease_end",  # the lease expired or was revoked elsewhere
        "exit",  # the worker reported it exited
        "done",  # a plan-run worker finished its run
        "cancel",  # the durable job was cancelled from another surface
        "provision_failed",  # the sandbox never came up
        "disabled",  # the S-2 cascade: account disabled / removed from project
    }
)


def audit_venue(
    event: str,
    *,
    actor: str,
    session_id: str,
    owner: str = "",
    project: str = "",
    outcome: str = "success",
    scope: str = "user",
    **details: Any,
) -> None:
    """Write one ``venue.<event>`` row; never raises."""
    payload = {"session_id": session_id, "owner": owner, **details}
    try:
        from umrs.api.audit_events import emit_audit

        emit_audit(
            f"venue.{event}",
            actor=actor or "system",
            project_id=project or "",
            details=payload,
            outcome=outcome,
            scope=scope,
        )
    except Exception:  # audit never breaks the transition it describes
        logger.warning(
            "venue audit %s for %s not written", event, session_id, exc_info=True
        )


def audit_session_stop(
    *,
    cause: str,
    actor: str,
    session_id: str,
    owner: str,
    detail: str = "",
    **extra: Any,
) -> None:
    """The one ``venue.session.stop`` row of a session end."""
    audit_venue(
        "session.stop",
        actor=actor,
        session_id=session_id,
        owner=owner,
        outcome="error" if cause in ("exit", "provision_failed") else "success",
        cause=cause if cause in STOP_CAUSES else "other",
        detail=(detail or "")[:400],
        **extra,
    )
