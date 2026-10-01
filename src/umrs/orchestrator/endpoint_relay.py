"""Walltime relay — keep a roster endpoint alive across the Slurm time limit.

A serving job on a Slurm venue dies at its walltime (RCCS caps GB10 jobs at
24 h; the site submits with ``scheduler.time_limit: 23:00:00``). For an
endpoint that must stay up — a decision model every agent turn depends on —
UMRS hands the endpoint over to a successor job before the deadline instead of
letting the gateway answer "No active endpoint" until someone runs ``r4
endpoint up`` again.

A roster entry opts in with a ``keep_alive`` block::

    - name: bespoke-nimble-9b
      model: Bespoke-Nimble-9B
      keep_alive:
        project_id: default        # registry owner (endpoints are per project)
        user: mnagaso              # optional: Slurm submitter for per-user delegation
        renew_before_sec: 7200     # start the successor this long before the deadline
        handover_grace_sec: 300    # successor READY this long before the old one retires

The relay is a planner here and an executor in ``umrs.server`` (this
module is foundation: no ``umrs.api`` / ``umrs.server`` imports). Each cycle:

1. no live and no pending endpoint → provision one (also covers a crashed job);
2. every live endpoint is inside its renew window and no successor is pending
   → provision the successor;
3. a successor has been READY for ``handover_grace_sec`` and is outside the
   renew window → retire the old endpoints (idle ones at once, busy ones once
   they are within ``drain_floor_sec`` of the deadline).

The deadline is ``created_at + walltime``: ``created_at`` is the submit time,
never later than the Slurm start, so the estimate is never late — at worst the
successor starts by the job's queue time early.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

from .endpoint_registry import EndpointStatus, RegisteredEndpoint

logger = logging.getLogger(__name__)

_LIVE = (EndpointStatus.READY, EndpointStatus.DEGRADED)
_PENDING = (
    EndpointStatus.PENDING,
    EndpointStatus.PROVISIONING,
    EndpointStatus.STARTING,
)

#: Busy old endpoints still retire once this close to their deadline: the
#: walltime kill would end their requests anyway.
DRAIN_FLOOR_SEC = 300


@dataclass(frozen=True)
class KeepAlivePolicy:
    """One roster entry's ``keep_alive`` block, resolved."""

    model: str
    project_id: str
    user: str | None = None
    renew_before_sec: int = 7200
    handover_grace_sec: int = 300
    walltime_sec: int | None = None  # None = the site's scheduler.time_limit


@dataclass(frozen=True)
class RelayAction:
    """One step the executor performs."""

    action: Literal["provision", "retire"]
    model: str
    project_id: str
    reason: str
    job_id: str | None = None


def parse_slurm_duration(value: str) -> int | None:
    """Seconds in a Slurm ``--time`` value; ``None`` for unlimited/unparseable.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and
    ``D-HH:MM:SS``.
    """
    text = str(value or "").strip()
    if not text or text.upper() in ("UNLIMITED", "INFINITE"):
        return None
    match = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", text)
    if not match:
        return None
    days_s, a, b, c = match.groups()
    days = int(days_s or 0)
    if days_s is not None:  # D-HH[:MM[:SS]]
        hours, minutes, seconds = int(a), int(b or 0), int(c or 0)
    elif c is not None:  # HH:MM:SS
        hours, minutes, seconds = int(a), int(b), int(c)
    else:  # MM or MM:SS
        hours, minutes, seconds = 0, int(a), int(b or 0)
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _positive(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def keep_alive_policies(entries: Iterable[dict[str, Any]]) -> list[KeepAlivePolicy]:
    """Policies for the roster entries that carry a ``keep_alive`` block.

    One policy per model (the first entry wins, as in the provision roster).
    An entry without ``keep_alive.project_id`` is skipped with a warning:
    endpoints are per project, so the relay cannot guess the owner.
    """
    policies: dict[str, KeepAlivePolicy] = {}
    for entry in entries:
        block = entry.get("keep_alive")
        model = str(entry.get("model") or "")
        if not isinstance(block, dict) or not model or model in policies:
            continue
        project_id = str(block.get("project_id") or "").strip()
        if not project_id:
            logger.warning("keep_alive for %s has no project_id; relay skipped", model)
            continue
        walltime = block.get("walltime")
        policies[model] = KeepAlivePolicy(
            model=model,
            project_id=project_id,
            user=(str(block["user"]).strip() or None) if block.get("user") else None,
            renew_before_sec=_positive(block.get("renew_before_sec"), 7200),
            handover_grace_sec=_positive(block.get("handover_grace_sec"), 300),
            walltime_sec=parse_slurm_duration(walltime) if walltime else None,
        )
    return list(policies.values())


def _seconds_left(
    endpoint: RegisteredEndpoint, walltime_sec: int, now: datetime
) -> float:
    deadline = endpoint.created_at + timedelta(seconds=walltime_sec)
    return (deadline - now).total_seconds()


def plan_relay(
    policy: KeepAlivePolicy,
    endpoints: Iterable[RegisteredEndpoint],
    *,
    walltime_sec: int | None,
    now: datetime,
) -> list[RelayAction]:
    """Relay actions for one policy given the registry rows (pure)."""
    walltime = policy.walltime_sec or walltime_sec
    mine = [
        ep
        for ep in endpoints
        if ep.project_id == policy.project_id
        and (
            ep.model == policy.model
            or ep.metadata.get("served_model_name") == policy.model
        )
    ]
    live = [ep for ep in mine if ep.status in _LIVE]
    pending = [ep for ep in mine if ep.status in _PENDING]

    def act(
        action: Literal["provision", "retire"], reason: str, job_id: str | None = None
    ):
        return RelayAction(action, policy.model, policy.project_id, reason, job_id)

    if not live:
        if pending:
            return []
        return [act("provision", "no live or pending endpoint")]
    if walltime is None:
        return []  # no deadline to relay across; only the respawn above applies

    expiring = [
        ep for ep in live if _seconds_left(ep, walltime, now) <= policy.renew_before_sec
    ]
    fresh = [ep for ep in live if ep not in expiring]
    if not fresh:
        if pending:
            return []
        soonest = min(_seconds_left(ep, walltime, now) for ep in expiring)
        return [
            act(
                "provision",
                f"successor: live endpoint reaches walltime in {int(soonest)}s",
            )
        ]

    handed_over = any(
        ep.status == EndpointStatus.READY
        and ep.ready_at is not None
        and (now - ep.ready_at).total_seconds() >= policy.handover_grace_sec
        for ep in fresh
    )
    if not handed_over:
        return []
    actions = []
    for ep in expiring:
        left = _seconds_left(ep, walltime, now)
        if ep.active_requests == 0 or left <= DRAIN_FLOOR_SEC:
            actions.append(
                act("retire", f"handed over; {int(left)}s to walltime", ep.job_id)
            )
    return actions
