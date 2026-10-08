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

The deadline is ``metadata["slurm_deadline"]`` when the executor has read it
from Slurm (``squeue %L`` of the running job: a ``--time-min`` start before a
reservation ends early), else ``created_at + walltime``: ``created_at`` is the
submit time, never later than the Slurm start, so the estimate is never late —
at worst the successor starts by the job's queue time early.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
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

#: A DEGRADED endpoint counts as live only this long after it degraded
#: (``metadata["degraded_since"]``, set by the janitor's engine probe). On
#: 2026-10-07 a Nimble job vanished 15 h into its walltime; its row stayed
#: DEGRADED, the relay counted it live, nothing was respawned, and every
#: decision call failed until the row was removed by hand. Past the grace it
#: no longer blocks a respawn, and it is retired once a READY endpoint exists.
DEGRADED_GRACE_SEC = 900


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


_DURATION = re.compile(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?")


def _hms(
    days_s: str | None, a: str, b: str | None, c: str | None
) -> tuple[int, int, int]:
    if days_s is not None:  # D-HH[:MM[:SS]]
        return int(a), int(b or 0), int(c or 0)
    if c is not None:  # HH:MM:SS
        return int(a), int(b or 0), int(c)
    return 0, int(a), int(b or 0)  # MM or MM:SS


def parse_slurm_duration(value: str) -> int | None:
    """Seconds in a Slurm ``--time`` value; ``None`` for unlimited/unparseable.

    Accepts ``MM``, ``MM:SS``, ``HH:MM:SS``, ``D-HH``, ``D-HH:MM`` and
    ``D-HH:MM:SS``. A zero duration is Slurm's "no time limit" (bdrt sites
    set ``time_limit: "0"``), so it is ``None`` too: a 0 s walltime would put
    every live endpoint inside its renew window and the relay would provision
    a successor every cycle.
    """
    text = str(value or "").strip()
    match = _DURATION.fullmatch(text)
    if not match:  # also "", UNLIMITED, INFINITE
        return None
    days_s, a, b, c = match.groups()
    hours, minutes, seconds = _hms(days_s, a, b, c)
    total = ((int(days_s or 0) * 24 + hours) * 60 + minutes) * 60 + seconds
    return total or None


def _positive(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _policy_from_entry(entry: dict[str, Any]) -> KeepAlivePolicy | None:
    block = entry.get("keep_alive")
    model = str(entry.get("model") or "")
    if not isinstance(block, dict) or not model:
        return None
    project_id = str(block.get("project_id") or "").strip()
    if not project_id:
        logger.warning("keep_alive for %s has no project_id; relay skipped", model)
        return None
    walltime = block.get("walltime")
    return KeepAlivePolicy(
        model=model,
        project_id=project_id,
        user=str(block.get("user") or "").strip() or None,
        renew_before_sec=_positive(block.get("renew_before_sec"), 7200),
        handover_grace_sec=_positive(block.get("handover_grace_sec"), 300),
        walltime_sec=parse_slurm_duration(walltime) if walltime else None,
    )


def keep_alive_policies(entries: Iterable[dict[str, Any]]) -> list[KeepAlivePolicy]:
    """Policies for the roster entries that carry a ``keep_alive`` block.

    One policy per model (the first entry wins, as in the provision roster).
    An entry without ``keep_alive.project_id`` is skipped with a warning:
    endpoints are per project, so the relay cannot guess the owner.
    """
    policies: dict[str, KeepAlivePolicy] = {}
    for entry in entries:
        if str(entry.get("model") or "") in policies:
            continue
        policy = _policy_from_entry(entry)
        if policy is not None:
            policies[policy.model] = policy
    return list(policies.values())


#: Metadata key holding the Slurm-granted end of a running job (naive UTC ISO).
DEADLINE_KEY = "slurm_deadline"


def _granted_deadline(endpoint: RegisteredEndpoint) -> datetime | None:
    raw = (endpoint.metadata or {}).get(DEADLINE_KEY)
    try:
        return datetime.fromisoformat(str(raw)) if raw else None
    except ValueError:
        return None


def _seconds_left(
    endpoint: RegisteredEndpoint, walltime_sec: int, now: datetime
) -> float:
    deadline = _granted_deadline(endpoint) or (
        endpoint.created_at + timedelta(seconds=walltime_sec)
    )
    return (deadline - now).total_seconds()


def needs_deadline(policy: KeepAlivePolicy, endpoint: RegisteredEndpoint) -> bool:
    """A READY endpoint of this policy whose Slurm deadline is not read yet."""
    return (
        _owned(policy, endpoint)
        and endpoint.status == EndpointStatus.READY
        and bool(endpoint.job_id)
        and _granted_deadline(endpoint) is None
    )


def deadline_metadata(now: datetime, time_left_sec: int) -> dict[str, str]:
    """Registry metadata recording a running job's Slurm end (naive UTC)."""
    return {DEADLINE_KEY: (now + timedelta(seconds=time_left_sec)).isoformat()}


def _degraded_since(ep: RegisteredEndpoint) -> datetime | None:
    raw = (ep.metadata or {}).get("degraded_since")
    try:
        return datetime.fromisoformat(str(raw)) if raw else None
    except ValueError:
        return None


def _stale_degraded(ep: RegisteredEndpoint, now: datetime) -> bool:
    """DEGRADED for longer than the grace (rows without a timestamp never are)."""
    since = _degraded_since(ep) if ep.status == EndpointStatus.DEGRADED else None
    return since is not None and (now - since).total_seconds() >= DEGRADED_GRACE_SEC


def _owned(policy: KeepAlivePolicy, ep: RegisteredEndpoint) -> bool:
    if ep.project_id != policy.project_id:
        return False
    return (
        ep.model == policy.model or ep.metadata.get("served_model_name") == policy.model
    )


def _handed_over(
    policy: KeepAlivePolicy, fresh: list[RegisteredEndpoint], now: datetime
) -> bool:
    return any(
        ep.status == EndpointStatus.READY
        and ep.ready_at is not None
        and (now - ep.ready_at).total_seconds() >= policy.handover_grace_sec
        for ep in fresh
    )


def _retirements(
    policy: KeepAlivePolicy,
    expiring: list[RegisteredEndpoint],
    walltime: int,
    now: datetime,
) -> list[RelayAction]:
    actions = []
    for ep in expiring:
        left = _seconds_left(ep, walltime, now)
        if ep.active_requests == 0 or left <= DRAIN_FLOOR_SEC:
            actions.append(
                RelayAction(
                    "retire",
                    policy.model,
                    policy.project_id,
                    f"handed over; {int(left)}s to walltime",
                    ep.job_id,
                )
            )
    return actions


@dataclass(frozen=True)
class _Fleet:
    live: list[RegisteredEndpoint]
    pending: bool
    expiring: list[RegisteredEndpoint]
    fresh: list[RegisteredEndpoint]
    stale: tuple[RegisteredEndpoint, ...] = ()


def _split_by_deadline(
    policy: KeepAlivePolicy,
    live: list[RegisteredEndpoint],
    walltime: int,
    now: datetime,
) -> tuple[list[RegisteredEndpoint], list[RegisteredEndpoint]]:
    expiring, fresh = [], []
    for ep in live:
        near = _seconds_left(ep, walltime, now) <= policy.renew_before_sec
        (expiring if near else fresh).append(ep)
    return expiring, fresh


def _fleet(
    policy: KeepAlivePolicy,
    endpoints: Iterable[RegisteredEndpoint],
    walltime: int | None,
    now: datetime,
) -> _Fleet:
    mine = [ep for ep in endpoints if _owned(policy, ep)]
    live, stale = _live_and_stale(mine, now)
    pending = any(ep.status in _PENDING for ep in mine)
    if walltime is None:
        return _Fleet(live, pending, [], live, stale)
    expiring, fresh = _split_by_deadline(policy, live, walltime, now)
    return _Fleet(live, pending, expiring, fresh, stale)


def _live_and_stale(
    mine: list[RegisteredEndpoint], now: datetime
) -> tuple[list[RegisteredEndpoint], tuple[RegisteredEndpoint, ...]]:
    """Live rows (READY, or DEGRADED within the grace) and the stale DEGRADED ones."""
    stale = tuple(ep for ep in mine if _stale_degraded(ep, now))
    return [ep for ep in mine if ep.status in _LIVE and ep not in stale], stale


def _stale_retirements(policy: KeepAlivePolicy, fleet: _Fleet) -> list[RelayAction]:
    """Retire stale DEGRADED rows once a READY endpoint serves the model."""
    if not any(ep.status == EndpointStatus.READY for ep in fleet.live):
        return []
    return [
        RelayAction(
            "retire",
            policy.model,
            policy.project_id,
            f"degraded for over {DEGRADED_GRACE_SEC}s; a READY endpoint serves",
            ep.job_id,
        )
        for ep in fleet.stale
    ]


def plan_relay(
    policy: KeepAlivePolicy,
    endpoints: Iterable[RegisteredEndpoint],
    *,
    walltime_sec: int | None,
    now: datetime,
) -> list[RelayAction]:
    """Relay actions for one policy given the registry rows (pure)."""
    walltime = policy.walltime_sec or walltime_sec
    fleet = _fleet(policy, endpoints, walltime, now)
    return _plan_live(policy, fleet, walltime, now) + _stale_retirements(policy, fleet)


def _plan_live(
    policy: KeepAlivePolicy, fleet: _Fleet, walltime: int | None, now: datetime
) -> list[RelayAction]:
    """Respawn, successor and handover actions over the live endpoints."""
    provision = RelayAction("provision", policy.model, policy.project_id, "")
    if not fleet.live:
        return (
            []
            if fleet.pending
            else [replace(provision, reason="no live or pending endpoint")]
        )
    if walltime is None:
        return []  # no deadline to relay across; only the respawn above applies
    if not fleet.fresh:
        if fleet.pending:
            return []
        soonest = min(_seconds_left(ep, walltime, now) for ep in fleet.expiring)
        reason = f"successor: live endpoint reaches walltime in {int(soonest)}s"
        return [replace(provision, reason=reason)]
    if not _handed_over(policy, fleet.fresh, now):
        return []
    return _retirements(policy, fleet.expiring, walltime, now)
