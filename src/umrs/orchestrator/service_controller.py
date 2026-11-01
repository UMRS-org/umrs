"""Replica controller for service-tier models (ADR-106 D-3/D-4).

A roster entry with a ``service`` block keeps ``replicas`` copies of a model
READY and replaces them one at a time, so a restart never drops the service::

    - name: svc-chat-glm53
      model: glm-5.3-flash
      aliases: [svc-chat]
      service:
        project_id: default
        user: mnagaso
        replicas: 2            # target READY count
        min_available: 1       # never drain below this many handed-over READY replicas
        max_surge: 1           # extra jobs allowed while replacing (0 = drain first)
        renew_before_sec: 7200 # walltime venues: start replacing this long before the end
        handover_grace_sec: 300
        drain_timeout_sec: 900 # retire a draining replica after this even if busy
        max_age: 7-00:00:00    # planned roll on venues without a walltime (bdrt)

Pure planner (foundation: no ``umrs.api`` / ``umrs.server`` imports); the
executor is ``umrs.server.endpoint_relay``. Each cycle it does at most one
thing per kind:

1. **retire** a DRAINING replica once it has no in-flight request, its drain
   timed out, or its walltime is ``DRAIN_FLOOR_SEC`` away;
2. **respawn** while fewer than ``replicas`` jobs run or wait (one job at a
   time; DRAINING ones do not count);
3. **roll** the most urgent *due* replica: provision a surge successor while
   ``live < replicas + max_surge``, else **drain** it, but only when at least
   ``min_available`` other READY replicas have been handed over
   (``handover_grace_sec``). One replica drains at a time;
4. **trim** a replica above ``replicas`` once nothing is due.

A replica is *due* when it is inside its walltime renew window, serves an
older recipe (``service_digest`` differs from the entry's digest), is older
than ``max_age``, carries ``roll_requested``, or (``replicas > 1`` on a
walltime venue) is the oldest of a set whose deadlines are aligned. In the
last case it rolls early once, so the replicas' deadlines spread out.

An upgrade stops itself: while a replica of the new recipe is FAILED, an
outdated digest is no longer a reason to roll (the canary failed; the old
replicas keep serving).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .endpoint_registry import EndpointStatus, RegisteredEndpoint
from .endpoint_relay import (
    _LIVE,
    _PENDING,
    DRAIN_FLOOR_SEC,
    KeepAlivePolicy,
    RelayAction,
    _owned,
    _seconds_left,
    _stale_degraded,
)

#: Metadata the executor writes: the recipe digest a replica was provisioned
#: from, when it started draining, and an operator's roll request.
DIGEST_KEY = "service_digest"
DRAINING_SINCE_KEY = "draining_since"
ROLL_KEY = "roll_requested"


@dataclass
class _View:
    live: list[RegisteredEndpoint]
    pending: list[RegisteredEndpoint]
    draining: list[RegisteredEndpoint]
    stale: list[RegisteredEndpoint]
    held: bool
    due: list[tuple[RegisteredEndpoint, str]] = field(default_factory=list)


def _iso(raw: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(raw)) if raw else None
    except ValueError:
        return None


def _age_sec(ep: RegisteredEndpoint, now: datetime) -> float:
    return (now - (ep.ready_at or ep.created_at)).total_seconds()


def _upgrade_held(policy: KeepAlivePolicy, mine: list[RegisteredEndpoint]) -> bool:
    return any(
        ep.status == EndpointStatus.FAILED
        and (ep.metadata or {}).get(DIGEST_KEY) == policy.digest
        for ep in mine
    )


def _with_status(
    rows: list[RegisteredEndpoint], statuses: Iterable[EndpointStatus], skip=()
) -> list[RegisteredEndpoint]:
    wanted = tuple(statuses)
    return [ep for ep in rows if ep.status in wanted and ep not in skip]


def _view(policy: KeepAlivePolicy, endpoints: Iterable[RegisteredEndpoint], now: datetime) -> _View:
    mine = [ep for ep in endpoints if _owned(policy, ep)]
    stale = [ep for ep in mine if _stale_degraded(ep, now)]
    return _View(
        live=_with_status(mine, _LIVE, skip=stale),
        pending=_with_status(mine, _PENDING),
        draining=_with_status(mine, (EndpointStatus.DRAINING,)),
        stale=stale,
        held=_upgrade_held(policy, mine),
    )


def _outdated(policy: KeepAlivePolicy, ep: RegisteredEndpoint, held: bool) -> bool:
    digest = (ep.metadata or {}).get(DIGEST_KEY)
    return bool(digest) and digest != policy.digest and not held


def _due_reason(
    policy: KeepAlivePolicy,
    ep: RegisteredEndpoint,
    walltime: int | None,
    view: _View,
    now: datetime,
) -> str | None:
    """Why this live replica should be replaced now, or ``None``."""
    if walltime and _seconds_left(ep, walltime, now) <= policy.renew_before_sec:
        return f"walltime in {int(_seconds_left(ep, walltime, now))}s"
    if _outdated(policy, ep, view.held):
        return "serves an older recipe (upgrade)"
    if (ep.metadata or {}).get(ROLL_KEY):
        return "roll requested"
    if policy.max_age_sec and _age_sec(ep, now) >= policy.max_age_sec:
        return f"older than max_age ({int(_age_sec(ep, now))}s)"
    return None


def _stagger_due(
    policy: KeepAlivePolicy, view: _View, walltime: int | None, now: datetime
) -> list[tuple[RegisteredEndpoint, str]]:
    """The oldest replica of an aligned set, once it is past 1/N of its walltime."""
    if not walltime or policy.replicas < 2 or len(view.live) < policy.replicas:
        return []
    lefts = sorted((_seconds_left(ep, walltime, now), ep.job_id, ep) for ep in view.live)
    spread = lefts[-1][0] - lefts[0][0]
    oldest_left, _, oldest = lefts[0]
    if spread < walltime / (2 * policy.replicas) and oldest_left <= walltime * (
        1 - 1 / policy.replicas
    ):
        return [(oldest, "stagger: replica deadlines are aligned")]
    return []


def _due(policy: KeepAlivePolicy, view: _View, walltime: int | None, now: datetime):
    due = []
    for ep in view.live:
        reason = _due_reason(policy, ep, walltime, view, now)
        if reason:
            due.append((ep, reason))
    if not due:
        due = _stagger_due(policy, view, walltime, now)
    due.sort(key=lambda pair: _seconds_left(pair[0], walltime, now) if walltime else 0)
    return due


def _action(policy: KeepAlivePolicy, kind: str, reason: str, job_id: str | None = None):
    return RelayAction(kind, policy.model, policy.project_id, reason, job_id)


def _drained(
    policy: KeepAlivePolicy, ep: RegisteredEndpoint, walltime: int | None, now: datetime
) -> str | None:
    if ep.active_requests == 0:
        return "drained"
    since = _iso((ep.metadata or {}).get(DRAINING_SINCE_KEY))
    if since is not None and (now - since).total_seconds() >= policy.drain_timeout_sec:
        return f"drain timed out with {ep.active_requests} in flight"
    if walltime and _seconds_left(ep, walltime, now) <= DRAIN_FLOOR_SEC:
        return "walltime floor reached while draining"
    return None


def _retire_drained(policy: KeepAlivePolicy, view: _View, walltime: int | None, now: datetime):
    actions = []
    for ep in view.draining:
        reason = _drained(policy, ep, walltime, now)
        if reason:
            actions.append(_action(policy, "retire", reason, ep.job_id))
    return actions


def _retire_stale(policy: KeepAlivePolicy, view: _View):
    if not any(ep.status == EndpointStatus.READY for ep in view.live):
        return []
    return [
        _action(policy, "retire", "degraded past the grace; a READY replica serves", ep.job_id)
        for ep in view.stale
    ]


def _handed_over_ready(
    policy: KeepAlivePolicy, view: _View, target: RegisteredEndpoint, now: datetime
) -> int:
    return sum(
        1
        for ep in view.live
        if ep is not target
        and ep.status == EndpointStatus.READY
        and ep.ready_at is not None
        and (now - ep.ready_at).total_seconds() >= policy.handover_grace_sec
    )


def _roll_one(policy: KeepAlivePolicy, view: _View, now: datetime):
    target, reason = view.due[0]
    if len(view.live) < policy.replicas + policy.max_surge:
        return [_action(policy, "provision", f"surge for {target.job_id}: {reason}")]
    if _handed_over_ready(policy, view, target, now) >= max(policy.min_available, 1):
        return [_action(policy, "drain", reason, target.job_id)]
    return []


def _trim(policy: KeepAlivePolicy, view: _View):
    if len(view.live) <= policy.replicas:
        return []
    oldest = min(view.live, key=lambda ep: ep.ready_at or ep.created_at)
    return [_action(policy, "drain", f"above {policy.replicas} replicas", oldest.job_id)]


def _capacity(policy: KeepAlivePolicy, view: _View, now: datetime):
    """Respawn, roll or trim: at most one capacity action per cycle."""
    if len(view.live) + len(view.pending) < policy.replicas:
        if view.pending:
            return []
        short = policy.replicas - len(view.live)
        return [_action(policy, "provision", f"respawn: {short} of {policy.replicas} missing")]
    if view.pending or view.draining:
        return []
    if view.due:
        return _roll_one(policy, view, now)
    return _trim(policy, view)


def plan_service(
    policy: KeepAlivePolicy,
    endpoints: Iterable[RegisteredEndpoint],
    *,
    walltime_sec: int | None,
    now: datetime,
) -> list[RelayAction]:
    """Controller actions for one ``service`` policy given the registry rows (pure)."""
    walltime = policy.walltime_sec or walltime_sec
    view = _view(policy, endpoints, now)
    view.due = _due(policy, view, walltime, now)
    return (
        _retire_drained(policy, view, walltime, now)
        + _retire_stale(policy, view)
        + _capacity(policy, view, now)
    )
