"""Executor for the walltime relay (``umrs.orchestrator.endpoint_relay``).

Runs beside the janitor in the server lifespan. Each cycle reads the
``keep_alive`` roster entries, plans against the registry and executes the
actions through the same provision/deprovision handlers ``r4 endpoint up`` and
``r4 endpoint down`` reach, so quota, recipe policy, partition validation and
audit stay on one path.

Environment:
    UMRS_ENDPOINT_RELAY           "false" disables the loop (default enabled;
                                  it is a no-op without keep_alive entries)
    UMRS_ENDPOINT_RELAY_INTERVAL  seconds between cycles (default 120)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from ..orchestrator.endpoint_registry import get_registry
from ..orchestrator.endpoint_relay import (
    _LIVE,
    _PENDING,
    KeepAlivePolicy,
    RelayAction,
    _owned,
    deadline_metadata,
    keep_alive_policies,
    needs_deadline,
    parse_slurm_duration,
    plan_relay,
)
from ..orchestrator.provision_roster import role_alias_models, roster_entries

logger = logging.getLogger(__name__)


def _site_walltime_sec() -> int | None:
    try:
        from ..orchestrator.site_config import get_site_config

        return parse_slurm_duration(get_site_config().scheduler.time_limit)
    except Exception as exc:  # noqa: BLE001 - no site config = no deadline
        logger.debug("endpoint relay: no site time_limit (%s)", exc)
        return None


def _relay_backend():
    """The configured orchestrator backend, or ``None`` (no deadline reads)."""
    try:
        from ..orchestrator.factory import (
            get_orchestrator,
            resolve_configured_backend_name,
        )

        configured = resolve_configured_backend_name()
        return get_orchestrator(configured) if configured else None
    except Exception as exc:  # noqa: BLE001 - deadlines are an optimisation
        logger.debug("endpoint relay: no backend for deadline reads (%s)", exc)
        return None


async def _read_deadline(backend, endpoint, now: datetime) -> None:
    try:
        left = await backend.time_left_sec(endpoint.job_id)
    except Exception as exc:  # noqa: BLE001 - fall back to created_at + walltime
        logger.debug(
            "endpoint relay: time left of %s unknown (%s)", endpoint.job_id, exc
        )
        return
    if left:
        metadata = deadline_metadata(now, left)
        get_registry().update_metadata(endpoint.job_id, metadata)
        endpoint.metadata.update(metadata)


async def refresh_deadlines(policies, endpoints, now: datetime) -> None:
    """Record the Slurm-granted end of each READY keep-alive endpoint once.

    A job started with a shortened limit (``--time-min`` before a
    reservation) ends before ``created_at + time_limit``; without its real
    deadline the relay would start the successor too late.
    """
    due = [ep for ep in endpoints for p in policies if needs_deadline(p, ep)]
    if not due:
        return
    backend = _relay_backend()
    if backend is None or not hasattr(backend, "time_left_sec"):
        return
    for endpoint in due:
        await _read_deadline(backend, endpoint, now)


def provision_result(result: Any) -> dict[str, Any]:
    """``status`` / ``job_id`` / ``error`` of a ``provision_endpoint`` answer.

    A ``wait=False`` submit answers a 202 ``JSONResponse`` (the body carries
    the job id), a ready or reused endpoint a ``ProvisionResponse``. Reading
    only the attribute lost the job id of every relay successor since #720,
    so the replica's recipe digest (ADR-106 D-4) was never recorded.
    """
    body = getattr(result, "body", None)
    if isinstance(body, (bytes, bytearray)):
        try:
            data = json.loads(body)
        except ValueError:
            data = {}
    else:
        data = {k: getattr(result, k, None) for k in ("status", "job_id", "error")}
    return data if isinstance(data, dict) else {}


def replica_metadata(policy: KeepAlivePolicy, action: RelayAction) -> dict:
    """What a new service replica records: its recipe digest (ADR-106 D-4)
    and, for a surge that retries a failed co-serving, that mark (ADR-109 D-2)."""
    from ..orchestrator.service_controller import CO_RETRY_KEY, DIGEST_KEY

    metadata: dict = {DIGEST_KEY: policy.digest}
    if action.co_retry:
        metadata[CO_RETRY_KEY] = True
    return metadata


async def _provision(
    policy: KeepAlivePolicy, action: RelayAction, now: datetime
) -> str | None:
    from ..api.endpoints import ProvisionRequest, provision_endpoint

    request = ProvisionRequest(
        model=policy.model,
        project_id=policy.project_id,
        tier="system",
        ttl_sec=0,
        prefer_warm=False,  # the successor must not short-circuit to the old endpoint
        readiness_probe=policy.readiness_probe,  # ADR-106 D-7 (None = provision default)
        # Return at submit: readiness finishes in the background and the next
        # cycle counts the row as pending. Waiting here held the whole relay
        # (every role's drain, retire, respawn) on one successor's model load,
        # and up to the 24 h readiness timeout while it queued in Slurm.
        wait=False,
    )
    result = await provision_endpoint(
        request=request,
        project_id=policy.project_id,
        is_admin=True,
        requesting_user=policy.user,
        registry=get_registry(),
    )
    job_id = provision_result(result).get("job_id")
    if policy.service and job_id:
        # ADR-106 D-4: the replica remembers the recipe it was built from;
        # a later change to the entry makes it due for an upgrade roll.
        get_registry().update_metadata(str(job_id), replica_metadata(policy, action))
    logger.info(
        "endpoint relay: provisioned %s for %s (%s) job=%s",
        policy.model,
        policy.project_id,
        action.reason,
        job_id or "-",
    )
    return str(job_id) if job_id else None


async def _drain(
    policy: KeepAlivePolicy, action: RelayAction, now: datetime
) -> None:
    """ADR-106 D-5: stop routing new requests; retire follows once drained."""
    from ..orchestrator.service_controller import DRAINING_SINCE_KEY

    registry = get_registry()
    registry.mark_draining(str(action.job_id))
    registry.update_metadata(str(action.job_id), {DRAINING_SINCE_KEY: now.isoformat()})
    logger.info(
        "endpoint relay: draining %s job=%s (%s)",
        action.model,
        action.job_id,
        action.reason,
    )


async def _retire(
    policy: KeepAlivePolicy, action: RelayAction, now: datetime
) -> None:
    from ..api.endpoints import deprovision_endpoint

    await deprovision_endpoint(
        job_id=str(action.job_id),
        force=False,
        project_id=action.project_id,
        registry=get_registry(),
    )
    logger.info(
        "endpoint relay: retired %s job=%s (%s)",
        action.model,
        action.job_id,
        action.reason,
    )


async def run_relay_once(now: datetime | None = None) -> list[RelayAction]:
    """Plan and execute one relay cycle; returns the actions attempted."""
    policies = keep_alive_policies(roster_entries())
    if not policies:
        return []
    # Registry timestamps are naive UTC (datetime.utcnow defaults).
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    walltime = _site_walltime_sec()
    endpoints = get_registry().list_all()
    await refresh_deadlines(policies, endpoints, now)
    attempted: list[RelayAction] = []
    for policy in policies:
        for action in plan_relay(policy, endpoints, walltime_sec=walltime, now=now):
            attempted.append(action)
            try:
                await _EXECUTORS[action.action](policy, action, now)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the loop
                logger.warning(
                    "endpoint relay: %s %s failed: %s", action.action, action.model, exc
                )
    return attempted


_EXECUTORS = {"provision": _provision, "drain": _drain, "retire": _retire}


def service_policies_for(name: str) -> list[KeepAlivePolicy]:
    """The ``service`` policies serving *name*: a role alias (``svc-*``,
    every model whose entry carries it) or a model name. Roster order.

    ADR-109 D-2: a co-served model (``svc-vision2`` = SAM3.1) is served by
    its primary's replicas, so its primary's policy answers for it.
    """
    models = set(role_alias_models(name)) or {name}
    return [
        policy
        for policy in keep_alive_policies(roster_entries())
        if policy.service and models & {policy.model, *policy.co_serve}
    ]


def _has_replica(policies: list[KeepAlivePolicy], endpoints) -> bool:
    return any(
        _owned(policy, ep) and ep.status in (*_LIVE, *_PENDING)
        for policy in policies
        for ep in endpoints
    )


def _service_admits(policy: KeepAlivePolicy, caller: Any) -> bool:
    """Would the caller be routed to this service's replicas? The recipe's
    ``access`` (ADR-078 D-4) decides; no caller = the relay itself."""
    if caller is None:
        return True
    from types import SimpleNamespace

    from ..orchestrator.endpoint_access import can_use

    entry = next((e for e in roster_entries() if e.get("model") == policy.model), {})
    row = SimpleNamespace(
        project_id=policy.project_id, metadata={"access": entry.get("access") or []}
    )
    return can_use(row, caller)


async def ensure_service_replica(
    name: str, *, reason: str, now: datetime | None = None, caller: Any = None
) -> dict[str, Any]:
    """ADR-109 D-4: at least one replica of the service *name* now.

    The relay stays the one owner of service lifetime (ADR-106 D-3): the
    replica is provisioned exactly as a relay cycle would (system tier, the
    entry's project and user, the recipe digest recorded). An on-demand
    entry (``service.replicas: 0``) is never respawned, rolled or trimmed by
    the controller, so its replica lives until ``r4 endpoint down`` or its
    Slurm walltime. A *caller* the service's access list would not route to
    starts nothing (``forbidden``). Answers ``{"outcome": "no_service" |
    "forbidden" | "present" | "provisioned" | "failed", "model", "job_id"}``.
    """
    policies = service_policies_for(name)
    if not policies:
        return {"outcome": "no_service", "model": name, "job_id": None}
    policy = policies[0]
    if not _service_admits(policy, caller):
        return {"outcome": "forbidden", "model": policy.model, "job_id": None}
    if _has_replica(policies, get_registry().list_all()):
        return {"outcome": "present", "model": policy.model, "job_id": None}
    action = RelayAction("provision", policy.model, policy.project_id, reason)
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    job_id = await _provision(policy, action, now)
    outcome = "provisioned" if job_id else "failed"
    return {"outcome": outcome, "model": policy.model, "job_id": job_id}


class EndpointRelay:
    """Background loop around :func:`run_relay_once`."""

    def __init__(self, interval_sec: int | None = None):
        self.interval_sec = interval_sec or int(
            os.getenv("UMRS_ENDPOINT_RELAY_INTERVAL", "120")
        )
        self._task: asyncio.Task | None = None

    def start(self) -> bool:
        if os.getenv("UMRS_ENDPOINT_RELAY", "true").lower() in ("0", "false", "no"):
            return False
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
        return True

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await run_relay_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("endpoint relay cycle failed")
            await asyncio.sleep(self.interval_sec)


_relay: EndpointRelay | None = None


def start_endpoint_relay() -> bool:
    global _relay
    if _relay is None:
        _relay = EndpointRelay()
    return _relay.start()


def stop_endpoint_relay() -> None:
    if _relay is not None:
        _relay.stop()
