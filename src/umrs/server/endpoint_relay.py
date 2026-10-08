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
import logging
import os
from datetime import datetime, timezone

from ..orchestrator.endpoint_registry import get_registry
from ..orchestrator.endpoint_relay import (
    KeepAlivePolicy,
    RelayAction,
    deadline_metadata,
    keep_alive_policies,
    needs_deadline,
    parse_slurm_duration,
    plan_relay,
)
from ..orchestrator.provision_roster import roster_entries

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


async def _provision(policy: KeepAlivePolicy, action: RelayAction) -> None:
    from ..api.endpoints import ProvisionRequest, provision_endpoint

    request = ProvisionRequest(
        model=policy.model,
        project_id=policy.project_id,
        tier="system",
        ttl_sec=0,
        prefer_warm=False,  # the successor must not short-circuit to the old endpoint
    )
    result = await provision_endpoint(
        request=request,
        project_id=policy.project_id,
        is_admin=True,
        requesting_user=policy.user,
        registry=get_registry(),
    )
    job_id = getattr(result, "job_id", None)
    logger.info(
        "endpoint relay: provisioned %s for %s (%s) job=%s",
        policy.model,
        policy.project_id,
        action.reason,
        job_id or "-",
    )


async def _retire(action: RelayAction) -> None:
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
                if action.action == "provision":
                    await _provision(policy, action)
                else:
                    await _retire(action)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the loop
                logger.warning(
                    "endpoint relay: %s %s failed: %s", action.action, action.model, exc
                )
    return attempted


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
