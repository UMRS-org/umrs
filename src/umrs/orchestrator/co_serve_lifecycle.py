"""Registry rows and role probes of co-served jobs (ADR-109 D-2).

The orchestration service registers the primary row of a job; this module
adds one row per co-serving of the same job and runs each one's role probe
once its primary is READY. The rows' way down (drain, failure, removal, TTL
renewal, location) follows the primary inside the registry
(``EndpointRegistry._follow_status`` and friends), so lifecycle actions keep
acting on the job.
"""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import Any

from .co_serve import co_job_id, co_port, from_metadata, row_metadata
from .endpoint_registry import EndpointRegistry, EndpointStatus, RegisteredEndpoint

logger = logging.getLogger(__name__)

#: How long a co-serving's role probe may take after its primary is READY
#: (the sidecar is up by then; this covers the first model load).
READY_TIMEOUT_ENV = "UMRS_CO_SERVE_READY_TIMEOUT_SEC"
_DEFAULT_READY_TIMEOUT_SEC = 900

_UNFINISHED = (
    EndpointStatus.PENDING,
    EndpointStatus.PROVISIONING,
    EndpointStatus.STARTING,
)


def ready_timeout_sec() -> int:
    try:
        value = int(os.getenv(READY_TIMEOUT_ENV, "") or _DEFAULT_READY_TIMEOUT_SEC)
    except ValueError:
        return _DEFAULT_READY_TIMEOUT_SEC
    return value if value > 0 else _DEFAULT_READY_TIMEOUT_SEC


def register_co_served(
    registry: EndpointRegistry,
    primary: RegisteredEndpoint,
    backend_job_id: str | None,
) -> list[RegisteredEndpoint]:
    """Register one row per co-serving of ``primary``'s job (none without)."""
    specs = from_metadata(primary.metadata)
    if not specs:
        return []
    from .co_serve import JOB_GROUP_KEY

    registry.update_metadata(primary.job_id, {JOB_GROUP_KEY: primary.job_id})
    port = co_port(int(primary.metadata.get("allocated_port") or primary.port))
    rows = []
    for spec in specs:
        row = registry.register(
            project_id=primary.project_id,
            model=spec.model,
            job_id=co_job_id(primary.job_id, spec.name),
            host=primary.host,
            port=port,
            backend=primary.backend,
            ttl_sec=0,  # the primary's expiry, set below (renewals follow it)
            metadata=row_metadata(
                spec, primary.job_id, primary.metadata, primary.host, port
            ),
        )
        if primary.expires_at is not None:
            registry.extend_ttl(row.job_id, primary.expires_at)
        if backend_job_id:
            registry.set_backend_job_id(row.job_id, backend_job_id)
        rows.append(row)
    logger.info(
        "co-served rows of job %s: %s", primary.job_id, [r.job_id for r in rows]
    )
    return rows


def _probe_spec(row: RegisteredEndpoint) -> dict[str, str | None]:
    """The role probe of a co-served row: its adapter's segmentation."""
    metadata = row.metadata or {}
    return {
        "model": str(metadata.get("vision_adapter") or row.model),
        "kind": str(metadata.get("role_probe") or metadata.get("readiness_probe") or "http"),
        "path": str(metadata.get("readiness_path") or "/readyz"),
    }


async def _await_role_probe(service: Any, job_id: str, row: RegisteredEndpoint, host: str):
    loop = asyncio.get_running_loop()
    return await service._wait_for_ready_with_budget(
        job_id=row.job_id,
        host=host,
        port=row.port,
        deadline=loop.time() + ready_timeout_sec(),
        readiness_probe=_probe_spec(row),
        liveness=partial(service._backend_job_running, job_id),
    )


async def probe_co_served_row(
    service: Any, job_id: str, row: RegisteredEndpoint, host: str
) -> bool:
    """Run one co-served row's role probe; READY or FAILED (only that row)."""
    registry = service.registry
    registry.update_status(row.job_id, EndpointStatus.STARTING)
    registry.update_metadata(row.job_id, {"detail": "role probe: one segmentation"})
    try:
        result = await _await_role_probe(service, job_id, row, host)
    except Exception as exc:  # noqa: BLE001 - a failed sidecar never fails the job
        registry.mark_failed(row.job_id, f"role probe error: {exc}")
        return False
    if result.is_ready:
        registry.mark_ready(row.job_id)
        registry.update_metadata(row.job_id, {"detail": "role probe passed"})
        return True
    error = result.error or f"role probe {result.state.value}"
    registry.mark_failed(row.job_id, error)
    logger.warning("co-served %s of job %s failed: %s", row.model, job_id, error)
    return False


async def probe_co_served(service: Any, job_id: str) -> int:
    """Role-probe every unfinished co-served row of a READY job; READY count."""
    primary = service.registry.get_by_job_id(job_id)
    if primary is None or primary.status != EndpointStatus.READY:
        return 0
    ready = 0
    for row in service.registry.co_rows(job_id):
        if row.status in _UNFINISHED:
            ready += await probe_co_served_row(service, job_id, row, primary.host)
    return ready


def start_co_served_probes(service: Any, job_id: str) -> asyncio.Task | None:
    """Probe the job's co-served rows in the background (None without any)."""
    if not service.registry.co_rows(job_id):
        return None
    task = asyncio.create_task(probe_co_served(service, job_id))
    service._resume_tasks.add(task)
    task.add_done_callback(service._resume_tasks.discard)
    return task


def resume_co_served(service: Any) -> list[str]:
    """After a restart: probe the unfinished co-served rows of READY jobs.

    A job that is not READY yet resumes through its own provision, which
    starts these probes when it turns READY.
    """
    jobs = {
        str(row.metadata.get("co_served_by"))
        for row in service.registry.list_all(status_filter=list(_UNFINISHED))
        if (row.metadata or {}).get("co_served_by")
    }
    return [job for job in sorted(jobs) if start_co_served_probes(service, job)]
