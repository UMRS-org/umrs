"""A restart must not orphan a provision (2026-10-02, live rccs).

Each server restart marked a still-queued relay successor FAILED without
cancelling it. The relay then submitted another; the orphans ran
unregistered and callers got 503. Now the boot reconciler defers a row
whose job is alive, the server's loop finishes it (READY, or FAILED with
the job cancelled), and a non-expiring endpoint waits out the queue.
"""

import asyncio
from types import SimpleNamespace

import pytest
from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessResult, ReadinessState
from umrs.orchestrator.service import OrchestrationService


class _Backend:
    """Scripted job states; records cancels."""

    def __init__(self, states, host="node-1"):
        self._states = list(states)
        self._host = host
        self.cancelled: list[str] = []
        self.config = SimpleNamespace(readiness_poll_interval_sec=1, startup_timeout_sec=1)

    async def get_job_status(self, job_id):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        node = "pending" if state == JobStatus.PENDING else self._host
        return SimpleNamespace(
            job_id=job_id,
            status=state,
            error_message=None,
            endpoint=SimpleNamespace(node=node, litellm_port=11434),
        )

    async def cleanup(self, job_id, force=False):
        self.cancelled.append(f"{job_id}!" if force else job_id)


class _Poller:
    def __init__(self, state=ReadinessState.READY):
        self.config = ReadinessConfig(timeout_sec=1)
        self._state = state

    async def wait_for_ready(self, **kwargs):
        return ReadinessResult(
            state=self._state,
            job_id=kwargs["job_id"],
            host=kwargs["host"],
            port=kwargs["port"],
            models=["m"],
        )


def _service(backend, poller=None, host="pending", ttl_sec=0):
    registry = EndpointRegistry()
    registry.register(
        project_id="p",
        model="m",
        job_id="391972",
        host=host,
        port=11434,
        backend="slurm",
        ttl_sec=ttl_sec,
    )
    service = OrchestrationService(registry=registry, readiness_poller=poller or _Poller())
    service._backend = backend
    service._backend_type = SimpleNamespace(value="slurm")
    service._port_manager = SimpleNamespace(release=lambda *_: None)
    return service, registry


@pytest.mark.asyncio
async def test_a_queued_job_is_deferred_at_once_not_failed():
    backend = _Backend([JobStatus.PENDING])
    service, registry = _service(backend)

    result = await service.reconcile_stale_endpoints(timeout_sec=300)

    assert result["deferred"] == 1 and result["marked_failed"] == 0
    assert registry.get_by_job_id("391972").status == EndpointStatus.PROVISIONING
    assert backend.cancelled == []


@pytest.mark.asyncio
async def test_a_running_job_still_loading_is_deferred():
    backend = _Backend([JobStatus.RUNNING])
    service, registry = _service(backend, _Poller(ReadinessState.TIMEOUT), host="node-1")

    result = await service.reconcile_stale_endpoints(timeout_sec=1)

    assert result["deferred"] == 1
    assert registry.get_by_job_id("391972").status != EndpointStatus.FAILED


@pytest.mark.asyncio
async def test_resume_finishes_a_deferred_row_to_ready():
    backend = _Backend([JobStatus.PENDING, JobStatus.PENDING, JobStatus.RUNNING])
    service, registry = _service(backend)

    resumed = await service.resume_unfinished_provisions()
    assert resumed == ["391972"]
    await asyncio.gather(*service._resume_tasks)

    endpoint = registry.get_by_job_id("391972")
    assert endpoint.status == EndpointStatus.READY and endpoint.host == "node-1"


@pytest.mark.asyncio
async def test_resume_that_gives_up_cancels_the_job():
    backend = _Backend([JobStatus.RUNNING])
    service, registry = _service(backend, _Poller(ReadinessState.TIMEOUT), host="node-1")

    await service.resume_unfinished_provisions()
    await asyncio.gather(*service._resume_tasks)

    assert backend.cancelled == ["391972"]  # nothing left running unregistered
    assert registry.get_by_job_id("391972") is None


@pytest.mark.asyncio
async def test_resume_skips_a_job_that_already_ended():
    service, _registry = _service(_Backend([JobStatus.STOPPED]))
    assert await service.resume_unfinished_provisions() == []


@pytest.mark.asyncio
async def test_a_non_expiring_endpoint_waits_out_the_queue():
    # Queued past the 1 s budget, then placed: the relay successor's case.
    states = [JobStatus.PENDING] * 3 + [JobStatus.RUNNING]
    service, _registry = _service(_Backend(states))
    loop = asyncio.get_running_loop()

    host, _port = await service._resolve_endpoint_location(
        "391972", "pending", 11434, loop.time(), queue_patient=True
    )
    assert host == "node-1"


@pytest.mark.asyncio
async def test_an_expiring_endpoint_still_times_out_in_the_queue():
    service, _registry = _service(_Backend([JobStatus.PENDING]))
    loop = asyncio.get_running_loop()

    with pytest.raises(TimeoutError):
        await service._resolve_endpoint_location("391972", "pending", 11434, loop.time())
