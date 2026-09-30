"""Live bdrt 2026-09-27: endpoints never became READY. Serve jobs were
cancelled a minute or two after start, but provisioning waited on HTTP probes
alone, inside one budget sized max(readiness, startup timeout), which the host
.env set to 48 h.

Pinned here: the provision budget bounds the queue wait only; readiness gets
its own clock capped by the recipe's readiness_timeout_sec; the wait asks the
backend whether the job still exists (and so does the location wait)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.orchestrator.readiness import (
    BACKEND_JOB_ENDED,
    ReadinessConfig,
    ReadinessPoller,
    ReadinessResult,
    ReadinessState,
)
from umrs.orchestrator.service import OrchestrationService, ProvisionRequest

_48H = 48 * 3600


class _Backend:
    """Plain get_job_status says the job is alive; the before-READY view
    (squeue does not list it, sacct errors) says it ended."""

    def __init__(self, before_ready: JobStatus = JobStatus.FAILED):
        self.config = SimpleNamespace(readiness_poll_interval_sec=1)
        self.before_ready = before_ready
        self.cleanup_calls: list[str] = []

    async def get_job_status(self, job_id):
        return SimpleNamespace(job_id=job_id, status=JobStatus.RUNNING, endpoint=None)

    async def get_job_status_before_ready(self, job_id):
        return SimpleNamespace(
            job_id=job_id,
            status=self.before_ready,
            endpoint=None,
            error_message="Slurm no longer lists the job and sacct is unavailable",
        )

    async def cleanup(self, job_id, force=False):
        self.cleanup_calls.append(job_id)


class _PortManager:
    def release(self, port, host):
        pass


def _service(backend, poller_config: ReadinessConfig) -> OrchestrationService:
    registry = EndpointRegistry()
    registry.register(
        project_id="p",
        model="m",
        job_id="4242",
        host="gb10-01",
        port=12001,
        backend="slurm-remote",
    )
    return OrchestrationService(
        registry=registry,
        port_manager=_PortManager(),
        readiness_poller=ReadinessPoller(poller_config),
        backend=backend,
    )


async def _finalize(service, request, *, start_offset: float, budget: float):
    loop = asyncio.get_running_loop()
    start_time = loop.time() - start_offset
    return await service._finalize_provision(
        request=request,
        job_id="4242",
        target_host="pending",
        actual_host="gb10-01",
        actual_port=12001,
        deadline=start_time + budget,
        progress=None,
        start_time=start_time,
    )


async def test_ended_backend_job_fails_provision_and_cleans_up(monkeypatch):
    async def never_ready(self, *_a, **_k):
        return False, False, [], "connection refused"

    async def no_sleep(self, *_a, **_k):
        return None

    monkeypatch.setattr(ReadinessPoller, "_check_once", never_ready)
    monkeypatch.setattr(ReadinessPoller, "_sleep_before_next_attempt", no_sleep)
    backend = _Backend()
    service = _service(
        backend, ReadinessConfig(timeout_sec=_48H, liveness_interval_sec=0.0)
    )

    result = await asyncio.wait_for(
        _finalize(service, ProvisionRequest("p", "m"), start_offset=0, budget=_48H),
        5,
    )

    assert result.success is False
    assert result.error == BACKEND_JOB_ENDED
    assert backend.cleanup_calls == ["4242"]
    assert service.registry.get_by_job_id("4242") is None


def _capture_readiness_timeout(monkeypatch) -> dict:
    captured: dict = {}

    class _Poller:
        def __init__(self, config):
            captured["timeout_sec"] = config.timeout_sec
            self.config = config

        async def wait_for_ready(self, **kwargs):
            captured["liveness"] = kwargs.get("liveness")
            return ReadinessResult(
                state=ReadinessState.READY,
                job_id=kwargs["job_id"],
                host=kwargs["host"],
                port=kwargs["port"],
            )

        async def close(self):
            pass

    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _Poller)
    return captured


def _plan(readiness_timeout_sec: int) -> dict:
    return {"serving_plan": {"readiness_timeout_sec": readiness_timeout_sec}}


@pytest.mark.parametrize(
    ("metadata", "budget", "expected"),
    [
        ({}, _48H, 3600),  # no recipe: ResolvedEndpointSpec default
        (_plan(900), _48H, 900),  # recipe timeout caps a 48 h budget
        (_plan(7200), _48H, 7200),  # a slow-loading recipe keeps its hour two
        (_plan(7200), 2400, 2400),  # never longer than the old budget
        ({"serving_plan": {"readiness_timeout_sec": "junk"}}, _48H, 3600),
    ],
)
async def test_readiness_timeout_is_capped_by_the_recipe(
    monkeypatch, metadata, budget, expected
):
    captured = _capture_readiness_timeout(monkeypatch)
    service = _service(_Backend(), ReadinessConfig(timeout_sec=budget))
    request = ProvisionRequest("p", "m", metadata=metadata)

    result = await _finalize(service, request, start_offset=0, budget=budget)

    assert result.success is True
    assert expected - 1 <= captured["timeout_sec"] <= expected
    assert captured["liveness"] is not None


async def test_readiness_clock_starts_when_the_host_is_known(monkeypatch):
    # 3000 s of a 3600 s budget went to the queue; readiness still gets its
    # full hour instead of the 600 s left over.
    captured = _capture_readiness_timeout(monkeypatch)
    service = _service(_Backend(), ReadinessConfig(timeout_sec=3600))

    await _finalize(
        service, ProvisionRequest("p", "m"), start_offset=3000, budget=3600
    )

    assert captured["timeout_sec"] >= 3599


async def test_location_wait_ends_when_the_job_vanished_before_ready():
    service = _service(_Backend(), ReadinessConfig(timeout_sec=300))
    deadline = asyncio.get_running_loop().time() + 2

    with pytest.raises(RuntimeError, match="failed before publishing an endpoint"):
        await service._resolve_endpoint_location("4242", "pending", 12001, deadline)


async def test_location_wait_falls_back_to_get_job_status():
    class _PlainBackend:
        config = SimpleNamespace(readiness_poll_interval_sec=1)

        async def get_job_status(self, job_id):
            return SimpleNamespace(
                job_id=job_id,
                status=JobStatus.RUNNING,
                endpoint=SimpleNamespace(node="gb10-02", litellm_port=12002),
            )

    service = _service(_PlainBackend(), ReadinessConfig(timeout_sec=300))
    deadline = asyncio.get_running_loop().time() + 2

    location = await service._resolve_endpoint_location(
        "4242", "pending", 12001, deadline
    )

    assert location == ("gb10-02", 12002)
