from types import SimpleNamespace

import pytest

from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessResult, ReadinessState
from umrs.orchestrator.service import OrchestrationService


@pytest.mark.asyncio
async def test_wait_for_ready_budget_expands_timeout(monkeypatch):
    captured = {}

    class _TempPoller:
        def __init__(self, config):
            captured["timeout_sec"] = config.timeout_sec
            self.config = config

        async def wait_for_ready(self, **kwargs):
            captured["kwargs"] = kwargs
            return ReadinessResult(
                state=ReadinessState.READY,
                job_id=kwargs["job_id"],
                host=kwargs["host"],
                port=kwargs["port"],
                models=[kwargs["probe_model"]],
            )

        async def close(self):
            captured["closed"] = True

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(readiness_poller=base_poller)
    loop = pytest.importorskip("asyncio").get_running_loop()

    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _TempPoller)

    result = await service._wait_for_ready_with_budget(
        job_id="job-123",
        host="ready.test",
        port=8001,
        deadline=loop.time() + 600,
        probe_model="qwen3-8b",
        probe_kind="chat",
    )

    assert result.is_ready
    assert captured["timeout_sec"] >= 599
    assert captured["kwargs"]["probe_model"] == "qwen3-8b"
    assert captured["kwargs"]["probe_kind"] == "chat"
    assert captured["closed"] is True


class _Backend:
    def __init__(self, status: JobStatus):
        self.status = status
        self.cleanup_calls = []

    async def cleanup(self, job_id: str, force: bool = False):
        self.cleanup_calls.append((job_id, force))

    async def get_job_status(self, job_id: str):
        return SimpleNamespace(job_id=job_id, status=self.status)


class _PortManager:
    def __init__(self):
        self.released = []

    def release(self, port: int, host: str):
        self.released.append((host, port))


@pytest.mark.asyncio
async def test_deprovision_passes_force_and_preserves_registry_when_backend_active():
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="e5-base-v2",
        job_id="175407",
        host="ng-dgx-m0-08",
        port=8001,
        backend="slurm-podman",
    )
    backend = _Backend(JobStatus.RUNNING)
    port_manager = _PortManager()
    service = OrchestrationService(
        registry=registry,
        port_manager=port_manager,
        backend=backend,
    )

    success = await service.deprovision("175407", force=True)

    assert success is False
    assert backend.cleanup_calls == [("175407", True)]
    assert registry.get_by_job_id("175407") is not None
    assert registry.get_by_job_id("175407").metadata["cleanup_status"] == "running"
    assert port_manager.released == []


@pytest.mark.asyncio
async def test_deprovision_removes_registry_only_after_backend_stopped():
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="e5-base-v2",
        job_id="175408",
        host="ng-dgx-m0-08",
        port=8001,
        backend="slurm-podman",
    )
    backend = _Backend(JobStatus.STOPPED)
    port_manager = _PortManager()
    service = OrchestrationService(
        registry=registry,
        port_manager=port_manager,
        backend=backend,
    )

    success = await service.deprovision("175408", force=True)

    assert success is True
    assert backend.cleanup_calls == [("175408", True)]
    assert registry.get_by_job_id("175408") is None
    assert port_manager.released == [("ng-dgx-m0-08", 8001)]