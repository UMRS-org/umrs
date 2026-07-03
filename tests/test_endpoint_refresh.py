"""Tests for OrchestrationService.refresh_endpoint — recovery of stuck endpoints.

The refresh_endpoint method recovers endpoints stuck in PROVISIONING/STARTING
with host="pending" after a _finalize_provision task died (e.g. UMRS restart).
"""

from types import SimpleNamespace

import pytest

from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessResult, ReadinessState
from umrs.orchestrator.service import OrchestrationService


class _FakeBackend:
    """Fake backend that returns a real host from get_job_status."""

    def __init__(self, host: str = "compute-01", status: JobStatus = JobStatus.RUNNING):
        self._host = host
        self._status = status

    async def get_job_status(self, job_id: str):
        return SimpleNamespace(
            job_id=job_id,
            status=self._status,
            endpoint=SimpleNamespace(node=self._host, litellm_port=11434),
        )


class _FakeReadinessPoller:
    """Fake readiness poller that returns READY after one call."""

    def __init__(self):
        self.config = ReadinessConfig(timeout_sec=300)

    async def wait_for_ready(self, **kwargs):
        return ReadinessResult(
            state=ReadinessState.READY,
            job_id=kwargs["job_id"],
            host=kwargs["host"],
            port=kwargs["port"],
            models=["test-model"],
        )


@pytest.mark.asyncio
async def test_refresh_resolves_pending_host(monkeypatch):
    """Refresh should re-resolve host from backend when host is 'pending'."""
    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="test-model",
        job_id="job-123",
        host="pending",
        port=11434,
        backend="slurm",
        metadata={"readiness_probe": "chat"},
    )

    service = OrchestrationService(
        registry=registry,
        readiness_poller=_FakeReadinessPoller(),
    )
    service._backend = _FakeBackend(host="compute-01")
    service._backend_type = SimpleNamespace(value="slurm")

    endpoint = await service.refresh_endpoint("job-123")

    assert endpoint is not None
    assert endpoint.host == "compute-01"
    assert endpoint.status == EndpointStatus.READY


@pytest.mark.asyncio
async def test_refresh_handles_already_resolved_host(monkeypatch):
    """Refresh should work when host is already resolved (not 'pending')."""
    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="test-model",
        job_id="job-456",
        host="compute-02",
        port=11434,
        backend="slurm",
        metadata={"readiness_probe": "chat"},
    )

    service = OrchestrationService(
        registry=registry,
        readiness_poller=_FakeReadinessPoller(),
    )
    service._backend = _FakeBackend(host="compute-02")
    service._backend_type = SimpleNamespace(value="slurm")

    endpoint = await service.refresh_endpoint("job-456")

    assert endpoint is not None
    assert endpoint.host == "compute-02"
    assert endpoint.status == EndpointStatus.READY


@pytest.mark.asyncio
async def test_refresh_returns_none_for_nonexistent_endpoint():
    """Refresh should return None if the endpoint doesn't exist in the registry."""
    registry = EndpointRegistry()

    service = OrchestrationService(
        registry=registry,
        readiness_poller=_FakeReadinessPoller(),
    )

    endpoint = await service.refresh_endpoint("nonexistent-job")

    assert endpoint is None
