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

    def __init__(self, host="compute-01", status=JobStatus.RUNNING):
        self._host = host
        self._status = status
        self.config = SimpleNamespace(readiness_poll_interval_sec=1)

    async def get_job_status(self, job_id):
        return SimpleNamespace(
            job_id=job_id,
            status=self._status,
            endpoint=SimpleNamespace(node=self._host, litellm_port=11434),
        )


class _FakePoller:
    """Fake readiness poller that returns READY immediately."""

    def __init__(self, config):
        self.config = config

    async def wait_for_ready(self, **kwargs):
        return ReadinessResult(
            state=ReadinessState.READY,
            job_id=kwargs["job_id"],
            host=kwargs["host"],
            port=kwargs["port"],
            models=["test-model"],
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_refresh_resolves_pending_host(monkeypatch):
    """Refresh should re-resolve host from backend when host is 'pending'."""
    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

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

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
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
    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

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

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
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

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
    )

    endpoint = await service.refresh_endpoint("nonexistent-job")

    assert endpoint is None
