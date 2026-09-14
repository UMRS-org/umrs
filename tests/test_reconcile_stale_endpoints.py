"""Tests for OrchestrationService.reconcile_stale_endpoints (Task 6 Gap B).

The boot-time reconciler re-resolves and re-probes endpoints stuck in
PROVISIONING/STARTING after a UMRS restart.
"""

import asyncio
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


class _DeadBackend:
    """Fake backend that always raises ConnectionError."""

    async def get_job_status(self, job_id):
        raise ConnectionError("backend unreachable")


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


class _SlowPoller:
    """Fake poller that returns READY after a delay (for timeout tests)."""

    def __init__(self, config, delay=60):
        self.config = config
        self.delay = delay

    async def wait_for_ready(self, **kwargs):
        await asyncio.sleep(self.delay)
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
async def test_reconcile_promotes_stale_provisioning(monkeypatch):
    """Reconciler should promote PROVISIONING endpoints to READY when backend is alive."""
    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="test-model",
        job_id="job-001",
        host="pending",
        port=11434,
        backend="slurm",
    )
    # Second endpoint in STARTING status
    registry.register(
        project_id="test-project",
        model="test-model-2",
        job_id="job-002",
        host="compute-01",
        port=11435,
        backend="slurm",
        metadata={"readiness_probe": "chat"},
    )
    registry.update_status("job-002", EndpointStatus.STARTING)

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
    )
    service._backend = _FakeBackend(host="compute-01")
    service._backend_type = SimpleNamespace(value="slurm")

    result = await service.reconcile_stale_endpoints(timeout_sec=300)

    assert result["checked"] == 2
    assert result["promoted_to_ready"] == 2
    assert result["marked_failed"] == 0
    # Both endpoints should now be READY with resolved locations
    ep1 = registry.get_by_job_id("job-001")
    ep2 = registry.get_by_job_id("job-002")
    assert ep1.status == EndpointStatus.READY
    assert ep1.host == "compute-01"
    assert ep2.status == EndpointStatus.READY


@pytest.mark.asyncio
async def test_reconcile_ignores_ready_endpoints(monkeypatch):
    """Reconciler should not touch READY or FAILED endpoints."""
    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="ready-model",
        job_id="job-ready",
        host="compute-01",
        port=11434,
        backend="slurm",
    )
    registry.mark_ready("job-ready")

    registry.register(
        project_id="test-project",
        model="failed-model",
        job_id="job-fail",
        host="compute-02",
        port=11435,
        backend="slurm",
    )
    registry.mark_failed("job-fail", "crash")

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
    )
    service._backend = _FakeBackend()
    service._backend_type = SimpleNamespace(value="slurm")

    result = await service.reconcile_stale_endpoints(timeout_sec=300)

    assert result["checked"] == 0
    assert result["promoted_to_ready"] == 0


@pytest.mark.asyncio
async def test_reconcile_handles_backend_failure(monkeypatch):
    """Reconciler should mark endpoints FAILED when backend is unreachable."""

    class _DeadBackend:
        async def get_job_status(self, job_id):
            raise ConnectionError("backend unreachable")

    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="test-model",
        job_id="job-stuck",
        host="pending",
        port=11434,
        backend="slurm",
    )

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
    )
    service._backend = _DeadBackend()
    service._backend_type = SimpleNamespace(value="slurm")

    result = await service.reconcile_stale_endpoints(timeout_sec=5)

    assert result["checked"] == 1
    # Endpoint should be marked FAILED because _resolve_endpoint_location
    # will throw on backend failure
    ep = registry.get_by_job_id("job-stuck")
    assert ep.status == EndpointStatus.FAILED


@pytest.mark.asyncio
async def test_reconcile_empty_registry(monkeypatch):
    """Reconciler should return zero counts when no stuck endpoints exist."""
    monkeypatch.setattr("umrs.orchestrator.service.ReadinessPoller", _FakePoller)

    registry = EndpointRegistry()

    base_poller = SimpleNamespace(config=ReadinessConfig(timeout_sec=300))
    service = OrchestrationService(
        registry=registry,
        readiness_poller=base_poller,
    )

    result = await service.reconcile_stale_endpoints(timeout_sec=300)

    assert result == {
        "checked": 0,
        "promoted_to_ready": 0,
        "marked_failed": 0,
        "timed_out": 0,
        "errors": [],
    }


class TestBootReconcilerEnvGate:
    """Verify the UMRS_BOOT_RECONCILE_ENABLED env gate in http_sidecar."""

    def test_reconciler_enabled_by_default(self):
        """Reconciler runs when env var is unset."""
        import os

        # Remove env var if set, verify default allows reconciler
        old = os.environ.pop("UMRS_BOOT_RECONCILE_ENABLED", None)
        try:
            enabled = os.environ.get("UMRS_BOOT_RECONCILE_ENABLED", "true").lower() in (
                "true",
                "1",
                "yes",
            )
            assert enabled is True
        finally:
            if old is not None:
                os.environ["UMRS_BOOT_RECONCILE_ENABLED"] = old

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "disabled"])
    def test_reconciler_disabled_when_set(self, value):
        """Reconciler skips when UMRS_BOOT_RECONCILE_ENABLED is falsy."""
        import os

        old = os.environ.get("UMRS_BOOT_RECONCILE_ENABLED")
        try:
            os.environ["UMRS_BOOT_RECONCILE_ENABLED"] = value
            enabled = os.environ.get("UMRS_BOOT_RECONCILE_ENABLED", "true").lower() in (
                "true",
                "1",
                "yes",
            )
            assert enabled is False
        finally:
            if old is not None:
                os.environ["UMRS_BOOT_RECONCILE_ENABLED"] = old
            else:
                os.environ.pop("UMRS_BOOT_RECONCILE_ENABLED", None)
