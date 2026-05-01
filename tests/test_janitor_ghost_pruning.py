"""Focused tests for F-012: Ghost endpoint pruning in the Janitor.

Verifies that the Janitor's liveness cleanup removes READY endpoints
whose backend jobs have terminated, while preserving endpoints that are
still alive or cannot be verified.
"""

from dataclasses import dataclass
from typing import Any, Dict

import pytest

from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.janitor import CleanupResult, Janitor, JanitorConfig

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeJobInfo:
    status: Any


class FakeBackend:
    """Minimal backend stub for testing liveness checks."""

    def __init__(self, statuses: Dict[str, str]):
        """statuses maps job_id → status string.

        Use ``"READY"`` / ``"RUNNING"`` for alive, ``"COMPLETED"`` / ``"FAILED"``
        for dead, and ``"ERROR"`` to simulate a backend error.
        """
        from umrs.orchestrator.base import JobStatus

        self._statuses = statuses
        self._job_status_enum = JobStatus

    async def get_job_status(self, job_id: str) -> FakeJobInfo:
        raw = self._statuses.get(job_id, "UNKNOWN")
        if raw == "ERROR":
            raise RuntimeError(f"Simulated backend error for {job_id}")
        try:
            status = self._job_status_enum(raw.lower())
        except ValueError:
            status = raw
        return FakeJobInfo(status=status)

    async def list_jobs(self):
        return []

    async def cleanup(self, job_id: str):
        pass


def _register_ready(registry: EndpointRegistry, job_id: str, model: str = "test-model") -> None:
    registry.register(
        project_id="proj",
        model=model,
        job_id=job_id,
        host="node",
        port=8001,
        backend="compose",
    )
    registry.mark_ready(job_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJanitorGhostPruning:

    async def test_dead_ready_endpoint_is_pruned(self):
        """A READY endpoint whose backend job is COMPLETED is removed."""
        registry = EndpointRegistry()
        _register_ready(registry, "job-dead", model="m1")

        backend = FakeBackend({"job-dead": "COMPLETED"})
        config = JanitorConfig(
            enable_ttl_cleanup=False,
            enable_orphan_cleanup=False,
            enable_stuck_cleanup=False,
            enable_liveness_cleanup=True,
        )
        janitor = Janitor(config=config, registry=registry, backend=backend)

        result = await janitor.run_once()

        assert result.ghosts_pruned == 1
        assert registry.get_by_job_id("job-dead") is None

    async def test_alive_ready_endpoint_is_preserved(self):
        """A READY endpoint whose backend job is RUNNING is kept."""
        registry = EndpointRegistry()
        _register_ready(registry, "job-alive", model="m1")

        backend = FakeBackend({"job-alive": "RUNNING"})
        config = JanitorConfig(
            enable_ttl_cleanup=False,
            enable_orphan_cleanup=False,
            enable_stuck_cleanup=False,
            enable_liveness_cleanup=True,
        )
        janitor = Janitor(config=config, registry=registry, backend=backend)

        result = await janitor.run_once()

        assert result.ghosts_pruned == 0
        assert registry.get_by_job_id("job-alive") is not None

    async def test_backend_error_preserves_endpoint(self):
        """When the backend raises an error, the endpoint is preserved (fail-safe)."""
        registry = EndpointRegistry()
        _register_ready(registry, "job-unknown", model="m1")

        backend = FakeBackend({"job-unknown": "ERROR"})
        config = JanitorConfig(
            enable_ttl_cleanup=False,
            enable_orphan_cleanup=False,
            enable_stuck_cleanup=False,
            enable_liveness_cleanup=True,
        )
        janitor = Janitor(config=config, registry=registry, backend=backend)

        result = await janitor.run_once()

        assert result.ghosts_pruned == 0
        assert registry.get_by_job_id("job-unknown") is not None

    async def test_mixed_alive_and_dead(self):
        """Only dead endpoints are pruned; alive ones stay."""
        registry = EndpointRegistry()
        _register_ready(registry, "job-alive", model="m1")
        _register_ready(registry, "job-dead", model="m2")

        backend = FakeBackend({
            "job-alive": "RUNNING",
            "job-dead": "FAILED",
        })
        config = JanitorConfig(
            enable_ttl_cleanup=False,
            enable_orphan_cleanup=False,
            enable_stuck_cleanup=False,
            enable_liveness_cleanup=True,
        )
        janitor = Janitor(config=config, registry=registry, backend=backend)

        result = await janitor.run_once()

        assert result.ghosts_pruned == 1
        assert registry.get_by_job_id("job-alive") is not None
        assert registry.get_by_job_id("job-dead") is None

    def test_cleanup_result_includes_ghosts_in_total(self):
        """ghosts_pruned contributes to total_cleaned."""
        result = CleanupResult(ghosts_pruned=3)

        assert result.total_cleaned == 3
        assert result.to_dict()["ghosts_pruned"] == 3

    async def test_liveness_disabled_skips_pruning(self):
        """When enable_liveness_cleanup=False, no ghost pruning occurs."""
        registry = EndpointRegistry()
        _register_ready(registry, "job-dead", model="m1")

        backend = FakeBackend({"job-dead": "COMPLETED"})
        config = JanitorConfig(
            enable_ttl_cleanup=False,
            enable_orphan_cleanup=False,
            enable_stuck_cleanup=False,
            enable_liveness_cleanup=False,
        )
        janitor = Janitor(config=config, registry=registry, backend=backend)

        result = await janitor.run_once()

        assert result.ghosts_pruned == 0
        # Endpoint preserved because liveness check was disabled
        assert registry.get_by_job_id("job-dead") is not None
