"""Focused tests for F-012: Ghost endpoint pruning in the Janitor.

Verifies that the Janitor's liveness cleanup removes READY endpoints
whose backend jobs have terminated, while preserving endpoints that are
still alive or cannot be verified.
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from umrs.orchestrator.endpoint_registry import EndpointRegistry
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

    def __init__(self, statuses: dict[str, str]):
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

        backend = FakeBackend(
            {
                "job-alive": "RUNNING",
                "job-dead": "FAILED",
            }
        )
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


class TestOrphanFailureBackoff:
    """Venue-health backoff (2026-09-18): a wedged venue should not produce
    one ERROR per janitor tick — first error, then backoff, then recovery."""

    def _janitor(self):
        from umrs.orchestrator.janitor import Janitor, JanitorConfig

        backend = MagicMock()
        backend.backend_name = "slurm-remote"
        j = Janitor.__new__(Janitor)
        # Properties at class level back the accessor; set the underscore slot.
        j._backend = backend
        j.config = JanitorConfig()
        j._locks = {}
        return j, backend

    @staticmethod
    def _fail(j, times: int, caplog) -> list[tuple[int, str]]:
        import logging

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="umrs.orchestrator.janitor"):
            for _ in range(times):
                j._orphan_failures_record_failure(RuntimeError("Connection closed"))
        return [(r.levelno, r.getMessage()) for r in caplog.records]

    def test_first_failure_is_one_warning_line(self, caplog):
        """ops-watch #639: one SSH banner timeout is transient, not an ERROR."""
        j, _ = self._janitor()
        assert self._fail(j, 1, caplog) == [
            (
                30,
                "Orphan cleanup failed (backend=slurm-remote, streak=1): Connection closed",
            )
        ]

    def test_streak_below_threshold_warns_then_escalates_to_error(self, caplog):
        j, _ = self._janitor()
        records = self._fail(j, 3, caplog)
        assert [level for level, _ in records] == [30, 30, 40]
        # One message text at every level: the Ops Watch fingerprint
        # normalizes the streak number, so the ERROR keeps its issue.
        assert [message for _, message in records] == [
            f"Orphan cleanup failed (backend=slurm-remote, streak={n}): Connection closed"
            for n in (1, 2, 3)
        ]

    def test_backoff_mid_streak_only_at_summary_boundary(self, caplog):
        j, _ = self._janitor()
        records = self._fail(j, 64, caplog)
        errors = [message for level, message in records if level == 40]
        # streak 3 escalates; then quiet until each 32-failure summary.
        assert len(records) == 5
        assert len(errors) == 3
        assert "streak=32 consecutive" in errors[1]
        assert "streak=64 consecutive" in errors[2]

    def test_recovery_emits_info_log_with_streak_carried(self, caplog):
        j, _ = self._janitor()
        import logging

        with caplog.at_level(logging.INFO, logger="umrs.orchestrator.janitor"):
            j._orphan_failures_record_failure(RuntimeError("Connection closed"))
            j._orphan_failures_record_failure(RuntimeError("Connection closed"))
            caplog.clear()
            j._orphan_failures_record_success()
        lines = [r.getMessage() for r in caplog.records if r.levelno == 20]
        assert any("recovered" in line and "2" in line for line in lines)
        # second recovery attempt — no streak, silent.
        caplog.clear()
        j._orphan_failures_record_success()
        assert not any("recovered" in r.getMessage() for r in caplog.records)

    def test_streaks_are_per_backend(self, caplog):
        j, backend = self._janitor()
        import logging

        with caplog.at_level(logging.WARNING, logger="umrs.orchestrator.janitor"):
            j._orphan_failures_record_failure(RuntimeError("Connection closed"))
            backend.backend_name = "k8s"
            j._orphan_failures_record_failure(RuntimeError("dns lookup timeout"))
        warnings = [r.getMessage() for r in caplog.records if r.levelno == 30]
        # two different backends → each got its own first-failure WARNING
        assert len(warnings) == 2
        assert "backend=slurm-remote, streak=1" in warnings[0]
        assert "backend=k8s, streak=1" in warnings[1]
