"""Focused tests for R-HC1: READY-side engine probe in the Janitor.

The backend job-status check (ghost pruning) cannot see a rivault allocation
whose vllm/litellm stack exited underneath it. These tests pin the
READY ↔ DEGRADED transition semantics the cycle must hold so list surfaces
(registry, /endpoints, status bars, model pickers) carry ground truth.
"""

from __future__ import annotations

import pytest
import umrs.orchestrator.readiness as readiness_mod
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.janitor import Janitor, JanitorConfig

pytestmark = pytest.mark.asyncio


def _register_ready(
    registry: EndpointRegistry,
    job_id: str,
    model: str = "probe-model",
    port: int = 13000,
) -> None:
    registry.register(
        project_id="proj",
        model=model,
        job_id=job_id,
        host="127.0.0.1",
        port=port,
        backend="slurm-remote",
    )
    registry.mark_ready(job_id)


class _ScriptedPoller:
    """check_ready_detail replacement replaying scripted tuples per job_id port."""

    def __init__(self, script: dict[int, list[tuple[bool, bool, list[str], str]]]):
        self._script = script
        self.calls: list[tuple[str, int]] = []

    async def check_ready_detail(
        self,
        host: str,
        port: int,
        *,
        probe_model: str | None = None,
        probe_kind: str = "models",
        probe_path: str | None = None,
    ) -> tuple[bool, bool, list[str], str]:
        self.calls.append((host, port))
        seq = self._script.get(port, [])
        if not seq:
            return (False, False, [], "unscripted-call")
        return seq.pop(0) if seq else (False, False, [], "script empty")


class _JanitorProbeHarness:
    def __init__(self, poller: _ScriptedPoller, **config_kwargs):
        self.registry = EndpointRegistry()
        self.poller = poller
        base = {
            "enable_ttl_cleanup": False,
            "enable_orphan_cleanup": False,
            "enable_stuck_cleanup": False,
            "enable_liveness_cleanup": False,
            "enable_ttl_auto_extend": False,
            "enable_engine_probe": True,
            "engine_probe_degrade_after": 3,
            "engine_probe_timeout_sec": 5,
            # Default grace off so the classic READY→DEGRADED threshold tests
            # exercise the cycle immediately; the grace-aware tests override.
            "engine_probe_grace_sec": 0,
        }
        base.update(config_kwargs)
        config = JanitorConfig(**base)
        self.janitor = Janitor(config=config, registry=self.registry)

    async def probe(self) -> tuple[int, int]:
        return await self.janitor._probe_ready_engines()


@pytest.fixture(autouse=True)
def _patch_poller(monkeypatch):
    current: dict[str, _ScriptedPoller] = {}

    def _factory() -> _ScriptedPoller:
        return current["poller"]

    monkeypatch.setattr(readiness_mod, "get_readiness_poller", _factory)
    yield current


class TestJanitorEngineProbe:
    async def test_ready_degrades_only_after_threshold(self, _patch_poller):
        """READY flips to DEGRADED only after N consecutive probe failures."""
        port = 13001
        script = {
            port: [(False, False, [], "connection refused")] * 3,
        }
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(poller)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-x", port=port)
        endpoint0 = harness.registry.get_by_job_id("job-x")

        (d0, _) = await harness.probe()
        assert d0 == 0
        assert endpoint0.status == EndpointStatus.READY

        (d1, _) = await harness.probe()
        assert d1 == 0
        assert endpoint0.status == EndpointStatus.READY

        (d2, _) = await harness.probe()
        assert d2 == 1
        assert endpoint0.status == EndpointStatus.DEGRADED
        assert "connection refused" in (endpoint0.metadata.get("readiness_error") or "")

        assert [c[1] for c in poller.calls] == [port, port, port]

    async def test_degraded_recovers_to_ready(self, _patch_poller):
        """DEGRADED flips back to READY immediately on a healthy probe."""
        port = 13005
        script = {
            port: [
                (False, False, [], "connection refused"),
                (False, False, [], "connection refused"),
                (False, False, [], "connection refused"),
                (True, False, ["probe-model"], ""),
            ],
        }
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(poller)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-r", port=port)
        endpoint0 = harness.registry.get_by_job_id("job-r")

        await harness.probe()
        await harness.probe()
        (d2, _) = await harness.probe()
        assert d2 == 1
        assert endpoint0.status == EndpointStatus.DEGRADED

        (d3, r3) = await harness.probe()
        assert d3 == 0 and r3 == 1
        assert endpoint0.status == EndpointStatus.READY
        assert harness.janitor._probe_failures.get("job-r") is None
        assert (endpoint0.metadata.get("detail") or "").startswith("probe: recovered")

    async def test_degraded_since_is_stamped_once_and_cleared_on_recovery(self, _patch_poller):
        """The walltime relay reads ``degraded_since`` (2026-10-07: a vanished job
        stayed DEGRADED and blocked its own respawn)."""
        port = 13007
        refused = (False, False, [], "connection refused")
        script = {port: [refused] * 4 + [(True, False, ["probe-model"], "")]}
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(poller)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-s", port=port)
        endpoint0 = harness.registry.get_by_job_id("job-s")

        for _ in range(3):
            await harness.probe()
        stamped = endpoint0.metadata.get("degraded_since")
        assert endpoint0.status == EndpointStatus.DEGRADED and stamped

        await harness.probe()  # still failing: the first stamp is kept
        assert endpoint0.metadata.get("degraded_since") == stamped

        await harness.probe()  # recovered
        assert endpoint0.status == EndpointStatus.READY
        assert not endpoint0.metadata.get("degraded_since")

    async def test_pending_host_is_skipped(self, _patch_poller):
        """Endpoints without a concrete host are never probed and stay untouched."""
        poller = _ScriptedPoller({})
        harness = _JanitorProbeHarness(poller)
        _patch_poller["poller"] = poller
        registry = harness.registry
        registry.register(
            project_id="proj",
            model="m",
            job_id="job-pending",
            host="pending",
            port=0,
            backend="slurm-remote",
        )
        registry.mark_ready("job-pending")

        (d, r) = await harness.probe()
        assert (d, r) == (0, 0)
        assert registry.get_by_job_id("job-pending").status == EndpointStatus.READY
        assert poller.calls == []

    async def test_probe_disabled_is_noop(self, _patch_poller):
        poller = _ScriptedPoller({13000: [(False, False, [], "x")] * 3})
        harness = _JanitorProbeHarness(poller, enable_engine_probe=False)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-off", model="m")

        result = await harness.janitor._cleanup()  # full cycle with probe disabled
        assert result.engines_degraded == 0
        assert result.engines_recovered == 0
        assert poller.calls == []
        assert harness.registry.get_by_job_id("job-off").status == EndpointStatus.READY

    async def test_is_degraded_probe_counts_as_failure_with_detail(self, _patch_poller):
        """Reachable-but-unhealthy probes accumulate toward degradation."""
        port = 13110
        script = {
            port: [(False, True, [], "expected model absent")],
        }
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(poller, engine_probe_degrade_after=1)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-d", port=port)

        (d, _) = await harness.probe()
        assert d == 1
        endpoint0 = harness.registry.get_by_job_id("job-d")
        assert endpoint0.status == EndpointStatus.DEGRADED
        assert "expected model absent" in (endpoint0.metadata.get("readiness_error") or "")

    async def test_probe_timeout_is_failure_not_crash(self, _patch_poller):
        """TimeoutError from wait_for is a counted probe failure; the loop survives."""

        class _SlowPoller:
            def __init__(self):
                self.calls = 0

            async def check_ready_detail(self, host, port, **kwargs):
                self.calls += 1
                await asyncio.sleep(30)
                raise AssertionError("must be cancelled by wait_for bound")

        import asyncio

        poller = _SlowPoller()
        harness = _JanitorProbeHarness(poller, engine_probe_timeout_sec=1)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-t")

        (d1, _) = await harness.probe()
        (d2, _) = await harness.probe()
        (d3, _) = await harness.probe()
        assert (d1, d2, d3) == (0, 0, 1)
        assert poller.calls >= 1
        assert harness.registry.get_by_job_id("job-t").status == EndpointStatus.DEGRADED


class TestJanitorEngineProbeGrace:
    """Warm-up grace: a READY row younger than grace is never probed, so the
    probe can't degrade a still-loading engine (rivault bring-up ~12-20 min
    while the cycle cadence is 60 s)."""

    async def test_warmup_ready_endpoints_are_not_probed(self, _patch_poller):
        poller = _ScriptedPoller({13000: [(False, False, [], "loading")] * 3})
        harness = _JanitorProbeHarness(poller, engine_probe_grace_sec=3600)
        _patch_poller["poller"] = poller
        _register_ready(harness.registry, "job-warm")

        (d, r) = await harness.probe()
        assert (d, r) == (0, 0)
        assert poller.calls == []
        assert harness.registry.get_by_job_id("job-warm").status == EndpointStatus.READY

    async def test_degraded_rows_are_probed_regardless_of_age(self, _patch_poller):
        port = 13001
        script = {port: [(True, False, ["probe-model"], "")]}
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(poller, engine_probe_grace_sec=3600)
        _patch_poller["poller"] = poller
        registry = harness.registry
        _register_ready(registry, "job-d", port=port)
        registry.mark_degraded("job-d")

        (d, r) = await harness.probe()
        assert (d, r) == (0, 1)
        assert registry.get_by_job_id("job-d").status == EndpointStatus.READY

    async def test_post_grace_endpoints_are_probed(self, _patch_poller):
        """A READY row past the grace window accepts the probe normally."""
        from datetime import timedelta

        port = 13002
        script = {port: [(False, False, [], "want fail")] * 3}
        poller = _ScriptedPoller(script)
        harness = _JanitorProbeHarness(
            poller, engine_probe_grace_sec=3600, engine_probe_degrade_after=2
        )
        _patch_poller["poller"] = poller
        registry = harness.registry
        _register_ready(registry, "job-old", port=port)
        endpoint0 = registry.get_by_job_id("job-old")
        # simulate an endpoint 2h old by backdating its created_at
        endpoint0.created_at = endpoint0.created_at - timedelta(hours=2)

        await harness.probe()
        (d2, _) = await harness.probe()
        assert d2 == 1
        assert endpoint0.status == EndpointStatus.DEGRADED
        assert [c[1] for c in poller.calls] == [port, port]
