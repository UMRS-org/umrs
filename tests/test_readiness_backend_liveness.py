"""Live bdrt 2026-09-27: Slurm serve jobs were cancelled 83-136 s after start
by something outside UMRS, and the readiness poll never noticed. It probed
the dead host until ``2.0 ** attempt`` overflowed at attempt 1024 (~2.8 h),
and every probe error logged as a bare "Error: ".

Pinned here: the backoff exponent is capped, probe errors name their type,
and an optional liveness callback ends the wait FAILED once the backend job
is gone."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from umrs.orchestrator import readiness
from umrs.orchestrator.readiness import (
    BACKEND_JOB_ENDED,
    ReadinessConfig,
    ReadinessPoller,
    ReadinessState,
)


def _poller(**overrides) -> ReadinessPoller:
    return ReadinessPoller(ReadinessConfig(timeout_sec=3600, **overrides))


@pytest.mark.parametrize("attempt", [1024, 5000])
def test_backoff_delay_stays_finite_past_the_float_range(attempt):
    poller = _poller(max_delay_sec=5.0, jitter_factor=0.1)

    delay = poller._calculate_delay(attempt)

    assert 0.1 <= delay <= 5.0 * 1.1


async def test_probe_error_without_message_names_its_type(monkeypatch):
    class _Client:
        async def get(self, *_a, **_k):
            raise httpx.RemoteProtocolError("")  # str() is empty

    poller = _poller()

    async def client():
        return _Client()

    monkeypatch.setattr(poller, "_get_client", client)
    is_ready, is_degraded, models, error = await poller._check_once("127.0.0.1", 1)

    assert (is_ready, is_degraded, models) == (False, False, [])
    assert error == "RemoteProtocolError: "


def _never_ready(poller: ReadinessPoller, monkeypatch, probes: list[int]) -> None:
    async def not_ready(*_a, **_k):
        probes.append(1)
        return False, False, [], "connection refused"

    async def no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(poller, "_check_once", not_ready)
    monkeypatch.setattr(poller, "_sleep_before_next_attempt", no_sleep)


async def test_ended_backend_job_fails_the_wait(monkeypatch):
    poller = _poller(liveness_interval_sec=0.0)
    probes: list[int] = []
    _never_ready(poller, monkeypatch, probes)

    async def job_gone() -> bool:
        return False

    result = await asyncio.wait_for(
        poller.wait_for_ready("job-1", "127.0.0.1", 1, liveness=job_gone), 5
    )

    assert result.state == ReadinessState.FAILED
    assert result.error == BACKEND_JOB_ENDED
    assert probes == []  # no probe of a dead job


async def test_liveness_is_asked_once_per_interval(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(readiness.time, "monotonic", lambda: clock["now"])
    poller = _poller(liveness_interval_sec=60.0)
    probes: list[int] = []
    _never_ready(poller, monkeypatch, probes)

    async def ten_seconds_per_attempt(*_a, **_k):
        clock["now"] += 10.0

    monkeypatch.setattr(poller, "_sleep_before_next_attempt", ten_seconds_per_attempt)
    answers = iter([True, False])
    asked_at: list[float] = []

    async def liveness() -> bool:
        asked_at.append(clock["now"] - 1000.0)
        return next(answers)

    result = await asyncio.wait_for(
        poller.wait_for_ready("job-2", "127.0.0.1", 1, liveness=liveness), 5
    )

    assert asked_at == [60.0, 120.0]
    assert result.state == ReadinessState.FAILED
    assert len(probes) == 12


async def test_a_failing_liveness_check_keeps_the_wait_going(monkeypatch):
    poller = _poller(liveness_interval_sec=0.0)
    calls: list[int] = []

    async def ready_on_third(*_a, **_k):
        calls.append(1)
        return len(calls) >= 3, False, ["m"], ""

    async def no_sleep(*_a, **_k):
        return None

    async def backend_unreachable() -> bool:
        raise ConnectionError("ssh down")

    monkeypatch.setattr(poller, "_check_once", ready_on_third)
    monkeypatch.setattr(poller, "_sleep_before_next_attempt", no_sleep)

    result = await asyncio.wait_for(
        poller.wait_for_ready("job-3", "127.0.0.1", 1, liveness=backend_unreachable),
        5,
    )

    assert result.state == ReadinessState.READY
    assert len(calls) == 3
