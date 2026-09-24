"""Live rccs 2026-09-24: `r4 endpoint down` deprovisioned job 387426 while a
readiness poll kept probing it for 33+ minutes (its own timeout was the only
exit). Backend cleanup now cancels the poll; it stops at the next attempt."""

from __future__ import annotations

import asyncio

from umrs.orchestrator import readiness
from umrs.orchestrator.readiness import (
    ReadinessConfig,
    ReadinessPoller,
    ReadinessState,
    cancel_readiness,
)


async def test_cleanup_cancels_an_in_flight_poll(monkeypatch):
    poller = ReadinessPoller(ReadinessConfig(timeout_sec=3600))
    calls = []

    async def never_ready(*_a, **_k):
        calls.append(1)
        if len(calls) == 2:
            cancel_readiness("job-1")  # the endpoint is torn down mid-wait
        return False, False, [], "connection refused"

    async def no_sleep(*_a, **_k):
        return None

    monkeypatch.setattr(poller, "_check_once", never_ready)
    monkeypatch.setattr(poller, "_sleep_before_next_attempt", no_sleep)
    result = await asyncio.wait_for(poller.wait_for_ready("job-1", "127.0.0.1", 1), 5)
    assert result.state == ReadinessState.CANCELLED
    assert len(calls) == 2
    assert "job-1" not in readiness._CANCELLED_JOBS  # consumed, not sticky


async def test_other_jobs_keep_polling(monkeypatch):
    cancel_readiness("job-other")
    poller = ReadinessPoller(ReadinessConfig(timeout_sec=3600))

    async def ready(*_a, **_k):
        return True, False, ["m"], ""

    monkeypatch.setattr(poller, "_check_once", ready)
    result = await poller.wait_for_ready("job-2", "127.0.0.1", 1)
    assert result.state == ReadinessState.READY
    readiness._CANCELLED_JOBS.discard("job-other")
