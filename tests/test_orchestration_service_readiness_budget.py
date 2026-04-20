from types import SimpleNamespace

import pytest

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