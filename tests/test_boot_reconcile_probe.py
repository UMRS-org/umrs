"""The pre-bind boot reconcile probes each stuck endpoint briefly.

2026-10-22: it waited up to 300 s per endpoint, serially, before uvicorn
bound the HTTP port, so bdrt's API was down 15 min after a restart while its
endpoints answered 502. Live-but-loading rows are deferred and finished by
resume_unfinished_provisions on the server loop, so the boot pass needs only
a short probe.
"""

from __future__ import annotations

import pytest
from umrs.orchestrator import service as service_module
from umrs.server import http_sidecar


class _FakeService:
    def __init__(self) -> None:
        self.timeouts: list[int] = []

    async def reconcile_stale_endpoints(self, timeout_sec: int = 300) -> dict:
        self.timeouts.append(timeout_sec)
        return {"checked": 4, "promoted_to_ready": 0, "marked_failed": 0, "errors": []}


@pytest.fixture
def fake_service(monkeypatch: pytest.MonkeyPatch) -> _FakeService:
    fake = _FakeService()
    monkeypatch.setattr(service_module, "get_orchestration_service", lambda: fake)
    monkeypatch.delenv("UMRS_RECONCILE_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("UMRS_BOOT_RECONCILE_ENABLED", raising=False)
    return fake


def test_the_boot_pass_probes_briefly_by_default(fake_service: _FakeService) -> None:
    http_sidecar._run_boot_reconcile()
    assert fake_service.timeouts == [http_sidecar._BOOT_RECONCILE_PROBE_SEC]
    assert http_sidecar._BOOT_RECONCILE_PROBE_SEC <= 15


def test_the_env_still_sets_the_probe(
    fake_service: _FakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UMRS_RECONCILE_TIMEOUT_SEC", "42")
    http_sidecar._run_boot_reconcile()
    assert fake_service.timeouts == [42]


def test_a_disabled_boot_pass_calls_nothing(
    fake_service: _FakeService, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UMRS_BOOT_RECONCILE_ENABLED", "false")
    http_sidecar._run_boot_reconcile()
    assert fake_service.timeouts == []
