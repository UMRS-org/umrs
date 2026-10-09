"""One durable job runtime per process.

STATE JOBS-20261009-001: ``server_umrs.py`` starts the durable runtime, and
the HTTP sidecar's lifespan starts it again — every call built a NEW runtime
(own job store and instance id, scheduler, monitor, boot reconciliation). Two
schedulers dispatched from one database, and the second boot reconciliation
took the first's live attempts for stale ones and requeued them: an understand
job ran twice on the stand-alone stack, and its embed raced the first run.
"""

from __future__ import annotations

import threading

import pytest
from umrs.server import durable_job_runtime as djr


class _Runtime:
    built = 0

    def __init__(self, **_kw):
        _Runtime.built += 1
        self._shutdown_called = False

    def start(self, **_kw):
        return object()


@pytest.fixture
def fake_runtime(monkeypatch):
    _Runtime.built = 0
    monkeypatch.delenv("UMRS_DISABLE_DURABLE_RUNTIME", raising=False)
    monkeypatch.setattr(djr, "DurableJobRuntime", _Runtime)
    monkeypatch.setattr(djr, "build_durable_job_db_pool_from_env", lambda: object())
    monkeypatch.setattr("umrs.api.catalog_store.init_catalog_meta_store", lambda _p: None)
    monkeypatch.setattr(
        "umrs.orchestrator.detection.detect_backend", lambda: type("D", (), {"backend": "x"})()
    )
    monkeypatch.setattr(djr, "_global_runtime", None)
    return _Runtime


def test_a_second_start_reuses_the_running_runtime(fake_runtime):
    first = djr.start_durable_job_runtime_from_env()
    second = djr.start_durable_job_runtime_from_env()
    assert first is second and fake_runtime.built == 1


def test_concurrent_starts_build_one_runtime(fake_runtime):
    """The main thread and the sidecar thread start it at nearly the same time."""
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(djr.start_durable_job_runtime_from_env()))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake_runtime.built == 1 and len({id(r) for r in results}) == 1


def test_a_shut_down_runtime_is_replaced(fake_runtime):
    first = djr.start_durable_job_runtime_from_env()
    first._shutdown_called = True
    second = djr.start_durable_job_runtime_from_env()
    assert second is not first and fake_runtime.built == 2


def test_the_opt_out_still_starts_nothing(fake_runtime, monkeypatch):
    monkeypatch.setenv("UMRS_DISABLE_DURABLE_RUNTIME", "1")
    assert djr.start_durable_job_runtime_from_env() is None and fake_runtime.built == 0
