"""The sidecar lifespan stops the durable runtime it started; a stopped
runtime leaves the process registry.

Before 2026-11-24 nothing stopped it: every app lifespan (each
``TestClient(app)`` in the suite) left a live runtime registered in
``durable_job_runtime._global_runtime``. A later loop start in the same xdist
worker was then job-hosted on it, never thread-hosted, and
``test_plan_loop_server::TestTheStartRoute`` saw no driver registered
(UMRS-org main run 45280814153). A runtime shut down by its owner also stayed
registered, so ``get_runtime()`` kept handing out a stopped runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from umrs.server import durable_job_runtime as djr
from umrs.server.http_sidecar import _create_app


class _Runtime:
    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1
        djr._clear_runtime_if(self)


@pytest.fixture
def no_runtime(monkeypatch):
    monkeypatch.setattr(djr, "_global_runtime", None)


def _starting(runtime):
    def start():
        djr._set_runtime(runtime)
        return runtime

    return start


def _boot_and_exit() -> None:
    with TestClient(_create_app()):
        pass


def test_the_lifespan_stops_the_runtime_it_started(monkeypatch, no_runtime):
    runtime = _Runtime()
    monkeypatch.setattr(djr, "start_durable_job_runtime_from_env", _starting(runtime))

    _boot_and_exit()

    assert runtime.shutdowns == 1
    assert djr.get_runtime() is None


def test_the_lifespan_leaves_a_reused_runtime_to_its_owner(monkeypatch, no_runtime):
    owner = _Runtime()  # server_umrs.py started it first; it stops it too
    djr._set_runtime(owner)
    monkeypatch.setattr(djr, "start_durable_job_runtime_from_env", lambda: owner)

    _boot_and_exit()

    assert owner.shutdowns == 0
    assert djr.get_runtime() is owner


def test_a_shut_down_runtime_is_no_longer_the_process_runtime(no_runtime):
    service = SimpleNamespace(job_service=SimpleNamespace(job_store=None))
    runtime = djr.DurableJobRuntime(None, service=service)
    djr._set_runtime(runtime)

    runtime.shutdown()

    assert djr.get_runtime() is None


def test_shutting_down_another_runtime_keeps_the_registered_one(no_runtime):
    service = SimpleNamespace(job_service=SimpleNamespace(job_store=None))
    registered = djr.DurableJobRuntime(None, service=service)
    other = djr.DurableJobRuntime(None, service=service)
    djr._set_runtime(registered)

    other.shutdown()

    assert djr.get_runtime() is registered
