"""A second dispatch of a run still running here attaches to it (peer review 2026-09-26).

A scheduler requeue after a lost submit acknowledgement, or a boot reconcile,
re-dispatches the same execution id. Two live runs of one id would cancel each
other's step jobs on the cluster and race on the record.
"""

from __future__ import annotations

import threading

from umrs.server import workflow_registry_handler as wrh
from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry


def _running(registry: WorkflowExecutionRegistry, release: threading.Event):
    record = registry.create(workflow_name="w", user_id="alice", execution_id="wf-attach")
    started = threading.Event()

    def runner(rec):
        started.set()
        release.wait(10)
        rec.update(status="succeeded")

    registry.launch(record, runner)
    assert started.wait(5)
    return record


def test_running_here_follows_the_runner_thread():
    registry, release = WorkflowExecutionRegistry(), threading.Event()
    record = _running(registry, release)
    assert registry.running_here("wf-attach") is record
    release.set()
    assert registry.join("wf-attach", timeout=5)
    assert registry.running_here("wf-attach") is None
    assert registry.running_here("wf-unknown") is None


def test_a_second_dispatch_attaches_instead_of_launching(monkeypatch):
    registry, release = WorkflowExecutionRegistry(), threading.Event()
    record = _running(registry, release)
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    monkeypatch.setattr(wrh, "_workflow_run_context", lambda *_a, **_k: ("/src", {}, 1))
    launched = []
    monkeypatch.setattr(registry, "launch", lambda *args, **_k: launched.append(args))
    try:
        got, source, steps = wrh._launch_workflow_from_catalog(
            workflow_name="w", client_id="alice", params={}, execution_id="wf-attach"
        )
        assert got is record and (source, steps) == ("/src", 1)
        assert launched == []
        assert registry.get("wf-attach") is record  # no replacement record
    finally:
        release.set()
        registry.join("wf-attach", timeout=5)
