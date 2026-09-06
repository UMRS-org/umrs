"""UMRS WorkflowExecutionPort pins (real-mode REST loop composition).

- trigger creates a WORKFLOW_RUN durable job with loop params preserved
- wait blocks until a terminal registry status and carries durable_job_id
- results maps registry record into the engine's expected payload shape
- cancel is cooperative + best-effort
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort


class _FakeRecord:
    """Execution-registry stand-in with mutable status."""

    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self.status = "running"
        self.results = {"outputs": {"answer": 42}, "metrics": {"m": 1}}
        self.user_id = "research-loop"

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "results": self.results,
            "artifacts": [{"name": "out.bin"}],
            "error": "",
        }


class _FakeRegistry:
    def __init__(self) -> None:
        self.records: dict[str, _FakeRecord] = {}

    def get(self, execution_id: str) -> _FakeRecord | None:
        rec = self.records.get(execution_id)
        return rec

    def gain(self, execution_id: str) -> _FakeRecord:
        record = _FakeRecord(execution_id)
        self.records[execution_id] = record
        return record

    def cancel(self, execution_id: str) -> None:
        if execution_id in self.records:
            self.records[execution_id].status = "cancelled"


@pytest.fixture
def store() -> MagicMock:
    s = MagicMock()
    s.create_job.return_value = "job-port-1"
    s.get_job_with_attempts.return_value = (MagicMock(job_id="job-port-1"), [])
    return s


@pytest.fixture
def registry() -> _FakeRegistry:
    return _FakeRegistry()


def _patch_registry(registry: _FakeRegistry):
    return patch(
        "umrs.server.workflow_execution_registry.get_execution_registry",
        return_value=registry,
    )


def test_trigger_creates_workflow_run_job(store, registry):
    port = UmrsWorkflowExecutionPort(store)
    exec_id = port.trigger(
        "demo08_yield_eval", {"a": 1, "project_id": "proj-x", "user_id": "alice"}
    )
    store.create_job.assert_called_once()
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["project_id"] == "proj-x"
    assert kwargs["spec"]["workflow_name"] == "demo08_yield_eval"
    assert kwargs["spec"]["execution_id"] == exec_id
    assert kwargs["max_attempts"] == 1
    # user_id consumed as identity, not forwarded inside params
    assert "user_id" not in kwargs["spec"]["params"]


def test_wait_returns_terminal_with_durable_job_id(store, registry):
    port = UmrsWorkflowExecutionPort(store, poll_interval=0.01)
    exec_id = port.trigger("wf", {"project_id": "p"})
    record = registry.gain(exec_id)

    def complete_later():
        time.sleep(0.05)
        record.status = "succeeded"

    threading.Thread(target=complete_later, daemon=True).start()
    with _patch_registry(registry):
        data = port.wait(exec_id, timeout=2.0)
    assert data["status"] == "succeeded"
    assert data["durable_job_id"] == "job-port-1"


def test_wait_timeout_is_honest(store, registry):
    port = UmrsWorkflowExecutionPort(store, poll_interval=0.01)
    exec_id = port.trigger("wf", {"project_id": "p"})
    registry.gain(exec_id)  # stays running
    with _patch_registry(registry):
        data = port.wait(exec_id, timeout=0.05)
    assert data["status"] == "timeout"
    assert data["durable_job_id"] == "job-port-1"


def test_results_maps_engine_payload_shape(store, registry):
    port = UmrsWorkflowExecutionPort(store)
    exec_id = port.trigger("wf", {"project_id": "p"})
    record = registry.gain(exec_id)
    record.status = "succeeded"
    with _patch_registry(registry):
        results = port.results(exec_id)
    assert results["status"] == "succeeded"
    assert results["outputs"] == {"answer": 42}
    assert results["metrics"] == {"m": 1}
    assert results["artifacts"] == [{"name": "out.bin"}]
    assert results["durable_job_id"] == "job-port-1"


def test_cancel_cooperates_with_registry_and_store(store, registry):
    from types import SimpleNamespace

    port = UmrsWorkflowExecutionPort(store)
    exec_id = port.trigger("wf", {"project_id": "p"})
    registry.gain(exec_id)
    # cancel_workflow_execution_for_job reads execution_id from job.spec —
    # SimpleNamespace: Mock treats ``spec`` as a reserved construction param.
    job = SimpleNamespace(job_id="job-port-1", spec={"execution_id": exec_id})
    store.get_job_with_attempts.return_value = (job, [])
    with _patch_registry(registry):
        outcome = port.cancel(exec_id)
    assert outcome["status"] == "cancelled"
    assert registry.records[exec_id].status == "cancelled"
    store.cancel_job.assert_called_once_with("job-port-1")


def test_results_unknown_execution(store):
    port = UmrsWorkflowExecutionPort(store)
    with _patch_registry(_FakeRegistry()):
        assert port.results("exec-ghost")["status"] == "unknown"


def test_trigger_enqueues_the_created_job():
    """WP-07 live finding: create_job leaves PLANNED; the scheduler claims QUEUED only.

    Without the enqueue every server-driven real-mode loop blocked in wait().
    """
    from unittest.mock import MagicMock

    from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort

    store = MagicMock()
    store.create_job.return_value = "job-1"
    port = UmrsWorkflowExecutionPort(store)
    execution_id = port.trigger("demo08_yield_eval", {"project_id": "p", "iteration": 1})
    assert execution_id.startswith("exec-")
    store.create_job.assert_called_once()
    store.enqueue_job.assert_called_once_with("job-1")
