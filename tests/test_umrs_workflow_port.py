"""UMRS WorkflowExecutionPort pins (real-mode REST loop composition).

- trigger creates a WORKFLOW_RUN durable job with loop params preserved, for
  the loop's verified owner with the workflow resolved owner-scoped (ADR-074
  D-8, review round 3): a planner-written ``user_id`` param names nobody
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


#: The catalog the port resolves against: (owner, name) -> source_path.
CATALOG = {
    ("alice", "demo08_yield_eval"): "/wf/alice/demo08_yield_eval",
    ("bob", "private"): "/wf/bob/private",
    ("*", "wf"): "/wf/system/wf",
    ("*", "demo08_yield_eval"): "/wf/system/demo08_yield_eval",
}


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    """Owner-scoped resolution (the real rule: the caller's own entry, else a
    system one; never another owner's)."""
    from umrs.server import workflow_registry_handler as handler

    def resolve(name, owner, *, is_admin=False):
        path = CATALOG.get((owner, name)) or CATALOG.get(("*", name))
        return {"source_path": path} if path else None

    monkeypatch.setattr(handler, "resolve_workflow_for_caller", resolve)


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
    exec_id = port.trigger_as("demo08_yield_eval", {"a": 1, "project_id": "proj-x"}, owner="alice")
    store.create_job.assert_called_once()
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["project_id"] == "proj-x"
    assert kwargs["spec"]["workflow_name"] == "demo08_yield_eval"
    assert kwargs["spec"]["execution_id"] == exec_id
    # the owner's own workflow, resolved now: the backend never re-resolves
    assert kwargs["spec"]["source_path"] == "/wf/alice/demo08_yield_eval"
    assert kwargs["max_attempts"] == 1


def test_a_planner_user_id_param_names_nobody(store):
    # Experiment parameters come from the LLM planner (steerable through the
    # research goal): a user_id there must not run another owner's code.
    port = UmrsWorkflowExecutionPort(store)
    port.trigger_as("wf", {"project_id": "p", "user_id": "bob"}, owner="alice")
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert "user_id" not in kwargs["spec"]["params"]
    port.trigger("wf", {"project_id": "p", "user_id": "bob"})
    assert store.create_job.call_args.kwargs["user_id"] == "research-loop"


def test_another_owners_workflow_fails_by_name_before_any_job(store):
    port = UmrsWorkflowExecutionPort(store)
    with pytest.raises(RuntimeError, match="'private' is not one the loop's owner 'alice'"):
        port.trigger_as("private", {"project_id": "p", "user_id": "bob"}, owner="alice")
    with pytest.raises(RuntimeError, match="ADR-074 D-8"):
        port.trigger("private", {"project_id": "p"})  # no verified owner: system only
    store.create_job.assert_not_called()


def test_the_engine_runs_experiments_as_the_loops_owner(store):
    from types import SimpleNamespace

    from umrs.knowledge.loop import AutonomousResearchLoop

    port = UmrsWorkflowExecutionPort(store)
    engine = AutonomousResearchLoop(workflow_port=port)
    loop = SimpleNamespace(metadata={"owner_user_id": "alice"})
    engine._trigger_workflow(loop, "demo08_yield_eval", {"project_id": "p"})
    assert store.create_job.call_args.kwargs["user_id"] == "alice"


def test_the_engine_keeps_a_client_ports_plain_trigger():
    from types import SimpleNamespace

    from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
    from umrs.knowledge.loop import AutonomousResearchLoop

    port = FakeWorkflowExecutionPort()
    engine = AutonomousResearchLoop(workflow_port=port)
    loop = SimpleNamespace(metadata={"owner_user_id": "alice"})
    engine._trigger_workflow(loop, "wf", {"project_id": "p"})
    assert port.calls[0][0] == "trigger"


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
    from umrs.jobs.models import JobType

    # cancel_workflow_execution_for_job reads execution_id from job.spec —
    # SimpleNamespace: Mock treats ``spec`` as a reserved construction param.
    # The port's jobs are workflow_run jobs (the registry cancel keys on it).
    job = SimpleNamespace(
        job_id="job-port-1", spec={"execution_id": exec_id}, job_type=JobType.WORKFLOW_RUN
    )
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


def test_an_experiment_naming_another_user_still_runs_as_the_loops_owner(store):
    from umrs.knowledge.loop import (
        AutonomousResearchLoop,
        LoopIteration,
        LoopPhase,
        LoopStatus,
        ResearchLoop,
    )
    from umrs.knowledge.planner import ResearchExperiment

    engine = AutonomousResearchLoop(simulated=False, workflow_port=UmrsWorkflowExecutionPort(store))
    engine._cancel_check = lambda: True  # noqa: SLF001 — stop right after the trigger
    loop = ResearchLoop(
        loop_id="loop-owner", project_id="p", research_goal="g", status=LoopStatus.RUNNING,
        phase=LoopPhase.EXPERIMENT_EXECUTION, max_iterations=1, current_iteration=1,
        metadata={"mode": "auto", "workflow_name": "demo08_yield_eval",
                  "owner_user_id": "alice"},
    )  # fmt: skip
    experiment = ResearchExperiment(experiment_id="e1", parameters={"user_id": "bob", "n": 2})
    iteration = LoopIteration(iteration_id="it", loop_id=loop.loop_id, iteration_number=1)
    engine._execute_experiment(loop, iteration, experiment)  # noqa: SLF001
    kwargs = store.create_job.call_args.kwargs
    assert kwargs["user_id"] == "alice"
    assert kwargs["spec"]["source_path"] == "/wf/alice/demo08_yield_eval"
    assert kwargs["spec"]["params"]["n"] == 2 and "user_id" not in kwargs["spec"]["params"]
