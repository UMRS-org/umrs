"""W5 tests — manual approval gates (`gate: true` steps) end to end.

Covers:

- ExecutionRecord gate mechanics (prepare/resolve/audit/to_dict)
- Registry ``resolve_approval`` forward
- Runner gate behavior: approve → succeed, reject → fail,
  deadline → fail, cancel-during-wait → execution cancelled
- StepSpec ``gate`` field round-trip + entrypoint waiver
- WorkflowDefinition ``triggers`` round-trip (W4 field parity)
"""

from __future__ import annotations

import threading
import time

import pytest
from umrs.pipeline_sdk.workflow import StepSpec, WorkflowDefinition
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


def _gate_workflow(timeout: float | None = None) -> dict:
    step = {"name": "gate1", "gate": True}
    if timeout is not None:
        step["gate_timeout_seconds"] = timeout
    return {"name": "gate-wf", "version": "0.1", "steps": [step]}


def _run_in_thread(record, tmp_path, workflow_json):
    thread = threading.Thread(
        target=run_workflow_record,
        kwargs={
            "record": record,
            "source_path": str(tmp_path),
            "workflow_json": workflow_json,
        },
        daemon=True,
    )
    thread.start()
    return thread


def _wait_step_status(record, step_name, wanted, timeout=15.0):
    """Wait until a step reaches ``wanted`` status (suite-load tolerant).

    Under full-suite load the runner thread can take >5s to reach the
    gate park (thread scheduling + one-off pool resolution ahead of the
    wave); the semantic assertion is that the gate EVENTUALLY parks,
    not how quickly.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        step = record.steps.get(step_name)
        if step is not None and step.status == wanted:
            return step
        time.sleep(0.02)
    raise TimeoutError(f"step {step_name} never reached {wanted}")


def _wait_done(thread, timeout=5.0):
    thread.join(timeout)
    assert not thread.is_alive(), "runner thread should have finished"


# ── Record mechanics ───────────────────────────────────────────────────────


class TestRecordGates:
    def test_resolve_gate_records_decision_and_audit(self):
        registry = get_execution_registry()
        record = registry.create(workflow_name="w", user_id="alice")
        event = record.prepare_gate("gate1")
        assert not event.is_set()

        record.resolve_gate("gate1", approved=True, approver="bob", comment="ok")

        assert event.is_set()
        decision = record.gate_decisions["gate1"]
        assert decision["approved"] is True
        assert decision["approver"] == "bob"
        audit = record.results["gate_audit"]
        assert len(audit) == 1
        assert audit[0]["step"] == "gate1"

    def test_first_decision_wins(self):
        registry = get_execution_registry()
        record = registry.create(workflow_name="w", user_id="alice")
        record.resolve_gate("gate1", approved=True, approver="bob", comment=None)
        record.resolve_gate("gate1", approved=False, approver="eve", comment=None)
        decision = record.gate_decisions["gate1"]
        assert decision["approved"] is True
        assert decision["approver"] == "bob"

    def test_to_dict_includes_gates(self):
        registry = get_execution_registry()
        record = registry.create(workflow_name="w", user_id="alice")
        record.resolve_gate("gate1", approved=False, approver="bob", comment="no")
        payload = record.to_dict()
        assert payload["gates"]["gate1"]["approved"] is False

    def test_registry_resolve_approval_forwards(self):
        registry = get_execution_registry()
        record = registry.create(workflow_name="w", user_id="alice")
        record.prepare_gate("gate1")
        updated = registry.resolve_approval(
            record.execution_id,
            "gate1",
            approved=True,
            approver="alice",
            comment="go",
        )
        assert updated is record
        assert record._gate_events["gate1"].is_set()
        assert record.gate_decisions["gate1"]["approved"] is True

    def test_registry_resolve_approval_unknown_execution(self):
        registry = get_execution_registry()
        assert (
            registry.resolve_approval(
                "wf-nope", "gate1", approved=True, approver=None, comment=None
            )
            is None
        )


# ── Runner gate behavior ───────────────────────────────────────────────────


class TestRunnerGates:
    def test_approve_completes_execution(self, tmp_path):
        registry = get_execution_registry()
        record = registry.create(workflow_name="gate-wf", user_id="alice")
        thread = _run_in_thread(record, tmp_path, _gate_workflow())

        _wait_step_status(record, "gate1", "waiting_approval")
        registry.resolve_approval(
            record.execution_id,
            "gate1",
            approved=True,
            approver="alice",
            comment="looks good",
        )
        _wait_done(thread)

        assert record.status == "succeeded", record.error
        step = record.steps["gate1"]
        assert step.status == "succeeded"
        assert step.extra["gate"]["approver"] == "alice"
        assert step.extra["waited_seconds"] >= 0.0
        audit = record.results["gate_audit"]
        assert audit[0]["comment"] == "looks good"

    def test_reject_fails_execution_with_reason(self, tmp_path):
        registry = get_execution_registry()
        record = registry.create(workflow_name="gate-wf", user_id="alice")
        thread = _run_in_thread(record, tmp_path, _gate_workflow())

        _wait_step_status(record, "gate1", "waiting_approval")
        registry.resolve_approval(
            record.execution_id,
            "gate1",
            approved=False,
            approver="security-bot",
            comment="missing review",
        )
        _wait_done(thread)

        assert record.status == "failed"
        assert "rejected by security-bot: missing review" in (record.error or "")

    def test_deadline_expiry_fails_execution(self, tmp_path):
        registry = get_execution_registry()
        record = registry.create(workflow_name="gate-wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_gate_workflow(timeout=0.2),
        )
        assert record.status == "failed"
        assert "approval deadline" in (record.error or "")
        step = record.steps["gate1"]
        assert step.status == "failed"
        # Load-insensitive deadline proof: measure the step's OWN lifecycle
        # window (runner-time), never test-process wall time — suite load adds
        # startup costs (pool resolution, imports) to wall time without
        # touching the gate deadline at all.
        assert 0.0 <= step.finished_at - step.started_at < 2.0

    def test_cancel_during_wait_marks_execution_cancelled(self, tmp_path):
        registry = get_execution_registry()
        record = registry.create(workflow_name="gate-wf", user_id="alice")
        thread = _run_in_thread(record, tmp_path, _gate_workflow())

        _wait_step_status(record, "gate1", "waiting_approval")
        registry.cancel(record.execution_id)
        _wait_done(thread)

        assert record.status == "cancelled"


# ── Model parity ────────────────────────────────────────────────────────────


class TestGateModel:
    def test_gate_stepspec_roundtrip(self):
        spec = StepSpec(name="review", gate=True)
        d = spec.to_dict()
        assert d["gate"] is True
        assert StepSpec.from_dict(d).gate is True
        spec.validate()  # no entrypoint required for gates

    def test_non_gate_requires_entrypoint(self):
        spec = StepSpec(name="run")
        with pytest.raises(ValueError, match="entrypoint"):
            spec.validate()

    def test_gate_field_omitted_when_false(self):
        spec = StepSpec(name="run", entrypoint="pkg.mod:Cls")
        assert "gate" not in spec.to_dict()

    def test_workflow_definition_triggers_roundtrip(self):
        definition = {
            "name": "wf",
            "source": {"type": "local_path", "path": "/tmp/x"},
            "steps": [{"name": "a", "entrypoint": "pkg.mod:Cls"}],
            "triggers": [{"on": "ingest", "bucket": "uploads"}],
        }
        wf = WorkflowDefinition.from_dict(definition)
        assert wf.triggers == [{"on": "ingest", "bucket": "uploads"}]
        assert wf.to_dict()["triggers"] == definition["triggers"]

    def test_workflow_definition_triggers_omitted_by_default(self):
        definition = {
            "name": "wf",
            "source": {"type": "local_path", "path": "/tmp/x"},
            "steps": [{"name": "a", "entrypoint": "pkg.mod:Cls"}],
        }
        wf = WorkflowDefinition.from_dict(definition)
        assert "triggers" not in wf.to_dict()
