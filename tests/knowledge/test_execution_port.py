"""Tests for WorkflowExecutionPort and FakeWorkflowExecutionPort."""

from __future__ import annotations

import pytest
from umrs.knowledge.execution_port import (
    FakeWorkflowExecutionPort,
    WorkflowExecutionPort,
)


def test_fake_is_workflow_execution_port() -> None:
    port = FakeWorkflowExecutionPort()
    assert isinstance(port, WorkflowExecutionPort)


def test_fake_trigger_wait_results_include_durable_job_id() -> None:
    port = FakeWorkflowExecutionPort(durable_job_id="job-abc")
    execution_id = port.trigger("demo08_yield_eval", {"x": 1})
    assert execution_id.startswith("exec-fake-")

    waited = port.wait(execution_id)
    assert waited["durable_job_id"] == "job-abc"
    assert waited["execution_id"] == execution_id
    assert waited["status"] == "succeeded"

    results = port.results(execution_id)
    assert results["durable_job_id"] == "job-abc"
    assert results["outputs"]

    status = port.status(execution_id)
    assert status["durable_job_id"] == "job-abc"


def test_fake_records_calls() -> None:
    port = FakeWorkflowExecutionPort()
    eid = port.trigger("wf", {"a": 2})
    port.wait(eid, timeout=5.0)
    port.results(eid)
    port.cancel(eid)

    names = [c[0] for c in port.calls]
    assert names == ["trigger", "wait", "results", "cancel"]
    assert port.calls[0][1] == ("wf", {"a": 2})
    assert port.calls[1][2]["timeout"] == 5.0


def test_fake_require_durable_fails_when_missing() -> None:
    port = FakeWorkflowExecutionPort(durable_job_id="", require_durable=True)
    with pytest.raises(RuntimeError, match="durable job backend required"):
        port.trigger("wf", {})
