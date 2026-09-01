"""WORKFLOW_EXECUTIONS — project-scoped execution id listing.

ACL parity with WORKFLOW_STATUS (can_access_execution):
- records whose provenance is ``unverified`` are invisible (fail-closed),
  even to their own owner;
- records carrying a project are visible only to callers claiming the same
  project in the payload (`project_id`);
- everyone is pinned to their own owner scope on the ZMQ rail (no admin
  promotion over ZMQ today).
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from umrs.server.workflow_execution_registry import (
    ExecutionRecord,
    WorkflowExecutionRegistry,
)
from umrs.server import workflow_registry_handler as handler


@pytest.fixture(autouse=True)
def _legacy_auth_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-strict mode: the deployed umrs CLI path runs against non-strict
    catalogs, matching its environment; strict-mode provenance stays tested
    in the auth module itself."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "legacy")


def _project_record(
    name: str,
    *,
    user_id: str = "alice",
    project_id: str = "proj1",
    status: str = "running",
) -> ExecutionRecord:
    rec = ExecutionRecord(
        execution_id=name,
        workflow_name="wf-echo",
        user_id=user_id,
        params={"project_id": project_id},
    )
    rec.status = status
    rec.identity_provenance = "mvp_claim"
    return rec


@pytest.fixture(autouse=True)
def _seed() -> list[ExecutionRecord]:
    alice_running = _project_record("wf-a-running")
    alice_running.set_step("ingest", status="succeeded")
    alice_running.set_step("train", status="running")
    alice_running.progress = 0.5

    alice_done = _project_record("wf-a-done", status="succeeded")
    alice_done.finished_at = 1.0

    alice_other = _project_record("wf-a-proj2", project_id="proj2")

    unverified = _project_record("wf-a-unverified")
    unverified.identity_provenance = "unverified"

    bob = _project_record("wf-b-1", user_id="bob")

    return [alice_running, alice_done, alice_other, unverified, bob]


def _list(caller: str, payload: dict, records: list[ExecutionRecord]) -> dict:
    reg = WorkflowExecutionRegistry()
    for record in records:
        reg._records[record.execution_id] = record
    body = json.dumps(payload) if payload else ""
    with mock.patch(
        "umrs.server.workflow_execution_registry.get_execution_registry",
        return_value=reg,
    ):
        return json.loads(handler._handle_workflow_executions(caller, body))


def test_lists_own_project_scoped_records_with_step_counts(_seed):
    resp = _list("alice", {"project_id": "proj1"}, _seed)
    assert resp["status"] == "ok"
    ids = [row["execution_id"] for row in resp["executions"]]
    # wf-a-proj2 (wrong project), wf-a-unverified (fail-closed), wf-b-1 (other
    # owner) are all hidden.
    assert sorted(ids) == ["wf-a-done", "wf-a-running"]
    running = next(r for r in resp["executions"] if r["execution_id"] == "wf-a-running")
    assert running["steps_done"] == 1
    assert running["steps_total"] == 2
    assert running["user_id"] == "alice"
    assert running["workflow_name"] == "wf-echo"


def test_project_visibility_requires_matching_project_claim(_seed):
    resp = _list("alice", {"project_id": "proj2"}, _seed)
    ids = [row["execution_id"] for row in resp["executions"]]
    assert ids == ["wf-a-proj2"]


def test_unverified_provenance_is_invisible_even_to_owner(_seed):
    resp = _list("alice", {"project_id": "proj1"}, _seed)
    assert all(r["execution_id"] != "wf-a-unverified" for r in resp["executions"])


def test_other_owners_records_never_listed(_seed):
    resp = _list("alice", {"project_id": "proj1"}, _seed)
    assert all(r["user_id"] == "alice" for r in resp["executions"])
    resp2 = _list("bob", {"project_id": "proj1"}, _seed)
    assert [r["execution_id"] for r in resp2["executions"]] == ["wf-b-1"]


def test_status_and_limit_filters_honored(_seed):
    resp = _list("alice", {"project_id": "proj1", "status": "succeeded"}, _seed)
    assert [r["execution_id"] for r in resp["executions"]] == ["wf-a-done"]
    capped = _list("alice", {"project_id": "proj1", "limit": 1}, _seed)
    assert capped["count"] == 1


def test_invalid_json_returns_error():
    with mock.patch(
        "umrs.server.workflow_execution_registry.get_execution_registry",
        return_value=WorkflowExecutionRegistry(),
    ):
        resp = json.loads(handler._handle_workflow_executions("alice", "{not-json"))
    assert resp["status"] == "error"
