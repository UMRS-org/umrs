"""Route-level tests for W5 gate approval + W4 triggers introspection."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.server.workflow_execution_registry import (
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_registry_handler import register_workflow


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _tmp_catalog(tmp_path, monkeypatch):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    monkeypatch.setattr(
        "umrs.server.workflow_registry_handler._catalog_path",
        lambda: catalog_file,
    )
    return catalog_file


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


def _headers(user: str = "alice", project: str = "proj-a") -> dict[str, str]:
    return {"X-User-ID": user, "X-Verified-Project-ID": project}


def _register(tmp_path: Path, name: str, workflow_json: dict) -> None:
    wf_dir = tmp_path / name
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "workflow.json").write_text(json.dumps(workflow_json))
    register_workflow(
        workflow_name=name,
        version=str(workflow_json.get("version", "1.0")),
        user_id="alice",
        source_path=str(wf_dir),
        workflow_json=workflow_json,
        scope="user",
        description="",
    )


def _gate_workflow(**extra) -> dict:
    wf = {
        "name": "gate-wf",
        "version": "1.0",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "review", "gate": True, "gate_timeout_seconds": 30}],
    }
    wf.update(extra)
    return wf


def _trigger(client: TestClient, name: str) -> str:
    resp = client.post(
        f"/umrs/workflows/{name}/trigger",
        json={"parameters": {}},
        headers=_headers(),
    )
    assert resp.status_code in (200, 202), resp.text
    data = resp.json()
    execution = data.get("execution") or data
    return execution["execution_id"]


def _wait_for_gate(client: TestClient, execution_id: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/umrs/workflows/executions/{execution_id}", headers=_headers())
        payload = resp.json()["execution"]
        for step in payload.get("steps", []):
            if step["name"] == "review" and step["status"] == "waiting_approval":
                return
        time.sleep(0.05)
    raise TimeoutError("gate step never reached waiting_approval")


# ── W5: approve route ───────────────────────────────────────────────────────


class TestApproveRoute:
    def test_full_approve_flow_completes_execution(self, client, tmp_path):
        _register(tmp_path, "gate-wf", _gate_workflow())
        execution_id = _trigger(client, "gate-wf")
        _wait_for_gate(client, execution_id)

        resp = client.post(
            f"/umrs/workflows/executions/{execution_id}/steps/review/approve",
            json={"approved": True, "approver": "alice", "comment": "lgtm"},
            headers=_headers(),
        )
        assert resp.status_code == 200, resp.text
        decision = resp.json()["execution"]["gates"]["review"]
        assert decision["approved"] is True
        assert decision["approver"] == "alice"

    def test_unknown_execution_returns_404(self, client):
        resp = client.post(
            "/umrs/workflows/executions/wf-nope/steps/review/approve",
            json={"approved": True},
            headers=_headers(),
        )
        assert resp.status_code == 404

    def test_unknown_step_returns_404(self, client, tmp_path):
        _register(tmp_path, "gate-wf", _gate_workflow())
        execution_id = _trigger(client, "gate-wf")
        _wait_for_gate(client, execution_id)

        resp = client.post(
            f"/umrs/workflows/executions/{execution_id}/steps/other/approve",
            json={"approved": True},
            headers=_headers(),
        )
        assert resp.status_code == 404

    def test_non_waiting_step_returns_409(self, client, tmp_path):
        _register(tmp_path, "gate-wf", _gate_workflow())
        execution_id = _trigger(client, "gate-wf")
        _wait_for_gate(client, execution_id)

        ok = client.post(
            f"/umrs/workflows/executions/{execution_id}/steps/review/approve",
            json={"approved": True},
            headers=_headers(),
        )
        assert ok.status_code == 200

        # Second decision on the same gate: conflict.
        resp = client.post(
            f"/umrs/workflows/executions/{execution_id}/steps/review/approve",
            json={"approved": False},
            headers=_headers(),
        )
        assert resp.status_code == 409

    def test_reject_flow_fails_execution(self, client, tmp_path):
        _register(tmp_path, "gate-wf", _gate_workflow())
        execution_id = _trigger(client, "gate-wf")
        _wait_for_gate(client, execution_id)

        resp = client.post(
            f"/umrs/workflows/executions/{execution_id}/steps/review/approve",
            json={"approved": False, "comment": "insufficient evidence"},
            headers=_headers(),
        )
        assert resp.status_code == 200
        decision = resp.json()["execution"]["gates"]["review"]
        assert decision["approved"] is False


# ── W4: triggers introspection ──────────────────────────────────────────────


class TestTriggersIntrospection:
    def test_declared_triggers_listed(self, client, tmp_path):
        wf = _gate_workflow(triggers=[{"on": "ingest", "bucket": "uploads", "prefix": "img/"}])
        _register(tmp_path, "gate-wf", wf)

        resp = client.get("/umrs/workflows/gate-wf/triggers", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["triggers"] == [{"on": "ingest", "bucket": "uploads", "prefix": "img/"}]

    def test_no_triggers_defaults_empty(self, client, tmp_path):
        _register(tmp_path, "gate-wf", _gate_workflow())
        resp = client.get("/umrs/workflows/gate-wf/triggers", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["triggers"] == []

    def test_unknown_workflow_404(self, client):
        resp = client.get("/umrs/workflows/nope/triggers", headers=_headers())
        assert resp.status_code == 404

    def test_describe_includes_triggers(self, client, tmp_path):
        wf = _gate_workflow(triggers=[{"on": "ingest"}])
        _register(tmp_path, "gate-wf", wf)
        resp = client.get("/umrs/workflows/gate-wf", headers=_headers())
        assert resp.status_code == 200
        assert resp.json()["triggers"] == [{"on": "ingest"}]
