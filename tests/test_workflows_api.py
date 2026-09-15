"""Route-level tests for UMRS workflow HTTP routes."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.workflows import router
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
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
def tmp_catalog(tmp_path):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=catalog_file,
    ):
        yield catalog_file


def _workflow_json(name: str = "ct-analysis"):
    return {
        "name": name,
        "version": "1.0",
        "description": "workflow test",
        "source": {"type": "local_path", "path": "."},
        "steps": [
            {"name": "step-1", "entrypoint": "my_module:StepOne"},
            {"name": "step-2", "entrypoint": "my_module:StepTwo"},
        ],
    }


def _workflow_zip_bytes(name: str = "ct-analysis") -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", json.dumps(_workflow_json(name)))
        zf.writestr("steps.py", "class StepOne: pass\nclass StepTwo: pass\n")
    return buffer.getvalue()


def test_list_workflows_supports_filters(client, tmp_path):
    alice_dir = tmp_path / "alice-workflow"
    alice_dir.mkdir()
    (alice_dir / "workflow.json").write_text(json.dumps(_workflow_json("ct-analysis")))

    bob_dir = tmp_path / "bob-workflow"
    bob_dir.mkdir()
    (bob_dir / "workflow.json").write_text(json.dumps(_workflow_json("video-analysis")))

    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(alice_dir),
        workflow_json=_workflow_json("ct-analysis"),
        scope="user",
    )
    register_workflow(
        workflow_name="video-analysis",
        version="1.0",
        user_id="bob",
        source_path=str(bob_dir),
        workflow_json=_workflow_json("video-analysis"),
        scope="system",
    )

    response = client.get(
        "/umrs/workflows",
        params={"scope": "user", "name": "ct", "user_id": "alice"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["count"] == 1
    assert data["workflows"][0]["name"] == "ct-analysis"


def test_trigger_workflow_returns_acceptance_payload(client, tmp_path):
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(json.dumps(_workflow_json()))

    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=_workflow_json(),
        scope="user",
    )

    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        json={
            "user_id": "alice",
            "params": {"bucket": "demo-bucket", "project_id": "project-a"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["workflow_name"] == "ct-analysis"
    assert data["steps"] == 2
    assert data["params"]["bucket"] == "demo-bucket"


def test_trigger_workflow_uses_verified_identity_in_strict_mode(client, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    reset_execution_registry_for_tests()
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(json.dumps(_workflow_json()))

    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=_workflow_json(),
        scope="user",
    )

    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "project-a",
        },
        json={
            "user_id": "attacker",
            "params": {"project_id": "evil-project", "bucket": "demo-bucket"},
        },
    )

    assert response.status_code == 200
    data = response.json()
    record = get_execution_registry().get(data["execution_id"])
    assert record is not None
    assert record.user_id == "alice"
    assert record.params["project_id"] == "project-a"


def test_trigger_workflow_rejects_query_identity_in_strict_mode(client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        json={"user_id": "alice", "params": {"project_id": "project-a"}},
    )

    assert response.status_code == 401


def test_trigger_missing_workflow_returns_404(client):
    response = client.post(
        "/umrs/workflows/missing-workflow/trigger",
        json={"user_id": "alice", "params": {}},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_upload_workflow_registers_and_returns_accepted(client):
    response = client.post(
        "/umrs/workflows/upload",
        data={"user_id": "alice", "workflow_name": "ct-analysis"},
        files={"file": ("ct-analysis.zip", _workflow_zip_bytes(), "application/zip")},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "ok"
    assert data["accepted"] is True
    assert data["workflow_name"] == "ct-analysis"

    listed = client.get("/umrs/workflows", params={"user_id": "alice"})
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["workflows"][0]["name"] == "ct-analysis"


def test_upload_workflow_rejects_invalid_archive(client):
    response = client.post(
        "/umrs/workflows/upload",
        data={"user_id": "alice", "workflow_name": "broken-workflow"},
        files={"file": ("broken.zip", b"not-a-zip", "application/zip")},
    )

    assert response.status_code == 400
    assert "zip" in response.json()["detail"].lower()
