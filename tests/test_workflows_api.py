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
def _handshake_secret(monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
    yield


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=catalog_file,
    ):
        yield catalog_file


def _workflow_json(name: str = "ct-analysis", *, with_required_bucket: bool = False):
    payload = {
        "name": name,
        "version": "1.0",
        "description": "workflow test",
        "source": {"type": "local_path", "path": "."},
        "steps": [
            {"name": "step-1", "entrypoint": "my_module:StepOne"},
            {"name": "step-2", "entrypoint": "my_module:StepTwo"},
        ],
    }
    if with_required_bucket:
        payload["parameter_schema"] = {
            "type": "object",
            "properties": {"bucket": {"type": "string"}},
            "required": ["bucket"],
        }
    return payload


def _workflow_zip_bytes(name: str = "ct-analysis") -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("workflow.json", json.dumps(_workflow_json(name)))
        zf.writestr("steps.py", "class StepOne: pass\nclass StepTwo: pass\n")
    return buffer.getvalue()


def _auth_headers(
    user: str = "alice", project: str = "project-a", admin: bool = False
) -> dict[str, str]:
    """Gateway-equivalent signed identity headers (wave-6 F1)."""
    import os

    from r4_contracts.handshake_hmac import sign_claims

    secret = os.environ.get("UMRS_HANDSHAKE_HMAC_SECRET", "")
    headers = {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
    }
    if secret:
        headers["X-Verified-Is-Admin"] = "true" if admin else "false"
        headers["X-Verified-HMAC"] = sign_claims(secret, user, project, admin)
    return headers


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
        headers=_auth_headers("alice"),
        params={"scope": "user", "name": "ct"},
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
        headers=_auth_headers("alice", "project-a"),
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

    # Wave-6 F1: unsigned (un-HMAC-signed) verified headers are now rejected
    # at the identity gate BEFORE project authorization — 401 strictly
    # dominates the old 403. The 403 contract for signed callers whose
    # declared project falls outside verified scope is covered by
    # test_declared_project_rejected_outside_memberships.
    assert response.status_code == 401


def test_trigger_workflow_rejects_query_identity_in_strict_mode(client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        json={"user_id": "alice", "params": {"project_id": "project-a"}},
    )

    assert response.status_code == 401


# ── P0-1: workflow-declared project_id authority matrix ──────────────────────


def _register_ct(tmp_path):
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


def test_declared_project_wins_for_admin(client, tmp_path, monkeypatch):
    """Admin caller + declared project-B ≠ verified default → project-B lands.

    Wave-6 note: memberships/admin-declared project authority without HMAC
    signatures is an mvp-mode contract; strict mode requires the signed
    HMAC tuple (see test_workflows_hmac_gate.py).
    """
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    _register_ct(tmp_path)
    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "default",
            "X-Verified-Is-Admin": "true",
            "X-Verified-Project-Memberships": "",
        },
        json={"user_id": "alice", "params": {"project_id": "project-b", "bucket": "b"}},
    )
    assert response.status_code == 200
    record = get_execution_registry().get(response.json()["execution_id"])
    assert record.params["project_id"] == "project-b"


def test_declared_project_wins_within_memberships(client, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    _register_ct(tmp_path)
    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "default",
            "X-Verified-Is-Admin": "false",
            "X-Verified-Project-Memberships": "project-a,project-b",
        },
        json={"user_id": "alice", "params": {"project_id": "project-b", "bucket": "b"}},
    )
    assert response.status_code == 200
    record = get_execution_registry().get(response.json()["execution_id"])
    assert record.params["project_id"] == "project-b"


def test_declared_project_rejected_outside_memberships(client, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    _register_ct(tmp_path)
    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "default",
            "X-Verified-Is-Admin": "false",
            "X-Verified-Project-Memberships": "project-a",
        },
        json={"user_id": "alice", "params": {"project_id": "project-x", "bucket": "b"}},
    )
    assert response.status_code == 403
    assert "project-x" in response.json()["detail"]


def test_verified_project_wins_when_nothing_declared(client, tmp_path):
    _register_ct(tmp_path)
    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers=_auth_headers("alice", "project-a"),
        json={"user_id": "alice", "params": {"bucket": "demo-bucket"}},
    )
    assert response.status_code == 200
    record = get_execution_registry().get(response.json()["execution_id"])
    assert record.params["project_id"] == "project-a"


def test_trigger_missing_workflow_returns_404(client):
    response = client.post(
        "/umrs/workflows/missing-workflow/trigger",
        headers=_auth_headers("alice", "project-a"),
        json={"user_id": "alice", "params": {}},
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_upload_workflow_registers_and_returns_accepted(client):
    # ADR-027 role gate: uploads require workflow_author claim or admin.
    headers = _auth_headers("alice", "project-a", admin=True)
    response = client.post(
        "/umrs/workflows/upload",
        headers=headers,
        data={"user_id": "alice", "workflow_name": "ct-analysis"},
        files={"file": ("ct-analysis.zip", _workflow_zip_bytes(), "application/zip")},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "ok"
    assert data["accepted"] is True
    assert data["workflow_name"] == "ct-analysis"

    listed = client.get("/umrs/workflows", headers=headers)
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["workflows"][0]["name"] == "ct-analysis"


def test_upload_workflow_rejects_invalid_archive(client):
    # ADR-027 role gate: uploads require workflow_author claim or admin.
    response = client.post(
        "/umrs/workflows/upload",
        headers=_auth_headers("alice", "project-a", admin=True),
        data={"user_id": "alice", "workflow_name": "broken-workflow"},
        files={"file": ("broken.zip", b"not-a-zip", "application/zip")},
    )

    assert response.status_code == 400
    assert "zip" in response.json()["detail"].lower()


def test_describe_workflow_returns_parameter_schema(client, tmp_path):
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    workflow_json = _workflow_json(with_required_bucket=True)
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow_json))

    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=workflow_json,
        scope="user",
    )

    response = client.get(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("alice", "project-a"),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["workflow_name"] == "ct-analysis"
    assert data["parameter_schema"]["required"] == ["bucket"]


def test_trigger_workflow_rejects_missing_required_params(client, tmp_path):
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    workflow_json = _workflow_json(with_required_bucket=True)
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow_json))

    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=workflow_json,
        scope="user",
    )

    response = client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers=_auth_headers("alice", "project-a"),
        json={"user_id": "alice", "params": {}},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["code"] == "workflow_parameters_invalid"
    assert "bucket" in detail["missing"]


def _register(client_workflow_json, tmp_path, *, name, user, scope="user"):
    workflow_dir = tmp_path / f"{user}-{name}"
    workflow_dir.mkdir(exist_ok=True)
    (workflow_dir / "workflow.json").write_text(json.dumps(client_workflow_json))
    register_workflow(
        workflow_name=name,
        version="1.0",
        user_id=user,
        source_path=str(workflow_dir),
        workflow_json=client_workflow_json,
        scope=scope,
    )


def test_delete_workflow_owner_can_delete(client, tmp_path):
    _register(_workflow_json("ct-analysis"), tmp_path, name="ct-analysis", user="alice")

    deleted = client.delete(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("alice", "project-a"),
    )
    assert deleted.status_code == 200
    assert deleted.json()["removed"] == 1

    gone = client.get(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("alice", "project-a"),
    )
    assert gone.status_code == 404


def test_delete_workflow_missing_returns_404(client):
    response = client.delete(
        "/umrs/workflows/no-such-workflow",
        headers=_auth_headers("alice", "project-a"),
    )
    assert response.status_code == 404


def test_delete_workflow_non_owner_forbidden(client, tmp_path):
    _register(_workflow_json("ct-analysis"), tmp_path, name="ct-analysis", user="alice")

    response = client.delete(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("bob", "project-a"),
    )
    assert response.status_code == 403
    # Owner's registration survives the attempt.
    still_there = client.get(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("alice", "project-a"),
    )
    assert still_there.status_code == 200


def test_delete_workflow_admin_may_delete_other_users_workflow(client, tmp_path):
    _register(_workflow_json("ct-analysis"), tmp_path, name="ct-analysis", user="alice")

    response = client.delete(
        "/umrs/workflows/ct-analysis",
        headers=_auth_headers("admin", "project-a", admin=True),
    )
    assert response.status_code == 200
    assert response.json()["removed"] == 1


def test_delete_workflow_system_scope_requires_admin(client, tmp_path):
    _register(
        _workflow_json("shared-wf"),
        tmp_path,
        name="shared-wf",
        user="bob",
        scope="system",
    )

    # Non-admin cannot delete system workflows, even authenticated.
    forbidden = client.delete(
        "/umrs/workflows/shared-wf",
        headers=_auth_headers("alice", "project-a"),
    )
    assert forbidden.status_code == 403

    # Admin can.
    allowed = client.delete(
        "/umrs/workflows/shared-wf",
        headers=_auth_headers("admin", "project-a", admin=True),
    )
    assert allowed.status_code == 200


# ── Run definitions (persisted invocation templates) ─────────────────────────


@pytest.fixture(autouse=True)
def tmp_definitions(tmp_path):
    definitions_file = str(tmp_path / "run_definition_catalog.json")
    with patch(
        "umrs.server.run_definition_registry._definitions_path",
        return_value=definitions_file,
    ):
        yield definitions_file


def _save_def(client, headers, name="nightly-seg", workflow="ct-analysis", params=None):
    return client.put(
        "/umrs/workflows/run-definitions",
        headers=headers,
        json={
            "name": name,
            "workflow_name": workflow,
            "params": params if params is not None else {"bucket": "b1"},
            "description": "nightly replay",
        },
    )


def test_run_definition_roundtrip(client, tmp_path):
    _register(
        _workflow_json("ct-analysis", with_required_bucket=True),
        tmp_path,
        name="ct-analysis",
        user="alice",
    )

    saved = _save_def(client, _auth_headers("alice", "project-a"))
    assert saved.status_code == 200
    assert saved.json()["definition"]["workflow_name"] == "ct-analysis"

    listed = client.get(
        "/umrs/workflows/run-definitions",
        headers=_auth_headers("alice", "project-a"),
    )
    assert listed.status_code == 200
    assert listed.json()["count"] == 1

    got = client.get(
        "/umrs/workflows/run-definitions/nightly-seg",
        headers=_auth_headers("alice", "project-a"),
    )
    assert got.status_code == 200
    assert got.json()["definition"]["params"] == {"bucket": "b1"}


def test_run_definition_rejects_unregistered_workflow(client):
    response = _save_def(client, _auth_headers("alice", "project-a"), workflow="ghost-wf")
    assert response.status_code == 404
    assert "registered workflow" in response.json()["detail"]


def test_run_definition_is_owner_visible_only(client, tmp_path):
    _register(_workflow_json("ct-analysis"), tmp_path, name="ct-analysis", user="alice")
    _save_def(client, _auth_headers("alice", "project-a"))

    # Non-owner does not see it (hidden existence).
    not_found = client.get(
        "/umrs/workflows/run-definitions/nightly-seg",
        headers=_auth_headers("bob", "project-a"),
    )
    assert not_found.status_code == 404

    # Owner list excludes others; admin list is global.
    bob_list = client.get(
        "/umrs/workflows/run-definitions",
        headers=_auth_headers("bob", "project-a"),
    )
    assert bob_list.json()["count"] == 0
    admin_list = client.get(
        "/umrs/workflows/run-definitions",
        headers=_auth_headers("admin", "project-a", admin=True),
    )
    assert admin_list.json()["count"] == 1


def test_run_definition_delete_owner_and_admin(client, tmp_path):
    _register(_workflow_json("ct-analysis"), tmp_path, name="ct-analysis", user="alice")
    _save_def(client, _auth_headers("alice", "project-a"), name="def-one")
    _save_def(client, _auth_headers("alice", "project-a"), name="def-two")

    deleted = client.delete(
        "/umrs/workflows/run-definitions/def-one",
        headers=_auth_headers("alice", "project-a"),
    )
    assert deleted.status_code == 200

    admin_deleted = client.delete(
        "/umrs/workflows/run-definitions/def-two",
        headers=_auth_headers("admin", "project-a", admin=True),
    )
    assert admin_deleted.status_code == 200

    listed = client.get(
        "/umrs/workflows/run-definitions",
        headers=_auth_headers("alice", "project-a"),
    )
    assert listed.json()["count"] == 0


def test_run_definition_trigger_replays_exact_params(client, tmp_path):
    _register(
        _workflow_json("ct-analysis", with_required_bucket=True),
        tmp_path,
        name="ct-analysis",
        user="alice",
    )
    _save_def(client, _auth_headers("alice", "project-a"), params={"bucket": "b1"})

    # Trigger must succeed with the stored params even though the request body
    # carries none — prove the replay path supplies them.
    triggered = client.post(
        "/umrs/workflows/run-definitions/nightly-seg/trigger",
        headers=_auth_headers("alice", "project-a"),
    )
    assert triggered.status_code in (200, 201, 202)
    payload = triggered.json()
    assert payload.get("status") == "ok"
    assert payload.get("execution_id") or payload.get("result", {}).get("execution_id")
