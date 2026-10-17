"""H-1 memberships lane (2026-10-18): a workflow may declare a project from
the caller's memberships only when the gateway signed the list.

Before: strict mode ignored ``X-Verified-Project-Memberships`` entirely
(wave-6 F1b), because no HMAC covered it, so multi-project users could not
declare a member project; mvp trusted it as written. After: strict mode
honours the list when ``X-Verified-Memberships-HMAC`` verifies against the
signed (user, project, admin) tuple, and ignores it otherwise.
"""

from __future__ import annotations

import json
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

from r4_contracts.handshake_hmac import sign_claims, sign_memberships

SECRET = "h1r-memberships-wf-secret"


def _identity(user: str, project: str, admin: bool = False, hmac: str | None = None) -> dict:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": hmac or sign_claims(SECRET, user, project, admin),
    }


def _with_members(headers: dict, members: str, signed_for: str | None = None) -> dict:
    sig = sign_memberships(
        SECRET,
        headers["X-User-ID"],
        headers["X-Verified-Project-ID"],
        headers["X-Verified-Is-Admin"] == "true",
        members if signed_for is None else signed_for,
    )
    return {
        **headers,
        "X-Verified-Project-Memberships": members,
        "X-Verified-Memberships-HMAC": sig,
    }


ALICE = _identity("alice", "project-a")
SIGNED = _with_members(ALICE, "project-a,project-b")
UNSIGNED = {**ALICE, "X-Verified-Project-Memberships": "project-a,project-b"}
TAMPERED = _with_members(ALICE, "project-a,project-b", signed_for="project-a")
FORGED = {
    **_identity("alice", "project-b", admin=True, hmac="0" * 64),
    "X-Verified-Project-Memberships": "project-b",
    "X-Verified-Memberships-HMAC": "0" * 64,
}


@pytest.fixture(autouse=True)
def _strict(monkeypatch, tmp_path):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    # No job store: the run is recorded in the execution registry this test
    # reads (a full run may leave a Postgres-backed store bound).
    monkeypatch.setattr("umrs.api.workflows.get_optional_job_store", lambda: None)
    reset_execution_registry_for_tests()
    with (
        patch(
            "umrs.server.workflow_registry_handler._catalog_path",
            return_value=str(tmp_path / "workflow_catalog.json"),
        ),
        patch(
            "umrs.server.run_definition_registry._definitions_path",
            return_value=str(tmp_path / "run_definition_catalog.json"),
        ),
    ):
        yield
    reset_execution_registry_for_tests()


@pytest.fixture
def client(tmp_path) -> TestClient:
    workflow = {
        "name": "ct-analysis",
        "version": "1.0",
        "description": "workflow test",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "s1", "entrypoint": "m:StepOne", "depends_on": []}],
    }
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow))
    register_workflow(
        workflow_name="ct-analysis",
        version="1.0",
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=workflow,
        scope="user",
    )
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _trigger(client, headers, project="project-b"):
    return client.post(
        "/umrs/workflows/ct-analysis/trigger",
        headers=headers,
        json={"user_id": "alice", "params": {"project_id": project, "bucket": "b"}},
    )


def test_signed_memberships_admit_a_declared_member_project(client):
    resp = _trigger(client, SIGNED)
    assert resp.status_code == 200, resp.text
    record = get_execution_registry().get(resp.json()["execution_id"])
    assert record.params["project_id"] == "project-b"


@pytest.mark.parametrize("headers", [UNSIGNED, TAMPERED], ids=["unsigned", "tampered"])
def test_unsigned_or_tampered_memberships_are_ignored(client, headers):
    resp = _trigger(client, headers)
    assert resp.status_code == 403
    assert "project-b" in resp.text


def test_signed_list_never_admits_an_unlisted_project(client):
    assert _trigger(client, SIGNED, project="project-x").status_code == 403


def test_forged_tuple_is_refused(client):
    assert _trigger(client, FORGED).status_code == 401


def test_run_definition_replay_follows_the_same_rule(client):
    saved = client.put(
        "/umrs/workflows/run-definitions",
        headers=ALICE,
        json={
            "name": "nightly",
            "workflow_name": "ct-analysis",
            "params": {"project_id": "project-b", "bucket": "b"},
        },
    )
    assert saved.status_code == 200, saved.text
    path = "/umrs/workflows/run-definitions/nightly/trigger"
    assert client.post(path, headers=UNSIGNED).status_code == 403
    assert client.post(path, headers=SIGNED).status_code == 200
