"""ADR-108 W2c (D-6): workflow upload provenance and scoped-run stamping.

- An upload's ``provenance`` form field (repo, branch, commit, dirty) is
  cleaned and kept on the version's catalog row; a malformed one is dropped
  and never fails the upload.
- A trigger under a scope token records the scope's path and its
  repo/branch/commit on the durable job spec, and the ingest of the run's
  outputs carries ``scope_path`` so each artifact is stamped with it.
- A ZMQ client's own ``_scope`` claim is ignored.
"""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.scopes import tokens as scope_tokens
from umrs.server import workflow_registry_handler as wrh
from umrs.server.workflow_execution_registry import reset_execution_registry_for_tests

from umrs_protocol.handshake_hmac import sign_claims

SECRET = "test-secret"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setattr("umrs.api.workflows._check_workflow_quota", lambda *_a: None)
    monkeypatch.setattr("umrs.api.workflows.get_optional_job_store", lambda: None)
    monkeypatch.setattr("umrs.server.step_reaper.durable_job_store", lambda: None)
    wrh.set_workflow_circles_resolver(lambda _user: [])
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _h(user: str = "alice", project: str = "proj-a") -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
    }


def _zip(version: str) -> bytes:
    definition = {
        "name": "wf",
        "version": version,
        "description": "W2c provenance test: one step, no inputs, no outputs.",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "s1", "entrypoint": "platform_steps:S1", "depends_on": []}],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("workflow.json", json.dumps(definition))
    return buffer.getvalue()


def _upload(client: TestClient, version: str, provenance: str | None = None):
    data = {"workflow_name": "wf"}
    if provenance is not None:
        data["provenance"] = provenance
    return client.post(
        "/umrs/workflows/upload",
        headers=_h(),
        data=data,
        files={"file": ("wf.zip", _zip(version), "application/zip")},
    )


def _row(version: str) -> dict[str, Any]:
    rows = wrh._load_catalog().get("workflows", [])
    return next(r for r in rows if r["name"] == "wf" and r["version"] == version)


# ── upload provenance ─────────────────────────────────────────────────────


def test_upload_keeps_the_cleaned_provenance_on_the_version_row(client) -> None:
    sent = {
        "repo": "github.com/org/wf",
        "branch": "feat/x",
        "commit": COMMIT.upper(),
        "dirty": False,
        "token": "secret",
    }
    response = _upload(client, "1.0.0", json.dumps(sent))
    assert response.status_code == 202, response.text
    assert _row("1.0.0")["provenance"] == {
        "repo": "github.com/org/wf",
        "branch": "feat/x",
        "commit": COMMIT,
        "dirty": False,
        "client_reported": True,
    }


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '{"commit": "zz"}', "null", ""])
def test_a_malformed_provenance_is_dropped_never_a_failed_upload(client, bad) -> None:
    response = _upload(client, "1.0.0", bad)
    assert response.status_code == 202, response.text
    assert "provenance" not in _row("1.0.0")


def test_an_upload_without_provenance_has_none(client) -> None:
    response = _upload(client, "1.0.0")
    assert response.status_code == 202, response.text
    assert "provenance" not in _row("1.0.0")


# ── scoped trigger → durable spec → output ingest ─────────────────────────


def _token(**claims: str) -> str:
    token, _ = scope_tokens.issue(
        user="alice",
        project="proj-a",
        path="s-w2c/loop-1",
        grants=("workflow:wf",),
        secret=SECRET,
        **claims,
    )
    return token


@pytest.fixture
def captured(monkeypatch) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def _fake(cmd, client_id, payload_str, job_backend=None, **_kw):
        sent.append(json.loads(payload_str))
        return json.dumps({"status": "ok", "execution_id": "e-1", "workflow_name": "wf"})

    monkeypatch.setattr("umrs.api.workflows.handle_workflow_command", _fake)
    return sent


def test_a_scoped_trigger_hands_its_verified_scope_to_the_run(client, captured) -> None:
    _upload(client, "1.0.0")
    token = _token(repo="github.com/org/wf", branch="feat/x", commit=COMMIT)
    response = client.post(
        "/umrs/workflows/wf/trigger", json={}, headers={**_h(), "x-r4-scope": token}
    )
    assert response.status_code == 200, response.text
    assert captured[-1]["_scope"] == {
        "path": "s-w2c/loop-1",
        "repo": "github.com/org/wf",
        "branch": "feat/x",
        "commit": COMMIT,
    }


def test_an_unscoped_trigger_carries_no_scope(client, captured) -> None:
    _upload(client, "1.0.0")
    assert client.post("/umrs/workflows/wf/trigger", json={}, headers=_h()).status_code == 200
    assert "_scope" not in captured[-1]


class _Backend:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def submit_workflow_run(self, _client_id: str, payload: str) -> str:
        self.payloads.append(json.loads(payload))
        return json.dumps({"status": "ok", "job": {"job_id": "j-1"}})


def _submit(scope: Any) -> dict[str, Any]:
    backend = _Backend()
    wrh._submit_durable_workflow_run(
        client_id="alice",
        workflow_name="wf",
        params={},
        execution_id="e-1",
        source_path="/x",
        step_count=1,
        job_backend=backend,  # type: ignore[arg-type]
        scope=scope,
    )
    return backend.payloads[-1]["spec"]


def test_the_durable_spec_records_the_scope_path_and_commit() -> None:
    spec = _submit({"path": "s-w2c/loop-1", "commit": COMMIT, "junk": "x"})
    assert spec["scope_path"] == "s-w2c/loop-1"
    assert spec["scope"] == {"path": "s-w2c/loop-1", "commit": COMMIT}
    unscoped = _submit(None)
    assert "scope_path" not in unscoped and "scope" not in unscoped


def test_a_zmq_scope_claim_is_ignored() -> None:
    payload = {"_scope": {"path": "forged"}}
    assert wrh._verified_scope(payload, identity_verified=False) is None
    assert wrh._verified_scope(payload, identity_verified=True) == {"path": "forged"}


def test_the_output_ingest_carries_the_runs_scope_path(monkeypatch) -> None:
    from umrs.jobs import knowledge_adapters as ka
    from umrs.jobs import service

    submitted: list[Any] = []
    fake = SimpleNamespace(submit_job=lambda sub: submitted.append(sub) or "ingest-1")
    monkeypatch.setattr(service, "get_job_service", lambda: fake)
    monkeypatch.setattr(ka, "_store_artifact_provenance", lambda *_a, **_k: None)
    run = SimpleNamespace(spec={"scope_path": "s-w2c/loop-1"}, priority=0, job_id="j-1")
    art = [{"kind": "document", "source_ref": "s3://b/k", "artifact_id": "a", "content_hash": "h"}]
    assert ka._enqueue_ingest_job(art, "b", "proj-a", "alice", run, None, "wf", "*") == "ingest-1"
    assert submitted[-1].spec["scope_path"] == "s-w2c/loop-1"
    run.spec = {}
    ka._enqueue_ingest_job(art, "b", "proj-a", "alice", run, None, "wf", "*")
    assert "scope_path" not in submitted[-1].spec
