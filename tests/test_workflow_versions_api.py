"""Workflow versions over REST (ADR-113 WV1/WV2): the contract the r4 lane
implements against — upload, list, describe, trigger/plan, executions,
delete, trigger pins, run definitions, adopt, package/export."""

from __future__ import annotations

import io
import json
import zipfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.server import workflow_registry_handler as wrh
from umrs.server.workflow_execution_registry import reset_execution_registry_for_tests

CODE = "x-platform-error-code"
SECRET = "test-secret"


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setattr("umrs.api.workflows._check_workflow_quota", lambda *_a: None)
    monkeypatch.setattr("umrs.api.workflows.get_optional_job_store", lambda: None)
    monkeypatch.setattr("umrs.server.step_reaper.durable_job_store", lambda: None)
    monkeypatch.setattr(
        "umrs.server.workflow_runner.run_workflow_record",
        lambda record, **_kw: record.update(status="succeeded"),
    )
    wrh.set_workflow_circles_resolver(lambda _user: [])
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _h(user: str = "alice", project: str = "proj-a", admin: bool = False) -> dict:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _zip(version: str | None, code: str = "", **extra) -> bytes:
    definition = {
        "name": "wf",
        "description": "api versions test",
        "source": {"type": "local_path", "path": "."},
        "steps": [{"name": "s1", "entrypoint": "platform_steps:S1", "depends_on": []}],
        **extra,
    }
    if version is not None:
        definition["version"] = version
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("workflow.json", json.dumps(definition))
        zf.writestr("notes.txt", f"notes {code}\n")
    return buffer.getvalue()


def _upload(client, version, *, name="wf", user="alice", code="", **extra):
    return client.post(
        "/umrs/workflows/upload",
        headers=_h(user),
        data={"workflow_name": name},
        files={"file": (f"{name}.zip", _zip(version, code, **extra), "application/zip")},
    )


# ── upload ────────────────────────────────────────────────────────────────


def test_upload_reports_version_digest_and_warnings(client):
    r = _upload(client, "1.0.0", name="cnn-v1.0.0")
    assert r.status_code == 202, r.text
    body = r.json()
    assert (body["status"], body["version"]) == ("ok", "1.0.0")
    assert len(body["source_sha256"]) == 64
    assert body["warnings"] and "version-like suffix" in body["warnings"][0]
    again = _upload(client, "1.0.0", name="cnn-v1.0.0")
    assert (again.status_code, again.json()["status"]) == (200, "unchanged")
    assert again.json()["source_sha256"] == body["source_sha256"]


def test_upload_refusals_are_typed(client):
    assert _upload(client, "1.0.0").status_code == 202
    clash = _upload(client, "1.0.0", code="other")
    assert (clash.status_code, clash.headers[CODE]) == (409, "workflows.version_exists")
    assert "1.0.0" in clash.json()["detail"]
    for version in (None, "1.0"):
        bad = _upload(client, version)
        assert (bad.status_code, bad.headers[CODE]) == (400, "workflows.invalid_version")


# ── list / describe ───────────────────────────────────────────────────────


def test_list_one_row_per_name_and_all_versions(client):
    for version in ("1.0.0", "1.2.0", "1.10.0"):
        assert _upload(client, version, code=version).status_code == 202
    rows = client.get("/umrs/workflows", headers=_h()).json()["workflows"]
    assert len(rows) == 1
    assert (rows[0]["version"], rows[0]["versions"]) == ("1.10.0", ["1.10.0", "1.2.0", "1.0.0"])
    every = client.get("/umrs/workflows?all_versions=true", headers=_h()).json()["workflows"]
    assert [r["version"] for r in every] == ["1.10.0", "1.2.0", "1.0.0"]
    assert len({r["source_sha256"] for r in every}) == 3


def test_describe_names_a_version_and_lists_them(client):
    _upload(client, "1.0.0", code="a")
    _upload(client, "2.0.0", code="b")
    default = client.get("/umrs/workflows/wf", headers=_h()).json()
    assert (default["version"], default["versions"]) == ("2.0.0", ["2.0.0", "1.0.0"])
    one = client.get("/umrs/workflows/wf?version=1.0.0", headers=_h()).json()
    assert one["version"] == "1.0.0" and one["source_sha256"] != default["source_sha256"]
    assert client.get("/umrs/workflows/wf@1.0.0", headers=_h()).json()["version"] == "1.0.0"
    missing = client.get("/umrs/workflows/wf?version=9.0.0", headers=_h())
    assert (missing.status_code, missing.headers[CODE]) == (404, "workflows.version_not_found")
    assert "2.0.0, 1.0.0" in missing.json()["detail"]


# ── trigger / executions ──────────────────────────────────────────────────


def _trigger(client, name="wf", **body):
    return client.post(f"/umrs/workflows/{name}/trigger", json=body, headers=_h())


def test_trigger_runs_the_version_asked_for_and_reports_it(client):
    first = _upload(client, "1.0.0", code="a").json()
    second = _upload(client, "2.0.0", code="b").json()
    pinned = _trigger(client, version="1.0.0")
    assert pinned.status_code == 200, pinned.text
    assert (pinned.json()["workflow_version"], pinned.json()["source_sha256"]) == (
        "1.0.0",
        first["source_sha256"],
    )
    assert _trigger(client).json()["workflow_version"] == "2.0.0"
    assert _trigger(client, "wf@1.0.0").json()["workflow_version"] == "1.0.0"
    execution_id = _trigger(client).json()["execution_id"]
    status = client.get(f"/umrs/workflows/executions/{execution_id}", headers=_h()).json()
    assert status["execution"]["workflow_version"] == "2.0.0"
    results = client.get(f"/umrs/workflows/executions/{execution_id}/results", headers=_h())
    assert (results.json()["workflow_version"], results.json()["source_sha256"]) == (
        "2.0.0",
        second["source_sha256"],
    )
    missing = _trigger(client, version="3.0.0")
    assert (missing.status_code, missing.headers[CODE]) == (404, "workflows.version_not_found")


def test_zmq_run_workflow_takes_a_version(client):
    from umrs.core.TaskCmdStatus import TaskCmd

    _upload(client, "1.0.0", code="a")
    _upload(client, "2.0.0", code="b")
    payload = json.dumps({"workflow_name": "wf", "version": "1.0.0", "params": {}})
    reply = json.loads(wrh.handle_workflow_command(TaskCmd.RUN_WORKFLOW, "alice", payload))
    assert reply["workflow_version"] == "1.0.0"


# ── delete ────────────────────────────────────────────────────────────────


def test_delete_one_version_or_all(client):
    _upload(client, "1.0.0", code="a")
    _upload(client, "2.0.0", code="b")
    _upload(client, "3.0.0", code="c")
    one = client.delete("/umrs/workflows/wf?version=1.0.0", headers=_h())
    assert (one.status_code, one.json()["versions"]) == (200, ["1.0.0"])
    assert client.get("/umrs/workflows/wf", headers=_h()).json()["versions"] == ["3.0.0", "2.0.0"]
    missing = client.delete("/umrs/workflows/wf?version=1.0.0", headers=_h())
    assert (missing.status_code, missing.headers[CODE]) == (404, "workflows.version_not_found")
    every = client.delete("/umrs/workflows/wf", headers=_h())
    assert every.json()["versions"] == ["3.0.0", "2.0.0"]
    assert client.get("/umrs/workflows/wf", headers=_h()).status_code == 404


# ── triggers pin ──────────────────────────────────────────────────────────


def test_trigger_pin_put_and_get(client):
    ingest = [{"on": "ingest", "bucket": "up"}]
    _upload(client, "1.0.0", code="a", triggers=ingest)
    _upload(client, "2.0.0", code="b")
    shown = client.get("/umrs/workflows/wf/triggers", headers=_h()).json()
    assert (shown["version"], shown["trigger_version"], shown["triggers"]) == ("2.0.0", None, [])
    put = client.put("/umrs/workflows/wf/triggers", json={"version": "1.0.0"}, headers=_h())
    assert put.json()["trigger_version"] == "1.0.0"
    shown = client.get("/umrs/workflows/wf/triggers", headers=_h()).json()
    assert (shown["version"], shown["trigger_version"], shown["triggers"]) == (
        "1.0.0",
        "1.0.0",
        ingest,
    )
    bad = client.put("/umrs/workflows/wf/triggers", json={"version": "9.0.0"}, headers=_h())
    assert bad.headers[CODE] == "workflows.version_not_found"
    reset = client.put("/umrs/workflows/wf/triggers", json={"version": None}, headers=_h())
    assert reset.json()["trigger_version"] is None
    foreign = client.put("/umrs/workflows/wf/triggers", json={"version": None}, headers=_h("bob"))
    assert foreign.status_code == 404


# ── run definitions ───────────────────────────────────────────────────────


def test_a_run_definition_pins_and_replays_its_version(client):
    first = _upload(client, "1.0.0", code="a").json()
    saved = client.put(
        "/umrs/workflows/run-definitions",
        json={"name": "nightly", "workflow_name": "wf", "params": {}},
        headers=_h(),
    ).json()["definition"]
    assert (saved["workflow_version"], saved["source_sha256"]) == ("1.0.0", first["source_sha256"])
    _upload(client, "2.0.0", code="b")  # the default moves on
    replay = client.post("/umrs/workflows/run-definitions/nightly/trigger", headers=_h())
    assert replay.status_code == 200, replay.text
    assert replay.json()["workflow_version"] == "1.0.0"
    pinned = client.put(
        "/umrs/workflows/run-definitions",
        json={"name": "v2", "workflow_name": "wf", "version": "2.0.0", "params": {}},
        headers=_h(),
    ).json()["definition"]
    assert pinned["workflow_version"] == "2.0.0"


# ── adopt ─────────────────────────────────────────────────────────────────


def test_adopt_route(client):
    _upload(client, "0.1.0", name="cnn-v0.2.9")
    r = client.post(
        "/umrs/workflows/adopt",
        json={"from_name": "cnn-v0.2.9", "name": "cnn", "version": "0.2.9"},
        headers=_h(),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["name"], body["version"], body["from_name"]) == ("cnn", "0.2.9", "cnn-v0.2.9")
    assert (
        client.get("/umrs/workflows/cnn", headers=_h()).json()["source_sha256"]
        == body["source_sha256"]
    )
    old = client.get("/umrs/workflows/cnn-v0.2.9/triggers", headers=_h()).json()
    assert (old["triggers_disabled"], old["adopted_as"]) == (True, "cnn@0.2.9")
    bad = client.post(
        "/umrs/workflows/adopt",
        json={"from_name": "cnn-v0.2.9", "name": "cnn", "version": "latest"},
        headers=_h(),
    )
    assert bad.headers[CODE] == "workflows.invalid_version"
    foreign = client.post(
        "/umrs/workflows/adopt",
        json={"from_name": "cnn-v0.2.9", "name": "x", "version": "1.0.0"},
        headers=_h("bob"),
    )
    assert foreign.status_code == 403


# ── package / export ──────────────────────────────────────────────────────


def test_package_and_export_take_a_version(client):
    _upload(client, "1.0.0", code="old")
    _upload(client, "2.0.0", code="new")
    package = client.get("/umrs/workflows/wf/package?version=1.0.0", headers=_h()).json()
    notes = next(f for f in package["files"] if f["path"] == "notes.txt")
    assert (package["version"], notes["content"]) == ("1.0.0", "notes old\n")
    exported = client.get("/umrs/workflows/wf/export?version=1.0.0", headers=_h())
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as zf:
        assert zf.read("notes.txt") == b"notes old\n"
        manifest = json.loads(zf.read("EXPORT.json"))
    assert manifest["version"] == "1.0.0" and len(manifest["source_sha256"]) == 64


# ── grants ────────────────────────────────────────────────────────────────


def test_grant_writability_finds_the_callers_row():
    """Before: the first row of the name decided, so bob's same-named row
    listed first made alice's own workflow not writable."""
    from unittest.mock import patch

    from umrs.api.grants import CallerScope, _workflow_writable

    catalog = {
        "workflows": [
            {"name": "wf", "user_id": "bob", "scope": "user"},
            {"name": "wf", "user_id": "alice", "scope": "user", "version": "1.0.0"},
        ]
    }
    with patch.object(wrh, "_load_catalog", return_value=catalog):
        alice = CallerScope(user_id="alice", project_id="p", is_admin=False)
        assert _workflow_writable("wf", alice) is True
        assert _workflow_writable("wf@1.0.0", alice) is True
