"""The A2L conservation ledger answers only on a verified identity, within scope.

Found 2026-10-09 while making ``r4 artifact`` work on the stand-alone stack:
every ``/umrs/a2l/ledger/*`` route took no caller identity — on the UMRS port
it answered an anonymous request (200), listed every user's rows, and wrote as
whatever ``user_id`` a body named. Reads are now limited to the caller's
readable projects (knowledge.access_scope), writes are made as the caller, a
row the caller may not see answers 404, and job-scoped reads need job access.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.a2l import AgentExecution, ArtifactRecord
from umrs.api import a2l_ledger
from umrs.api.jobs import CallerIdentity, get_caller_identity

_MINE = ArtifactRecord(id="a-mine", type="report", uri="s3://b/m", user_id="ana", project_id="p1")
_THEIRS = ArtifactRecord(id="a-theirs", type="report", uri="s3://b/t", user_id="bo", project_id="p2")


class _Store:
    def __init__(self):
        self.calls: dict = {}
        self.artifacts = {a.id: a for a in (_MINE, _THEIRS)}
        self.executions = {
            "e-theirs": AgentExecution(
                id="e-theirs", agent_name="x", user_id="bo", project_id="p2", status="running"
            )
        }

    async def list_artifacts(self, **kw):
        self.calls["list_artifacts"] = kw
        return [_MINE]

    async def get_artifact(self, artifact_id):
        return self.artifacts.get(artifact_id)

    async def derive_artifact(self, req):
        self.calls["derive"] = req
        return _MINE

    async def create_artifact(self, record):
        self.calls["create"] = record
        return record

    async def update_artifact_status(self, artifact_id, *_a, **_kw):
        return self.artifacts[artifact_id]

    async def lineage(self, artifact_id, **_kw):
        return {
            "root_id": artifact_id,
            "nodes": [_MINE.to_dict(), _THEIRS.to_dict()],
            "edges": [{"parent_id": "a-theirs", "child_id": "a-mine", "relationship_type": "d"}],
        }

    async def list_executions(self, **kw):
        self.calls["list_executions"] = kw
        return []

    async def get_execution(self, execution_id):
        return self.executions.get(execution_id)

    async def create_execution(self, req):
        self.calls["create_execution"] = req
        return self.executions["e-theirs"]

    async def list_decisions(self, **_kw):
        return []

    async def job_conservation_view(self, job_id):
        return {"spine": {"job_id": job_id}, "artifacts": [], "executions": [], "decisions": []}


def _client(monkeypatch, *, identity=None, jobs=None):
    store = _Store()
    app = FastAPI()
    app.include_router(a2l_ledger.router)

    async def fake_store():
        return store

    app.dependency_overrides[a2l_ledger._get_store] = fake_store
    if identity is not None:
        app.dependency_overrides[get_caller_identity] = lambda: identity
    job_store = SimpleNamespace(get_job=lambda job_id: (jobs or {}).get(job_id))
    monkeypatch.setattr(a2l_ledger, "get_optional_job_store", lambda: job_store)
    return TestClient(app, raise_server_exceptions=False), store


_ANA = CallerIdentity(user_id="ana", project_id="p1", is_admin=False)
_ADMIN = CallerIdentity(user_id="root", project_id="p1", is_admin=True)


def test_an_anonymous_request_is_refused(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    client, _ = _client(monkeypatch)
    assert client.get("/umrs/a2l/ledger/artifacts").status_code in (401, 403)


def test_lists_are_limited_to_readable_projects(monkeypatch):
    client, store = _client(monkeypatch, identity=_ANA)
    assert client.get("/umrs/a2l/ledger/artifacts").status_code == 200
    assert store.calls["list_artifacts"]["readable_projects"] == ["p1", "system", "user:ana"]
    client.get("/umrs/a2l/ledger/agent-executions")
    assert store.calls["list_executions"]["readable_projects"] == ["p1", "system", "user:ana"]


def test_an_admin_lists_everything(monkeypatch):
    client, store = _client(monkeypatch, identity=_ADMIN)
    client.get("/umrs/a2l/ledger/artifacts")
    assert store.calls["list_artifacts"]["readable_projects"] is None


def test_naming_another_project_to_list_is_forbidden(monkeypatch):
    client, _ = _client(monkeypatch, identity=_ANA)
    assert client.get("/umrs/a2l/ledger/artifacts?project_id=p2").status_code == 403


def test_another_projects_row_is_not_found(monkeypatch):
    client, _ = _client(monkeypatch, identity=_ANA)
    assert client.get("/umrs/a2l/ledger/artifacts/a-theirs").status_code == 404
    assert client.get("/umrs/a2l/ledger/artifacts/a-mine").status_code == 200
    patch = client.patch("/umrs/a2l/ledger/artifacts/a-theirs/status", json={"status": "failed"})
    assert patch.status_code == 404
    assert client.get("/umrs/a2l/ledger/agent-executions/e-theirs").status_code == 404


_DERIVE = {
    "parents": ["a-mine"],
    "operation": "summarise",
    "type": "report",
    "uri": "s3://b/r",
    "input_data_hash": "h",
}


def test_derive_is_made_as_the_caller(monkeypatch):
    client, store = _client(monkeypatch, identity=_ANA)
    spoofed = client.post("/umrs/a2l/ledger/artifacts/derive", json={**_DERIVE, "user_id": "bo"})
    assert spoofed.status_code == 403
    ok = client.post("/umrs/a2l/ledger/artifacts/derive", json={**_DERIVE, "user_id": "ana"})
    assert ok.status_code == 201
    assert (store.calls["derive"].user_id, store.calls["derive"].project_id) == ("ana", "p1")


def test_derive_into_another_project_is_forbidden(monkeypatch):
    client, _ = _client(monkeypatch, identity=_ANA)
    body = {**_DERIVE, "user_id": "ana", "project_id": "p2"}
    assert client.post("/umrs/a2l/ledger/artifacts/derive", json=body).status_code == 403


def test_create_is_admin_only_and_gets_a_deterministic_id(monkeypatch):
    body = {"type": "dataset", "uri": "s3://b/d.h5"}
    client, _ = _client(monkeypatch, identity=_ANA)
    assert client.post("/umrs/a2l/ledger/artifacts", json=body).status_code == 403
    client, store = _client(monkeypatch, identity=_ADMIN)
    first = client.post("/umrs/a2l/ledger/artifacts", json=body).json()
    second = client.post("/umrs/a2l/ledger/artifacts", json=body).json()
    assert first["id"] == second["id"] and len(first["id"]) == 64
    assert store.calls["create"].project_id == "p1"


def test_lineage_hides_nodes_the_caller_may_not_read(monkeypatch):
    client, _ = _client(monkeypatch, identity=_ANA)
    graph = client.get("/umrs/a2l/ledger/artifacts/a-mine/lineage").json()
    assert [n["id"] for n in graph["nodes"]] == ["a-mine"]
    assert graph["edges"] == []


def test_job_scoped_reads_need_job_access(monkeypatch):
    jobs = {
        "j-ana": SimpleNamespace(project_id="p1", user_id="ana"),
        "j-bo": SimpleNamespace(project_id="p2", user_id="bo"),
    }
    client, _ = _client(monkeypatch, identity=_ANA, jobs=jobs)
    assert client.get("/umrs/a2l/ledger/jobs/j-bo/view").status_code == 404
    assert client.get("/umrs/a2l/ledger/jobs/j-ana/view").status_code == 200
    assert client.get("/umrs/a2l/ledger/decisions").status_code == 400
    assert client.get("/umrs/a2l/ledger/decisions?job_id=j-bo").status_code == 404
    decided = {"decided_by": "bo", "decision": "approve", "job_id": "j-ana"}
    assert client.post("/umrs/a2l/ledger/decisions", json=decided).status_code == 403


def test_health(monkeypatch):
    client, _ = _client(monkeypatch, identity=_ANA)
    assert client.get("/umrs/a2l/ledger/health").json() == {"status": "ok"}


@pytest.mark.parametrize("project", [None, "", "p1"])
def test_execution_create_is_made_as_the_caller(monkeypatch, project):
    client, store = _client(monkeypatch, identity=_ANA)
    body = {"agent_name": "x", "user_id": "ana"}
    if project is not None:
        body["project_id"] = project
    assert client.post("/umrs/a2l/ledger/agent-executions", json=body).status_code == 201
    assert store.calls["create_execution"].project_id == "p1"


def test_the_cli_derive_body_is_enough(monkeypatch):
    """`r4 artifact derive` sends only type/uri/metadata/project_id/parents."""
    client, store = _client(monkeypatch, identity=_ANA)
    body = {"type": "report", "uri": "s3://b/r.md", "metadata": {}, "parents": ["a-mine"]}
    assert client.post("/umrs/a2l/ledger/artifacts/derive", json=body).status_code == 201
    req = store.calls["derive"]
    assert (req.user_id, req.operation) == ("ana", "derive") and len(req.input_data_hash) == 64
