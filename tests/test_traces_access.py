"""Execution traces: every route is the verified caller's own jobs only, and
the Postgres path writes what it says it writes.

Found 2026-10-12 (ADR-092 slice 7 review): ``GET /umrs/traces/{id}``,
``/execution/{id}``, ``/step/{name}``, ``/{id}/lineage``, ``/{id}/children``,
``/{id}/inputs``, ``/{id}/outputs`` and ``PATCH /{id}`` checked no job access,
so any authenticated caller read (or changed) another project's traces;
``DELETE /expired`` let any caller purge every user's expired traces. On
Postgres, ``_db_insert_trace`` never wrote ``inputs_snapshot``, and
``_db_update_trace`` passed ``outputs_snapshot`` as a dict psycopg2 cannot
adapt, so a PATCH with outputs fell back to memory and answered 404.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import jobs as jobs_api
from umrs.api import traces as traces_api
from umrs.jobs.execution_trace_store import ExecutionTraceStore
from umrs.jobs.store import JobNotFoundError


class _Caller:
    def __init__(self, user_id, project_id, is_admin=False):
        self.user_id, self.project_id, self.is_admin = user_id, project_id, is_admin

    def can_access_job(self, job):
        if self.is_admin:
            return True
        return job.project_id == self.project_id and job.user_id == self.user_id


ALICE = _Caller("alice", "proj-a")
BOB = _Caller("bob", "proj-b")
ADMIN = _Caller("root", "ops", is_admin=True)


class _Jobs:
    def __init__(self):
        self.jobs = {
            "job-a": SimpleNamespace(job_id="job-a", project_id="proj-a", user_id="alice"),
            "job-b": SimpleNamespace(job_id="job-b", project_id="proj-b", user_id="bob"),
        }

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        return self.jobs[job_id]

    def get_jobs_batch(self, job_ids):
        return {j: self.jobs[j] for j in job_ids if j in self.jobs}


@pytest.fixture()
def world():
    traces = ExecutionTraceStore(db_url=None)
    a_root = traces.create_trace("job-a", "s1", execution_id="ex-1", inputs={"q": "a"})
    a_child = traces.create_trace(
        "job-a", "s2", execution_id="ex-1", parent_trace_id=a_root.trace_id
    )
    b_root = traces.create_trace("job-b", "s1", execution_id="ex-1", inputs={"q": "b"})
    orphan = traces.create_trace("job-gone", "s1", execution_id="ex-1")
    jobs = _Jobs()
    return SimpleNamespace(traces=traces, jobs=jobs, a=a_root, a2=a_child, b=b_root, orphan=orphan)


def _client(world, caller=None):
    app = FastAPI()
    app.include_router(traces_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: world.jobs
    app.dependency_overrides[traces_api.get_trace_store] = lambda: world.traces
    if caller is not None:
        app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return TestClient(app)


SINGLE = ["", "/lineage", "/children", "/inputs", "/outputs"]


@pytest.mark.parametrize("suffix", SINGLE)
def test_another_projects_trace_is_not_found(world, suffix):
    res = _client(world, BOB).get(f"/umrs/traces/{world.a.trace_id}{suffix}")
    assert res.status_code == 404
    assert world.a.trace_id in res.json()["detail"]
    assert "alice" not in res.text and '"q"' not in res.text


@pytest.mark.parametrize("suffix", SINGLE)
def test_the_owner_still_reads_its_trace(world, suffix):
    assert _client(world, ALICE).get(f"/umrs/traces/{world.a.trace_id}{suffix}").status_code == 200


def test_a_trace_whose_job_is_gone_is_not_found(world):
    res = _client(world, ADMIN).get(f"/umrs/traces/{world.orphan.trace_id}")
    assert res.status_code == 404


def test_another_projects_trace_cannot_be_changed(world):
    res = _client(world, BOB).patch(
        f"/umrs/traces/{world.a.trace_id}", json={"status": "failed", "error": "x"}
    )
    assert res.status_code == 404
    assert world.traces.get_trace(world.a.trace_id).status == "pending"


def test_a_child_cannot_be_hung_under_another_projects_trace(world):
    res = _client(world, BOB).post(
        "/umrs/traces",
        json={"job_id": "job-b", "step_name": "s", "parent_trace_id": world.a.trace_id},
    )
    assert res.status_code == 404


@pytest.mark.parametrize("path", ["/umrs/traces/execution/ex-1", "/umrs/traces/step/s1"])
def test_lists_carry_only_the_callers_jobs(world, path):
    bob = _client(world, BOB).get(path).json()
    assert {t["job_id"] for t in bob["traces"]} == {"job-b"}
    assert bob["count"] == 1
    admin = _client(world, ADMIN).get(path).json()
    assert {t["job_id"] for t in admin["traces"]} == {"job-a", "job-b"}  # job-gone: no job


def test_children_and_lineage_leave_out_unreadable_jobs(world):
    # a child of alice's trace hung in bob's job (written before the parent check)
    b_child = world.traces.create_trace("job-b", "s3", parent_trace_id=world.a.trace_id)
    alice = _client(world, ALICE)
    kids = alice.get(f"/umrs/traces/{world.a.trace_id}/children").json()
    assert [t["trace_id"] for t in kids["traces"]] == [world.a2.trace_id]
    lineage = _client(world, BOB).get(f"/umrs/traces/{b_child.trace_id}/lineage").json()
    assert [t["trace_id"] for t in lineage["traces"]] == [b_child.trace_id]


def test_expired_cleanup_is_an_admin_operation(world):
    assert _client(world, BOB).delete("/umrs/traces/expired").status_code == 403
    assert _client(world, ADMIN).delete("/umrs/traces/expired").status_code == 200


def test_the_routes_need_the_gateway_verified_identity(world, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    client = _client(world)  # the real get_caller_identity
    for path in (f"/umrs/traces/{world.a.trace_id}", "/umrs/traces/execution/ex-1"):
        res = client.get(path, params={"caller_user_id": "alice", "caller_project_id": "proj-a"})
        assert res.status_code == 401, path


# ── the Postgres path ───────────────────────────────────────────────────────


class _Conn:
    def __init__(self, log, fail=False, row=None):
        self.log, self.fail, self.row = log, fail, row

    def cursor(self):
        conn = self

        class _Cur:
            description = [("trace_id",), ("job_id",), ("step_name",)]

            def execute(self, sql, params=None):
                if conn.fail:
                    raise RuntimeError("db down")
                for p in params or ():
                    if isinstance(p, dict):  # psycopg2: can't adapt type 'dict'
                        raise TypeError("can't adapt type 'dict'")
                conn.log.append((" ".join(sql.split()), params))

            def fetchone(self):
                return conn.row

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return _Cur()

    def commit(self):
        pass

    def close(self):
        pass


def _pg_store(monkeypatch, fail=False):
    log: list = []
    store = ExecutionTraceStore(db_url=None)
    store._db_url = "postgresql://umrs@db/umrs"
    monkeypatch.setattr(ExecutionTraceStore, "db_enabled", property(lambda _s: True))
    monkeypatch.setattr(store, "_get_conn", lambda: _Conn(log, fail, ("trc-1", "job-a", "s1")))
    return store, log


def test_the_insert_writes_the_inputs_snapshot(monkeypatch):
    store, log = _pg_store(monkeypatch)
    store.create_trace("job-a", "s1", inputs={"zeta": 1, "alpha": 2})
    ((sql, params),) = log
    assert "inputs_snapshot" in sql
    assert '{"zeta": 1, "alpha": 2}' in params


def test_an_update_with_outputs_reaches_postgres_as_json(monkeypatch):
    store, log = _pg_store(monkeypatch)
    record = store.complete_trace("trc-1", outputs={"answer": 42})
    update = next((sql, p) for sql, p in log if sql.startswith("UPDATE execution_traces"))
    assert "outputs_snapshot = %s" in update[0]
    assert json.dumps({"answer": 42}) in update[1]
    assert record is not None and record.trace_id == "trc-1"
    assert store._traces == {}


@pytest.mark.parametrize("write", ["create", "update", "cleanup"])
def test_a_database_failure_with_a_dsn_raises_never_falls_back_to_memory(monkeypatch, write):
    store, _log = _pg_store(monkeypatch, fail=True)
    calls = {
        "create": lambda: store.create_trace("job-a", "s1"),
        "update": lambda: store.fail_trace("trc-1", "x"),
        "cleanup": store.cleanup_expired,
    }
    with pytest.raises(RuntimeError, match="db down"):
        calls[write]()
    assert store._traces == {}


def test_a_dsn_without_a_driver_refuses_the_memory_fallback():
    store = ExecutionTraceStore(db_url=None)
    store._db_url = "postgresql://umrs@db/umrs"  # set, but psycopg2 never loaded
    with pytest.raises(RuntimeError, match="refusing the in-memory fallback"):
        store.create_trace("job-a", "s1")


def test_without_a_dsn_the_dev_memory_store_still_works():
    store = ExecutionTraceStore(db_url=None)
    record = store.create_trace("job-a", "s1")
    assert store.complete_trace(record.trace_id, outputs={"a": 1}).outputs_snapshot == {"a": 1}


def test_a_patch_whose_write_fails_is_named_not_a_404(world, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("db down")

    monkeypatch.setattr(world.traces, "complete_trace", boom)
    res = _client(world, ALICE).patch(
        f"/umrs/traces/{world.a.trace_id}", json={"status": "succeeded", "outputs": {"a": 1}}
    )
    assert res.status_code == 503
    assert "trace update failed" in res.json()["detail"]
