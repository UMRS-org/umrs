"""ADR-105 S2 (umrs half, D-5): pushed job events, no second job store.

- JobStore terminal transitions fan out: the auto-ingest pair keeps firing
  and every ``add_job_terminal_listener`` / process-wide listener fires for
  succeeded, failed (not requeued), cancelled and lost (job ended, no retry).
- The emitter writes one ``platform_notifications`` row (kind ``job.<state>``)
  carrying job_id, job_type, execution_id, origin session id.
- ``NotificationStore.list_after`` / ``max_id`` are the durable cursor read.
- ``GET /umrs/jobs/events`` is the caller's own rows only, matched before
  ``/{job_id}``, empty (200) when no pool is wired; ``tail`` starts at now.
- ``x-r4-session-id`` on POST /umrs/jobs and the workflow trigger stamps
  ``spec.origin.session_id``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs import job_events
from umrs.jobs.models import Job, JobAttempt, JobStatus, JobType
from umrs.jobs.reconciler import Reconciler, ReconcileResult
from umrs.jobs.store import (
    JobStore,
    register_process_job_terminal_listener,
    unregister_process_job_terminal_listener,
)
from umrs.platform_notifications import NotificationStore

from r4_contracts.handshake_hmac import sign_claims
from umrs import platform_notifications

SECRET = "job-events-secret"


def _job(
    job_type: JobType = JobType.WORKFLOW_RUN,
    *,
    spec: dict[str, Any] | None = None,
    max_attempts: int = 1,
    status: JobStatus = JobStatus.RUNNING,
) -> Job:
    return Job(
        job_id="job-1",
        project_id="proj-a",
        user_id="alice",
        job_type=job_type,
        spec=spec
        if spec is not None
        else {
            "workflow_name": "relax",
            "execution_id": "wf-abc",
            "origin": {"session_id": "sess-1"},
        },
        idempotency_key="k",
        status=status,
        desired_max_attempts=max_attempts,
        last_state_change_at=datetime(2026, 11, 2, 12, 0, 0),
    )


# ── store fan-out ───────────────────────────────────────────────────────────


class _StoreCursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 1

    def execute(self, sql, _params=None):
        self._conn.sql.append(sql)

    def fetchone(self):
        return self._conn.rows.pop(0) if self._conn.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _StoreConn:
    def __init__(self, rows):
        self.rows = list(rows)
        self.sql: list[str] = []

    def cursor(self, **_kw):
        return _StoreCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def _store(monkeypatch, job: Job, rows: list) -> tuple[JobStore, dict]:
    monkeypatch.setattr(JobStore, "_ensure_schema", lambda _self: None)
    store = JobStore(MagicMock(), "inst")
    conn = _StoreConn(rows)
    held = {"n": 0}

    @contextmanager
    def _get_conn():
        held["n"] += 1
        try:
            yield conn
        finally:
            held["n"] -= 1

    @contextmanager
    def _lock(_conn, _job_id, *, tries=1):
        yield

    attempt = JobAttempt(attempt_id="att-1", job_id="job-1", attempt_no=1, backend="b")
    monkeypatch.setattr(store, "_get_conn", _get_conn)
    monkeypatch.setattr(store, "_advisory_lock", _lock)
    monkeypatch.setattr(store, "_reconcile_fan_out_parent", lambda *_a: None)
    monkeypatch.setattr(store, "_persist_attempt_manifest", lambda *_a: None)
    monkeypatch.setattr(store, "_record_attempt_timing", lambda *_a: None)
    monkeypatch.setattr(store, "_row_to_job", lambda _row: job)
    monkeypatch.setattr(store, "get_job", lambda _id: job)
    monkeypatch.setattr(store, "get_attempt", lambda _id: attempt)
    calls: dict[str, list] = {
        "ingest_ok": [],
        "ingest_fail": [],
        "listener": [],
        "held_at_listener": [],
    }
    store.set_post_completion_callbacks(
        on_succeeded=lambda j, a: calls["ingest_ok"].append((j.job_id, a.attempt_id)),
        on_failed=lambda j, n, r: calls["ingest_fail"].append((j.job_id, n, r)),
    )
    store.add_job_terminal_listener(
        lambda j, state, a, reason: calls["listener"].append(
            (j.job_id, state, getattr(a, "attempt_id", None), reason)
        )
    )
    store.add_job_terminal_listener(lambda *_a: calls["held_at_listener"].append(held["n"]))
    return store, calls


def _fail_rows() -> list:
    return [{"job_id": "job-1", "attempt_no": 1}, {"job_id": "job-1"}]


@pytest.mark.parametrize("path", ["succeeded", "failed", "cancelled"])
def test_listeners_run_after_the_pooled_connection_is_returned(monkeypatch, path):
    """Review U2: inside ``with self._get_conn()`` an exhausted pool would
    drop the event (the listener needs a connection of its own)."""
    if path == "failed":
        store, calls = _store(monkeypatch, _job(max_attempts=1), _fail_rows())
        store.mark_attempt_failed("att-1", "oom")
    elif path == "succeeded":
        store, calls = _store(monkeypatch, _job(), [("job-1",)])
        store.mark_attempt_succeeded("att-1")
    else:
        store, calls = _store(monkeypatch, _job(), [("job-1",)])
        store.cancel_job("job-1")
    assert calls["held_at_listener"] == [0]


def test_succeeded_fires_ingest_and_listener(monkeypatch):
    store, calls = _store(monkeypatch, _job(), [("job-1",)])
    store.mark_attempt_succeeded("att-1")
    assert calls["ingest_ok"] == [("job-1", "att-1")]
    assert calls["listener"] == [("job-1", "succeeded", "att-1", "success")]


def test_terminal_failure_fires_ingest_and_listener(monkeypatch):
    store, calls = _store(monkeypatch, _job(max_attempts=1), _fail_rows())
    assert store.mark_attempt_failed("att-1", "oom") is False
    assert calls["ingest_fail"] == [("job-1", 1, "oom")]
    assert calls["listener"] == [("job-1", "failed", "att-1", "oom")]


def test_requeued_failure_fires_nothing(monkeypatch):
    store, calls = _store(monkeypatch, _job(max_attempts=3), _fail_rows())
    assert store.mark_attempt_failed("att-1", "node died") is True
    assert calls["ingest_fail"] == [] and calls["listener"] == []


def test_cancel_fires_listener_only(monkeypatch):
    store, calls = _store(monkeypatch, _job(), [("job-1",)])
    store.cancel_job("job-1", reason="user_cancelled")
    assert calls["listener"] == [("job-1", "cancelled", None, "user_cancelled")]
    assert calls["ingest_ok"] == [] and calls["ingest_fail"] == []


def test_listener_error_never_blocks_the_transition(monkeypatch):
    store, calls = _store(monkeypatch, _job(), [("job-1",)])

    def _boom(*_a):
        raise RuntimeError("listener down")

    store.add_job_terminal_listener(_boom)
    store.mark_attempt_succeeded("att-1")  # no raise
    assert calls["ingest_ok"] and calls["listener"]


def test_process_listener_fires_for_every_store(monkeypatch):
    seen: list = []

    def fn(job, state, attempt, reason):
        seen.append(state)

    register_process_job_terminal_listener(fn)
    register_process_job_terminal_listener(fn)  # idempotent
    try:
        store, _ = _store(monkeypatch, _job(), [("job-1",)])
        store.cancel_job("job-1")
    finally:
        unregister_process_job_terminal_listener(fn)
    assert seen == ["cancelled"]


def _reconciler(monkeypatch, job: Job) -> tuple[Reconciler, list, list]:
    monkeypatch.setattr(JobStore, "_ensure_schema", lambda _self: None)
    store = JobStore(MagicMock(), "inst")
    fired: list = []
    monkeypatch.setattr(store, "mark_attempt_lost", lambda *_a: None)
    monkeypatch.setattr(store, "get_job", lambda _id: job)
    monkeypatch.setattr(store, "get_attempt", lambda _id: None)
    store.add_job_terminal_listener(lambda _j, s, _a, r: fired.append((s, r)))
    rec = Reconciler(store)
    moves: list = []
    monkeypatch.setattr(rec, "_fail_job", lambda jid, _reason: moves.append(("fail", jid)) or True)
    monkeypatch.setattr(
        rec, "_requeue_job", lambda jid, _when: moves.append(("requeue", jid)) or True
    )
    return rec, fired, moves


def test_lost_attempt_that_ends_the_job_fires_lost(monkeypatch):
    rec, fired, moves = _reconciler(monkeypatch, _job(max_attempts=1))
    rec._process_stale_attempt("att-1", "job-1", 1, "old", ReconcileResult())
    assert moves == [("fail", "job-1")]
    assert [s for s, _ in fired] == ["lost"]
    assert "exhausted retries" in fired[0][1]


def _job_id_and_guard(rest: list, jobs: dict) -> tuple[str, list]:
    ids = [i for i, value in enumerate(rest) if isinstance(value, str) and value in jobs]
    at = ids[0]
    return rest[at], rest[at + 1 :]


class _JobsCursor:
    """``UPDATE jobs ... WHERE job_id = %s [AND status = %s]`` on a dict."""

    def __init__(self, jobs):
        self._jobs = jobs
        self._row = None

    def execute(self, _sql, params):
        # (status, job_id, guard) for _fail_job; (status, next_run, job_id,
        # guard) for _requeue_job; the unguarded pre-fix SQL lacks the guard.
        new_status, *rest = params
        job_id, guard = _job_id_and_guard(rest, self._jobs)
        current = self._jobs.get(job_id)
        self._row = None
        if current is None or (guard and current != guard[0]):
            return
        self._jobs[job_id] = new_status
        self._row = (job_id,)

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _JobsPool:
    def __init__(self, jobs):
        self.jobs = jobs

    def getconn(self):
        conn = MagicMock()
        conn.cursor.return_value = _JobsCursor(self.jobs)
        return conn

    def putconn(self, _conn):
        pass


def _guarded_reconciler(monkeypatch, status_after_read: str, max_attempts: int = 1):
    """The reconciler reads the job RUNNING; by the time it fails or
    requeues it, the row holds ``status_after_read`` (a concurrent cancel
    committed)."""
    rec, fired, _ = _reconciler(monkeypatch, _job(max_attempts=max_attempts))
    pool = _JobsPool({"job-1": status_after_read})
    rec.job_store.db_pool = pool
    # the real, guarded transitions (the _reconciler helper stubs them)
    monkeypatch.setattr(rec, "_fail_job", Reconciler._fail_job.__get__(rec))
    monkeypatch.setattr(rec, "_requeue_job", Reconciler._requeue_job.__get__(rec))
    return rec, fired, pool


def test_a_concurrently_cancelled_job_is_never_requeued(monkeypatch):
    """A cancel that committed between the reconciler's read and its requeue
    must win: the unguarded UPDATE set CANCELLED back to QUEUED, so the
    cancelled job ran again."""
    rec, fired, pool = _guarded_reconciler(monkeypatch, JobStatus.CANCELLED.value, max_attempts=3)
    result = ReconcileResult()
    rec._process_stale_attempt("att-1", "job-1", 1, "old", result)
    assert pool.jobs["job-1"] == JobStatus.CANCELLED.value
    assert result.requeued_jobs == [] and fired == []


def test_a_still_running_lost_job_with_retries_is_requeued(monkeypatch):
    rec, fired, pool = _guarded_reconciler(monkeypatch, JobStatus.RUNNING.value, max_attempts=3)
    result = ReconcileResult()
    rec._process_stale_attempt("att-1", "job-1", 1, "old", result)
    assert pool.jobs["job-1"] == JobStatus.QUEUED.value
    assert result.requeued_jobs == ["job-1"] and fired == []


def test_a_concurrently_cancelled_job_is_not_overwritten_and_emits_no_lost(monkeypatch):
    """Review U3: cancel_job committed (and fired ``cancelled``) between the
    reconciler's read and its fail; the unguarded UPDATE turned CANCELLED
    into FAILED and fired ``lost`` as well."""
    rec, fired, pool = _guarded_reconciler(monkeypatch, JobStatus.CANCELLED.value)
    result = ReconcileResult()
    rec._process_stale_attempt("att-1", "job-1", 1, "old", result)
    assert pool.jobs["job-1"] == JobStatus.CANCELLED.value
    assert fired == [] and result.failed_jobs == []


def test_a_still_running_lost_job_fails_and_emits_lost(monkeypatch):
    rec, fired, pool = _guarded_reconciler(monkeypatch, JobStatus.RUNNING.value)
    result = ReconcileResult()
    rec._process_stale_attempt("att-1", "job-1", 1, "old", result)
    assert pool.jobs["job-1"] == JobStatus.FAILED.value
    assert [s for s, _ in fired] == ["lost"] and result.failed_jobs == ["job-1"]


def test_lost_attempt_that_requeues_fires_nothing(monkeypatch):
    rec, fired, moves = _reconciler(monkeypatch, _job(max_attempts=3))
    rec._process_stale_attempt("att-1", "job-1", 1, "old", ReconcileResult())
    assert moves == [("requeue", "job-1")] and fired == []


# ── emitter ─────────────────────────────────────────────────────────────────


@pytest.fixture
def notified(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(platform_notifications, "_EMITTER", object())
    monkeypatch.setattr(platform_notifications, "notify", lambda **kw: rows.append(kw) or True)
    return rows


def test_emitter_row_shape_for_a_workflow_run(notified):
    assert job_events.emit_job_terminal_event(_job(), "succeeded", None, "success")
    (row,) = notified
    assert row["kind"] == "job.succeeded"
    assert row["title"] == "workflow relax finished: succeeded"
    assert (row["user_id"], row["project_id"]) == ("alice", "proj-a")
    assert row["detail"] == {
        "job_id": "job-1",
        "job_type": "workflow_run",
        "status": "succeeded",
        "execution_id": "wf-abc",
        "session_id": "sess-1",
        "exit_reason": "success",
        "finished_at": "2026-11-02T12:00:00+00:00",
    }


def test_emitter_non_workflow_job_has_no_execution_id(notified):
    job = _job(JobType.CLEANUP, spec={"execution_id": "x"})
    job_events.emit_job_terminal_event(job, "cancelled", None, "user_cancelled")
    (row,) = notified
    assert row["kind"] == "job.cancelled"
    assert row["detail"]["execution_id"] is None
    assert row["detail"]["session_id"] is None
    assert row["title"] == "cleanup job-1 finished: cancelled"


def test_emitter_unwired_writes_nothing(monkeypatch):
    monkeypatch.setattr(platform_notifications, "_EMITTER", None)
    with patch.object(platform_notifications, "notify") as notify:
        assert job_events.emit_job_terminal_event(_job(), "failed", None, "x") is False
    notify.assert_not_called()


def test_install_registers_the_emitter_once(monkeypatch):
    from umrs.jobs import store as store_mod

    monkeypatch.setattr(store_mod, "_PROCESS_TERMINAL_LISTENERS", [])
    job_events.install_job_event_listener()
    job_events.install_job_event_listener()
    assert [job_events.emit_job_terminal_event] == store_mod._PROCESS_TERMINAL_LISTENERS


def test_service_boot_installs_the_emitter():
    import inspect

    from umrs.jobs import service

    assert "install_job_event_listener()" in inspect.getsource(service.JobService.startup)


def test_sidecar_job_store_init_installs_the_emitter(monkeypatch):
    from umrs.jobs import store as store_mod
    from umrs.server import http_sidecar

    monkeypatch.setattr(store_mod, "_PROCESS_TERMINAL_LISTENERS", [])
    monkeypatch.setattr(JobStore, "_ensure_schema", lambda _self: None)
    monkeypatch.setattr("umrs.api.jobs._job_store", None)
    http_sidecar._init_job_store(MagicMock())
    assert [job_events.emit_job_terminal_event] == store_mod._PROCESS_TERMINAL_LISTENERS


# ── NotificationStore cursor read ───────────────────────────────────────────


def _unlike(pattern: str) -> str:
    return pattern[:-1].replace("\\_", "_").replace("\\%", "%").replace("\\\\", "\\")


class _NCursor:
    def __init__(self, conn):
        self._conn = conn
        self._result: list = []

    def execute(self, sql, params=()):
        self._conn.sql.append((sql, params))
        if "MAX(id)" in sql:
            user, project, pattern = params
            ids = [r["id"] for r in self._conn.match(user, project, _unlike(pattern))]
            self._result = [(max(ids, default=0),)]
            return
        user, project, after, pattern, *rest = params
        sid = rest[0] if len(rest) == 2 else None
        rows = [
            r
            for r in self._conn.match(user, project, _unlike(pattern))
            if r["id"] > after and (sid is None or r["detail"].get("session_id") == sid)
        ]
        self._result = [self._conn.as_tuple(r) for r in sorted(rows, key=lambda r: r["id"])][
            : rest[-1]
        ]

    def fetchone(self):
        return self._result[0] if self._result else None

    def fetchall(self):
        return list(self._result)


class _NConn:
    def __init__(self, rows):
        self.rows = rows
        self.sql: list = []
        self.rolled_back = 0

    def cursor(self):
        return _NCursor(self)

    def rollback(self):
        self.rolled_back += 1

    def match(self, user, project, prefix):
        return [
            r
            for r in self.rows
            if r["user_id"] == user and r["project_id"] == project and r["kind"].startswith(prefix)
        ]

    @staticmethod
    def as_tuple(r):
        return (
            r["id"],
            r["kind"],
            r["title"],
            r["detail"],
            r["user_id"],
            r["project_id"],
            "t",
            None,
        )


def _row(i, kind="job.succeeded", user="alice", project="proj-a", session="s1"):
    return {
        "id": i,
        "kind": kind,
        "title": f"t{i}",
        "detail": {"job_id": f"job-{i}", "session_id": session},
        "user_id": user,
        "project_id": project,
    }


ROWS = [
    _row(3),
    _row(1),
    _row(2, session="s2"),
    _row(4, kind="budget.stop"),
    _row(5, user="bob"),
    _row(6, project="proj-b"),
    _row(7, kind="job.failed"),
]


def test_list_after_ascending_scoped_and_prefixed():
    store = NotificationStore(_NConn(ROWS))
    got = store.list_after(user_id="alice", project_id="proj-a", after=0)
    assert [r["id"] for r in got] == [1, 2, 3, 7]
    got = store.list_after(user_id="alice", project_id="proj-a", after=2)
    assert [r["id"] for r in got] == [3, 7]


def test_list_after_session_and_kind_filters():
    store = NotificationStore(_NConn(ROWS))
    got = store.list_after(user_id="alice", project_id="proj-a", after=0, session_id="s2")
    assert [r["id"] for r in got] == [2]
    got = store.list_after(user_id="alice", project_id="proj-a", after=0, kind_prefix="job.failed")
    assert [r["id"] for r in got] == [7]


def test_list_after_caps_the_limit():
    conn = _NConn(ROWS)
    NotificationStore(conn).list_after(user_id="alice", project_id="proj-a", after=0, limit=10_000)
    sql, params = conn.sql[-1]
    assert params[-1] == 100 and "ORDER BY id ASC" in sql


def test_like_prefix_escapes_wildcards():
    assert platform_notifications.like_prefix("job_x%") == "job\\_x\\%%"


def test_max_id_is_the_scope_newest_or_zero():
    store = NotificationStore(_NConn(ROWS))
    assert store.max_id(user_id="alice", project_id="proj-a") == 7
    assert store.max_id(user_id="carol", project_id="proj-a") == 0


def test_dashboard_list_can_leave_job_events_out():
    captured = MagicMock()
    captured.fetchall.return_value = []
    conn = MagicMock()
    conn.cursor.return_value = captured
    NotificationStore(conn).list(project_id="p", unread_only=True, exclude_kind_prefix="job.")
    sql, params = captured.execute.call_args.args
    assert "kind NOT LIKE %s" in sql and "job.%" in params


# ── GET /umrs/jobs/events ───────────────────────────────────────────────────


def _signed(user: str, project: str = "proj-a", **extra) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
        **extra,
    }


class _Pool:
    def __init__(self, rows):
        self.conn = _NConn(rows)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        pass


class _CreateStore:
    _get_conn = None  # no catalog autopin
    site_name = ""

    def __init__(self):
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    def enqueue_job(self, _job_id):
        pass

    def get_job(self, _job_id):
        raise AssertionError("/events must not reach GET /{job_id}")


@pytest.fixture
def app(monkeypatch):
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    application = FastAPI()
    application.include_router(jobs_router)
    store = _CreateStore()
    application.dependency_overrides[get_job_store] = lambda: store
    application.state.umrs_db_pool = _Pool(ROWS)
    application.state.create_store = store
    return application


def test_events_route_returns_the_callers_page(app):
    r = TestClient(app).get("/umrs/jobs/events?after=1&limit=2", headers=_signed("alice"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["id"] for e in body["events"]] == [2, 3]
    assert body["cursor"] == 3
    first = body["events"][0]
    assert first["kind"] == "job.succeeded" and first["project_id"] == "proj-a"
    assert first["detail"]["job_id"] == "job-2"
    assert app.state.umrs_db_pool.conn.rolled_back == 1


def test_events_route_never_shows_another_users_rows(app):
    r = TestClient(app).get("/umrs/jobs/events", headers=_signed("bob"))
    assert [e["id"] for e in r.json()["events"]] == [5]
    r = TestClient(app).get("/umrs/jobs/events", headers=_signed("alice", "proj-b"))
    assert [e["id"] for e in r.json()["events"]] == [6]


def test_events_route_session_filter_and_empty_cursor(app):
    client = TestClient(app)
    r = client.get("/umrs/jobs/events?session_id=s2", headers=_signed("alice"))
    assert [e["id"] for e in r.json()["events"]] == [2]
    r = client.get("/umrs/jobs/events?after=7", headers=_signed("alice"))
    assert r.json() == {"events": [], "cursor": 7}


def test_events_route_tail_starts_at_now(app):
    r = TestClient(app).get("/umrs/jobs/events?tail=true", headers=_signed("alice"))
    assert r.json() == {"events": [], "cursor": 7}


@pytest.mark.parametrize(
    "query", ["kind_prefix=budget.", "session_id=bad%20id", "limit=0", "limit=101", "after=-1"]
)
def test_events_route_refuses_bad_filters(app, query):
    r = TestClient(app).get(f"/umrs/jobs/events?{query}", headers=_signed("alice"))
    assert r.status_code in (400, 422), r.text


def test_events_route_requires_verified_identity(app):
    r = TestClient(app).get("/umrs/jobs/events")
    assert r.status_code == 401


def test_events_route_is_not_swallowed_by_job_id(app):
    from umrs.api.jobs import router as jobs_router

    paths = [route.path for route in jobs_router.routes if "GET" in route.methods]
    assert paths.index("/umrs/jobs/events") < paths.index("/umrs/jobs/{job_id}")
    op = next(r for r in jobs_router.routes if r.path == "/umrs/jobs/events")
    assert op.operation_id == "job_events"


def test_events_route_unwired_answers_empty(app):
    app.state.umrs_db_pool = None
    r = TestClient(app).get("/umrs/jobs/events?after=4", headers=_signed("alice"))
    assert r.status_code == 200 and r.json() == {"events": [], "cursor": 4}


def test_events_route_store_fault_is_a_typed_503(app):
    class _Broken:
        def getconn(self):
            raise RuntimeError("db down")

    app.state.umrs_db_pool = _Broken()
    r = TestClient(app).get("/umrs/jobs/events", headers=_signed("alice"))
    assert r.status_code == 503


# ── origin stamping (x-r4-session-id) ───────────────────────────────────────


def _create_body(spec=None) -> dict:
    return {"project_id": "proj-a", "user_id": "alice", "job_type": "cleanup", "spec": spec or {}}


def test_create_stamps_the_session_header(app):
    client = TestClient(app)
    headers = _signed("alice", **{"x-r4-session-id": "sess-42"})
    r = client.post("/umrs/jobs", json=_create_body({"origin": {"venue": "v"}}), headers=headers)
    assert r.status_code == 201, r.text
    spec = app.state.create_store.created[0]["spec"]
    assert spec["origin"] == {"venue": "v", "session_id": "sess-42"}


@pytest.mark.parametrize("value", [None, "bad id", "x" * 129])
def test_create_without_a_valid_header_leaves_spec_alone(app, value):
    headers = _signed("alice")
    if value is not None:
        headers["x-r4-session-id"] = value
    r = TestClient(app).post("/umrs/jobs", json=_create_body(), headers=headers)
    assert r.status_code == 201, r.text
    assert "origin" not in app.state.create_store.created[0]["spec"]


@pytest.fixture
def workflow_client(monkeypatch, tmp_path):
    from test_workflow_durable_jobs import FakeJobStore
    from umrs.api.workflows import router as workflows_router
    from umrs.server.workflow_execution_registry import reset_execution_registry_for_tests
    from umrs.server.workflow_registry_handler import register_workflow

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    reset_execution_registry_for_tests()
    workflow_dir = tmp_path / "relax"
    workflow_dir.mkdir()
    workflow_json = {"name": "relax", "version": "1.0", "steps": []}
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow_json))
    store = FakeJobStore()
    with (
        patch(
            "umrs.server.workflow_registry_handler._catalog_path",
            return_value=str(tmp_path / "workflow_catalog.json"),
        ),
        patch("umrs.api.workflows.get_optional_job_store", return_value=store),
    ):
        register_workflow(
            workflow_name="relax",
            version="1.0",
            user_id="alice",
            source_path=str(workflow_dir),
            workflow_json=workflow_json,
        )
        application = FastAPI()
        application.include_router(workflows_router)
        yield TestClient(application), store


def test_trigger_stamps_the_durable_workflow_run(workflow_client):
    client, store = workflow_client
    headers = _signed("alice", **{"x-r4-session-id": "sess-7"})
    r = client.post("/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=headers)
    assert r.status_code == 200, r.text
    (job,) = store.jobs.values()
    assert job.job_type == JobType.WORKFLOW_RUN
    assert job.spec["origin"] == {"session_id": "sess-7"}


def test_trigger_without_header_keeps_spec(workflow_client):
    client, store = workflow_client
    r = client.post(
        "/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=_signed("alice")
    )
    assert r.status_code == 200, r.text
    (job,) = store.jobs.values()
    assert "origin" not in job.spec


# ── migration 0130: the cursor read's index ────────────────────────────────


def test_0130_indexes_the_cursor_read_and_follows_0129():
    from umrs import migrations

    path = migrations.MIGRATIONS_DIR / "0130_platform_notifications_cursor_index.sql"
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0130") == 1
    assert versions.index("0130") == versions.index("0129") + 1
    code = " ".join(
        line for line in path.read_text().splitlines() if not line.lstrip().startswith("--")
    )
    assert "CREATE INDEX IF NOT EXISTS platform_notifications_user_project_id_idx" in code
    assert "ON platform_notifications (user_id, project_id, id)" in code
    assert "WHERE" not in code  # not partial: job.* rows are never read_at-stamped
