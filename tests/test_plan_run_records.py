"""A team plan run's durable record in UMRS (ADR-092 slice 7, MD-4).

The run is one client-hosted ``research_loop_job`` (no new table, one job
store); each portable journal record is one ``execution_traces`` row keyed
by its D-8 content key, its JSON text kept verbatim. Routes are the traces
family's, the caller's own run only. The store's attempt state machine is
faked here (as the other route tests do); the reopen SQL and the strict
trace insert are pinned against a fake connection.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import agent_quota
from umrs.api import jobs as jobs_api
from umrs.api import traces as traces_api
from umrs.jobs import plan_run_records as prr
from umrs.jobs.execution_trace_store import ExecutionTraceStore
from umrs.jobs.models import AttemptStatus, JobStatus, JobType, RestartPolicy
from umrs.jobs.research_loop_backend import (
    ResearchLoopStatusChecker,
    validate_research_loop_spec,
)
from umrs.jobs.store import (
    AttemptNotFoundError,
    DuplicateJobError,
    InvalidTransitionError,
    JobNotFoundError,
    JobStore,
)

RUN = "review-0a1b2c3d"
KEY = "a" * 64


class _Caller:
    def __init__(self, user_id="alice", project_id="proj-x", is_admin=False):
        self.user_id = user_id
        self.project_id = project_id
        self.is_admin = is_admin

    def can_access_job(self, job):
        return self.is_admin or (job.project_id == self.project_id and job.user_id == self.user_id)

    @property
    def is_anonymous(self):
        return not self.project_id


class _FakeJobStore:
    """The JobStore surface the plan-run records use, with its transitions."""

    def __init__(self):
        self.jobs: dict[str, SimpleNamespace] = {}
        self.attempts: dict[str, SimpleNamespace] = {}
        self.calls: list[tuple] = []

    # jobs
    def get_job_by_idempotency_key(self, project_id, key):
        for job in self.jobs.values():
            if job.project_id == project_id and job.idempotency_key == key:
                return job
        return None

    def create_job(self, project_id, user_id, job_type, spec, **kw):
        if self.get_job_by_idempotency_key(project_id, kw["idempotency_key"]):
            raise DuplicateJobError("dup")
        job_id = f"job-{len(self.jobs) + 1}"
        self.jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=json.loads(json.dumps(spec)),
            idempotency_key=kw["idempotency_key"],
            restart_policy=kw["restart_policy"],
            max_attempts=kw["max_attempts"],
            status=JobStatus.PLANNED,
            active_attempt_id=None,
            commit_attempt_id=None,
        )
        self.calls.append(("create_job", job_id))
        return job_id

    def get_job(self, job_id):
        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        return self.jobs[job_id]

    def reopen_client_job(self, job_id):
        job = self.jobs[job_id]
        if not prr.is_client_hosted(job.spec) or job.status == JobStatus.CANCELLED:
            raise InvalidTransitionError("not reopenable")
        superseded = []
        for att in self.attempts.values():
            if att.job_id == job_id and att.status in (
                AttemptStatus.CREATED,
                AttemptStatus.SUBMITTED,
                AttemptStatus.RUNNING,
            ):
                att.status = AttemptStatus.LOST
                superseded.append(att.attempt_id)
        job.status = JobStatus.PLANNED
        job.active_attempt_id = None
        return superseded

    # attempts
    def create_attempt(self, job_id, backend):
        job = self.jobs[job_id]
        if job.status not in (JobStatus.PLANNED, JobStatus.QUEUED):
            raise InvalidTransitionError("not planned")
        no = 1 + sum(a.job_id == job_id for a in self.attempts.values())
        attempt_id = f"att-{job_id}-{no}"
        self.attempts[attempt_id] = SimpleNamespace(
            attempt_id=attempt_id,
            job_id=job_id,
            attempt_no=no,
            backend=backend,
            backend_ref=None,
            status=AttemptStatus.CREATED,
            exit_reason=None,
            metrics={},
        )
        job.status = JobStatus.RUNNING
        job.active_attempt_id = attempt_id
        return attempt_id

    def mark_attempt_submitted(self, attempt_id, backend_ref):
        att = self.attempts[attempt_id]
        assert att.status == AttemptStatus.CREATED
        att.status, att.backend_ref = AttemptStatus.SUBMITTED, backend_ref

    def mark_attempt_running(self, attempt_id, status_detail=None):
        att = self.attempts[attempt_id]
        assert att.status == AttemptStatus.SUBMITTED
        att.status, att.status_detail = AttemptStatus.RUNNING, status_detail

    def get_attempt(self, attempt_id):
        if attempt_id not in self.attempts:
            raise AttemptNotFoundError(attempt_id)
        return self.attempts[attempt_id]

    def get_attempts_for_job(self, job_id):
        return [a for a in self.attempts.values() if a.job_id == job_id]

    def mark_attempt_succeeded(self, attempt_id, commit=True, metrics=None):
        att = self.attempts[attempt_id]
        att.status, att.metrics = AttemptStatus.SUCCEEDED, dict(metrics or {})
        job = self.jobs[att.job_id]
        job.status, job.active_attempt_id = JobStatus.COMPLETED, None
        if commit:
            job.commit_attempt_id = attempt_id

    def mark_attempt_failed(self, attempt_id, exit_reason, retry_delay=None):
        assert retry_delay is None  # NEVER retried
        att = self.attempts[attempt_id]
        att.status, att.exit_reason = AttemptStatus.FAILED, exit_reason
        job = self.jobs[att.job_id]
        job.status, job.active_attempt_id = JobStatus.FAILED, None
        return False

    def update_attempt_metrics(self, attempt_id, metrics):
        self.attempts[attempt_id].metrics.update(metrics)

    # test helpers
    def umrs_restart(self):
        """The reconciler on boot: active attempts LOST, NEVER jobs FAILED."""
        for att in self.attempts.values():
            if att.status in (AttemptStatus.SUBMITTED, AttemptStatus.RUNNING):
                att.status = AttemptStatus.LOST
                job = self.jobs[att.job_id]
                job.status, job.active_attempt_id = JobStatus.FAILED, None


def _app(store, traces, caller):
    app = FastAPI()
    app.include_router(traces_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    app.dependency_overrides[traces_api.get_trace_store] = lambda: traces
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return app


@pytest.fixture()
def store():
    return _FakeJobStore()


@pytest.fixture()
def traces():
    return ExecutionTraceStore(db_url=None)


@pytest.fixture()
def client(store, traces):
    return TestClient(_app(store, traces, _Caller()))


DEFINITION = {"plan": {"name": "review"}, "params": {"x": 1}, "manifest": {"engine": "plan"}}


def _open(client, attempt=1, run=RUN, **body):
    payload = {"attempt": attempt, "definition": DEFINITION, **body}
    return client.post(f"/umrs/traces/plan-runs/{run}/attempts", json=payload)


def _record(key=KEY, ok=True, **extra):
    # Keys deliberately NOT sorted: the text must come back byte-identical.
    record = {
        "call_key": key,
        "label": "finder",
        "step": "find",
        "phase": "p1",
        "node": "find",
        "ok": ok,
        "output": {"zeta": 1, "alpha": [2, 1], "mid": "x"},
        "reason": "" if ok else "schema_invalid: nope",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "model": "llama3.2:3b",
        "attempt": "attempt-1",
        "retrieval": [{"query": "q", "source_id": "s1"}],
    }
    record.update(extra)
    return json.dumps(record, ensure_ascii=False)


def _append(client, attempt_id, text, run=RUN):
    return client.post(
        f"/umrs/traces/plan-runs/{run}/records",
        json={"attempt_id": attempt_id, "record": text},
    )


class TestOpenAttempt:
    def test_first_open_creates_one_client_hosted_research_loop_job(self, client, store):
        res = _open(client, required=True)
        assert res.status_code == 201, res.text
        body = res.json()
        job = store.jobs[body["job_id"]]
        attempt = store.attempts[body["attempt_id"]]
        assert {
            "created": body["created"],
            "type": job.job_type,
            "hosting": job.spec["hosting"],
            "required": job.spec["durable_required"],
            "definition": job.spec["definition"],
            "policy": job.restart_policy,
            "attempt": (attempt.status, attempt.backend, attempt.backend_ref),
            "active": job.active_attempt_id,
        } == {
            "created": True,
            "type": JobType.RESEARCH_LOOP,
            "hosting": "client",
            "required": True,
            "definition": DEFINITION,
            "policy": RestartPolicy.NEVER,
            "attempt": (
                AttemptStatus.RUNNING,
                "r4_local",
                f"research_loop_job|plan_run={RUN}|driver=client|attempt=1",
            ),
            "active": body["attempt_id"],
        }

    def test_a_resume_supersedes_the_active_attempt_and_reopens_the_job(self, client, store):
        first = _open(client).json()
        second = _open(client, attempt=2).json()
        assert second["created"] is False
        assert second["job_id"] == first["job_id"]
        assert second["superseded"] == [first["attempt_id"]]
        assert store.attempts[first["attempt_id"]].status == AttemptStatus.LOST
        assert store.jobs[first["job_id"]].active_attempt_id == second["attempt_id"]
        assert second["attempt_no"] == 2

    def test_a_finished_or_failed_run_reopens_for_its_next_attempt(self, client, store):
        first = _open(client).json()
        client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts/{first['attempt_id']}/finish",
            json={"status": "failed", "error": "boom"},
        )
        assert store.jobs[first["job_id"]].status == JobStatus.FAILED
        second = _open(client, attempt=2)
        assert second.status_code == 201
        assert store.jobs[first["job_id"]].status == JobStatus.RUNNING

    def test_the_definition_is_kept_from_the_first_open_only(self, client, store):
        first = _open(client).json()
        _open(client, attempt=2, definition={"plan": {"name": "other"}})
        assert store.jobs[first["job_id"]].spec["definition"] == DEFINITION

    def test_a_cancelled_record_is_never_reopened(self, client, store):
        first = _open(client).json()
        store.jobs[first["job_id"]].status = JobStatus.CANCELLED
        res = _open(client, attempt=2)
        assert res.status_code == 409
        assert "plan_run_cancelled" in res.json()["detail"]

    @pytest.mark.parametrize("run", ["../x", "a|b", "a=b", "-lead", "x" * 129])
    def test_unsafe_run_ids_are_refused(self, client, run):
        res = client.post(
            f"/umrs/traces/plan-runs/{run}/attempts", json={"attempt": 1, "definition": {}}
        )
        assert res.status_code in (400, 404), res.text

    def test_another_user_never_reaches_the_run(self, store, traces):
        _open(TestClient(_app(store, traces, _Caller())))
        bob = TestClient(_app(store, traces, _Caller(user_id="bob")))
        assert bob.get(f"/umrs/traces/plan-runs/{RUN}/records").status_code == 404
        # bob's own open makes bob's own job: never alice's
        assert _open(bob).json()["created"] is True
        assert len(store.jobs) == 2

    def test_an_oversized_definition_is_refused(self, client):
        res = _open(client, definition={"blob": "x" * (prr.MAX_DEFINITION_BYTES + 1)})
        assert res.status_code == 400
        assert "definition_too_large" in res.json()["detail"]


class TestRecords:
    def test_a_record_is_one_trace_row_keyed_by_its_d8_key(self, client, traces):
        opened = _open(client).json()
        text = _record()
        res = _append(client, opened["attempt_id"], text)
        assert res.status_code == 201, res.text
        (row,) = traces.get_traces_for_job(opened["job_id"])
        assert {
            "hash": row.inputs_hash,
            "attempt": row.attempt_id,
            "type": row.step_type,
            "status": row.status,
            "step": row.step_name,
            "model": row.model_ref,
            "retrieval": row.retrieval_refs,
            "snapshot": row.outputs_snapshot,
        } == {
            "hash": KEY,
            "attempt": opened["attempt_id"],
            "type": "plan_agent",
            "status": "succeeded",
            "step": "find",
            "model": {"name": "llama3.2:3b"},
            "retrieval": [{"query": "q", "source_id": "s1"}],
            "snapshot": {"format": prr.RECORD_FORMAT, "record": text},
        }

    def test_records_come_back_verbatim_in_write_order(self, client):
        opened = _open(client).json()
        texts = [_record(), _record(key="b" * 64, ok=False), _record(key="g1", kind="gate")]
        for text in texts:
            assert _append(client, opened["attempt_id"], text).status_code == 201
        body = client.get(f"/umrs/traces/plan-runs/{RUN}/records").json()
        assert [r["record"] for r in body["records"]] == texts
        assert [r["status"] for r in body["records"]] == ["succeeded", "failed", "succeeded"]
        assert body["count"] == 3

    def test_a_null_names_its_reason(self, client, traces):
        opened = _open(client).json()
        _append(client, opened["attempt_id"], _record(ok=False))
        (row,) = traces.get_traces_for_job(opened["job_id"])
        assert row.error == "schema_invalid: nope"

    def test_a_write_to_a_superseded_attempt_is_refused(self, client):
        first = _open(client).json()
        _open(client, attempt=2)
        res = _append(client, first["attempt_id"], _record())
        assert res.status_code == 409
        assert "attempt_not_active" in res.json()["detail"]

    def test_a_write_after_a_umrs_restart_is_refused(self, client, store):
        opened = _open(client).json()
        store.umrs_restart()
        res = _append(client, opened["attempt_id"], _record())
        assert res.status_code == 409
        assert "attempt_not_active" in res.json()["detail"]

    def test_an_unknown_attempt_is_refused(self, client):
        _open(client)
        assert _append(client, "att-nope", _record()).status_code == 404

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            json.dumps([1]),
            json.dumps({"ok": True}),
            json.dumps({"call_key": "k" * 65}),
            json.dumps({"call_key": "workspace:w1", "kind": "workspace"}),
            json.dumps({"call_key": "m", "kind": "merge_back"}),
        ],
    )
    def test_malformed_or_venue_local_records_are_refused(self, client, text):
        opened = _open(client).json()
        assert _append(client, opened["attempt_id"], text).status_code == 400

    def test_an_oversized_record_is_refused_by_name(self, client):
        opened = _open(client).json()
        text = _record(output="x" * prr.MAX_RECORD_BYTES)
        res = _append(client, opened["attempt_id"], text)
        assert res.status_code == 400
        assert "record_too_large" in res.json()["detail"]

    def test_a_store_failure_is_named_never_acknowledged(self, client, traces, monkeypatch):
        opened = _open(client).json()

        def boom(**_kw):
            raise RuntimeError("db down")

        monkeypatch.setattr(traces, "record_completed", boom)
        res = _append(client, opened["attempt_id"], _record())
        assert res.status_code == 503
        assert "durable record write failed" in res.json()["detail"]

    def test_an_unknown_run_is_404(self, client):
        assert client.get("/umrs/traces/plan-runs/none-00000000/records").status_code == 404
        assert client.get("/umrs/traces/plan-runs/none-00000000").status_code == 404


class TestRunAndFinish:
    def test_the_run_view_carries_definition_and_r4_attempt_labels(self, client):
        _open(client)
        _open(client, attempt=3)
        view = client.get(f"/umrs/traces/plan-runs/{RUN}").json()
        assert view["definition"] == DEFINITION
        assert view["job_type"] == "research_loop_job"
        assert [(a["label"], a["status"]) for a in view["attempts"]] == [
            (1, "lost"),
            (3, "running"),
        ]

    def test_success_commits_the_attempt(self, client, store):
        opened = _open(client).json()
        res = client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts/{opened['attempt_id']}/finish",
            json={"status": "success", "summary": {"agents_ok": 3}},
        )
        assert res.status_code == 200, res.text
        assert res.json()["job_status"] == "completed"
        job = store.jobs[opened["job_id"]]
        assert job.commit_attempt_id == opened["attempt_id"]
        assert store.attempts[opened["attempt_id"]].metrics["plan_status"] == "success"

    def test_a_cancelled_attempt_fails_with_its_reason_and_never_commits(self, client, store):
        opened = _open(client).json()
        client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts/{opened['attempt_id']}/finish",
            json={"status": "cancelled", "error": "user cancel"},
        )
        attempt = store.attempts[opened["attempt_id"]]
        assert attempt.status == AttemptStatus.FAILED
        assert attempt.exit_reason == "plan_cancelled: user cancel"
        assert store.jobs[opened["job_id"]].commit_attempt_id is None

    def test_finish_twice_or_with_an_unknown_status_is_refused(self, client):
        opened = _open(client).json()
        url = f"/umrs/traces/plan-runs/{RUN}/attempts/{opened['attempt_id']}/finish"
        assert client.post(url, json={"status": "weird"}).status_code == 400
        assert client.post(url, json={"status": "success"}).status_code == 200
        assert client.post(url, json={"status": "success"}).status_code == 409


class TestTheKindStaysClientHosted:
    def test_the_backend_never_hosts_a_client_hosted_spec(self):
        with pytest.raises(ValueError, match="client_hosted_run"):
            validate_research_loop_spec({"hosting": "client", "plan_run_id": RUN})

    def test_the_checker_never_fails_a_client_hosted_attempt(self):
        attempt = SimpleNamespace(backend_ref="research_loop_job|plan_run=x|driver=client")
        job = SimpleNamespace(spec={"hosting": "client"})
        result = asyncio.run(ResearchLoopStatusChecker("me").check_status(attempt, job))
        assert result.status == "running"

    def test_a_server_hosted_loop_is_still_checked_as_before(self):
        attempt = SimpleNamespace(backend_ref="bogus", umrs_instance_id=None)
        result = asyncio.run(
            ResearchLoopStatusChecker("me").check_status(attempt, SimpleNamespace(spec={}))
        )
        assert result.status == "failed"

    def test_client_hosted_runs_never_count_against_the_agent_process_budget(self):
        seen = []

        class _Cur:
            def execute(self, sql, params):
                seen.append((sql, params))

            def fetchone(self):
                return (0,)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class _Conn:
            def cursor(self):
                return _Cur()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        pool = SimpleNamespace(getconn=lambda: _Conn(), putconn=lambda _c: None)
        agent_quota.count_active_agent_units(pool, user_id="alice")
        sql, params = seen[0]
        assert "spec->>'hosting'" in sql and "<> ALL(%s)" in sql
        # slice 8: a venue-hosted run is not a unit either (its worker's
        # frontend_session job is)
        assert params[-1] == ["client", "venue"]


class _SqlCur:
    def __init__(self, log, rows):
        self.log = log
        self.rows = rows

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        return [("att-old",)]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _SqlConn:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows
        self.committed = self.rolled_back = False

    def cursor(self):
        return _SqlCur(self.log, self.rows)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _sql_store(rows):
    log: list = []
    conn = _SqlConn(log, rows)
    store = JobStore.__new__(JobStore)
    store.db_pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    return store, conn, log


def _first_sql(log, prefix):
    """The first (sql, params) the store executed that starts with ``prefix``."""
    return next(entry for entry in log if entry[0].startswith(prefix))


class TestReopenSql:
    def test_reopen_is_client_hosted_only_never_cancelled_never_queued(self):
        # advisory lock taken, then the job UPDATE matched one row
        store, conn, log = _sql_store([(True,), ("job-1",)])
        assert store.reopen_client_job("job-1") == ["att-old"]
        sql, params = _first_sql(log, "UPDATE jobs")
        _lost_sql, lost_params = _first_sql(log, "UPDATE job_attempts")
        assert {
            # slice 8: venue too
            "hosting clause": "spec->>'hosting' IN ('client', 'venue')" in sql,
            "job to": params[0],
            "cancelled in params": JobStatus.CANCELLED.value in params,
            "queued in params": JobStatus.QUEUED.value in params,  # taken OUT of the queue
            "attempt to": lost_params[0],
            "committed": conn.committed,
        } == {
            "hosting clause": True,
            "job to": JobStatus.PLANNED.value,
            "cancelled in params": False,
            "queued in params": True,
            "attempt to": AttemptStatus.LOST.value,
            "committed": True,
        }

    def test_reopen_of_another_job_raises_and_changes_nothing(self):
        store, conn, log = _sql_store([(True,), None])
        with pytest.raises(InvalidTransitionError):
            store.reopen_client_job("job-1")
        assert conn.rolled_back and not conn.committed
        assert not any(sql.startswith("UPDATE job_attempts") for sql, _ in log)


class _TraceConn:
    def __init__(self, log, fail=False):
        self.log, self.fail = log, fail

    def cursor(self):
        conn = self

        class _C:
            def execute(self, sql, params):
                if conn.fail:
                    raise RuntimeError("insert failed")
                conn.log.append((sql, params))

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return _C()

    def commit(self):
        pass

    def close(self):
        pass


class TestStrictTraceInsert:
    def _store(self, monkeypatch, fail=False):
        log: list = []
        store = ExecutionTraceStore(db_url=None)
        monkeypatch.setattr(ExecutionTraceStore, "db_enabled", property(lambda _s: True))
        monkeypatch.setattr(store, "_get_conn", lambda: _TraceConn(log, fail))
        return store, log

    def _write(self, store):
        return store.record_completed(
            job_id="job-1",
            attempt_id="att-1",
            step_name="find",
            step_type="plan_agent",
            inputs_hash=KEY,
            status="succeeded",
            outputs_snapshot={"format": prr.RECORD_FORMAT, "record": _record()},
            outputs_hash="h",
        )

    def test_one_insert_with_the_callers_key_and_the_text_as_a_json_string(self, monkeypatch):
        store, log = self._store(monkeypatch)
        self._write(store)
        ((sql, params),) = log
        assert "INSERT INTO execution_traces" in sql and "outputs_snapshot" in sql
        assert KEY in params
        snapshot = json.loads(next(p for p in params if isinstance(p, str) and "format" in p))
        assert snapshot["record"] == _record()

    def test_a_database_failure_raises_never_falls_back_to_memory(self, monkeypatch):
        store, _log = self._store(monkeypatch, fail=True)
        with pytest.raises(RuntimeError):
            self._write(store)
        assert store._traces == {}
