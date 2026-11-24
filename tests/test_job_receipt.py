"""ADR-105 D-10 (A105-S7, 2026-11-24): job receipts and step-level ingest.

``GET /umrs/jobs/{id}/receipt`` joins, at read time, the job, its commit
attempt's stored outputs, its artifact-registry and a2l rows, and the
knowledge ingest it chained. Before this slice the route did not exist (404
for every id) and every stored output of a run was ingested whatever its
steps declared.

The Postgres case runs against ``UMRS_TEST_DB_URL`` in a scratch schema
(skipped without it); the rest is hermetic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import jobs as jobs_api
from umrs.api.jobs import CallerIdentity
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs import workflow_run as wr
from umrs.jobs.models import AttemptStatus, Job, JobAttempt, JobStatus, JobType
from umrs.jobs.store import JobNotFoundError
from umrs.server.workflow_runner import _stamp_ingest_selection
from umrs.shared.step_ingest import ingest_selection

JOB_ID = "job-receipt000001"
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)
ALICE = CallerIdentity(user_id="alice", project_id="pa")
BOB = CallerIdentity(user_id="bob", project_id="pb")
ALICE_ELSEWHERE = CallerIdentity(user_id="alice", project_id="pb")
ADMIN = CallerIdentity(user_id="root", project_id="ops", is_admin=True)

OUTPUTS = [
    {
        "artifact_id": "wfart-aaaa",
        "content_hash": "a" * 64,
        "kind": "s3_uri",
        "source_ref": "s3://b/jobs/job-receipt000001/attempt_1/results/t.csv",
        "name": "results/t.csv",
        "size_bytes": 12,
    },
    {
        "artifact_id": "wfart-bbbb",
        "content_hash": "b" * 64,
        "kind": "s3_uri",
        "source_ref": "s3://b/jobs/job-receipt000001/attempt_1/scratch/x.bin",
        "name": "scratch/x.bin",
        "size_bytes": 3,
        "ingest": False,
    },
]


def _job() -> Job:
    return Job(
        job_id=JOB_ID,
        project_id="pa",
        user_id="alice",
        job_type=JobType.WORKFLOW_RUN,
        spec={"execution_id": "wf-1", "workflow_name": "demo", "workflow_version": "1.2.0"},
        idempotency_key="k",
        status=JobStatus.COMPLETED,
        commit_attempt_id="att-2",
    )


def _attempt(attempt_id: str, no: int, metrics: dict) -> JobAttempt:
    return JobAttempt(
        attempt_id=attempt_id,
        job_id=JOB_ID,
        attempt_no=no,
        backend="workflow_run",
        status=AttemptStatus.SUCCEEDED,
        ended_at=NOW,
        metrics=metrics,
    )


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.rows: list[tuple] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.conn.executed.append((sql, params))
        table = sql.split(" FROM ", 1)[1].split()[0]
        if table in self.conn.broken:
            raise RuntimeError(f"relation {table} does not exist")
        self.rows = self.conn.tables.get(table, [])

    def fetchall(self) -> list[tuple]:
        return self.rows

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: Any) -> None:
        return None


@dataclass
class _Conn:
    tables: dict[str, list[tuple]]
    broken: set[str] = field(default_factory=set)
    executed: list[tuple] = field(default_factory=list)
    rollbacks: int = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def rollback(self) -> None:
        self.rollbacks += 1


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.returned = 0

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, _conn: _Conn) -> None:
        self.returned += 1


def _tables() -> dict[str, list[tuple]]:
    return {
        "artifacts": [("0b6c…-uuid", "results/t.csv", "s3://b/k/t.csv", 12, "a" * 64)],
        "a2l_artifacts": [("a2l-1", "result", "s3://b/k/t.csv", "completed")],
        "jobs": [
            ("job-ingest01", "knowledge_ingest", "succeeded"),
            ("job-underst1", "knowledge_understand", "running"),
        ],
        "knowledge_artifacts": [("wfart-aaaa",)],
    }


class _Store:
    def __init__(self, conn: _Conn | None = None) -> None:
        self.db_pool = _Pool(conn or _Conn(_tables()))

    def get_job(self, job_id: str) -> Job:
        if job_id != JOB_ID:
            raise JobNotFoundError(job_id)
        return _job()

    def get_attempts_for_job(self, _job_id: str) -> list[JobAttempt]:
        return [
            _attempt("att-1", 1, {"artifacts": [{"artifact_id": "stale"}]}),
            _attempt("att-2", 2, {"artifacts": OUTPUTS}),
        ]


def _client(store: Any, caller: CallerIdentity) -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return TestClient(app)


def _receipt(store: Any, caller: CallerIdentity = ALICE, job_id: str = JOB_ID) -> Any:
    return _client(store, caller).get(f"/umrs/jobs/{job_id}/receipt")


# ── the route ──────────────────────────────────────────────────────────────


def _col(rows: list[dict[str, Any]], key: str) -> list[Any]:
    return [row[key] for row in rows]


def _sections(body: dict[str, Any]) -> dict[str, list[Any]]:
    """Each joined section by its identifying column."""
    return {
        "artifacts": _col(body["artifacts"], "name"),
        "a2l_artifacts": _col(body["a2l_artifacts"], "id"),
        "knowledge_jobs": _col(body["knowledge_jobs"], "job_id"),
        "knowledge_artifact_ids": body["knowledge_artifact_ids"],
        "warnings": body["warnings"],
    }


def test_receipt_names_the_job_and_its_commit_attempt() -> None:
    resp = _receipt(_Store())
    assert resp.status_code == 200, resp.text  # 404 before A105-S7: no route
    body = resp.json()
    head = ("job_id", "job_type", "status", "execution_id", "workflow_version")
    assert [body[k] for k in head] == [JOB_ID, "workflow_run", "completed", "wf-1", "1.2.0"]
    assert body["commit_attempt"]["attempt_id"] == "att-2"
    assert _col(body["outputs"], "name") == ["results/t.csv", "scratch/x.bin"]
    assert _col(body["outputs"], "ingest") == [True, False]


def test_receipt_joins_every_record_of_the_job() -> None:
    assert _sections(_receipt(_Store()).json()) == {
        "artifacts": ["results/t.csv"],
        "a2l_artifacts": ["a2l-1"],
        "knowledge_jobs": ["job-ingest01", "job-underst1"],
        "knowledge_artifact_ids": ["wfart-aaaa"],
        "warnings": [],
    }


def test_every_joined_row_is_bound_to_the_jobs_project() -> None:
    conn = _Conn(_tables())
    _receipt(_Store(conn))
    assert len(conn.executed) == 4
    for sql, params in conn.executed:
        assert params[:2] == (JOB_ID, "pa"), sql
        assert "project_id" in sql
    # The pooled connection goes back, with no transaction left open.
    assert conn.rollbacks == 4


@pytest.mark.parametrize("caller", [BOB, ALICE_ELSEWHERE], ids=["other-user", "other-project"])
def test_a_foreign_job_reads_like_an_unknown_one(caller: CallerIdentity) -> None:
    conn = _Conn(_tables())
    foreign = _receipt(_Store(conn), caller)
    unknown = _receipt(_Store(conn), caller, job_id="job-zzzzzzzzzzzz")
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.text.replace(JOB_ID, "<id>") == unknown.text.replace("job-zzzzzzzzzzzz", "<id>")
    assert conn.executed == []  # nothing joined for a refused caller


def test_admin_reads_any_receipt() -> None:
    assert _receipt(_Store(), ADMIN).status_code == 200


def test_an_unreadable_section_is_a_named_warning() -> None:
    conn = _Conn(_tables(), broken={"a2l_artifacts"})
    body = _receipt(_Store(conn)).json()
    assert body["a2l_artifacts"] == []
    assert body["warnings"] == [{"section": "a2l_artifacts", "reason": "RuntimeError"}]
    assert [a["name"] for a in body["artifacts"]] == ["results/t.csv"]
    assert body["knowledge_artifact_ids"] == ["wfart-aaaa"]


def test_no_commit_attempt_lists_no_outputs() -> None:
    store = _Store()
    store.get_job = lambda _id: Job(**{**_job().__dict__, "commit_attempt_id": None})
    body = _receipt(store).json()
    assert body["commit_attempt"] is None
    assert body["outputs"] == []


# ── step-level ingest (StepOutput.ingest under the steps' allowlist) ───────


def _steps(*outputs: list | None) -> list[dict[str, Any]]:
    steps = []
    for i, declared in enumerate(outputs):
        step: dict[str, Any] = {"name": f"s{i}", "entrypoint": "m:C"}
        if declared is not None:
            step["outputs"] = declared
        steps.append(step)
    return steps


def _results(**keys: list[str]) -> dict[str, Any]:
    return {name: SimpleNamespace(output_keys=k) for name, k in keys.items()}


def test_no_declared_flag_keeps_ingesting_everything() -> None:
    assert ingest_selection(_steps(None, [{"name": "o"}]), _results(s0=["a"], s1=["b"])) is None


def test_declared_flags_select_the_steps_keys() -> None:
    steps = _steps(
        [{"name": "table", "path_or_ref": "results/", "ingest": True}],
        [{"name": "scratch", "ingest": False}],
        [{"name": "all", "ingest": True}],
    )
    results = _results(
        s0=["results/t.csv", "results", "resultsX/y", "tmp/z"], s1=["scratch/x"], s2=["c/d"]
    )
    assert ingest_selection(steps, results) == ["c/d", "results", "results/t.csv"]


def _record(blobs: dict[str, bytes]) -> SimpleNamespace:
    return SimpleNamespace(
        results={"blobs": blobs}, execution_id="wf-1", workflow_name="demo", user_id="alice"
    )


def test_runner_records_the_selection_only_when_flags_are_declared() -> None:
    declared = _record({})
    levels = [_steps([{"name": "o", "path_or_ref": "keep", "ingest": True}])]
    _stamp_ingest_selection(declared, levels, _results(s0=["keep/a", "drop/b"]))
    assert declared.results["ingest_keys"] == ["keep/a"]
    legacy = _record({})
    _stamp_ingest_selection(legacy, [_steps(None)], _results(s0=["keep/a"]))
    assert "ingest_keys" not in legacy.results


class _S3:
    def put_object(self, **_: Any) -> dict:
        return {}


def _persist(record: SimpleNamespace) -> list[dict[str, Any]]:
    return wr._persist_run_artifacts(
        record=record,
        job=SimpleNamespace(job_id=JOB_ID, user_id="alice", project_id="pa", spec={}),
        attempt=SimpleNamespace(attempt_id="att-2", attempt_no=2, output_prefix="jobs/x/a2"),
        s3_client=_S3(),
        artifact_store=SimpleNamespace(create=lambda _a: SimpleNamespace(id=uuid.uuid4())),
    )


def test_outputs_outside_the_selection_are_stored_but_marked() -> None:
    record = _record({"bkt/keep/a": b"1", "bkt/drop/b": b"2"})
    record.results["ingest_keys"] = ["keep/a"]
    stored = {d["name"]: d for d in _persist(record)}
    assert set(stored) == {"keep/a", "drop/b"}  # both kept and registered
    assert "ingest" not in stored["keep/a"]
    assert stored["drop/b"]["ingest"] is False


def test_legacy_descriptors_keep_their_shape() -> None:
    (desc,) = _persist(_record({"bkt/keep/a": b"1"}))
    assert "ingest" not in desc


class _MetricsPool:
    def __init__(self, metrics: dict) -> None:
        self.metrics = metrics

    async def fetchval(self, _sql: str, _att: str) -> dict:
        return self.metrics


def _collected(monkeypatch: pytest.MonkeyPatch, artifacts: list[dict]) -> list[dict]:
    _metrics_hold(monkeypatch, artifacts)
    return ka._collect_artifact_ids(SimpleNamespace(job_id=JOB_ID))


def _metrics_hold(monkeypatch: pytest.MonkeyPatch, artifacts: list[dict]) -> None:
    """The commit attempt's metrics hold *artifacts* (no database)."""
    import asyncio
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _pool():
        yield _MetricsPool({"artifacts": artifacts})

    async def _attempts(_pool: Any, _job: Any) -> list[str]:
        return ["att-2"]

    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgresql://x")
    monkeypatch.setattr(ka, "_open_pool", _pool)
    monkeypatch.setattr(ka, "_resolve_succeeded_attempts", _attempts)
    monkeypatch.setattr(ka, "_run_coroutine_sync", lambda coro, _l="": asyncio.run(coro))


def test_auto_ingest_skips_outputs_outside_the_selection(monkeypatch) -> None:
    assert [a["artifact_id"] for a in _collected(monkeypatch, OUTPUTS)] == ["wfart-aaaa"]


def test_a_denied_ingest_makes_no_job(monkeypatch: pytest.MonkeyPatch) -> None:
    denied = [{**o, "ingest": False} for o in OUTPUTS]
    submitted: list[Any] = []
    _metrics_hold(monkeypatch, denied)
    monkeypatch.setattr(ka, "_enqueue_ingest_chain", lambda *a, **_k: submitted.append(a) or ["j"])
    job = SimpleNamespace(job_id=JOB_ID, project_id="pa", user_id="alice", spec={})
    assert ka.enqueue_knowledge_ingest(job, SimpleNamespace(attempt_id="att-2")) == []
    assert submitted == []


def test_the_ingest_job_names_the_run_it_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    from umrs.jobs import service

    seen: list[Any] = []
    fake = SimpleNamespace(submit_job=lambda sub: seen.append(sub) or "job-ingest01")
    monkeypatch.setattr(service, "get_job_service", lambda: fake)
    monkeypatch.setattr(ka, "_store_artifact_provenance", lambda *_a, **_k: None)
    job = SimpleNamespace(job_id=JOB_ID, priority=0, spec={})
    ka._enqueue_ingest_job(OUTPUTS[:1], "b", "pa", "alice", job, None, "demo", "*")
    assert seen[0].spec["parent_job_id"] == JOB_ID


# ── Postgres: the SQL itself, in a scratch schema ──────────────────────────

_DDL = (
    "CREATE TABLE artifacts (id UUID PRIMARY KEY, name TEXT, uri TEXT, size_bytes BIGINT,"
    " checksum TEXT, project_id TEXT, job_id TEXT, created_at TIMESTAMPTZ DEFAULT now())",
    "CREATE TABLE a2l_artifacts (id TEXT PRIMARY KEY, type TEXT, uri TEXT, status TEXT,"
    " project_id TEXT, job_id TEXT, deleted_at TIMESTAMPTZ,"
    " created_at TIMESTAMPTZ DEFAULT now())",
    "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, project_id TEXT, job_type TEXT, status TEXT,"
    " spec JSONB, created_at TIMESTAMPTZ DEFAULT now())",
    "CREATE TABLE knowledge_artifacts (artifact_id TEXT PRIMARY KEY, project_id TEXT,"
    " metadata JSONB)",
)
_ROWS = (
    "INSERT INTO artifacts (id, name, uri, project_id, job_id) VALUES"
    f" (gen_random_uuid(), 'mine', 's3://b/m', 'pa', '{JOB_ID}'),"
    f" (gen_random_uuid(), 'other-project', 's3://b/o', 'pb', '{JOB_ID}')",
    "INSERT INTO a2l_artifacts (id, type, uri, status, project_id, job_id, deleted_at) VALUES"
    f" ('a1', 'result', 's3://b/m', 'completed', 'pa', '{JOB_ID}', NULL),"
    f" ('a2', 'result', 's3://b/d', 'completed', 'pa', '{JOB_ID}', now())",
    "INSERT INTO jobs (job_id, project_id, job_type, status, spec) VALUES"
    f" ('ki', 'pa', 'knowledge_ingest', 'succeeded', '{{\"parent_job_id\": \"{JOB_ID}\"}}'),"
    f" ('ku', 'pb', 'knowledge_understand', 'queued', '{{\"parent_job_id\": \"{JOB_ID}\"}}'),"
    f" ('kx', 'pa', 'workflow_run', 'queued', '{{\"parent_job_id\": \"{JOB_ID}\"}}')",
    "INSERT INTO knowledge_artifacts VALUES"
    f" ('k1', 'pa', '{{\"produced_by_job_id\": \"{JOB_ID}\"}}'),"
    f" ('k2', 'pb', '{{\"produced_by_job_id\": \"{JOB_ID}\"}}')",
)


class _SchemaPool:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def getconn(self) -> Any:
        return self.conn

    def putconn(self, _conn: Any) -> None:
        return None


@pytest.fixture
def scratch_schema(db_connection: Any) -> Any:
    """A committed scratch schema holding the four joined tables (the
    receipt's own rollback would drop uncommitted rows); dropped after."""
    schema = f"receipt_{uuid.uuid4().hex[:10]}"
    with db_connection.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}, public")
        cur.execute(";".join(_DDL + _ROWS))
    db_connection.commit()
    yield db_connection
    with db_connection.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    db_connection.commit()


def test_receipt_sql_on_postgres(scratch_schema: Any) -> None:
    store = _Store()
    store.db_pool = _SchemaPool(scratch_schema)
    assert _sections(_receipt(store).json()) == {
        "artifacts": ["mine"],
        "a2l_artifacts": ["a1"],
        "knowledge_jobs": ["ki"],
        "knowledge_artifact_ids": ["k1"],
        "warnings": [],
    }
