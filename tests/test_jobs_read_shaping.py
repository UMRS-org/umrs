"""Server-side read shaping (ADR-105 D-6, slice S6a; decisions-2026-11-24 §2.4).

``GET /umrs/jobs`` (``job_list``) and ``GET /umrs/workflows/executions``
(``workflow_executions``) accept ``fields=`` (projection), ``cursor=`` (an
opaque keyset position from ``next_cursor``) and ``summary=true``
(aggregates). Acceptance: summary aggregates; projection; a cursor walk over
1001 jobs without duplicates; no scope widening (a cursor never carries a
scope, and one minted for another caller or filter set is refused).

Fail-before: on the code before S6a, ``fields``/``cursor``/``summary`` were
unknown query parameters (ignored), no ``next_cursor`` existed, and
``/umrs/workflows/executions`` was read as a workflow named "executions".
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from typing import Any

import pytest
from _a105_s6a_pending import ALL_TASKS, pending
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import jobs as jobs_api
from umrs.api import workflows as workflows_api
from umrs.api.jobs import CallerIdentity
from umrs.jobs.models import Job, JobStatus, JobType
from umrs.jobs.read_query import JobKeysetPosition, JobReadFilters
from umrs.jobs.read_summary import SummaryGroup
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)

from umrs_protocol.errors import ERROR_CODE_HEADER

pytestmark = pending(*ALL_TASKS)

T0 = datetime(2026, 11, 1)
ALICE = CallerIdentity(user_id="alice", project_id="pa", is_admin=False)
BOB = CallerIdentity(user_id="bob", project_id="pa", is_admin=False)
ADMIN = CallerIdentity(user_id="root", project_id="pa", is_admin=True)


# ── fake job store (SQL-faithful: filters, total order, keyset) ────────────


def _job(i: int, user: str = "alice", project: str = "pa", **kw: Any) -> Job:
    created = T0 + timedelta(minutes=i // 10)  # ten jobs share each timestamp
    return Job(
        job_id=f"job-{user}-{i:05d}",
        project_id=project,
        user_id=user,
        job_type=kw.get("job_type", JobType.KNOWLEDGE_EMBED),
        spec={"n": i},
        idempotency_key=f"k-{user}-{project}-{i}",
        status=kw.get("status", JobStatus.COMPLETED),
        created_at=created,
        updated_at=created,
    )


def _matches(job: Job, f: JobReadFilters) -> bool:
    return (
        (f.user_id is None or job.user_id == f.user_id)
        and (not f.project_id or job.project_id == f.project_id)
        and (not f.status or job.status.value == f.status)
        and (not f.job_type or job.job_type.value == f.job_type)
    )


class FakeStore:
    def __init__(self, rows: list[Job]) -> None:
        self.rows = rows
        self.seen: list[JobReadFilters] = []

    def _ordered(self, f: JobReadFilters) -> list[Job]:
        self.seen.append(f)
        rows = [r for r in self.rows if _matches(r, f)]
        return sorted(rows, key=lambda r: (r.created_at, r.job_id), reverse=True)

    def list_jobs(
        self, project_id=None, status=None, job_type=None, limit=100, offset=0, user_id=None
    ):
        f = JobReadFilters(
            project_id=project_id,
            user_id=user_id,
            status=status.value if status else None,
            job_type=job_type.value if job_type else None,
        )
        return self._ordered(f)[offset : offset + limit]

    def list_jobs_page(self, filters, *, after=None, limit=100, offset=0):
        rows = self._ordered(filters)
        if after is not None:
            key = (after.created_at, after.job_id)
            rows = [r for r in rows if (r.created_at, r.job_id) < key]
        return rows[offset : offset + limit]

    def summarize_jobs(self, filters):
        groups: dict[tuple[str, str], list[Job]] = {}
        for r in self._ordered(filters):
            groups.setdefault((r.status.value, r.job_type.value), []).append(r)
        return [
            SummaryGroup(s, t, len(v), min(j.created_at for j in v), max(j.created_at for j in v))
            for (s, t), v in sorted(groups.items())
        ]


def _jobs_client(store: Any, caller: CallerIdentity) -> TestClient:
    app = FastAPI()
    app.include_router(jobs_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return TestClient(app)


def _walk(client: TestClient, params: dict[str, Any]) -> tuple[list[str], int]:
    """Follow next_cursor to the end; the job ids seen and the page count."""
    seen: list[str] = []
    body = client.get("/umrs/jobs", params=params).json()
    pages = 1
    seen.extend(j["job_id"] for j in body["jobs"])
    while body.get("next_cursor"):
        r = client.get("/umrs/jobs", params={**params, "cursor": body["next_cursor"]})
        assert r.status_code == 200, r.text
        body = r.json()
        pages += 1
        seen.extend(j["job_id"] for j in body["jobs"])
        assert pages < 100
    return seen, pages


# ── job_list ────────────────────────────────────────────────────────────────


def test_cursor_walks_1001_jobs_without_duplicates() -> None:
    store = FakeStore([_job(i) for i in range(1001)])
    seen, pages = _walk(_jobs_client(store, ALICE), {"limit": 100})
    assert len(seen) == 1001
    assert len(set(seen)) == 1001
    assert pages == 11


def test_last_page_has_no_next_cursor() -> None:
    store = FakeStore([_job(i) for i in range(5)])
    body = _jobs_client(store, ALICE).get("/umrs/jobs", params={"limit": 5}).json()
    assert len(body["jobs"]) == 5
    assert body["next_cursor"] is None


def test_offset_page_cursor_continues_the_same_order() -> None:
    store = FakeStore([_job(i) for i in range(30)])
    client = _jobs_client(store, ALICE)
    first = client.get("/umrs/jobs", params={"limit": 10, "offset": 5}).json()
    nxt = client.get("/umrs/jobs", params={"limit": 10, "cursor": first["next_cursor"]}).json()
    everything = client.get("/umrs/jobs", params={"limit": 30}).json()
    ids = [j["job_id"] for j in everything["jobs"]]
    assert [j["job_id"] for j in first["jobs"]] == ids[5:15]
    assert [j["job_id"] for j in nxt["jobs"]] == ids[15:25]


def test_fields_projects_rows_into_items() -> None:
    store = FakeStore([_job(i) for i in range(3)])
    body = _jobs_client(store, ALICE).get("/umrs/jobs", params={"fields": "status,job_type"}).json()
    assert body["jobs"] == []
    assert body["total"] == 3
    assert body["items"][0] == {
        "job_id": "job-alice-00002",
        "status": "completed",
        "job_type": "knowledge_embed",
    }
    assert all(list(item) == ["job_id", "status", "job_type"] for item in body["items"])


def test_unknown_field_is_a_typed_400() -> None:
    store = FakeStore([_job(0)])
    r = _jobs_client(store, ALICE).get("/umrs/jobs", params={"fields": "status,attempts"})
    assert r.status_code == 400
    assert r.headers[ERROR_CODE_HEADER] == "jobs.bad_request"
    assert "unknown attempts" in r.text


def test_summary_counts_every_matching_job() -> None:
    rows = [_job(i) for i in range(4)]
    rows += [_job(i, status=JobStatus.FAILED, job_type=JobType.WORKFLOW_RUN) for i in range(4, 6)]
    rows += [_job(i, user="bob") for i in range(6, 9)]
    store = FakeStore(rows)
    body = (
        _jobs_client(store, ALICE).get("/umrs/jobs", params={"summary": "true", "limit": 1}).json()
    )
    assert (body["jobs"], body["total"]) == ([], 6)
    assert body["summary"] == {
        "total": 6,
        "by_status": {"completed": 4, "failed": 2},
        "by_type": {"knowledge_embed": 4, "workflow_run": 2},
        "oldest": T0.isoformat(),
        "newest": T0.isoformat(),
    }
    assert store.seen[-1] == JobReadFilters(project_id="pa", user_id="alice")


def test_non_admin_scope_is_pinned_in_every_mode() -> None:
    store = FakeStore([_job(i) for i in range(3)] + [_job(9, project="pb")])
    client = _jobs_client(store, ALICE)
    for params in ({"project_id": "pb"}, {"project_id": "pb", "summary": "true"}):
        r = client.get("/umrs/jobs", params=params)
        assert r.status_code == 200
        assert store.seen[-1].project_id == "pa"
        assert store.seen[-1].user_id == "alice"


def test_a_cursor_of_another_caller_is_refused() -> None:
    store = FakeStore([_job(i) for i in range(5)] + [_job(i, user="bob") for i in range(5)])
    token = _jobs_client(store, ALICE).get("/umrs/jobs", params={"limit": 2}).json()["next_cursor"]
    assert token
    r = _jobs_client(store, BOB).get("/umrs/jobs", params={"limit": 2, "cursor": token})
    assert r.status_code == 400
    assert "cursor belongs to another query" in r.text


def test_a_cursor_for_other_filters_is_refused() -> None:
    rows = [_job(i) for i in range(5)] + [_job(i, project="pb") for i in range(5, 10)]
    client = _jobs_client(FakeStore(rows), ADMIN)
    token = client.get("/umrs/jobs", params={"limit": 2, "project_id": "pa"}).json()["next_cursor"]
    r = client.get("/umrs/jobs", params={"limit": 2, "project_id": "pb", "cursor": token})
    assert r.status_code == 400
    assert "cursor belongs to another query" in r.text


def test_cursor_with_offset_is_refused() -> None:
    client = _jobs_client(FakeStore([_job(i) for i in range(5)]), ALICE)
    token = client.get("/umrs/jobs", params={"limit": 2}).json()["next_cursor"]
    r = client.get("/umrs/jobs", params={"limit": 2, "offset": 1, "cursor": token})
    assert r.status_code == 400
    assert "cursor and offset cannot be combined" in r.text


@pytest.mark.parametrize("token", ["garbage!", "x" * 600, "eyJ2IjoxfQ"])
def test_malformed_cursor_is_a_typed_400(token: str) -> None:
    r = _jobs_client(FakeStore([_job(0)]), ALICE).get("/umrs/jobs", params={"cursor": token})
    assert r.status_code == 400
    assert r.headers[ERROR_CODE_HEADER] == "jobs.bad_request"
    assert "cursor is malformed" in r.text


# ── workflow_executions ────────────────────────────────────────────────────


@pytest.fixture
def registry():
    reset_execution_registry_for_tests()
    yield get_execution_registry()
    reset_execution_registry_for_tests()


def _wf_client(caller: CallerIdentity) -> TestClient:
    app = FastAPI()
    app.include_router(workflows_api.router)
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return TestClient(app)


def _execution(reg, eid: str, ts: float, user: str = "alice", project: str = "pa", **kw: Any):
    record = reg.create(
        workflow_name=kw.get("wf", "align"),
        user_id=user,
        params={"project_id": project},
        execution_id=eid,
        identity_provenance=kw.get("provenance", "gateway_verified"),
    )
    record.submitted_at = ts
    record.status = kw.get("status", "running")
    return record


def test_executions_route_is_not_a_workflow_name(registry) -> None:
    _execution(registry, "wf-1", 1.0)
    r = _wf_client(ALICE).get("/umrs/workflows/executions")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert [e["execution_id"] for e in body["executions"]] == ["wf-1"]


def test_executions_visibility_is_per_record(registry) -> None:
    _execution(registry, "own", 5.0)
    _execution(registry, "other-user", 4.0, user="bob")
    _execution(registry, "other-project", 3.0, project="pb")
    _execution(registry, "unverified", 2.0, provenance="unverified")
    body = _wf_client(ALICE).get("/umrs/workflows/executions", params={"user_id": "bob"}).json()
    assert [e["execution_id"] for e in body["executions"]] == ["own"]
    summary = (
        _wf_client(ALICE)
        .get("/umrs/workflows/executions", params={"summary": "true"})
        .json()["summary"]
    )
    assert summary["total"] == 1


def test_admin_sees_every_owner_or_one(registry) -> None:
    _execution(registry, "a", 2.0)
    _execution(registry, "b", 1.0, user="bob", project="pb")
    every = _wf_client(ADMIN).get("/umrs/workflows/executions").json()
    assert [e["execution_id"] for e in every["executions"]] == ["a", "b"]
    one = _wf_client(ADMIN).get("/umrs/workflows/executions", params={"user_id": "bob"}).json()
    assert [e["execution_id"] for e in one["executions"]] == ["b"]


def test_executions_cursor_walk(registry) -> None:
    for i in range(12):
        _execution(registry, f"wf-{i:02d}", float(i // 3))
    client = _wf_client(ALICE)
    seen: list[str] = []
    body = client.get("/umrs/workflows/executions", params={"limit": 5}).json()
    counts = [body["count"]]
    seen += [e["execution_id"] for e in body["executions"]]
    while body["next_cursor"]:
        body = client.get(
            "/umrs/workflows/executions", params={"limit": 5, "cursor": body["next_cursor"]}
        ).json()
        counts.append(body["count"])
        seen += [e["execution_id"] for e in body["executions"]]
    assert counts == [5, 5, 2]
    assert sorted(seen) == [f"wf-{i:02d}" for i in range(12)]


def test_executions_cursor_of_another_caller_is_refused(registry) -> None:
    for i in range(4):
        _execution(registry, f"a{i}", float(i))
        _execution(registry, f"b{i}", float(i), user="bob")
    token = (
        _wf_client(ALICE)
        .get("/umrs/workflows/executions", params={"limit": 2})
        .json()["next_cursor"]
    )
    r = _wf_client(BOB).get("/umrs/workflows/executions", params={"limit": 2, "cursor": token})
    assert r.status_code == 400
    assert r.headers[ERROR_CODE_HEADER] == "workflows.bad_request"
    assert "cursor belongs to another query" in r.text


def test_executions_fields_and_status_filter(registry) -> None:
    _execution(registry, "ok", 2.0, status="succeeded")
    _execution(registry, "bad", 1.0, status="failed")
    body = (
        _wf_client(ALICE)
        .get("/umrs/workflows/executions", params={"fields": "status", "status": "SUCCEEDED"})
        .json()
    )
    assert body["executions"] == [{"execution_id": "ok", "status": "succeeded"}]
    r = _wf_client(ALICE).get("/umrs/workflows/executions", params={"fields": "params"})
    assert r.status_code == 400
    assert "unknown params" in r.text


def test_executions_summary(registry) -> None:
    _execution(registry, "a", 2.0, status="succeeded")
    _execution(registry, "b", 3.0, status="succeeded", wf="blast")
    _execution(registry, "c", 1.0, status="failed")
    body = _wf_client(ALICE).get("/umrs/workflows/executions", params={"summary": "true"}).json()
    assert body["count"] == 0
    assert body["executions"] == []
    assert body["summary"]["by_status"] == {"failed": 1, "succeeded": 2}
    assert body["summary"]["by_type"] == {"align": 2, "blast": 1}


# ── real Postgres: the SQL itself ──────────────────────────────────────────


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


needs_pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture
def pg_store():
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.jobs.store import JobStore

    pool = ThreadedConnectionPool(1, 4, _db_url())
    project = f"s6a-{uuid.uuid4().hex[:10]}"
    store = JobStore(pool)
    yield store, project
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE project_id = %s", (project,))
        conn.commit()
    finally:
        pool.putconn(conn)
        pool.closeall()


def _seed(store: Any, project: str, n: int, user: str) -> None:
    for i in range(n):
        store.create_job(
            project_id=project,
            user_id=user,
            job_type=JobType.KNOWLEDGE_EMBED,
            spec={"n": i},
            idempotency_key=f"{project}-{user}-{i}",
        )
    conn = store.db_pool.getconn()
    try:
        with conn.cursor() as cur:
            # Ten jobs per timestamp: the keyset must break ties on job_id.
            cur.execute(
                "UPDATE jobs SET created_at = %s + (substring(idempotency_key from "
                "'([0-9]+)$')::int / 10) * interval '1 minute' WHERE project_id = %s",
                (T0, project),
            )
        conn.commit()
    finally:
        store.db_pool.putconn(conn)


@needs_pg
def test_pg_cursor_walks_1001_jobs_without_duplicates(pg_store) -> None:
    store, project = pg_store
    _seed(store, project, 1001, "alice")
    _seed(store, project, 3, "bob")
    caller = CallerIdentity(user_id="alice", project_id=project, is_admin=False)
    client = _jobs_client(store, caller)
    seen, pages = _walk(client, {"limit": 100})
    assert (len(seen), len(set(seen)), pages) == (1001, 1001, 11)


@needs_pg
def test_pg_summary_and_fields(pg_store) -> None:
    store, project = pg_store
    _seed(store, project, 30, "alice")
    _seed(store, project, 3, "bob")
    caller = CallerIdentity(user_id="alice", project_id=project, is_admin=False)
    client = _jobs_client(store, caller)
    summary = client.get("/umrs/jobs", params={"summary": "true"}).json()["summary"]
    assert (summary["total"], sum(summary["by_status"].values())) == (30, 30)
    assert summary["by_type"] == {"knowledge_embed": 30}
    assert datetime.fromisoformat(summary["oldest"]).replace(tzinfo=None) == T0
    items = client.get("/umrs/jobs", params={"fields": "status", "limit": 2}).json()["items"]
    assert [list(i) for i in items] == [["job_id", "status"], ["job_id", "status"]]


@needs_pg
def test_pg_page_after_a_tie_boundary(pg_store) -> None:
    store, project = pg_store
    _seed(store, project, 25, "alice")
    f = JobReadFilters(project_id=project, user_id="alice")
    full = store.list_jobs_page(f, limit=25)
    cut = full[14]  # inside a ten-job tie group
    rest = store.list_jobs_page(
        f, after=JobKeysetPosition(created_at=cut.created_at, job_id=cut.job_id), limit=25
    )
    assert [j.job_id for j in rest] == [j.job_id for j in full[15:]]
    assert [j.job_id for j in store.list_jobs(project_id=project, limit=25)] == [
        j.job_id for j in full
    ]
