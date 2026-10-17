"""H-1 web-shell lane: dashboard job counts come from JobStore.count_jobs.

Before this lane the overview asked the store for ``count_by_status`` and the
job list for ``count_all``; no JobStore implements either, so every caller saw
zeros (and a non-admin's job list total was the page size). ``count_jobs``
uses the same WHERE clause as ``list_jobs``; the dashboard passes the caller's
owner filter (admins count platform-wide).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs.models import JobStatus, JobType
from umrs.jobs.store import JobStore

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-webshell-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
ROOT = _headers("root", "ops", admin=True)
FORGED = {
    "X-User-ID": "mallory",
    "X-Verified-Project-ID": "proj-b",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": "0" * 64,
}


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── JobStore.count_jobs (SQL shape) ─────────────────────────────────────────


class _Cursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), list(params or [])))

    def fetchone(self):
        return (7,)

    def fetchall(self):
        return []


class _Pool:
    def __init__(self):
        self.log: list[tuple[str, list]] = []

    def getconn(self):
        log = self.log
        return SimpleNamespace(cursor=lambda **_kw: _Cursor(log))

    def putconn(self, _conn):
        return None


class _NoSchemaStore(JobStore):
    def _ensure_schema(self):
        return None


def test_count_jobs_without_filters_counts_every_row():
    pool = _Pool()
    assert _NoSchemaStore(pool).count_jobs() == 7
    sql, params = pool.log[-1]
    assert sql == "SELECT COUNT(*) FROM jobs"
    assert params == []


def test_count_jobs_uses_the_list_jobs_filters():
    pool = _Pool()
    store = _NoSchemaStore(pool)
    count = store.count_jobs(
        project_id="proj-a", user_id="alice", status=JobStatus.RUNNING, job_type=JobType.INDEX_BUILD
    )
    assert count == 7
    sql, params = pool.log[-1]
    assert sql == (
        "SELECT COUNT(*) FROM jobs WHERE user_id = %s AND project_id = %s"
        " AND status = %s AND job_type = %s"
    )
    assert params == ["alice", "proj-a", "running", JobType.INDEX_BUILD.value]


def test_count_jobs_accepts_a_status_string():
    pool = _Pool()
    _NoSchemaStore(pool).count_jobs(status="queued")
    assert pool.log[-1][1] == ["queued"]


# Review follow-up: pin "same WHERE clause as list_jobs" against list_jobs
# itself, not a hand-written string, so a filter added to one drifts loudly.

_FILTER_CASES = [
    {},
    {"user_id": "alice"},
    {"project_id": "proj-a"},
    {"status": JobStatus.RUNNING},
    {"job_type": JobType.PIPELINE_RUN},
    {"user_id": "", "project_id": "proj-a"},
    {
        "project_id": "proj-a",
        "user_id": "alice",
        "status": JobStatus.FAILED,
        "job_type": JobType.INDEX_BUILD,
    },
]


def _where(sql: str) -> str:
    """The text between ``FROM jobs`` and ``ORDER BY`` (or the end)."""
    return sql.split("FROM jobs", 1)[1].split("ORDER BY", 1)[0].strip()


@pytest.mark.parametrize("filters", _FILTER_CASES)
def test_count_jobs_where_clause_matches_list_jobs(filters):
    pool = _Pool()
    store = _NoSchemaStore(pool)
    store.list_jobs(limit=5, offset=10, **filters)
    store.count_jobs(**filters)
    (list_sql, list_params), (count_sql, count_params) = pool.log[-2:]
    assert list_params[-2:] == [5, 10]
    assert _where(count_sql) == _where(list_sql)
    assert count_params == list_params[:-2]


def test_count_jobs_takes_every_list_jobs_filter():
    import inspect

    def _names(fn) -> set[str]:
        return set(inspect.signature(fn).parameters) - {"self"}

    assert _names(JobStore.count_jobs) == _names(JobStore.list_jobs) - {"limit", "offset"}


# ── dashboard overview + job list totals ────────────────────────────────────


def _job(job_id: str, project: str, user: str, status: JobStatus, job_type=JobType.INDEX_BUILD):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        job_id=job_id,
        project_id=project,
        user_id=user,
        status=status,
        job_type=job_type,
        ttl_seconds=None,
        fan_out_child_job_ids=[],
        created_at=now,
        updated_at=now,
        active_attempt_id=None,
    )


class _CountingJobStore:
    """In-memory store with list_jobs + count_jobs (same filters)."""

    def __init__(self, jobs):
        self.jobs = jobs
        self.count_calls: list[dict] = []

    def _match(self, job, project_id, user_id, status, job_type=None) -> bool:
        wanted = getattr(status, "value", status)
        return (
            (not project_id or job.project_id == project_id)
            and (user_id is None or job.user_id == user_id)
            and (wanted is None or job.status.value == wanted)
            and (job_type is None or job.job_type == job_type)
        )

    def list_jobs(
        self, project_id=None, status=None, job_type=None, limit=100, offset=0, user_id=None
    ):
        rows = [j for j in self.jobs if self._match(j, project_id, user_id, status, job_type)]
        return rows[offset : offset + limit]

    def count_jobs(self, project_id=None, user_id=None, status=None, job_type=None):
        self.count_calls.append({"project_id": project_id, "user_id": user_id})
        rows = [j for j in self.jobs if self._match(j, project_id, user_id, status, job_type)]
        return len(rows)


@pytest.fixture
def dash(monkeypatch):
    from umrs.api import web_dashboard

    store = _CountingJobStore(
        [
            _job("a-run", "proj-a", "alice", JobStatus.RUNNING),
            _job("a-fail", "proj-a", "alice", JobStatus.FAILED),
            _job("b-run", "proj-a", "bob", JobStatus.RUNNING),
            _job("c-queue", "proj-b", "carol", JobStatus.QUEUED),
            _job("c-done", "proj-b", "carol", JobStatus.COMPLETED),
        ]
    )
    monkeypatch.setattr(web_dashboard, "get_job_store", lambda: store)
    monkeypatch.setattr(web_dashboard, "get_optional_job_store", lambda: store)
    monkeypatch.setattr(web_dashboard, "get_optional_model_store", lambda: None)
    monkeypatch.setattr(web_dashboard, "_workflow_execution_counts", lambda _scope: None)
    app = FastAPI()
    app.include_router(web_dashboard.router)
    return TestClient(app), store


def _counts(body: dict) -> tuple[int, int, int, int, int]:
    keys = ("running_jobs", "queued_jobs", "completed_jobs", "failed_jobs", "total_jobs")
    return tuple(body[k] for k in keys)


def test_overview_counts_the_callers_own_jobs(dash):
    client, store = dash
    body = client.get("/umrs/web/dashboard/overview", headers=ALICE).json()
    assert _counts(body) == (1, 0, 0, 1, 2)
    assert {(c["project_id"], c["user_id"]) for c in store.count_calls} == {("proj-a", "alice")}


def test_overview_counts_are_platform_wide_for_a_verified_admin(dash):
    client, store = dash
    body = client.get("/umrs/web/dashboard/overview", headers=ROOT).json()
    assert _counts(body) == (2, 1, 1, 1, 5)
    assert {(c["project_id"], c["user_id"]) for c in store.count_calls} == {(None, None)}


def test_overview_forged_admin_tuple_is_refused(dash):
    client, store = dash
    assert client.get("/umrs/web/dashboard/overview", headers=FORGED).status_code == 401
    assert store.count_calls == []


def test_job_list_total_is_the_callers_own_count(dash):
    client, _ = dash
    body = client.get("/umrs/web/dashboard/jobs?limit=1", headers=ALICE).json()
    assert [j["job_id"] for j in body["items"]] == ["a-run"]
    assert body["pagination"]["total"] == 2
    assert body["pagination"]["has_next"] is True


def test_job_list_total_is_platform_wide_for_a_verified_admin(dash):
    client, _ = dash
    body = client.get("/umrs/web/dashboard/jobs?limit=2&status=running", headers=ROOT).json()
    assert body["pagination"]["total"] == 2
    assert body["pagination"]["has_next"] is False


def test_job_list_forged_admin_tuple_is_refused(dash):
    client, store = dash
    assert client.get("/umrs/web/dashboard/jobs", headers=FORGED).status_code == 401
    assert store.count_calls == []


# Review follow-up: the total honours the list's ?job_type= and ?user_id=
# filters too (it ignored both, so pagination.total/has_next were wrong).


def test_job_list_total_honours_the_job_type_filter(dash):
    client, store = dash
    store.jobs.append(_job("p-run", "proj-b", "carol", JobStatus.RUNNING, JobType.PIPELINE_RUN))
    url = "/umrs/web/dashboard/jobs?limit=1&job_type=pipeline_run"
    body = client.get(url, headers=ROOT).json()
    assert [j["job_id"] for j in body["items"]] == ["p-run"]
    assert body["pagination"]["total"] == 1
    assert body["pagination"]["has_next"] is False


def test_job_list_total_honours_an_admins_user_filter(dash):
    client, _ = dash
    url = "/umrs/web/dashboard/jobs?limit=1&user_id=carol"
    body = client.get(url, headers=ROOT).json()
    assert body["pagination"]["total"] == 2
    assert body["pagination"]["has_next"] is True


def test_job_list_total_is_zero_for_another_users_filter(dash):
    client, store = dash
    body = client.get("/umrs/web/dashboard/jobs?user_id=bob", headers=ALICE).json()
    assert body["items"] == []
    assert body["pagination"]["total"] == 0
    assert body["pagination"]["has_next"] is False
    assert store.count_calls == []


def test_job_list_total_with_the_callers_own_user_filter(dash):
    client, _ = dash
    body = client.get("/umrs/web/dashboard/jobs?user_id=alice", headers=ALICE).json()
    assert body["pagination"]["total"] == 2


def test_job_list_total_for_an_unknown_job_type_is_zero(dash):
    client, _ = dash
    body = client.get("/umrs/web/dashboard/jobs?job_type=nope", headers=ROOT).json()
    assert body["items"] == []
    assert body["pagination"]["total"] == 0


def test_counts_degrade_to_the_old_view_when_the_store_has_no_count_jobs(dash, monkeypatch):
    client, store = dash
    monkeypatch.setattr(store, "count_jobs", None)
    body = client.get("/umrs/web/dashboard/overview", headers=ALICE).json()
    assert _counts(body) == (0, 0, 0, 0, 0)
    jobs = client.get("/umrs/web/dashboard/jobs?limit=1", headers=ALICE).json()
    assert jobs["pagination"]["total"] == 1
