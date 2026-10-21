"""Cycle-2 umrs-api lane (2026-10-21): a non-admin's job list pages over
their own jobs, not over the whole project's.

Found live (two users in one project, gateway): alice had 2 jobs, carol
then submitted 5; ``GET /gw/jobs?limit=3`` as alice returned 0 jobs
(total 0) while ``limit=1000`` returned her 2. The route asked the store for
the project's newest ``limit`` jobs and only then dropped other users'
rows, so a busy shared project hid the caller's own jobs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import jobs as jobs_api
from umrs.api._verified_headers import CallerIdentity
from umrs.jobs.models import Job, JobType

T0 = datetime(2026, 10, 21)


class _Store:
    """SQL-faithful list_jobs: filters, newest first, then LIMIT/OFFSET."""

    def __init__(self, rows: list[Job]) -> None:
        self.rows = rows

    def list_jobs(self, project_id=None, status=None, job_type=None, limit=100, offset=0, user_id=None):
        del status, job_type
        rows = [r for r in self.rows if project_id in (None, r.project_id)]
        rows = [r for r in rows if user_id in (None, r.user_id)]
        rows.sort(key=lambda r: r.created_at, reverse=True)
        return rows[offset : offset + limit]


def _job(i: int, user: str) -> Job:
    return Job(
        job_id=f"job-{user}-{i}", project_id="pq", user_id=user,
        job_type=JobType.KNOWLEDGE_EMBED, spec={}, idempotency_key=f"k{i}",
        created_at=T0 + timedelta(minutes=i), updated_at=T0 + timedelta(minutes=i),
    )


def test_own_jobs_are_not_paged_out_by_colleagues() -> None:
    rows = [_job(0, "alice"), _job(1, "alice")] + [_job(10 + i, "carol") for i in range(5)]
    app = FastAPI()
    app.include_router(jobs_api.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: _Store(rows)
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: CallerIdentity(
        user_id="alice", project_id="pq", is_admin=False
    )
    r = TestClient(app).get("/umrs/jobs", params={"limit": 3})
    assert r.status_code == 200, r.text
    assert [j["job_id"] for j in r.json()["jobs"]] == ["job-alice-1", "job-alice-0"]
