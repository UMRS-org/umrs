"""Job responses carry the claiming site (multi-HPC plan S3, 2026-10-24).

All umrs-front servers share one ``jobs`` table, so a client that lists jobs
through several gateways sees each row once per gateway. ``venue`` on the
response lets it deduplicate by ``job_id`` and label each row with the site
that runs it. Before this, ``_row_to_job`` dropped the column (every job read
back with ``venue=""``) and ``JobResponse`` had no field for it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from umrs.api.jobs import _job_to_response
from umrs.jobs.models import JobType
from umrs.jobs.store import JobStore

_NOW = datetime(2026, 10, 24, tzinfo=timezone.utc)


def _row(**over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "job_id": "job-1",
        "project_id": "p",
        "user_id": "alice",
        "job_type": "research_loop_job",
        "spec": {},
        "idempotency_key": None,
        "status": "queued",
        "priority": 0,
        "desired_max_attempts": 1,
        "restart_policy": "on_crash",
        "next_run_at": None,
        "active_attempt_id": None,
        "commit_attempt_id": None,
        "created_at": _NOW,
        "updated_at": _NOW,
        "venue": "bdrt-front",
    }
    row.update(over)
    return row


def test_row_to_job_keeps_the_venue() -> None:
    store = JobStore.__new__(JobStore)
    assert store._row_to_job(_row()).venue == "bdrt-front"
    assert store._row_to_job(_row(venue=None)).venue == ""


def test_job_response_carries_the_venue() -> None:
    job = JobStore.__new__(JobStore)._row_to_job(_row())
    assert _job_to_response(job).venue == "bdrt-front"
    assert _job_to_response(job).model_dump()["venue"] == "bdrt-front"


def _env_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    return pool


def test_listed_jobs_carry_their_venue_on_postgres() -> None:
    pool = _env_pool_or_skip()
    try:
        store = JobStore(pool, "test-instance")
        store.site_name = "rccs-front"
        job_id = store.create_job(
            project_id="proj-venue",
            user_id="alice-venue",
            job_type=JobType.RESEARCH_LOOP,
            spec={"loop_id": "l1", "venue": "cluster:bdrt-front"},
            idempotency_key=None,
        )
        assert store.get_job(job_id).venue == "bdrt-front"
        listed = store.list_jobs(user_id="alice-venue", project_id="proj-venue", limit=10)
        assert {j.job_id: _job_to_response(j).venue for j in listed}[job_id] == "bdrt-front"
    finally:
        pool.closeall()
