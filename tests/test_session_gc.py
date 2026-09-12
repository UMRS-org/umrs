"""SESSION_GC job type pins (fail-without-change).

Backend is archive-only; it uses the store's own archive cascade (never a
straight DELETE), per-attempt report lives on the shared backend instance,
and the status checker maps reports to succeeded with the counters.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umrs.jobs.models import Job, JobAttempt, JobStatus, JobType, RestartPolicy, AttemptStatus
from umrs.jobs.session_gc import SessionGcBackend, SessionGcStatusChecker


def _job(spec: dict, job_id: str = "job-g1") -> Job:
    return Job(
        job_id=job_id,
        job_type=JobType.SESSION_GC,
        project_id="proj-9",
        user_id="u-9",
        spec=spec,
        idempotency_key="",
        status=JobStatus.QUEUED,
        restart_policy=RestartPolicy.ON_CRASH,
        desired_max_attempts=1,
    )


def _attempt() -> JobAttempt:
    return JobAttempt(
        attempt_id="att-g1",
        job_id="job-g1",
        attempt_no=1,
        backend="session_gc",
        status=AttemptStatus.RUNNING,
    )


class TestSpecChain:
    def test_max_days_required(self):
        backend = SessionGcBackend(session_store=MagicMock())
        with pytest.raises(ValueError, match="max_days"):
            asyncio.run(backend.submit_job(_job({}), _attempt()))

    def test_max_days_must_be_positive_int(self):
        backend = SessionGcBackend(session_store=MagicMock())
        with pytest.raises(ValueError, match="> 0"):
            asyncio.run(backend.submit_job(_job({"max_days": 0}), _attempt()))
        with pytest.raises(ValueError, match="integer"):
            asyncio.run(backend.submit_job(_job({"max_days": "not-a-number"}), _attempt()))

    def test_missing_store_fails_honestly(self):
        backend = SessionGcBackend(session_store=None)
        with pytest.raises(RuntimeError, match="session store"):
            asyncio.run(backend.submit_job(_job({"max_days": 30}), _attempt()))


class TestSweepLane:
    def test_archive_only_calls_into_cascade(self):
        store = MagicMock()
        store.archive_aged_sessions = MagicMock(return_value=["s-1", "s-2"])
        backend = SessionGcBackend(session_store=store)
        with patch(
            "umrs.api.companion.get_trust_store",
            return_value=MagicMock(),
        ):
            asyncio.run(backend.submit_job(_job({"max_days": 45}), _attempt()))
        args, kwargs = store.archive_aged_sessions.call_args
        assert kwargs["max_days"] == 45
        assert kwargs["trust_store"] is not None
        # Handle stays durable per attempt_id (readable per-attempt).
        report = backend.report_for("job-g1")
        assert report["archived"] == 2
        assert report["max_days"] == 45
        assert report["lane"].startswith("archive-only")
        assert (
            store.delete_over_sessions.call_count == 0
            if hasattr(store, "delete_over_sessions")
            else True
        )

    def test_report_per_job_isolated(self):
        store = MagicMock()
        store.archive_aged_sessions = MagicMock(return_value=["a"])
        backend = SessionGcBackend(session_store=store)
        with patch("umrs.api.companion.get_trust_store", return_value=MagicMock()):
            asyncio.run(backend.submit_job(_job({"max_days": 7}, job_id="job-g2"), _attempt()))
        assert set(backend.report_for("job-g2").keys()) >= {"max_days", "archived"}
        assert backend.report_for("the-other-attempt") == {}


class TestStatusChecker:
    def test_succeeded_carries_report_metrics(self):
        backend = SessionGcBackend(session_store=MagicMock())
        checker = SessionGcStatusChecker(backend=backend)
        backend._reports["job-g1"] = {"archived": 4, "max_days": 30}
        result = asyncio.run(checker.check_status(_attempt(), _job({})))
        assert result.status == "succeeded"
        assert result.metrics["archived"] == 4

    def test_storeless_backend_reports_empty(self):
        checker = SessionGcStatusChecker(backend=None)
        result = asyncio.run(checker.check_status(_attempt(), _job({})))
        assert result.status == "succeeded"
        assert result.metrics == {}
