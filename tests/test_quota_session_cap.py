"""ADR-099 D-6 (RS-1 lane Q): live remote sessions per user, 12 by default.

- the start's quota check is the session cap of the start's resolved policy:
  the 13th live ``frontend_session`` of a user at this venue is a typed 429
  ``quota.concurrent_sessions_exceeded`` (before: the 4-process budget);
- ``frontend_session`` left the ADR-051 agent-process budget, so a 5th live
  session no longer trips ``quota.cpu_process_exceeded``; companion workers
  and loop jobs still count there.

Lane P's ``StartCheck`` is a plain stand-in here (contract §1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException
from umrs.api.agent_quota import (
    AGENT_JOB_KINDS,
    count_active_agent_units,
    enforce_spawn_budget,
)
from umrs.api.companion_venue import _quota_check
from umrs.quotas.session_cap import (
    check_concurrent_sessions,
    concurrent_per_user_default,
    count_live_sessions,
    session_limits,
)


@dataclass(frozen=True)
class StartCheck:  # the shape of lane P's umrs.venues.start_checks.StartCheck
    owner: str
    project: str
    venue: str
    session_id: str
    policy: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Job:
    user_id: str
    job_type: str
    status: str = "running"
    venue: str = "rccs"
    project_id: str = "default"
    hosting: str = ""


class _JobsPool:
    """Answers the two COUNT(*) FROM jobs statements from an in-memory table."""

    def __init__(self, jobs):
        self.jobs = list(jobs)
        self.sql: list = []

    def getconn(self):
        return _JobsConn(self)

    def putconn(self, _conn):
        pass


class _JobsConn:
    def __init__(self, pool):
        self.pool = pool
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params):
        self.pool.sql.append((sql, params))
        match = _agent_match if "spec->>'hosting'" in sql else _session_match
        self._row = (sum(1 for job in self.pool.jobs if match(sql, params, job)),)

    def fetchone(self):
        return self._row


def _agent_match(_sql, params, job):
    """agent_quota.count_active_agent_units' statement."""
    user, kinds, statuses, hostings = params
    return (
        job.user_id == user
        and job.job_type in kinds
        and job.status in statuses
        and job.hosting not in hostings
    )


def _session_match(sql, params, job):
    """session_cap.count_live_sessions' statement: ``col = %s`` / ``col = ANY(%s)``."""
    columns = [c.strip().split(" ")[0] for c in sql.split("WHERE", 1)[1].split(" AND ")]
    for column, value in zip(columns, params, strict=False):
        have = getattr(job, column)
        if (have not in value) if isinstance(value, list) else (have != value):
            return False
    return True


def _sessions(n, *, user="alice", venue="rccs", project="default", status="running"):
    return [
        _Job(
            user_id=user,
            job_type="frontend_session",
            venue=venue,
            project_id=project,
            status=status,
        )
        for _ in range(n)
    ]


def _check(policy=None, *, owner="alice", project="opossum"):
    return StartCheck(
        owner=owner,
        project=project,
        venue="rccs",
        session_id="sess-new",
        policy=policy if policy is not None else {},
    )


def _policy(user=12, project=None):
    return {"sessions": {"concurrent_user": user, "concurrent_project": project}}


class TestSessionCap:
    def test_the_13th_live_session_is_refused(self):
        pool = _JobsPool(_sessions(12))
        with pytest.raises(HTTPException) as exc:
            check_concurrent_sessions(pool, _check(_policy()))
        err = exc.value
        assert err.status_code == 429
        assert err.headers["x-platform-error-code"] == "quota.concurrent_sessions_exceeded"
        assert "12 of 12 live sessions" in err.detail and "user alice at rccs" in err.detail

    def test_the_12th_is_admitted(self):
        check_concurrent_sessions(_JobsPool(_sessions(11)), _check(_policy()))

    def test_only_live_sessions_of_this_user_at_this_venue_count(self):
        jobs = (
            _sessions(11)
            + _sessions(5, status="succeeded")
            + _sessions(5, user="bob")
            + _sessions(5, venue="ai4s")
            + [_Job(user_id="alice", job_type="companion_worker")]
        )
        pool = _JobsPool(jobs)
        assert count_live_sessions(pool, venue="rccs", user_id="alice") == 11
        check_concurrent_sessions(pool, _check(_policy()))

    def test_project_cap_when_set(self):
        pool = _JobsPool(_sessions(3, user="bob", project="opossum"))
        with pytest.raises(HTTPException) as exc:
            check_concurrent_sessions(pool, _check(_policy(user=12, project=3)))
        assert "project opossum" in exc.value.detail

    def test_null_cap_means_no_limit(self):
        check_concurrent_sessions(_JobsPool(_sessions(40)), _check(_policy(user=None)))

    def test_missing_key_is_the_site_default(self, monkeypatch):
        monkeypatch.delenv("UMRS_SESSION_CONCURRENT_PER_USER", raising=False)
        assert session_limits({})["concurrent_user"] == 12
        with pytest.raises(HTTPException):
            check_concurrent_sessions(_JobsPool(_sessions(12)), _check({}))
        monkeypatch.setenv("UMRS_SESSION_CONCURRENT_PER_USER", "2")
        with pytest.raises(HTTPException):
            check_concurrent_sessions(_JobsPool(_sessions(2)), _check({}))

    @pytest.mark.parametrize(
        ("raw", "want"), [("", 12), ("x", 12), ("-3", 12), ("0", None), ("20", 20)]
    )
    def test_env_default(self, raw, want):
        assert concurrent_per_user_default({"UMRS_SESSION_CONCURRENT_PER_USER": raw}) == want

    def test_start_wiring_uses_the_session_cap(self):
        hook = _quota_check(_JobsPool(_sessions(12)))
        with pytest.raises(HTTPException) as exc:
            hook(_check(_policy()))
        assert exc.value.headers["x-platform-error-code"] == ("quota.concurrent_sessions_exceeded")
        hook = _quota_check(_JobsPool(_sessions(4)))
        hook(_check(_policy()))  # 4 live sessions: under 12, no process budget


class TestAgentProcessBudget:
    def test_frontend_session_left_the_budget(self):
        assert "frontend_session" not in AGENT_JOB_KINDS
        assert set(AGENT_JOB_KINDS) == {"companion_worker", "research_loop_job"}

    def test_a_5th_session_no_longer_trips_cpu_process_exceeded(self, monkeypatch):
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "4")
        pool = _JobsPool(_sessions(5))
        assert count_active_agent_units(pool, user_id="alice") == 0
        enforce_spawn_budget(pool, user_id="alice")

    def test_workers_and_loops_still_count(self, monkeypatch):
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "4")
        jobs = [_Job(user_id="alice", job_type="companion_worker") for _ in range(2)] + [
            _Job(user_id="alice", job_type="research_loop_job") for _ in range(2)
        ]
        with pytest.raises(HTTPException) as exc:
            enforce_spawn_budget(_JobsPool(jobs + _sessions(3)), user_id="alice")
        assert exc.value.headers["x-platform-error-code"] == "quota.cpu_process_exceeded"
