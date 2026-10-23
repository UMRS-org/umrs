"""Agent CPU-process budget (ADR-051 D-2).

Accounting model: dorable jobs are the ONLY source — count non-terminal
agent jobs owned by ONE user on the frontend (companion-worker jobs +
loop-hosted jobs). The claim closes constitutionally at a terminal
status; a process crash can NOT silently pin someone's budget, and there
is no heartbeat-tale second ledger anywhere.

Enforcement: 503 typed at spawn once per spawn point on the frontend —
never a side pass, never a background quota thread.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException

from umrs.api.typed_errors import PlatformError
from umrs.jobs.plan_run_records import PLAN_RUN_HOSTINGS


def agent_max_processes() -> int:
    """Spawn ceiling for concurrent frontend agent units (0 = uncapped)."""
    return max(0, int(os.environ.get("UMRS_AGENT_PROCESSES_PER_USER_MAX", "4")))


# Job kinds counted as CPU-process units on this venue.
# JobType VALUES (the 2026-09-21 audit found "companion_worker_job" here, which
# never matched JobType.COMPANION_WORKER = "companion_worker" — companion units
# were never counted). ADR-099 D-6 (2026-10-23): ``frontend_session`` (a remote
# companion sandbox, ADR-046 A2) left this budget; remote sessions have their
# own cap (``umrs.quotas.session_cap``, 12 per user), which a budget of 4
# would otherwise undercut.
AGENT_JOB_KINDS: tuple[str, ...] = (
    "companion_worker",
    "research_loop_job",
)

# Live job statuses still consuming a unit (terminal closes the claim).
LIVE_JOB_STATUSES: tuple[str, ...] = ("planned", "queued", "running")

# ADR-092 slice 7: a research_loop_job its client hosts (a team plan run's
# durable record, ``spec.hosting = 'client'``) runs no process here, and a
# client that dies leaves it RUNNING until a resume or a restart: never a unit.
# Slice 8: a run its venue worker hosts (``'venue'``) is not a unit either;
# its worker's ``frontend_session`` job counts toward the remote-session cap.
PLAN_RUN_RECORD_HOSTINGS = sorted(PLAN_RUN_HOSTINGS)


def count_active_agent_units(pool: Any, *, user_id: str) -> int:
    """Count LIVE agent jobs owned by one user (never fabricated).

    Uses the sync psycopg pool; the pool being absent answers 0 decently
    (never a fabricated 503)."""
    if pool is None:
        return 0
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT COUNT(*) FROM jobs
                       WHERE user_id = %s
                         AND job_type = ANY(%s)
                         AND status = ANY(%s)
                         AND COALESCE(spec->>'hosting', '') <> ALL(%s)""",
                    (
                        user_id,
                        list(AGENT_JOB_KINDS),
                        list(LIVE_JOB_STATUSES),
                        PLAN_RUN_RECORD_HOSTINGS,
                    ),
                )
                row = cur.fetchone()
    finally:
        pool.putconn(conn)
    return int(row[0]) if row else 0


def enforce_spawn_budget(pool: Any, *, user_id: str) -> dict[str, int]:
    """Check the spawn budget; typed 503 when the ceiling is reached.

    Returns ``{"current": n, "limit": m}`` for honest logging."""
    limit = agent_max_processes()
    current = count_active_agent_units(pool, user_id=user_id) if limit > 0 else 0
    if limit > 0 and current >= limit:
        raise HTTPException(
            **PlatformError.quota(
                "cpu_process_exceeded",
                f"agent process budget exceeded for user {user_id}: "
                f"current={current} limit={limit}",
            ).to_http_parts()
        )
    return {"current": current, "limit": limit}


__all__ = [
    "AGENT_JOB_KINDS",
    "LIVE_JOB_STATUSES",
    "agent_max_processes",
    "count_active_agent_units",
    "enforce_spawn_budget",
]
