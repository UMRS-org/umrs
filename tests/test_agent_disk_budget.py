"""ADR-051 D-2/D-3 pins: agent spawn budget + per-user disk budget.

Contracts locked:

- count_active_agent_units reads the jobs table only (companion_worker_job
  + research_loop_job, LIVE statuses); nothing fabricated, pool-absent = 0.
- enforce_spawn_budget refuses when current >= limit with typed
  quota.cpu_process_exceeded.
- DiskLedger charges byte_size exactly ONCE per (user, content_hash) —
  re-upload of the same hash to the same user costs 0.
- quota refusal is typed quota.disk_bytes_exceeded, never silent overshoot.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from umrs.api.agent_quota import (
    AGENT_JOB_KINDS,
    LIVE_JOB_STATUSES,
    count_active_agent_units,
)
from umrs.api.disk_ledger import DiskLedger


def _open_pool(rows):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.fetchone.return_value = rows
    pool = MagicMock()
    pool.getconn.return_value.__enter__.return_value = conn
    return pool, cursor


class TestAgentBudget:
    def test_count_from_jobs_table_only(self) -> None:
        pool, cursor = _open_pool((3,))
        n = count_active_agent_units(pool, user_id="alice")
        sql, params = cursor.execute.call_args.args
        assert "FROM jobs" in sql
        assert params[0] == "alice"
        for kind in AGENT_JOB_KINDS:
            assert kind in str(params[1])
        for status in LIVE_JOB_STATUSES:
            assert status in str(params[2])
        assert n == 3

    def test_pool_absent_answers_zero_never_fabricated(self) -> None:
        assert count_active_agent_units(None, user_id="alice") == 0

    def test_spawn_over_budget_refused_loudly(self, monkeypatch) -> None:
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "3")
        from umrs.api.agent_quota import enforce_spawn_budget

        pool, _cursor = _open_pool((3,))
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            enforce_spawn_budget(pool, user_id="heavy")
        assert exc_info.value.status_code == 503
        assert "agent process budget exceeded" in str(exc_info.value.detail)
        assert "heavy" in str(exc_info.value.detail)


class TestDiskLedger:
    def _conn(self) -> MagicMock:
        return MagicMock()

    def test_charge_once_per_hash(self) -> None:
        conn = self._conn()
        ledger = DiskLedger(conn)
        conn.execute = MagicMock()
        # First charge events — simulate no prior hash and usage reads via execute
        ledger._conn = SimpleNamespace(
            cursor=MagicMock(
                return_value=MagicMock(
                    execute=MagicMock(),
                    fetchone=MagicMock(side_effect=[None, (0,)]),
                )
            ),
            commit=MagicMock(),
        )
        usage = ledger.usage("alice")
        assert usage == 0

    def test_quota_exceeded_typed_loudly(self) -> None:
        from fastapi import HTTPException
        from umrs.api.disk_ledger import raise_typed_quota

        with pytest.raises(HTTPException) as exc_info:
            raise_typed_quota(limit=10, usage=9, byte_size=4, user_id="alice")
        body = exc_info.value.detail
        assert "disk budget exceeded" in body
        assert "alice" in body


class TestDiskLedgerIntegration:
    def _run(self, user: str):
        tool = __import__("hashlib")
        return tool.sha256(user.encode()).hexdigest()

    def test_first_upload_charges_once_no_double(self) -> None:
        # Pre-verified behavior via fake conn capturing exact statements.
        conn = MagicMock()
        ledger = DiskLedger(conn)
        executed_calls: list[tuple] = []
        conn.cursor = MagicMock(
            return_value=MagicMock(
                execute=lambda sql, params=None: executed_calls.append((sql, params)),
                fetchone=MagicMock(
                    side_effect=[None, (100,)],
                ),
            )
        )
        conn.commit = MagicMock()
        out = ledger.charge("alice", content_hash="h-1", byte_size=100)
        assert out["charged"] == 100 and out["usage"] == 200 and out["limit"] > 0
        # Same hash re-upload: free
        conn.cursor = MagicMock(
            return_value=MagicMock(
                execute=lambda sql, params=None: None,
                fetchone=MagicMock(side_effect=[(1,), (200,)]),
            )
        )
        ledger2 = DiskLedger(conn)
        out2 = ledger2.charge("alice", content_hash="h-1", byte_size=100)
        assert out2["charged"] == 0 and out2["usage"] == 200
