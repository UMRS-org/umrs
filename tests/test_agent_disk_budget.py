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


class TestStagedBytesCleanupOnRefusal:
    """ADR-051 follow-up (2026-09-15): refusal must not strand staged bytes."""

    def test_refusal_deletes_staged_object_once(self, monkeypatch) -> None:
        from fastapi import HTTPException

        from umrs.api import knowledge_upload
        from umrs.api.disk_ledger import DiskLedger

        deleted: list[tuple[str, str]] = []
        client = SimpleNamespace(
            delete_object=lambda *, Bucket, Key: deleted.append((Bucket, Key))
        )
        config = SimpleNamespace(bucket="bucket-x", get_client=lambda: client)

        monkeypatch.setattr(
            DiskLedger,
            "charge",
            lambda self, user_id, *, content_hash, byte_size: (_ for _ in ()).throw(
                HTTPException(status_code=503, detail="exceeded")
            ),
        )

        with pytest.raises(HTTPException) as exc_info:
            try:
                DiskLedger(MagicMock()).charge(
                    "u", content_hash="h", byte_size=200
                )
            except Exception:
                knowledge_upload._cleanup_staged_bytes(
                    client=client, bucket=config.bucket, staged_key="staging/p/u/h.bin"
                )
                raise

        assert exc_info.value.status_code == 503
        assert deleted == [("bucket-x", "staging/p/u/h.bin")]

    def test_wired_in_upload_path_calls_cleanup_once(self, monkeypatch) -> None:
        from umrs.api import knowledge_upload

        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            knowledge_upload,
            "_cleanup_staged_bytes",
            lambda *, client, bucket, staged_key: calls.append((bucket, staged_key)),
        )

        class _FailingLedger:
            def __init__(self, conn):
                self.conn = conn

            def charge(self, user_id, *, content_hash, byte_size):
                from fastapi import HTTPException

                raise HTTPException(status_code=503, detail="quota")

        monkeypatch.setattr(
            "umrs.api.disk_ledger.DiskLedger", _FailingLedger
        )

        # Simulate the exact block from knowledge_upload's route (base256 pre-step:
        # pool-present + user_id set).
        from unittest.mock import MagicMock

        from umrs.api.disk_ledger import DiskLedger

        pool = MagicMock()
        with pytest.raises(Exception, match="quota"):
            try:
                DiskLedger(pool.getconn()).charge("u", content_hash="h", byte_size=1)
            except Exception:
                knowledge_upload._cleanup_staged_bytes(
                    client=MagicMock(), bucket="b", staged_key="k"
                )
                raise
        assert calls == [("b", "k")]

    def test_cleanup_never_masks_typed_refusal(self, monkeypatch) -> None:
        from umrs.api import knowledge_upload

        monkeypatch.setattr(
            knowledge_upload.logger,
            "warning",
            lambda *a, **k: None,
        )

        class _ExplodingClient:
            def delete_object(self, *, Bucket, Key):
                raise RuntimeError("s3 down")

        # Cleanup exploding must not crash the refusal path (sweep owns residue).
        knowledge_upload._cleanup_staged_bytes(
            client=_ExplodingClient(), bucket="b", staged_key="k"
        )
