"""ADR-114 session bus + ADR-046 worker pop on real Postgres (issue r4#849).

The bus lock joined ``companion_session_archives`` with ``LEFT JOIN`` and
locked with a bare ``FOR UPDATE``; Postgres refuses that ("FOR UPDATE cannot
be applied to the nullable side of an outer join"), so every bind, send,
inbox read and the venue worker's queue pop answered 500 on the fronts from
2026-11-10. The fake pool in ``test_companion_session_bus`` never parses SQL.

Env-gated like ``test_companion_snapshot_lww_pg``: runs where the CI job's
``POSTGRES_*`` variables (or ``UMRS_TEST_DB_URL``) point at a database.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from test_companion_snapshot_lww_pg import _db_url
from umrs.api._companion_session_bus import (
    SessionBusBindBody,
    SessionBusInboxBody,
    SessionBusSendBody,
    _locked,
    bind_presence_row,
    inbox_row,
    send_message_row,
    worker_pop_row,
)
from umrs.api.companion_sessions import queue_mutate_row, upsert_snapshot_row

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_USER = "bus-pg-test"
_SIDS = ("bus-pg-a", "bus-pg-b", "bus-pg-archived")
_SCHEMA = (
    "0079_companion_trust_session_cascade.sql",
    "0080_companion_session_snapshots.sql",
    "0084_companion_session_registry.sql",
)

pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


def _clean(connection) -> None:
    with connection.cursor() as cur:
        cur.execute("DELETE FROM companion_session_snapshots WHERE user_id = %s", (_USER,))
        cur.execute(
            "DELETE FROM companion_session_archives WHERE session_id = ANY(%s)", (list(_SIDS),)
        )
    connection.commit()


def _connect():
    import psycopg2

    return psycopg2.connect(_db_url())


@pytest.fixture()
def conn():
    connection = _connect()
    with connection.cursor() as cur:
        for name in _SCHEMA:
            cur.execute((_MIGRATIONS / name).read_text())
    connection.commit()
    _clean(connection)
    yield connection
    connection.rollback()
    _clean(connection)
    connection.close()


def _seed(conn, sid: str) -> None:
    upsert_snapshot_row(
        conn,
        _USER,
        sid,
        name=sid,
        payload=json.dumps({"id": sid, "metadata": {}}),
        updated_at="2026-11-11T00:00:00+00:00",
        venue="local",
        state="ongoing",
    )


def _bind(conn, sid: str) -> dict:
    return bind_presence_row(conn, _USER, sid, SessionBusBindBody(bus_session_id=sid, name=sid))


def test_worker_pop_pops_the_queued_turn(conn):
    _seed(conn, "bus-pg-a")
    queue_mutate_row(conn, _USER, "bus-pg-a", action="append", content="next turn")
    result = worker_pop_row(conn, _USER, "bus-pg-a")
    assert result["front"]["content"] == "next turn"
    assert result["queue"] == []


def test_bind_send_and_take_round_trip(conn):
    _seed(conn, "bus-pg-a")
    _seed(conn, "bus-pg-b")
    assert _bind(conn, "bus-pg-a")["bound"] is True
    assert _bind(conn, "bus-pg-b")["bound"] is True
    send = SessionBusSendBody(message_id="msg-pg849aaaaaaa", recipient_id="bus-pg-b", body="hello")
    assert send_message_row(conn, _USER, "bus-pg-a", send)["duplicate"] is False
    taken = inbox_row(conn, _USER, "bus-pg-b", SessionBusInboxBody(action="take"))
    assert [m["body"] for m in taken["messages"]] == ["hello"]


def test_archived_tombstone_still_refuses_presence(conn):
    _seed(conn, "bus-pg-archived")
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO companion_session_archives (session_id) VALUES (%s)",
            ("bus-pg-archived",),
        )
    conn.commit()
    with pytest.raises(HTTPException) as refused:
        _bind(conn, "bus-pg-archived")
    assert refused.value.status_code == 409
    assert "not accepting" in str(refused.value.detail)


def test_bus_lock_still_holds_the_snapshot_row(conn):
    import psycopg2.errors

    _seed(conn, "bus-pg-a")
    queue_mutate_row(conn, _USER, "bus-pg-a", action="append", content="held")
    with conn.cursor() as cur:
        _locked(cur, _USER, "bus-pg-a")
        other = _connect()
        try:
            with other.cursor() as probe, pytest.raises(psycopg2.errors.LockNotAvailable):
                probe.execute(
                    "SELECT 1 FROM companion_session_snapshots"
                    " WHERE user_id = %s AND session_id = %s FOR UPDATE NOWAIT",
                    (_USER, "bus-pg-a"),
                )
        finally:
            other.rollback()
            other.close()
    conn.rollback()
