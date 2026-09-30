"""Pins for drop_db / drop_db_user (deprovision path, 2026-09-30).

Found live: `admin user delete --purge --deprovision` died on
``ImportError: cannot import name 'drop_db'`` during the batch-benchmark
cleanup — the deprovision path imported helpers that never existed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from umrs.server import sql as server_sql


def _fake_cfg():
    return {
        "host": "h",
        "port": 5432,
        "superuser": "su",
        "superuser_password": "pw",
        "user_password": "upw",
    }


def test_drop_db_terminates_sessions_and_drops():
    cur = MagicMock()
    cur.fetchone.return_value = (1,)  # database exists
    conn = MagicMock()
    conn.cursor.return_value = cur
    with (
        patch.object(server_sql, "get_db_config", return_value=_fake_cfg()),
        patch.object(server_sql.psycopg2, "connect", return_value=conn),
        patch("psycopg2.sql.SQL") as mock_sql_cls,
    ):
        server_sql.drop_db("default_bench_bench")
    terminated = any(
        "pg_terminate_backend" in str(c.args[0]) for c in cur.execute.mock_calls
    )
    assert terminated
    mock_sql_cls.assert_called_with("DROP DATABASE {};")
    conn.close.assert_called_once()


def test_drop_db_noop_when_absent():
    cur = MagicMock()
    cur.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cur
    with (
        patch.object(server_sql, "get_db_config", return_value=_fake_cfg()),
        patch.object(server_sql.psycopg2, "connect", return_value=conn),
        patch("psycopg2.sql.SQL") as mock_sql_cls,
    ):
        server_sql.drop_db("ghost_db")
    mock_sql_cls.assert_not_called()


def test_drop_db_user_drops_existing_role():
    cur = MagicMock()
    conn = MagicMock()
    conn.cursor.return_value = cur
    with (
        patch.object(server_sql, "get_db_config", return_value=_fake_cfg()),
        patch.object(server_sql.psycopg2, "connect", return_value=conn),
        patch.object(server_sql, "db_user_exists", return_value=True),
        patch("psycopg2.sql.SQL") as mock_sql_cls,
    ):
        server_sql.drop_db_user("benchuser")
    mock_sql_cls.assert_called_with("DROP USER {};")
    conn.close.assert_called_once()
