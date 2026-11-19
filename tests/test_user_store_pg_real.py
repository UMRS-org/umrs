"""Real-Postgres twin of ``test_user_store_pg.py`` for the 0110/0113 lane.

The tests are env-gated like ``test_companion_snapshot_lww_pg``, but require
its explicit ``UMRS_TEST_DB_URL`` form so ambient ``POSTGRES_*`` settings can
never select a database. Every statement runs inside a scratch schema.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_companion_snapshot_lww_pg import _db_url
from umrs.admin.user_store import (
    UserRecord,
    UserStore,
    configure_user_store_pg,
    mirror_record_to_pg,
    pg_get,
    pg_lane_active,
    reset_user_store_pg,
)

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_M0110 = _MIGRATIONS / "0110_users.sql"
_M0113 = _MIGRATIONS / "0113_users_venue_scoping.sql"
_SCHEMA = f"wp_d06_{os.getpid()}"

pytestmark = pytest.mark.skipif(
    os.environ.get("UMRS_TEST_DB_URL") is None,
    reason="UMRS_TEST_DB_URL is not set",
)


def _test_db_url() -> str:
    explicit = os.environ["UMRS_TEST_DB_URL"]
    assert _db_url() == explicit
    return explicit


def _connect():
    import psycopg2

    return psycopg2.connect(_test_db_url(), options=f"-c search_path={_SCHEMA}")


def _admin_connect():
    import psycopg2

    return psycopg2.connect(_test_db_url())


class _SchemaPool:
    def getconn(self):
        return _connect()

    def putconn(self, conn):
        conn.close()


def _sql(conn, text, params=None):
    with conn.cursor() as cur:
        cur.execute(text, params)
        rows = cur.fetchall() if cur.description is not None else []
    conn.commit()
    return rows


def _drop_schema(conn) -> None:
    from psycopg2 import sql

    with conn.cursor() as cur:
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(_SCHEMA)))
    conn.commit()


@pytest.fixture()
def schema():
    from psycopg2 import sql

    conn = _admin_connect()
    try:
        _drop_schema(conn)
        with conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(_SCHEMA)))
        conn.commit()
    finally:
        conn.close()
    yield
    reset_user_store_pg()
    conn = _admin_connect()
    try:
        _drop_schema(conn)
    finally:
        conn.close()


@pytest.fixture()
def migrated(schema):
    conn = _connect()
    try:
        _sql(conn, _M0110.read_text())
        _sql(conn, _M0113.read_text())
    finally:
        conn.close()
    yield _SchemaPool()


def _user(tmp_path, site, name, **over):
    store = UserStore(tmp_path / site)
    store.create(name, **over)
    return store


def test_0113_keeps_pre_0113_rows_under_the_empty_venue(schema):
    conn = _connect()
    try:
        _sql(conn, _M0110.read_text())
        _sql(conn, "INSERT INTO users (username, role) VALUES ('old', 'admin')")
        _sql(conn, _M0113.read_text())
        rows = _sql(conn, "SELECT venue, username, role FROM users")
        key_columns = _sql(
            conn,
            """SELECT a.attname FROM pg_index i
               JOIN pg_attribute a ON a.attrelid = i.indrelid
                                  AND a.attnum = ANY(i.indkey)
               WHERE i.indrelid = 'users'::regclass AND i.indisprimary
               ORDER BY array_position(i.indkey, a.attnum)""",
        )
    finally:
        conn.close()
    assert rows == [("", "old", "admin")]
    assert [row[0] for row in key_columns] == ["venue", "username"]


def test_both_migrations_reapply_without_error_or_row_loss(migrated, tmp_path):
    record = _user(tmp_path, "rccs", "alice").get("alice")
    mirror_record_to_pg(migrated, record, venue="rccs")
    conn = _connect()
    try:
        _sql(conn, _M0110.read_text())
        _sql(conn, _M0113.read_text())
        count = _sql(conn, "SELECT count(*) FROM users")
    finally:
        conn.close()
    assert count == [(1,)]


def test_boot_sync_twice_is_idempotent_and_stays_in_its_venue(migrated, tmp_path):
    rccs = _user(tmp_path, "rccs", "alice")
    _user(tmp_path, "rccs", "carol")
    _user(tmp_path, "ai4s", "bob")
    assert configure_user_store_pg(migrated, store=rccs, venue="rccs") == 2
    assert configure_user_store_pg(migrated, store=rccs, venue="rccs") == 2
    ai4s = UserStore(tmp_path / "ai4s")
    assert configure_user_store_pg(migrated, store=ai4s, venue="ai4s") == 1
    conn = _connect()
    try:
        rows = _sql(conn, "SELECT venue, username FROM users ORDER BY 1, 2")
    finally:
        conn.close()
    assert rows == [("ai4s", "bob"), ("rccs", "alice"), ("rccs", "carol")]


def test_every_field_round_trips_through_the_real_row(migrated, tmp_path):
    store = _user(
        tmp_path,
        "rccs",
        "alice",
        fullname="A L",
        email="a@example.org",
        role="admin",
        slurm_account="grp-a",
        projects=["p1", "p2"],
        quota={"cpu": 4},
        oidc_subject="241723",
        extras={"s3_configuration_file": "x.cfg"},
    )
    store.update(
        "alice",
        slurm_accounts={"rccs": "grp-r"},
        curve_public_key="k" * 40,
        curve_key_revoked_at="2026-01-01T00:00:00Z",
    )
    assert configure_user_store_pg(migrated, store=store, venue="rccs") == 1
    record = pg_get("alice")
    assert record is not None
    expected = json.loads((tmp_path / "rccs" / "alice.json").read_text())
    assert record.to_json() == expected


def test_one_oidc_subject_per_venue_is_enforced_by_the_index(migrated, tmp_path):
    import psycopg2

    alice = _user(tmp_path, "rccs", "alice", oidc_subject="s1").get("alice")
    bob = UserRecord(username="bob", oidc_subject="s1")
    mirror_record_to_pg(migrated, alice, venue="rccs")
    mirror_record_to_pg(migrated, bob, venue="ai4s")
    with pytest.raises(psycopg2.errors.UniqueViolation):
        mirror_record_to_pg(migrated, bob, venue="rccs")


def test_a_failed_first_sync_is_repaired_by_the_next_boot(migrated, tmp_path):
    class _UnavailablePool:
        def getconn(self):
            raise RuntimeError("pg down")

        def putconn(self, _conn):
            pass

    store = _user(tmp_path, "rccs", "alice")
    assert configure_user_store_pg(_UnavailablePool(), store=store, venue="rccs") == 0
    assert pg_lane_active() is False
    assert configure_user_store_pg(migrated, store=store, venue="rccs") == 1
    assert pg_lane_active() is True
    record = pg_get("alice")
    assert record is not None
    assert record.username == "alice"


def test_unwiring_the_lane_loses_nothing_json_has_every_write(migrated, tmp_path):
    store = _user(tmp_path, "rccs", "alice")
    assert configure_user_store_pg(migrated, store=store, venue="rccs") == 1
    store.set_disabled("alice", True)
    store.add_project("alice", "p9")
    mirrored = pg_get("alice")
    assert mirrored is not None
    assert mirrored.disabled is True
    assert "p9" in mirrored.projects
    reset_user_store_pg()
    recovered = store.get("alice")
    assert recovered.disabled is True
    assert "p9" in recovered.projects
