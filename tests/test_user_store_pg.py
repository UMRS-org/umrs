"""D-06 first-leg pins: accounts into Postgres beside projects (0110).

The JSON lane stays the bootstrap/recovery lane; the PG lane goes
read-first once wired; writes dual-write; unwired = the exact
pre-0110 behavior (the existing admin suites ride that path).
"""

from __future__ import annotations

import pytest
from umrs.admin.user_store import (
    UserStore,
    configure_user_store_pg,
    pg_lane_active,
    reset_user_store_pg,
)


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result: list = []

    def execute(self, sql, params=None):
        self._conn._handle(sql, params or (), self)

    def fetchone(self):
        return self._result.pop(0) if self._result else None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Conn:
    """psycopg-shaped fake over a dict keyed by username."""

    def __init__(self):
        self.rows: dict[str, dict] = {}

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def _handle(self, sql, params, cur):
        if "INSERT INTO users" in sql and "ON CONFLICT" in sql:
            row = dict(zip(_COLS, params, strict=False))
            self.rows[row["username"]] = row
            cur._result = []
        elif "FROM users WHERE username = %s" in sql:
            cur._result = (
                [tuple(self.rows[params[0]][c] for c in _COLS)] if params[0] in self.rows else []
            )
        elif sql.strip().startswith("SELECT"):
            cur._result = [tuple(r[c] for c in _COLS) for r in self.rows.values()]


_COLS = [
    "username",
    "fullname",
    "email",
    "role",
    "disabled",
    "slurm_account",
    "slurm_accounts",
    "projects",
    "quota",
    "oidc_subject",
    "curve_public_key",
    "curve_key_revoked_at",
    "extras",
    "created_at",
    "last_updated",
]


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, _c):
        pass


@pytest.fixture(autouse=True)
def _unwire():
    reset_user_store_pg()
    yield
    reset_user_store_pg()


def _make_user(tmp_path, name="alice", **over):
    store = UserStore(tmp_path)
    store.create(name, **over)
    return store


def test_sync_carries_json_rows_into_pg(tmp_path):
    _make_user(tmp_path, "alice", role="admin", slurm_account="group-a")
    _make_user(tmp_path, "bob")
    conn = _Conn()
    synced = configure_user_store_pg(_Pool(conn), store=UserStore(tmp_path))
    assert synced == 2
    assert conn.rows["alice"]["role"] == "admin"
    assert conn.rows["bob"]["role"] == "user"


def test_get_is_pg_first_when_wired(tmp_path, monkeypatch):
    """The wired lane reads PG first; an offline-CLI JSON-only row still falls back."""
    _make_user(tmp_path, "alice")
    conn = _Conn()
    configure_user_store_pg(_Pool(conn), store=UserStore(tmp_path))
    assert pg_lane_active() is True
    # PG carries the synced row — mutate it, the read must see the PG truth
    conn.rows["alice"]["role"] = "admin"
    assert UserStore(tmp_path).get("alice").role == "admin"
    # a JSON-only row (offline CLI created it after the sync) falls back
    store = UserStore(tmp_path)
    store.create("offline-user")
    assert UserStore(tmp_path).get("offline-user").username == "offline-user"


def test_writes_dual_write(tmp_path):
    _make_user(tmp_path, "alice")
    conn = _Conn()
    configure_user_store_pg(_Pool(conn), store=UserStore(tmp_path))
    UserStore(tmp_path).set_role("alice", "admin")
    assert conn.rows["alice"]["role"] == "admin"  # PG mirrored
    assert UserStore(tmp_path).get("alice").role == "admin"  # JSON truth too


def test_unwired_is_the_pre_0110_behavior(tmp_path):
    assert pg_lane_active() is False
    _make_user(tmp_path, "alice")
    assert UserStore(tmp_path).get("alice").username == "alice"
    # create/update/get all ride the JSON lane exactly as before


def test_sync_failure_fails_open_to_json(tmp_path, caplog):
    class _BoomPool:
        def getconn(self):
            raise RuntimeError("pg down")

        def putconn(self, _c):
            pass

    _make_user(tmp_path, "alice")
    assert configure_user_store_pg(_BoomPool()) == 0
    assert pg_lane_active() is False  # unwired — JSON authoritative
    assert any("stays authoritative" in r.getMessage() for r in caplog.records)
    assert UserStore(tmp_path).get("alice").username == "alice"


def test_curve_authority_fields_round_trip_through_pg(tmp_path):
    """D-03's account keystore fields ride the same sync (0110 columns)."""
    store = _make_user(tmp_path, "alice")
    import tempfile

    from umrs.core.zmq_security import generate_curve_keypair

    with tempfile.TemporaryDirectory() as td:
        pub, _ = generate_curve_keypair(td, "alice")
        from umrs.core.zmq_security import load_public_key_z85

        pub_z85 = load_public_key_z85(pub)
        store.update("alice", curve_public_key=pub_z85)
    conn = _Conn()
    configure_user_store_pg(_Pool(conn), store=UserStore(tmp_path))
    assert conn.rows["alice"]["curve_public_key"] == pub_z85
