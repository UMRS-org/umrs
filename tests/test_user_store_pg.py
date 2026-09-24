"""D-06 first-leg pins: accounts into Postgres beside projects (0110).

The JSON lane stays the bootstrap/recovery lane; the PG lane goes
read-first once wired; writes dual-write; unwired = the exact
pre-0110 behavior (the existing admin suites ride that path).
"""

from __future__ import annotations

import pytest
from umrs.admin.user_store import (
    UserStore,
    UserStoreError,
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

    def fetchall(self):
        rows, self._result = self._result, []
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Conn:
    """psycopg-shaped fake over a table keyed by (venue, username) — 0113.

    ``rows`` is the venue-less view of the unscoped lane (venue ""), kept so
    the pre-0113 pins read unchanged; ``table`` is the full keyed store.
    """

    def __init__(self):
        self.table: dict[tuple[str, str], dict] = {}

    @property
    def rows(self) -> dict[str, dict]:
        return {u: r for (v, u), r in self.table.items() if v == ""}

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def _handle(self, sql, params, cur):
        if "INSERT INTO users" in sql and "ON CONFLICT (venue, username)" in sql:
            row = dict(zip(_COLS, params, strict=True))
            self.table[(row["venue"], row["username"])] = row
            cur._result = []
        elif "FROM users WHERE venue = %s AND username = %s" in sql:
            row = self.table.get((params[0], params[1]))
            cur._result = [_select_row(row)] if row else []
        elif "FROM users WHERE venue = %s" in sql:
            cur._result = self._venue_rows(params[0])
        else:
            raise AssertionError(f"unscoped users SQL: {sql}")

    def _venue_rows(self, venue):
        return [_select_row(r) for (v, _u), r in sorted(self.table.items()) if v == venue]


def _select_row(row):
    """The SELECT column order (every column but the venue scope)."""
    return tuple(row[c] for c in _COLS[1:])


_COLS = [
    "venue",
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


# ── 0113: venue scoping — servers sharing one umrs_db never mix accounts ────


def _site_dir(tmp_path, site, **users):
    d = tmp_path / site
    store = UserStore(d)
    for name, over in users.items():
        store.create(name, **over)
    return d


def test_two_sites_sharing_one_db_keep_their_own_rows(tmp_path):
    """The umrs-front shape: four servers, one umrs_db, separate JSON dirs."""
    rccs = _site_dir(tmp_path, "rccs", alice={"role": "admin", "slurm_account": "rccs-a"})
    ai4s = _site_dir(tmp_path, "ai4s", alice={"role": "user", "slurm_account": "ai4s-a"})
    conn = _Conn()
    configure_user_store_pg(_Pool(conn), store=UserStore(rccs), venue="rccs")
    configure_user_store_pg(_Pool(conn), store=UserStore(ai4s), venue="ai4s")
    assert conn.table[("rccs", "alice")]["role"] == "admin"
    assert conn.table[("ai4s", "alice")]["role"] == "user"  # last boot did not win

    # the ai4s server (wired last) reads its own row, never rccs's admin role
    record = UserStore(ai4s).get("alice")
    assert record.role == "user"
    assert record.slurm_account == "ai4s-a"


def test_dual_write_stays_in_its_venue(tmp_path):
    rccs = _site_dir(tmp_path, "rccs", alice={})
    conn = _Conn()
    conn.table[("ai4s", "alice")] = dict.fromkeys(_COLS, "") | {
        "venue": "ai4s",
        "username": "alice",
        "role": "user",
        "disabled": False,
    }
    configure_user_store_pg(_Pool(conn), store=UserStore(rccs), venue="rccs")
    UserStore(rccs).set_disabled("alice", True)
    assert conn.table[("rccs", "alice")]["disabled"] is True
    assert conn.table[("ai4s", "alice")]["disabled"] is False  # other site untouched


def test_other_venue_row_is_absent_not_borrowed(tmp_path):
    """A user known only to another site falls back to this site's JSON lane."""
    rccs = _site_dir(tmp_path, "rccs")
    conn = _Conn()
    conn.table[("ai4s", "bob")] = dict.fromkeys(_COLS, "") | {
        "venue": "ai4s",
        "username": "bob",
        "role": "admin",
        "disabled": False,
    }
    configure_user_store_pg(_Pool(conn), store=UserStore(rccs), venue="rccs")
    with pytest.raises(UserStoreError):
        UserStore(rccs).get("bob")


def test_reset_clears_the_venue(tmp_path):
    from umrs.admin.user_store import pg_venue

    configure_user_store_pg(_Pool(_Conn()), store=UserStore(tmp_path), venue="rccs")
    assert pg_venue() == "rccs"
    reset_user_store_pg()
    assert pg_venue() == ""


def test_migration_0113_rekeys_users_by_venue():
    from pathlib import Path

    import umrs.migrations as migrations

    sql = (Path(migrations.__file__).parent / "0113_users_venue_scoping.sql").read_text()
    assert "PRIMARY KEY (venue, username)" in sql
    assert "(venue, oidc_subject)" in sql
    assert "DELETE" not in sql.upper()  # non-destructive by contract


# ── admin CLI --from-pg (the D-06 tail verb) ─────────────────────────────────


def _cli_rows(monkeypatch, conn, venue):
    import importlib

    import umrs.admin.user_store as us

    admin_main = importlib.import_module("umrs.cli.admin.main")

    monkeypatch.setattr(us, "site_venue", lambda: venue)
    monkeypatch.setattr(
        "umrs.api._asyncpg_pool._resolve_knowledge_dsn", lambda: ("postgresql://x/y", True)
    )

    class _ClosingConn(_Conn):
        def close(self):
            pass

    closing = _ClosingConn()
    closing.table = conn.table
    monkeypatch.setattr("psycopg2.connect", lambda *_a, **_kw: closing)
    return admin_main


def test_cli_from_pg_lists_only_this_site(tmp_path, monkeypatch, capsys):
    rccs = _site_dir(tmp_path, "rccs", alice={}, carol={})
    UserStore(rccs).set_disabled("carol", True)
    ai4s = _site_dir(tmp_path, "ai4s", bob={})
    conn = _Conn()
    configure_user_store_pg(_Pool(conn), store=UserStore(rccs), venue="rccs")
    configure_user_store_pg(_Pool(conn), store=UserStore(ai4s), venue="ai4s")
    admin_main = _cli_rows(monkeypatch, conn, "rccs")
    import argparse

    args = argparse.Namespace(json=False, role=None, disabled_only=False, from_pg=True)
    assert admin_main._cmd_user_list(args, UserStore(rccs)) == 0
    out = capsys.readouterr().out
    assert "alice" in out and "carol" in out and "(disabled)" in out
    assert "bob" not in out  # another site's account
    assert "2 row(s) from Postgres (0110, venue 'rccs')" in out


def test_cli_from_pg_laneless_refuses_by_name(tmp_path, monkeypatch, capsys):
    import argparse
    import importlib

    admin_main = importlib.import_module("umrs.cli.admin.main")
    monkeypatch.setattr("umrs.api._asyncpg_pool._resolve_knowledge_dsn", lambda: ("", False))
    args = argparse.Namespace(json=False, role=None, disabled_only=False, from_pg=True)
    assert admin_main._cmd_user_list(args, UserStore(tmp_path)) == 2
    assert "--from-pg requires the Postgres lane" in capsys.readouterr().err
