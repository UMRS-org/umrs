"""The account cascade's HPC site-account leg (ADR-112 wave B1).

A disabled account loses its transport keys at this venue (the files: its
gate can no longer be reached), its site account is revoked and its project
links are suspended; a project removal suspends only that project's link.
The leg is best-effort like the jobs leg and lands in the one audit event.
"""

from __future__ import annotations

import json
import shutil

import pytest
from umrs.admin import account_cascade as ac
from umrs.hpc_accounts import transport_keys as tk
from umrs.hpc_accounts.store import InMemorySiteAccountStore

pytestmark = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="needs ssh-keygen")

SITE = "site-b1"


class _Broker:
    venue = SITE

    def revoke_by_owner(self, *_a, **_kw):
        return []


class _AuditConn:
    def __init__(self) -> None:
        self.rows: list = []

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

            def execute(self, sql, params):
                if "INSERT INTO audit_events" in sql:
                    conn.rows.append(params)

        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


class _Pool:
    def __init__(self) -> None:
        self.conn = _AuditConn()

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        pass


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv(tk.KEY_DIR_ENV, str(tmp_path))
    shared = InMemorySiteAccountStore(SITE).tables
    here = InMemorySiteAccountStore(SITE, shared)
    elsewhere = InMemorySiteAccountStore("other-site", shared)
    keys = {}
    for store in (here, elsewhere):
        for user in ("alice", "bob"):
            key = tk.generate(store.venue, user)
            keys[(store.venue, user)] = key
            store.issue_key(user, key.key_id, key.fingerprint)
            store.upsert_link(user, "p1", "grp")
            store.upsert_link(user, "p2", "grp")
    return here, elsewhere, keys


def _cascade(store, project=None, pool=None):
    return ac.run_cascade(
        "alice",
        project=project,
        reason=ac.ACCOUNT_DISABLED,
        actor="admin-cli:root",
        trigger="admin-cli",
        broker=_Broker(),
        audit_pool=pool,
        site_accounts=store,
    )


def test_a_disabled_account_loses_its_keys_account_and_links_here(world):
    here, _elsewhere, keys = world
    pool = _Pool()
    result = _cascade(here, pool=pool)
    assert (result.transport_keys_destroyed, result.hpc_links_suspended) == (1, 2)
    assert not keys[(SITE, "alice")].path.exists()
    assert here.get_account("alice")["status"] == "revoked"
    assert {x["status"] for x in here.list_links("alice")} == {"suspended"}
    details = json.loads(pool.conn.rows[0][9])
    counts = (details["transport_keys_destroyed"], details["hpc_links_suspended"])
    assert counts == (1, 2)


def test_other_people_and_other_venues_keep_theirs(world):
    here, elsewhere, keys = world
    _cascade(here)
    kept = [keys[(SITE, "bob")].path, keys[("other-site", "alice")].path]
    assert [p.exists() for p in kept] == [True, True]
    assert elsewhere.get_account("alice")["status"] == "pending"
    assert {x["status"] for x in elsewhere.list_links("alice")} == {"active"}


def test_a_project_removal_suspends_only_that_link_and_keeps_the_key(world):
    here, _elsewhere, keys = world
    result = _cascade(here, project="p1")
    assert (result.transport_keys_destroyed, result.hpc_links_suspended) == (0, 1)
    assert keys[(SITE, "alice")].path.exists()
    assert here.get_account("alice")["status"] == "pending"
    assert {x["project_id"]: x["status"] for x in here.list_links("alice")} == {
        "p1": "suspended",
        "p2": "active",
    }


def test_a_failing_store_never_stops_the_cascade(world):
    class _Broken(InMemorySiteAccountStore):
        def suspend_links(self, *_a, **_kw):
            raise RuntimeError("table gone")

    result = _cascade(_Broken(SITE), pool=_Pool())
    assert (result.hpc_links_suspended, result.audited) == (0, True)


def test_without_a_database_or_venue_the_leg_is_skipped():
    assert ac.site_account_store(None, SITE) is None
    assert ac.site_account_store(_Pool(), "") is None
    result = _cascade(None)
    assert (result.transport_keys_destroyed, result.hpc_links_suspended) == (0, 0)


def test_the_default_store_is_this_venues_postgres_store():
    store = ac.site_account_store(_Pool(), SITE)
    assert type(store).__name__ == "SiteAccountStore" and store.venue == SITE


def test_the_leg_against_postgres(tmp_path, monkeypatch):
    """Skipped without a database: the cascade's own pool reaches the rows."""
    from test_hpc_accounts_store import _pg_cleanup, _pg_pool_or_skip
    from umrs.api.audit_store import ensure_audit_schema
    from umrs.hpc_accounts.store import SiteAccountStore

    pool = _pg_pool_or_skip()
    conn = pool.getconn()
    try:
        ensure_audit_schema(conn)
    finally:
        pool.putconn(conn)
    monkeypatch.setenv(tk.KEY_DIR_ENV, str(tmp_path))
    store = SiteAccountStore(pool, SITE)
    try:
        key = tk.generate(SITE, "alice")
        store.issue_key("alice", key.key_id, key.fingerprint)
        store.upsert_link("alice", "p1", "grp")
        result = _cascade(None, pool=pool)
        assert (result.transport_keys_destroyed, result.hpc_links_suspended) == (1, 1)
        assert store.get_account("alice")["status"] == "revoked"
        assert not key.path.exists() and result.audited
    finally:
        _pg_cleanup(pool, [SITE])


# ── the standing sweep reaches site-account holders (B1 review, 2026-11-08) ─


class _NoLeases(_Broker):
    def active_owner_projects(self):
        return set()


class _Record:
    def __init__(self, username: str, disabled: bool) -> None:
        self.username = username
        self.disabled = disabled
        self.role = "user"
        self.projects: list = []
        self.oidc_subject = ""


def _sweep(store, disabled=("alice",)):
    return ac.AccountStandingSweep(
        pool=None,
        broker=_NoLeases(),
        record_lookup=lambda user: _Record(user, user in disabled),
        curve_provider=lambda: None,
        site_accounts=store,
    )


def test_the_sweep_revokes_a_disabled_site_account_holder_without_a_lease(world):
    here, _elsewhere, keys = world
    results = _sweep(here).run_once()
    assert [(r.user, r.project, r.transport_keys_destroyed) for r in results] == [
        ("alice", "", 1)
    ]
    assert here.get_account("alice")["status"] == "revoked"
    assert not keys[(SITE, "alice")].path.exists()
    assert {x["status"] for x in here.list_links("alice")} == {"suspended"}


def test_the_sweep_leaves_others_alone_and_converges(world):
    here, elsewhere, _keys = world
    _sweep(here).run_once()
    assert here.get_account("bob")["status"] == "pending"
    assert elsewhere.get_account("alice")["status"] == "pending"
    assert _sweep(here).run_once() == []  # nothing left to end


def test_a_failed_site_account_leg_is_retried_on_the_next_tick(world):
    here, _elsewhere, keys = world

    class _Flaky(InMemorySiteAccountStore):
        fail = True

        def revoke_account(self, username):
            if _Flaky.fail:
                raise RuntimeError("database blip")
            return super().revoke_account(username)

    flaky = _Flaky(SITE, here.tables)
    _sweep(flaky).run_once()
    assert here.get_account("alice")["status"] != "revoked"
    _Flaky.fail = False
    assert [r.user for r in _sweep(flaky).run_once()] == ["alice"]
    assert here.get_account("alice")["status"] == "revoked"


def test_an_ex_member_loses_that_projects_link_under_enforce(world, monkeypatch):
    here, _elsewhere, keys = world
    monkeypatch.setenv("UMRS_PROJECT_MEMBERSHIP", "enforce")
    monkeypatch.setattr(ac, "_member_of", lambda _pool, _record, _user: {"p2"})
    results = _sweep(here, disabled=()).run_once()
    assert sorted((r.user, r.project) for r in results) == [("alice", "p1"), ("bob", "p1")]
    assert {x["project_id"]: x["status"] for x in here.list_links("alice")} == {
        "p1": "suspended",
        "p2": "active",
    }
    assert keys[(SITE, "alice")].path.exists()


def test_site_account_holders_skips_revoked_accounts_and_suspended_links(world):
    here, _elsewhere, _keys = world
    here.revoke_account("bob")
    here.suspend_links("alice", "x", project_id="p2")
    assert ac.site_account_holders(here) == {"alice": {"p1"}, "bob": {"p1", "p2"}}
