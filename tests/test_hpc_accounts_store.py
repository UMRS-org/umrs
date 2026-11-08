"""``umrs.hpc_accounts.store`` (ADR-112 wave B1; migrations 0134/0135).

One contract, two implementations: the psycopg2 store over a migrated
Postgres (skipped without one) and the in-memory twin the route tests use.
Both are venue-scoped (invariant I-11): stores of two fronts over one
database never see each other's rows.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

import pytest
from umrs.hpc_accounts.store import (
    ACCOUNT_VIEW,
    LINK_COLUMNS,
    InMemorySiteAccountStore,
    SiteAccountStore,
)

from umrs import migrations

_VERIFIED = {
    "site_login": "alice1",
    "verified_login": "alice1",
    "verified_assocs": [{"cluster": "c1", "account": "grp-a", "partition": "p", "qos": "n"}],
    "gate_version": "1",
    "gate_sha256": "a" * 64,
    "signer_fingerprints": ["SHA256:abc"],
    "uv_required": True,
}


def _pg_pool_or_skip() -> Any:
    pytest.importorskip("psycopg2")
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn
    from umrs.cli.admin.cascade import _ConnPerCallPool

    dsn, _ = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("no Postgres configured")
    pool = _ConnPerCallPool(dsn)
    try:
        conn = pool.getconn()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")
    try:
        with conn.cursor() as cur:
            for name in ("0134_site_accounts.sql", "0135_project_site_allocations.sql"):
                cur.execute((migrations.MIGRATIONS_DIR / name).read_text())
        conn.commit()
    finally:
        pool.putconn(conn)
    return pool


def _pg_cleanup(pool: Any, venues: list[str]) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM project_site_allocations WHERE venue = ANY(%s)", (venues,))
            cur.execute("DELETE FROM site_accounts WHERE venue = ANY(%s)", (venues,))
        conn.commit()
    finally:
        pool.putconn(conn)


@pytest.fixture(params=["memory", "postgres"])
def stores(request: pytest.FixtureRequest) -> Iterator[tuple[Any, Any]]:
    """Two fronts' stores over one database: (front A, front B)."""
    tag = uuid.uuid4().hex[:8]
    venue_a, venue_b = f"a-{tag}", f"b-{tag}"
    if request.param == "memory":
        shared = InMemorySiteAccountStore(venue_a).tables
        yield InMemorySiteAccountStore(venue_a, shared), InMemorySiteAccountStore(venue_b, shared)
        return
    pool = _pg_pool_or_skip()
    try:
        yield SiteAccountStore(pool, venue_a), SiteAccountStore(pool, venue_b)
    finally:
        _pg_cleanup(pool, [venue_a, venue_b])


def test_a_new_key_makes_a_pending_account_with_the_contract_view(stores) -> None:
    a, _ = stores
    row = a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    assert tuple(row) == ACCOUNT_VIEW
    assert (row["venue"], row["status"], row["method"], row["gate_probe"]) == (
        a.venue,
        "pending",
        "gated_key",
        "unknown",
    )
    assert (row["transport_key_id"], row["transport_fingerprint"]) == (
        "tk-000000000001",
        "SHA256:fp1",
    )
    assert a.get_account("alice") == row


def test_verify_activates_only_while_the_key_is_current(stores) -> None:
    a, _ = stores
    a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    assert a.record_verified("alice", "tk-00000000000f", _VERIFIED) is None
    row = a.record_verified("alice", "tk-000000000001", _VERIFIED)
    assert row is not None and row["status"] == "active" and row["gate_probe"] == "refused"
    assert row["verified_assocs"] == _VERIFIED["verified_assocs"]
    assert row["signer_fingerprints"] == ["SHA256:abc"] and row["uv_required"] is True
    assert row["verified_at"]


def test_a_rotation_returns_the_account_to_pending(stores) -> None:
    a, _ = stores
    a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    a.record_verified("alice", "tk-000000000001", _VERIFIED)
    row = a.issue_key("alice", "tk-000000000002", "SHA256:fp2")
    assert (row["status"], row["transport_key_id"]) == ("pending", "tk-000000000002")
    assert row["verified_login"] == "alice1"  # the last report stays on record


def test_probe_and_revoke(stores) -> None:
    a, _ = stores
    assert a.revoke_account("alice") is False
    a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    a.record_probe("alice", "bypassed")
    assert a.get_account("alice")["gate_probe"] == "bypassed"
    assert a.revoke_account("alice") is True
    assert a.get_account("alice")["status"] == "revoked"


@pytest.mark.parametrize("probe", ["bypassed", "unknown"])
def test_a_probe_the_gate_did_not_refuse_suspends_an_active_account(stores, probe) -> None:
    a, _ = stores
    a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    a.record_probe("alice", probe)
    assert a.get_account("alice")["status"] == "pending"  # never active: stays pending
    a.record_verified("alice", "tk-000000000001", _VERIFIED)
    a.record_probe("alice", "refused")
    assert a.get_account("alice")["status"] == "active"
    a.record_probe("alice", probe)
    assert (a.get_account("alice")["status"], a.get_account("alice")["gate_probe"]) == (
        "suspended",
        probe,
    )
    assert a.record_verified("alice", "tk-000000000001", _VERIFIED)["status"] == "active"


def test_suspend_account_touches_only_an_active_account(stores) -> None:
    a, _ = stores
    assert a.suspend_account("alice") is False
    a.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    assert a.suspend_account("alice") is False
    a.record_verified("alice", "tk-000000000001", _VERIFIED)
    assert a.suspend_account("alice") is True
    assert a.get_account("alice")["status"] == "suspended"


def _two_links(store) -> dict:
    store.issue_key("alice", "tk-000000000001", "SHA256:fp1")
    first = store.upsert_link("alice", "p1", "grp-a")
    store.upsert_link("alice", "p2", "grp-a")
    return first


def _statuses(store) -> dict:
    return {x["project_id"]: x["status"] for x in store.list_links("alice")}


def test_a_link_has_the_contract_view(stores) -> None:
    a, _ = stores
    link = _two_links(a)
    assert tuple(link) == LINK_COLUMNS
    got = (link["project_id"], link["slurm_account"], link["status"])
    assert got == ("p1", "grp-a", "active")
    assert [x["project_id"] for x in a.list_links("alice")] == ["p1", "p2"]


def test_suspending_one_project_is_idempotent_and_a_relink_resumes(stores) -> None:
    a, _ = stores
    _two_links(a)
    counts = [a.suspend_links("alice", "membership", project_id="p1") for _ in range(2)]
    assert (counts, _statuses(a)) == ([1, 0], {"p1": "suspended", "p2": "active"})
    assert a.upsert_link("alice", "p1", "grp-a")["status"] == "active"


def test_suspending_the_account_and_deleting(stores) -> None:
    a, _ = stores
    _two_links(a)
    assert a.suspend_links("alice", "account_revoked") == 2
    assert [a.delete_link("alice", "p1"), a.delete_link("alice", "p1")] == [True, False]


def test_another_front_sees_none_of_this_fronts_rows(stores) -> None:
    a, b = stores
    _two_links(a)
    seen = (b.get_account("alice"), b.list_accounts(), b.list_links())
    assert seen == (None, [], [])
    changed = (
        b.revoke_account("alice"),
        b.suspend_links("alice", "x"),
        b.delete_link("alice", "p1"),
    )
    assert changed == (False, 0, False)


def test_another_fronts_key_leaves_this_fronts_account(stores) -> None:
    a, b = stores
    _two_links(a)
    b.issue_key("alice", "tk-000000000009", "SHA256:fp9")
    assert a.get_account("alice")["transport_key_id"] == "tk-000000000001"
    assert [r["username"] for r in a.list_accounts()] == ["alice"]


def test_list_links_without_a_user_lists_the_venue(stores) -> None:
    a, _ = stores
    for user in ("bob", "alice"):
        a.issue_key(user, "tk-000000000001", "SHA256:fp")
        a.upsert_link(user, "p1", "grp")
    assert [(x["username"], x["project_id"]) for x in a.list_links()] == [
        ("alice", "p1"),
        ("bob", "p1"),
    ]


def test_the_memory_twin_refuses_a_link_without_an_account() -> None:
    with pytest.raises(ValueError, match="site account"):
        InMemorySiteAccountStore("v").upsert_link("nobody", "p1", "grp")


def test_a_store_needs_its_venue() -> None:
    with pytest.raises(ValueError, match="venue"):
        SiteAccountStore(object(), "")


def test_the_postgres_store_names_the_venue_in_every_statement() -> None:
    """Without a database: every statement the store sends carries the venue."""

    class _Cur:
        description = None

        def __init__(self, seen: list) -> None:
            self.seen = seen

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def execute(self, sql, params):
            self.seen.append((sql, params))

    class _Conn:
        def __init__(self) -> None:
            self.seen: list = []

        def cursor(self):
            return _Cur(self.seen)

        def commit(self):
            pass

        def rollback(self):
            pass

    class _Pool:
        def __init__(self) -> None:
            self.conn = _Conn()

        def getconn(self):
            return self.conn

        def putconn(self, _conn):
            pass

    pool = _Pool()
    store = SiteAccountStore(pool, "front-x")
    store.get_account("u")
    store.list_accounts()
    store.record_probe("u", "unknown")
    store.revoke_account("u")
    store.list_links("u")
    store.delete_link("u", "p")
    store.suspend_links("u", "r")
    store.record_verified("u", "tk-000000000001", _VERIFIED)
    store.suspend_account("u")
    assert len(pool.conn.seen) == 9
    for sql, params in pool.conn.seen:
        assert "venue = %s" in sql and "front-x" in params, sql
