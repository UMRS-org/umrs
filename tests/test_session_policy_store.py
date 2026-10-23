"""ADR-099: the rule store (memory and Postgres shapes), the proxy's cached
lookup, and migration 0125.

The Postgres integration test runs with ``UMRS_TEST_DB_URL`` (or
``--db-url``): it applies 0125 inside a rolled-back transaction over a
scratch ``companion_venue_sessions`` and round-trips a rule through the
store's own SQL.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import pytest
from tests.fake_pg_pool import FakePool
from umrs.session_policy import (
    LimitsCache,
    PolicyError,
    PolicyRule,
    PolicyRuleStore,
    effective_limits_cached,
    effective_limits_for,
)

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0125_session_policy.sql"


def _row(venue="rccs", scope="user", subject="alice", layer="admin", doc=None):
    when = datetime(2026, 10, 23, tzinfo=timezone.utc)
    return (venue, scope, subject, layer, json.dumps(doc or {}), "root", "", when)


class TestMemoryStore:
    def test_put_get_list_delete(self):
        store = PolicyRuleStore()
        rule = store.put(PolicyRule("rccs", "user", "alice", "admin", {"budget": {"rpm_user": 6}}))
        assert rule.updated_at is not None
        assert store.get("rccs", "user", "alice", "admin") == rule
        assert store.list(scope="user") == [rule]
        assert store.list(venue="") == []
        assert store.delete("rccs", "user", "alice", "admin") is True
        assert store.delete("rccs", "user", "alice", "admin") is False

    def test_applicable_picks_this_users_projects_and_sites_rules(self):
        store = PolicyRuleStore()
        for rule in (
            PolicyRule("rccs", "site", "", "admin", {}),
            PolicyRule("", "site", "", "admin", {}),
            PolicyRule("bdrt", "site", "", "admin", {}),
            PolicyRule("rccs", "user", "alice", "admin", {}),
            PolicyRule("rccs", "user", "bob", "admin", {}),
            PolicyRule("rccs", "project", "opossum", "owner", {}),
        ):
            store.put(rule)
        got = store.applicable(venue="rccs", user_id="alice", project_id="opossum")
        assert sorted(r.key for r in got) == [
            ("", "site", "", "admin"),
            ("rccs", "project", "opossum", "owner"),
            ("rccs", "site", "", "admin"),
            ("rccs", "user", "alice", "admin"),
        ]

    def test_put_refuses_a_bad_key(self):
        with pytest.raises(PolicyError) as err:
            PolicyRuleStore().put(PolicyRule("rccs", "user", "alice", "owner", {}))
        assert err.value.detail["field"] == "layer"


class TestPostgresShapes:
    def test_applicable_is_one_query_on_the_key(self):
        pool = FakePool(script=[[_row(doc={"budget": {"rpm_user": 6}})]])
        rules = PolicyRuleStore(pool).applicable(
            venue="rccs", user_id="alice", project_id="opossum"
        )
        assert rules[0].doc == {"budget": {"rpm_user": 6}}
        ((sql, params),) = pool.record
        assert "FROM session_policy_rules WHERE venue IN (%s, '')" in sql
        assert params == ("rccs", "opossum", "opossum", "alice", "alice")

    def test_put_upserts_and_returns_the_row(self):
        pool = FakePool(script=[[_row(doc={"web": {"allow": ["none"]}})]])
        rule = PolicyRuleStore(pool).put(
            PolicyRule("rccs", "user", "alice", "admin", {"web": {"allow": ["none"]}}, "root")
        )
        assert rule.doc == {"web": {"allow": ["none"]}}
        ((sql, params),) = pool.record
        assert "ON CONFLICT (venue, scope, subject, layer) DO UPDATE" in sql
        assert params[:4] == ("rccs", "user", "alice", "admin")
        assert json.loads(params[4]) == {"web": {"allow": ["none"]}}

    def test_delete_reports_whether_a_row_went(self):
        pool = FakePool(script=[[(1,)], []])
        store = PolicyRuleStore(pool)
        assert store.delete("rccs", "user", "alice", "admin") is True
        assert store.delete("rccs", "user", "alice", "admin") is False


class TestLimitsCache:
    def test_cached_for_the_ttl_then_read_again(self):
        store = PolicyRuleStore()
        clock = [100.0]
        cache = LimitsCache(store, ttl_s=30, clock=lambda: clock[0], env={})
        assert (
            cache.get(venue="rccs", user_id="alice", project_id="p")["budget"]["rpm_user"] is None
        )
        store.put(PolicyRule("rccs", "user", "alice", "admin", {"budget": {"rpm_user": 6}}))
        clock[0] = 129.0
        assert (
            cache.get(venue="rccs", user_id="alice", project_id="p")["budget"]["rpm_user"] is None
        )
        clock[0] = 131.0
        assert cache.get(venue="rccs", user_id="alice", project_id="p")["budget"]["rpm_user"] == 6

    def test_bounded(self):
        cache = LimitsCache(PolicyRuleStore(), max_keys=2, env={})
        for user in ("a", "b", "c"):
            cache.get(venue="rccs", user_id=user, project_id="")
        assert len(cache._entries) == 2
        assert ("rccs", "a", "") not in cache._entries

    def test_the_proxy_provider_reads_the_pool(self):
        pool = FakePool(script=[[_row(doc={"budget": {"tokens_per_day_user": 1000}})]])
        limits = effective_limits_cached(pool)("rccs", "alice", "opossum")
        assert limits["budget"]["tokens_per_day_user"] == 1000
        assert limits["origin"]["budget.tokens_per_day_user"] == "user:admin"
        assert len(pool.record) == 1

    def test_store_errors_propagate_for_the_caller_to_fail_closed(self):
        class Broken(PolicyRuleStore):
            def applicable(self, **_kw):
                raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            effective_limits_for(Broken(), venue="rccs", user_id="a", project_id="")

    def test_a_failed_reread_serves_the_last_good_limits(self, caplog):
        """ADR-099 D-5 (Q4): a blip after the TTL keeps the proxy on the last
        limits read (old code: the error propagated and every proxied call
        answered 503 quota.unavailable)."""
        store = _Flaky()
        clock = [100.0]
        cache = LimitsCache(store, ttl_s=30, clock=lambda: clock[0], env={})
        store.put(PolicyRule("rccs", "user", "alice", "admin", {"budget": {"rpm_user": 6}}))
        assert cache.get(venue="rccs", user_id="alice", project_id="p")["budget"]["rpm_user"] == 6
        store.down = True
        clock[0] = 200.0
        limits = cache.get(venue="rccs", user_id="alice", project_id="p")
        assert limits["budget"]["rpm_user"] == 6
        assert cache.stats == {"stale_served": 1, "unavailable": 0}
        assert "serving last good limits" in caplog.text
        store.down = False
        store.put(PolicyRule("rccs", "user", "alice", "admin", {"budget": {"rpm_user": 9}}))
        assert cache.get(venue="rccs", user_id="alice", project_id="p")["budget"]["rpm_user"] == 9

    def test_a_key_never_read_still_fails_closed(self):
        store = _Flaky()
        store.down = True
        cache = LimitsCache(store, env={})
        with pytest.raises(ConnectionError):
            cache.get(venue="rccs", user_id="bob", project_id="")
        assert cache.stats == {"stale_served": 0, "unavailable": 1}


class _Flaky(PolicyRuleStore):
    down = False

    def applicable(self, **kw):
        if self.down:
            raise ConnectionError("down")
        return super().applicable(**kw)


# ── migration 0125 ────────────────────────────────────────────────────────


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def test_0125_follows_0124_and_is_alone_with_its_number():
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0125") == 1
    assert versions.index("0125") == versions.index("0124") + 1


def test_0125_keys_and_constraints_match_the_store():
    code = _code()
    assert "PRIMARY KEY (venue, scope, subject, layer)" in code
    assert "CHECK (layer = 'admin' OR scope = 'project')" in code
    assert "CHECK (scope <> 'site' OR subject = '')" in code
    assert re.search(
        r"ALTER TABLE companion_venue_sessions\s+ADD COLUMN IF NOT EXISTS policy JSONB "
        r"NOT NULL DEFAULT '\{\}'::jsonb;",
        code,
    )
    assert "GRANT SELECT ON session_policy_rules TO umrs_infer" in code
    assert "DROP " not in code.upper()


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url", default=None) or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("no UMRS_TEST_DB_URL")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


class _OneConnPool:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def getconn(self) -> Any:
        return self.conn

    def putconn(self, _conn: Any) -> None:
        return None


class _NoCommit:
    """The test's transaction stays open (rolled back at the end)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self, *a: Any, **k: Any) -> Any:
        return self._conn.cursor(*a, **k)


@pytest.mark.integration
def test_0125_applies_and_the_store_round_trips(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS companion_venue_sessions (session_id TEXT PRIMARY KEY)"
        )
        cur.execute(MIGRATION.read_text())
        cur.execute(MIGRATION.read_text())  # idempotent
    store = PolicyRuleStore(_OneConnPool(_NoCommit(pg_conn)))
    store.put(PolicyRule("rccs", "user", "alice", "admin", {"budget": {"rpm_user": 6}}, "root"))
    store.put(PolicyRule("", "site", "", "admin", {"sessions": {"concurrent_user": 4}}))
    assert store.get("rccs", "user", "alice", "admin").doc == {"budget": {"rpm_user": 6}}
    assert len(store.applicable(venue="rccs", user_id="alice", project_id="")) == 2
    assert [r.scope for r in store.list(venue="")] == ["site"]
    assert store.delete("", "site", "", "admin") is True
    with pg_conn.cursor() as cur, pytest.raises(Exception, match="owner_is_project"):
        cur.execute(
            "INSERT INTO session_policy_rules (venue, scope, subject, layer) "
            "VALUES ('', 'user', 'alice', 'owner')"
        )


@pytest.mark.integration
def test_the_session_row_keeps_its_policy_through_a_job_restart(pg_conn):
    from umrs.venues.venue_sessions import VenueSessionStore

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_name = "
            "'companion_venue_sessions' AND column_name = 'policy'"
        )
        if cur.fetchone() is None:
            pytest.skip("database not migrated to 0125 (python -m umrs.migrations)")
    store = VenueSessionStore(_OneConnPool(_NoCommit(pg_conn)))
    reserve = {
        "session_id": "rs1-policy-test",
        "user_id": "alice",
        "subnet": "172.30.250.0/24",
        "source_names": ("openai-main",),
        "source_ids": ("11111111-1111-4111-8111-111111111111",),
        "default_source_id": "11111111-1111-4111-8111-111111111111",
        "default_source_name": "openai-main",
        "web_policy": "none",
    }
    policy = {"web": {"mode": "none", "domains": []}, "origin": {"web.mode": "choice"}}
    assert store.upsert_provisioning(**reserve, policy=policy).policy == policy
    assert store.upsert_provisioning(**reserve, job_id="job-2").policy == policy
    assert store.get("rs1-policy-test").to_dict()["policy"] == policy
