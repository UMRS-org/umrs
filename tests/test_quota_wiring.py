"""ADR-099 (RS-1 fix Q3): the production wiring of lanes P and Q, end to end.

The lane tests check each side against a stand-in; these build the real
pieces together, so a broken join fails here:

- ``companion_venue._quota_check(pool)`` inside a real
  ``FrontendVenueService``: a start resolves P's policy and the 13th live
  session is refused by Q's cap through the real ``StartCheck``;
- ``infer_proxy.__main__.build_quota_enforcer(pool)`` over P's real
  ``effective_limits_cached``: a user rule read from the pool caps the user.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from tests.fake_pg_pool import FakePool
from tests.test_infer_proxy import _lease
from tests.test_quota_session_cap import _JobsPool, _sessions
from tests.test_session_policy_start import _harness, _start
from umrs.api.companion_venue import _quota_check
from umrs.infer_proxy.__main__ import build_quota_enforcer

CODE = "x-platform-error-code"


class TestStartWiring:
    def test_the_real_hook_refuses_the_13th_live_session(self, monkeypatch):
        h = _harness(monkeypatch, quota_check=_quota_check(_JobsPool(_sessions(12))))
        r = _start(h, web_policy="none")
        assert r.status_code == 429, r.text
        assert r.headers[CODE] == "quota.concurrent_sessions_exceeded"
        assert "12 of 12 live sessions" in r.json()["detail"]
        assert h.jobs.created == []

    def test_the_real_hook_admits_a_start_under_the_cap(self, monkeypatch):
        h = _harness(monkeypatch, quota_check=_quota_check(_JobsPool(_sessions(11))))
        r = _start(h, web_policy="none")
        assert r.status_code == 200, r.text
        assert len(h.jobs.created) == 1

    def test_a_site_rule_reaches_the_hook_through_the_resolved_policy(self, monkeypatch):
        from umrs.session_policy import PolicyRule, PolicyRuleStore

        store = PolicyRuleStore()
        store.put(PolicyRule("rccs", "site", "", "admin", {"sessions": {"concurrent_user": 2}}))
        hook = _quota_check(_JobsPool(_sessions(2)))
        h = _harness(monkeypatch, store=store, quota_check=hook)
        r = _start(h, web_policy="none")
        assert r.status_code == 429 and "2 of 2" in r.json()["detail"]


def _rule_row(doc):
    when = datetime(2026, 10, 23, tzinfo=timezone.utc)
    return ("rccs", "user", "alice", "admin", json.dumps(doc), "root", "", when)


class TestProxyWiring:
    def test_the_built_enforcer_reads_the_rules_and_the_ledger(self, monkeypatch):
        monkeypatch.setenv("UMRS_QUOTA_DAY_TZ", "UTC")
        pool = FakePool(script=[[_rule_row({"budget": {"tokens_per_day_user": 1000}})], [(1500,)]])
        pool.conn.commit = lambda: None  # psycopg2 parity: the ledger read commits
        enforcer = build_quota_enforcer(pool)
        denial = enforcer.check(_lease(owner_id="alice", project="opossum", venue="rccs"))
        assert denial is not None
        assert (denial.code, denial.status) == ("quota.tokens_per_day_exceeded", 429)
        assert (denial.limit, denial.used, denial.scope) == (1000, 1500, "user")
        rules_sql, sum_sql = (sql for sql, _params in pool.record)
        assert "FROM session_policy_rules" in rules_sql
        assert "FROM llm_usage_log" in sum_sql

    def test_no_rule_means_no_ledger_query(self):
        pool = FakePool(script=[[]])
        enforcer = build_quota_enforcer(pool)
        assert enforcer.check(_lease(owner_id="alice", project="p", venue="rccs")) is None
        assert len(pool.record) == 1  # the rules read; no SUM
