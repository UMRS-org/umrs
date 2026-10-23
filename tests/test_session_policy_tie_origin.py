"""ADR-099 D-7, Amendment 3 (RS-1 round-3 review, finding N3): a project
owner's limit equal to the site default is recorded as the owner's.

The idle checker re-reads ``UMRS_VENUE_IDLE_SECONDS`` for a value whose
origin is ``default``. On a tie the fold kept ``default``, so an owner's
``idle_s`` of 1800 (the default) followed the operator's later 4 h and a
running session idled at 4 h, wider than the owner's rule. Now the tie names
the narrowing layer; a looser owner value and an admin value keep theirs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from test_frontend_session_backend import NOW, _job, _spec, _submit
from test_quota_lifetime import _backend, _check
from umrs.jobs.frontend_session_backend import FrontendSessionStatusChecker
from umrs.session_policy import (
    PolicyRule,
    WebChoice,
    effective_limits,
    resolve_session_policy,
    site_defaults,
)

DEFAULTS = site_defaults({})


def _owner(doc):
    return PolicyRule(venue="rccs", scope="project", subject="opossum", layer="owner", doc=doc)


def _admin(doc):
    return PolicyRule(venue="rccs", scope="site", subject="", layer="admin", doc=doc)


def _limits(*rules, defaults=DEFAULTS):
    return effective_limits(
        list(rules), venue="rccs", user_id="alice", project_id="opossum", defaults=defaults
    )


class TestTheTieNamesTheOwner:
    def test_an_owner_value_equal_to_the_default_is_the_owners(self):
        limits = _limits(_owner({"lifetime": {"idle_s": DEFAULTS["lifetime"]["idle_s"]}}))
        assert limits["lifetime"]["idle_s"] == 1800
        assert limits["origin"]["lifetime.idle_s"] == "project:owner"

    def test_an_owner_value_looser_than_a_lowered_default_stays_default(self):
        defaults = site_defaults({"UMRS_VENUE_IDLE_SECONDS": "600"})
        limits = _limits(_owner({"lifetime": {"idle_s": 1800}}), defaults=defaults)
        assert limits["lifetime"]["idle_s"] == 600
        assert limits["origin"]["lifetime.idle_s"] == "default"
        # ...bounded by the owner's rule (Amendment 4, round-4 R4-2)
        assert limits["bounds"] == {"lifetime.idle_s": 1800}

    def test_an_admin_tie_keeps_the_admin(self):
        limits = _limits(
            _admin({"lifetime": {"idle_s": 900}}), _owner({"lifetime": {"idle_s": 900}})
        )
        assert limits["origin"]["lifetime.idle_s"] == "site:admin"


class TestARaisedSiteIdleDoesNotWidenTheOwnersRule:
    def test_the_running_session_keeps_the_owners_1800s(self):
        limits = _limits(_owner({"lifetime": {"idle_s": 1800}}))
        policy = {
            "schema": 1,
            "lifetime": dict(limits["lifetime"]),
            "origin": dict(limits["origin"]),
        }
        backend, prov, broker = _backend(idle=4 * 3600)  # the operator raised it
        job = _job(spec=_spec(policy=policy))
        ref = _submit(backend, job)
        broker.add("lease-1", last=NOW - timedelta(seconds=2 * 3600))
        res = _check(FrontendSessionStatusChecker(backend), ref, job)
        assert res.status == "succeeded" and res.metrics["end"] == "idle"
        assert prov.torn == ["sess-1"]


# ── Amendment 4 (RS-1 round-4 review, R4-2) ─────────────────────────────────
# An owner's idle_s ABOVE the site default lost to it (origin ``default``),
# so the checker read the live env alone: the operator raising
# UMRS_VENUE_IDLE_SECONDS to 4 h let a running session idle past the owner's
# 3600 s. ``bounds`` keeps the owner's value; the checker takes the smaller.


def _running_policy(owner_idle_s: int) -> dict:
    limits = _limits(_owner({"lifetime": {"idle_s": owner_idle_s}}))
    assert limits["origin"]["lifetime.idle_s"] == "default"
    return {
        "schema": 1,
        "lifetime": dict(limits["lifetime"]),
        "origin": dict(limits["origin"]),
        "bounds": dict(limits.get("bounds") or {}),
    }


def _idle_check(policy: dict, *, site_idle_s: int, idle_for_s: int):
    backend, prov, broker = _backend(idle=site_idle_s)
    job = _job(spec=_spec(policy=policy))
    ref = _submit(backend, job)
    broker.add("lease-1", last=NOW - timedelta(seconds=idle_for_s))
    return _check(FrontendSessionStatusChecker(backend), ref, job), prov


class TestARaisedSiteIdleStopsAtALooserOwnerRule:
    def test_the_owners_3600s_ends_a_session_the_raised_default_would_keep(self):
        res, prov = _idle_check(_running_policy(3600), site_idle_s=4 * 3600, idle_for_s=2 * 3600)
        assert res.status == "succeeded" and res.metrics["end"] == "idle"
        assert prov.torn == ["sess-1"]

    def test_below_the_owners_rule_the_raised_default_applies(self):
        res, prov = _idle_check(_running_policy(3600), site_idle_s=4 * 3600, idle_for_s=2400)
        assert res.status != "succeeded" and prov.torn == []

    def test_a_lowered_default_still_reaches_the_running_session(self):
        res, prov = _idle_check(_running_policy(3600), site_idle_s=600, idle_for_s=900)
        assert res.status == "succeeded" and prov.torn == ["sess-1"]

    def test_the_resolved_policy_carries_the_bounds(self):
        limits = _limits(_owner({"lifetime": {"idle_s": 3600}}))
        policy = resolve_session_policy(
            limits,
            WebChoice(mode="none", domains=()),
            require_choice=True,
            legacy_default="none",
            now=datetime(2026, 10, 23, tzinfo=timezone.utc),
        )
        assert policy["bounds"] == {"lifetime.idle_s": 3600}
