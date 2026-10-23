"""ADR-099 D-7, Amendment 3 (RS-1 round-3 review, finding N3): a project
owner's limit equal to the site default is recorded as the owner's.

The idle checker re-reads ``UMRS_VENUE_IDLE_SECONDS`` for a value whose
origin is ``default``. On a tie the fold kept ``default``, so an owner's
``idle_s`` of 1800 (the default) followed the operator's later 4 h and a
running session idled at 4 h, wider than the owner's rule. Now the tie names
the narrowing layer; a looser owner value and an admin value keep theirs.
"""

from __future__ import annotations

from datetime import timedelta

from test_frontend_session_backend import NOW, _job, _spec, _submit
from test_quota_lifetime import _backend, _check
from umrs.jobs.frontend_session_backend import FrontendSessionStatusChecker
from umrs.session_policy import PolicyRule, effective_limits, site_defaults

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
