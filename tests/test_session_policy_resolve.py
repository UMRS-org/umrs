"""ADR-099 D-1..D-3: rule validation, the fold to effective limits, the choice.

The resolution table of the contract (§6, lane P): default, site override,
a user raised above the site, project and user both set (strictest), owner
narrows, owner widening refused, a domain cap removes ``public``, a choice
wider than the caps refused.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from umrs.session_policy import (
    PolicyError,
    PolicyRule,
    WebChoice,
    check_owner_rule,
    domain_within,
    effective_limits,
    resolve_session_policy,
    site_defaults,
    validate_rule_doc,
    web_choice_required,
)

NOW = datetime(2026, 10, 23, 9, 0, tzinfo=timezone.utc)
DEFAULTS = site_defaults({})


def _rule(scope, subject, doc, *, layer="admin", venue="rccs"):
    return PolicyRule(venue=venue, scope=scope, subject=subject, layer=layer, doc=doc)


def _limits(*rules, user="alice", project="opossum", venue="rccs"):
    return effective_limits(
        list(rules), venue=venue, user_id=user, project_id=project, defaults=DEFAULTS
    )


def _choose(limits, mode, *domains, required=True, legacy="public"):
    return resolve_session_policy(
        limits,
        WebChoice(mode, tuple(domains)),
        require_choice=required,
        legacy_default=legacy,
        now=NOW,
    )


class TestSiteDefaults:
    def test_defaults_are_todays_values(self):
        assert DEFAULTS["lifetime"] == {"idle_s": 1800, "wall_clock_s": 604800}
        assert DEFAULTS["sessions"] == {"concurrent_user": 12, "concurrent_project": None}
        assert set(DEFAULTS["budget"].values()) == {None}

    def test_env_sets_the_defaults(self):
        env = {
            "UMRS_VENUE_IDLE_SECONDS": "900",
            "UMRS_VENUE_MAX_LIFETIME_SECONDS": "86400",
            "UMRS_SESSION_CONCURRENT_PER_USER": "3",
        }
        got = site_defaults(env)
        assert got["lifetime"] == {"idle_s": 900, "wall_clock_s": 86400}
        assert got["sessions"]["concurrent_user"] == 3

    def test_choice_required_defaults_true(self):
        assert web_choice_required({}) is True
        assert web_choice_required({"UMRS_SESSION_WEB_CHOICE_REQUIRED": "false"}) is False


class TestResolutionTable:
    def test_default(self):
        limits = _limits()
        assert limits["web"]["allow"] == ["none", "allowlist", "public"]
        assert limits["web"]["domains"] == []
        assert limits["sessions"]["concurrent_user"] == 12
        assert limits["budget"]["tokens_per_day_user"] is None
        assert set(limits["origin"].values()) == {"default"}

    def test_site_override(self):
        limits = _limits(_rule("site", "", {"budget": {"tokens_per_day_user": 1000}}))
        assert limits["budget"]["tokens_per_day_user"] == 1000
        assert limits["origin"]["budget.tokens_per_day_user"] == "site:admin"

    def test_user_raised_above_the_site(self):
        limits = _limits(
            _rule("site", "", {"sessions": {"concurrent_user": 4}}),
            _rule("user", "alice", {"sessions": {"concurrent_user": 20}}),
        )
        assert limits["sessions"]["concurrent_user"] == 20
        assert limits["origin"]["sessions.concurrent_user"] == "user:admin"

    def test_user_rule_null_lifts_a_site_cap(self):
        limits = _limits(
            _rule("site", "", {"budget": {"tokens_per_day_user": 1000}}),
            _rule("user", "alice", {"budget": {"tokens_per_day_user": None}}),
        )
        assert limits["budget"]["tokens_per_day_user"] is None

    def test_project_and_user_both_set_the_strictest_wins(self):
        limits = _limits(
            _rule("user", "alice", {"budget": {"rpm_user": 120}}),
            _rule("project", "opossum", {"budget": {"rpm_user": 30}}),
        )
        assert limits["budget"]["rpm_user"] == 30
        assert limits["origin"]["budget.rpm_user"] == "project:admin"

    def test_other_users_and_projects_do_not_bind(self):
        limits = _limits(
            _rule("user", "bob", {"budget": {"rpm_user": 1}}),
            _rule("project", "other", {"budget": {"rpm_user": 2}}),
        )
        assert limits["budget"]["rpm_user"] is None

    def test_owner_narrows(self):
        limits = _limits(
            _rule("site", "", {"lifetime": {"idle_s": 3600}}),
            _rule("project", "opossum", {"lifetime": {"idle_s": 600}}, layer="owner"),
        )
        assert limits["lifetime"]["idle_s"] == 600
        assert limits["origin"]["lifetime.idle_s"] == "project:owner"

    def test_owner_never_widens_in_the_fold(self):
        limits = _limits(
            _rule("site", "", {"lifetime": {"idle_s": 600}}),
            _rule("project", "opossum", {"lifetime": {"idle_s": 3600}}, layer="owner"),
        )
        assert limits["lifetime"]["idle_s"] == 600
        assert limits["origin"]["lifetime.idle_s"] == "site:admin"

    def test_domain_cap_removes_public(self):
        limits = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}))
        assert limits["web"]["allow"] == ["none", "allowlist"]
        assert limits["web"]["domains"] == [".arxiv.org"]
        assert limits["origin"]["web.allow"] == "project:admin"

    def test_disjoint_domain_caps_leave_only_none(self):
        limits = _limits(
            _rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}),
            _rule("user", "alice", {"web": {"domains": ["pypi.org"]}}),
        )
        assert limits["web"]["allow"] == ["none"]

    def test_domain_caps_intersect_by_dstdomain(self):
        limits = _limits(
            _rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}),
            _rule("user", "alice", {"web": {"domains": ["export.arxiv.org", "pypi.org"]}}),
        )
        assert limits["web"]["domains"] == ["export.arxiv.org"]

    def test_this_sites_rule_beats_the_all_sites_rule(self):
        limits = _limits(
            _rule("site", "", {"sessions": {"concurrent_user": 2}}, venue=""),
            _rule("site", "", {"sessions": {"concurrent_user": 6}}),
        )
        assert limits["sessions"]["concurrent_user"] == 6
        other = _limits(
            _rule("site", "", {"sessions": {"concurrent_user": 2}}, venue=""),
            _rule("site", "", {"sessions": {"concurrent_user": 6}}),
            venue="bdrt",
        )
        assert other["sessions"]["concurrent_user"] == 2

    def test_web_allow_intersects_and_none_stays(self):
        limits = _limits(
            _rule("project", "opossum", {"web": {"allow": ["public"]}}),
            _rule("user", "alice", {"web": {"allow": ["allowlist"]}}),
        )
        assert limits["web"]["allow"] == ["none"]


class TestOwnerRule:
    def test_owner_narrowing_passes(self):
        admin = _limits(_rule("project", "opossum", {"budget": {"rpm_user": 60}}), user="")
        check_owner_rule({"budget": {"rpm_user": 10}}, admin)

    def test_owner_widening_refused(self):
        admin = _limits(_rule("project", "opossum", {"budget": {"rpm_user": 60}}), user="")
        with pytest.raises(PolicyError) as err:
            check_owner_rule({"budget": {"rpm_user": 100}}, admin)
        assert err.value.code == "widening_refused"
        assert err.value.detail == {
            "field": "budget.rpm_user",
            "requested": 100,
            "cap": 60,
            "capped_by": "project:admin",
        }

    def test_owner_cannot_add_a_mode_or_leave_the_domain_cap(self):
        admin = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}), user="")
        with pytest.raises(PolicyError, match="web.allow"):
            check_owner_rule({"web": {"allow": ["public"]}}, admin)
        with pytest.raises(PolicyError, match="web.domains"):
            check_owner_rule({"web": {"domains": ["pypi.org"]}}, admin)
        check_owner_rule({"web": {"domains": ["export.arxiv.org"], "allow": ["none"]}}, admin)


class TestChoice:
    def test_choice_inside_the_caps(self):
        limits = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}))
        policy = _choose(limits, "allowlist", "export.arxiv.org", "ARXIV.org.")
        assert policy["web"] == {"mode": "allowlist", "domains": ["export.arxiv.org", "arxiv.org"]}
        assert policy["origin"]["web.mode"] == policy["origin"]["web.domains"] == "choice"
        assert "web.allow" not in policy["origin"]
        assert policy["resolved_at"] == "2026-10-23T09:00:00Z"
        assert policy["sessions"]["concurrent_user"] == 12

    def test_choice_widening_refused(self):
        limits = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}))
        with pytest.raises(PolicyError) as err:
            _choose(limits, "public")
        assert err.value.code == "widening_refused"
        assert err.value.detail["field"] == "web.mode"
        with pytest.raises(PolicyError) as err:
            _choose(limits, "allowlist", "pypi.org")
        assert err.value.detail["field"] == "web.domains"
        assert err.value.detail["capped_by"] == "project:admin"

    def test_no_choice_while_required(self):
        limits = _limits(
            _rule(
                "project",
                "opossum",
                {"web": {"domains": [".arxiv.org"], "suggested_domains": [".arxiv.org"]}},
            )
        )
        with pytest.raises(PolicyError) as err:
            _choose(limits, "")
        assert err.value.code == "web_choice_required"
        assert err.value.detail == {
            "choices": ["none", "allowlist"],
            "domains_cap": [".arxiv.org"],
            "suggested_domains": [".arxiv.org"],
        }

    def test_flag_off_falls_to_the_legacy_default_clipped(self):
        assert _choose(_limits(), "", required=False)["web"]["mode"] == "public"
        capped = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}))
        policy = _choose(capped, "", required=False)
        assert policy["web"] == {"mode": "allowlist", "domains": [".arxiv.org"]}
        assert policy["origin"]["web.mode"] == "default"

    def test_the_legacy_allowlist_keeps_the_domains_the_client_sent(self):
        """P5: an old client sends domains without a mode; with the flag off
        the legacy allowlist uses those inside the cap (old code: dropped,
        the whole cap applied instead)."""
        capped = _limits(_rule("project", "opossum", {"web": {"domains": [".arxiv.org"]}}))
        policy = _choose(capped, "", "export.arxiv.org", "pypi.org", required=False)
        assert policy["web"] == {"mode": "allowlist", "domains": ["export.arxiv.org"]}
        outside_only = _choose(capped, "", "pypi.org", required=False)
        assert outside_only["web"] == {"mode": "allowlist", "domains": [".arxiv.org"]}
        uncapped = _choose(_limits(), "", "pypi.org", required=False, legacy="allowlist")
        assert uncapped["web"] == {"mode": "allowlist", "domains": ["pypi.org"]}

    def test_the_choice_required_message_stands_alone(self):
        """P3: the typed detail never reaches a client (only the message
        does), so the message names the choices, the cap and the hints."""
        limits = _limits(
            _rule(
                "project",
                "opossum",
                {"web": {"domains": [".arxiv.org"], "suggested_domains": [".arxiv.org"]}},
            )
        )
        with pytest.raises(PolicyError) as err:
            _choose(limits, "")
        message = err.value.message
        assert "none | allowlist" in message and "within: .arxiv.org" in message
        assert "suggested: .arxiv.org" in message and "--web" in message

    def test_allowlist_without_domains_is_invalid(self):
        with pytest.raises(PolicyError) as err:
            _choose(_limits(), "allowlist")
        assert (err.value.code, err.value.detail["field"]) == ("invalid", "web.domains")

    def test_unknown_mode_is_invalid(self):
        with pytest.raises(PolicyError) as err:
            _choose(_limits(), "wide-open")
        assert err.value.code == "invalid"

    def test_none_and_public_drop_domains(self):
        assert _choose(_limits(), "public", "pypi.org")["web"]["domains"] == []
        assert _choose(_limits(), "none")["web"] == {"mode": "none", "domains": []}


class TestValidation:
    @pytest.mark.parametrize(
        ("doc", "field"),
        [
            ({"resources": {"cpus": 4}}, "resources"),
            ({"budget": {"usd": 3}}, "budget.usd"),
            ({"budget": {"rpm_user": -1}}, "budget.rpm_user"),
            ({"budget": {"rpm_user": True}}, "budget.rpm_user"),
            ({"budget": {"rpm_user": "60"}}, "budget.rpm_user"),
            ({"lifetime": {"idle_s": 30}}, "lifetime.idle_s"),
            ({"web": {"allow": []}}, "web.allow"),
            ({"web": {"allow": ["everything"]}}, "web.allow"),
            ({"web": {"domains": ["not a domain"]}}, "web.domains"),
            ({"web": {"domains": ["x.org"] * 65}}, "web.domains"),
            ({"web": "public"}, "web"),
        ],
    )
    def test_bad_documents_are_invalid(self, doc, field):
        with pytest.raises(PolicyError) as err:
            validate_rule_doc(doc, scope="site", layer="admin")
        assert (err.value.code, err.value.detail["field"]) == ("invalid", field)

    def test_project_fields_refused_in_a_user_rule(self):
        with pytest.raises(PolicyError, match="budget.tokens_per_day_project"):
            validate_rule_doc(
                {"budget": {"tokens_per_day_project": 5}}, scope="user", layer="admin"
            )

    def test_owner_layer_only_for_projects(self):
        with pytest.raises(PolicyError) as err:
            validate_rule_doc({}, scope="user", layer="owner")
        assert err.value.detail["field"] == "layer"

    def test_domains_are_normalized_and_deduplicated(self):
        doc = validate_rule_doc(
            {"web": {"domains": [" PyPI.org. ", "pypi.org"], "allow": ["public", "none"]}},
            scope="project",
            layer="owner",
        )
        assert doc == {"web": {"domains": ["pypi.org"], "allow": ["none", "public"]}}

    def test_null_means_no_limit_at_this_layer(self):
        doc = validate_rule_doc({"budget": {"rpm_user": None}}, scope="site", layer="admin")
        assert doc == {"budget": {"rpm_user": None}}


@pytest.mark.parametrize(
    ("domain", "cap", "inside"),
    [
        ("arxiv.org", ["arxiv.org"], True),
        ("export.arxiv.org", ["arxiv.org"], False),
        ("arxiv.org", [".arxiv.org"], True),
        ("export.arxiv.org", [".arxiv.org"], True),
        (".export.arxiv.org", [".arxiv.org"], True),
        (".arxiv.org", ["arxiv.org"], False),
        ("evilarxiv.org", [".arxiv.org"], False),
    ],
)
def test_domain_within_follows_squid_dstdomain(domain, cap, inside):
    assert domain_within(domain, cap) is inside
