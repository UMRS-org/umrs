"""Host placement (ADR-098 D-7, D-9; decision SH-D5, slice SH-H2).

Acceptance: a pinned down host is refused (no front fallback); a sensitive
project with only crun/squid-only hosts is refused; the least-loaded host is
chosen; a single node is unchanged. Plus the ADR-099 ``placement.labels`` and
``mcp.max_instances`` policy fields.

Fail before SH-H2: ``umrs.hosts.placement`` did not exist, the venue start
never consulted the hosts registry (a sensitive project started on a
squid-only front; a drained front still took sessions), and the policy
refused both new fields as unknown.
"""

from __future__ import annotations

import pytest
from test_companion_venue_routes import FakeSealed, _meta, _start
from test_companion_venue_routes import harness as _venue_harness  # noqa: F401 — fixture
from test_hosts_janitor_pg import _db_url, _session
from test_hosts_janitor_pg import pool as _pg_pool  # noqa: F401 — fixture
from umrs.hosts import HostRecord, HostRegistry
from umrs.hosts.placement import (
    DEFAULT_ISOLATION,
    HOST_UNAVAILABLE,
    NO_HOST,
    PlacementRefused,
    PlacementRequest,
    inline_host_profile,
    label_matches,
    meets_isolation_floor,
    parse_cpus,
    parse_memory_gb,
    place,
)
from umrs.hosts.registry import _Change
from umrs.session_policy import (
    PolicyError,
    PolicyRule,
    WebChoice,
    check_owner_rule,
    effective_limits,
    mcp_max_instances,
    placement_labels,
    resolve_session_policy,
    site_defaults,
    validate_rule_doc,
)


@pytest.fixture()
def harness(_venue_harness):  # noqa: F811 — the imported fixture, by name
    """The venue routes' harness (test_companion_venue_routes)."""
    return _venue_harness


@pytest.fixture()
def pool(_pg_pool):  # noqa: F811 — the imported fixture, by name
    """A scratch-schema Postgres pool (test_hosts_janitor_pg)."""
    return _pg_pool


BOTH = ("inline", "agent")
NFT = {"runtime": "crun", "egress": "nft"}


def host(host_id: str, **kw) -> HostRecord:
    return HostRecord(host_id=host_id, venue="site-a", **kw)


def agent(host_id: str, **kw) -> HostRecord:
    return host(host_id, launch_mode="agent", **kw)


def refused(hosts, req, load=None) -> PlacementRefused:
    with pytest.raises(PlacementRefused) as exc:
        place(hosts, req, load)
    return exc.value


# ── stage 3: least loaded ──────────────────────────────────────────────────


def test_single_inline_host_is_placed_unchanged():
    """A single node: its own inline host, whatever it reports."""
    assert place([host("local")], PlacementRequest(cpus=8, memory_gb=16)).host_id == "local"


def test_least_loaded_active_host_is_chosen():
    hosts = [agent("h-a"), agent("h-b"), agent("h-c")]
    load = {"h-a": 3, "h-b": 1, "h-c": 2}
    assert place(hosts, PlacementRequest(launch_modes=BOTH), load).host_id == "h-b"


def test_equal_load_breaks_ties_by_host_id():
    hosts = [agent("h-b"), agent("h-a")]
    assert place(hosts, PlacementRequest(launch_modes=BOTH), {}).host_id == "h-a"


def test_draining_and_down_hosts_are_never_placed():
    hosts = [agent("h-a", state="draining"), agent("h-b", state="down"), agent("h-c")]
    load = {"h-c": 9}
    assert place(hosts, PlacementRequest(launch_modes=BOTH), load).host_id == "h-c"


def test_no_active_host_is_no_host():
    exc = refused([host("local", state="draining")], PlacementRequest())
    assert exc.code == NO_HOST
    assert exc.detail["considered"] == 0


# ── stage 1: role, launch mode, resources ──────────────────────────────────


def test_agent_hosts_are_not_placed_until_the_agent_mode_exists():
    """This server drives inline hosts only (umrs-hostd is SH-H3)."""
    exc = refused([agent("h-a")], PlacementRequest())
    assert exc.code == NO_HOST
    assert exc.detail["refused"] == {"launch_mode": 1}


def test_role_mismatch_refused():
    exc = refused([host("local", roles=("mcp",))], PlacementRequest(role="session"))
    assert exc.detail["refused"] == {"role": 1}


@pytest.mark.parametrize(
    "capacity",
    [{"cpus": 4}, {"memory_gb": 8}, {"max_sessions": 2}],
)
def test_resources_refuse_a_host_short_of_capacity(capacity):
    hosts = [agent("small", capacity=capacity), agent("big", capacity={"cpus": 64})]
    req = PlacementRequest(cpus=8, memory_gb=16, launch_modes=BOTH)
    load = {"small": 2, "big": 5}
    assert place(hosts, req, load).host_id == "big"


def test_full_host_refused_and_named():
    exc = refused([host("local", capacity={"max_sessions": 1})], PlacementRequest(), {"local": 1})
    assert exc.code == NO_HOST
    assert exc.detail["refused"] == {"resources": 1}
    assert "resources" in exc.message


def test_unreported_capacity_is_no_limit():
    assert place([host("local", capacity={"cpus": "junk"})], PlacementRequest(cpus=99)).host_id


@pytest.mark.parametrize(
    ("raw", "gib"),
    [("16g", 16.0), ("512m", 0.5), ("1t", 1024.0), (str(2 * 1024**3), 2.0), ("", 0.0), ("x", 0.0)],
)
def test_parse_memory(raw, gib):
    assert parse_memory_gb(raw) == pytest.approx(gib)


@pytest.mark.parametrize(
    ("raw", "gib"),
    [("16gb", 16.0), ("16GiB", 16.0), ("16Gb", 16.0), ("512MB", 0.5), ("1024kb", 1 / 1024)],
)
def test_parse_memory_reads_every_podman_unit_spelling(raw, gib):
    """Review SH-H2: podman's ``--memory`` takes ``16gb``/``16GiB``; they
    parsed as 0 GiB, so a host's ``memory_gb`` capacity never refused them."""
    assert parse_memory_gb(raw) == pytest.approx(gib)


def test_memory_capacity_refuses_a_gb_spelled_worker_size():
    req = PlacementRequest(memory_gb=parse_memory_gb("16gb"))
    exc = refused([host("local", capacity={"memory_gb": 8})], req)
    assert exc.detail["refused"] == {"resources": 1}


def test_parse_cpus():
    assert parse_cpus("8") == 8.0
    assert parse_cpus("") == 0.0
    assert parse_cpus(True) == 0.0


# ── stage 2: labels and the isolation floor ────────────────────────────────


def test_required_labels_select_hosts():
    hosts = [agent("cpu-a", labels={"arch": "x86_64"}), agent("arm-a", labels={"arch": "aarch64"})]
    req = PlacementRequest(labels=("arch=aarch64",), launch_modes=BOTH)
    assert place(hosts, req, {"arm-a": 5}).host_id == "arm-a"


def test_missing_label_is_no_host():
    exc = refused([host("local")], PlacementRequest(labels=("gpu",)))
    assert exc.detail["refused"] == {"labels": 1}


def test_label_match_forms():
    labels = {"gpu": True, "zone": "a", "off": False}
    assert label_matches(labels, "gpu")
    assert label_matches(labels, "gpu=true")
    assert label_matches(labels, "zone=a")
    assert not label_matches(labels, "zone=b")
    assert not label_matches(labels, "off")
    assert not label_matches(labels, "absent")


def test_sensitive_project_refused_on_crun_squid_only_hosts():
    """D-7: the floor refuses a sensitive project where egress is squid-only,
    and a host reporting nothing counts as crun/squid-only."""
    hosts = [host("local"), agent("h-a", isolation={"runtime": "runsc", "egress": "squid-only"})]
    exc = refused(hosts, PlacementRequest(sensitive=True, launch_modes=BOTH))
    assert exc.code == NO_HOST
    assert exc.detail["refused"] == {"isolation": 2}
    assert "isolation" in exc.message


def test_sensitive_project_lands_on_an_nft_host():
    hosts = [host("local"), agent("h-nft", isolation=NFT)]
    req = PlacementRequest(sensitive=True, launch_modes=BOTH)
    assert place(hosts, req, {"local": 0, "h-nft": 7}).host_id == "h-nft"


def test_site_declared_sensitive_tier_admits_the_inline_front():
    front = host("local", isolation={**DEFAULT_ISOLATION, "tier": "sensitive"})
    assert meets_isolation_floor(front)
    assert place([front], PlacementRequest(sensitive=True)).host_id == "local"


def test_agent_host_cannot_declare_the_site_tier_for_itself():
    liar = agent("h-a", isolation={"egress": "squid-only", "tier": "sensitive"})
    assert not meets_isolation_floor(liar)


def test_non_sensitive_project_ignores_the_floor():
    assert place([host("local")], PlacementRequest(sensitive=False)).host_id == "local"


# ── pins ───────────────────────────────────────────────────────────────────


def _pin(pin: str, **kw) -> PlacementRequest:
    return PlacementRequest(pin=pin, declared=frozenset({"atom-0"}), launch_modes=BOTH, **kw)


def test_pinned_down_host_refused_without_front_fallback():
    hosts = [host("local"), agent("atom-0", state="down")]
    exc = refused(hosts, _pin("atom-0"))
    assert exc.code == HOST_UNAVAILABLE
    assert exc.detail == {"pin": "atom-0", "reason": "down"}


@pytest.mark.parametrize(
    ("hosts", "reason"),
    [
        ([host("local"), agent("atom-0", state="draining")], "draining"),
        ([host("local")], "unregistered"),
        ([host("local"), agent("atom-0", isolation={"egress": "squid-only"})], "isolation"),
    ],
)
def test_pin_refusals_are_named(hosts, reason):
    exc = refused(hosts, _pin("atom-0", sensitive=True))
    assert exc.code == HOST_UNAVAILABLE
    assert exc.detail["reason"] == reason


def test_undeclared_pin_refused_even_when_registered():
    exc = refused([agent("ghost")], _pin("ghost"))
    assert (exc.code, exc.detail["reason"]) == (HOST_UNAVAILABLE, "undeclared")


def test_pin_to_an_agent_host_refused_until_the_agent_mode_exists():
    req = PlacementRequest(pin="atom-0", declared=frozenset({"atom-0"}))
    exc = refused([host("local"), agent("atom-0")], req)
    assert (exc.code, exc.detail["reason"]) == (HOST_UNAVAILABLE, "launch_mode")


def test_active_declared_pin_is_placed_even_when_loaded():
    hosts = [host("local"), agent("atom-0")]
    assert place(hosts, _pin("atom-0"), {"atom-0": 9}).host_id == "atom-0"


# ── the inline host's profile (site.yaml extra.hosts.inline) ───────────────


def test_inline_profile_defaults_to_crun_squid_only():
    assert inline_host_profile(None) == {
        "labels": {},
        "capacity": {},
        "isolation": {"runtime": "crun", "egress": "squid-only"},
    }


def test_inline_profile_reads_the_site_declaration():
    extra = {
        "hosts": {
            "inline": {
                "labels": {"arch": "x86_64"},
                "capacity": {"max_sessions": 6},
                "isolation": {"tier": "sensitive"},
                "junk": 1,
            }
        }
    }
    profile = inline_host_profile(extra)
    assert profile["labels"] == {"arch": "x86_64"}
    assert profile["capacity"] == {"max_sessions": 6}
    assert profile["isolation"] == {"runtime": "crun", "egress": "squid-only", "tier": "sensitive"}


def test_inline_profile_ignores_malformed_sections():
    profile = inline_host_profile({"hosts": {"inline": {"labels": ["x"], "isolation": "nft"}}})
    assert profile["labels"] == {}
    assert profile["isolation"] == DEFAULT_ISOLATION


def _boot_inline(monkeypatch, loader) -> HostRecord:
    from umrs.hosts.janitor import start_inline_host

    monkeypatch.setattr("umrs.orchestrator.site_config.load_site_config", loader)
    maintenance = start_inline_host(None, broker=object(), venue="site-a")
    maintenance.stop()
    record = maintenance.registry.get("local")
    assert record is not None
    return record


def test_inline_host_registers_the_site_profile(monkeypatch):
    from types import SimpleNamespace

    extra = {"hosts": {"inline": {"isolation": {"egress": "nft"}, "labels": {"zone": "a"}}}}
    record = _boot_inline(monkeypatch, lambda: SimpleNamespace(extra=extra))
    assert record.isolation == {"runtime": "crun", "egress": "nft"}
    assert record.labels == {"zone": "a"}


def test_inline_host_without_a_site_file_reports_the_default(monkeypatch):
    def missing():
        raise FileNotFoundError("no site.yaml")

    record = _boot_inline(monkeypatch, missing)
    assert record.isolation == DEFAULT_ISOLATION


def test_registry_rows_feed_placement():
    registry = HostRegistry(None, venue="site-a")
    registry.register("local")
    registry.drain("local", "maintenance")
    with pytest.raises(PlacementRefused) as exc:
        place(registry.hosts(), PlacementRequest())
    assert exc.value.code == NO_HOST


# ── ADR-099: placement.labels and mcp.max_instances ────────────────────────


def _rule(scope, subject, layer, doc):
    norm = validate_rule_doc(doc, scope=scope, layer=layer)
    return PolicyRule(venue="", scope=scope, subject=subject, layer=layer, doc=norm)


def _limits(*rules):
    return effective_limits(
        list(rules), venue="site-a", user_id="u", project_id="p", defaults=site_defaults({})
    )


def test_policy_accepts_the_new_fields():
    doc = {"placement": {"labels": ["arch=aarch64", "gpu", "gpu"]}, "mcp": {"max_instances": 3}}
    norm = validate_rule_doc(doc, scope="site", layer="admin")
    assert norm == {"placement": {"labels": ["arch=aarch64", "gpu"]}, "mcp": {"max_instances": 3}}


@pytest.mark.parametrize(
    "doc",
    [
        {"placement": {"labels": "gpu"}},
        {"placement": {"labels": ["Bad Label"]}},
        {"placement": {"labels": [f"k{i}" for i in range(17)]}},
        {"placement": {"hosts": ["x"]}},
        {"mcp": {"max_instances": 0}},
    ],
)
def test_policy_refuses_bad_values(doc):
    with pytest.raises(PolicyError) as exc:
        validate_rule_doc(doc, scope="site", layer="admin")
    assert exc.value.code == "invalid"


def test_site_defaults_require_nothing():
    limits = _limits()
    assert limits["placement"] == {"labels": None}
    assert limits["mcp"] == {"max_instances": None}
    assert placement_labels(limits) == ()
    assert mcp_max_instances(limits) is None


def test_label_requirements_add_up_across_layers():
    limits = _limits(
        _rule("site", "", "admin", {"placement": {"labels": ["arch=x86_64"]}}),
        _rule("project", "p", "owner", {"placement": {"labels": ["gpu"]}}),
    )
    assert placement_labels(limits) == ("arch=x86_64", "gpu")
    assert limits["origin"]["placement.labels"] == "project:owner"


def test_admin_null_lifts_a_lower_label_requirement():
    limits = _limits(
        _rule("site", "", "admin", {"placement": {"labels": ["gpu"]}}),
        _rule("user", "u", "admin", {"placement": {"labels": None}}),
    )
    assert placement_labels(limits) == ()


def test_a_project_admin_label_rule_replaces_the_site_admin_rule():
    """Labels fold like every admin limit (ADR-099 amendment, corrected in
    the SH-H2 review): the specific admin rule replaces the site's; the
    project and user admin rules and the owner rule add up."""
    limits = _limits(
        _rule("site", "", "admin", {"placement": {"labels": ["zone=a"]}}),
        _rule("project", "p", "admin", {"placement": {"labels": ["gpu"]}}),
        _rule("user", "u", "admin", {"placement": {"labels": ["arch=x86_64"]}}),
        _rule("project", "p", "owner", {"placement": {"labels": ["fast"]}}),
    )
    assert placement_labels(limits) == ("gpu", "arch=x86_64", "fast")


def test_owner_labels_never_count_as_widening():
    admin = _limits(_rule("site", "", "admin", {"placement": {"labels": ["gpu"]}}))
    check_owner_rule({"placement": {"labels": ["zone=a"]}}, admin)


def test_owner_cannot_raise_mcp_max_instances_past_the_admin():
    admin = _limits(_rule("site", "", "admin", {"mcp": {"max_instances": 2}}))
    with pytest.raises(PolicyError) as exc:
        check_owner_rule({"mcp": {"max_instances": 5}}, admin)
    assert exc.value.code == "widening_refused"


def test_mcp_max_instances_folds_to_the_strictest():
    limits = _limits(
        _rule("project", "p", "admin", {"mcp": {"max_instances": 4}}),
        _rule("user", "u", "admin", {"mcp": {"max_instances": 2}}),
    )
    assert mcp_max_instances(limits) == 2


def test_resolved_policy_carries_the_new_sections():
    from datetime import datetime, timezone

    limits = _limits(_rule("site", "", "admin", {"placement": {"labels": ["gpu"]}}))
    policy = resolve_session_policy(
        limits,
        WebChoice("none"),
        require_choice=True,
        legacy_default="public",
        now=datetime.now(timezone.utc),
    )
    assert placement_labels(policy) == ("gpu",)
    assert policy["mcp"] == {"max_instances": None}


# ── the venue start (FrontendVenueService, routes harness) ─────────────────


def _registry(**inline) -> HostRegistry:
    registry = HostRegistry(None, venue="site-a")
    registry.register("local", **inline)
    return registry


def _own_model_start(h):
    """A sensitive project's start with a self-deployed source (the ADR-062
    source gate admits it), so only placement can refuse it."""
    own = _meta("own-model", allow_host="https://models.example.org")
    own.provider_kind = "vllm"
    h.service._sealed = FakeSealed({("alice", "own-model"): own})
    h.pool.seed("alice", "sess-1")
    h.pool.projects = {"proj-s": "sensitive"}
    return _start(h, source_refs=["own-model"], project="proj-s")


def test_single_node_start_unchanged_and_names_its_host(harness):
    harness.pool.seed("alice", "sess-1")
    r = _start(harness)
    assert r.status_code == 200, r.text
    assert harness.jobs.created[0]["spec"]["host_id"] == "local"


def test_drained_front_takes_no_new_session(harness):
    harness.pool.seed("alice", "sess-1")
    registry = _registry()
    registry.drain("local", "maintenance")
    harness.service.host_registry = registry
    r = _start(harness)
    assert r.status_code == 503, r.text
    assert r.headers["x-platform-error-code"] == "companionvenue.no_host"
    assert not harness.jobs.created


def test_sensitive_project_refused_on_a_squid_only_front(harness):
    """The inline front reports crun/squid-only unless the site declares a
    tier: a sensitive project's session is refused by name, no job made."""
    r = _own_model_start(harness)
    assert r.status_code == 503, r.text
    assert r.headers["x-platform-error-code"] == "companionvenue.no_host"
    assert "isolation" in r.json()["detail"]
    assert not harness.jobs.created


def test_sensitive_project_starts_on_a_declared_sensitive_front(harness):
    harness.service.host_registry = _registry(isolation={"tier": "sensitive"})
    r = _own_model_start(harness)
    assert r.status_code == 200, r.text


def test_sensitive_project_starts_on_an_nft_front(harness):
    harness.service.host_registry = _registry(isolation=NFT)
    assert _own_model_start(harness).status_code == 200


def test_pinned_session_refused_when_its_host_is_down(harness, monkeypatch):
    """A cpu:<host> session whose host is down: 409 host_unavailable, and
    never a job on the front, which is active and empty."""
    harness.pool.seed("alice", "sess-1")
    harness.pool.table[("alice", "sess-1")]["venue"] = "cpu:atom-0"
    monkeypatch.setattr("umrs_r4.venues.frontend_service.declared_cpu_hosts", lambda: {"atom-0"})
    registry = _registry()
    registry.register("atom-0", launch_mode="agent")
    registry._backend.transition(  # what the janitor's mark_down leaves
        "site-a", "atom-0", _Change(("active",), "down", "no heartbeat", "")
    )
    harness.service.host_registry = registry
    r = _start(harness)
    assert r.status_code == 409, r.text
    assert r.headers["x-platform-error-code"] == "companionvenue.host_unavailable"
    assert not harness.jobs.created


def test_policy_labels_the_front_lacks_refuse_the_start(harness):
    from umrs.session_policy import PolicyRuleStore

    harness.pool.seed("alice", "sess-1")
    store = PolicyRuleStore()
    store.put(
        PolicyRule(
            venue="", scope="site", subject="", layer="admin",
            doc={"placement": {"labels": ["gpu"]}},
        )
    )  # fmt: skip
    harness.service.policy_store = store
    r = _start(harness)
    assert r.status_code == 503, r.text
    assert r.headers["x-platform-error-code"] == "companionvenue.no_host"
    assert "labels" in r.json()["detail"]


# ── the load map on Postgres (env-gated like the other *_pg suites) ────────


@pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")
def test_host_load_counts_live_sessions_per_host_at_this_site(pool):
    from umrs_r4.quotas.host_load import host_load

    _session(pool, "s1", host="local", site="site-a", state="running")
    _session(pool, "s2", host="local", site="site-a", state="paused")
    _session(pool, "s3", host="h-b", site="site-a", state="provisioning")
    _session(pool, "s4", host="h-b", site="site-a", state="stopped")
    _session(pool, "s5", host="local", site="site-b", state="running")
    assert host_load(pool, "site-a") == {"local": 2, "h-b": 1}
    assert host_load(pool, "site-c") == {}
    assert host_load(None, "site-a") == {}


# ── the hosts registry switched off (review SH-H2) ─────────────────────────


def test_hosts_registry_switched_off_places_on_the_inline_host(monkeypatch):
    """``UMRS_HOSTS_ENABLED=false`` turns the registry loop off, so the
    inline host never registers: placement read the empty Postgres registry
    and refused every start ``no_host``. Off now means this process's inline
    host alone (the service's in-memory registry), as without a pool."""
    from umrs_r4.api.companion_venue import _host_sources

    monkeypatch.setenv("UMRS_HOSTS_ENABLED", "false")
    registry, load = _host_sources(object(), "site-a")
    assert registry is None
    assert callable(load)
    monkeypatch.setenv("UMRS_HOSTS_ENABLED", "true")
    registry, _ = _host_sources(object(), "site-a")
    assert isinstance(registry, HostRegistry)


def test_hosts_enabled_flag():
    from umrs.hosts.janitor import hosts_enabled

    assert hosts_enabled({}) is True
    for off in ("false", "0", "no", " FALSE "):
        assert hosts_enabled({"UMRS_HOSTS_ENABLED": off}) is False
    assert hosts_enabled({"UMRS_HOSTS_ENABLED": "true"}) is True
