"""Walltime relay (ADR-076): an always-on roster endpoint survives the 24 h
Slurm limit by handing over to a successor job before the deadline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from umrs.orchestrator import provision_roster as pr
from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint
from umrs.orchestrator.endpoint_relay import (
    KeepAlivePolicy,
    keep_alive_policies,
    parse_slurm_duration,
    plan_relay,
)

POOL = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"
SITE = Path(__file__).resolve().parents[1] / "deploy/rccs-front/site.yaml"
NOW = datetime(2026, 10, 2, 12, 0, 0)
WALLTIME = 23 * 3600
POLICY = KeepAlivePolicy(model="Bespoke-Nimble-9B", project_id="default")


def _ep(job_id, status, *, age_h, ready_ago_min=None, active=0, project="default"):
    return RegisteredEndpoint(
        project_id=project,
        model="Bespoke-Nimble-9B",
        job_id=job_id,
        host="10.0.0.1",
        port=4000,
        status=status,
        created_at=NOW - timedelta(hours=age_h),
        ready_at=(NOW - timedelta(minutes=ready_ago_min)) if ready_ago_min is not None else None,
        active_requests=active,
    )


def _plan(*endpoints, policy=POLICY, walltime=WALLTIME):
    return [
        (a.action, a.job_id) for a in plan_relay(policy, endpoints, walltime_sec=walltime, now=NOW)
    ]


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("23:00:00", 23 * 3600),
        ("1-00:00:00", 24 * 3600),
        ("2-00:00:00", 48 * 3600),
        ("1-12", 36 * 3600),
        ("90", 90 * 60),
        ("05:30", 5 * 60 + 30),
        ("UNLIMITED", None),
        ("", None),
        ("soon", None),
    ],
)
def test_parse_slurm_duration(text, seconds):
    assert parse_slurm_duration(text) == seconds


def test_nothing_running_provisions():
    assert _plan() == [("provision", None)]


def test_crashed_endpoint_respawns():
    assert _plan(_ep("j1", EndpointStatus.FAILED, age_h=3)) == [("provision", None)]


def test_pending_first_job_waits():
    assert _plan(_ep("j1", EndpointStatus.PROVISIONING, age_h=0.2)) == []


def test_young_endpoint_needs_nothing():
    assert _plan(_ep("j1", EndpointStatus.READY, age_h=5, ready_ago_min=240)) == []


def test_successor_starts_inside_renew_window():
    # 23 h walltime, 2 h window: at 21.5 h the successor is due.
    assert _plan(_ep("j1", EndpointStatus.READY, age_h=21.5, ready_ago_min=1200)) == [
        ("provision", None)
    ]


def test_successor_not_doubled_while_pending():
    old = _ep("j1", EndpointStatus.READY, age_h=21.5, ready_ago_min=1200)
    new = _ep("j2", EndpointStatus.STARTING, age_h=0.3)
    assert _plan(old, new) == []


def test_old_kept_until_successor_ready_for_grace():
    old = _ep("j1", EndpointStatus.READY, age_h=22, ready_ago_min=1250)
    new = _ep("j2", EndpointStatus.READY, age_h=0.5, ready_ago_min=2)  # < 300 s grace
    assert _plan(old, new) == []


def test_old_retired_after_handover():
    old = _ep("j1", EndpointStatus.READY, age_h=22, ready_ago_min=1250)
    new = _ep("j2", EndpointStatus.READY, age_h=0.8, ready_ago_min=10)
    assert _plan(old, new) == [("retire", "j1")]


def test_busy_old_drains_until_the_floor():
    new = _ep("j2", EndpointStatus.READY, age_h=0.8, ready_ago_min=10)
    busy = _ep("j1", EndpointStatus.READY, age_h=22, ready_ago_min=1250, active=2)
    assert _plan(busy, new) == []
    last_minutes = _ep("j1", EndpointStatus.READY, age_h=22.95, ready_ago_min=1300, active=2)
    assert _plan(last_minutes, new) == [("retire", "j1")]


def test_other_projects_are_not_counted():
    foreign = _ep("j9", EndpointStatus.READY, age_h=1, ready_ago_min=30, project="opossum")
    assert _plan(foreign) == [("provision", None)]


def test_no_walltime_only_respawns():
    old = _ep("j1", EndpointStatus.READY, age_h=100, ready_ago_min=5000)
    assert _plan(old, walltime=None) == []
    assert _plan(walltime=None) == [("provision", None)]


def test_policy_walltime_overrides_site():
    policy = KeepAlivePolicy(model="Bespoke-Nimble-9B", project_id="default", walltime_sec=4 * 3600)
    assert _plan(_ep("j1", EndpointStatus.READY, age_h=2.5, ready_ago_min=100), policy=policy) == [
        ("provision", None)
    ]


def test_policies_require_a_project():
    entries = [
        {
            "model": "A",
            "keep_alive": {"project_id": "p", "renew_before_sec": 60, "walltime": "1-00:00:00"},
        },
        {"model": "A", "keep_alive": {"project_id": "other"}},  # first entry per model wins
        {"model": "B", "keep_alive": {}},  # no owner → skipped
        {"model": "C"},
    ]
    policies = keep_alive_policies(entries)
    assert [(p.model, p.project_id, p.renew_before_sec, p.walltime_sec) for p in policies] == [
        ("A", "p", 60, 24 * 3600)
    ]


@pytest.fixture
def rccs_roster(monkeypatch):
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(POOL))
    pr._cache.clear()
    yield
    pr._cache.clear()


def test_rccs_nimble_entry_is_kept_alive_on_ng_dgx_s(rccs_roster):
    (policy,) = keep_alive_policies(pr.roster_entries())
    assert (policy.model, policy.project_id) == ("Bespoke-Nimble-9B", "default")
    # The successor window must cover a cold start (~45 min Lustre load).
    assert policy.renew_before_sec >= 3600
    defaults = pr.resolve_model_provision_defaults("Bespoke-Nimble-9B")
    assert (defaults["partition"], defaults["ttl_sec"], defaults["max_model_len"]) == (
        "ng-dgx-s",
        0,
        8192,
    )
    env = defaults["env_vars"]
    # logprob_token_ids on /v1/completions needs vLLM >= 0.26 (a pinned tag).
    major, minor = env["VLLM_DOCKER_IMAGE"].rsplit(":v", 1)[1].split(".")[:2]
    assert (int(major), int(minor)) >= (0, 26)
    assert float(env["VLLM_GPU_MEM_UTIL"]) <= 0.78
    assert env["VLLM_LANGUAGE_MODEL_ONLY"] == "1"


def test_ng_dgx_s_is_a_valid_rccs_front_partition_but_not_a_default():
    import yaml

    site = yaml.safe_load(SITE.read_text())
    assert "ng-dgx-s" in site["partition_profiles"]
    assert "ng-dgx-s" not in site["scheduler"]["partition"].split(",")
    # The relay's deadline comes from the site time_limit; it must sit under
    # the 1-00:00:00 partition maximum.
    assert parse_slurm_duration(site["scheduler"]["time_limit"]) < 24 * 3600


def test_roster_entries_dedupes_aliases(tmp_path, monkeypatch):
    roster = tmp_path / "pool.yaml"
    roster.write_text(
        "endpoints:\n  - name: a\n    model: M\n    aliases: [m-alias]\n  - model: N\n"
    )
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(roster))
    pr._cache.clear()
    try:
        assert sorted(e["model"] for e in pr.roster_entries()) == ["M", "N"]
    finally:
        pr._cache.clear()


def test_runner_provisions_through_the_api_handler(rccs_roster, monkeypatch):
    """The executor reaches the same handler as `r4 endpoint up`, as an admin
    system-tier provision that bypasses warm reuse (else the successor would
    short-circuit to the expiring endpoint)."""
    import asyncio

    from umrs.api import endpoints as api
    from umrs.server import endpoint_relay as runner

    calls = []

    async def fake_provision_endpoint(**kwargs):
        calls.append(kwargs)
        return api.ProvisionResponse(status="provisioning", job_id="j-new")

    class _Registry:
        def list_all(self):
            return [_ep("j1", EndpointStatus.READY, age_h=21.5, ready_ago_min=1200)]

    monkeypatch.setattr(api, "provision_endpoint", fake_provision_endpoint)
    monkeypatch.setattr(runner, "get_registry", lambda: _Registry())
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: WALLTIME)

    actions = asyncio.run(runner.run_relay_once(now=NOW))

    assert [a.action for a in actions] == ["provision"]
    (call,) = calls
    request = call["request"]
    assert (request.model, request.tier, request.ttl_sec, request.prefer_warm) == (
        "Bespoke-Nimble-9B",
        "system",
        0,
        False,
    )
    assert (call["project_id"], call["is_admin"], call["requesting_user"]) == (
        "default",
        True,
        "mnagaso",
    )
