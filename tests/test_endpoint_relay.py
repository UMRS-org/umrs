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
BDRT_POOL = Path(__file__).resolve().parents[1] / "deploy/bdrt/pool-bdrt-step.yaml"
BDRT_SITE = Path(__file__).resolve().parents[1] / "deploy/bdrt-front/site.yaml"
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
        ("0", None),  # Slurm "no limit" (bdrt time_limit)
        ("00:00:00", None),
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


def _degraded(job_id, *, age_h, degraded_min):
    ep = _ep(job_id, EndpointStatus.DEGRADED, age_h=age_h)
    ep.metadata["degraded_since"] = (NOW - timedelta(minutes=degraded_min)).isoformat()
    return ep


def test_briefly_degraded_endpoint_still_counts_as_live():
    assert _plan(_degraded("j1", age_h=3, degraded_min=5)) == []


def test_long_degraded_endpoint_is_respawned():
    """2026-10-07: a vanished Nimble job stayed DEGRADED and blocked the respawn."""
    assert _plan(_degraded("j1", age_h=15, degraded_min=20)) == [("provision", None)]


def test_long_degraded_without_timestamp_keeps_the_old_behaviour():
    assert _plan(_ep("j1", EndpointStatus.DEGRADED, age_h=15)) == []


def test_stale_degraded_is_retired_once_a_ready_endpoint_serves():
    fresh = _ep("j2", EndpointStatus.READY, age_h=0.2, ready_ago_min=3)
    assert _plan(_degraded("j1", age_h=15, degraded_min=30), fresh) == [("retire", "j1")]


def test_stale_degraded_waits_while_the_respawn_is_pending():
    pending = _ep("j2", EndpointStatus.PROVISIONING, age_h=0.1)
    assert _plan(_degraded("j1", age_h=15, degraded_min=30), pending) == []


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


def test_rccs_nimble_entry_is_kept_alive_on_a_gb10_partition(rccs_roster):
    (policy,) = keep_alive_policies(pr.roster_entries())
    assert (policy.model, policy.project_id) == ("Bespoke-Nimble-9B", "default")
    # The successor window must cover a cold start (~45 min Lustre load).
    assert policy.renew_before_sec >= 3600
    defaults = pr.resolve_model_provision_defaults("Bespoke-Nimble-9B")
    # ng-dgx-s (8 nodes) first, so it wins ties; every other GB10 partition
    # follows, so a successor never queues behind one busy partition (the
    # 2-node ng-dgx-m1 pin did, 2026-10-08, dgx-spark-fleet-plan §1). UMRS
    # submits to ONE of them (2026-10-24, earliest --test-only start).
    import yaml

    candidates = defaults["partition"].split(",")
    site_candidates = yaml.safe_load(SITE.read_text())["scheduler"]["partition"].split(",")
    # ng-dgx-m3 last: the step lane's partition.
    assert (candidates[0], candidates[-1]) == ("ng-dgx-s", "ng-dgx-m3")
    assert sorted(candidates) == sorted(site_candidates)
    assert (defaults["ttl_sec"], defaults["max_model_len"]) == (0, 8192)
    env = defaults["env_vars"]
    # logprob_token_ids on /v1/completions needs vLLM >= 0.26 (a pinned tag).
    major, minor = env["VLLM_DOCKER_IMAGE"].rsplit(":v", 1)[1].split(".")[:2]
    assert (int(major), int(minor)) >= (0, 26)
    assert float(env["VLLM_GPU_MEM_UTIL"]) <= 0.78
    assert env["VLLM_LANGUAGE_MODEL_ONLY"] == "1"


def test_rccs_front_chooses_among_every_gb10_partition_ng_dgx_s_first_m3_last():
    import yaml

    site = yaml.safe_load(SITE.read_text())
    candidates = site["scheduler"]["partition"].split(",")
    # 2026-10-24: every GB10 partition is a candidate; RCCS refuses a list at
    # sbatch, so the site must choose one per submit. Ties go to ng-dgx-s
    # (singletons on the standalone nodes, pairs kept for pair work) and
    # never first to ng-dgx-m3, the step lane's only partition.
    assert candidates[0] == "ng-dgx-s"
    assert candidates[-1] == site["step_executor"]["partition"] == "ng-dgx-m3"
    assert {"ng-dgx-m0", "ng-dgx-m1", "ng-dgx-m2", "ng-dgx-m3"} < set(candidates)
    assert all(cand in site["partition_profiles"] for cand in candidates)
    assert site["scheduler"]["partition_choice"] == "earliest_start"
    # The relay's deadline comes from the site time_limit; it must sit under
    # the 1-00:00:00 partition maximum.
    assert parse_slurm_duration(site["scheduler"]["time_limit"]) < 24 * 3600


def test_zero_walltime_site_only_respawns():
    """bdrt-style sites set ``time_limit: "0"`` (no Slurm limit): a keep-alive
    endpoint there is only respawned, never relayed (a 0 s walltime would
    provision a successor every cycle)."""
    import yaml

    site = yaml.safe_load(BDRT_SITE.read_text())
    walltime = parse_slurm_duration(site["scheduler"]["time_limit"])
    assert walltime is None
    old = _ep("j1", EndpointStatus.READY, age_h=100, ready_ago_min=5000)
    assert plan_relay(POLICY, [old], walltime_sec=walltime, now=NOW) == []


def test_nimble_is_kept_alive_on_rccs_only(monkeypatch):
    """Maintainer 2026-10-08: the decision model runs on rccs-cloud, not bdrt.
    A bdrt entry without the model staged there made the bdrt relay submit
    a failing Slurm job every cycle (jobs 7843, 7844, ...)."""
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(BDRT_POOL))
    pr._cache.clear()
    try:
        assert keep_alive_policies(pr.roster_entries()) == []
    finally:
        pr._cache.clear()


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
    monkeypatch.setattr(runner, "_relay_backend", lambda: None)

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


# ── Slurm-granted deadlines (--time-min before a reservation, 2026-10-08) ──


def _with_deadline(ep, hours_left):
    ep.metadata["slurm_deadline"] = (NOW + timedelta(hours=hours_left)).isoformat()
    return ep


def test_granted_deadline_wins_over_created_at_plus_walltime():
    """nakao-1009: a job started at 01:17 with --time-min ends at 13:00, not 00:17
    the next day. One hour left → the successor is due now."""
    young = _ep("j1", EndpointStatus.READY, age_h=1, ready_ago_min=40)
    assert _plan(young) == []
    assert _plan(_with_deadline(young, 1)) == [("provision", None)]


def test_unreadable_deadline_falls_back_to_walltime():
    ep = _ep("j1", EndpointStatus.READY, age_h=1, ready_ago_min=40)
    ep.metadata["slurm_deadline"] = "soon"
    assert _plan(ep) == []


def test_needs_deadline_only_for_ready_owned_rows_without_one():
    from umrs.orchestrator.endpoint_relay import needs_deadline

    ready = _ep("j1", EndpointStatus.READY, age_h=1, ready_ago_min=40)
    assert needs_deadline(POLICY, ready)
    assert not needs_deadline(POLICY, _with_deadline(_ep("j2", EndpointStatus.READY, age_h=1), 5))
    assert not needs_deadline(POLICY, _ep("j3", EndpointStatus.PROVISIONING, age_h=0.1))
    assert not needs_deadline(POLICY, _ep("j4", EndpointStatus.READY, age_h=1, project="other"))


def test_runner_reads_the_deadline_once_and_relays_by_it(rccs_roster, monkeypatch):
    import asyncio

    from umrs.api import endpoints as api
    from umrs.server import endpoint_relay as runner

    asked, updates, calls = [], [], []
    ep = _ep("401", EndpointStatus.READY, age_h=1, ready_ago_min=40)

    class _Backend:
        async def time_left_sec(self, job_id):
            asked.append(job_id)
            return 3600

    class _Registry:
        def list_all(self):
            return [ep]

        def update_metadata(self, job_id, metadata):
            updates.append((job_id, metadata))

    async def fake_provision_endpoint(**kwargs):
        calls.append(kwargs)
        return api.ProvisionResponse(status="provisioning", job_id="402")

    monkeypatch.setattr(api, "provision_endpoint", fake_provision_endpoint)
    monkeypatch.setattr(runner, "get_registry", lambda: _Registry())
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: WALLTIME)
    monkeypatch.setattr(runner, "_relay_backend", lambda: _Backend())

    actions = asyncio.run(runner.run_relay_once(now=NOW))
    assert [a.action for a in actions] == ["provision"]
    assert asked == ["401"]
    assert updates == [("401", {"slurm_deadline": (NOW + timedelta(hours=1)).isoformat()})]

    asyncio.run(runner.run_relay_once(now=NOW))
    assert asked == ["401"]  # read once: the deadline of a running job is fixed


def test_runner_without_a_deadline_answer_keeps_the_walltime_estimate(rccs_roster, monkeypatch):
    import asyncio

    from umrs.server import endpoint_relay as runner

    class _Backend:
        async def time_left_sec(self, _job_id):
            raise OSError("ssh down")

    class _Registry:
        def list_all(self):
            return [_ep("401", EndpointStatus.READY, age_h=1, ready_ago_min=40)]

    monkeypatch.setattr(runner, "get_registry", lambda: _Registry())
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: WALLTIME)
    monkeypatch.setattr(runner, "_relay_backend", lambda: _Backend())
    assert asyncio.run(runner.run_relay_once(now=NOW)) == []


def _remote_backend(stdout, ok=True, seen=None):
    from types import SimpleNamespace

    from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)

    class _SSH:
        async def run_remote(self, command):
            if seen is not None:
                seen.append(command)
            return SimpleNamespace(ok=ok, stdout=stdout)

    async def get_ssh():
        return _SSH()

    backend._get_ssh = get_ssh
    return backend


@pytest.mark.parametrize(
    ("stdout", "ok", "expected"),
    [
        ("11:42:58\n", True, 11 * 3600 + 42 * 60 + 58),
        ("1-02:00:00", True, 26 * 3600),
        ("UNLIMITED", True, None),
        ("", True, None),  # not running (pending or gone)
        ("11:00:00", False, None),
    ],
)
def test_remote_backend_time_left(stdout, ok, expected):
    import asyncio

    seen = []
    backend = _remote_backend(stdout, ok, seen)
    assert asyncio.run(backend.time_left_sec("395814")) == expected
    assert seen == ['squeue -j 395814 -h -t RUNNING -o "%L"']


def test_remote_backend_time_left_refuses_a_non_numeric_job_id():
    import asyncio

    seen = []
    backend = _remote_backend("1:00:00", True, seen)
    assert asyncio.run(backend.time_left_sec("1; rm -rf /")) is None
    assert seen == []


def test_rccs_front_submits_with_a_time_min_below_the_limit():
    import yaml
    from umrs.orchestrator.site_config import SchedulerConfig, _dict_to_dataclass

    sched = _dict_to_dataclass(SchedulerConfig, yaml.safe_load(SITE.read_text())["scheduler"])
    assert sched.time_min
    assert 0 < parse_slurm_duration(sched.time_min) < parse_slurm_duration(sched.time_limit)


@pytest.mark.parametrize(("time_min", "present"), [("03:00:00", True), ("", False)])
def test_wrapper_sbatch_carries_time_min_only_when_set(time_min, present):
    from types import SimpleNamespace

    from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend
    from umrs.orchestrator.site_config import SchedulerConfig

    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    sched = SchedulerConfig(time_limit="23:00:00", time_min=time_min, gpus=0, memory="")
    args = backend._build_wrapper_sbatch_args(
        site=SimpleNamespace(scheduler=sched, partition_profiles={}),
        output_dir="/tmp/x",
        gpu_count=None,
        node_count=None,
        partition_override="ng-dgx-s",
        slurm_account=None,
        nodelist_hint=None,
    )
    assert "--time=23:00:00" in args
    assert ("--time-min=03:00:00" in args) is present
