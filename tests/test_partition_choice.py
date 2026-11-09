"""One Slurm partition per submission, chosen among the candidates.

Live rccs 2026-10-24: every service model landed on ng-dgx-m0 (the first
listed candidate) while ng-dgx-m1..m3 and half of ng-dgx-s idled, and a
Bonsai-8B provision failed with "Quota exceeded on partition 'ng-dgx-m0'
(limit: 2 x 72 node-hours per user)". RCCS also refuses a comma list
("Multiple partitions are not allowed"), so UMRS must pick ONE: the earliest
``sbatch --test-only`` start, the next one on a quota refusal.
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from umrs.orchestrator import partition_choice as pc
from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend
from umrs.orchestrator.site_config import SchedulerConfig

RCCS = "ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3,ng-dgx-s"
OK_LINE = (
    "sbatch: Job 426027 to start at {start} using 20 processors on nodes "
    "{part}-10 in partition {part}"
)
QUOTA = (
    "sbatch: error: Quota exceeded on partition '{part}' (limit: 2 x 72 node-hours per user)\n"
    "sbatch: error: Batch job submission failed: Unspecified error"
)


@pytest.fixture(autouse=True)
def _fresh_recent_picks():
    """The process-wide pick record must not leak between tests."""
    pc.recent_picks.clear()
    yield
    pc.recent_picks.clear()


def _site(partition=RCCS, choice="earliest_start", profiles=None):
    sched = SchedulerConfig(
        partition=partition,
        partition_choice=choice,
        time_limit="23:00:00",
        gpus=0,
        memory="",
        account="cloud-users",
    )
    return SimpleNamespace(scheduler=sched, partition_profiles=profiles or {}, extra={})


class _FakeSSH:
    """Answers --test-only per partition; real submits from a queue."""

    def __init__(self, probes=None, submits=None, name="service"):
        self.probes = probes or {}
        self.submits = list(submits or [SimpleNamespace(ok=True, stdout="777", stderr="")])
        self.cmds: list[str] = []
        self.timeouts: list[int | None] = []
        self.name = name
        self.in_flight = 0
        self.max_in_flight = 0

    async def run_remote(self, cmd, timeout=None):
        self.cmds.append(cmd)
        self.timeouts.append(timeout)
        if "--test-only" in cmd:
            return await self._probe(cmd)
        return self.submits.pop(0)

    async def _probe(self, cmd):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0)  # let every concurrent probe start first
        self.in_flight -= 1
        part = cmd.split("--partition=", 1)[1].split()[0]
        answer = self.probes.get(part, ("ok", "2026-10-24T23:00:00"))
        if answer == "timeout":
            raise asyncio.TimeoutError("probe timed out")
        kind, value = answer
        if kind == "ok":
            return SimpleNamespace(
                ok=True, returncode=0, stdout="", stderr=OK_LINE.format(start=value, part=part)
            )
        return SimpleNamespace(ok=False, returncode=1, stdout="", stderr=value.format(part=part))

    @property
    def real(self):
        return [c for c in self.cmds if "--test-only" not in c]

    @property
    def probed(self):
        return [c.split("--partition=", 1)[1].split()[0] for c in self.cmds if "--test-only" in c]


def _backend(ssh):
    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    backend._job_info_dir = lambda: "/jobs"  # type: ignore[method-assign]
    backend._control_dir = lambda: "/jobs/control"  # type: ignore[method-assign]

    async def _get_ssh():
        return ssh

    backend._get_ssh = _get_ssh  # type: ignore[method-assign]
    return backend


_ENV = {"MODEL_NAME": "Bonsai-8B", "VLLM_GPU_MEM_UTIL": "0.9"}


def _wrapper_job(backend, **kwargs):
    return backend._submit_wrapper_job(
        model_path="/models/Bonsai-8B",
        wrapper_script="/deploy/serve-model.slurm",
        requesting_user=kwargs.pop("requesting_user", None),
        env=kwargs.pop("env", dict(_ENV)),
        **kwargs,
    )


def _submit(backend, site, **kwargs):
    with patch("umrs.orchestrator.site_config.load_site_config", return_value=site):
        return asyncio.run(_wrapper_job(backend, **kwargs))


def _partition_of(cmd):
    parts = [t for t in cmd.split() if t.startswith("--partition=")]
    assert len(parts) == 1, cmd
    return parts[0].split("=", 1)[1]


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_parse_test_only_success_line():
    out = OK_LINE.format(start="2026-10-24T23:01:59", part="ng-dgx-m1")
    assert pc.parse_test_only(out) == (datetime(2026, 10, 24, 23, 1, 59), "ng-dgx-m1")


@pytest.mark.parametrize(
    "out",
    [
        QUOTA.format(part="ng-dgx-m0"),
        "",
        "garbage to start at soon",
        "to start at 2026-13-99T99:00:00",
    ],
)
def test_parse_test_only_rejects_refusals_and_garbage(out):
    assert pc.parse_test_only(out) is None


def test_quota_line_names_its_partition():
    assert pc.quota_exceeded_partition(QUOTA.format(part="ng-dgx-m0")) == "ng-dgx-m0"
    assert pc.quota_exceeded_partition("sbatch: error: invalid partition specified") is None


def test_rank_earliest_first_ties_keep_order_refused_dropped():
    t1, t2 = datetime(2026, 10, 24, 12), datetime(2026, 10, 24, 13)
    starts = {"a": t2, "b": None, "c": t1, "d": t1}
    assert pc.rank_by_start(["a", "b", "c", "d"], starts) == ["c", "d", "a"]


def test_rank_starts_within_the_tolerance_tie_and_keep_list_order():
    # Concurrent probes of two idle partitions answer a second apart.
    late, early = datetime(2026, 10, 24, 23, 2, 0), datetime(2026, 10, 24, 23, 1, 59)
    assert pc.rank_by_start(["ng-dgx-s", "ng-dgx-m1"], {"ng-dgx-s": late, "ng-dgx-m1": early}) == [
        "ng-dgx-s",
        "ng-dgx-m1",
    ]
    much_later = datetime(2026, 10, 24, 23, 3, 0)
    starts = {"ng-dgx-s": much_later, "ng-dgx-m1": early}
    assert pc.rank_by_start(["ng-dgx-s", "ng-dgx-m1"], starts) == ["ng-dgx-m1", "ng-dgx-s"]


def test_rank_ties_go_to_the_least_recently_picked():
    t = datetime(2026, 10, 24, 23)
    starts = dict.fromkeys(["a", "b", "c"], t)
    assert pc.rank_by_start(["a", "b", "c"], starts, {"a": 2, "b": 1}) == ["c", "b", "a"]


def test_recent_picks_count_and_expire():
    now = [100.0]
    picks = pc.RecentPicks(ttl_sec=120, clock=lambda: now[0])
    picks.note("a")
    picks.note("a")
    picks.note("b")
    assert picks.counts() == {"a": 2, "b": 1}
    now[0] = 221.0
    assert picks.counts() == {}


def test_redact_exports_hides_every_export_value():
    args = ["sbatch", "--partition=a", "--export=ALL,LITELLM_MASTER_KEY=sk-abc123", "/w"]
    assert pc.redact_exports(args) == "sbatch --partition=a --export=<redacted> /w"


def test_probe_args_swap_partition_and_drop_export_and_parsable():
    args = [
        "sbatch",
        "--parsable",
        "--partition=a,b",
        "--time=1:00:00",
        "--export=ALL,K_HEX=00",
        "/w.slurm",
    ]
    assert pc.probe_args(args, "b") == [
        "sbatch",
        "--test-only",
        "--partition=b",
        "--time=1:00:00",
        "/w.slurm",
    ]


def test_default_partition_keeps_the_list_only_when_the_site_chooses():
    assert pc.default_partition(_site(" a , b ,c")) == "a,b,c"
    assert pc.default_partition(_site("a,b,c", choice="first")) == "a"
    assert pc.default_partition(SimpleNamespace(scheduler=SimpleNamespace(partition="a,b"))) == "a"


# ── DeploymentSpec / validation ──────────────────────────────────────────────


def _spec(site, profile=None):
    from umrs.model_registry.models import ModelEntry, ModelScope
    from umrs.orchestrator.deployment_spec import DeploymentSpec

    site.runtime = SimpleNamespace(rivault_dir="/rv")
    entry = ModelEntry(
        model_id="m-1", name="Bonsai-8B", scope=ModelScope.SYSTEM, storage_ref="/models/Bonsai-8B"
    )
    return DeploymentSpec.from_registry(entry, profile=profile or {}, site_config=site)


def test_deployment_spec_keeps_every_candidate_on_a_choosing_site():
    assert _spec(_site()).partition == RCCS


def test_deployment_spec_keeps_the_first_candidate_elsewhere():
    assert _spec(_site(choice="first")).partition == "ng-dgx-m0"


def test_deployment_spec_caps_gpu_util_by_the_strictest_listed_partition():
    profiles = {"ng-dgx-m0": SimpleNamespace(gpu_mem_util_max=0.78)}
    spec = _spec(_site("ng-dgx-m1,ng-dgx-m0", profiles=profiles), {"env_vars": {}})
    assert spec.gpu_memory_utilization == 0.78


def test_validate_partition_accepts_a_list_of_known_partitions():
    from umrs.jobs.adapters import _validate_partition

    profiles = {"ng-dgx-s": SimpleNamespace()}
    _validate_partition(
        "ng-dgx-s,ng-dgx-m1,ng-dgx-m0", _site("ng-dgx-m0,ng-dgx-m1", profiles=profiles)
    )


def test_validate_partition_names_the_bad_element_of_a_list():
    from umrs.jobs.adapters import _validate_partition

    with pytest.raises(RuntimeError, match="'ng-dgx-zz'"):
        _validate_partition("ng-dgx-m0,ng-dgx-zz", _site())


def test_endpoint_request_accepts_a_listed_partition_list(monkeypatch):
    from fastapi import HTTPException
    from umrs.api import endpoints as e

    monkeypatch.setattr("umrs.orchestrator.site_config.load_site_config", lambda: _site())
    e._validate_partition_allowed("ng-dgx-s,ng-dgx-m1")
    with pytest.raises(HTTPException) as exc:
        e._validate_partition_allowed("ng-dgx-s,ng-dgx-zz")
    assert "'ng-dgx-zz'" in str(exc.value.detail)


def test_preflight_checks_every_listed_partition_profile():
    from umrs.orchestrator.preflight import PreflightValidator

    validator = PreflightValidator.__new__(PreflightValidator)
    validator.site = _site(profiles={"b": SimpleNamespace(gpus=1)})
    validator.spec = SimpleNamespace(partition="a,b", gpus_per_node=4)
    assert validator._validate_partition_profile() == [
        "Partition 'b' has 1 GPU/node, but spec requests 4"
    ]


# ── remote submit ────────────────────────────────────────────────────────────


def test_list_submits_once_to_the_earliest_start_skipping_quota():
    ssh = _FakeSSH(
        probes={
            "ng-dgx-m0": ("err", QUOTA),
            "ng-dgx-m1": ("ok", "2026-10-25T08:00:00"),
            "ng-dgx-m2": ("ok", "2026-10-24T23:01:59"),
            "ng-dgx-m3": ("ok", "2026-10-24T23:01:59"),
            "ng-dgx-s": ("err", "sbatch: error: Batch job submission failed"),
        }
    )
    job = _submit(_backend(ssh), _site(), partition_override=RCCS)
    assert job.job_id == "777"
    assert sorted(ssh.probed) == sorted(RCCS.split(","))
    assert ssh.timeouts == [30] * 5 + [60]  # bounded probes, then the real submit
    assert [_partition_of(c) for c in ssh.real] == ["ng-dgx-m2"]  # tie: list order


def test_probes_carry_no_export_the_real_submit_does():
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"))
    probes = [c for c in ssh.cmds if "--test-only" in c]
    assert len(probes) == 2
    assert not any("--export=" in c for c in probes)
    assert "--export=" in ssh.real[0]


def test_no_override_takes_the_site_list_not_its_first_element():
    ssh = _FakeSSH(probes={"ng-dgx-m0": ("err", QUOTA)})
    _submit(_backend(ssh), _site())
    assert [_partition_of(c) for c in ssh.real] == ["ng-dgx-m1"]


def test_no_candidate_passes_submits_in_configured_order():
    ssh = _FakeSSH(
        probes=dict.fromkeys(RCCS.split(","), ("err", QUOTA)),
        submits=[
            SimpleNamespace(ok=False, stdout="", stderr=QUOTA.format(part=part))
            for part in RCCS.split(",")
        ],
    )
    with pytest.raises(RuntimeError, match="Quota exceeded on partition 'ng-dgx-s'"):
        _submit(_backend(ssh), _site())
    assert [_partition_of(c) for c in ssh.real] == RCCS.split(",")


def test_every_probe_timing_out_still_walks_the_list_on_quota():
    # Slow slurmctld (or refused ssh sessions): no probe answers, the first
    # candidate is over quota, the second takes the job.
    refused = SimpleNamespace(ok=False, stdout="", stderr=QUOTA.format(part="ng-dgx-m0"))
    ok = SimpleNamespace(ok=True, stdout="888", stderr="")
    ssh = _FakeSSH(
        probes=dict.fromkeys(["ng-dgx-m0", "ng-dgx-m1", "ng-dgx-m2"], "timeout"),
        submits=[refused, ok],
    )
    job = _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1,ng-dgx-m2"))
    assert job.job_id == "888"
    assert [_partition_of(c) for c in ssh.real] == ["ng-dgx-m0", "ng-dgx-m1"]


def test_probes_run_at_most_three_at_a_time():
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site())
    assert len(ssh.probed) == 5
    assert ssh.max_in_flight == 3


def test_back_to_back_submits_spread_over_idle_partitions():
    # --test-only cannot see a job that is still pending, so every probe of
    # an idle partition says "now"; the recent picks break the tie.
    site = _site("ng-dgx-m0,ng-dgx-m1,ng-dgx-m2")
    chosen = []
    for _ in range(3):
        ssh = _FakeSSH()
        _submit(_backend(ssh), site)
        chosen.append(_partition_of(ssh.real[0]))
    assert chosen == ["ng-dgx-m0", "ng-dgx-m1", "ng-dgx-m2"]


def test_concurrent_submits_spread_over_idle_partitions():
    site = _site("ng-dgx-m0,ng-dgx-m1")
    sshes = [_FakeSSH(), _FakeSSH()]

    async def _both():
        return await asyncio.gather(*(_wrapper_job(_backend(ssh)) for ssh in sshes))

    with patch("umrs.orchestrator.site_config.load_site_config", return_value=site):
        asyncio.run(_both())
    assert sorted(_partition_of(ssh.real[0]) for ssh in sshes) == ["ng-dgx-m0", "ng-dgx-m1"]


def test_submit_log_hides_export_values(caplog):
    refused = SimpleNamespace(ok=False, stdout="", stderr=QUOTA.format(part="ng-dgx-m0"))
    ok = SimpleNamespace(ok=True, stdout="888", stderr="")
    ssh = _FakeSSH(submits=[refused, ok])
    env = {**_ENV, "LITELLM_MASTER_KEY": "sk-abc123"}
    with caplog.at_level("INFO", logger="umrs.orchestrator.remote_slurm_backend"):
        _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"), env=env)
    assert all("sk-abc123" in cmd for cmd in ssh.real)  # the job still gets it
    assert caplog.text.count("Submitting wrapper job") == 2
    assert "sk-abc123" not in caplog.text


def test_probe_timeout_counts_as_a_failed_candidate():
    ssh = _FakeSSH(probes={"ng-dgx-m0": "timeout"})
    _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"))
    assert [_partition_of(c) for c in ssh.real] == ["ng-dgx-m1"]


def test_real_quota_refusal_moves_to_the_next_ranked_candidate():
    refused = SimpleNamespace(ok=False, stdout="", stderr=QUOTA.format(part="ng-dgx-m0"))
    ok = SimpleNamespace(ok=True, stdout="888", stderr="")
    ssh = _FakeSSH(submits=[refused, ok])
    job = _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"))
    assert job.job_id == "888"
    assert [_partition_of(c) for c in ssh.real] == ["ng-dgx-m0", "ng-dgx-m1"]


def test_other_real_failures_raise_without_retry():
    bad = SimpleNamespace(ok=False, stdout="", stderr="sbatch: error: invalid account")
    ssh = _FakeSSH(submits=[bad])
    with pytest.raises(RuntimeError, match="invalid account"):
        _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"))
    assert len(ssh.real) == 1


def test_single_partition_is_one_sbatch_without_a_probe():
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site(), partition_override="ng-dgx-m3")
    assert len(ssh.cmds) == 1
    assert _partition_of(ssh.cmds[0]) == "ng-dgx-m3"


def test_a_site_without_the_opt_in_submits_its_first_candidate_unprobed():
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site("mi300x,h200", choice="first"))
    assert len(ssh.cmds) == 1
    assert _partition_of(ssh.cmds[0]) == "mi300x"


def test_ai4s_sizing_routing_is_unchanged():
    profiles = {"1n4gpu": SimpleNamespace(gpus=4), "1n1gpu": SimpleNamespace(gpus=1)}
    site = _site("1n1gpu,1n4gpu", profiles=profiles)
    assert RemoteSlurmBackend._select_partition(site, 4, 1) == "1n4gpu"
    ssh = _FakeSSH()
    _submit(_backend(ssh), site, gpu_count=4, node_count=1)
    assert len(ssh.cmds) == 1
    assert _partition_of(ssh.cmds[0]) == "1n4gpu"


# ── ADR-112 wave A: one platform login, the user is attribution only ──────


def test_a_requesting_user_never_selects_the_ssh_identity():
    service = _FakeSSH(probes={"ng-dgx-m0": ("err", QUOTA)}, name="service")
    _submit(_backend(service), _site("ng-dgx-m0,ng-dgx-m1"), requesting_user="alice")
    assert sorted(service.probed) == ["ng-dgx-m0", "ng-dgx-m1"]
    assert [_partition_of(c) for c in service.real] == ["ng-dgx-m1"]
    # Attribution only: the comment and UMRS_USER, never the login.
    assert "--comment=user=alice" in service.real[0]
    assert "UMRS_USER=alice" in service.real[0]


def test_no_per_user_ssh_path_exists():
    # The per-user connection and its silent fallback are deleted (I-1).
    assert not hasattr(RemoteSlurmBackend, "_get_ssh_for_user")
    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    assert not hasattr(backend, "_user_ssh_pool")
    assert not hasattr(backend, "_delegation_mode")


def test_a_failed_connect_raises_and_never_reroutes():
    from umrs.orchestrator.ssh_control import SSHControlConfig

    # A site file still saying per_user_ssh (deprecated, ignored since ADR-112).
    legacy = _site("ng-dgx-m0")
    legacy.scheduler.slurm_delegation = "per_user_ssh"
    legacy.scheduler.service_pubkey_path = "/keys/umrs-service.pub"
    with patch("umrs.orchestrator.site_config.load_site_config", return_value=legacy):
        backend = RemoteSlurmBackend(
            SimpleNamespace(extra={}, model_path=""),
            ssh_config=SSHControlConfig(ssh_target="cluster", ssh_user="operator"),
        )
    attempts: list[str] = []

    class _Refused:
        def __init__(self, config):
            self.config = config

        async def ensure_connected(self):
            attempts.append(self.config.ssh_user)
            raise ConnectionError("Permission denied (publickey)")

    with (
        patch("umrs.orchestrator.remote_slurm_backend.SSHControlManager", _Refused),
        pytest.raises(ConnectionError, match="publickey"),
    ):
        _submit(backend, legacy, requesting_user="alice")
    # One connect, as the platform login; no second identity was tried.
    assert attempts == ["operator"]


def _declared_site(login="operator"):
    from umrs.orchestrator.site_config import HpcAccessConfig, ServiceAccountAccess

    site = _site("ng-dgx-m0,ng-dgx-m1")
    site.name = "cluster-a"
    site.extra = {"ssh_user": login}
    site.scheduler.hpc_access = HpcAccessConfig(
        modes_allowed=["service_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(login=login, personal=True),
    )
    return site


def test_wrapper_submit_records_the_platform_identity():
    ssh = _FakeSSH()
    job = _submit(_backend(ssh), _declared_site(), requesting_user="alice", slurm_account="grp-a")
    assert job.metadata["hpc_identity"] == {
        "site": "cluster-a",
        "lane": "platform",
        "access_mode": "service_account",
        "site_login": "operator",
        "slurm_account": "grp-a",
        "personal": True,
        "declared": True,
    }


def test_the_recorded_login_is_the_one_the_connection_used():
    from umrs.orchestrator.ssh_control import SSHControlConfig

    backend = _backend(_FakeSSH())
    backend._ssh_config = SSHControlConfig(ssh_target="cluster", ssh_user="svc")
    job = _submit(backend, _declared_site(login="operator"), requesting_user="alice")
    identity = job.metadata["hpc_identity"]
    assert (identity["site_login"], identity["lane"]) == ("svc", "platform")
    # The site's master account when the caller names none.
    assert identity["slurm_account"] == "cloud-users"


def test_direct_submit_records_the_platform_identity():
    class _StdinSSH:
        async def run_remote_with_stdin(self, _cmd, _data, **_kw):
            return SimpleNamespace(ok=True, stdout="4242", stderr="")

    backend = _backend(_StdinSSH())
    backend.config = SimpleNamespace(model_path="/m")
    backend._get_wrapper_script = lambda **_kw: None  # type: ignore[method-assign]
    backend._build_sbatch_script = lambda _model, **_kw: "#!/bin/sh\n"  # type: ignore[method-assign]
    with patch("umrs.orchestrator.site_config.load_site_config", return_value=_declared_site()):
        job = asyncio.run(backend.submit_job("/m", requesting_user="alice"))
    assert job.job_id == "4242"
    assert job.metadata["hpc_identity"]["lane"] == "platform"
    assert job.metadata["hpc_identity"]["site_login"] == "operator"


def test_probe_and_submit_stay_in_the_proxy_charset():
    from umrs.orchestrator.remote_slurm_backend import _PROXY_PASSTHRU

    allowed = set(_PROXY_PASSTHRU) | set(" =,")
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1"))
    for cmd in ssh.cmds:
        assert set(cmd) <= allowed, cmd


def test_gpu_util_cap_follows_the_candidates_the_submit_may_use():
    profiles = {"ng-dgx-m1": SimpleNamespace(gpu_mem_util_max=0.78)}
    ssh = _FakeSSH()
    _submit(_backend(ssh), _site("ng-dgx-m0,ng-dgx-m1", profiles=profiles))
    export = next(t for t in ssh.real[0].split() if t.startswith("--export="))
    assert "VLLM_GPU_MEM_UTIL=0.78" in export


# ── legacy direct launcher (remote path) ─────────────────────────────────────


def _launcher_run(answers, seen):
    def _run(site, command, timeout=60):
        seen.append((command, timeout))
        part = command.split("--partition=", 1)[1].split()[0]
        answer = answers.get(part)
        if answer == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if answer is None:
            return SimpleNamespace(returncode=1, stdout="", stderr=QUOTA.format(part=part))
        line = OK_LINE.format(start=answer, part=part)
        return SimpleNamespace(returncode=0, stdout="", stderr=line)

    return _run


def test_launcher_narrows_a_list_to_the_earliest_start(monkeypatch):
    from umrs.server import rivault_launcher as rl

    seen: list = []
    answers = {
        "ng-dgx-m1": "2026-10-25T01:00:00",
        "ng-dgx-m2": "2026-10-24T23:00:00",
        "ng-dgx-m3": "timeout",
    }
    monkeypatch.setattr(rl, "_run_remote_slurm_command", _launcher_run(answers, seen))
    args = [
        "sbatch",
        "--partition=ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3",
        "--export=ALL,A=1",
        "/w",
    ]
    out = rl._choose_remote_partition(_site(), args)
    assert out == ["sbatch", "--partition=ng-dgx-m2", "--export=ALL,A=1", "/w"]
    assert all("--test-only" in cmd and t == 30 for cmd, t in seen)


def test_launcher_leaves_a_list_alone_without_the_opt_in(monkeypatch):
    from umrs.server import rivault_launcher as rl

    seen: list = []
    monkeypatch.setattr(rl, "_run_remote_slurm_command", _launcher_run({}, seen))
    args = ["sbatch", "--partition=spark,spark-double", "/w"]
    assert rl._choose_remote_partition(_site("spark,spark-double", choice="first"), args) == args
    assert seen == []
    assert rl._first_site_partition(_site("spark,spark-double", choice="first")) == "spark"
    assert rl._first_site_partition(_site()) == RCCS


def test_launcher_submit_log_hides_export_values(monkeypatch, caplog):
    from umrs.server import rivault_launcher as rl

    site = _site("ng-dgx-m0", choice="first")
    site.extra = {"slurm_wrapper_script": "/deploy/serve-model.slurm"}
    seen: list = []

    def _refuse(_site, command, timeout=60):
        seen.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="sbatch: error: nope")

    monkeypatch.setattr(rl, "_run_remote_slurm_command", _refuse)
    with caplog.at_level("INFO", logger=rl.logger.name), pytest.raises(RuntimeError):
        rl.get_remote_slurm_node(site, "Bonsai-8B", sbatch_env={"LITELLM_MASTER_KEY": "sk-abc"})
    assert "sk-abc" in seen[0]
    assert "[remote-slurm]" in caplog.text
    assert "sk-abc" not in caplog.text


def test_a_preferred_partition_wins_a_tie_over_recent_picks():
    # rccs (2026-11-03): ng-dgx-s wins whenever it starts as early as an m pair,
    # even if it was picked more often recently.
    t = datetime(2026, 11, 3, 18)
    starts = dict.fromkeys(["ng-dgx-m1", "ng-dgx-s", "ng-dgx-m2"], t)
    load = {"ng-dgx-s": 3}
    assert (
        pc.rank_by_start(["ng-dgx-m1", "ng-dgx-s", "ng-dgx-m2"], starts, load, {"ng-dgx-s"})[0]
        == "ng-dgx-s"
    )
    assert pc.rank_by_start(["ng-dgx-m1", "ng-dgx-s", "ng-dgx-m2"], starts, load)[0] == "ng-dgx-m1"


def test_a_preferred_partition_that_starts_later_falls_back():
    early, late = datetime(2026, 11, 3, 18), datetime(2026, 11, 3, 19)
    starts = {"ng-dgx-s": late, "ng-dgx-m1": early}
    assert pc.rank_by_start(["ng-dgx-s", "ng-dgx-m1"], starts, {}, {"ng-dgx-s"}) == [
        "ng-dgx-m1",
        "ng-dgx-s",
    ]


def test_preferred_partitions_come_from_the_site_scheduler():
    site = SimpleNamespace(scheduler=SimpleNamespace(preferred_partitions="ng-dgx-s, ng-dgx-x"))
    assert pc.preferred_partitions(site) == frozenset({"ng-dgx-s", "ng-dgx-x"})
    assert pc.preferred_partitions(SimpleNamespace(scheduler=SimpleNamespace())) == frozenset()


# ── per-partition GPU request (ADR-109 D-3) ──────────────────────────────────

GB10 = SimpleNamespace(gpus=0, gpu_request="none", gres="", exclusive=False)
PRO6000 = SimpleNamespace(gpus=2, gpu_request="gres", gres="gpu:PRO6000-96GB:2", exclusive=True)
LEGACY = SimpleNamespace(gpus=1)  # a profile that predates gpu_request


def _gres_of(cmd):
    return [t for t in cmd.split() if t.startswith("--gres=") or t == "--exclusive"]


def test_gres_request_follows_the_partition_style():
    assert pc.gres_request("none", "gpu:x:2", 4, "gpu:4") is None
    assert pc.gres_request("", "", 4, "gpu:4") == "gpu:4"  # legacy: the site rule
    assert pc.gres_request("", "", 0, None) is None
    assert pc.gres_request("gres", "gpu:PRO6000-96GB:2", 0, None) == "gpu:PRO6000-96GB:2"
    assert pc.gres_request("gres", "", 2, None) == "gpu:2"
    assert pc.gres_request("gres", "", 0, None) is None
    with pytest.raises(ValueError, match="gres"):
        pc.gres_request("gres", "gpu:x;id", 1, None)


def test_gpu_request_style_refuses_an_unknown_style():
    assert pc.gpu_request_style(None) == ""
    assert pc.gpu_request_style(LEGACY) == ""
    assert pc.gpu_request_style(PRO6000) == "gres"
    with pytest.raises(ValueError, match="gpus"):
        pc.gpu_request_style(SimpleNamespace(gpu_request="gpus"))


@pytest.mark.parametrize(
    ("gres", "count"),
    [("gpu:PRO6000-96GB:2", 2), ("gpu:4", 4), ("gpu:a100", 1), ("gpu", 1), ("", 0), ("mps:9", 0)],
)
def test_gres_gpu_count(gres, count):
    assert pc.gres_gpu_count(gres) == count


def test_partition_gpus_reads_the_gres_then_gpus():
    assert pc.partition_gpus(PRO6000) == 2
    assert pc.partition_gpus(GB10) == 0
    assert pc.partition_gpus(LEGACY) == 1
    assert pc.partition_gpus(None) == 0


def test_with_gpu_request_leaves_a_legacy_profile_untouched():
    args = ["sbatch", "--partition=x", "--gres=gpu:1", "s.slurm"]
    assert pc.with_gpu_request(args, LEGACY, 1) == args
    assert pc.with_gpu_request(args, None, 1) == args


def test_with_gpu_request_drops_or_sets_the_gres_line():
    args = ["sbatch", "--partition=ng-dgx-m3", "--time=1:00:00", "--gres=gpu:1", "s.slurm"]
    assert pc.with_gpu_request(args, GB10, 1) == [
        "sbatch", "--partition=ng-dgx-m3", "--time=1:00:00", "s.slurm",
    ]  # fmt: skip
    plain = ["sbatch", "--partition=ai-pro6000", "--time=1:00:00", "s.slurm"]
    assert pc.with_gpu_request(plain, PRO6000, 0) == [
        "sbatch", "--partition=ai-pro6000", "--gres=gpu:PRO6000-96GB:2", "--exclusive",
        "--time=1:00:00", "s.slurm",
    ]  # fmt: skip


def test_gb10_serving_args_are_unchanged_by_their_profiles():
    # Live today (rccs-front: scheduler gpus 0, every candidate a GB10): the
    # GB10 profiles' gpu_request none changes no byte of probe or submit.
    gb10 = dict.fromkeys(("ng-dgx-m0", "ng-dgx-m1", "ng-dgx-m2", "ng-dgx-m3", "ng-dgx-s"), GB10)
    before, after = _FakeSSH(), _FakeSSH()
    _submit(_backend(before), _site())
    pc.recent_picks.clear()
    _submit(_backend(after), _site(profiles=gb10))
    assert after.cmds == before.cmds
    assert not any(_gres_of(c) for c in after.cmds)


def test_ai_pro6000_gets_its_gres_on_probe_and_submit_even_with_site_gpus_zero():
    ssh = _FakeSSH(probes={"ng-dgx-m3": ("ok", "2026-11-05T10:00:00")})
    site = _site("ng-dgx-m3,ai-pro6000", profiles={"ng-dgx-m3": GB10, "ai-pro6000": PRO6000})
    _submit(_backend(ssh), site)
    probes = {_partition_of(c): _gres_of(c) for c in ssh.cmds if "--test-only" in c}
    assert probes == {
        "ng-dgx-m3": [],
        "ai-pro6000": ["--gres=gpu:PRO6000-96GB:2", "--exclusive"],
    }
    [real] = ssh.real
    assert _partition_of(real) == "ai-pro6000"  # the earlier start; one partition only
    assert _gres_of(real) == ["--gres=gpu:PRO6000-96GB:2", "--exclusive"]


def test_a_gb10_submit_next_to_a_pro6000_candidate_carries_no_gres():
    ssh = _FakeSSH(probes={"ai-pro6000": ("ok", "2026-11-05T10:00:00")})
    site = _site("ng-dgx-m3,ai-pro6000", profiles={"ng-dgx-m3": GB10, "ai-pro6000": PRO6000})
    _submit(_backend(ssh), site)
    [real] = ssh.real
    assert _partition_of(real) == "ng-dgx-m3"
    assert _gres_of(real) == []


def test_a_legacy_site_keeps_its_site_wide_gres():
    ssh = _FakeSSH()
    site = _site("1n1gpu", profiles={"1n1gpu": LEGACY})
    site.scheduler.gpus = 1
    _submit(_backend(ssh), site)
    assert _gres_of(ssh.real[0]) == ["--gres=gpu:1"]


def test_the_endpoint_adapter_keeps_the_identity_on_the_attempt():
    # ADR-112 I-10: the scheduler persists attempt.metrics after the submit.
    from umrs.jobs.adapters import _record_hpc_identity

    attempt = SimpleNamespace(metrics={"other": 1})
    info = SimpleNamespace(metadata={"hpc_identity": {"lane": "platform", "site_login": "op"}})
    _record_hpc_identity(attempt, info)
    assert attempt.metrics == {
        "other": 1,
        "hpc_identity": {"lane": "platform", "site_login": "op"},
    }
    bare = SimpleNamespace(metrics={})
    _record_hpc_identity(bare, SimpleNamespace(metadata={}))
    assert bare.metrics == {}


def test_a_provision_submit_records_the_backend_identity_on_the_attempt():
    """Plan A.6 (review 2026-11-08): ``EndpointProvisionBackend.submit_job``
    itself copies the backend's ``hpc_identity`` into ``attempt.metrics``,
    which the scheduler then persists; only the helper was tested before."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from test_llm_model_limits import _durable_spec
    from umrs.jobs import JobAttempt, JobType
    from umrs.jobs.adapters import EndpointProvisionBackend
    from umrs.jobs.models import Job
    from umrs.jobs.scheduler import Scheduler
    from umrs.orchestrator.deployment_spec import DeploymentSpec
    from umrs.orchestrator.preflight import PreflightResult

    identity = {"lane": "platform", "access_mode": "service_account", "site_login": "op"}
    orchestrator = AsyncMock()
    orchestrator.backend_type = "slurm"
    orchestrator.submit_job = AsyncMock(
        return_value=MagicMock(
            job_id="slurm-1",
            status=MagicMock(value="pending"),
            endpoint=MagicMock(node="gpu-0"),
            metadata={"hpc_identity": identity},
        )
    )
    backend = EndpointProvisionBackend(
        orchestrator=orchestrator,
        registry=MagicMock(),
        port_manager=MagicMock(allocate=MagicMock(return_value=11434)),
    )
    spec = {"model": "glm-5.3", "model_path": "/models/glm-5.3", "ttl_sec": 0}
    job = Job(
        job_id="job-glm",
        project_id="proj-1",
        user_id="user-1",
        job_type=JobType.ENDPOINT_PROVISION,
        spec=spec,
        idempotency_key="k-glm",
    )
    attempt = JobAttempt(attempt_id="att-glm", job_id="job-glm", attempt_no=1, backend="slurm")
    persisted: dict = {}
    store = MagicMock(create_attempt=MagicMock(return_value="att-glm"))
    store.get_attempt.return_value = attempt
    store.update_attempt_metrics.side_effect = lambda _id, metrics: persisted.update(metrics)
    catalog_entry = MagicMock(storage_ref="/models/glm-5.3")
    catalog_entry.to_dict.return_value = {"name": "glm-5.3"}
    site = MagicMock()
    site.scheduler.partition = "test-gpu"
    site.partition_profiles = {}
    with (
        patch.object(backend, "_determine_target_host", return_value="gpu-0"),
        patch("umrs.jobs.adapters._resolve_model_entry", return_value=catalog_entry),
        patch(
            "umrs.jobs.adapters._check_active_certification",
            return_value={"cert_id": "cert-1", "smoke_test_passed": True},
        ),
        patch("umrs.orchestrator.site_config.load_site_config", return_value=site),
        patch.object(DeploymentSpec, "from_registry", return_value=_durable_spec()),
        patch(
            "umrs.orchestrator.preflight.PreflightValidator.run_all",
            new=AsyncMock(return_value=PreflightResult(passed=True, checks=[])),
        ),
    ):
        asyncio.run(Scheduler(store, lambda _t: backend)._schedule_job("job-glm", job))
    store.mark_attempt_submitted.assert_called_once()
    assert attempt.metrics["hpc_identity"] == identity
    assert persisted["hpc_identity"] == identity
