"""Blank-partition moves for pending provisions (2026-11-04).

Maintainer: every ng-dgx-s / ng-dgx-m* partition is equivalent, so a service
model should start wherever one is blank. Job 456170 (2-node DeepSeek) waited
on ng-dgx-m0 for a 21:55 start while m2 and m3 sat idle: the submit-time
choice was never revisited. The janitor now offers each pending provision to
``RemoteSlurmBackend.move_if_blank``, which moves it with ``scontrol update``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from umrs.orchestrator import partition_choice as pc
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.orchestrator.janitor import Janitor, JanitorConfig
from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend
from umrs.orchestrator.service import ProvisionRequest, _placement_metadata
from umrs.orchestrator.site_config import SchedulerConfig

NOW = datetime(2026, 11, 4, 15, 0, 0)
PAIRS = "ng-dgx-m0,ng-dgx-m2,ng-dgx-m1"
SINGLE = "ng-dgx-s,ng-dgx-m1,ng-dgx-m2,ng-dgx-m0,ng-dgx-m3"
OK_LINE = "sbatch: Job 1 to start at {start} using 2 processors on nodes {part}-1 in partition {part}"


@pytest.fixture(autouse=True)
def _fresh_recent_picks():
    pc.recent_picks.clear()
    yield
    pc.recent_picks.clear()


def _at(minutes: float) -> datetime:
    return NOW + timedelta(minutes=minutes)


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_blank_target_takes_a_candidate_that_starts_now():
    starts = {"ng-dgx-m2": _at(0.5), "ng-dgx-m1": _at(400)}
    assert pc.blank_target("ng-dgx-m0", starts, NOW) == "ng-dgx-m2"


def test_blank_target_is_none_when_every_other_candidate_waits():
    starts = {"ng-dgx-m2": _at(30), "ng-dgx-m1": None}
    assert pc.blank_target("ng-dgx-m0", starts, NOW) is None


def test_blank_target_never_returns_the_current_partition():
    assert pc.blank_target("ng-dgx-m0", {"ng-dgx-m0": _at(0)}, NOW) is None


def test_blank_target_prefers_the_preferred_partition_among_blanks():
    starts = {"ng-dgx-m1": _at(0), "ng-dgx-s": _at(0.5)}
    assert pc.blank_target("ng-dgx-m0", starts, NOW, {"ng-dgx-s"}) == "ng-dgx-s"


def test_blank_target_spreads_by_recent_picks_then_order():
    pc.recent_picks.note("ng-dgx-m2")
    starts = {"ng-dgx-m2": _at(0), "ng-dgx-m1": _at(0)}
    assert pc.blank_target("ng-dgx-m0", starts, NOW) == "ng-dgx-m1"


@pytest.mark.parametrize(
    ("reason", "movable"),
    [
        ("Resources", True),
        ("Priority", True),
        ("QOSMaxNodePerUserLimit", True),
        ("JobHeldUser", False),
        ("Dependency", False),
        ("BeginTime", False),
    ],
)
def test_movable_reason(reason, movable):
    assert pc.movable_reason(reason) is movable


def test_parse_clock():
    assert pc.parse_clock("2026-11-04T15:00:00\n") == NOW
    assert pc.parse_clock("date: not found") is None


# ── backend: move_if_blank ───────────────────────────────────────────────────


def _site(preferred=""):
    sched = SchedulerConfig(
        partition=SINGLE,
        partition_choice="earliest_start",
        preferred_partitions=preferred,
        time_limit="23:00:00",
        gpus=0,
        memory="",
        account="cloud-users",
    )
    return SimpleNamespace(scheduler=sched, partition_profiles={}, extra={})


def _ok(stdout="", stderr=""):
    return SimpleNamespace(ok=True, returncode=0, stdout=stdout, stderr=stderr)


def _fail(stderr):
    return SimpleNamespace(ok=False, returncode=1, stdout="", stderr=stderr)


class _FakeSSH:
    """squeue, --test-only per partition, the cluster clock and scontrol."""

    def __init__(self, queued="PENDING|ng-dgx-m0|Resources", starts=None, update=None):
        self.queued = queued
        self.starts = starts or {}
        self.update = update or _ok()
        self.cmds: list[str] = []

    async def run_remote(self, cmd, **_kwargs):
        self.cmds.append(cmd)
        if cmd.startswith("squeue"):
            return _ok(stdout=f"{self.queued}\n" if self.queued else "")
        if "--test-only" in cmd:
            return self._probe(cmd)
        if cmd.startswith("date"):
            return _ok(stdout=f"{NOW.isoformat()}\n")
        return self.update

    def _probe(self, cmd):
        part = cmd.split("--partition=", 1)[1].split()[0]
        start = self.starts.get(part)
        if start is None:
            return _fail("sbatch: error: Requested node configuration is not available")
        return _ok(stderr=OK_LINE.format(start=start.isoformat(), part=part))

    @property
    def probes(self):
        return [c for c in self.cmds if "--test-only" in c]

    @property
    def updates(self):
        return [c for c in self.cmds if c.startswith("scontrol")]


def _backend(ssh):
    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    backend.config = SimpleNamespace(extra={"slurm_wrapper_script": "/deploy/serve-model.slurm"})
    backend._job_info_dir = lambda: "/jobs"  # type: ignore[method-assign]

    async def _get_ssh():
        return ssh

    backend._get_ssh = _get_ssh  # type: ignore[method-assign]
    return backend


def _move(ssh, placement, site=None, job_id="456170"):
    with patch("umrs.orchestrator.site_config.load_site_config", return_value=site or _site()):
        return asyncio.run(_backend(ssh).move_if_blank(job_id, placement))


TWO_NODE = {"partition_override": PAIRS, "node_count": 2, "gpu_count": 1}


def _probed(ssh):
    return sorted(p.split("--partition=", 1)[1].split()[0] for p in ssh.probes)


def test_a_two_node_job_moves_to_the_blank_pair():
    ssh = _FakeSSH(starts={"ng-dgx-m2": NOW, "ng-dgx-m1": _at(400)})

    assert _move(ssh, TWO_NODE) == "ng-dgx-m2"

    assert ssh.updates == ["scontrol update JobId=456170 Partition=ng-dgx-m2"]
    assert _probed(ssh) == ["ng-dgx-m1", "ng-dgx-m2"]  # never the current one, never m3
    assert pc.recent_picks.counts() == {"ng-dgx-m2": 1}


def test_the_probes_carry_the_job_shape_and_wrapper():
    ssh = _FakeSSH(starts={"ng-dgx-m2": NOW})
    _move(ssh, TWO_NODE)
    shapes = {("--nodes=2" in p, "/deploy/serve-model.slurm" in p) for p in ssh.probes}
    assert shapes == {(True, True)}


def test_no_blank_candidate_leaves_the_job_alone():
    ssh = _FakeSSH(starts={"ng-dgx-m2": _at(60), "ng-dgx-m1": _at(400)})
    assert _move(ssh, TWO_NODE) is None
    assert ssh.updates == []


def test_a_single_node_role_moves_to_a_blank_s_before_a_blank_m():
    ssh = _FakeSSH(
        queued="PENDING|ng-dgx-m1|Resources",
        starts={"ng-dgx-m2": NOW, "ng-dgx-s": NOW, "ng-dgx-m3": NOW},
    )
    placement = {"partition_override": SINGLE, "node_count": 1}
    assert _move(ssh, placement, _site(preferred="ng-dgx-s")) == "ng-dgx-s"


@pytest.mark.parametrize(
    "queued",
    [
        "RUNNING|ng-dgx-m0|None",
        "PENDING|ng-dgx-m0|JobHeldUser",
        "PENDING|ng-dgx-m3|Resources",  # moved by hand outside the list
        "",
    ],
)
def test_only_a_movable_pending_job_on_a_listed_partition_is_probed(queued):
    ssh = _FakeSSH(queued=queued, starts={"ng-dgx-m2": NOW})
    assert _move(ssh, TWO_NODE) is None
    assert ssh.probes == [] and ssh.updates == []


def test_a_single_candidate_or_a_non_slurm_id_is_never_queried():
    ssh = _FakeSSH(starts={"ng-dgx-m2": NOW})
    assert _move(ssh, {"partition_override": "ng-dgx-m0", "node_count": 2}) is None
    assert _move(ssh, TWO_NODE, job_id="rivault-abc") is None
    assert ssh.cmds == []


def test_a_refused_update_is_logged_and_returns_none(caplog):
    ssh = _FakeSSH(
        starts={"ng-dgx-m2": NOW},
        update=_fail("Job violates accounting/QOS policy"),
    )
    assert _move(ssh, TWO_NODE) is None
    assert "not moved from ng-dgx-m0 to ng-dgx-m2" in caplog.text
    assert pc.recent_picks.counts() == {}


# ── janitor: _move_queued ────────────────────────────────────────────────────


class _Mover:
    def __init__(self, answer="ng-dgx-m2"):
        self.answer = answer
        self.calls: list[tuple[str, dict]] = []

    async def move_if_blank(self, job_id, placement):
        self.calls.append((job_id, dict(placement)))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _config(**overrides) -> JanitorConfig:
    base = {
        "enable_ttl_cleanup": False,
        "enable_orphan_cleanup": False,
        "enable_stuck_cleanup": False,
        "enable_liveness_cleanup": False,
        "enable_engine_probe": False,
        "enable_ttl_auto_extend": False,
        "enable_failed_retention": False,
        "partition_move_after_sec": 120,
        "partition_move_cooldown_sec": 300,
    }
    base.update(overrides)
    return JanitorConfig(**base)


def _queued(registry, job_id, age_sec=600, **metadata):
    registry.register(
        project_id="default",
        model="m",
        job_id=job_id,
        host="pending",
        port=0,
        backend="slurm-remote",
        metadata={"partition_override": PAIRS, "node_count": 2, **metadata},
    )
    endpoint = registry.get_by_job_id(job_id)
    endpoint.created_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        seconds=age_sec
    )


def _janitor(registry, backend, **overrides):
    return Janitor(config=_config(**overrides), registry=registry, backend=backend)


@pytest.mark.asyncio
async def test_janitor_offers_each_queued_listed_row_once_per_cooldown():
    registry, mover = EndpointRegistry(), _Mover()
    _queued(registry, "1")
    janitor = _janitor(registry, mover)

    first = await janitor.run_once()
    second = await janitor.run_once()

    assert first.queued_moved == 1 and second.queued_moved == 0
    assert mover.calls == [("1", {"partition_override": PAIRS, "node_count": 2})]
    assert first.to_dict()["queued_moved"] == 1
    assert first.total_cleaned == 0  # a move is not a cleanup


@pytest.mark.asyncio
async def test_janitor_skips_young_single_candidate_and_pinned_rows():
    registry, mover = EndpointRegistry(), _Mover()
    _queued(registry, "young", age_sec=30)
    _queued(registry, "single", partition_override="ng-dgx-m0")
    _queued(registry, "pinned", nodelist_hint="ng-dgx-m0-08")
    registry.register(
        project_id="default", model="m", job_id="legacy", host="pending", port=0,
        backend="slurm-remote",
    )
    registry.get_by_job_id("legacy").created_at -= timedelta(hours=1)

    await _janitor(registry, mover).run_once()

    assert mover.calls == []


@pytest.mark.asyncio
async def test_janitor_skips_ready_rows_and_survives_a_backend_error():
    registry = EndpointRegistry()
    _queued(registry, "ready")
    registry.mark_ready("ready")
    _queued(registry, "boom")
    mover = _Mover(answer=RuntimeError("ssh down"))

    result = await _janitor(registry, mover).run_once()

    assert [call[0] for call in mover.calls] == ["boom"]
    assert result.queued_moved == 0


@pytest.mark.asyncio
async def test_janitor_without_a_mover_or_disabled_does_nothing():
    registry, mover = EndpointRegistry(), _Mover()
    _queued(registry, "1")

    assert (await _janitor(registry, object()).run_once()).queued_moved == 0
    off = await _janitor(registry, mover, enable_partition_move=False).run_once()

    assert off.queued_moved == 0 and mover.calls == []


# ── service: the row records where its job may run ───────────────────────────


def test_placement_metadata_records_the_request_shape():
    request = ProvisionRequest(
        project_id="default", model="m", partition_override=PAIRS, node_count=2, gpu_count=1
    )
    assert _placement_metadata(request) == {
        "partition_override": PAIRS,
        "node_count": 2,
        "gpu_count": 1,
    }
