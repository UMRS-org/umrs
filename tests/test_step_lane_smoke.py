"""The step-lane smoke (receipt R2, rccs step-lane spec §11)."""

from __future__ import annotations

import glob
import json
import socket
import subprocess
import time
from types import SimpleNamespace

import pytest
from step_lane_harness import make_cluster
from umrs.server import step_lane_smoke as smoke
from umrs.server.step_executor import StepExecutionError
from umrs.server.step_lane_probe import SLEEP_MARKER, probe
from umrs.server.step_staging import register_bucket_registry

GOOD = {
    "uid": 65534,
    "rootfs_read_only": True,
    "env_keys": sorted({*smoke.STAGED_ENV, "PATH", "HOME", "HOSTNAME", "container"}),
    "connect": {"10.0.0.1:443": "OSError", "login1:9000": "gaierror"},
    "gpu_devices": [],
}


@pytest.fixture(autouse=True)
def _no_bucket_registry():
    register_bucket_registry(None)
    yield
    register_bucket_registry(None)


def test_a_hardened_run_passes():
    assert smoke.check(GOOD, gpu=False) == []
    assert smoke.check({**GOOD, "gpu_devices": ["/dev/nvidia0"]}, gpu=True) == []


@pytest.mark.parametrize(
    ("change", "gpu", "problem"),
    [
        ({"uid": 0}, False, "uid 0"),
        ({"rootfs_read_only": False}, False, "writable"),
        ({"env_keys": [*GOOD["env_keys"], "LITELLM_MASTER_KEY"]}, False, "LITELLM_MASTER_KEY"),
        ({"env_keys": ["PATH"]}, False, "staged names missing"),
        ({"connect": {"10.0.0.1:443": "open"}}, False, "connections opened: 10.0.0.1:443"),
        ({}, True, "no /dev/nvidia*"),
    ],
)
def test_each_escape_is_named(change, gpu, problem):
    [found] = smoke.check({**GOOD, **change}, gpu=gpu)
    assert problem in found


def test_the_sites_process_bound_must_hold():
    assert smoke.check({**GOOD, "nproc_soft": 4096}, gpu=False, nproc_limit=4096) == []
    [found] = smoke.check({**GOOD, "nproc_soft": -1}, gpu=False, nproc_limit=4096)
    assert "RLIMIT_NPROC is -1, not the site's 4096" in found


def test_the_probe_reports_names_never_values(monkeypatch):
    monkeypatch.setenv("UMRS_PROBE_SECRET", "s3cr3t-value")
    facts = probe([])
    assert "UMRS_PROBE_SECRET" in facts["env_keys"]
    assert "s3cr3t-value" not in repr(facts)


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_the_smoke_runs_the_probe_through_the_lane(tmp_path, monkeypatch):
    # The fake cluster runs the real helper and worker; its job keeps only
    # PATH/HOME and the four staged names, as the hardened pod does.
    cluster = make_cluster(tmp_path, monkeypatch)
    target = ("127.0.0.1", _closed_port())
    reports = smoke.run_plan(
        [("probe", "ng-dgx-m0", False), ("probe", "ng-dgx-m1", True)],
        targets=[target],
        timeout=60,
        make_executor=lambda p: cluster.executor(partition=p),
    )
    plain, gpu = reports
    assert plain["ok"], plain
    assert plain["partition"] == "ng-dgx-m0" and plain["job"]
    assert plain["probe"]["connect"] == {f"127.0.0.1:{target[1]}": "ConnectionRefusedError"}
    if not glob.glob("/dev/nvidia*"):
        assert gpu["problems"] == ["the GPU run saw no /dev/nvidia*"]
    assert len(cluster.calls("sbatch")) == 2
    assert cluster.workdirs() == []


def test_a_failed_run_is_reported_and_the_others_go_on(tmp_path, monkeypatch):
    cluster = make_cluster(tmp_path, monkeypatch)

    def broken(partition):
        if partition == "bad":
            raise RuntimeError("no ssh target")
        return cluster.executor(partition=partition)

    bad, chain, good = smoke.run_plan(
        [("probe", "bad", False), ("chain", "bad", False), ("probe", "ng-dgx-m0", False)],
        targets=[],
        timeout=60,
        make_executor=broken,
    )
    assert not bad["ok"] and "no ssh target" in bad["problems"][0]
    assert not chain["ok"] and "no ssh target" in chain["problems"][0]
    assert good["ok"], good


def test_usage():
    assert smoke._plan("", "m0") == []
    assert smoke._plan("m0,m1", "m0") == [
        ("probe", "m0", False),
        ("probe", "m1", False),
        ("probe", "m0", True),
        ("chain", "m0", False),
        ("kill", "m1", False),
    ]
    assert smoke.main(["--partitions", ","]) == 2


def test_the_chain_hands_a_live_parent_to_b_and_its_tag_holds(tmp_path, monkeypatch):
    cluster = make_cluster(tmp_path, monkeypatch)
    [report] = smoke.run_plan(
        [("chain", "ng-dgx-m0", False)],
        targets=[],
        timeout=60,
        make_executor=lambda p: cluster.executor(partition=p),
    )
    assert report["ok"], report
    assert report["tag_stable"] is True and all(report["jobs"])
    assert len(cluster.calls("sbatch")) == 2 and cluster.workdirs() == []


class _StubLane:
    """What the kill check drives: the sleeping step runs until the run is
    cancelled; its log carries the marker (when *started*); it settles as
    *final*; the inspector job reports *report* from node gb10-m3."""

    workdir_root, partition, account = "/r", "ng-dgx-m3", "cloud-users"
    venue_script = "/c/umrs/src/umrs/mini-rivault-llm/venue.sh"

    def __init__(self, report, *, started=True, final="CANCELLED"):
        self.report, self.started, self.final = report, started, final
        self._settlers, self.scripts, self.removed = [], [], []

    def bind_run(self, _run_id, *, cancelled, on_job):
        self.cancelled, self.on_job = cancelled, on_job

    def execute(self, *, entrypoint, config, **_ctx):
        assert config["sleep_seconds"] > 0 and entrypoint == smoke.PROBE_ENTRYPOINT
        self.on_job("sleeper", {"job_id": "901", "workdir": "/r/wd-sleep", "state": "RUNNING"})
        while not self.cancelled():
            time.sleep(0.01)
        raise StepExecutionError("run", "step sleeper cancelled: the workflow run was cancelled")

    def _helper(self, verb, *_args, **_kw):
        assert verb == "tail-log"
        return SimpleNamespace(returncode=0, stdout=(SLEEP_MARKER if self.started else "").encode())

    def _job_state(self, job_id):
        return self.final if job_id == "901" else "COMPLETED"

    def _mkwd(self, _job):
        return "/r/wd-inspect"

    def _run_cmd(self, _cmd, **_kw):
        return SimpleNamespace(returncode=0, stdout="gb10-m3\n")

    def _put(self, _job, _name, data):
        self.scripts.append(data.decode())

    def _submit(self, job):
        job.job_id = "902"

    def _read_reply_at(self, _workdir, _job):
        return json.dumps(self.report).encode()

    def _rmwd(self, workdir):
        self.removed.append(workdir)
        return True


CLEAN = {"host": "gb10-m3", "containers": [], "processes": [], "workdir_left": False}


def _kill(lane, **kw):
    return smoke.run_kill_check(
        "ng-dgx-m3", timeout=60, make_executor=lambda _p: lane, poll=0.01, **kw
    )


def test_a_cancelled_step_that_leaves_nothing_passes_the_kill_check():
    lane = _StubLane(CLEAN)
    report = _kill(lane)
    assert report["ok"], report
    assert report["state"] == "CANCELLED" and report["inspect"]["node"] == "gb10-m3"
    [script] = lane.scripts
    assert "#SBATCH --nodelist=gb10-m3" in script and "ctr=umrs-step-901" in script
    assert "[[ -e /r/wd-sleep ]]" in script and "rivault_venue_prepare" in script
    assert lane.removed == ["/r/wd-inspect"]  # the inspector cleans up after itself


def test_the_inspector_script_is_valid_bash():
    lane = _StubLane(CLEAN)
    _kill(lane)
    proc = subprocess.run(["bash", "-n"], input=lane.scripts[0].encode(), capture_output=True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    ("report", "kw", "problem"),
    [
        ({**CLEAN, "containers": ["umrs-step-901|Up 3 minutes"]}, {}, "still running"),
        ({**CLEAN, "processes": ["4242 conmon -n umrs-step-901"]}, {}, "processes survived"),
        ({**CLEAN, "workdir_left": True}, {}, "the workdir survived"),
        (CLEAN, {"final": "COMPLETED"}, "settled as COMPLETED"),
    ],
)
def test_anything_that_outlives_the_cancel_fails_the_kill_check(report, kw, problem):
    result = _kill(_StubLane(report, **kw))
    assert not result["ok"] and any(problem in p for p in result["problems"]), result


def test_an_exited_record_is_a_leftover_not_a_survivor():
    # scancel's SIGKILL takes the podman client before --rm (peer review):
    # the record is reported, and the next step job on the node removes it.
    result = _kill(_StubLane({**CLEAN, "containers": ["umrs-step-901|Exited (137) 1 minute ago"]}))
    assert result["ok"], result
    assert result["inspect"]["leftover"] == ["umrs-step-901|Exited (137) 1 minute ago"]


def test_a_step_that_never_starts_fails_the_kill_check():
    result = _kill(_StubLane(CLEAN, started=False), start_timeout=0.2)
    assert "the step container never started" in result["problems"]
