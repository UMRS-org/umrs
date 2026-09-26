"""The step-lane smoke (receipt R2, rccs step-lane spec §11)."""

from __future__ import annotations

import glob
import socket

import pytest
from step_lane_harness import make_cluster
from umrs.server import step_lane_smoke as smoke
from umrs.server.step_lane_probe import probe
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
