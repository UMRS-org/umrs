"""The step lane's reaper (build spec §8; review round 2, 2026-09-26).

A step's jobs were reconciled only when the same step of the same run was
submitted again. A run that ended or was refused first (the lane switched
off, a cancel or failed upstream step after a restart, a settle thread that
died with the process, a crash between ``mkwd`` and ``submit``) left
uploaded code running and its workdir on the quota-less Lustre. Through the
fake cluster of ``step_lane_harness``: the reaper ends those with the lane
disabled, never touches another run's job or one this process still chases,
records the step cleaned, and the hourly sweep removes idle unused workdirs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from step_lane_harness import HELPER, make_cluster, write_step_tree
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepResult
from umrs.server import remote_slurm_step_executor as rse
from umrs.server import step_reaper, step_trust
from umrs.server import workflow_runner as wr
from umrs.server.remote_slurm_step_executor import safe_segment
from umrs.server.step_reply import reply_to_json
from umrs.server.workflow_execution_registry import ExecutionRecord

RUN = "wf-0123456789ab"
JOB_NAME = f"umrs-step-{safe_segment(RUN, 'run')}-s1"
PREFIX = f"{safe_segment(RUN, 'run')}-s1-"


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    return make_cluster(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def _no_process_reaper():
    step_reaper.set_step_lane_reaper(None)
    yield
    step_reaper.set_step_lane_reaper(None)


def _off(cluster):
    """The lane switched off: not enabled, no image (the converge dropped it)."""
    return cluster.executor(enabled=False, image="", settle_seconds=5.0)


def _workdir(cluster, attempt=1) -> Path:
    proc = subprocess.run(
        [sys.executable, "-I", str(HELPER), "mkwd", str(cluster.root), f"{PREFIX}{attempt}-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return Path(proc.stdout.decode().strip())


def _facts(wd: Path, job_id: str = "900", state: str = "RUNNING") -> dict:
    return {"job_id": job_id, "workdir": str(wd), "job_name": JOB_NAME, "attempt": 1,
            "state": state, "cleaned": False}  # fmt: skip


def _record(status="failed", durable="job-1", submitted_at=None, **steps) -> ExecutionRecord:
    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice",
                             status=status, durable_job_id=durable)  # fmt: skip
    if submitted_at is not None:
        record.submitted_at = submitted_at
    for name, facts in steps.items():
        record.set_step(name, status="failed")
        step_reaper.record_step_job(record, name, facts)
    return record


# ── one step ────────────────────────────────────────────────────────────────


def test_a_live_job_of_an_ended_run_is_cancelled_with_the_lane_off(cluster):
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd)) is True
    assert [c[1:] for c in cluster.calls("scancel")] == [["900"]]
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"]
    assert cluster.workdirs() == []


def test_a_pending_job_is_cancelled_before_it_ever_starts(cluster):
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["PENDING"])
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd, state="SUBMITTED")) is True
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"] and cluster.workdirs() == []


def test_a_kept_completed_workdir_goes_once_its_run_ended(cluster):
    wd = _workdir(cluster)
    (wd / "reply" / "output.json").write_bytes(
        reply_to_json({"status": "ok", "result": StepResult(output_keys=["r"])})
    )
    cluster.add_job("900", JOB_NAME, str(wd), ["COMPLETED"])
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd, state="COMPLETED")) is True
    assert cluster.calls("scancel") == [] and cluster.workdirs() == []


def test_a_job_the_record_never_saw_is_found_by_name(cluster):
    # sbatch outcome unknown: the record holds the workdir, not the job id.
    wd = _workdir(cluster)
    cluster.add_job("901", JOB_NAME, str(wd), ["RUNNING"])
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd, job_id="", state="")) is True
    assert [c[1:] for c in cluster.calls("scancel")] == [["901"]]
    assert cluster.workdirs() == []


def test_an_earlier_attempts_job_of_the_step_is_reaped_too(cluster):
    # Each attempt's workdir is recorded right after mkwd (review round 3):
    # the reaper ends the jobs the record names (review round 4).
    older, latest = _workdir(cluster, 1), _workdir(cluster, 2)
    cluster.add_job("900", JOB_NAME, str(older), ["RUNNING"])
    cluster.add_job("902", JOB_NAME, str(latest), ["FAILED"])
    facts = {**_facts(latest, job_id="902", state="FAILED"), "attempt": 2}
    assert _off(cluster).reap_step(RUN, "s1", _facts(older, state=""), facts) is True
    assert [c[1:] for c in cluster.calls("scancel")] == [["900"]]
    assert cluster.workdirs() == []


def test_a_durable_retrys_job_of_the_same_step_is_never_reaped(cluster):
    # A durable retry re-runs the same execution id: its job has the same
    # name, workdir stem and comment. The ended dispatch's reaper touches
    # only the jobs its record names (review round 4).
    ended, retry = _workdir(cluster, 1), _workdir(cluster, 1)
    cluster.add_job("900", JOB_NAME, str(ended), ["FAILED"])
    cluster.add_job("910", JOB_NAME, str(retry), ["RUNNING"], comment=f"umrs-step:{RUN}:1")
    assert _off(cluster).reap_step(RUN, "s1", _facts(ended, state="FAILED")) is True
    assert cluster.calls("scancel") == []
    assert cluster.jobs()["910"]["states"] == ["RUNNING"] and cluster.workdirs() == [retry]


def test_another_runs_job_is_left_alone(cluster):
    wd = _workdir(cluster)
    cluster.add_job("905", JOB_NAME, str(wd), ["RUNNING"], comment="umrs-step:wf-0123456789ac:1")
    _off(cluster).reap_step(RUN, "s1", _facts(wd, job_id="905"))
    assert cluster.calls("scancel") == []
    assert wd.exists()


def test_a_job_this_process_still_settles_is_left_to_it(cluster):
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    with rse._SETTLING_LOCK:
        rse._SETTLING.add("900")
    try:
        assert _off(cluster).reap_step(RUN, "s1", _facts(wd)) is False
    finally:
        with rse._SETTLING_LOCK:
            rse._SETTLING.discard("900")
    assert cluster.calls("scancel") == [] and wd.exists()


def test_without_find_jobs_the_recorded_job_is_still_ended(cluster):
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    cluster.behave(find_rc=1)
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd)) is True
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"] and cluster.workdirs() == []


def test_without_find_jobs_or_a_job_id_the_workdir_waits_for_the_sweep(cluster):
    wd = _workdir(cluster)
    cluster.behave(find_rc=1)
    assert _off(cluster).reap_step(RUN, "s1", _facts(wd, job_id="", state="")) is False
    assert wd.exists()


# ── the run ─────────────────────────────────────────────────────────────────


def test_reaping_a_run_records_each_step_cleaned_and_skips_cleaned_ones(cluster):
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    step_reaper.set_step_lane_reaper(_off(cluster))
    done = {**_facts(wd, job_id="1", state="COMPLETED"), "cleaned": True}
    record = _record(durable=None, s1=_facts(wd), s2=done)  # no retry will come
    thread = step_reaper.reap_run_jobs(record)
    thread.join(timeout=60)
    assert step_reaper.step_jobs(record.steps["s1"])[str(wd)]["cleaned"] is True
    assert cluster.workdirs() == []
    assert all(c[1:] != ["1"] for c in cluster.calls("scancel"))
    assert step_reaper.reap_run_jobs(record) is None  # nothing left: no thread, no SSH


def test_a_step_run_to_completion_is_recorded_cleaned(cluster, tmp_path):
    from step_lane_harness import FakeS3
    from umrs.pipeline_sdk.step import StepContext
    from umrs.server.step_executor import EXECUTOR_SOURCE_DIR_METADATA_KEY
    from umrs.server.step_staging import STAGE_REQUEST_METADATA_KEY, register_bucket_registry

    register_bucket_registry(None)
    tree = write_step_tree(tmp_path / "bundle")
    ex, recorded = cluster.executor(), []
    ex.bind_run(RUN, on_job=lambda _step, job: recorded.append(dict(job)))
    ctx = StepContext(
        s3=FakeS3(), input_bucket="alice.proj.data", input_keys=[],
        output_bucket="alice.proj.data-output", run_id=RUN,
        metadata={EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
                  STAGE_REQUEST_METADATA_KEY: {"owner": "alice", "step": "s1"},
                  rse.STEP_TIMEOUT_METADATA_KEY: 60},
    )  # fmt: skip
    ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert recorded[-1]["cleaned"] is True and recorded[-1]["state"] == "COMPLETED"


def test_orphaned_runs_are_the_ended_and_the_undurable_ones_from_before_boot():
    facts = {"job_id": "9", "workdir": "/r/x", "cleaned": False}
    before = time.time() - 100
    ended = _record("failed", submitted_at=before, s1=facts)
    undurable = _record("running", durable=None, submitted_at=before, s1=facts)
    redispatched = _record("running", submitted_at=before, s1=facts)
    fresh = _record("failed", submitted_at=time.time() + 100, s1=facts)
    clean = _record("succeeded", submitted_at=before, s1={**facts, "cleaned": True})
    runs = [ended, undurable, redispatched, fresh, clean]
    assert step_reaper.orphaned_runs(runs, time.time()) == [ended, undurable]


def _age(path: Path, seconds: float) -> None:
    when = time.time() - seconds
    for target in (path, path / "log", path / "reply", path / "out"):
        os.utime(target, (when, when))


def test_the_boot_pass_reaps_orphans_then_the_sweep_removes_idle_workdirs(cluster):
    reaper = _off(cluster)
    orphan_wd, idle_wd, live_wd = _workdir(cluster, 1), _workdir(cluster, 2), _workdir(cluster, 3)
    cluster.add_job("900", JOB_NAME, str(orphan_wd), ["RUNNING"])
    cluster.add_job("903", "umrs-step-wf-other-s1", str(live_wd), ["RUNNING"])
    for wd in (idle_wd, live_wd):
        _age(wd, reaper.sweep_age() + 60)
    registry = SimpleNamespace(list=lambda: [_record(submitted_at=0.0, s1=_facts(orphan_wd))])
    stop = threading.Event()
    stop.set()  # one pass
    spent = lambda _job: "failed"
    step_reaper.reap_loop(reaper, registry, time.time(), stop=stop, durable_status=spent)
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"]
    assert cluster.workdirs() == [live_wd]  # a live job's workdir is never swept


def test_the_sweep_age_covers_queue_run_and_transfers(cluster):
    reaper = _off(cluster)
    assert reaper.sweep_age() > 6 * 3600 + 3600  # queue_timeout + time_limit
    assert reaper.sweep_age() >= 3600  # the helper's floor


def test_the_reaper_falls_back_to_site_yaml_when_the_env_lines_are_gone(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_HELPER", "")  # compose maps the dropped .env line to ""
    monkeypatch.setenv("UMRS_SLURM_WORKDIR_ROOT", "")
    site = SimpleNamespace(helper_path="/c/umrs/deploy/_common/umrs_step_helper.py",
                           workdir_root="/lvs0/x/umrs_steps")  # fmt: skip
    monkeypatch.setattr(step_reaper, "_site_step_executor", lambda: site)
    monkeypatch.setattr(rse, "_site_ssh_config", lambda: SSHControlConfig(ssh_target="login"))
    reaper = step_reaper.step_lane_reaper()
    assert reaper.helper_path == site.helper_path
    assert reaper.workdir_root == site.workdir_root
    assert reaper.enabled is False  # built for reaping, never for running a step


def test_no_reaper_on_a_server_without_a_remote_lane(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_HELPER", "")
    monkeypatch.setenv("UMRS_SLURM_WORKDIR_ROOT", "")
    monkeypatch.setattr(step_reaper, "_site_step_executor", lambda: SimpleNamespace())
    assert step_reaper.step_lane_reaper() is None
    assert step_reaper.start_step_lane_reaper(registry=SimpleNamespace(list=list)) is None


# ── the runner reaps when a run ends or is refused ──────────────────────────


class _Reaped(list):
    """The records reaped, and the ``for_retry`` each reap was given."""

    def __init__(self) -> None:
        super().__init__()
        self.for_retry: list = []

    def __call__(self, record, for_retry=None) -> None:
        self.append(record)
        self.for_retry.append(for_retry)


@pytest.fixture
def reaped(monkeypatch):
    calls = _Reaped()
    monkeypatch.setattr(step_reaper, "reap_run_jobs", calls)
    return calls


def _upload(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "lanemod.py").write_text("class UserStep: pass\n")
    return str(root)


def test_a_refused_redispatch_reaps_what_the_earlier_process_left(tmp_path, monkeypatch, reaped):
    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: False)
    monkeypatch.setattr(step_trust, "uploaded_code_policy", lambda: "refuse")
    record = _record("running", s1={"job_id": "900", "workdir": "/r/x", "cleaned": False})
    workflow = {"steps": [{"name": "s1", "entrypoint": "lanemod:UserStep"}]}
    wr.run_workflow_record(record, source_path=_upload(tmp_path / "wf"), workflow_json=workflow)
    assert record.status == "failed" and "connected HPC" in record.error
    assert reaped == [record]
    # A refusal is a kill switch: the earlier jobs end even for a durable run
    # a retry would re-dispatch (peer review F2).
    assert reaped.for_retry == [False]


class _Fails:
    name = "slurm"

    def execute(self, **_kw):
        from umrs.server.step_executor import StepExecutionError

        raise StepExecutionError("run", "boom")


@pytest.mark.parametrize("cancel", [False, True])
def test_a_run_that_ends_reaps_its_step_jobs(tmp_path, reaped, cancel):
    record = _record("running", s0={"job_id": "900", "workdir": "/r/x", "cleaned": False})
    record.cancel_requested = cancel
    workflow = {"steps": [{"name": "s1", "entrypoint": "umrs.pipeline_sdk.step:Step"}]}
    wr.run_workflow_record(record, source_path=None, workflow_json=workflow, executor=_Fails())
    assert record.is_terminal()
    assert reaped == [record]


def test_a_run_without_step_jobs_never_builds_a_reaper(tmp_path, monkeypatch):
    monkeypatch.setattr(step_reaper, "step_lane_reaper", lambda: pytest.fail("built a reaper"))
    record = _record("running")
    workflow = {"steps": [{"name": "s1", "entrypoint": "umrs.pipeline_sdk.step:Step"}]}
    wr.run_workflow_record(record, source_path=None, workflow_json=workflow, executor=_Fails())
    assert record.status == "failed"


# ── a durable retry replaces the record (review round 4) ───────────────────


def test_a_durable_retry_marks_the_record_it_replaces():
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry

    registry = WorkflowExecutionRegistry()
    first = registry.create("w", "alice", execution_id=RUN, durable_job_id="job-1")
    second = registry.create("w", "alice", execution_id=RUN, durable_job_id="job-1")
    assert first.superseded_by is second and second.superseded_by is None
    assert step_reaper.live_record(first) is second


def test_a_replaced_records_late_job_facts_persist_only_the_live_record():
    # The old run's settle thread reports its job ended after the retry
    # replaced the record: the stale snapshot must never be upserted over
    # the live row (review round 4).
    from umrs.workflow.pg_registry import DurableExecutionRecord

    persisted = []
    old, live = (DurableExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice",
                                        status=status) for status in ("failed", "running"))  # fmt: skip
    old._persist_hook = lambda _r: persisted.append("old")
    live._persist_hook = lambda _r: persisted.append("live")
    old.superseded_by = live
    wr._record_step_job(old, "s1", {**_facts(Path("/r/wd1")), "cleaned": True})
    assert persisted == ["live"] and "s1" not in old.steps
    assert step_reaper.step_jobs(live.steps["s1"])["/r/wd1"]["cleaned"] is True


def test_the_reaper_stops_walking_a_record_a_retry_replaced():
    old = _record(s0=_facts(Path("/r/wd0")), s1=_facts(Path("/r/wd1"), job_id="901"))
    live = _record("running")
    calls = []

    def reap_step(_run, step, *_facts, **_kw):
        calls.append(step)
        step_reaper.carry_step_jobs(old, live)  # the retry arrives meanwhile
        old.superseded_by = live
        return True

    step_reaper.reap_record(SimpleNamespace(reap_step=reap_step), old)
    assert calls == ["s0"]  # s1 is the live dispatch's now
    assert step_reaper.step_jobs(live.steps["s0"])["/r/wd0"]["cleaned"] is True
    assert step_reaper.unreaped_steps(live) == {"s1": [_facts(Path("/r/wd1"), job_id="901")]}
    assert step_reaper.unreaped_steps(old) == {"s0": [_facts(Path("/r/wd0"))],
                                               "s1": [_facts(Path("/r/wd1"), job_id="901")]}  # fmt: skip


# ── open runs nothing will dispatch again (review round 4) ──────────────────


@pytest.mark.parametrize(
    ("status", "orphaned"),
    [("failed", True), ("cancelled", True), (None, True), ("completed", True),
     ("queued", False), ("running", False), ("planned", False)],
)  # fmt: skip
def test_an_open_run_is_orphaned_unless_its_durable_job_runs_again(status, orphaned):
    record = _record("running", submitted_at=time.time() - 100, s1=_facts(Path("/r/x")))
    runs = step_reaper.orphaned_runs([record], time.time(), lambda _job: status)
    assert runs == ([record] if orphaned else [])


def test_an_unreadable_job_store_leaves_the_open_run_to_its_dispatch():
    def unreadable(_job):
        raise RuntimeError("db down")

    record = _record("running", submitted_at=time.time() - 100, s1=_facts(Path("/r/x")))
    assert step_reaper.orphaned_runs([record], time.time(), unreadable) == []


@pytest.mark.parametrize(("status", "ended"), [("failed", "failed"), ("cancelled", "cancelled")])
def test_the_boot_pass_reaps_and_ends_an_open_run_whose_job_will_not_run_again(
    cluster, status, ended
):
    # Its last durable attempt was lost in the crash (or it was cancelled
    # before the poll sent scancel): its job ran on to --time.
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    record = _record("running", submitted_at=0.0, s1=_facts(wd))
    stop = threading.Event()
    stop.set()
    step_reaper.reap_loop(_off(cluster), SimpleNamespace(list=lambda: [record]), time.time(),
                          stop=stop, durable_status=lambda _job: status)  # fmt: skip
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"] and cluster.workdirs() == []
    assert record.status == ended and "will not run again" in record.error
    assert step_reaper.unreaped_steps(record) == {}


def test_a_replaced_open_record_is_never_ended_over_the_live_one():
    record, live = _record("running"), _record("running")
    record.superseded_by = live
    step_reaper.close_abandoned_run(record, lambda _job: "failed")
    assert record.status == "running" and live.status == "running"


# ── one registry, one lane per server (per-site wiring review, 2026-09-27) ──


def test_a_reaper_takes_only_the_jobs_under_its_own_workdir_root():
    # Every umrs-front server reads the one shared execution registry; each
    # reaper reaches only its own cluster, where another cluster's job id is
    # an unrelated job.
    rccs, bdrt = "/lvs0/x/umrs_steps", "/home/m/umrs_steps"
    record = _record(submitted_at=0.0, s1=_facts(Path(f"{rccs}/wf-a")),
                     s2=_facts(Path(f"{bdrt}/wf-b"), job_id="901"))  # fmt: skip
    seen = []

    def reap_step(_run, step, *facts, **_kw):
        seen.append((step, [f["job_id"] for f in facts]))
        return True

    step_reaper.reap_record(SimpleNamespace(workdir_root=bdrt + "/", reap_step=reap_step),
                            record, for_retry=False)  # fmt: skip
    assert seen == [("s2", ["901"])]
    assert step_reaper.unreaped_steps(record) == {"s1": [_facts(Path(f"{rccs}/wf-a"))]}
    assert step_reaper.orphaned_runs([record], time.time(), root=rccs) == [record]
    assert step_reaper.orphaned_runs([record], time.time(), root=bdrt) == []
    assert step_reaper.unreaped_steps(record, "") == {}  # an empty root is no lane
    assert step_reaper.lane_root(SimpleNamespace()) is None  # a double naming none


def _open_run_with_a_live_job(cluster) -> tuple[Path, ExecutionRecord]:
    wd = _workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["RUNNING"])
    return wd, _record("running", submitted_at=0.0, s1=_facts(wd))


def _boot_pass(reaper, record) -> None:
    """One boot pass over *record*, whose durable job will not run again."""
    stop = threading.Event()
    stop.set()
    step_reaper.reap_loop(reaper, SimpleNamespace(list=lambda: [record]), time.time(),
                          stop=stop, durable_status=lambda _job: "failed")  # fmt: skip


def _foreign_lane(cluster, tmp_path):
    """Another cluster's reaper: its own root, the same account and job ids."""
    other = tmp_path / "other" / "umrs_steps"
    other.mkdir(parents=True)
    return cluster.executor(enabled=False, image="", workdir_root=str(other))


def test_another_lanes_boot_pass_neither_reaps_nor_ends_this_lanes_run(cluster, tmp_path):
    # The bdrt server restarts while an rccs run has a live job: its reaper
    # must not scancel the job nor close the run as its own orphan.
    wd, record = _open_run_with_a_live_job(cluster)
    _boot_pass(_foreign_lane(cluster, tmp_path), record)
    assert (cluster.calls("scancel"), cluster.jobs()["900"]["states"]) == ([], ["RUNNING"])
    assert (record.status, cluster.workdirs()) == ("running", [wd])
    assert step_reaper.unreaped_steps(record) == {"s1": [_facts(wd)]}


def test_the_runs_own_lane_still_reaps_and_ends_it_after_another_lanes_pass(cluster, tmp_path):
    wd, record = _open_run_with_a_live_job(cluster)
    _boot_pass(_foreign_lane(cluster, tmp_path), record)
    _boot_pass(_off(cluster), record)
    assert (cluster.jobs()["900"]["states"], cluster.workdirs()) == (["CANCELLED"], [])
    assert (record.status, step_reaper.unreaped_steps(record)) == ("failed", {})


def test_the_server_gives_the_boot_reaper_the_job_stores_status(monkeypatch):
    import server_umrs

    seen = {}
    monkeypatch.setattr(step_reaper, "start_step_lane_reaper", lambda **kw: seen.update(kw))
    runtime = SimpleNamespace(job_status=lambda _job: "failed")
    server_umrs._start_step_lane_reaper(1.0, runtime)
    assert seen["durable_status"] is runtime.job_status and seen["booted_at"] == 1.0
    server_umrs._start_step_lane_reaper(1.0, None)
    assert seen["durable_status"] is None


def test_the_runtime_reports_a_durable_jobs_status_and_none_when_missing():
    from umrs.jobs.models import JobStatus
    from umrs.jobs.store import JobNotFoundError
    from umrs.server.durable_job_runtime import DurableJobRuntime

    def get_job(job_id):
        if job_id != "job-1":
            raise JobNotFoundError(job_id)
        return SimpleNamespace(status=JobStatus.CANCELLED)

    store = SimpleNamespace(get_job=get_job)
    runtime = DurableJobRuntime(None, service=SimpleNamespace(
        job_service=SimpleNamespace(job_store=store)))  # fmt: skip
    assert runtime.job_status("job-1") == "cancelled"
    assert runtime.job_status("job-2") is None


# ── the job facts survive retries, results and re-dispatch (review round 3) ─


def test_a_retry_or_a_result_never_replaces_the_steps_job_facts():
    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice")
    wr._record_step_job(record, "s1", _facts(Path("/r/wd1"), state=rse.CANCEL_PENDING))
    record.set_step("s1", status="running", extra={"attempt": 2, "last_error": "x"})
    record.set_step("s1", status="succeeded", extra={"output_keys": [], "extra": {}})
    assert record.steps["s1"].extra["output_keys"] == []
    [left] = step_reaper.unreaped_steps(record)["s1"]
    assert left["state"] == rse.CANCEL_PENDING


def test_an_older_attempts_late_settle_never_overwrites_a_newer_job():
    # attempt 1's settle thread reports its job ended after attempt 2 submitted
    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice")
    newer = {**_facts(Path("/r/wd2"), job_id="902", state="RUNNING"), "attempt": 2}
    wr._record_step_job(record, "s1", newer)
    wr._record_step_job(record, "s1", {**_facts(Path("/r/wd1")), "cleaned": True})
    assert step_reaper.unreaped_steps(record) == {"s1": [newer]}


def test_job_facts_recorded_from_many_threads_are_all_kept():
    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice")
    threads = [
        threading.Thread(
            target=lambda i=i: wr._record_step_job(record, "s1", _facts(Path(f"/r/wd{i}")))
        )
        for i in range(32)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(step_reaper.step_jobs(record.steps["s1"])) == 32


def test_every_recorded_job_of_a_step_is_reaped_with_one_find_jobs(cluster):
    older, latest = _workdir(cluster, 1), _workdir(cluster, 2)
    cluster.add_job("900", JOB_NAME, str(older), ["RUNNING"])
    facts = [_facts(older, state=rse.CANCEL_PENDING), _facts(latest, job_id="", state="")]
    assert _off(cluster).reap_step(RUN, "s1", *facts) is True
    assert cluster.jobs()["900"]["states"] == ["CANCELLED"] and cluster.workdirs() == []
    assert cluster.helper_verbs().count("find-jobs") == 1


def test_a_retry_never_loses_the_first_attempts_job_to_the_reaper(cluster, tmp_path, monkeypatch):
    # Attempt 1's COMPLETED job keeps its workdir (collect-out lost its
    # transport); attempt 2 fails before it submits (find-jobs down in the
    # same outage); the run fails. The kept workdir must still be reaped.
    from umrs.server.step_staging import register_bucket_registry

    register_bucket_registry(None)
    tree = write_step_tree(tmp_path / "bundle")
    ex = cluster.executor()
    local, lost = ex._remote_argv, []

    def outage(command):
        if " collect-out " in command or (lost and " find-jobs " in command):
            lost.append(command)
            return ["bash", "-c", "exit 255"]
        return local(command)

    ex._remote_argv = outage
    step_reaper.set_step_lane_reaper(cluster.executor())
    threads, real = [], step_reaper.reap_run_jobs
    monkeypatch.setattr(
        step_reaper, "reap_run_jobs", lambda r, for_retry=None: threads.append(real(r, for_retry))
    )
    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice")
    step = {"name": "s1", "entrypoint": "steps.lane:Upper", "max_retries": 1, "retry_delay": 0}
    wr.run_workflow_record(record, source_path=str(tree), workflow_json={"steps": [step]},
                           executor=ex)  # fmt: skip
    assert record.status == "failed" and "find-jobs" in record.steps["s1"].error
    assert len(cluster.calls("sbatch")) == 1
    for thread in threads:
        thread.join(timeout=60)
    assert threads and cluster.workdirs() == []
    [facts] = step_reaper.step_jobs(record.steps["s1"]).values()
    assert facts["state"] == "COMPLETED" and facts["cleaned"] is True


def test_a_redispatched_run_keeps_the_jobs_its_earlier_process_left(tmp_path, monkeypatch, reaped):
    # The re-dispatch replaces the record; step s2's job (a step the refused
    # run never reaches) must stay on it for the reaper.
    import asyncio
    import json as _json

    import umrs.server.workflow_registry_handler
    from umrs.jobs.models import Job, JobAttempt, JobType
    from umrs.jobs.workflow_run import WorkflowRunBackend
    from umrs.server.workflow_execution_registry import (
        get_execution_registry,
        reset_execution_registry_for_tests,
    )

    monkeypatch.setattr(step_trust, "hpc_lane_available", lambda: False)
    monkeypatch.setattr(step_trust, "uploaded_code_policy", lambda: "refuse")
    reset_execution_registry_for_tests()
    registry = get_execution_registry()
    prior = registry.create("w", "alice", execution_id=RUN, durable_job_id="job-1")
    left = _facts(Path("/r/s2-wd"), job_id="905", state="PENDING")
    step_reaper.record_step_job(prior, "s2", left)
    step_reaper.record_step_job(prior, "s1", {**_facts(Path("/r/s1-wd")), "cleaned": True})
    source = _upload(tmp_path / "wf")
    steps = [{"name": "s1", "entrypoint": "lanemod:UserStep"},
             {"name": "s2", "entrypoint": "lanemod:UserStep", "depends_on": ["s1"]}]  # fmt: skip
    (tmp_path / "wf" / "workflow.json").write_text(_json.dumps({"name": "w", "steps": steps}))
    job = Job(job_id="job-1", project_id="p", user_id="alice", job_type=JobType.WORKFLOW_RUN,
              spec={"execution_id": RUN, "workflow_name": "w", "params": {},
                    "source_path": source}, idempotency_key="k")  # fmt: skip
    attempt = JobAttempt(attempt_id="a2", job_id="job-1", attempt_no=2, backend="workflow-run")
    asyncio.run(WorkflowRunBackend().submit_job(job, attempt))
    assert registry.join(RUN, timeout=30)
    record = registry.get(RUN)
    assert record is not prior and record.status == "failed"
    assert reaped == [record]
    assert step_reaper.unreaped_steps(record) == {"s2": [left]}


def test_a_durable_record_persists_each_job_and_reads_it_back():
    import json as _json

    from umrs.workflow.pg_registry import (
        DurableExecutionRecord,
        _record_from_payload,
        _record_json,
    )

    record = DurableExecutionRecord(execution_id=RUN, workflow_name="w", user_id="alice")
    persisted = []
    record._persist_hook = lambda rec: persisted.append(_record_json(rec))
    wr._record_step_job(record, "s1", _facts(Path("/r/wd1")))
    assert len(persisted) == 1
    back = _record_from_payload(_json.loads(persisted[-1]))
    assert step_reaper.unreaped_steps(back) == {"s1": [_facts(Path("/r/wd1"))]}


def test_a_completed_workdir_a_durable_retry_may_adopt_is_kept(cluster):
    """Round-3 review (2026-09-26): the run-end reap of a failed run deleted
    the COMPLETED workdir the durable retry would adopt."""
    wd = _workdir(cluster)
    cluster.add_job("909", JOB_NAME, str(wd), ["COMPLETED"])
    ex = _off(cluster)
    assert ex.reap_step(RUN, "s1", _facts(wd, state="COMPLETED"), for_retry=True) is False
    assert wd.exists() and cluster.calls("scancel") == []


def test_a_live_job_a_durable_retry_may_resume_is_left_running(cluster):
    """Peer review B1 (2026-09-26): the run-end reap of a failed run
    cancelled the live job its durable retry would resume."""
    wd = _workdir(cluster)
    cluster.add_job("910", JOB_NAME, str(wd), ["RUNNING"])
    ex = _off(cluster)
    assert ex.reap_step(RUN, "s1", _facts(wd, job_id="910"), for_retry=True) is False
    assert wd.exists() and cluster.calls("scancel") == []


def test_a_failed_job_is_reaped_even_for_a_retry(cluster):
    wd = _workdir(cluster)
    cluster.add_job("911", JOB_NAME, str(wd), ["FAILED"])
    ex = _off(cluster)
    assert ex.reap_step(RUN, "s1", _facts(wd, job_id="911"), for_retry=True) is True
    assert not wd.exists()


def test_a_cancel_between_durable_retries_ends_the_jobs_left_for_the_retry(cluster):
    """Peer review F2 (2026-09-26): the record had ended, so cancel returned
    early and the live job kept for the retry ran on to --time."""
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry

    wd = _workdir(cluster)
    cluster.add_job("912", JOB_NAME, str(wd), ["RUNNING"])
    step_reaper.set_step_lane_reaper(_off(cluster))
    registry = WorkflowExecutionRegistry()
    record = registry.create(workflow_name="w", user_id="alice", execution_id=RUN,
                             durable_job_id="job-1")  # fmt: skip
    record.update(status="failed", finished_at=time.time())
    step_reaper.record_step_job(record, "s1", _facts(wd, job_id="912"))
    threads, real = [], step_reaper.reap_run_jobs
    step_reaper.reap_run_jobs = lambda r, for_retry=None: threads.append(real(r, for_retry))
    try:
        assert registry.cancel(RUN) is record
    finally:
        step_reaper.reap_run_jobs = real
    for thread in threads:
        thread.join(timeout=60)
    assert threads and cluster.jobs()["912"]["states"] == ["CANCELLED"]
    assert not wd.exists()


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return bool(condition())


def _redispatched(registry):
    """An ended run that left a live job, and the durable re-dispatch that
    replaced its record and carried the job over."""
    prior = registry.create("w", "alice", execution_id=RUN, durable_job_id="job-1")
    prior.update(status="failed", finished_at=time.time())
    step_reaper.record_step_job(prior, "s1", _facts(Path("/r/wd")))
    record = registry.create("w", "alice", execution_id=RUN, durable_job_id="job-1")
    step_reaper.carry_step_jobs(prior, record)
    return prior, record


def test_a_cancel_in_the_redispatch_submitted_window_ends_the_carried_jobs(reaped):
    """Peer review F2 (2026-09-26): a cancel between the re-dispatch's record
    and its launch marked the run cancelled, and the launch returned without
    reaping the jobs it carried."""
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry

    registry = WorkflowExecutionRegistry()
    _prior, record = _redispatched(registry)
    assert registry.cancel(RUN) is record and record.status == "cancelled"
    ran = []
    registry.launch(record, ran.append)
    assert _wait_for(lambda: record in reaped), reaped
    assert ran == [] and reaped.for_retry[reaped.index(record)] is False


def test_a_cancel_racing_a_redispatch_cancels_the_record_it_put_in_place(reaped):
    """Peer review F2 (2026-09-26): the cancel read the old record, the
    re-dispatch (which found the durable job not yet cancelled) replaced it,
    and the new run resumed the live job although the user cancelled."""
    from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry

    registry = WorkflowExecutionRegistry()
    prior, record = _redispatched(registry)
    reads = iter([prior])  # the cancel's first read predates the replacement
    live_get = registry.get
    registry.get = lambda eid: next(reads, None) or live_get(eid)
    assert registry.cancel(RUN) is record
    assert (prior, False) == (reaped[0], reaped.for_retry[0])  # the ended run's jobs
    assert record.cancel_requested and record.status == "cancelled"  # its launch runs nothing


def test_the_run_end_reap_of_a_spent_durable_job_ends_its_jobs(monkeypatch):
    """Peer review (2026-09-26): the last retry's reap kept the job to --time."""
    from umrs.api import jobs as api_jobs

    ok = SimpleNamespace(status="failed", cancel_requested=False, durable_job_id="job-1")
    job = SimpleNamespace(status="queued", can_retry=lambda n: n < 3)  # a retry pending
    store = SimpleNamespace(get_job=lambda _id: job, get_attempts_for_job=lambda _id: [1, 2])
    monkeypatch.setattr(api_jobs, "get_optional_job_store", lambda: store)
    assert step_reaper._may_retry(ok)  # attempt 2 of 3: a retry may come
    store.get_attempts_for_job = lambda _id: [1, 2, 3]
    assert not step_reaper._may_retry(ok)  # spent
    store.get_attempts_for_job = lambda _id: [1]
    job.status = "cancelled"
    assert not step_reaper._may_retry(ok)
    store.get_job = lambda _id: None
    assert not step_reaper._may_retry(ok)
    job.status = "failed"  # job-level: every retry spent (budget exhausted too)
    store.get_job = lambda _id: job
    assert not step_reaper._may_retry(ok)

    def missing(_id):
        from umrs.jobs.store import JobNotFoundError

        raise JobNotFoundError(_id)  # what JobStore.get_job does in production

    store.get_job = missing
    assert not step_reaper._may_retry(ok)

    def unreadable(_id):
        raise RuntimeError("db down")

    store.get_job = unreadable
    assert step_reaper._may_retry(ok)  # unreadable: leave the jobs to --time and the sweep


def test_without_a_sidecar_store_the_durable_runtimes_store_decides(monkeypatch):
    """Peer review (2026-09-26): with no in-process sidecar store every
    attempt kept its jobs, though the durable runtime's store knew the job
    was spent."""
    from umrs.api import jobs as api_jobs
    from umrs.server import durable_job_runtime

    ok = SimpleNamespace(status="failed", cancel_requested=False, durable_job_id="job-1")
    job = SimpleNamespace(status="cancelled", can_retry=lambda n: n < 3)
    store = SimpleNamespace(get_job=lambda _id: job, get_attempts_for_job=lambda _id: [1])
    runtime = durable_job_runtime.DurableJobRuntime(None, service=SimpleNamespace(
        job_service=SimpleNamespace(job_store=store)))  # fmt: skip
    monkeypatch.setattr(api_jobs, "get_optional_job_store", lambda: None)
    monkeypatch.setattr(durable_job_runtime, "_global_runtime", runtime)
    assert durable_job_runtime.get_runtime_job_store() is store
    assert not step_reaper._may_retry(ok)
    job.status = "queued"
    assert step_reaper._may_retry(ok)
    monkeypatch.setattr(durable_job_runtime, "_global_runtime", None)
    assert step_reaper._may_retry(ok)  # neither store: leave the jobs


def test_only_a_failed_uncancelled_durable_run_may_retry(monkeypatch):
    # No job store in this process (CI has one, with no job-1 in it: "spent").
    monkeypatch.setattr(step_reaper, "durable_job_store", lambda: None)
    ok = SimpleNamespace(status="failed", cancel_requested=False, durable_job_id="job-1")
    assert step_reaper._may_retry(ok)
    assert not step_reaper._may_retry(SimpleNamespace(**{**vars(ok), "cancel_requested": True}))
    assert not step_reaper._may_retry(SimpleNamespace(**{**vars(ok), "durable_job_id": None}))
    assert not step_reaper._may_retry(SimpleNamespace(**{**vars(ok), "status": "succeeded"}))
    # at boot, only while the scheduler still dispatches the durable job
    assert step_reaper._may_retry(ok, lambda _job: "queued")
    assert not step_reaper._may_retry(ok, lambda _job: "failed")
    assert not step_reaper._may_retry(ok, lambda _job: None)

    def unreadable(_job):
        raise RuntimeError("db down")

    assert step_reaper._may_retry(ok, unreadable)  # as _redispatched: leave them
