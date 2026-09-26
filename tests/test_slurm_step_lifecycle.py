"""Staged step lane lifecycle (build spec §8).

Through the fake cluster of ``step_lane_harness`` (the real helper, fake
Slurm CLIs): reconciliation by run-scoped job name (adopt a COMPLETED job
with its reply that ran on the current attempt's inputs, ``scancel`` a live
one, never two jobs, never another run's job, also behind a symlinked root), ``scancel`` on the queue clock, the
step's deadline (on the run clock), a workflow cancel and a newer attempt —
re-sent until it reaches Slurm — every terminal state, workdir removal only
after a confirmed terminal state (and never of the only copy of a COMPLETED
job's outputs), sbatch outcomes the server could not see, and the separate
poll clocks.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import time

import pytest
from step_lane_harness import HELPER, FakeS3, make_cluster, write_step_tree
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server import remote_slurm_step_executor as rse
from umrs.server import slurm_step_executor as sse
from umrs.server.remote_slurm_step_executor import (
    CANCEL_PENDING,
    STEP_TIMEOUT_METADATA_KEY,
    RemoteSlurmStepExecutor,
    _StepJob,
    _Unreachable,
    safe_segment,
)
from umrs.server.slurm_step_executor import (
    TERMINAL_FAILURES,
    UNKNOWN,
    UNREACHABLE,
    WALL_GRACE_SEC,
    PollClock,
    SlurmStepExecutor,
)
from umrs.server.step_executor import EXECUTOR_SOURCE_DIR_METADATA_KEY, StepExecutionError
from umrs.server.step_reply import reply_to_json
from umrs.server.step_staging import (
    DEFAULT_CAPS,
    STAGE_REQUEST_METADATA_KEY,
    build_stage_plan,
    inputs_tag,
    register_bucket_registry,
)
from umrs.server.workflow_runner import _run_executor_attempt

OWNER = "alice"
IN_BUCKET = "alice.proj.data"
OUT_BUCKET = "alice.proj.data-output"
RUN = "wf-0123456789ab"
JOB_NAME = f"umrs-step-{safe_segment(RUN, 'run')}-s1"
PREFIX = f"{safe_segment(RUN, 'run')}-s1-"


@pytest.fixture(autouse=True)
def _no_bucket_registry():
    register_bucket_registry(None)
    yield
    register_bucket_registry(None)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    return make_cluster(tmp_path, monkeypatch)


@pytest.fixture
def tree(tmp_path):
    return write_step_tree(tmp_path / "bundle")


def _ctx(tree, s3=None, **metadata) -> StepContext:
    meta = {
        EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
        STAGE_REQUEST_METADATA_KEY: {"owner": OWNER, "step": "s1"},
        **metadata,
    }
    return StepContext(
        s3=s3 if s3 is not None else FakeS3(),
        input_bucket=IN_BUCKET,
        input_keys=[],
        output_bucket=OUT_BUCKET,
        metadata=meta,
        run_id=RUN,
    )


def _run(cluster, tree, ex=None, **metadata):
    """One step; a 60 s step deadline turns a lifecycle regression that would
    poll forever into a failure."""
    ex = ex or cluster.executor()
    metadata.setdefault(STEP_TIMEOUT_METADATA_KEY, 60)
    return ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=_ctx(tree, **metadata))


def _tag(cluster, tree, s3=None, config=None) -> str:
    """The inputs tag the executor gives this step (its workdir name carries it)."""
    ex, ctx = cluster.executor(), _ctx(tree, s3)
    payload = ex._payload("steps.lane:Upper", config or {}, ctx)
    return inputs_tag(build_stage_plan(ctx, caps=ex.caps), payload, str(tree), ex.caps)


def _earlier_workdir(cluster, attempt=1, tag=""):
    prefix = f"{PREFIX}{attempt}-" + (f"{tag}-" if tag else "")
    proc = subprocess.run(
        [sys.executable, "-I", str(HELPER), "mkwd", str(cluster.root), prefix],
        capture_output=True,
        check=True,
    )
    from pathlib import Path

    return Path(proc.stdout.decode().strip())


def _join(ex):
    for thread in list(ex._settlers):
        thread.join(timeout=30)


def _completed_earlier(cluster, tag, job_id="900", states=("COMPLETED",), physical=False, **job):
    """An earlier attempt's job that finished with a reply and one output on
    the inputs *tag* names; *physical* records its workdir as Slurm does
    without ``--chdir``."""
    wd = _earlier_workdir(cluster, tag=tag)
    (wd / "reply" / "output.json").write_bytes(
        reply_to_json({"status": "ok", "result": StepResult(output_keys=["r.txt"])})
    )
    (wd / "out" / OUT_BUCKET).mkdir()
    (wd / "out" / OUT_BUCKET / "r.txt").write_bytes(b"earlier")
    recorded = os.path.realpath(wd) if physical else str(wd)
    cluster.add_job(job_id, JOB_NAME, recorded, list(states), **job)
    return wd


# ── reconciliation ──────────────────────────────────────────────────────────


def test_a_completed_earlier_job_with_its_reply_is_adopted_not_rerun(cluster, tree):
    wd = _earlier_workdir(cluster, tag=_tag(cluster, tree))
    (wd / "reply" / "output.json").write_bytes(
        reply_to_json({"status": "ok", "result": StepResult(output_keys=["r.txt"])})
    )
    (wd / "out" / OUT_BUCKET).mkdir()
    (wd / "out" / OUT_BUCKET / "r.txt").write_bytes(b"earlier")
    cluster.add_job("900", JOB_NAME, str(wd), ["COMPLETED"])
    s3 = FakeS3()
    ex = cluster.executor()
    result = ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=_ctx(tree, s3))
    assert result.output_keys == [f"runs/{RUN}/s1/r.txt"]
    assert s3.objects == {(OUT_BUCKET, f"runs/{RUN}/s1/r.txt"): b"earlier"}
    assert result.extra["slurm_job"]["job_id"] == "900"
    assert cluster.calls("sbatch") == []
    assert cluster.workdirs() == []


def test_a_completed_job_that_ran_on_other_inputs_is_not_adopted(cluster, tree):
    # A restart re-ran an upstream step: its outputs (this step's inputs) may
    # differ now, so the earlier job's reply does not belong to this run.
    wd = _completed_earlier(cluster, "0" * 12)
    result = _run(cluster, tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(cluster.calls("sbatch")) == 1
    assert not wd.exists() and cluster.workdirs() == []


def test_the_inputs_tag_follows_what_the_job_computes_from(cluster, tree):
    base = _tag(cluster, tree)
    assert _tag(cluster, tree) == base
    assert _tag(cluster, tree, config={"out": "other.txt"}) != base
    (tree / "steps" / "lane.py").write_text("changed = True\n")
    assert _tag(cluster, tree) != base


def test_the_inputs_tag_follows_an_upstream_output_in_the_runs_own_store(cluster, tree):
    # The in-memory store has no S3 etag of its own: it gives a content tag, so
    # a re-dispatched run that recomputed a parent's output changes the tag.
    import threading

    from umrs.server.workflow_execution_registry import ExecutionRecord
    from umrs.server.workflow_runner import _InMemoryS3Client

    record = ExecutionRecord(execution_id=RUN, workflow_name="w", user_id=OWNER)
    s3 = _InMemoryS3Client(record, state_lock=threading.Lock())
    request = {"owner": OWNER, "step": "s1", "params": {"f": "in/a.txt"},
               "stage_inputs": {"params": ["f"]}}  # fmt: skip

    def tag():
        ctx = StepContext(
            s3=s3, input_bucket="wf-runner", input_keys=[], output_bucket="wf-runner-output",
            metadata={STAGE_REQUEST_METADATA_KEY: request}, run_id=RUN,
        )  # fmt: skip
        plan = build_stage_plan(ctx)
        return inputs_tag(plan, cluster.executor()._payload("m:S", {}, ctx), None)

    s3.put_object(Bucket="wf-runner", Key="in/a.txt", Body=b"first")
    first = tag()
    s3.put_object(Bucket="wf-runner", Key="in/a.txt", Body=b"other")
    assert tag() != first
    s3.put_object(Bucket="wf-runner", Key="in/a.txt", Body=b"first")
    assert tag() == first


def test_a_completed_earlier_job_without_a_reply_is_cleaned_and_the_step_runs(cluster, tree):
    wd = _earlier_workdir(cluster)
    cluster.add_job("900", JOB_NAME, str(wd), ["COMPLETED"])
    result = _run(cluster, tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(cluster.calls("sbatch")) == 1
    assert cluster.workdirs() == []


def test_a_live_earlier_job_is_cancelled_before_the_one_submit(cluster, tree):
    wd = _earlier_workdir(cluster)
    cluster.add_job("901", JOB_NAME, str(wd), ["RUNNING"])
    ex = cluster.executor()
    _run(cluster, tree, ex)
    assert [c[1:] for c in cluster.calls("scancel")] == [["901"]]
    assert len(cluster.calls("sbatch")) == 1
    _join(ex)
    assert cluster.workdirs() == []
    assert cluster.jobs()["901"]["states"] == ["CANCELLED"]


def test_an_earlier_job_that_cannot_be_cancelled_blocks_the_submit(cluster, tree):
    wd = _earlier_workdir(cluster)
    cluster.add_job("901", JOB_NAME, str(wd), ["PENDING"])
    cluster.behave(scancel_rc=1)
    with pytest.raises(StepExecutionError, match="refusing to submit the step twice"):
        _run(cluster, tree)
    assert cluster.calls("sbatch") == []
    assert cluster.workdirs() == [wd]


def test_a_failed_earlier_job_leaves_no_workdir_and_a_foreign_one_is_untouched(
    cluster, tree, tmp_path
):
    wd = _earlier_workdir(cluster)
    cluster.add_job("902", JOB_NAME, str(wd), ["NODE_FAIL"])
    foreign = tmp_path / "elsewhere" / f"{PREFIX}1-abcdefgh"
    foreign.mkdir(parents=True)
    cluster.add_job("903", JOB_NAME, str(foreign), ["RUNNING"])
    _run(cluster, tree)
    assert not wd.exists() and foreign.exists()
    assert all(c[1:] != ["903"] for c in cluster.calls("scancel"))


def test_find_jobs_failing_means_no_submit_at_all(cluster, tree):
    cluster.behave(find_rc=1)
    with pytest.raises(StepExecutionError, match="find-jobs failed"):
        _run(cluster, tree)
    assert cluster.calls("sbatch") == [] and cluster.workdirs() == []


def test_a_sacct_outage_never_reruns_a_step_whose_outputs_are_kept(cluster, tree):
    # The kept COMPLETED job is past squeue's MinJobAge: only sacct knows it.
    wd = _completed_earlier(cluster, _tag(cluster, tree))
    cluster.behave(find_sacct_rc=1)
    with pytest.raises(StepExecutionError, match="find-jobs failed.*sacct failed"):
        _run(cluster, tree)
    assert cluster.calls("sbatch") == [] and cluster.workdirs() == [wd]
    cluster.behave()  # slurmdbd back: the next attempt adopts it
    assert _run(cluster, tree).output_keys == [f"runs/{RUN}/s1/r.txt"]
    assert cluster.calls("sbatch") == [] and cluster.workdirs() == []


# ── scancel on every give-up ────────────────────────────────────────────────


def test_the_queue_clock_cancels_a_job_that_never_starts(cluster, tree):
    cluster.behave(run=False, states=["PENDING"])
    with pytest.raises(StepExecutionError, match="still queued after 1s"):
        _run(cluster, tree, cluster.executor(queue_timeout="00:00:01"))
    assert len(cluster.calls("scancel")) == 1
    assert cluster.workdirs() == []


def test_the_steps_own_deadline_cancels_its_job(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    with pytest.raises(StepExecutionError, match="0.5s deadline"):
        _run(cluster, tree, **{STEP_TIMEOUT_METADATA_KEY: 0.5})
    assert len(cluster.calls("scancel")) == 1
    assert cluster.workdirs() == []


def test_a_workflow_cancel_cancels_the_job(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    polls = []
    ex = cluster.executor()
    ex.bind_run(RUN, cancelled=lambda: polls.append(1) or len(polls) > 2)
    with pytest.raises(StepExecutionError, match="workflow run was cancelled") as info:
        _run(cluster, tree, ex)
    assert info.value.phase == "run"
    assert len(cluster.calls("scancel")) == 1
    assert cluster.workdirs() == []


def test_a_newer_attempt_of_the_step_ends_the_old_ones_job(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor()

    def newer_attempt_starts():
        ex._attempts[(RUN, "s1")] += 1
        return False

    ex.bind_run(RUN, cancelled=newer_attempt_starts)
    with pytest.raises(StepExecutionError, match="abandoned: attempt"):
        _run(cluster, tree, ex)
    assert cluster.calls("sbatch") == []  # superseded before it was even submitted


def test_a_newer_attempt_while_polling_cancels_the_running_job(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor()
    calls = []

    def newer_attempt_after_submit():
        calls.append(1)
        if len(calls) == 3:
            ex._attempts[(RUN, "s1")] += 1
        return False

    ex.bind_run(RUN, cancelled=newer_attempt_after_submit)
    with pytest.raises(StepExecutionError, match="abandoned: attempt 2"):
        _run(cluster, tree, ex)
    assert len(cluster.calls("sbatch")) == 1 and len(cluster.calls("scancel")) == 1


#: Build spec §8, written out: the set in the code must not shrink the test.
SPEC_TERMINAL_FAILURES = (
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
)  # fmt: skip


def test_the_terminal_set_is_the_spec_set():
    assert frozenset(SPEC_TERMINAL_FAILURES) == TERMINAL_FAILURES


@pytest.mark.parametrize("state", SPEC_TERMINAL_FAILURES)
def test_every_terminal_failure_ends_the_poll_with_the_log_tail(cluster, tree, state):
    cluster.behave(run=False, states=["RUNNING", state], files={"log/slurm.err": "boom on node\n"})
    with pytest.raises(StepExecutionError, match=f"status {state}. Error log: boom on node"):
        _run(cluster, tree)
    assert cluster.calls("scancel") == []  # terminal already: nothing to cancel
    assert cluster.workdirs() == []


# ── workdir removal only after a confirmed terminal state ───────────────────


def test_a_job_still_completing_after_scancel_is_cleaned_once_it_ends(cluster, tree):
    cluster.behave(
        run=False, states=["RUNNING"], cancel_to=["COMPLETING", "COMPLETING", "CANCELLED"]
    )
    ex = cluster.executor()
    with pytest.raises(StepExecutionError, match="deadline"):
        _run(cluster, tree, ex, **{STEP_TIMEOUT_METADATA_KEY: 0.3})
    assert ex._settlers, "the workdir must wait for the job to end"
    _join(ex)
    assert cluster.workdirs() == []


def test_a_job_that_never_ends_after_scancel_keeps_its_workdir(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"], cancel_to=["COMPLETING"])
    ex = cluster.executor(settle_seconds=0.5)
    with pytest.raises(StepExecutionError, match="deadline"):
        _run(cluster, tree, ex, **{STEP_TIMEOUT_METADATA_KEY: 0.3})
    _join(ex)
    assert len(cluster.workdirs()) == 1


def test_an_unreachable_login_node_never_loses_a_live_jobs_workdir(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor(settle_seconds=0.5, unreachable_budget=0.5)
    local = ex._remote_argv

    def flaky(command):
        argv = local(command)
        if cluster.calls("sbatch") and (" status " in command or " cancel " in command):
            return ["bash", "-c", "exit 255"]
        return argv

    ex._remote_argv = flaky
    with pytest.raises(StepExecutionError, match="unreadable over SSH"):
        _run(cluster, tree, ex)
    _join(ex)
    assert len(cluster.workdirs()) == 1


# ── sbatch outcomes the server could not see ────────────────────────────────


def test_a_job_queued_despite_an_sbatch_timeout_is_found_by_its_workdir(cluster, tree):
    cluster.behave(sbatch_rc=1, queue_anyway=True)
    result = _run(cluster, tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(cluster.calls("sbatch")) == 1
    assert cluster.workdirs() == []


def test_an_sbatch_timeout_that_queued_nothing_fails_cleanly(cluster, tree):
    cluster.behave(sbatch_rc=1)
    with pytest.raises(StepExecutionError, match="no job was queued"):
        _run(cluster, tree)
    assert cluster.workdirs() == []


def test_an_sbatch_refusal_is_a_plain_failure(cluster, tree):
    cluster.behave(sbatch_rc=1, sbatch_err="sbatch: error: Invalid account")
    with pytest.raises(StepExecutionError, match="sbatch refused the step job: .*Invalid account"):
        _run(cluster, tree)
    assert cluster.workdirs() == []


def test_an_unknown_outcome_that_cannot_be_checked_keeps_the_workdir(cluster, tree):
    ex = cluster.executor()
    submit = ex._submit

    def submit_with_slurm_down(job):
        cluster.behave(sbatch_rc=1, find_rc=1)
        return submit(job)

    ex._submit = submit_with_slurm_down
    with pytest.raises(StepExecutionError, match="workdir .* is kept"):
        _run(cluster, tree, ex)
    assert len(cluster.workdirs()) == 1


def test_staging_refuses_a_nearly_full_filesystem(cluster, tree):
    ex = cluster.executor(caps=dataclasses.replace(DEFAULT_CAPS, min_free_bytes=10**18))
    with pytest.raises(StepExecutionError, match="staging needs at least"):
        _run(cluster, tree, ex)
    assert cluster.calls("sbatch") == [] and cluster.workdirs() == []


def test_the_record_hook_failing_never_fails_the_step(cluster, tree):
    ex = cluster.executor()

    def broken(step, job):
        raise RuntimeError("registry down")

    ex.bind_run(RUN, on_job=broken)
    assert _run(cluster, tree, ex).output_keys


# ── poll clocks ─────────────────────────────────────────────────────────────


@pytest.fixture
def clock_time(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(sse.time, "monotonic", lambda: now[0])
    return now


def test_the_queue_clock_runs_until_the_job_starts(clock_time):
    clock = PollClock(queue_timeout=100, wall_seconds=50, submitted=0.0)
    clock.observe("PENDING")
    clock_time[0] = 100
    assert clock.expired() is None
    clock_time[0] = 101
    assert "still queued" in clock.expired()


def test_the_run_clock_starts_at_running_and_allows_the_grace(clock_time):
    clock = PollClock(queue_timeout=100, wall_seconds=50, submitted=0.0)
    clock_time[0] = 90
    clock.observe("RUNNING")
    clock_time[0] = 90 + 50 + WALL_GRACE_SEC
    assert clock.expired() is None  # queue time never counts against --time
    clock_time[0] += 1
    assert "ran past its 50s time limit" in clock.expired()


def test_a_requeued_job_restarts_both_clocks(clock_time):
    clock = PollClock(queue_timeout=100, wall_seconds=50, submitted=0.0)
    clock_time[0] = 10
    clock.observe("RUNNING")
    clock_time[0] = 20
    clock.observe("REQUEUED")
    assert clock.running_since is None
    clock_time[0] = 120
    assert clock.expired() is None
    clock_time[0] = 121
    assert "still queued" in clock.expired()


def test_unknown_counts_as_queued_and_unreachable_has_its_own_budget(clock_time):
    clock = PollClock(queue_timeout=1000, wall_seconds=50, submitted=0.0, unreachable_budget=60)
    clock.observe(UNKNOWN)
    assert clock.running_since is None
    clock.observe(UNREACHABLE)
    clock_time[0] = 59
    clock.observe(UNREACHABLE)
    assert clock.expired() is None
    clock_time[0] = 60
    assert "unreadable over SSH for 60s" in clock.expired()
    clock.observe("PENDING")
    assert clock.expired() is None


def test_the_local_lane_ends_on_every_terminal_failure_too(monkeypatch):
    ex = SlurmStepExecutor(image="img:1", poll_interval=0, keep_workdir=False)
    monkeypatch.setattr(ex, "_submit_sbatch", lambda *_: "5")
    monkeypatch.setattr(ex, "_check_job_status", lambda *_: "NODE_FAIL")
    ctx = StepContext(s3=None, input_bucket="in", input_keys=[], output_bucket="out", run_id="r")
    with pytest.raises(StepExecutionError, match="status NODE_FAIL"):
        ex.execute(entrypoint="m:S", config={}, ctx=ctx)


# ── job identity: the whole run id, and --comment when Slurm kept it ────────


def test_the_job_name_and_comment_carry_the_whole_run_id(cluster, tree):
    _run(cluster, tree)
    [job] = cluster.jobs().values()
    assert job["name"] == f"umrs-step-{RUN}-s1"
    assert job["comment"] == f"umrs-step:{RUN}:1"
    assert os.path.basename(job["workdir"]).startswith(f"{RUN}-s1-1-")


def test_a_job_whose_comment_names_another_run_is_left_alone(cluster, tree):
    wd = _earlier_workdir(cluster)
    other = "umrs-step:wf-0123456789ac:1"  # same name and workdir stem, not this run
    cluster.add_job("905", JOB_NAME, str(wd), ["RUNNING"], comment=other)
    _run(cluster, tree)
    assert all(c[1:] != ["905"] for c in cluster.calls("scancel"))
    assert wd.exists()


def test_a_live_job_whose_comment_names_this_run_is_cancelled(cluster, tree):
    wd = _earlier_workdir(cluster)
    cluster.add_job("906", JOB_NAME, str(wd), ["RUNNING"], comment=f"umrs-step:{RUN}:1")
    ex = cluster.executor()
    _run(cluster, tree, ex)
    assert [c[1:] for c in cluster.calls("scancel")] == [["906"]]
    _join(ex)


# ── a workdir root behind a symlink (RCCS /lvs0/rccs-dmp -> /lvs0/dne1/...) ──


@pytest.fixture
def linked(tmp_path, monkeypatch):
    return make_cluster(tmp_path, monkeypatch, symlinked_root=True)


def test_behind_a_symlink_a_live_earlier_job_is_still_cancelled(linked, tree):
    wd = _earlier_workdir(linked)
    # recorded as Slurm does without --chdir: the physical path
    linked.add_job("901", JOB_NAME, os.path.realpath(wd), ["RUNNING"])
    ex = linked.executor()
    _run(linked, tree, ex)
    assert [c[1:] for c in linked.calls("scancel")] == [["901"]]
    assert len(linked.calls("sbatch")) == 1  # never two jobs per attempt
    _join(ex)
    assert linked.workdirs() == []


def test_behind_a_symlink_a_completed_earlier_job_is_adopted(linked, tree):
    wd = _completed_earlier(linked, _tag(linked, tree))
    s3 = FakeS3()
    result = linked.executor().execute(entrypoint="steps.lane:Upper", config={}, ctx=_ctx(tree, s3))
    assert result.output_keys == [f"runs/{RUN}/s1/r.txt"]
    assert linked.calls("sbatch") == [] and not wd.exists()


def test_behind_a_symlink_a_physical_completed_job_is_adopted(linked, tree):
    wd = _completed_earlier(linked, _tag(linked, tree), job_id="907", physical=True)
    result = _run(linked, tree)
    assert result.extra["slurm_job"]["job_id"] == "907"
    assert linked.calls("sbatch") == [] and not wd.exists()


def test_behind_a_symlink_the_job_is_submitted_with_the_named_workdir(linked, tree):
    _run(linked, tree)
    [job] = linked.jobs().values()
    assert job["workdir"].startswith(str(linked.root) + "/")  # --chdir, not getcwd()


def test_behind_a_symlink_a_job_queued_despite_an_sbatch_timeout_is_found(linked, tree):
    linked.behave(sbatch_rc=1, queue_anyway=True)
    result = _run(linked, tree)
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(linked.calls("sbatch")) == 1
    assert linked.workdirs() == []


# ── the step's deadline runs on the job's run clock ─────────────────────────


def test_queue_wait_never_counts_against_the_steps_deadline(cluster, tree):
    cluster.behave(states=["PENDING"] * 3 + ["RUNNING", "COMPLETED"], pending_sleep=0.4)
    result = _run(cluster, tree, **{STEP_TIMEOUT_METADATA_KEY: 1.0})
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert cluster.calls("scancel") == []


def test_the_runner_leaves_the_step_deadline_to_an_executor_that_owns_it():
    class _Slow:
        def execute(self, **_kw):
            time.sleep(0.3)
            return StepResult()

    class _OwnsIt(_Slow):
        owns_step_deadline = True

    kw = {"entrypoint": "m:S", "config": {}, "ctx": None, "step_name": "s",
          "timeout_seconds": 0.05}  # fmt: skip
    assert isinstance(_run_executor_attempt(_OwnsIt(), **kw), StepResult)
    with pytest.raises(StepExecutionError, match="deadline"):
        _run_executor_attempt(_Slow(), **kw)
    assert RemoteSlurmStepExecutor.owns_step_deadline is True


# ── scancel reaches Slurm or stays on record ────────────────────────────────


def _flaky_cancel(cluster, ex, down):
    local = ex._remote_argv

    def flaky(command):
        argv = local(command)
        if down["calls"] > 0 and (" cancel " in command or " status " in command):
            down["calls"] -= 1
            return ["bash", "-c", "exit 255"]
        return argv

    ex._remote_argv = flaky


def test_a_scancel_lost_to_an_ssh_blip_is_sent_again(cluster, tree):
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor(cancel_patience=20.0)
    down = {"calls": 0}
    _flaky_cancel(cluster, ex, down)
    polls, recorded = [], []

    def cancel_on_the_second_poll():
        polls.append(1)
        if len(polls) == 2:
            down["calls"] = 2  # the scancel and the state read right after it fail
            return True
        return False

    ex.bind_run(
        RUN, cancelled=cancel_on_the_second_poll,
        on_job=lambda _step, job: recorded.append(job["state"]),
    )  # fmt: skip
    with pytest.raises(StepExecutionError, match="cancelled"):
        _run(cluster, tree, ex)
    _join(ex)
    [job] = cluster.jobs().values()
    assert job["states"] == ["CANCELLED"]
    assert cluster.workdirs() == []
    assert CANCEL_PENDING in recorded and recorded[-1] == "CANCELLED"


def test_an_outage_past_the_budget_records_the_job_kept_not_cancelled(cluster, tree):
    # Round-3 review (2026-09-26): an unreadable state is no stop decision —
    # the job is recorded with its workdir kept for a retry's reconcile.
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor(settle_seconds=0.5, unreachable_budget=0.5, cancel_patience=1.0)
    local = ex._remote_argv

    def down_after_submit(command):
        argv = local(command)
        if cluster.calls("sbatch") and (" cancel " in command or " status " in command):
            return ["bash", "-c", "exit 255"]
        return argv

    ex._remote_argv = down_after_submit
    recorded = []
    ex.bind_run(RUN, on_job=lambda _step, job: recorded.append(dict(job)))
    with pytest.raises(StepExecutionError, match="unreadable over SSH"):
        _run(cluster, tree, ex)
    _join(ex)
    assert recorded[-1]["state"] == UNREACHABLE
    assert recorded[-1]["job_id"] and recorded[-1]["workdir"]
    assert cluster.calls("scancel") == []
    assert len(cluster.workdirs()) == 1


# ── a COMPLETED job's outputs are the only copy ─────────────────────────────


def test_a_lost_collect_out_keeps_the_workdir_and_the_retry_adopts_it(cluster, tree):
    ex = cluster.executor()
    local = ex._remote_argv
    ex._remote_argv = lambda c: ["bash", "-c", "exit 255"] if " collect-out " in c else local(c)
    with pytest.raises(StepExecutionError, match=r"collecting /work/out failed \(rc=255\).*kept"):
        _run(cluster, tree, ex)
    assert len(cluster.workdirs()) == 1
    result = _run(cluster, tree, cluster.executor())  # the runner's retry, or a restart
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(cluster.calls("sbatch")) == 1  # adopted, not run again
    assert cluster.workdirs() == []


_HELPER_FAILED = (
    """echo '{"error": "failed", "detail": "[Errno 5] Input/output error"}' >&2; exit 1"""
)


@pytest.mark.parametrize(
    ("verb", "message"),
    [
        ("collect-out", r"collecting /work/out failed \(rc=1\): \[Errno 5\].*kept"),
        ("read-reply", r"reply could not be read \(rc=1\): \[Errno 5\].*kept"),
    ],
)
def test_a_helper_failure_after_completed_keeps_the_workdir_for_adoption(
    cluster, tree, verb, message
):
    ex = cluster.executor()
    local = ex._remote_argv
    ex._remote_argv = lambda c: ["bash", "-c", _HELPER_FAILED] if f" {verb} " in c else local(c)
    with pytest.raises(StepExecutionError, match=message):
        _run(cluster, tree, ex)
    assert len(cluster.workdirs()) == 1
    result = _run(cluster, tree, cluster.executor())
    assert result.output_keys == [f"runs/{RUN}/s1/result/out.txt"]
    assert len(cluster.calls("sbatch")) == 1  # adopted, not run again
    assert cluster.workdirs() == []


def test_a_refused_reply_removes_the_workdir(cluster, tree):
    ex = cluster.executor()
    local = ex._remote_argv
    refused = """echo '{"error": "refused", "detail": "output.json is a symlink"}' >&2; exit 4"""
    ex._remote_argv = lambda c: ["bash", "-c", refused] if " read-reply " in c else local(c)
    with pytest.raises(StepExecutionError, match="step reply refused: output.json is a symlink"):
        _run(cluster, tree, ex)
    assert cluster.workdirs() == []


@pytest.mark.parametrize(
    ("rel", "mode", "message"),
    [
        ("reply", "000", "step reply refused: reply/ is not readable: the step hid it"),
        (f"out/{OUT_BUCKET}/result/out.txt", "000", "not readable: the step hid it"),
        (f"out/{OUT_BUCKET}/result", "300", "not readable: the step hid it"),
    ],
)
def test_permissions_the_step_withheld_are_a_refusal_not_a_helper_failure(
    cluster, tree, rel, mode, message
):
    # A failure would keep the workdir, and every retry would adopt the same
    # job and fail the same way; a refusal removes it (build spec §5).
    if os.geteuid() == 0:
        pytest.skip("root reads past any mode")
    cluster.behave(chmod_after={rel: mode})
    with pytest.raises(StepExecutionError, match=message):
        _run(cluster, tree)
    assert cluster.workdirs() == []


def test_the_workdir_is_on_record_before_anything_is_staged(cluster, tree):
    ex, recorded = cluster.executor(), []
    ex.bind_run(RUN, on_job=lambda _step, job: recorded.append(dict(job)))
    local = ex._remote_argv
    ex._remote_argv = lambda c: ["bash", "-c", "exit 255"] if " stage-in " in c else local(c)
    with pytest.raises(StepExecutionError, match="stage-in failed"):
        _run(cluster, tree, ex)
    assert recorded[0]["workdir"] and recorded[0]["job_id"] == ""
    assert recorded[0]["cleaned"] is False
    assert recorded[-1]["workdir"] == recorded[0]["workdir"] and recorded[-1]["cleaned"]


def test_a_stream_whose_remote_side_never_exits_is_killed_by_its_timer(cluster):
    ex = cluster.executor()
    ex._remote_argv = lambda _c: ["bash", "-c", "exec >&- 2>&-; exec sleep 600"]

    def refuse(_source):
        raise StepExecutionError("worker", "out/result.csv is not under the output bucket")

    import threading

    done = {}
    thread = threading.Thread(
        target=lambda: done.setdefault("r", ex._stream("x", timeout=1.0, drain=refuse)),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=20)
    assert not thread.is_alive(), "the wait for ssh to exit was unbounded"
    assert done["r"].returncode < 0 and "not under the output bucket" in done["r"].error.message


def test_a_failed_upload_keeps_the_workdir(cluster, tree):
    class _Down(FakeS3):
        def upload_fileobj(self, *_args, **_kw):
            raise ConnectionError("seaweedfs down")

    ex = cluster.executor()
    ctx = _ctx(tree, _Down(), **{STEP_TIMEOUT_METADATA_KEY: 60})
    with pytest.raises(StepExecutionError, match="seaweedfs down; workdir .* is kept"):
        ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert len(cluster.workdirs()) == 1


def test_a_refused_output_larger_than_the_pipe_keeps_its_named_reason(cluster, tree):
    ctx = _ctx(tree, **{STEP_TIMEOUT_METADATA_KEY: 60})
    with pytest.raises(StepExecutionError, match="not under the output bucket") as info:
        cluster.executor().execute(entrypoint="steps.lane:Stray", config={}, ctx=ctx)
    assert "Broken pipe" not in info.value.message
    assert cluster.workdirs() == []  # a refusal is not worth keeping


# ── a COMPLETING earlier job is waited for, not cancelled ───────────────────


def test_a_completing_earlier_job_is_adopted_once_it_completes(cluster, tree):
    _completed_earlier(
        cluster, _tag(cluster, tree), states=("COMPLETING", "COMPLETING", "COMPLETED")
    )
    result = cluster.executor().execute(
        entrypoint="steps.lane:Upper", config={}, ctx=_ctx(tree, FakeS3())
    )
    assert result.output_keys == [f"runs/{RUN}/s1/r.txt"]
    assert cluster.calls("scancel") == [] and cluster.calls("sbatch") == []


def test_an_earlier_job_stuck_completing_blocks_the_submit_without_a_scancel(cluster, tree):
    wd = _completed_earlier(cluster, _tag(cluster, tree), states=("COMPLETING",))
    with pytest.raises(StepExecutionError, match="still COMPLETING"):
        _run(cluster, tree, cluster.executor(settle_seconds=0.5))
    assert cluster.calls("scancel") == [] and cluster.calls("sbatch") == []
    assert wd.exists()


# ── the reply read's unreachable budget is wall time ────────────────────────


def test_the_reply_read_budget_counts_time_not_attempts(monkeypatch):
    ex = RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login"), unreachable_budget=600, poll_interval=5
    )
    now = [0.0]
    monkeypatch.setattr(rse.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(rse.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    attempts = []

    def hanging(verb, *args, **kw):
        attempts.append(verb)
        now[0] += 90  # each attempt blocks until ssh gives up on the dead node
        raise _Unreachable("worker", "helper read-reply: the login node is unreachable")

    ex._helper = hanging
    job = _StepJob(RUN, "s1", 1, JOB_NAME, PREFIX, workdir="/r/x", job_id="7")
    with pytest.raises(StepExecutionError, match="is kept"):
        ex._read_reply_at("/r/x", job)
    assert job.keep and len(attempts) == 7  # 7 x (90 s + 5 s) >= 600 s


# ── every failure is a StepExecutionError (the runner retries only those) ───


def test_an_ssh_that_cannot_start_fails_the_step_by_name(cluster, tree):
    ex = cluster.executor()
    local = ex._remote_argv
    ex._remote_argv = lambda c: ["/nonexistent/ssh"] if " stage-in " in c else local(c)
    with pytest.raises(StepExecutionError, match=r"stage-in failed \(rc=-1\): ssh did not start"):
        _run(cluster, tree, ex)
    assert cluster.workdirs() == []


def test_any_other_exception_becomes_a_step_execution_error(cluster, tree, monkeypatch):
    def broken(*_a, **_k):
        raise KeyError("botocore surprise")

    monkeypatch.setattr(rse, "build_stage_plan", broken)
    with pytest.raises(StepExecutionError, match="KeyError: 'botocore surprise'"):
        _run(cluster, tree)


def test_rmwd_gets_the_transfer_timeout(cluster, tree):
    ex = cluster.executor()
    seen = []
    real = ex._ssh_run

    def spy(command, *, timeout, stdin=None):
        parts = command.split()
        seen.append((parts[3] if len(parts) > 3 else parts[0], timeout))
        return real(command, timeout=timeout, stdin=stdin)

    ex._ssh_run = spy
    _run(cluster, tree, ex)
    assert ("rmwd", rse._TRANSFER_TIMEOUT) in seen


# ── outages are no stop decision (round-3 review, 2026-09-26) ───────────────


def test_an_ssh_outage_past_the_budget_never_cancels_the_job(cluster, tree):
    """Past the unreachable budget the step fails, but the job keeps running
    under its own --time and its workdir stays for a retry to reconcile: no
    scancel, no rmwd (a finished job's outputs would be its only copy)."""
    cluster.behave(run=False, states=["RUNNING"])
    ex = cluster.executor(settle_seconds=0.5, unreachable_budget=0.3)
    local = ex._remote_argv

    def outage(command):
        if cluster.calls("sbatch") and " status " in command:
            return ["bash", "-c", "exit 255"]
        return local(command)

    ex._remote_argv = outage
    with pytest.raises(StepExecutionError, match="unreadable over SSH"):
        _run(cluster, tree, ex)
    _join(ex)
    assert cluster.calls("scancel") == []
    assert len(cluster.workdirs()) == 1


def test_a_job_that_completed_before_any_scancel_reached_it_keeps_its_outputs(cluster):
    wd = _earlier_workdir(cluster)
    cluster.add_job("907", JOB_NAME, str(wd), ["COMPLETED"])
    ex = cluster.executor(settle_seconds=0.5)
    assert ex._settle_until_terminal("907", str(wd), False, None) is False
    assert wd.exists()


def test_a_site_rewritten_comment_falls_back_to_the_workdir_stem(cluster, tree):
    wd = _earlier_workdir(cluster)
    cluster.add_job("908", JOB_NAME, str(wd), ["RUNNING"], comment="acct=cloud-users")
    ex = cluster.executor()
    _run(cluster, tree, ex)
    assert [c[1:] for c in cluster.calls("scancel")] == [["908"]]
    _join(ex)


@pytest.mark.parametrize(
    ("site_gpu", "hint", "expected"),
    [("1", True, True), ("1", "true", True), ("1", "false", False), ("1", None, False),
     ("0", True, False), ("", "true", False)],
)  # fmt: skip
def test_a_gpu_needs_a_true_hint_on_a_gpu_site(cluster, monkeypatch, site_gpu, hint, expected):
    """Peer review 2026-09-26: bool("false") asked for a GPU, and the site's
    gpu: false was ignored whenever a step carried any hint."""
    from umrs.pipeline_sdk.step import StepContext
    from umrs.server.step_executor import EXECUTOR_GPU_METADATA_KEY

    monkeypatch.setenv("UMRS_STEP_GPU", site_gpu)
    metadata = {} if hint is None else {EXECUTOR_GPU_METADATA_KEY: hint}
    ctx = StepContext(s3=None, input_bucket="in", input_keys=[], output_bucket="out",
                      metadata=metadata)  # fmt: skip
    assert cluster.executor()._needs_gpu(ctx) is expected


def test_a_broken_pipe_while_feeding_becomes_a_stream_error():
    """CI 36225352647: the remote exited before stage-in read its stdin, and
    the step failed with a bare BrokenPipeError instead of the named cause.
    A broken pipe is a broken stream: the remote's rc and stderr explain it."""
    from types import SimpleNamespace

    from umrs.server.step_staging import StageStreamError

    def feed(_pipe):
        raise BrokenPipeError(32, "Broken pipe")

    proc = SimpleNamespace(stdin=None, stdout=None)
    value, error, stdout = rse._pump(proc, feed, None)
    assert isinstance(error, StageStreamError) and stdout == b""
    assert isinstance(error, StepExecutionError)
