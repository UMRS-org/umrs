"""Fail-first regressions for the final ADR-112 server/ledger residuals.

These tests were run before the production fixes and their failures recorded
in /tmp/opencode/server-residual-before.txt. They intentionally exercise the
real in-memory ledger and lane write points, not just source string checks.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.hpc_accounts import usage_ledger as ul
from umrs.hpc_accounts.ledger_sweep import UsageLedgerSweep
from umrs.jobs import hpc_ledger
from umrs.knowledge import umrs_workflow_port
from umrs.orchestrator.hpc_identity import HpcIdentity, Lane
from umrs.pipeline_sdk.step import StepContext
from umrs.server import gated_step_lane as gated
from umrs.server.slurm_step_executor import SlurmStepExecutor
from umrs.server.step_ledger import LEDGER_METADATA_KEY, service_pending


def _unit(
    attempt: str,
    *,
    tag: str = "unique",
    project: str = "p1",
    method: str = "platform",
    venue: str = "v",
) -> ul.LedgerUnit:
    return ul.LedgerUnit(
        venue=venue,
        attempt_id=attempt,
        person="alice",
        lane="staged_step",
        tag=tag,
        project_id=project,
        method=method,
    )


def _profile(*, exclusive: bool = True, gpus: int = 8) -> Any:
    return SimpleNamespace(exclusive=exclusive, gpus=gpus, gres="", max_gpus=gpus)


def _pricing_site(profile: Any) -> Any:
    return SimpleNamespace(get_partition_profile=lambda _name: profile)


def _ctx() -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="data",
        input_keys=[],
        output_bucket="out",
        config={},
        metadata={LEDGER_METADATA_KEY: {"person": "alice", "project_id": "p1"}},
        run_id="wf-residual",
    )


# ── binding K4 prices ────────────────────────────────────────────────────────


def test_remote_service_pending_charges_the_exclusive_node() -> None:
    ledger = ul.InMemoryUsageLedger()
    ul.set_usage_ledger(ledger)
    job = SimpleNamespace(
        run_id="wf-residual",
        step="s1",
        attempt=1,
        dispatch_id="dispatch",
        job_name="job",
        site_login="operator",
        slurm_account="acct",
    )
    service_pending(job, _ctx(), 4, 1, 60, partition=_profile(gpus=8))
    (row,) = ledger.tables["rows"]
    assert (row["alloc_gpus"], row["elapsed_s"]) == (8, 60)


def test_hpc_script_estimate_charges_its_exclusive_partition(monkeypatch) -> None:
    profile = _profile(gpus=8)
    monkeypatch.setattr(
        "umrs.shared.hpc_user_work_access.current_site",
        lambda: _pricing_site(profile),
    )
    est = hpc_ledger.script_estimate(["--partition=whole", "--gpus=1", "--time=1"])
    assert (est.gpus, est.time_s) == (8, 60)


def test_local_slurm_pending_charges_its_exclusive_partition(monkeypatch, tmp_path) -> None:
    ledger = ul.InMemoryUsageLedger()
    ul.set_usage_ledger(ledger)
    profile = _profile(gpus=8)
    monkeypatch.setattr(
        "umrs.shared.hpc_user_work_access.current_site",
        lambda: _pricing_site(profile),
    )
    executor = SlurmStepExecutor(
        image="img:1", partition="whole", gpus=1, cpus=4, workdir_root=str(tmp_path)
    )
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    monkeypatch.setattr(executor, "_submit_sbatch", lambda _path, _wd: "777")
    job = executor._ledger_job(_ctx(), {"step_name": "s1"})
    executor._ledgered_submit(job, _ctx(), "run.sbatch", "wd")
    pending = ledger.tables["rows"][0]
    assert (pending["alloc_gpus"], pending["elapsed_s"]) == (8, 3600)


# ── one scheduler job, one owner, even under a first-association race ────────


class _RacingLedger(ul.InMemoryUsageLedger):
    """Force both threads through the old non-atomic lookup window.

    The fixed ledger exposes the scheduler-job lock used by the production
    write path, so its serialized execution needs no artificial interleaving.
    """

    barrier = threading.Barrier(2)

    def unit_for_scheduler_job(self, venue: str, scheduler_job_id: str):
        found = super().unit_for_scheduler_job(venue, scheduler_job_id)
        if not hasattr(self, "_scheduler_job_locked"):
            self.barrier.wait(timeout=5)
        return found


def test_concurrent_first_association_and_sweep_settle_one_job_once() -> None:
    ledger = _RacingLedger(tables={"rows": [], "budgets": {"p1": 0}})
    ul.set_usage_ledger(ledger)
    threads = [
        threading.Thread(
            target=ul.write_job_after,
            args=("submitted", _unit(f"r/s/{n}/dispatch"), "900", None),
        )
        for n in (1, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(ledger.current("v")) == 1
    usage = {
        "state": "COMPLETED",
        "elapsed": "00:10:00",
        "alloc_cpus": "1",
        "alloc_gpus": "1",
        "alloc_tres": "gres/gpu=1",
    }

    class Sacct:
        def match(self, _row, _since):
            return []

        def sacct(self, ids):
            return dict.fromkeys(ids, usage)

    UsageLedgerSweep(
        ledger,
        "v",
        gated=None,
        service=Sacct(),
        now=lambda: datetime.now(timezone.utc) + timedelta(minutes=1),
    ).run_once()
    assert len(ledger.current("v")) == 1
    assert ledger.project_totals("p1")["gpu_seconds"] == 600


class _Cursor:
    def __init__(self, calls):
        self.calls = calls

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, sql, params):
        self.calls.append((sql, params))


class _Conn:
    def __init__(self, calls):
        self.calls = calls
        self.commits = 0

    def cursor(self):
        return _Cursor(self.calls)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class _Pool:
    def __init__(self):
        self.calls = []
        self.conn = _Conn(self.calls)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        pass


def test_pg_scheduler_association_uses_an_advisory_job_lock() -> None:
    pool = _Pool()
    ledger = ul.PgUsageLedger(pool)
    with ledger._scheduler_job_locked("v", "900"):
        pass
    assert len(pool.calls) == 1
    assert "pg_advisory_xact_lock" in pool.calls[0][0]
    assert pool.calls[0][1] == ("hpc-ledger-job:v:900",)
    assert pool.conn.commits == 1


@pytest.mark.integration
def test_pg_concurrent_first_association_and_canonical_settlement() -> None:
    import uuid

    from hpc_ledger_pg import delete_ledger_rows, ledger_pool_or_skip

    pool = ledger_pool_or_skip()
    venue = f"race-{uuid.uuid4().hex[:8]}"
    ledger = ul.PgUsageLedger(pool)
    ul.set_usage_ledger(ledger)
    try:
        threads = [
            threading.Thread(
                target=ul.write_job_after,
                args=("submitted", _unit(f"r/s/{n}/dispatch", venue=venue), "900", None),
            )
            for n in (1, 2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(ledger.current(venue)) == 1
        usage = {
            "state": "COMPLETED",
            "elapsed_s": 600,
            "alloc_cpus": 1,
            "alloc_gpus": 1,
        }
        ul.write_job_after("settled", _unit("r/s/new/dispatch", venue=venue), "900", usage, "sacct")
        rows = ledger.current(venue)
        assert len(rows) == 1
        assert rows[0]["state"] == "COMPLETED"
        assert ledger.project_totals("p1")["gpu_seconds"] == 600
    finally:
        delete_ledger_rows(pool, [venue])


# ── respec: no eviction while active, cleanup at the final lane finish ───────


@pytest.fixture(autouse=True)
def _clean_respec():
    gated._RESPEC_RUNS.clear()
    active = getattr(gated, "_RESPEC_ACTIVE", None)
    if active is not None:
        active.clear()
    yield
    gated._RESPEC_RUNS.clear()
    if active is not None:
        active.clear()


def _grant_run(run_id: str, now: int, expires: int) -> gated.GatedRun:
    import base64
    import json

    body = {"run_id": run_id, "issued": now - 3500, "expires": expires, "nonce": "0" * 32}
    raw = base64.b64encode(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return gated.GatedRun("v", "alice", raw, "", "", kind="grant", key="key-" + run_id)


def test_parallel_respec_bytes_survive_other_active_runs(monkeypatch) -> None:
    now = int(time.time())
    counter = [0]

    def fresh(run, at):
        counter[0] += 1
        import base64
        import json

        body = {
            **run.body(),
            "issued": at,
            "expires": at + 3600,
            "nonce": format(counter[0], "032x"),
        }
        raw = base64.b64encode(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).decode()
        return raw, str(counter[0])

    monkeypatch.setattr(gated, "_fresh_at", fresh)
    original = _grant_run("old", now, now + 60)
    first = gated.fresh_run(original, now)
    for index in range(128):
        gated.fresh_run(_grant_run(f"other-{index}", now, now + 3600), now)
    second = gated.fresh_run(original, now)
    assert second.intent_b64 == first.intent_b64


def test_workflow_finish_removes_only_that_runs_respec_state(monkeypatch) -> None:
    now = int(time.time())

    def fresh(run, at):
        return run.intent_b64 + "fresh", "fresh-id"

    monkeypatch.setattr(gated, "_fresh_at", fresh)
    grant_id = "a" * 32
    run_id = "wf-aaaaaaaaaaaa"
    old = _grant_run(run_id, now, now + 60)
    old = replace(old, key=gated.grant_run_key(grant_id, run_id))
    other = _grant_run("other", now, now + 60)
    gated.fresh_run(other, now)
    gated.fresh_run(old, now)
    record = SimpleNamespace(execution_id=run_id, grant_id=grant_id)
    gated.clear_run_respec(record)
    assert old.key not in gated._RESPEC_RUNS
    assert "key-other" in gated._RESPEC_RUNS


# ── one taken-ID read per sweep, updated after each association ──────────────


class _CountingLedger(ul.InMemoryUsageLedger):
    current_calls = 0

    def current(self, *args, **kwargs):
        self.current_calls += 1
        return super().current(*args, **kwargs)


def test_sweep_loads_taken_scheduler_ids_once_per_cycle() -> None:
    ledger = _CountingLedger()
    for index in (1, 2):
        ledger.pending(
            _unit(f"r/s/{index}/dispatch", method="gated_key"),
            ul.Resources(1, 0, 60),
        )

    class Gate:
        def lookup(self, _row):
            return "job", "55"

        def usage(self, _row, _ids):
            return {}

    UsageLedgerSweep(
        ledger,
        "v",
        gated=Gate(),
        service=None,
        now=lambda: datetime.now(timezone.utc) + timedelta(minutes=20),
    ).run_once()
    assert ledger.current_calls == 1
    assert sorted(row["state"] for row in ledger.current("v")) == ["not_submitted", "submitted"]


# ── full sbatch estimate grammar ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("args", "gpus", "seconds"),
    [
        (["--time=1-02", "--gpus=1"], 1, 93600),
        (["--time=1-02:03", "--gpus=1"], 1, 93780),
        (["--time=1-02:03:04", "--gpus=1"], 1, 93840),
        (["--gpus-per-node=a100:4", "--nodes=2", "--time=60"], 8, 3600),
        (["--gpus-per-task=3", "--ntasks=4", "--time=60"], 12, 3600),
        (["--gpus-per-node=2", "--nodes=2-4", "--time=60"], 8, 3600),
    ],
)
def test_full_slurm_estimate_grammar(args, gpus, seconds) -> None:
    est = hpc_ledger.script_estimate(args)
    assert (est.gpus, est.time_s) == (gpus, seconds)


def test_unknown_gpu_multiplicity_makes_the_budget_estimate_incomplete() -> None:
    args = ["--gpus-per-node=2", "--nodes=many", "--time=60"]
    assert hpc_ledger.script_estimate_complete(args) is False


# ── unique loop/local recovery identity and nonzero loop walltime ────────────


IDENTITY = HpcIdentity("v", Lane.USER_WORK, "service_account", "op", "acct", False, True)


def _loop_job(job_id: str = "loop-job"):
    return SimpleNamespace(job_id=job_id, project_id="p1", user_id="alice", spec={})


def _loop_attempt(attempt_id: str = "att-1"):
    return SimpleNamespace(attempt_id=attempt_id, metrics={}, backend_ref="")


def test_cluster_loop_tag_is_submit_name_unique_and_has_real_walltime(monkeypatch) -> None:
    from umrs.jobs.cluster_venue import ClusterSlurmSubmitter
    from umrs.shared import handshake

    monkeypatch.setattr(handshake, "handshake_secret", lambda: "secret")
    calls = []

    class Backend:
        def __init__(self, _config):
            pass

        async def submit_job(self, *_args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(job_id="777")

    monkeypatch.setattr("umrs.orchestrator.remote_slurm_backend.RemoteSlurmBackend", Backend)
    ledger = ul.get_usage_ledger()
    sub = ClusterSlurmSubmitter(identity=lambda _job: IDENTITY)
    import asyncio

    ref = asyncio.run(sub.submit(job=_loop_job(), attempt=_loop_attempt(), spec={"loop_id": "L1"}))
    pending = ledger.tables["rows"][0]
    assert ref.endswith("slurm_id=777")
    assert pending["tag"] != "umrs-loop-job"
    assert pending["elapsed_s"] > 0
    assert calls[0]["job_name"] == pending["tag"]


class _LocalReturn:
    def __init__(self, answer):
        self.answer = answer

    def __call__(self, *_args, **_kwargs):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.parametrize(
    "answer",
    [
        SimpleNamespace(returncode=0, stdout="unparseable", stderr=""),
        subprocess.TimeoutExpired(["sbatch"], 60),
    ],
    ids=["unparseable", "timeout"],
)
def test_uncertain_local_submit_keeps_a_uniquely_named_workdir(
    monkeypatch, tmp_path, answer
) -> None:
    from umrs.server.step_executor import StepExecutionError

    executor = SlurmStepExecutor(image="img:1", poll_interval=0, workdir_root=str(tmp_path))
    match = None if isinstance(answer, Exception) else "Unexpected sbatch output"
    with pytest.raises((StepExecutionError, subprocess.TimeoutExpired), match=match):
        submit = _LocalReturn(answer)
        monkeypatch.setattr(executor, "_run_cmd", submit)
        executor.execute(entrypoint="m:S", config={"step_name": "s1"}, ctx=_ctx())
    workdirs = list(tmp_path.iterdir())
    assert len(workdirs) == 1
    pending = ul.get_usage_ledger().tables["rows"][0]
    assert pending["tag"] != "umrs-step"
    script = (workdirs[0] / "run_step.sbatch").read_text()
    assert f"#SBATCH --job-name={pending['tag']}" in script


def test_workflow_runner_finally_releases_the_completed_runs_cache() -> None:
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    run_id = "wf-bbbbbbbbbbbb"
    grant_id = "b" * 32
    key = gated.grant_run_key(grant_id, run_id)
    gated._RESPEC_RUNS[key] = ("old", "old-id")
    record = ExecutionRecord(
        execution_id=run_id,
        workflow_name="wf",
        user_id="alice",
        grant_id=grant_id,
    )
    wr.run_workflow_record(record, source_path=None, workflow_json={"steps": []})
    assert key not in gated._RESPEC_RUNS


# ── identity-scoped recovery tags and service-loop admission ─────────────────


def test_recovery_tags_include_attempt_person_and_project() -> None:
    base = hpc_ledger.recovery_tag("umrs-loop", "p1", "alice", "attempt")
    other_attempt = hpc_ledger.recovery_tag("umrs-loop", "p1", "alice", "other")
    other_person = hpc_ledger.recovery_tag("umrs-loop", "p1", "bob", "attempt")
    other_project = hpc_ledger.recovery_tag("umrs-loop", "p2", "alice", "attempt")
    assert len({base, other_attempt, other_person, other_project}) == 4
    assert all(
        tag.startswith("umrs-loop-") for tag in (base, other_attempt, other_person, other_project)
    )


# ── service-account research-loop admission before any durable row ──────────


def test_research_loop_service_run_uses_whole_run_service_budget(monkeypatch) -> None:
    identity = IDENTITY
    monkeypatch.setattr(
        "umrs.server.hpc_user_work.workflow_user_work_identity",
        lambda *_a, **_k: identity,
    )
    monkeypatch.setattr(
        umrs_workflow_port,
        "_grant_target",
        lambda _entry: ({"steps": [{"name": "s1", "entrypoint": "m:S"}]}, "/tmp/source"),
    )
    calls = []

    def refuse(target, ident, run, params):
        calls.append((target, ident, run, params))
        raise ul.BudgetExceeded("p1", 0, 0, 3600, 3600)

    monkeypatch.setattr("umrs.server.grant_admission.check_service_run_budget", refuse)
    entry = {"name": "wf", "version": "1.0.0"}
    with pytest.raises(RuntimeError, match="hpcaccess.budget_exceeded"):
        umrs_workflow_port._grant_run(entry, "alice", None, {"project_id": "p1"}, None)
    assert calls and calls[0][2]["project"] == "p1"
