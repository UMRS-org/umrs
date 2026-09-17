"""ADR-048 pins — per-job respawn budget ledger (WP-20 leg).

The durable retry chain multiplied engine-respawn capacity across attempts
(6b-7 / G-P4-6): every attempt arrived with a fresh per-attempt budget while
attempt N's stacks may still hold the GPU. The ledger keeps the accounting
node-local, flock-locked, and keyed to the DURABLE job id — so even N
attempts share exactly one budget per job lineage.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm" / "attempt_budget.sh"


def _sh(env: dict[str, str], script: str) -> subprocess.CompletedProcess:
    merged = dict(os.environ)
    merged.update(env)
    return subprocess.run(
        ["bash", "-c", f"source {LIB} && {script}"],
        env=merged,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture()
def info_dir(tmp_path):
    d = tmp_path / "budget-info"
    d.mkdir(parents=True)
    return str(d)


def test_claim_bursts_stop_at_budget_total(info_dir: str) -> None:
    env = {"JOB_INFO_DIR": info_dir, "RIVAULT_ENGINE_RESPAWN_MAX_PER_JOB": "3"}
    for idx in range(1, 4):
        r = _sh(env, f"budget_claim_job job-trial att-{idx}")
        assert r.returncode == 0, (idx, r.stderr)
    fourth = _sh(env, "budget_claim_job job-trial att-4")
    assert fourth.returncode != 0, "fourth claim must be refused once budget_total hits"
    ledger = json.loads((Path(info_dir) / "job-trial.budget").read_text())
    assert ledger["budget_used"] == 3
    assert len(ledger["claims"]) == 3


def test_marker_writes_reason_row_and_exits_125(info_dir: str) -> None:
    env = {"JOB_INFO_DIR": info_dir}
    r = _sh(env, "budget_mark_exhausted job-marco 4242")
    assert r.returncode == 125
    info = (Path(info_dir) / "rivault-job-4242.info").read_text()
    assert "RIVAULT_TERMINAL_REASON=budget_exhausted" in info
    assert "RIVAULT_BUDGET_TOTAL=3" in info


def test_concurrent_races_still_bounded_even_with_parallel_claims(info_dir: str) -> None:
    env = {"JOB_INFO_DIR": info_dir}
    # two workers race for the same job's last claim: flock(1) must serialize
    # them to exactly one winner — never both.
    cmd = (
        "(budget_claim_job race-job winner >/dev/null 2>&1; echo win:$?) & "
        "(budget_claim_job race-job loser >/dev/null 2>&1; echo lose:$?) & wait"
    )
    r = _sh(env, cmd)
    assert "win:0" in r.stdout + r.stderr or "lose:0" in r.stdout + r.stderr
    ledger = json.loads((Path(info_dir) / "race-job.budget").read_text())
    assert ledger["budget_used"] == 1


class _RealDictCursor:
    """psycopg2.RealDictCursor-shaped page for mark_attempt_failed's three SQLs."""

    def __init__(self, conn):
        self._conn = conn
        self._rows: list = []

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, sql, params=None):
        stitched = " ".join(sql.split()) if isinstance(sql, str) else str(sql)
        if "UPDATE job_attempts" in stitched and "RETURNING job_id, attempt_no" in stitched:
            self._rows = [{"job_id": self._conn.job["job_id"], "attempt_no": 1}]
            self._conn.updates.append(("attempt_failed", params))
        elif "SELECT * FROM jobs" in stitched:
            self._rows = [self._conn.job_row]
        elif "UPDATE jobs SET" in stitched:
            # status param slot tells the transition: queued vs failed
            self._conn.updates.append(("job_status", params))
            self._rows = []
        else:
            self._rows = []

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        out = self._rows[:]
        self._rows = []
        return out


class _BudgetConn:
    def __init__(self):
        self.job = {"job_id": "job-1", "job_type": "endpoint_provision"}
        self.job_row = self.job
        self.updates: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, **_kw):
        return _RealDictCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class _BudgetStore:
    def __init__(self, conn):
        self._conn = conn
        self._post_failure_callback = None

    def _get_conn(self):
        return _Ctx(self._conn)

    def _record_attempt_timing(self, *_a, **_kw):  # harness stub
        pass

    def _persist_attempt_manifest(self, *_a, **_kw):  # harness stub
        pass

    def _reconcile_fan_out_parent(self, *_a, **_kw):  # harness stub
        pass

    def _row_to_job(self, row):
        from umrs.jobs.models import Job, JobStatus, JobType, RestartPolicy

        return Job(
            job_id=row["job_id"],
            project_id="default",
            user_id="carol",
            job_type=JobType(row["job_type"]),
            spec={},
            idempotency_key="ik",
            status=JobStatus.RUNNING,
            active_attempt_id="att-1",
            restart_policy=RestartPolicy.ON_FAILURE,
            desired_max_attempts=10,
        )


class _Ctx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *_a):
        return False


def test_durable_mark_attempt_failed_skips_requeue_on_budget_marker() -> None:
    """Store D-3 boundary: exit_reason carrying `budget_exhausted` never requeues —
    the jobs row lands FAILED terminal (never QUEUED again)."""
    from umrs.jobs.store import JobStore

    conn = _BudgetConn()
    store = _BudgetStore(conn)
    # call the target function directly with the store as self — it uses only
    # self._get_conn() + self._row_to_job() from its class
    JobStore.mark_attempt_failed(store, "att-1", "budget_exhausted")
    job_updates = [u for u in conn.updates if u[0] == "job_status"]
    assert job_updates, "no jobs status update was written at all"
    status_params = job_updates[-1][1]
    assert status_params[0] == "failed", status_params
    assert all(p[0] != "queued" for _, p in job_updates), job_updates


def test_durable_mark_attempt_failed_requeues_other_reasons() -> None:
    """Counter-shape pin: ordinary failures still requeue (guard only fires
    on the literal marker)."""
    from umrs.jobs.store import JobStore

    conn = _BudgetConn()
    store = _BudgetStore(conn)
    JobStore.mark_attempt_failed(store, "att-1", "timeout watching engine")
    job_updates = [u for u in conn.updates if u[0] == "job_status"]
    assert job_updates, "updates were not emitted"
    assert job_updates[-1][1][0] != "failed"
    assert any(p[0] == "queued" for _, p in job_updates), job_updates


def test_remote_slurm_marker_short_circuit_before_ghost_sweep(monkeypatch) -> None:
    """JobInfo from an exhausted wrapper: FAILED + state_reason=budget_exhausted,
    and the ghost-file sweep never gets to delete the evidence."""
    import asyncio

    from umrs.orchestrator.base import JobStatus
    from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    marker = (
        "RIVAULT_STATUS=running\n"
        "RIVAULT_TERMINAL_REASON=budget_exhausted\n"
        "RIVAULT_JOB_ID=standalone-1\n"
    )

    async def fake_read(job_id):
        return marker

    async def fake_query(job_id):
        raise AssertionError("squeue fallback must not fire on a marker")

    monkeypatch.setattr(backend, "_read_remote_job_info", fake_read)
    monkeypatch.setattr(backend, "_query_slurm_job_status", fake_query)

    info = asyncio.get_event_loop().run_until_complete(
        backend.get_job_status("standalone-1")
    )
    assert info.status == JobStatus.FAILED
    assert info.state_reason == "budget_exhausted"
    assert info.metadata.get("RIVAULT_TERMINAL_REASON") == "budget_exhausted"


def test_wrapper_uses_budget_claim_and_marker_lib() -> None:
    """serve-model.slurm integration pins (shape analysis — same style as the
    egress static guards)."""
    wrapper = (
        Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm" / "serve-model.slurm"
    )
    text = wrapper.read_text()
    assert "attempt_budget.sh" in text
    assert "budget_claim_job" in text
    assert "budget_mark_exhausted" in text
    assert "_BUDGET_VIA_LIB" in text


def test_adapters_submit_env_carries_durable_job_id() -> None:
    """The orchestrator submit_env must carry UMRS_DURABLE_JOB_ID so the
    wrapper's claim file keys to the JOB lineage, never one attempt."""
    src = (
        Path(__file__).resolve().parents[1] / "src" / "umrs" / "jobs" / "adapters.py"
    ).read_text()
    assert 'submit_env["UMRS_DURABLE_JOB_ID"] = job.job_id' in src
