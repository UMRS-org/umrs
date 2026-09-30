"""A second dispatch of a run still running here attaches to it (peer review 2026-09-26).

A scheduler requeue after a lost submit acknowledgement, or a boot reconcile,
re-dispatches the same execution id. Two live runs of one id would cancel each
other's step jobs on the cluster and race on the record.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from umrs.api import jobs as api_jobs
from umrs.jobs.models import JobStatus, JobType
from umrs.server import step_reaper
from umrs.server import workflow_registry_handler as wrh
from umrs.server.workflow_execution_registry import WorkflowExecutionRegistry


def _running(registry: WorkflowExecutionRegistry, release: threading.Event):
    record = registry.create(workflow_name="w", user_id="alice", execution_id="wf-attach")
    started = threading.Event()

    def runner(rec):
        started.set()
        release.wait(10)
        rec.update(status="succeeded")

    registry.launch(record, runner)
    assert started.wait(5)
    return record


def test_running_here_follows_the_runner_thread():
    registry, release = WorkflowExecutionRegistry(), threading.Event()
    record = _running(registry, release)
    assert registry.running_here("wf-attach") is record
    release.set()
    assert registry.join("wf-attach", timeout=5)
    assert registry.running_here("wf-attach") is None
    assert registry.running_here("wf-unknown") is None


def test_a_second_dispatch_attaches_instead_of_launching(monkeypatch):
    registry, release = WorkflowExecutionRegistry(), threading.Event()
    record = _running(registry, release)
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    monkeypatch.setattr(wrh, "_workflow_run_context", lambda *_a, **_k: ("/src", {}, 1))
    launched = []
    monkeypatch.setattr(registry, "launch", lambda *args, **_k: launched.append(args))
    try:
        got, source, steps = wrh._launch_workflow_from_catalog(
            workflow_name="w", client_id="alice", params={}, execution_id="wf-attach"
        )
        assert got is record and (source, steps) == ("/src", 1)
        assert launched == []
        assert registry.get("wf-attach") is record  # no replacement record
    finally:
        release.set()
        registry.join("wf-attach", timeout=5)


def test_an_ending_runner_never_drops_a_newer_launch_of_the_same_id():
    # Peer review (2026-09-26): the finally of run A deleted the entry by id,
    # so run B (a requeue that launched meanwhile) lost it and a later
    # dispatch started a second run instead of attaching.
    registry, release = WorkflowExecutionRegistry(), threading.Event()
    _running(registry, release)
    newer = threading.Thread(target=release.wait, args=(10,))
    newer.start()
    with registry._lock:
        older = registry._workers["wf-attach"]
        registry._workers["wf-attach"] = newer
    release.set()
    older.join(timeout=5)  # run A's finally has run
    newer.join(timeout=5)
    with registry._lock:
        assert registry._workers.get("wf-attach") is newer


# ── a dispatch of a cancelled run (peer review F2, 2026-09-26) ─────────────


def _dispatch_env(monkeypatch, job_status=None):
    """A registry, a job store whose durable job has *job_status* (None: no
    store), and the step jobs the reaper was asked to end."""
    registry, reaped = WorkflowExecutionRegistry(), []
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    monkeypatch.setattr(wrh, "_workflow_run_context", lambda *_a, **_k: ("/src", {}, 0))
    monkeypatch.setattr(
        step_reaper, "reap_run_jobs", lambda r, for_retry=None: reaped.append((r, for_retry))
    )
    store = SimpleNamespace(get_job=lambda _id: SimpleNamespace(status=job_status))
    monkeypatch.setattr(api_jobs, "get_optional_job_store", lambda: store if job_status else None)
    return registry, reaped


def _dispatch(registry, execution_id="wf-attach"):
    record, _source, _steps = wrh._launch_workflow_from_catalog(
        workflow_name="w", client_id="alice", params={}, execution_id=execution_id,
        durable_job_id="job-1",
    )  # fmt: skip
    assert registry.join(execution_id, timeout=5)
    return record


def _reaped_for_real(reaped, record) -> bool:
    deadline = time.monotonic() + 5
    while (record, False) not in reaped and time.monotonic() < deadline:
        time.sleep(0.01)
    return (record, False) in reaped


def _ended_with_a_live_job(registry, cancelled=False):
    prior = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    prior.update(status="cancelled" if cancelled else "failed", finished_at=time.time(),
                 cancel_requested=cancelled)  # fmt: skip
    left = {"job_id": "900", "workdir": "/r/wd", "cleaned": False}
    step_reaper.record_step_job(prior, "s1", left)
    return prior, left


def test_a_dispatch_racing_a_cancel_runs_nothing_and_ends_the_carried_jobs(monkeypatch):
    # The cancel handler cancelled the durable job; the scheduler, which read
    # it before, re-dispatched the run before the registry cancel. The new
    # record replaced the old one (whose reaper stops as superseded) and
    # resumed the live job.
    registry, reaped = _dispatch_env(monkeypatch, job_status=JobStatus.CANCELLED)
    prior, left = _ended_with_a_live_job(registry)
    record = _dispatch(registry)
    assert record is not prior and record.status == "cancelled" and record.cancel_requested
    assert step_reaper.unreaped_steps(record) == {"s1": [left]}
    assert _reaped_for_real(reaped, record), reaped


def test_a_requeued_cancelled_run_runs_again(monkeypatch):
    # An admin retry or a reproduce requeues the cancelled durable job with
    # the same execution id: the earlier record's cancel does not refuse it.
    registry, _reaped = _dispatch_env(monkeypatch, job_status=JobStatus.QUEUED)
    prior, _left = _ended_with_a_live_job(registry, cancelled=True)
    record = _dispatch(registry)
    assert record is not prior and record.status == "succeeded"


def _cancel_order(monkeypatch, registry, job):
    """The order in which the durable job and the registry record are
    cancelled."""
    order: list[str] = []
    store = SimpleNamespace(get_job=lambda _id: job,
                            cancel_job=lambda *_a, **_k: order.append("durable job"))  # fmt: skip
    monkeypatch.setattr(api_jobs, "get_optional_job_store", lambda: store)
    monkeypatch.setattr(api_jobs, "_job_store", store)
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    real_cancel = registry.cancel
    monkeypatch.setattr(registry, "cancel", lambda eid: order.append("record") or real_cancel(eid))
    return order, store


def test_the_workflow_cancel_ends_the_durable_job_before_the_record(monkeypatch):
    # Record first let a re-dispatch land between the two with its durable
    # job still live: nothing would refuse it, and no cancel would reach it.
    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job = SimpleNamespace(job_id="job-1")
    order, _store = _cancel_order(monkeypatch, registry, job)
    monkeypatch.setattr(wrh, "_zmq_execution_access", lambda *_a: (record, job, None))
    reply = json.loads(wrh._handle_workflow_cancel("alice", '{"execution_id": "wf-attach"}'))
    assert reply["status"] == "ok" and order == ["durable job", "record"]


def test_the_http_workflow_cancel_ends_the_durable_job_before_the_record(monkeypatch):
    import asyncio

    from umrs.api import workflows as api_workflows

    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job = SimpleNamespace(job_id="job-1", status=JobStatus.RUNNING)
    order, store = _cancel_order(monkeypatch, registry, job)
    monkeypatch.setattr(api_workflows, "get_optional_job_store", lambda: store)
    monkeypatch.setattr(
        api_workflows, "resolve_execution_context", lambda *_a, **_k: (record, None)
    )
    monkeypatch.setattr(api_workflows, "enforce_http_execution_access", lambda *_a, **_k: None)
    monkeypatch.setattr(api_workflows, "_execution_payload", lambda rec, _job: rec.to_dict())
    reply = asyncio.run(api_workflows.cancel_execution("wf-attach", identity=None))
    assert reply["execution"]["status"] == "cancelled" and order == ["durable job", "record"]


def test_the_loop_ports_cancel_ends_the_durable_job_before_the_record(monkeypatch):
    from umrs.knowledge.umrs_workflow_port import UmrsWorkflowExecutionPort

    registry = WorkflowExecutionRegistry()
    registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job = SimpleNamespace(job_id="job-1", spec={"execution_id": "wf-attach"},
                          job_type=JobType.WORKFLOW_RUN)  # fmt: skip
    order, _store = _cancel_order(monkeypatch, registry, job)
    port_store = SimpleNamespace(get_job_with_attempts=lambda _id: (job, []),
                                 cancel_job=lambda _id: order.append("durable job"))  # fmt: skip
    port = UmrsWorkflowExecutionPort(port_store)
    port._jobs["wf-attach"] = "job-1"
    assert port.cancel("wf-attach")["status"] == "cancelled"
    assert order == ["durable job", "record"]


# ── the execution-id cancels after peer review of F2 (2026-09-26) ──────────

_CANCEL = '{"execution_id": "wf-attach"}'


def _http_cancel(monkeypatch, record, job):
    """The HTTP cancel_execution route with its lookup answering *record*
    and *job*."""
    import asyncio

    from umrs.api import workflows as api_workflows

    monkeypatch.setattr(api_workflows, "resolve_execution_context", lambda *_a, **_k: (record, job))
    monkeypatch.setattr(api_workflows, "enforce_http_execution_access", lambda *_a, **_k: None)
    monkeypatch.setattr(api_workflows, "_execution_payload", lambda rec, _job: rec.to_dict())
    return asyncio.run(api_workflows.cancel_execution("wf-attach", identity=None))


def test_the_workflow_cancel_reaches_a_record_registered_after_its_lookup(monkeypatch):
    # The lookup found no record (a first dispatch not yet registered); the
    # dispatch registered it and read its durable job still running before
    # the store cancel. Gating the registry cancel on the lookup left the run
    # going after the user was told it was cancelled.
    registry = WorkflowExecutionRegistry()
    job = SimpleNamespace(job_id="job-1")
    order, _store = _cancel_order(monkeypatch, registry, job)
    monkeypatch.setattr(wrh, "_zmq_execution_access", lambda *_a: (None, job, None))
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    reply = json.loads(wrh._handle_workflow_cancel("alice", _CANCEL))
    assert order == ["durable job", "record"] and record.cancel_requested
    assert reply["execution"]["status"] == "cancelled"


def test_the_http_workflow_cancel_reaches_a_record_registered_after_its_lookup(monkeypatch):
    registry = WorkflowExecutionRegistry()
    job = SimpleNamespace(job_id="job-1", status=JobStatus.RUNNING)
    order, _store = _cancel_order(monkeypatch, registry, job)
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    reply = _http_cancel(monkeypatch, None, job)
    assert order == ["durable job", "record"] and record.cancel_requested
    assert reply["execution"]["status"] == "cancelled"


def test_an_unreadable_job_after_the_durable_cancel_never_skips_the_registry(monkeypatch):
    # The reply's re-read sat between the committed durable cancel and the
    # registry cancel: a pool error there left the run going unflagged.
    def unreadable(_job_id):
        raise RuntimeError("connection lost")

    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job = SimpleNamespace(job_id="job-1")
    order, store = _cancel_order(monkeypatch, registry, job)
    store.get_job = unreadable
    monkeypatch.setattr(wrh, "_zmq_execution_access", lambda *_a: (record, job, None))
    reply = json.loads(wrh._handle_workflow_cancel("alice", _CANCEL))
    assert order == ["durable job", "record"] and record.cancel_requested
    assert reply["status"] == "ok"


def test_a_durable_cancel_that_failed_leaves_the_run_alone(monkeypatch):
    # A store error other than an ended job means the durable job lives on,
    # and a registry-only cancel is a rerun (the monitor requeues the failed
    # attempt): the cancel fails as a whole and the user may send it again.
    import pytest
    from umrs.jobs.store import JobStoreError

    def lock_held(*_a, **_k):
        raise JobStoreError("Could not acquire lock for job job-1")

    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job = SimpleNamespace(job_id="job-1", status=JobStatus.RUNNING)
    order, store = _cancel_order(monkeypatch, registry, job)
    store.cancel_job = lock_held
    monkeypatch.setattr(wrh, "_zmq_execution_access", lambda *_a: (record, job, None))
    with pytest.raises(JobStoreError):
        wrh._handle_workflow_cancel("alice", _CANCEL)
    with pytest.raises(JobStoreError):
        _http_cancel(monkeypatch, record, job)
    assert order == [] and not record.cancel_requested


def test_the_workflow_cancel_finds_a_job_past_the_execution_id_scan(monkeypatch):
    # The execution-id lookup scans only the newest workflow_run jobs: an
    # older run's durable job stayed queued for its retry, and the retry ran
    # the cancelled workflow again.
    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    order, _store = _cancel_order(monkeypatch, registry, SimpleNamespace(job_id="job-1"))
    monkeypatch.setattr(wrh, "_zmq_execution_access", lambda *_a: (record, None, None))
    json.loads(wrh._handle_workflow_cancel("alice", _CANCEL))
    assert order == ["durable job", "record"]


def test_without_a_sidecar_store_the_workflow_cancel_uses_the_runtimes(monkeypatch):
    # With no sidecar store the durable job was neither found nor cancelled:
    # the registry-only cancel failed the attempt and the monitor requeued it.
    from umrs.api import workflow_auth
    from umrs.server import durable_job_runtime

    registry = WorkflowExecutionRegistry()
    record = registry.create("w", "alice", execution_id="wf-attach", durable_job_id="job-1")
    job, cancelled = SimpleNamespace(job_id="job-1", spec={"execution_id": "wf-attach"}), []
    store = SimpleNamespace(list_jobs=lambda **_k: [job], get_job=lambda _id: job,
                            cancel_job=lambda job_id, **_k: cancelled.append(job_id))  # fmt: skip
    monkeypatch.setattr(api_jobs, "get_optional_job_store", lambda: None)
    monkeypatch.setattr(durable_job_runtime, "_global_runtime", SimpleNamespace(job_store=store))
    monkeypatch.setattr(
        "umrs.server.workflow_execution_registry.get_execution_registry", lambda: registry
    )
    monkeypatch.setattr(workflow_auth, "check_zmq_execution_access", lambda *_a, **_k: None)
    json.loads(wrh._handle_workflow_cancel("alice", _CANCEL))
    assert cancelled == ["job-1"] and record.cancel_requested


# ── the job store keeps a cancel a cancel (peer review F2, 2026-09-26) ─────


class _Conn:
    """A psycopg2 connection: each execute answers the next of *answers*
    (fetchone returns it); commits and rollbacks are recorded."""

    def __init__(self, answers):
        self.answers, self.sql, self.ended = list(answers), [], []

    def cursor(self, cursor_factory=None):
        return _Cursor(self)

    def commit(self):
        self.ended.append("commit")

    def rollback(self):
        self.ended.append("rollback")


class _Cursor:
    def __init__(self, conn):
        self.conn, self.row = conn, None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def execute(self, sql, params=None):
        self.conn.sql.append((" ".join(sql.split()), params))
        self.row = self.conn.answers.pop(0) if self.conn.answers else None

    def fetchone(self):
        return self.row


def _store(*answers):
    from umrs.jobs.store import JobStore

    conn = _Conn(answers)
    store = JobStore.__new__(JobStore)  # no schema bootstrap
    store.db_pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    store.umrs_instance_id = "umrs-test"
    return store, conn


#: create_attempt's answers: the lock, the attempt count, the INSERT, then the
#: job-state UPDATE's RETURNING row (None: the job is no longer startable).
def _attempt_answers(job_row):
    return ((True,), (0,), None, job_row)


def test_an_attempt_never_starts_a_job_cancelled_after_its_claim():
    # A cancel between claim_next_jobs and create_attempt was overwritten:
    # the unguarded UPDATE set the cancelled job RUNNING, and the dispatch's
    # durable_job_cancelled check then read it running.
    import pytest
    from umrs.jobs.store import InvalidTransitionError

    store, conn = _store(*_attempt_answers(None))
    with pytest.raises(InvalidTransitionError):
        store.create_attempt("job-1", backend="workflow-run")
    assert conn.ended == ["rollback"]  # no attempt row survives
    update = next(sql for sql, _p in conn.sql if sql.startswith("UPDATE jobs"))
    assert "AND status IN (%s, %s) RETURNING job_id" in update
    started, _conn = _store(*_attempt_answers(("job-1",)))
    assert started.create_attempt("job-1", backend="workflow-run").startswith("att")


def test_the_scheduler_never_submits_a_job_cancelled_after_its_claim():
    import asyncio

    from umrs.jobs.scheduler import Scheduler

    store, _conn = _store(*_attempt_answers(None))
    submitted = []

    async def submit_job(job, attempt):
        submitted.append(job.job_id)
        return "ref"

    backend = SimpleNamespace(backend_name="workflow-run", submit_job=submit_job)
    store.get_attempt = lambda _id: SimpleNamespace(attempt_id=_id)
    store.update_attempt_status_detail = store.mark_attempt_submitted = lambda *_a: None
    scheduler = Scheduler(store, backend_factory=lambda _t: backend)
    job = SimpleNamespace(job_id="job-1", job_type=JobType.WORKFLOW_RUN, next_run_at=None)
    asyncio.run(scheduler._schedule_job("job-1", job))
    assert submitted == []


def test_a_cancel_waits_out_a_short_hold_of_the_jobs_lock(monkeypatch):
    # The lock was try-once: a cancel landing while the scheduler's
    # create_attempt held it failed, exactly while the run was dispatched.
    import pytest
    from umrs.jobs import store as store_mod
    from umrs.jobs.store import JobStoreError

    monkeypatch.setattr(store_mod.time, "sleep", lambda _s: None)
    store, conn = _store((False,), (False,), (True,), ("job-1",), None, None)
    store.cancel_job("job-1", reason="r")
    assert conn.ended == ["commit"]
    held, _conn = _store(*[(False,)] * store_mod._CANCEL_LOCK_TRIES)
    with pytest.raises(JobStoreError, match="Could not acquire lock"):
        held.cancel_job("job-1", reason="r")
