"""D-37 scheduler leg pins: the hpc_script durable job kind (ROADMAP 3.3)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs.hpc_script import (
    HpcScriptBackend,
    HpcScriptStatusChecker,
    _sbatch_command,
    validate_hpc_script_spec,
)
from umrs.jobs.models import JobType
from umrs.orchestrator.hpc_identity import HpcAccessRefused
from umrs.orchestrator.site_config import HpcAccessConfig, ServiceAccountAccess
from umrs.shared import hpc_user_work_access as hpc_user_work


def _site(allowed=("p",)):
    """A site declaring the personal service account, *allowed* projects listed."""
    hpc = HpcAccessConfig(
        modes_allowed=["service_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(
            login="operator", personal=True, allowed_projects=list(allowed)
        ),
    )
    return SimpleNamespace(
        name="cluster-a",
        extra={"ssh_user": "operator"},
        scheduler=SimpleNamespace(hpc_access=hpc),
    )


@pytest.fixture(autouse=True)
def declared_site(monkeypatch):
    """This server's site lists project ``p``; nobody is an admin."""
    monkeypatch.setattr(hpc_user_work, "current_site", lambda: _site())
    monkeypatch.setattr(hpc_user_work, "user_is_admin", lambda _user: False)
    return monkeypatch


class _Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    @property
    def ok(self):
        return self.returncode == 0


class _SSH:
    def __init__(self, results):
        self._results = list(results)
        self.commands = []

    async def run_remote(self, command, **_kw):
        self.commands.append(command)
        return self._results.pop(0)


def _job(spec, project="p"):
    return SimpleNamespace(
        job_id="j-1", attempt_id=None, user_id="u", project_id=project, spec=spec
    )


def _attempt():
    return SimpleNamespace(attempt_id="a-1", metrics={}, backend_ref="")


def _run(coro):
    return asyncio.run(coro)


def test_spec_validation():
    good = validate_hpc_script_spec({"script_path": "/lvs0/x/run.sh"})
    assert good["script_path"] == "/lvs0/x/run.sh"
    assert good["sbatch_args"] == []
    with pytest.raises(ValueError, match="script_path"):
        validate_hpc_script_spec({})
    with pytest.raises(ValueError, match="sbatch_args"):
        validate_hpc_script_spec({"script_path": "x", "sbatch_args": "not-a-list"})


def test_sbatch_command_quotes_never_interpolates():
    cmd = _sbatch_command("/lvs0/x/my script.sh", ["-p", "short; rm -rf /"])
    assert "'/lvs0/x/my script.sh'" in cmd
    assert "'short; rm -rf /'" in cmd  # quoted — never executed by the shell


def test_backend_submits_and_parses_slurm_id():
    ssh = _SSH([_Result("Submitted batch job 385046\n")])

    async def factory():
        return ssh

    backend = HpcScriptBackend(ssh=factory)
    attempt = _attempt()
    ref = _run(backend.submit_job(_job({"script_path": "/lvs0/x/run.sh"}), attempt))
    assert ref == "hpc_script|script=/lvs0/x/run.sh|slurm=385046"
    assert attempt.metrics["slurm_job_id"] == "385046"
    assert ssh.commands[0].startswith("sbatch ")


def test_backend_sbatch_failure_raises_named():
    ssh = _SSH([_Result(stderr="Batch job submission failed", returncode=1)])

    async def factory():
        return ssh

    backend = HpcScriptBackend(ssh=factory)
    with pytest.raises(RuntimeError, match="sbatch failed"):
        _run(backend.submit_job(_job({"script_path": "/x"}), _attempt()))


def test_backend_degrades_without_ssh():
    backend = HpcScriptBackend(ssh=None)
    ref = _run(backend.submit_job(_job({"script_path": "/x"}), _attempt()))
    assert "degraded=no_ssh" in ref


def test_checker_maps_squeue_running_then_sacct_completed():
    ssh = _SSH(
        [_Result("RUNNING\n"), _Result("COMPLETED\n")]  # running cycle, then terminal
    )

    async def factory():
        return ssh

    checker = HpcScriptStatusChecker(ssh_factory=factory)
    attempt = _attempt()
    attempt.backend_ref = "hpc_script|script=/x|slurm=99"
    running = _run(checker.check_status(attempt, _job({})))
    assert running.status == "running"
    assert running.status_detail == "slurm state RUNNING"
    done = _run(checker.check_status(attempt, _job({})))
    assert done.status == "succeeded"
    assert done.metrics["slurm_state"] == "COMPLETED"


def _checker_for(state):
    ssh = _SSH([_Result(""), _Result(state + "\n")])  # squeue empty → sacct

    async def factory():
        return ssh

    return HpcScriptStatusChecker(ssh_factory=factory)


def test_checker_failed_terminal_states():
    for state in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"):
        checker = _checker_for(state)
        attempt = _attempt()
        attempt.backend_ref = "hpc_script|script=/x|slurm=1"
        result = _run(checker.check_status(attempt, _job({})))
        assert result.status == "failed", state


def test_checker_refuses_degraded_and_unknown():
    checker = HpcScriptStatusChecker(ssh_factory=None)
    attempt = _attempt()
    attempt.backend_ref = "hpc_script|script=/x|degraded=no_ssh"
    assert _run(checker.check_status(attempt, _job({}))).status == "failed"
    attempt.backend_ref = "hpc_script|script=/x|slurm="
    assert _run(checker.check_status(attempt, _job({}))).status == "failed"


def test_job_kind_registered_everywhere():
    from umrs.jobs.models import DEFAULT_BACKOFFS

    assert JobType.HPC_SCRIPT.value == "hpc_script"
    assert JobType.HPC_SCRIPT in DEFAULT_BACKOFFS
    assert JobType.HPC_SCRIPT not in ()  # dispatchable: adapters registered it


# ── ADR-112 A.3.2: hpc_script is user work on the site's login ──────────────


def _factory(ssh):
    async def factory():
        return ssh

    return factory


def test_submit_refuses_an_unlisted_project_before_any_ssh():
    ssh = _SSH([_Result("Submitted batch job 1\n")])
    attempt = _attempt()
    with pytest.raises(HpcAccessRefused) as refused:
        _run(
            HpcScriptBackend(ssh=_factory(ssh)).submit_job(
                _job({"script_path": "/x"}, project="other"), attempt
            )
        )
    assert refused.value.code == "hpcaccess.project_not_allowed"
    assert ssh.commands == []
    assert "hpc_identity" not in attempt.metrics


def test_submit_records_the_identity_of_a_listed_project():
    ssh = _SSH([_Result("Submitted batch job 7\n")])
    attempt = _attempt()
    _run(HpcScriptBackend(ssh=_factory(ssh)).submit_job(_job({"script_path": "/x"}), attempt))
    identity = attempt.metrics["hpc_identity"]
    assert (identity["lane"], identity["access_mode"]) == ("user_work", "service_account")
    assert (identity["site_login"], identity["personal"]) == ("operator", True)


def test_submit_lets_an_admin_run_an_unlisted_project(declared_site):
    declared_site.setattr(hpc_user_work, "user_is_admin", lambda user: user == "u")
    ssh = _SSH([_Result("Submitted batch job 8\n")])
    attempt = _attempt()
    backend = HpcScriptBackend(ssh=_factory(ssh))
    _run(backend.submit_job(_job({"script_path": "/x"}, project="other"), attempt))
    assert attempt.metrics["slurm_job_id"] == "8"


def test_submit_on_an_undeclared_site_refuses_users(declared_site):
    declared_site.setattr(hpc_user_work, "current_site", lambda: None)
    with pytest.raises(HpcAccessRefused, match="mode_undeclared"):
        _run(HpcScriptBackend(ssh=None).submit_job(_job({"script_path": "/x"}), _attempt()))


def test_admin_comes_from_the_user_record_and_unreadable_is_not_admin(monkeypatch):
    from umrs.admin import user_store

    monkeypatch.undo()  # the real user_is_admin, with a fake store
    roles = {"ops": SimpleNamespace(role="admin", disabled=False)}

    class _Store:
        def resolve_user(self, user):
            if user == "broken":
                raise OSError("store down")
            if user not in roles:
                raise user_store.UserStoreError("absent")
            return roles[user]

    monkeypatch.setattr(user_store, "UserStore", _Store)
    assert hpc_user_work.user_is_admin("ops") is True
    assert hpc_user_work.user_is_admin("nobody") is False
    assert hpc_user_work.user_is_admin("broken") is False
    assert hpc_user_work.user_is_admin(None) is False


# ── creation: POST /umrs/jobs refuses before the job exists ─────────────────

_SECRET = "hpc-script-create-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(_SECRET, user, project, admin),
    }


class _Store:
    _get_conn = None
    site_name = ""

    def __init__(self):
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id):
        pass


def _create(monkeypatch, headers, *, fan_out=False):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", _SECRET)
    store = _Store()
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_job_store] = lambda: store
    spec = {"script_path": "/lvs0/x/run.sh"}
    # The body claims the listed project; only the verified header counts.
    body = {
        "project_id": "p",
        "user_id": headers["X-User-ID"],
        "job_type": "hpc_script",
        "spec": spec,
        "enqueue_immediately": False,
    }
    if fan_out:
        body.update(job_type="cleanup", spec={})
        body["fan_out_jobs"] = [{"name": "c", "job_type": "hpc_script", "spec": spec}]
    response = TestClient(app).post("/umrs/jobs", json=body, headers=headers)
    return response, store


def test_create_refuses_an_unlisted_project_with_the_typed_code(declared_site):
    response, store = _create(declared_site, _headers("alice", "other"))
    assert response.status_code == 403, response.text
    assert response.headers["x-platform-error-code"] == "hpcaccess.project_not_allowed"
    assert store.created == []


def test_create_refuses_a_fan_out_child(declared_site):
    response, store = _create(declared_site, _headers("alice", "other"), fan_out=True)
    assert response.status_code == 403, response.text
    assert store.created == []


def test_create_admits_a_listed_project_and_an_admin(declared_site):
    declared_site.setattr(hpc_user_work, "user_is_admin", lambda user: user == "ops")
    response, store = _create(declared_site, _headers("alice", "p"))
    assert response.status_code == 201, response.text
    response, store = _create(declared_site, _headers("ops", "other", admin=True))
    assert response.status_code == 201, response.text
    assert len(store.created) == 1


def test_an_admitted_admin_job_is_not_refused_at_submit(declared_site):
    """Review 2026-11-08: creation read the token's admin flag and submit the
    user record, so an admitted admin job could fail at submit. One source
    (the record) now decides both: the job created is the job submitted."""
    declared_site.setattr(hpc_user_work, "user_is_admin", lambda user: user == "ops")
    response, store = _create(declared_site, _headers("ops", "other", admin=True))
    assert response.status_code == 201, response.text
    created = store.created[0]
    job = SimpleNamespace(
        job_id="j-ops",
        user_id=created["user_id"],
        project_id=created["project_id"],
        spec=created["spec"],
    )
    ssh = _SSH([_Result("Submitted batch job 9\n")])
    attempt = _attempt()
    _run(HpcScriptBackend(ssh=_factory(ssh)).submit_job(job, attempt))
    assert attempt.metrics["slurm_job_id"] == "9"


def test_a_token_admin_without_an_admin_record_is_refused_at_creation(declared_site):
    """The reverse half: refused at the door (typed), not at submit."""
    response, store = _create(declared_site, _headers("ops", "other", admin=True))
    assert response.status_code == 403, response.text
    assert response.headers["x-platform-error-code"] == "hpcaccess.project_not_allowed"
    assert store.created == []


def _loop_body(venue: str) -> dict:
    return {
        "project_id": "p",
        "user_id": "alice",
        "job_type": "research_loop_job",
        "spec": {"loop_id": "loop-1", "driver": "engine", "venue": venue},
        "enqueue_immediately": False,
    }


def test_create_refuses_a_cluster_venue_loop_job_of_an_unlisted_project(declared_site):
    """Review 2026-11-08: a ``research_loop_job`` on a ``cluster:`` venue
    sent straight to POST /umrs/jobs skipped the creation check and was
    refused only at submit (and retried)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    declared_site.setenv("UMRS_AUTH_MODE", "strict")
    declared_site.setenv("UMRS_HANDSHAKE_HMAC_SECRET", _SECRET)
    store = _Store()
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_job_store] = lambda: store
    client = TestClient(app)
    refused = client.post(
        "/umrs/jobs", json=_loop_body("cluster:cluster-a"), headers=_headers("alice", "other")
    )
    assert refused.status_code == 403, refused.text
    assert refused.headers["x-platform-error-code"] == "hpcaccess.project_not_allowed"
    assert store.created == []
    frontend = client.post(
        "/umrs/jobs", json=_loop_body("frontend"), headers=_headers("alice", "other")
    )
    assert frontend.status_code != 403, frontend.text


# ── the ZMQ JOB_SUBMIT lane cannot create user work at all ─────────────────


@pytest.mark.parametrize(
    ("job_type", "spec"),
    [
        ("hpc_script", {"script_path": "/lvs0/x/run.sh"}),
        ("research_loop_job", {"loop_id": "l", "driver": "engine", "venue": "cluster:c"}),
    ],
)
def test_zmq_submit_never_creates_user_work_on_the_login(job_type, spec):
    """Review 2026-11-08 asked for a creation check on ZMQ JOB_SUBMIT; the
    lane admits only five job types, none of them user work on the login,
    so POST /umrs/jobs is the only creation path to check. Pin that."""
    import json

    from umrs.server.job_tracking_handler import DurableJobCommandBackend

    store = _Store()
    payload = {"job_type": job_type, "project_id": "other", "spec": spec}
    reply = json.loads(DurableJobCommandBackend(store).submit("alice", json.dumps(payload)))
    assert reply["status"] == "error" and "Invalid job_type" in reply["message"]
    assert store.created == []


# ── the scheduler fails a submit-time refusal once, terminally ──────────────


class _SchedStore:
    def __init__(self):
        self.failed: list[tuple] = []
        self.submitted: list = []

    def create_attempt(self, **_kw):
        return "a-1"

    def get_attempt(self, _attempt_id):
        return _attempt()

    def update_attempt_status_detail(self, *_a):
        pass

    def mark_attempt_failed(self, attempt_id, exit_reason, retry_delay=None):
        self.failed.append((attempt_id, exit_reason, retry_delay))
        return False

    def mark_attempt_submitted(self, *a):
        self.submitted.append(a)


def test_the_scheduler_fails_a_refused_job_terminally_without_degrading():
    """Review 2026-11-08: a submit-time ``HpcAccessRefused`` went through the
    generic path: retry in 30 s up to max_attempts and a retry-burst /
    degradation signal. A refusal is final (the resolver's contract)."""
    from umrs.jobs import scheduler as scheduler_mod
    from umrs.jobs.models import JobType as JT
    from umrs.jobs.store import _terminal_exit_reason

    store = _SchedStore()
    sched = scheduler_mod.Scheduler(store, lambda _t: HpcScriptBackend(ssh=None))
    job = SimpleNamespace(
        job_id="j-1",
        user_id="u",
        project_id="other",
        spec={"script_path": "/x"},
        job_type=JT.HPC_SCRIPT,
        next_run_at=None,
    )
    for _ in range(3):
        _run(sched._schedule_job("j-1", job))
    assert [f[2] for f in store.failed] == [None, None, None]  # never a retry delay
    reason = store.failed[0][1]
    assert reason.startswith("hpcaccess.project_not_allowed")
    assert _terminal_exit_reason("a-1", reason) is True
    assert getattr(sched, "_consecutive_failures", 0) == 0
    assert store.submitted == []


def test_a_generic_submit_failure_still_retries():
    from umrs.jobs.store import _terminal_exit_reason

    assert _terminal_exit_reason("a-1", "backend_submission_failed: hpcaccess.x") is False


class _LedgerRefusingBackend:
    backend_name = "hpc_script"

    def __init__(self, code: str) -> None:
        self.code = code

    async def submit_job(self, _job, _attempt):
        from umrs.hpc_accounts.usage_ledger import LedgerRefused

        raise LedgerRefused(self.code, "nothing was submitted")


@pytest.mark.parametrize("code", ["hpcaccess.budget_exceeded", "hpcaccess.ledger_unavailable"])
def test_a_ledger_refusal_is_terminal_without_degradation(code):
    from umrs.jobs import scheduler as scheduler_mod
    from umrs.jobs.store import _terminal_exit_reason

    store = _SchedStore()
    sched = scheduler_mod.Scheduler(store, lambda _t: _LedgerRefusingBackend(code))
    attempt = _attempt()
    job = SimpleNamespace()
    _run(sched._submit_attempt("j-1", job, ("a-1", attempt), _LedgerRefusingBackend(code)))
    [(attempt_id, reason, retry_delay)] = store.failed
    assert attempt_id == "a-1" and reason.startswith(code)
    assert retry_delay is None and _terminal_exit_reason(attempt_id, reason) is True
    assert getattr(sched, "_consecutive_failures", 0) == 0
    assert store.submitted == []


class _MetricsDownStore(_SchedStore):
    def update_attempt_metrics(self, *_a):
        raise RuntimeError("store down")


class _AcceptingBackend:
    backend_name = "hpc_script"

    async def submit_job(self, _job, attempt):
        attempt.metrics["hpc_identity"] = {"login": "operator"}
        return "4242"


def test_a_metrics_store_error_never_fails_an_accepted_attempt():
    """Review 2026-11-08: every hpc_script, endpoint and cluster-loop submit
    now writes its identity into the attempt metrics. Before wave A a store
    error there marked the accepted attempt failed and retried it (a second
    submit of a job the cluster already runs); in the first wave A draft it
    escaped and left the attempt unmarked. It is now logged and dropped."""
    from umrs.jobs import scheduler as scheduler_mod

    store = _MetricsDownStore()
    sched = scheduler_mod.Scheduler(store, lambda _t: _AcceptingBackend())
    attempt = _attempt()
    _run(sched._submit_attempt("j-1", SimpleNamespace(), ("a-1", attempt), _AcceptingBackend()))
    assert store.submitted == [("a-1", "4242")]
    assert store.failed == []


# ── the attempt record names the account the script charges (I-10) ─────────


@pytest.mark.parametrize(
    ("args", "account"),
    [
        ([], None),
        (["-A", "alloc-1"], "alloc-1"),
        (["-Aalloc-2"], "alloc-2"),
        (["--account", "alloc-3"], "alloc-3"),
        (["-p", "short", "--account=alloc-4"], "alloc-4"),
    ],
)
def test_the_identity_records_the_account_the_sbatch_args_charge(args, account):
    ssh = _SSH([_Result("Submitted batch job 10\n")])
    attempt = _attempt()
    spec = {"script_path": "/x", "sbatch_args": args}
    _run(HpcScriptBackend(ssh=_factory(ssh)).submit_job(_job(spec), attempt))
    assert attempt.metrics["hpc_identity"]["slurm_account"] == account
