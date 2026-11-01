"""ADR-105 S3 (umrs half): plan -> commit for workflow runs (D-4).

- A plan is a pure computation, never a row: ``POST .../plan`` resolves and
  signs (HMAC over the canonical signed dict), writes no job and no catalog.
- ``POST .../commit`` verifies the digest, the expiry, the caller and the
  workflow, re-resolves (a changed version/lane/target is ``plan_stale``),
  checks ``auto`` against the project's threshold (fail closed), and runs
  the trigger path with the plan id as the idempotency key: a second commit
  answers the run already started.
- ``project_budgets.require_plan`` (migration 0131) refuses trigger,
  run-definition trigger and ZMQ ``RUN_WORKFLOW``; commit is the way in.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs.models import Job, JobStatus, RestartPolicy
from umrs.jobs.store import DuplicateJobError
from umrs.workflow_plans import (
    PLAN_TTL_SECONDS,
    SIGNED_FIELDS,
    PlanPolicy,
    plan_expired,
    quote_workflow,
    sign_plan,
    verify_plan,
)

from r4_contracts.handshake_hmac import sign_claims
from umrs import workflow_plans

SECRET = "plan-test-handshake"
CODE = "x-platform-error-code"
_STEP_ENV = (
    "UMRS_STEP_EXECUTOR",
    "UMRS_SLURM_STEP_SUBMIT",
    "UMRS_STEP_GPU",
    "UMRS_SLURM_GPUS",
    "UMRS_SLURM_PARTITION",
    "UMRS_SLURM_ACCOUNT",
    "UMRS_SLURM_TIME",
    "UMRS_STEP_EXECUTOR_TIMEOUT",
    "UMRS_STEP_REMOTE_ENABLED",
    "UMRS_STEP_UPLOADED_CODE",
    "UMRS_AGENT_PROCESSES_PER_USER_MAX",
)


# ── fakes: a job store with idempotency keys, a pool over project_budgets ──


class _Cursor:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool
        self._row: Any = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self._pool.reads.append(sql)
        if "project_budgets" in sql:
            self._row = self._pool.policies.get(params[0])
        else:  # agent_quota.count_active_agent_units
            self._row = (0,)

    def fetchone(self) -> Any:
        return self._row


class _Conn:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def cursor(self) -> _Cursor:
        if self._pool.broken:
            raise RuntimeError("database is down")
        return _Cursor(self._pool)


class _Pool:
    def __init__(self) -> None:
        self.policies: dict[str, tuple] = {}
        self.reads: list[str] = []
        self.broken = False

    def getconn(self) -> _Conn:
        return _Conn(self)

    def putconn(self, _conn: Any) -> None:
        return None


class _Store:
    """``create_job`` with the (project_id, idempotency_key) uniqueness of
    the real store: a duplicate key raises ``DuplicateJobError``."""

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.db_pool = _Pool()

    def create_job(self, project_id, user_id, job_type, spec, **options):
        key = options.get("idempotency_key")
        for job in self.jobs.values():
            if key and job.project_id == project_id and job.idempotency_key == key:
                raise DuplicateJobError(job.job_id)
        job_id = f"job-{len(self.jobs) + 1:03d}"
        self.jobs[job_id] = Job(
            job_id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=key or f"idem-{job_id}",
            status=JobStatus.PLANNED,
            priority=0,
            desired_max_attempts=options.get("max_attempts", 3),
            restart_policy=options.get("restart_policy", RestartPolicy.ON_CRASH),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        return job_id

    def enqueue_job(self, job_id):
        self.jobs[job_id].status = JobStatus.QUEUED

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_attempts_for_job(self, _job_id):
        return []


# ── fixtures ──────────────────────────────────────────────────────────────

_WORKFLOW = {
    "name": "relax",
    "version": "1.0",
    "steps": [
        {
            "name": "s1",
            "entrypoint": "m:S1",
            "timeout_seconds": 1800,
            "routing_hints": {"gpu": True},
        },
        {"name": "s2", "entrypoint": "m:S2", "timeout_seconds": 600, "max_retries": 1},
    ],
    "parameter_schema": {
        "type": "object",
        "properties": {"steps": {"type": "integer"}},
        "additionalProperties": False,
    },
}


def _step_env_clean(monkeypatch) -> None:
    for name in _STEP_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("umrs.shared.uploaded_code.load_site_env", lambda: None)
    monkeypatch.setattr("umrs.server.step_trust._load_site_env", lambda: None)


def _register(tmp_path: Path, *, version: str = "1.0", scope: str = "user") -> None:
    from umrs.server.workflow_registry_handler import register_workflow

    workflow_dir = tmp_path / "relax"
    workflow_dir.mkdir(exist_ok=True)
    (workflow_dir / "workflow.json").write_text(json.dumps({**_WORKFLOW, "version": version}))
    register_workflow(
        workflow_name="relax",
        version=version,
        user_id="alice",
        source_path=str(workflow_dir),
        workflow_json=_WORKFLOW,
        scope=scope,
    )


@pytest.fixture
def wf(monkeypatch, tmp_path):
    from umrs.api.workflows import router
    from umrs.server.workflow_execution_registry import reset_execution_registry_for_tests

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("UMRS_SECRET_KEY", "plan-signing-secret")
    _step_env_clean(monkeypatch)
    reset_execution_registry_for_tests()
    store = _Store()
    with (
        patch(
            "umrs.server.workflow_registry_handler._catalog_path",
            return_value=str(tmp_path / "workflow_catalog.json"),
        ),
        patch("umrs.api.workflows.get_optional_job_store", return_value=store),
    ):
        _register(tmp_path)
        app = FastAPI()
        app.include_router(router)
        yield TestClient(app), store, tmp_path


def _h(user: str = "alice", project: str = "proj-a", **extra: str) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
        **extra,
    }


def _plan(client: TestClient, params: dict | None = None, **headers: str) -> dict:
    r = client.post(
        "/umrs/workflows/relax/plan",
        json={"parameters": {"steps": 3} if params is None else params},
        headers=_h(**headers),
    )
    assert r.status_code == 200, r.text
    return r.json()


def _commit(client: TestClient, plan: dict, *, auto: bool = False, **headers: str):
    body = {"signed": plan["signed"], "digest": plan["digest"], "auto": auto}
    return client.post("/umrs/workflows/relax/commit", json=body, headers=_h(**headers))


# ── signing (pure) ────────────────────────────────────────────────────────

_FIELDS = {
    "project_id": "proj-a",
    "user_id": "alice",
    "workflow_name": "relax",
    "version": "1.0",
    "parameters": {"steps": 3},
    "input_files": None,
    "lane": "inproc",
    "partition": None,
    "account": None,
    "gpu_steps": 1,
    "max_wall_seconds": 3000,
}


def test_sign_and_verify_round_trip(monkeypatch):
    monkeypatch.setenv("UMRS_SECRET_KEY", "k1")
    signed, digest, plan_id = sign_plan(_FIELDS, now=1_000.0)
    assert list(signed) == list(SIGNED_FIELDS)
    assert signed["expires_at"] == "1970-01-01T00:31:40Z"  # 1000 + 900 s
    assert plan_id == "plan-" + digest[:24] and len(digest) == 64
    assert verify_plan(json.loads(json.dumps(signed)), digest)


@pytest.mark.parametrize("field", SIGNED_FIELDS)
def test_tampering_any_signed_field_breaks_the_digest(monkeypatch, field):
    monkeypatch.setenv("UMRS_SECRET_KEY", "k1")
    signed, digest, _ = sign_plan(_FIELDS)
    tampered = {**signed, field: "tampered"}
    assert not verify_plan(tampered, digest)


def test_another_key_does_not_verify(monkeypatch):
    monkeypatch.setenv("UMRS_SECRET_KEY", "k1")
    signed, digest, _ = sign_plan(_FIELDS)
    monkeypatch.setenv("UMRS_SECRET_KEY", "k2")
    assert not verify_plan(signed, digest)
    assert not verify_plan(signed, 123) and not verify_plan("x", digest)


def test_without_secret_a_per_process_key_signs(monkeypatch):
    monkeypatch.delenv("UMRS_SECRET_KEY", raising=False)
    monkeypatch.setattr(workflow_plans, "_process_key", None)
    signed, digest, _ = sign_plan(_FIELDS)
    assert verify_plan(signed, digest)
    monkeypatch.setenv("UMRS_SECRET_KEY", "k1")
    assert not verify_plan(signed, digest)


def test_expiry():
    signed, _, _ = sign_plan(_FIELDS, now=1_000.0)
    assert not plan_expired(signed, now=1_000.0 + PLAN_TTL_SECONDS - 1)
    assert plan_expired(signed, now=1_000.0 + PLAN_TTL_SECONDS)
    assert plan_expired({**signed, "expires_at": "soon"})


def test_auto_commit_policy_fails_closed():
    assert not PlanPolicy().auto_commit_allows(0.0)
    assert PlanPolicy(auto_commit_max_gpu_hours=1.0).auto_commit_allows(1.0)
    assert not PlanPolicy(auto_commit_max_gpu_hours=1.0).auto_commit_allows(1.01)


# ── quote (pure) ──────────────────────────────────────────────────────────


def test_quote_in_process_lane(monkeypatch, tmp_path):
    _step_env_clean(monkeypatch)
    quote = quote_workflow(_WORKFLOW, str(tmp_path))
    assert (quote.lane, quote.partition, quote.account) == ("inproc", None, None)
    assert [(s.name, s.gpu, s.attempt_seconds, s.attempts) for s in quote.steps] == [
        ("s1", True, 1800, 1),
        ("s2", False, 600, 2),
    ]
    assert quote.gpu_steps == 1
    assert quote.max_wall_seconds == 1800 + 1200
    assert quote.max_gpu_hours == 0.5


def test_quote_local_slurm_lane_uses_the_sbatch_time(monkeypatch, tmp_path):
    _step_env_clean(monkeypatch)
    monkeypatch.setenv("UMRS_SLURM_PARTITION", "ng-dgx-m3")
    monkeypatch.setenv("UMRS_SLURM_ACCOUNT", "acct")
    monkeypatch.setenv("UMRS_SLURM_GPUS", "2")
    monkeypatch.setenv("UMRS_STEP_EXECUTOR_TIMEOUT", "7200")
    quote = quote_workflow({**_WORKFLOW, "execution_mode": "slurm"}, str(tmp_path))
    assert (quote.lane, quote.partition, quote.account) == ("slurm", "ng-dgx-m3", "acct")
    assert [s.attempt_seconds for s in quote.steps] == [7200, 7200]
    assert quote.max_wall_seconds == 7200 * 3
    assert quote.max_gpu_hours == 4.0  # one GPU step, 2 GPUs, 2 h


def test_quote_remote_lane_needs_the_site_gpu_switch(monkeypatch, tmp_path):
    _step_env_clean(monkeypatch)
    monkeypatch.setenv("UMRS_SLURM_STEP_SUBMIT", "remote")
    monkeypatch.setenv("UMRS_SLURM_TIME", "02:00:00")
    workflow = {**_WORKFLOW, "execution_mode": "slurm"}
    quote = quote_workflow(workflow, str(tmp_path))
    assert quote.gpu_steps == 0  # UMRS_STEP_GPU off: no GPU even with the hint
    assert [s.attempt_seconds for s in quote.steps] == [1800, 600]  # step timeouts
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    assert quote_workflow(workflow, str(tmp_path)).gpu_steps == 1


def test_quote_refuses_uploaded_code_without_an_hpc_lane(monkeypatch, tmp_path):
    _step_env_clean(monkeypatch)
    (tmp_path / "m.py").write_text("")  # makes m:S1 uploaded code
    quote = quote_workflow(_WORKFLOW, str(tmp_path))
    assert quote.lane is None and "HPC lane" in quote.refusal


# ── plan route ────────────────────────────────────────────────────────────


def test_plan_signs_the_resolved_run(wf):
    client, store, _ = wf
    plan = _plan(client)
    assert plan["plan_id"] == "plan-" + plan["digest"][:24]
    assert verify_plan(plan["signed"], plan["digest"])
    signed = {k: plan["signed"][k] for k in ("project_id", "user_id", "parameters")}
    assert signed == {"project_id": "proj-a", "user_id": "alice", "parameters": {"steps": 3}}
    assert plan["expires_at"] == plan["signed"]["expires_at"]


def test_plan_reports_steps_bounds_quota_and_policy(wf):
    client, store, _ = wf
    plan = _plan(client)
    summary = {k: plan[k] for k in ("lane", "gpu_steps", "max_wall_seconds", "max_gpu_hours")}
    assert summary == {
        "lane": "inproc",
        "gpu_steps": 1,
        "max_wall_seconds": 3000,
        "max_gpu_hours": 0.5,
    }
    assert [s["name"] for s in plan["steps"]] == ["s1", "s2"]
    assert plan["quota"] == {"active_units": 0, "max_units": 4}
    policy = {
        k: plan[k] for k in ("require_plan", "auto_commit_allowed", "auto_commit_max_gpu_hours")
    }
    assert policy == {
        "require_plan": False,
        "auto_commit_allowed": False,
        "auto_commit_max_gpu_hours": None,
    }


def test_plan_has_no_side_effects(wf):
    client, store, tmp_path = wf
    catalog = (tmp_path / "workflow_catalog.json").read_bytes()
    with patch("umrs.api.workflows.handle_workflow_command") as run:
        _plan(client)
    run.assert_not_called()
    assert store.jobs == {}
    assert (tmp_path / "workflow_catalog.json").read_bytes() == catalog


def test_plan_refuses_bad_params_like_trigger(wf):
    client, store, _ = wf
    plan = client.post(
        "/umrs/workflows/relax/plan", json={"parameters": {"steps": "ten"}}, headers=_h()
    )
    trigger = client.post(
        "/umrs/workflows/relax/trigger", json={"parameters": {"steps": "ten"}}, headers=_h()
    )
    assert plan.status_code == trigger.status_code == 400
    assert plan.json() == trigger.json()
    assert plan.json()["detail"]["code"] == "workflow_parameters_invalid"
    assert store.jobs == {}


def test_plan_reports_the_project_policy(wf):
    client, store, _ = wf
    store.db_pool.policies["proj-a"] = (1.0, True)
    plan = _plan(client)
    assert plan["auto_commit_max_gpu_hours"] == 1.0
    assert plan["require_plan"] is True and plan["auto_commit_allowed"] is True


# ── commit route ──────────────────────────────────────────────────────────


def test_commit_starts_the_run_with_the_plan_id_as_key(wf):
    client, store, _ = wf
    plan = _plan(client)
    r = _commit(client, plan, **{"x-r4-session-id": "sess-9"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["plan_id"] == plan["plan_id"] and "existing_run" not in body
    (job,) = store.jobs.values()
    assert job.idempotency_key == plan["plan_id"]
    assert job.spec["execution_id"] == body["execution_id"]
    assert job.spec["params"] == {"steps": 3, "project_id": "proj-a"}
    assert job.spec["origin"] == {"session_id": "sess-9"}


def test_a_second_commit_returns_the_existing_run(wf):
    client, store, _ = wf
    plan = _plan(client)
    first = _commit(client, plan).json()
    second = _commit(client, plan)
    assert second.status_code == 200, second.text
    assert second.json()["execution_id"] == first["execution_id"]
    assert second.json()["durable_job_id"] == first["durable_job_id"]
    assert second.json()["existing_run"] is True
    assert len(store.jobs) == 1


def test_a_tampered_plan_is_refused(wf):
    client, store, _ = wf
    plan = _plan(client)
    plan["signed"]["parameters"] = {"steps": 300}
    r = _commit(client, plan)
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.plan_invalid")
    assert store.jobs == {}


def test_an_expired_plan_is_refused(wf):
    client, store, _ = wf
    plan = _plan(client)
    fields = {k: plan["signed"][k] for k in SIGNED_FIELDS[:-1]}
    signed, digest, _ = sign_plan(fields, now=time.time() - PLAN_TTL_SECONDS - 1)
    r = _commit(client, {"signed": signed, "digest": digest})
    assert (r.status_code, r.headers[CODE]) == (409, "workflows.plan_expired")
    assert store.jobs == {}


def test_another_user_cannot_commit_the_plan(wf):
    client, store, tmp_path = wf
    _register(tmp_path, scope="system")  # bob can resolve the workflow
    plan = _plan(client)
    r = _commit(client, plan, user="bob")
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.plan_mismatch")
    assert store.jobs == {}


def test_another_project_cannot_commit_the_plan(wf):
    client, store, _ = wf
    plan = _plan(client)
    r = _commit(client, plan, project="proj-b")
    assert r.status_code == 403
    assert store.jobs == {}


def test_a_plan_commits_only_its_own_workflow(wf):
    client, store, _ = wf
    plan = _plan(client)
    body = {"signed": plan["signed"], "digest": plan["digest"]}
    r = client.post("/umrs/workflows/other/commit", json=body, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.plan_mismatch")


def test_a_new_version_makes_the_plan_stale(wf):
    from umrs.server.workflow_registry_handler import unregister_workflow

    client, store, tmp_path = wf
    plan = _plan(client)
    unregister_workflow("relax", user_id="alice")
    _register(tmp_path, version="2.0")
    r = _commit(client, plan)
    assert (r.status_code, r.headers[CODE]) == (409, "workflows.plan_stale")
    assert "version" in r.json()["detail"]
    assert store.jobs == {}


def test_a_lane_change_makes_the_plan_stale(wf, monkeypatch):
    client, store, _ = wf
    plan = _plan(client)
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "subprocess")
    r = _commit(client, plan)
    assert (r.status_code, r.headers[CODE]) == (409, "workflows.plan_stale")
    assert "lane" in r.json()["detail"]


def test_auto_commit_without_a_threshold_is_refused(wf):
    client, store, _ = wf
    r = _commit(client, _plan(client), auto=True)
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.auto_commit_refused")
    assert "no auto-commit threshold" in r.json()["detail"]
    assert store.jobs == {}


def test_auto_commit_above_the_threshold_is_refused(wf):
    client, store, _ = wf
    store.db_pool.policies["proj-a"] = (0.25, False)
    r = _commit(client, _plan(client), auto=True)
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.auto_commit_refused")
    assert "threshold 0.25" in r.json()["detail"] and "0.5" in r.json()["detail"]
    assert store.jobs == {}


def test_auto_commit_within_the_threshold_runs(wf):
    client, store, _ = wf
    store.db_pool.policies["proj-a"] = (0.5, False)
    r = _commit(client, _plan(client), auto=True)
    assert r.status_code == 200, r.text
    assert len(store.jobs) == 1


def test_a_commit_needs_the_durable_store(wf):
    client, store, _ = wf
    plan = _plan(client)
    with patch("umrs.api.workflows.get_optional_job_store", return_value=None):
        r = _commit(client, plan)
    assert (r.status_code, r.headers[CODE]) == (503, "workflows.commit_unavailable")


# ── require_plan ──────────────────────────────────────────────────────────


def test_require_plan_refuses_trigger_but_not_commit(wf):
    client, store, _ = wf
    store.db_pool.policies["proj-a"] = (None, True)
    r = client.post("/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.plan_required")
    assert "/umrs/workflows/relax/plan" in r.json()["detail"]
    assert store.jobs == {}
    assert _commit(client, _plan(client)).status_code == 200
    assert len(store.jobs) == 1


def test_require_plan_refuses_a_run_definition_trigger(wf):
    client, store, _ = wf
    saved = client.put(
        "/umrs/workflows/run-definitions",
        json={"name": "nightly", "workflow_name": "relax", "params": {"steps": 1}},
        headers=_h(),
    )
    assert saved.status_code == 200, saved.text
    store.db_pool.policies["proj-a"] = (None, True)
    r = client.post("/umrs/workflows/run-definitions/nightly/trigger", headers=_h())
    assert (r.status_code, r.headers[CODE]) == (403, "workflows.plan_required")
    assert store.jobs == {}


def test_a_project_without_the_policy_triggers_as_before(wf):
    client, store, _ = wf
    store.db_pool.policies["proj-other"] = (None, True)
    r = client.post("/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=_h())
    assert r.status_code == 200, r.text


def test_an_unreadable_policy_fails_closed(wf):
    client, store, _ = wf
    store.db_pool.broken = True
    r = client.post("/umrs/workflows/relax/trigger", json={"parameters": {}}, headers=_h())
    assert (r.status_code, r.headers[CODE]) == (503, "workflows.policy_unavailable")
    assert store.jobs == {}


def test_require_plan_refuses_the_zmq_run_workflow(wf):
    from umrs.server.job_tracking_handler import DurableJobCommandBackend
    from umrs.server.workflow_registry_handler import (
        _handle_run_workflow,
        commit_workflow_run,
    )

    _client, store, _ = wf
    store.db_pool.policies["proj-a"] = (None, True)
    payload = json.dumps({"workflow_name": "relax", "params": {"project_id": "proj-a"}})
    backend = DurableJobCommandBackend(store)
    refused = json.loads(_handle_run_workflow("alice", payload, backend))
    assert refused["status"] == "error" and refused["code"] == "workflows.plan_required"
    assert store.jobs == {}
    # A ZMQ payload cannot carry the commit's key: only the server's commit does.
    forged = json.dumps({**json.loads(payload), "idempotency_key": "plan-x"})
    assert json.loads(_handle_run_workflow("alice", forged, backend))["status"] == "error"
    ok = json.loads(
        commit_workflow_run(
            "alice",
            payload,
            backend,
            trusted_identity=False,
            caller_is_admin=False,
            idempotency_key="plan-abc",
        )
    )
    assert ok["status"] == "ok" and len(store.jobs) == 1


# ── project budget: the two new fields ────────────────────────────────────


def _budget_client():
    from test_api_projects import _BudgetConn, _client
    from umrs.api.projects import ProjectStore

    store = ProjectStore(_BudgetConn())
    store.create("p-plan", owner_id="alice")
    return store, _client(store)


def test_budget_put_round_trips_the_plan_policy():
    from test_api_projects import _headers

    store, client = _budget_client()
    r = client.put(
        "/umrs/projects/p-plan/budget",
        json={"auto_commit_max_gpu_hours": 2.5, "require_plan": True},
        headers=_headers("alice"),
    )
    assert r.status_code == 200, r.text
    assert r.json()["budget"]["auto_commit_max_gpu_hours"] == 2.5
    assert r.json()["budget"]["require_plan"] is True
    got = client.get("/umrs/projects/p-plan/budget", headers=_headers("alice"))
    assert got.json()["budget"]["auto_commit_max_gpu_hours"] == 2.5
    assert got.json()["budget"]["require_plan"] is True


def test_budget_put_of_plan_fields_keeps_the_loop_caps_and_vice_versa():
    from test_api_projects import _headers

    store, client = _budget_client()
    alice = _headers("alice")
    client.put("/umrs/projects/p-plan/budget", json={"max_wall_clock_hours": 4.0}, headers=alice)
    client.put("/umrs/projects/p-plan/budget", json={"require_plan": True}, headers=alice)
    budget = store.get_budget("p-plan")
    assert budget["max_wall_clock_hours"] == 4.0 and budget["require_plan"] is True
    client.put("/umrs/projects/p-plan/budget", json={"max_gpu_hours": 1.0}, headers=alice)
    assert store.get_budget("p-plan")["require_plan"] is True
    client.put(
        "/umrs/projects/p-plan/budget",
        json={"auto_commit_max_gpu_hours": None},
        headers=alice,
    )
    assert store.get_budget("p-plan")["auto_commit_max_gpu_hours"] is None


def test_budget_put_refuses_a_negative_threshold_and_non_owners():
    from test_api_projects import _headers

    store, client = _budget_client()
    bad = client.put(
        "/umrs/projects/p-plan/budget",
        json={"auto_commit_max_gpu_hours": -1},
        headers=_headers("alice"),
    )
    assert bad.status_code == 422
    other = client.put(
        "/umrs/projects/p-plan/budget", json={"require_plan": True}, headers=_headers("bob")
    )
    assert other.status_code in (403, 404)
    assert store.get_budget("p-plan")["require_plan"] is False


def test_a_project_without_a_budget_row_has_no_policy():
    from test_api_projects import _BudgetConn
    from umrs.api.projects import ProjectStore

    budget = ProjectStore(_BudgetConn()).get_budget("nope")
    assert budget["auto_commit_max_gpu_hours"] is None and budget["require_plan"] is False
    assert workflow_plans.read_project_plan_policy(_Pool(), "nope") == PlanPolicy()
    assert workflow_plans.read_project_plan_policy(None, "p") == PlanPolicy()


# ── migration 0131 ────────────────────────────────────────────────────────


def _sql_code(path: Path) -> str:
    """The migration's statements on one line, comments dropped."""
    lines = [line.strip() for line in path.read_text().splitlines()]
    return " ".join(line for line in lines if not line.startswith("--"))


def test_0131_is_numbered_after_0130():
    from umrs import migrations

    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0131") == 1
    assert versions.index("0131") == versions.index("0130") + 1


def test_0131_adds_the_plan_policy_columns():
    from umrs import migrations

    code = _sql_code(migrations.MIGRATIONS_DIR / "0131_project_plan_policy.sql")
    assert (
        "ALTER TABLE project_budgets ADD COLUMN IF NOT EXISTS "
        "auto_commit_max_gpu_hours DOUBLE PRECISION NULL" in code
    )
    assert (
        "ALTER TABLE project_budgets ADD COLUMN IF NOT EXISTS "
        "require_plan BOOLEAN NOT NULL DEFAULT false" in code
    )
    assert "CONCURRENTLY" not in code.upper()
    assert workflow_plans._POLICY_SQL.count("auto_commit_max_gpu_hours, require_plan") == 1


# ── D-1: explicit, unique operation ids ───────────────────────────────────

PLAN_OPERATIONS = {
    "workflow_plan": ("post", "/umrs/workflows/{workflow_name}/plan"),
    "workflow_commit": ("post", "/umrs/workflows/{workflow_name}/commit"),
}


@pytest.fixture(scope="module")
def openapi_spec() -> dict:
    from umrs.server import http_sidecar

    original = http_sidecar._build_db_pool
    http_sidecar._build_db_pool = lambda _app: None
    try:
        return http_sidecar._create_app().openapi()
    finally:
        http_sidecar._build_db_pool = original


def _operation_ids(spec: dict) -> Counter:
    ops = [op for path_ops in spec["paths"].values() for op in path_ops.values()]
    return Counter(op.get("operationId") for op in ops if isinstance(op, dict))


@pytest.mark.parametrize("operation_id", sorted(PLAN_OPERATIONS))
def test_plan_and_commit_operation_ids_are_explicit_and_unique(openapi_spec, operation_id):
    method, path = PLAN_OPERATIONS[operation_id]
    assert _operation_ids(openapi_spec)[operation_id] == 1
    op = openapi_spec["paths"][path][method]
    assert op["operationId"] == operation_id
    assert op["responses"]["200"]["content"]["application/json"]["schema"]
    assert op["requestBody"]["content"]["application/json"]["schema"]
