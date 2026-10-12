"""A team plan run hosted on the server venue (ADR-092 slice 8, D-29, D-63).

UMRS hosts a plan run by starting one frontend-venue worker session (the
ADR-046 A2 path) whose command runs the r4 plan engine headless for that
run; the run's record is slice 7's (``research_loop_job``, ``hosting:
venue``), written only by that worker through the rail. Fakes as the venue
route tests use them: FakePodman (no live container), an in-memory broker
and registry, the sealed-source fake, and a job store with the attempt
state machine of the slice-7 tests. No live podman, no Postgres.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_companion_venue_routes import (
    READY_ENV,
    FakePodman,
    FakeSealed,
    _meta,
    _Pool,
)
from test_plan_run_records import _Caller, _FakeJobStore
from umrs.api import _rail_identity
from umrs.api import companion_venue as venue_api
from umrs.api import jobs as jobs_api
from umrs.api import plan_run_venue as prv_api
from umrs.api import traces as traces_api
from umrs.companion_trust.approval_inbox import ApprovalInboxStore
from umrs.jobs import plan_run_records as prr
from umrs.jobs.execution_trace_store import ExecutionTraceStore
from umrs.jobs.frontend_session_backend import (
    FrontendSessionBackend,
    FrontendSessionStatusChecker,
)
from umrs.jobs.models import JobStatus, JobType
from umrs.jobs.secret_broker import SecretBroker
from umrs.knowledge import tool_approvals
from umrs.venues import plan_runs
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

RUN = "tiny-0a1b2c3d"
OTHER_RUN = "other-9f8e7d6c"
PLAN = {
    "plan_version": 1,
    "name": "tiny",
    "description": "one agent",
    "budget": {"max_tokens": 1000, "max_wallclock_sec": 600},
    "phases": [
        {"id": "p1", "title": "P", "steps": [{"id": "s1", "kind": "agent", "prompt": "hi"}]}
    ],
    "_bound_params": {},
}
DEFINITION = {
    "plan": PLAN,
    "params": {},
    "manifest": {"run_id": RUN, "engine": "plan", "tool_capabilities": {}},
}
H = {"x-user-id": "alice"}


class VenueJobStore(_FakeJobStore):
    """The slice-7 fake plus the session-job verbs the venue service uses."""

    def __init__(self):
        super().__init__()
        self.enqueued: list[str] = []
        self.cancelled: list[tuple[str, str]] = []

    def create_job(self, project_id=None, user_id=None, job_type=None, spec=None, **kw):
        kw.setdefault("idempotency_key", f"k-{len(self.jobs)}")
        kw.setdefault("restart_policy", None)
        kw.setdefault("max_attempts", 1)
        return super().create_job(project_id, user_id, job_type, spec, **kw)

    def enqueue_job(self, job_id):
        self.enqueued.append(job_id)
        self.jobs[job_id].status = JobStatus.QUEUED

    def cancel_job(self, job_id, reason=""):
        job = self.jobs[job_id]
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            raise RuntimeError("already terminal")
        self.cancelled.append((job_id, reason))
        job.status, job.active_attempt_id = JobStatus.CANCELLED, None

    def stamp_spec_origin(self, _job_id, _origin):
        return True

    def find_job_id_by_spec_value(self, _job_type, _key, _value):
        return None

    def reopen_client_job(self, job_id):
        job = self.jobs[job_id]
        if not prr.is_plan_run_record(job.spec) or job.status == JobStatus.CANCELLED:
            raise prr.InvalidTransitionError("not reopenable")
        hosting = job.spec["hosting"]
        job.spec["hosting"] = "client"  # the parent's guard reads client only
        try:
            return super().reopen_client_job(job_id)
        finally:
            job.spec["hosting"] = hosting

    def session_jobs(self):
        return [j for j in self.jobs.values() if j.job_type == JobType.FRONTEND_SESSION]

    def run_job(self, run_id=RUN):
        return next(
            j
            for j in self.jobs.values()
            if (j.spec or {}).get("plan_run_id") == run_id and j.job_type == JobType.RESEARCH_LOOP
        )


class TokenPodman(FakePodman):
    """FakePodman that keeps every session token the provisioner sealed."""

    def __init__(self):
        super().__init__()
        self.tokens: dict[str, str] = {}

    def __call__(self, *args, input_text=None, **kw):
        if args[:2] == ("secret", "create"):
            self.tokens[args[2]] = input_text
        return super().__call__(*args, input_text=input_text, **kw)


@pytest.fixture(autouse=True)
def inbox():
    store = ApprovalInboxStore()
    tool_approvals.configure(store)
    yield store
    tool_approvals.configure(None)


@pytest.fixture
def h():
    pool = _Pool()
    broker = SecretBroker()
    podman = TokenPodman()
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="plan-venue-"))
    )
    jobs = VenueJobStore()
    traces = ExecutionTraceStore(db_url=None)
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=provisioner,
        pool=pool,
        sealed_store=FakeSealed({("alice", "openai-main"): _meta("openai-main")}),
        job_store=jobs,
        site_name="rccs",
    )
    budget = {"max_wall_clock_hours": 0.0}
    app = FastAPI()
    app.include_router(traces_api.router)
    app.include_router(prv_api.router)
    app.include_router(venue_api.router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    app.state.plan_run_budget_reader = lambda _project: dict(budget)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: jobs
    app.dependency_overrides[traces_api.get_trace_store] = lambda: traces
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: _Caller()
    return SimpleNamespace(
        client=TestClient(app),
        app=app,
        jobs=jobs,
        traces=traces,
        broker=broker,
        podman=podman,
        service=service,
        budget=budget,
    )


def _start(h, run=RUN, definition=DEFINITION, **over):
    body = {"definition": definition, "source_refs": ["openai-main"], **over}
    return h.client.post(f"/umrs/traces/plan-runs/{run}/venue", json=body, headers=H)


def _run_lane(h) -> list[str]:
    """Play the durable runtime: claim and provision every queued session job."""
    backend = FrontendSessionBackend(
        config=h.service.config,
        provisioner=h.service.provisioner,
        store=h.service.store,
        broker=h.broker,
        pool=None,
    )
    refs = []
    for job in [j for j in h.jobs.session_jobs() if j.status == JobStatus.QUEUED]:
        job.status = JobStatus.RUNNING
        refs.append(asyncio.run(backend.submit_job(job, SimpleNamespace(attempt_id="a1"))))
    return refs


def _worker(h, run=RUN):
    """The started worker: its session id, rail headers and podman argv."""
    sid = prr.venue_session_id("alice", run)
    token = h.podman.tokens[f"sess-{sid}-token"]
    argv = next(c for c in h.podman.calls if c[0] == "run" and f"worker-{sid}" in c)
    return SimpleNamespace(
        sid=sid,
        token=token,
        headers={"authorization": f"Bearer {token}"},
        argv=argv,
        gw=f"/umrs/companion/sessions/{sid}/rail/gw/traces/plan-runs/{run}",
    )


def _launched(h, run=RUN):
    assert _start(h, run=run).status_code == 202
    _run_lane(h)
    return _worker(h, run)


def _record(key="a" * 64):
    return json.dumps({"call_key": key, "kind": "agent", "node": "s1", "ok": True, "output": 1})


class TestLaunch:
    def test_start_records_the_run_and_launches_one_plan_worker(self, h):
        res = _start(h)
        assert res.status_code == 202, res.text
        body = res.json()
        run_job = h.jobs.jobs[body["job_id"]]
        (session_job,) = h.jobs.session_jobs()
        sid = prr.venue_session_id("alice", RUN)
        assert {
            "hosting": run_job.spec["hosting"],
            "type": run_job.job_type,
            "required": run_job.spec["durable_required"],
            "session": run_job.spec["venue_session_id"],
            "definition": run_job.spec["definition"] == DEFINITION,
            "run_status": run_job.status,
            "session_spec": (
                session_job.spec["session_id"],
                session_job.spec["plan_run_id"],
                session_job.spec["plan_run_job_id"],
                session_job.spec["project"],
            ),
            "enqueued": h.jobs.enqueued,
        } == {
            "hosting": "venue",
            "type": JobType.RESEARCH_LOOP,
            "required": True,
            "session": sid,
            "definition": True,
            "run_status": JobStatus.PLANNED,
            "session_spec": (sid, RUN, run_job.job_id, "proj-x"),
            "enqueued": [session_job.job_id],
        }
        # the request provisions nothing; the durable lane does
        assert h.podman.calls == []

    def test_the_lane_runs_the_plan_worker_command_in_the_sandbox(self, h):
        worker = _launched(h)
        assert worker.argv[-4:] == ("venue", "worker", "--plan-run", RUN)
        assert "--cap-drop" in worker.argv and f"sess-{worker.sid}" in worker.argv
        assert f"R4_VENUE_SESSION_ID={worker.sid}" in worker.argv

    def test_no_user_jwt_or_provider_key_reaches_the_worker_env(self, h):
        worker = _launched(h)
        envs = [worker.argv[i + 1] for i, a in enumerate(worker.argv) if a == "--env"]
        names = {e.split("=", 1)[0] for e in envs}
        assert names == {
            "OPENAI_BASE_URL",
            "ANTHROPIC_BASE_URL",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "NO_PROXY",
            "R4_VENUE_SESSION_TOKEN_FILE",
            "R4_VENUE_SESSION_ID",
            "R4_VENUE_RAIL_URL",
            "R4_ENDPOINT_URL",
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "R4_COMPANION_DB",
        }
        flat = " ".join(worker.argv).lower()
        assert worker.token not in " ".join(worker.argv)  # the token rides the secret only
        for leak in ("api_key", "authorization", "bearer", "jwt", "sk-"):
            assert leak not in flat

    def test_disabled_venue_refuses_by_name_and_creates_nothing(self, h):
        h.service.config = FrontendVenueConfig.from_env({})
        res = _start(h)
        assert res.status_code == 503
        assert res.headers["x-platform-error-code"] == "companionvenue.unavailable"
        detail = res.json()["detail"]
        assert detail.startswith("plan_iteration_needs_server_venue")
        assert "UMRS_VENUE_ENABLED" in detail and "frontend-venue play" in detail
        assert h.jobs.jobs == {}

    def test_a_plan_over_the_project_wall_clock_cap_is_refused_before_a_worker(self, h):
        h.budget["max_wall_clock_hours"] = 0.1  # 360 s < the plan's 600 s
        res = _start(h)
        assert res.status_code == 403
        assert "plan_exceeds_project_caps" in res.json()["detail"]
        assert h.jobs.jobs == {} and h.podman.calls == []

    def test_an_unbounded_plan_in_a_capped_project_is_refused(self, h):
        h.budget["max_wall_clock_hours"] = 1.0
        plan = {**PLAN, "budget": {"max_tokens": 1000}}
        res = _start(h, definition={**DEFINITION, "plan": plan})
        assert res.status_code == 403
        assert "unbounded" in res.json()["detail"]

    def test_a_plan_within_the_cap_starts_and_records_the_cap(self, h):
        h.budget["max_wall_clock_hours"] = 1.0
        assert _start(h).status_code == 202
        assert h.jobs.run_job().spec["caps"] == {"max_wall_clock_sec": 3600}

    def test_an_invalid_plan_is_refused_with_every_problem(self, h):
        plan = {**PLAN, "budget": {}}
        res = _start(h, definition={**DEFINITION, "plan": plan})
        assert res.status_code == 400
        assert "plan_invalid" in res.json()["detail"] and h.jobs.jobs == {}

    def test_a_running_worker_is_not_started_twice(self, h):
        assert _start(h).status_code == 202
        res = _start(h)
        assert res.status_code == 409
        assert res.headers["x-platform-error-code"] == "companionvenue.already_running"

    def test_a_client_hosted_run_is_not_resumed_on_the_venue(self, h):
        h.client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts", json={"attempt": 1, "definition": DEFINITION}
        )
        res = _start(h)
        assert res.status_code == 409 and "hosting_mismatch" in res.json()["detail"]


def _open_via_rail(h, worker, attempt=1):
    return h.client.post(
        f"{worker.gw}/attempts", json={"attempt": attempt, "definition": {}}, headers=worker.headers
    )


def _record_texts(h, worker) -> list:
    """The record texts the worker reads back through its rail."""
    read = h.client.get(f"{worker.gw}/records", headers=worker.headers).json()
    return [r["record"] for r in read["records"]]


class TestWorkerCredentialScope:
    def test_the_worker_writes_its_own_runs_record_through_the_rail(self, h):
        worker = _launched(h)
        opened = _open_via_rail(h, worker)
        assert opened.status_code == 201, opened.text
        attempt_id = opened.json()["attempt_id"]
        wrote = h.client.post(
            f"{worker.gw}/records",
            json={"attempt_id": attempt_id, "record": _record()},
            headers=worker.headers,
        )
        assert wrote.status_code == 201, wrote.text
        records = _record_texts(h, worker)
        done = h.client.post(
            f"{worker.gw}/attempts/{attempt_id}/finish",
            json={"status": "success", "summary": {"status": "success", "output": {"x": 1}}},
            headers=worker.headers,
        )
        view = h.client.get(f"/umrs/traces/plan-runs/{RUN}", headers=H).json()
        assert {
            "records": records,
            "finish": (done.status_code, done.json()["job_status"]),
            "hosting": view["hosting"],
            "plan_status": view["attempts"][0]["plan_status"],
            "output": view["attempts"][0]["summary"]["output"],
            "backend": h.jobs.attempts[attempt_id].backend,
            # every attestation lived only for its own in-process call
            "attestations": h.app.state.rail_attestations,
        } == {
            "records": [_record()],
            "finish": (200, "completed"),
            "hosting": "venue",
            "plan_status": "success",
            "output": {"x": 1},
            "backend": "r4_venue",
            "attestations": {},
        }

    def test_the_owners_own_client_cannot_write_a_venue_hosted_run(self, h):
        worker = _launched(h)
        attempt_id = _open_via_rail(h, worker).json()["attempt_id"]
        base = f"/umrs/traces/plan-runs/{RUN}"
        refused = [
            h.client.post(f"{base}/attempts", json={"attempt": 2}, headers=H),
            h.client.post(
                f"{base}/records", json={"attempt_id": attempt_id, "record": _record()}, headers=H
            ),
            h.client.post(
                f"{base}/attempts/{attempt_id}/finish", json={"status": "failed"}, headers=H
            ),
        ]
        assert [r.status_code for r in refused] == [403, 403, 403]
        assert all("venue_hosted_run" in r.json()["detail"] for r in refused)
        # reading stays the owner's
        assert h.client.get(f"{base}/records", headers=H).status_code == 200

    def test_a_worker_cannot_reach_another_runs_record(self, h):
        worker = _launched(h)
        _launched(h, run=OTHER_RUN)
        sid = worker.sid
        other = f"/umrs/companion/sessions/{sid}/rail/gw/traces/plan-runs/{OTHER_RUN}"
        for method, tail in (("POST", "/attempts"), ("GET", "/records"), ("GET", "")):
            res = h.client.request(
                method, other + tail, json={"attempt": 1}, headers=worker.headers
            )
            assert res.status_code == 403, (method, tail)
            assert "plan run its session hosts" in res.json()["detail"]

    def test_a_worker_cannot_start_resume_or_cancel_its_run(self, h):
        worker = _launched(h)
        for tail in ("/venue", "/cancel"):
            res = h.client.post(worker.gw + tail, json={}, headers=worker.headers)
            assert res.status_code == 403, tail
        res = h.client.get(
            f"/umrs/companion/sessions/{worker.sid}/rail/gw/traces/job/x", headers=worker.headers
        )
        assert res.status_code == 403

    def test_a_chat_sessions_worker_never_reaches_the_traces_family(self, h):
        worker = _launched(h)
        row = h.service.store.get(worker.sid)
        h.jobs.jobs[row.job_id].spec.pop("plan_run_id")  # as a chat session's job
        res = h.client.get(worker.gw, headers=worker.headers)
        assert res.status_code == 403

    def test_a_forged_attestation_header_is_stripped_and_worthless(self, h):
        worker = _launched(h)
        handle = _rail_identity.attest(h.app.state, worker.sid)
        _rail_identity.revoke_attestation(h.app.state, handle)
        res = h.client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts",
            json={"attempt": 1},
            headers={**H, _rail_identity.RAIL_ATTESTATION_HEADER: handle},
        )
        assert res.status_code == 403  # an expired handle names no session
        assert h.app.state.rail_attestations == {}  # nothing outlives its call

    def test_the_writer_rule(self):
        venue = SimpleNamespace(
            job_id="j", spec={"hosting": "venue", "plan_run_id": RUN, "venue_session_id": "s1"}
        )
        client = SimpleNamespace(job_id="k", spec={"hosting": "client", "plan_run_id": RUN})
        prr.check_writer(venue, "s1")
        prr.check_writer(client, None)
        for job, session in ((venue, None), (venue, "s2"), (client, "s1")):
            with pytest.raises(prr.PlanRunRecordError) as err:
                prr.check_writer(job, session)
            assert err.value.code == "forbidden"

    def test_a_worker_never_creates_a_run(self, h):
        handle = _rail_identity.attest(h.app.state, "s1")
        res = h.client.post(
            f"/umrs/traces/plan-runs/{RUN}/attempts",
            json={"attempt": 1, "definition": DEFINITION},
            headers={**H, _rail_identity.RAIL_ATTESTATION_HEADER: handle},
        )
        assert res.status_code == 403 and h.jobs.jobs == {}


class TestGatesGoToTheInbox:
    def _gate(self, h, worker, gate="g1"):
        return h.client.post(
            f"/umrs/companion/sessions/{worker.sid}/rail/plan-gates",
            json={"gate": gate, "trigger_class": "irreversible", "prompt": "merge?"},
            headers=worker.headers,
        )

    def test_a_gate_is_an_adr066_inbox_intent_the_owner_decides(self, h, inbox):
        worker = _launched(h)
        attempt_id = _open_via_rail(h, worker).json()["attempt_id"]
        res = self._gate(h, worker)
        assert res.status_code == 200, res.text
        approval = res.json()["approval"]
        record = inbox.get(approval["approval_id"])
        assert {
            "loop_id": record["loop_id"],
            "user": record["user_id"],
            "job": record["job_id"],
            "attempt": record["attempt_id"],
            "status": record["status"],
            "kind": record["action"]["kind"],
            "untrusted": record["action"]["untrusted"],
            "run": record["action"]["run_id"],
        } == {
            "loop_id": f"plan-run:{RUN}",
            "user": "alice",
            "job": h.jobs.run_job().job_id,
            "attempt": attempt_id,
            "status": "pending",
            "kind": "plan_gate",
            "untrusted": True,
            "run": RUN,
        }
        assert [p["id"] for p in tool_approvals.pending(user_id="alice")] == [
            approval["approval_id"]
        ]
        # the owner answers through the inbox; the worker reads the answer
        tool_approvals.decide(approval["approval_id"], "once", decided_by="alice")
        url = f"/umrs/companion/sessions/{worker.sid}/rail/plan-gates/{approval['approval_id']}"
        polled = h.client.get(url, headers=worker.headers).json()["approval"]
        assert (polled["status"], polled["decision"], polled["decided_by"]) == (
            "approved",
            "once",
            "alice",
        )

    def test_a_retried_gate_request_finds_the_same_intent(self, h):
        worker = _launched(h)
        _open_via_rail(h, worker)
        first = self._gate(h, worker).json()["approval"]["approval_id"]
        assert self._gate(h, worker).json()["approval"]["approval_id"] == first

    def test_a_gate_needs_an_open_attempt(self, h):
        worker = _launched(h)
        assert self._gate(h, worker).status_code == 409

    def test_a_worker_cannot_read_another_runs_gate(self, h):
        worker = _launched(h)
        other = _launched(h, run=OTHER_RUN)
        _open_via_rail(h, other)
        theirs = self._gate(h, other).json()["approval"]["approval_id"]
        url = f"/umrs/companion/sessions/{worker.sid}/rail/plan-gates/{theirs}"
        assert h.client.get(url, headers=worker.headers).status_code == 404

    def test_a_chat_session_raises_no_plan_gate(self, h):
        worker = _launched(h)
        row = h.service.store.get(worker.sid)
        h.jobs.jobs[row.job_id].spec.pop("plan_run_id")
        assert self._gate(h, worker).status_code == 403

    def test_the_rail_has_no_decide_verb(self):
        rail = _rail_routes()
        assert {
            "rail routes": bool(rail),
            "decide paths": _decide_paths(rail),
            # no inbox decide via the gw
            "companion via gw": "companion" in venue_api.RAIL_GW_FAMILIES,
            # create + read only: a gate is never answered over the rail
            "plan gates": _plan_gate_methods(rail),
        } == {
            "rail routes": True,
            "decide paths": [],
            "companion via gw": False,
            "plan gates": {
                "/umrs/companion/sessions/{session_id}/rail/plan-gates": {"POST"},
                "/umrs/companion/sessions/{session_id}/rail/plan-gates/{approval_id}": {"GET"},
            },
        }


def _rail_routes() -> list:
    return [r for r in venue_api.router.routes if "/rail/" in getattr(r, "path", "")]


def _decide_paths(rail: list) -> list[str]:
    return [r.path for r in rail if "decision" in r.path or "decide" in r.path]


def _plan_gate_methods(rail: list) -> dict[str, set[str]]:
    return {r.path: r.methods for r in rail if "plan-gates" in r.path}


class TestResumeAndCancel:
    def test_cancel_stops_the_worker_and_cancels_the_record(self, h):
        worker = _launched(h)
        _open_via_rail(h, worker)
        res = h.client.post(f"/umrs/traces/plan-runs/{RUN}/cancel", headers=H)
        assert res.status_code == 200, res.text
        body = res.json()
        run_job = h.jobs.run_job()
        (session_job,) = h.jobs.session_jobs()
        assert (body["cancelled"], body["job_status"], body["venue"]["state"]) == (
            True,
            "cancelled",
            "stopped",
        )
        assert run_job.status == JobStatus.CANCELLED
        assert session_job.status == JobStatus.CANCELLED
        assert ("rm", "-f", f"worker-{worker.sid}") in h.podman.calls
        # the lease is revoked: the worker's rail is closed
        res = h.client.get(worker.gw, headers=worker.headers)
        assert res.status_code == 401
        # and a cancelled run is never resumed
        again = _start(h, definition={})
        assert again.status_code == 409 and "plan_run_cancelled" in again.json()["detail"]

    def test_resume_starts_a_fresh_worker_on_the_recorded_definition(self, h):
        worker = _launched(h)
        attempt_id = _open_via_rail(h, worker).json()["attempt_id"]
        h.client.post(
            f"{worker.gw}/attempts/{attempt_id}/finish",
            json={"status": "failed", "error": "x"},
            headers=worker.headers,
        )
        h.service.stop(owner="alice", session_id=worker.sid, actor="test")
        res = _start(h, definition={})
        assert res.status_code == 202, res.text
        assert res.json()["created"] is False
        assert len(h.jobs.session_jobs()) == 2
        _run_lane(h)
        fresh = _worker(h)
        reopened = _open_via_rail(h, fresh, attempt=2)
        assert reopened.status_code == 201, reopened.text
        assert reopened.json()["attempt_no"] == 2

    def test_status_names_the_record_and_the_worker(self, h):
        _launched(h)
        view = h.client.get(f"/umrs/traces/plan-runs/{RUN}/venue", headers=H).json()
        assert (view["hosting"], view["venue"]["state"], view["venue"]["job"]["status"]) == (
            "venue",
            "running",
            "running",
        )

    def test_a_worker_that_reports_done_ends_its_session(self, h):
        worker = _launched(h)
        h.service.heartbeat(
            session_id=worker.sid,
            lease_id=h.service.store.get(worker.sid).lease_id,
            phase="done",
            detail="plan run success",
        )
        backend = FrontendSessionBackend(
            config=h.service.config,
            provisioner=h.service.provisioner,
            store=h.service.store,
            broker=h.broker,
            pool=None,
        )
        attempt = SimpleNamespace(
            backend_ref=f"frontend_session|session_id={worker.sid}|lease_id=x"
        )
        result = asyncio.run(
            FrontendSessionStatusChecker(backend).check_status(attempt, SimpleNamespace())
        )
        assert (result.status, h.service.store.get(worker.sid).state) == ("succeeded", "stopped")


class TestTheFoundationRules:
    def test_caps_are_the_wall_clock_only(self):
        assert plan_runs.caps_for({"max_wall_clock_hours": 2, "max_gpu_hours": 5}) == {
            "max_wall_clock_sec": 7200
        }
        assert plan_runs.caps_for({"max_wall_clock_hours": 0}) == {}
        assert plan_runs.caps_problem(DEFINITION, {}) == ""
        assert plan_runs.caps_problem(DEFINITION, {"max_wall_clock_sec": 600}) == ""
        assert plan_runs.caps_problem(DEFINITION, {"max_wall_clock_sec": 599})

    def test_the_backend_still_never_hosts_a_plan_run_record(self):
        from umrs.jobs.research_loop_backend import validate_research_loop_spec

        with pytest.raises(ValueError, match="client_hosted_run"):
            validate_research_loop_spec({"hosting": "venue", "plan_run_id": RUN})

    def test_the_session_spec_validates_the_run_id(self):
        from umrs.jobs.frontend_session_backend import (
            validate_frontend_session_spec,
            worker_command_for,
        )

        base = {
            "session_id": "s",
            "owner": "alice",
            "source_ids": ["i"],
            "source_names": ["n"],
        }
        assert worker_command_for(validate_frontend_session_spec(base)) == ("venue", "worker")
        with pytest.raises(prr.PlanRunRecordError):
            validate_frontend_session_spec({**base, "plan_run_id": "../x; rm"})


class TestTheLoopPathNamesTheVenue:
    """Without the venue the loop route's refusal keeps its code and names
    what the site lacks; with it, a plan loop is checked before it exists
    (slice 8b: tests/test_plan_loop_server.py runs one)."""

    @pytest.fixture
    def loops(self):
        from umrs.api import loop as loop_api
        from umrs.knowledge.loop import AutonomousResearchLoop

        loop_api.clear_loop_store()
        previous = loop_api._loop_engine
        loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
        app = FastAPI()
        app.include_router(loop_api.router)
        yield TestClient(app), loop_api
        loop_api._loop_engine = previous
        loop_api.clear_loop_store()

    def _start_loop(self, client):
        return client.post(
            "/umrs/loops",
            json={"project_id": "p", "research_goal": "g", "iteration_plan": "research-round"},
        )

    def test_without_the_venue_the_refusal_names_the_settings(self, loops, monkeypatch):
        client, loop_api = loops
        for name in READY_ENV:
            monkeypatch.delenv(name, raising=False)
        res = self._start_loop(client)
        assert res.status_code == 422
        assert "plan_iteration_needs_server_venue" in res.text
        assert "UMRS_VENUE_ENABLED" in res.text and "frontend-venue play" in res.text
        assert loop_api._loop_store.list_all() == []

    def test_with_the_venue_an_incomplete_plan_loop_is_refused_by_what_it_lacks(
        self, loops, monkeypatch
    ):
        client, loop_api = loops
        for name, value in READY_ENV.items():
            monkeypatch.setenv(name, value)
        res = self._start_loop(client)
        assert res.status_code == 422
        # slice 8b: the venue runs plan loops; this one names no source
        assert "plan_sources" in res.text and "UMRS_VENUE_ENABLED" not in res.text
        assert loop_api._loop_store.list_all() == []
