"""Frontend-venue routes: owner start/status/stop + the lease-token worker rail.

Fakes: in-memory broker + registry, FakePodman-driven provisioner, a
snapshot pool that answers exactly the SQL shapes the shared row helpers
issue, and the sqlite T4 approvals store. No live podman, no Postgres.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from umrs.api.companion_venue import router
from umrs.companion_trust.chat_turn_approvals import ChatTurnApprovalsStore
from umrs.jobs.frontend_session_backend import FrontendSessionBackend
from umrs.jobs.secret_broker import SecretBroker
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

SRC_ID = "11111111-1111-4111-8111-111111111111"
READY_ENV = {
    "UMRS_VENUE_ENABLED": "1",
    "UMRS_VENUE_INFER_HOST": "umrs-infer",
    "UMRS_VENUE_API_UPSTREAM": "169.254.1.2:8000",
    "UMRS_VENUE_WORKER_IMAGE": "localhost/r4-companion@sha256:abc",
    "UMRS_VENUE_NGINX_IMAGE": "docker.io/library/nginx@sha256:def",
    "UMRS_VENUE_SQUID_IMAGE": "docker.io/ubuntu/squid@sha256:123",
    "UMRS_VENUE_RAIL_IP_CHECK": "0",
}


# ── snapshot pool fake (only the helper SQL shapes) ─────────────────────────


class _Cursor:
    def __init__(self, table: dict, recorded: list, projects: dict | None = None):
        self._table = table
        self._recorded = recorded
        self._row = None
        self.rowcount = 0
        self._conn_projects = projects

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    _HANDLERS = (
        ("select 1 from companion_session_snapshots", "_exists"),
        ("select s.session_id, s.name, s.payload", "_detail"),
        ("select payload from companion_session_snapshots", "_payload"),
        ("update companion_session_snapshots set venue", "_set_venue"),
        ("update companion_session_snapshots set payload", "_set_payload"),
        ("insert into companion_session_snapshots", "_insert"),
        ("select data_class from projects", "_data_class"),
    )

    def _data_class(self, params):
        # projects dict lives on the pool (set per-test; absent row → None)
        projects = self._conn_projects or {}
        pid = params[0]
        self._row = (projects.get(pid),) if pid in projects else None

    def execute(self, sql, params=None):
        sql_l = " ".join(sql.split()).lower()
        self._recorded.append(sql_l)
        for needle, handler in self._HANDLERS:
            if needle in sql_l:
                getattr(self, handler)(params or ())
                return
        raise AssertionError(f"unexpected SQL in venue route test: {sql_l[:80]}")

    def _exists(self, params):
        sid, uid = params
        self._row = (1,) if (uid, sid) in self._table else None

    def _detail(self, params):
        uid, sid = params
        row = self._table.get((uid, sid))
        if row is None:
            self._row = None
            return
        self._row = (
            sid,
            row["name"],
            row["payload"],
            row["marker"],
            row["venue"],
            row["state"],
            None,
        )

    def _payload(self, params):
        uid, sid = params
        row = self._table.get((uid, sid))
        self._row = (row["payload"],) if row else None

    def _set_venue(self, params):
        venue, state, uid, sid = params
        row = self._table.get((uid, sid))
        if row:
            row["venue"], row["state"] = venue, state
        self.rowcount = 1 if row else 0

    def _set_payload(self, params):
        payload, uid, sid = params
        row = self._table.get((uid, sid))
        if row:
            row["payload"] = payload
        self.rowcount = 1 if row else 0

    def _insert(self, params):
        uid, sid, name, payload, marker, venue, state = params
        existing = self._table.get((uid, sid)) or {}
        if existing and existing["marker"] > marker:
            self.rowcount = 0
            return
        self._table[(uid, sid)] = {
            "name": name,
            "payload": payload,
            "marker": marker,
            "venue": venue or existing.get("venue", "local"),
            "state": state or existing.get("state", ""),
        }
        self.rowcount = 1

    def fetchone(self):
        return self._row


class _Conn:
    def __init__(self, table, recorded):
        self._table = table
        self._recorded = recorded
        self.commits = 0
        self.projects: dict = {}

    def cursor(self):
        return _Cursor(self._table, self._recorded, self.projects)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class _Pool:
    def __init__(self):
        self.table: dict = {}
        self.recorded: list = []
        self._conn = _Conn(self.table, self.recorded)

    @property
    def projects(self):
        return self._conn.projects

    @projects.setter
    def projects(self, value):
        self._conn.projects = value

    def getconn(self):
        return self._conn

    def putconn(self, _):
        pass

    def seed(self, uid: str, sid: str, *, queue=None, messages=None):
        payload = {"id": sid, "messages": messages or [], "metadata": {"chat_queue": queue or []}}
        self.table[(uid, sid)] = {
            "name": "S",
            "payload": json.dumps(payload),
            "marker": "2026-09-21T00:00:00+00:00",
            "venue": "local",
            "state": "",
        }


class FakePodman:
    def __init__(self, fail_at: str | None = None):
        self.calls: list[tuple[str, ...]] = []
        self.controllers = "[cpu io memory pids]"
        self.fail_at = fail_at

    def __call__(self, *args, input_text=None, **_):
        del input_text  # token never inspected here; the rail fixture captures it
        self.calls.append(tuple(args))
        if self.fail_at and any(self.fail_at in a for a in args):
            return subprocess.CompletedProcess(args, 1, "", f"boom {self.fail_at}")
        if args[0] == "info":
            return subprocess.CompletedProcess(args, 0, self.controllers + "\n", "")
        if args[:2] == ("pod", "inspect"):
            return subprocess.CompletedProcess(args, 0, "infra-abc123\n", "")
        if args[0] == "inspect":
            body = [{"NetworkSettings": {"Networks": {"egress-net": {"IPAddress": "172.20.8.2"}}}}]
            return subprocess.CompletedProcess(args, 0, json.dumps(body), "")
        return subprocess.CompletedProcess(args, 0, "", "")


class FakeSealed:
    def __init__(self, rows: dict[tuple[str, str], object]):
        self.rows = rows

    def get_meta(self, owner, name):
        return self.rows.get((owner, name))


def _meta(name: str, *, tombstoned=None):
    # provider_kind: pushed sources are public providers (ADR-062's class)
    return SimpleNamespace(id=SRC_ID, name=name, tombstoned_at=tombstoned, provider_kind="openai")


class FakeJobStore:
    """The JobSubmitter slice + the origin methods the venue routes touch."""

    def __init__(self):
        self.created: list[dict] = []
        self.jobs: dict[str, SimpleNamespace] = {}
        self.attempts: dict[str, list] = {}
        self.enqueued: list[str] = []
        self.cancelled: list[tuple[str, str]] = []
        self.stamped: list[tuple[str, dict]] = []
        self.by_spec: dict[tuple, str] = {}
        self.cancel_raises = False

    def create_job(self, **kw):
        job_id = f"job-{len(self.created) + 1}"
        self.created.append(kw)
        spec = kw["spec"]
        self.jobs[job_id] = SimpleNamespace(
            job_id=job_id, status=_status("planned"), venue=spec.get("venue", ""), spec=spec
        )
        return job_id  # the real JobStore.create_job answers the id, not a Job

    def enqueue_job(self, job_id):
        self.enqueued.append(job_id)
        self.jobs[job_id].status = _status("queued")

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_attempts_for_job(self, job_id):
        return list(self.attempts.get(job_id, []))

    def cancel_job(self, job_id, reason=""):
        if self.cancel_raises:
            raise RuntimeError("already terminal")
        self.cancelled.append((job_id, reason))
        self.jobs[job_id].status = _status("cancelled")

    def stamp_spec_origin(self, job_id, origin):
        self.stamped.append((job_id, dict(origin)))
        return True

    def find_job_id_by_spec_value(self, job_type, key, value):
        return self.by_spec.get((job_type.value, key, value))


def _status(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _attempt(status: str, exit_reason=None, ref=None) -> SimpleNamespace:
    return SimpleNamespace(
        status=_status(status), exit_reason=exit_reason, status_detail=None, backend_ref=ref
    )


def _run_lane(h) -> list[tuple[str, str]]:
    """Play the durable runtime: claim every queued session job on this harness."""
    backend = FrontendSessionBackend(
        config=h.service.config,
        provisioner=h.service.provisioner,
        store=h.service.store,
        broker=h.broker,
        pool=h.pool,
    )
    out: list[tuple[str, str]] = []
    for job in [j for j in h.jobs.jobs.values() if j.status.value == "queued"]:
        job.status = _status("running")
        try:
            ref = asyncio.run(backend.submit_job(job, SimpleNamespace(attempt_id="a1")))
        except Exception as exc:  # noqa: BLE001 — the attempt failed
            job.status = _status("failed")
            h.jobs.attempts[job.job_id] = [_attempt("failed", str(exc))]
            out.append(("failed", str(exc)))
            continue
        h.jobs.attempts[job.job_id] = [_attempt("running", None, ref)]
        out.append(("running", job.job_id))
    return out


def _probe_routes(app: FastAPI) -> None:
    """Fake platform families the rail gw re-dispatches to (echo identity)."""

    async def _echo(request: Request, tail: str = "") -> dict:
        return {
            "path": request.url.path,
            "method": request.method,
            "query": dict(request.query_params),
            "user": request.headers.get("x-user-id"),
            "project": request.headers.get("x-verified-project-id"),
            "hmac": request.headers.get("x-verified-hmac"),
            "admin": request.headers.get("x-verified-is-admin"),
            "auth": request.headers.get("authorization"),
            "body": (await request.body()).decode() or None,
        }

    async def _create_job() -> dict:
        return {"job_id": "job-9"}

    async def _run_wf() -> dict:
        return {"execution_id": "exec-1"}

    # job-creating POSTs (registered first: they shadow the echo catch-alls)
    app.add_api_route("/umrs/jobs", _create_job, methods=["POST"])
    app.add_api_route("/umrs/workflows/wf-1/run", _run_wf, methods=["POST"])
    app.add_api_route("/umrs/workflows/{tail:path}", _echo, methods=["GET", "POST", "DELETE"])
    app.add_api_route("/umrs/jobs/{tail:path}", _echo, methods=["GET", "POST"])

    # D-42 read-only families the rail now admits (echo GET; writes refused at
    # the rail layer itself, so no upstream route needed for POST).
    for _family in ("knowledge", "catalog", "scientific-catalogs", "rag"):
        app.add_api_route(f"/umrs/{_family}/{{tail:path}}", _echo, methods=["GET"])

    async def _secret(request: Request) -> dict:
        return {"leak": request.headers.get("x-user-id")}

    app.add_api_route("/umrs/secrets/probe", _secret, methods=["GET"])


@pytest.fixture
def harness(monkeypatch):
    approvals = ChatTurnApprovalsStore()
    pool = _Pool()
    broker = SecretBroker()
    podman = FakePodman()
    provisioner = FrontendPodmanProvisioner(
        broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="venue-route-"))
    )
    jobs = FakeJobStore()
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
    app = FastAPI()
    app.include_router(router)
    _probe_routes(app)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    app.state.companion_venue_approvals_factory = lambda _conn: approvals
    client = TestClient(app)
    return SimpleNamespace(
        client=client,
        pool=pool,
        broker=broker,
        podman=podman,
        service=service,
        approvals=approvals,
        jobs=jobs,
    )


H = {"x-user-id": "alice"}


def _start(h, sid="sess-1", **over):
    body = {"source_refs": ["openai-main"], **over}
    return h.client.post(f"/umrs/companion/sessions/{sid}/venue/start", json=body, headers=H)


class TestOwnerRoutes:
    def test_start_requires_identity(self, harness):
        r = harness.client.post(
            "/umrs/companion/sessions/s/venue/start", json={"source_refs": ["x"]}
        )
        assert r.status_code == 401

    def test_start_404_when_not_mirrored(self, harness):
        r = _start(harness)
        assert r.status_code == 404
        assert r.headers["x-platform-error-code"] == "companionvenue.not_found"

    def test_start_409_source_not_pushed(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness, source_refs=["openai-main", "missing-one"])
        assert r.status_code == 409
        assert r.headers["x-platform-error-code"] == "companionvenue.source_not_pushed"
        assert "missing-one" in r.json()["detail"]

    def test_start_400_bad_policy(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness, web_policy="wide-open")
        assert r.status_code == 400

    def test_start_503_when_not_configured(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.service.config = FrontendVenueConfig.from_env({})
        r = _start(harness)
        assert r.status_code == 503
        assert "UMRS_VENUE_ENABLED" in r.json()["detail"]

    def test_start_submits_the_session_job(self, harness):
        harness.pool.seed("alice", "sess-1")
        r = _start(harness)
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["state"], body["source_names"]) == ("provisioning", ["openai-main"])
        assert body["job"] == {
            "job_id": "job-1",
            "status": "queued",
            "venue": "rccs",
            "attempt_status": None,
            "attempt_detail": None,
        }
        created = harness.jobs.created[0]
        assert (created["job_type"].value, created["spec"]["venue"], created["spec"]["owner"]) == (
            "frontend_session",
            "rccs",
            "alice",
        )
        assert (created["max_attempts"], created["restart_policy"].value) == (1, "never")
        key = created["idempotency_key"]
        assert key.startswith("fs:sess-1:") and len(key) <= 64
        # the request itself provisions nothing; the lane does
        assert (harness.jobs.enqueued, harness.podman.calls) == (["job-1"], [])

    def test_lane_provisions_and_projects(self, harness):
        harness.pool.seed("alice", "sess-1")
        assert _start(harness).status_code == 200
        assert _run_lane(harness) == [("running", "job-1")]
        body = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert (body["state"], body["egress_ip"], body["lease"]["is_valid"]) == (
            "running",
            "172.20.8.2",
            True,
        )
        assert (body["job"]["status"], body["job"]["attempt_status"]) == ("running", "running")
        # registry projection: the snapshot row now says frontend/ongoing
        row = harness.pool.table[("alice", "sess-1")]
        assert (row["venue"], row["state"]) == ("frontend", "ongoing")

    def test_lane_worker_argv_carries_rail_env(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        # worker argv carries the loop command + rail env, never the token
        worker = next(c for c in harness.podman.calls if c[0] == "run" and "worker-sess-1" in c)
        assert worker[-2:] == ("venue", "worker")
        assert {"R4_VENUE_RAIL_URL=http://egress:8081", "R4_VENUE_SESSION_ID=sess-1"} <= set(worker)

    def test_second_start_is_409_already_running(self, harness):
        harness.pool.seed("alice", "sess-1")
        assert _start(harness).status_code == 200
        r = _start(harness)
        assert r.status_code == 409
        assert r.headers["x-platform-error-code"] == "companionvenue.already_running"

    def test_provision_failure_fails_attempt_and_row(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.podman.fail_at = "egress-sess-1"
        assert _start(harness).status_code == 200
        ((outcome, reason),) = _run_lane(harness)
        assert (outcome, "egress" in reason) == ("failed", True)
        status = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert (status["state"], "egress" in status["reason"], status["job"]["status"]) == (
            "failed",
            True,
            "failed",
        )
        assert "egress" in status["job"]["attempt_detail"]
        assert harness.pool.table[("alice", "sess-1")]["state"] == "error"
        # failed is terminal for that job: a deliberate restart submits a new one
        assert _start(harness).status_code == 200
        assert harness.jobs.enqueued == ["job-1", "job-2"]

    def test_idempotency_key_fits_the_column_for_uuid_sessions(self):
        from umrs.venues.frontend_service import _idempotency_key

        key = _idempotency_key("9341f89e-d714-48be-9571-b1e9e21844cd")
        assert len(key) <= 64 and key.startswith("fs:9341f89e-d714-48be-9571-b1e9e21844cd:")

    def test_unexpected_service_failure_is_a_typed_503(self, harness, monkeypatch):
        harness.pool.seed("alice", "sess-1")

        def boom(**_kw):
            raise RuntimeError("column job_id does not exist")

        monkeypatch.setattr(harness.service, "start", boom)
        r = _start(harness)
        assert r.status_code == 503
        assert r.headers["x-platform-error-code"] == "companionvenue.unavailable"
        assert "RuntimeError: column job_id does not exist" in r.json()["detail"]

    def test_job_dead_before_its_backend_ran_settles_the_row(self, harness):
        harness.pool.seed("alice", "sess-1")
        assert _start(harness).status_code == 200
        harness.jobs.jobs["job-1"].status = _status("failed")
        harness.jobs.attempts["job-1"] = [
            _attempt("failed", "frontend_session lane not configured")
        ]
        status = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert (status["state"], status["job"]["status"]) == ("failed", "failed")
        assert "lane not configured" in status["reason"]
        assert harness.pool.table[("alice", "sess-1")]["state"] == "error"
        # a deliberate restart is allowed again
        assert _start(harness).status_code == 200

    def test_start_refuses_without_job_store(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.service.job_store = None
        r = _start(harness)
        assert r.status_code == 503
        assert r.headers["x-platform-error-code"] == "companionvenue.unavailable"
        assert harness.service.store.get("sess-1") is None

    def test_start_refuses_unresolved_site(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.service.site_name = ""
        r = _start(harness)
        assert r.status_code == 503 and "site" in r.json()["detail"]
        assert harness.jobs.created == []

    def test_start_enforces_spawn_budget(self, harness, monkeypatch):
        from umrs.api.agent_quota import enforce_spawn_budget

        harness.pool.seed("alice", "sess-1")
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "1")
        monkeypatch.setattr("umrs.api.agent_quota.count_active_agent_units", lambda *_a, **_k: 1)
        harness.service._quota_check = lambda uid: enforce_spawn_budget(None, user_id=uid)
        r = _start(harness)
        assert r.status_code == 503, r.text
        assert r.headers["x-platform-error-code"] == "quota.cpu_process_exceeded"
        assert harness.jobs.created == []

    def test_status_owner_scoped(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        other = harness.client.get(
            "/umrs/companion/sessions/sess-1/venue", headers={"x-user-id": "mallory"}
        )
        assert other.status_code == 404
        mine = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H)
        assert mine.status_code == 200 and mine.json()["state"] == "running"

    def test_stop_tears_down_and_revokes(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        r = harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert (r.status_code, r.json()["state"]) == (200, "stopped")
        flattened = [" ".join(c) for c in harness.podman.calls]
        assert any("rm -f worker-sess-1" in c for c in flattened)
        assert all(not lease.is_valid for lease in harness.broker.list_leases(scope="inference"))
        assert harness.pool.table[("alice", "sess-1")]["state"] == "stopped"

    def test_stop_settles_the_job_and_is_idempotent(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert harness.jobs.cancelled == [("job-1", "stopped by user")]
        again = harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert (again.status_code, again.json()["state"]) == (200, "stopped")
        assert len(harness.jobs.cancelled) == 1  # the second stop cancels nothing new

    def test_stop_survives_a_job_already_terminal(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        harness.jobs.cancel_raises = True
        r = harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        assert r.status_code == 200 and r.json()["state"] == "stopped"


@pytest.fixture
def rail(harness):
    captured = {}

    class _Podman(FakePodman):
        def __call__(self, *args, input_text=None, **kw):
            if args[:2] == ("secret", "create"):
                captured["token"] = input_text
            return super().__call__(*args, input_text=input_text, **kw)

    podman = _Podman()
    harness.podman = podman
    harness.service.provisioner = FrontendPodmanProvisioner(
        harness.broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="venue-rail-"))
    )
    harness.pool.seed(
        "alice",
        "sess-1",
        queue=[{"id": "q-1", "content": "hello remote", "client_id": "tui", "added_at": "t"}],
    )
    assert _start(harness).status_code == 200
    assert _run_lane(harness) == [("running", "job-1")]
    harness.token = captured["token"]
    harness.rail_headers = {"authorization": f"Bearer {harness.token}"}
    return harness


class TestRail:
    def test_rail_rejects_without_token(self, rail):
        r = rail.client.get("/umrs/companion/sessions/sess-1/rail/snapshot")
        assert r.status_code == 401
        assert r.json()["detail"] == "token_invalid"

    def test_rail_rejects_user_identity_header_alone(self, rail):
        r = rail.client.get("/umrs/companion/sessions/sess-1/rail/snapshot", headers=H)
        assert r.status_code == 401

    def test_rail_rejects_other_session(self, rail):
        r = rail.client.get(
            "/umrs/companion/sessions/other/rail/snapshot", headers=rail.rail_headers
        )
        assert r.status_code == 401 and r.json()["detail"] == "sid_mismatch"

    def test_rail_snapshot_roundtrip(self, rail):
        r = rail.client.get(
            "/umrs/companion/sessions/sess-1/rail/snapshot", headers=rail.rail_headers
        )
        assert r.status_code == 200
        assert r.json()["session_id"] == "sess-1"
        put = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/snapshot",
            headers=rail.rail_headers,
            json={
                "name": "S",
                "updated_at": "2026-09-21T01:00:00+00:00",
                "payload": json.dumps(
                    {"id": "sess-1", "messages": [{"role": "user", "content": "x"}]}
                ),
                "venue": "local",  # ignored: rail pins frontend
                "state": "ongoing",
            },
        )
        assert put.status_code == 200 and put.json()["applied"] is True
        row = rail.pool.table[("alice", "sess-1")]
        assert row["venue"] == "frontend" and row["state"] == "ongoing"

    def test_rail_queue_pop(self, rail):
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/queue/pop", headers=rail.rail_headers
        )
        assert r.status_code == 200
        body = r.json()
        assert body["applied"] is True
        assert body["front"]["content"] == "hello remote"
        assert body["queue"] == []
        empty = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/queue/pop", headers=rail.rail_headers
        )
        assert empty.json()["front"] is None

    def test_rail_heartbeat_renews_and_records_runtime(self, rail):
        lease_before = next(iter(rail.broker.list_leases(scope="inference")))
        expires_before = lease_before.expires_at
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat",
            headers=rail.rail_headers,
            json={"phase": "running", "turn_seq": 2, "partial": "thinking…", "extend_seconds": 300},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["renewed"] is True and body["stop_requested"] is False
        assert rail.broker.get_lease(lease_before.lease_id).expires_at > expires_before
        status = rail.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert status["runtime"]["phase"] == "running"
        assert status["runtime"]["turn_seq"] == 2
        assert status["runtime"]["partial"] == "thinking…"

    def test_rail_heartbeat_carries_live_tool_activity(self, rail):
        # TUI visibility (2026-09-27): the worker's bounded activity lines
        # land verbatim on the row's runtime; honest absence otherwise.
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat",
            headers=rail.rail_headers,
            json={
                "phase": "running",
                "turn_seq": 3,
                "partial": "text…",
                "tool_activity": ["▸ hpc_submit sbatch run.slurm", "▸ rag_search 'defect' ✓ 1.2s"],
            },
        )
        assert r.status_code == 200
        status = rail.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert status["runtime"]["tool_activity"] == [
            "▸ hpc_submit sbatch run.slurm",
            "▸ rag_search 'defect' ✓ 1.2s",
        ]
        # A later plain heartbeat without the field keeps the last lines.
        rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat",
            headers=rail.rail_headers,
            json={"phase": "running", "turn_seq": 4, "partial": "more…"},
        )
        again = rail.client.get(
            "/umrs/companion/sessions/sess-1/venue", headers=H
        ).json()
        assert again["runtime"]["tool_activity"][0] == "▸ hpc_submit sbatch run.slurm"

    def test_rail_heartbeat_after_stop_requests_exit(self, rail):
        rail.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/heartbeat", headers=rail.rail_headers, json={}
        )
        # lease revoked by stop → the rail refuses the principal outright
        assert r.status_code == 401 and r.json()["detail"] == "lease_revoked"

    def test_rail_approvals_cycle_with_owner_decision(self, rail):
        r = rail.client.post(
            "/umrs/companion/sessions/sess-1/rail/approvals",
            headers=rail.rail_headers,
            json={"turn_id": "t3", "action": {"tool": "shell_exec", "description": "ls"}},
        )
        assert r.status_code == 200
        approval = r.json()["approval"]
        assert approval["status"] == "pending" and approval["user_id"] == "alice"
        aid = approval["approval_id"]
        pending = rail.client.get(
            f"/umrs/companion/sessions/sess-1/rail/approvals/{aid}", headers=rail.rail_headers
        )
        assert pending.status_code == 200 and pending.json()["approval"]["status"] == "pending"
        rail.approvals.decide(aid, "once", decided_by="alice")
        decided = rail.client.get(
            f"/umrs/companion/sessions/sess-1/rail/approvals/{aid}", headers=rail.rail_headers
        ).json()["approval"]
        assert decided["status"] == "approved" and decided["decision"] == "once"
        cross = rail.client.get(
            f"/umrs/companion/sessions/other/rail/approvals/{aid}", headers=rail.rail_headers
        )
        assert cross.status_code == 401  # sid_mismatch before any lookup


class TestRailGateway:
    """Platform verbs for the sandbox: lease → minted gateway identity → /umrs/<family>."""

    @pytest.fixture
    def gw(self, rail, monkeypatch):
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-handshake-secret")
        return rail

    def test_workflows_family_reaches_the_route_with_minted_identity(self, gw):
        r = gw.client.post(
            "/umrs/companion/sessions/sess-1/rail/gw/workflows/run?wait=true",
            headers={**gw.rail_headers, "x-user-id": "mallory", "x-verified-is-admin": "true"},
            json={"workflow_name": "wf-a"},
        )
        assert r.status_code == 200, r.text
        seen = r.json()
        assert (seen["path"], seen["method"], seen["query"]) == (
            "/umrs/workflows/run",
            "POST",
            {"wait": "true"},
        )
        # identity = the lease owner, never the sandbox's header; the session
        # token never reaches the platform route
        assert (seen["user"], seen["admin"], seen["auth"]) == ("alice", "false", None)
        assert json.loads(seen["body"]) == {"workflow_name": "wf-a"}
        from r4_contracts.handshake_hmac import verify_claims

        assert verify_claims("test-handshake-secret", seen["hmac"], "alice", seen["project"], False)

    def test_project_comes_from_the_lease(self, gw):
        lease = next(iter(gw.broker.list_leases(scope="inference")))
        lease.project = "proj-9"
        gw.broker._leases[lease.lease_id] = lease
        r = gw.client.get(
            "/umrs/companion/sessions/sess-1/rail/gw/jobs/j-1", headers=gw.rail_headers
        )
        assert r.status_code == 200
        assert r.json()["project"] == "proj-9"

    def test_disallowed_family_is_refused(self, gw):
        r = gw.client.get(
            "/umrs/companion/sessions/sess-1/rail/gw/secrets/probe", headers=gw.rail_headers
        )
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "companionvenue.forbidden"

    def test_gw_requires_the_lease(self, gw):
        r = gw.client.get("/umrs/companion/sessions/sess-1/rail/gw/jobs/j-1")
        assert r.status_code == 401

    def test_hook_sees_the_upstream_answer(self, gw):
        seen: list = []
        gw.client.app.state.companion_venue_rail_gw_hook = lambda p, m, path, up: seen.append(
            (p.owner, m, path, up.status_code)
        )
        gw.client.post(
            "/umrs/companion/sessions/sess-1/rail/gw/jobs/submit", headers=gw.rail_headers, json={}
        )
        assert seen == [("alice", "POST", "jobs/submit", 200)]

    def test_created_job_gets_the_session_origin(self, gw):
        r = gw.client.post(
            "/umrs/companion/sessions/sess-1/rail/gw/jobs",
            headers=gw.rail_headers,
            json={"job_type": "workflow_run"},
        )
        assert r.status_code == 200 and r.json()["job_id"] == "job-9"
        assert gw.jobs.stamped == [("job-9", {"session_id": "sess-1", "venue": "frontend"})]

    def test_workflow_run_origin_resolves_via_execution_id(self, gw):
        gw.jobs.by_spec[("workflow_run", "execution_id", "exec-1")] = "job-w"
        r = gw.client.post(
            "/umrs/companion/sessions/sess-1/rail/gw/workflows/wf-1/run",
            headers=gw.rail_headers,
            json={},
        )
        assert r.status_code == 200
        assert gw.jobs.stamped == [("job-w", {"session_id": "sess-1", "venue": "frontend"})]

    def test_reads_and_unresolved_answers_stamp_nothing(self, gw):
        gw.client.get("/umrs/companion/sessions/sess-1/rail/gw/jobs/j-1", headers=gw.rail_headers)
        gw.client.post(
            "/umrs/companion/sessions/sess-1/rail/gw/jobs/submit", headers=gw.rail_headers, json={}
        )
        assert gw.jobs.stamped == []


class TestRailGatewayD42ReadGating:
    """D-42: knowledge/catalog/scientific-catalogs/rag ride the rail READ-ONLY
    — a sandbox retrieves; writes stay on the lease owner's own surfaces."""

    @pytest.fixture
    def gw(self, rail, monkeypatch):
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-handshake-secret")
        return rail

    @pytest.mark.parametrize("family", ["knowledge", "catalog", "scientific-catalogs", "rag"])
    def test_read_families_get_reaches_upstream(self, gw, family):
        r = gw.client.get(
            f"/umrs/companion/sessions/sess-1/rail/gw/{family}/probe",
            headers=gw.rail_headers,
        )
        assert r.status_code == 200, (family, r.text)
        assert r.json()["path"] == f"/umrs/{family}/probe"

    @pytest.mark.parametrize("family", ["knowledge", "catalog", "scientific-catalogs", "rag"])
    def test_read_families_post_refused_before_upstream(self, gw, family):
        r = gw.client.post(
            f"/umrs/companion/sessions/sess-1/rail/gw/{family}/probe",
            headers=gw.rail_headers,
            json={},
        )
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "companionvenue.forbidden"

    def test_read_families_delete_refused(self, gw):
        r = gw.client.delete(
            "/umrs/companion/sessions/sess-1/rail/gw/knowledge/nodes/n-1",
            headers=gw.rail_headers,
        )
        assert r.status_code == 403

    def test_workflows_jobs_keep_write_verbs(self, gw):
        for family in ("workflows", "jobs"):
            r = gw.client.post(
                f"/umrs/companion/sessions/sess-1/rail/gw/{family}/probe",
                headers=gw.rail_headers,
                json={},
            )
            assert r.status_code == 200, (family, r.text)


class TestPauseResumeRoutes:
    def test_pause_resume_round_trip(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        podman_calls_before = list(harness.podman.calls)
        p = harness.client.post("/umrs/companion/sessions/sess-1/venue/pause", headers=H)
        assert (p.status_code, p.json()["state"]) == (200, "paused")
        # the sandbox stays up (pause touches NOTHING — no teardown, no job cancel)
        assert harness.podman.calls == podman_calls_before
        assert harness.jobs.cancelled == []
        st = harness.client.get("/umrs/companion/sessions/sess-1/venue", headers=H).json()
        assert st["state"] == "paused"
        r = harness.client.post("/umrs/companion/sessions/sess-1/venue/resume", headers=H)
        assert (r.status_code, r.json()["state"]) == (200, "running")
        # the mirror's running-state spelling is 'ongoing' (0084 vocabulary);
        # the venue row's own state is 'running' (VALID_STATES vocabulary).
        assert harness.pool.table[("alice", "sess-1")]["state"] == "ongoing"

    def test_pause_idempotent_and_resume_conflict(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        first = harness.client.post("/umrs/companion/sessions/sess-1/venue/pause", headers=H)
        again = harness.client.post("/umrs/companion/sessions/sess-1/venue/pause", headers=H)
        assert (first.status_code, again.status_code) == (200, 200)
        resumed = harness.client.post("/umrs/companion/sessions/sess-1/venue/resume", headers=H)
        twice = harness.client.post("/umrs/companion/sessions/sess-1/venue/resume", headers=H)
        assert (resumed.status_code, twice.status_code) == (200, 409)

    def test_pause_refused_from_terminal(self, harness):
        harness.pool.seed("alice", "sess-1")
        _start(harness)
        _run_lane(harness)
        harness.client.post("/umrs/companion/sessions/sess-1/venue/stop", headers=H)
        r = harness.client.post("/umrs/companion/sessions/sess-1/venue/pause", headers=H)
        assert r.status_code == 409


class TestDataClassGate:
    """ADR-062 (D-51 gate, first seam): sensitive projects refuse public-kind
    sources at venue start; no project claim → no gate (pre-gate behavior)."""

    def test_sensitive_project_refuses_public_sources(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.pool.projects = {"proj-s": "sensitive"}
        r = harness.client.post(
            "/umrs/companion/sessions/sess-1/venue/start",
            json={"source_refs": ["openai-main"], "project": "proj-s"},
            headers=H,
        )
        assert r.status_code == 403
        assert r.headers["x-platform-error-code"] == "companionvenue.source_class_refused"
        assert "sensitive" in r.json()["detail"]
        assert "openai-main" in r.json()["detail"]

    def test_non_sensitive_project_allows_public_sources(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.pool.projects = {"proj-n": "non_sensitive"}
        r = harness.client.post(
            "/umrs/companion/sessions/sess-1/venue/start",
            json={"source_refs": ["openai-main"], "project": "proj-n"},
            headers=H,
        )
        assert r.status_code == 200, r.text

    def test_no_project_claim_no_gate(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.pool.projects = {"proj-s": "sensitive"}
        r = harness.client.post(
            "/umrs/companion/sessions/sess-1/venue/start",
            json={"source_refs": ["openai-main"]},
            headers=H,
        )
        assert r.status_code == 200, r.text  # no claim → pre-gate behavior

    def test_unknown_project_no_gate(self, harness):
        harness.pool.seed("alice", "sess-1")
        harness.pool.projects = {}
        r = harness.client.post(
            "/umrs/companion/sessions/sess-1/venue/start",
            json={"source_refs": ["openai-main"], "project": "proj-ghost"},
            headers=H,
        )
        assert r.status_code == 200, r.text  # absent row → no gate (never invented)
