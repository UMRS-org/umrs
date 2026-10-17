"""H-1 (hardening plan, 2026-10-17), item 5, streams lane: the knowledge
events SSE router and the web dashboard router took no identity.

- GET /umrs/knowledge/projects/{project_id}/events (and /events/health):
  any caller subscribed to any project's knowledge events.
- /umrs/web/dashboard/*: /jobs, /jobs/{job_id}, /models/{model_id},
  /notifications, /workflows, /overview, /knowledge?project_id and the SSE
  snapshot channels served every project's rows to any caller.

Strict mode with gateway-signed headers; a foreign object reads as not found.
Streams never end, so they are driven through raw ASGI until the first body
frame, then disconnected.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-streams-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
ROOT = _headers("root", "ops", admin=True)


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


async def _first_frame(app, path: str, headers: dict[str, str] | None = None):
    """(status, first body bytes) of a possibly endless response."""
    started: dict[str, int] = {}
    body = bytearray()
    done = asyncio.Event()
    sent_request = False

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await done.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            started["status"] = message["status"]
        elif message["type"] == "http.response.body":
            body.extend(message.get("body", b""))
            if body or not message.get("more_body", False):
                done.set()

    path_only, _, query = path.partition("?")
    scope = {
        "type": "http",
        "method": "GET",
        "path": path_only,
        "raw_path": path_only.encode(),
        "query_string": query.encode(),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "http_version": "1.1",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1234),
        "root_path": "",
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return started.get("status"), bytes(body)


# ── knowledge events SSE ─────────────────────────────────────────────────────


@pytest.fixture
def events_app(monkeypatch):
    from umrs.knowledge import events_sse

    # The module-level broadcast queue is bound to the first test loop that
    # touched it; the fan-out loop is not under test here.
    monkeypatch.setattr(events_sse, "_start_broadcast_loop", lambda: None)

    saved_clients = dict(events_sse._clients)
    saved_projects = dict(events_sse._projects)
    events_sse._clients.clear()
    events_sse._projects.clear()
    app = FastAPI()
    app.include_router(events_sse.router)
    yield app
    events_sse._clients.clear()
    events_sse._clients.update(saved_clients)
    events_sse._projects.clear()
    events_sse._projects.update(saved_projects)


def _seed_listener(project: str) -> None:
    from umrs.knowledge import events_sse

    cid = f"seed-{project}"
    events_sse._clients[cid] = {
        "project": project,
        "queue": asyncio.Queue(),
        "connected_at": 1e18,  # never evicted
        "last_event_id": None,
        "buffer": [],
    }
    events_sse._projects[project].append(cid)


async def test_events_stream_needs_identity_in_strict_mode(events_app):
    from umrs.knowledge import events_sse

    status, _ = await _first_frame(events_app, "/umrs/knowledge/projects/proj-b/events")
    assert status == 401
    assert events_sse._clients == {}


async def test_events_stream_of_another_project_is_refused(events_app):
    from umrs.knowledge import events_sse

    status, _ = await _first_frame(events_app, "/umrs/knowledge/projects/proj-b/events", ALICE)
    assert status == 403
    assert events_sse._clients == {}


async def test_events_stream_of_own_project_opens(events_app):
    status, body = await _first_frame(events_app, "/umrs/knowledge/projects/proj-a/events", ALICE)
    assert status == 200
    assert body.startswith(b": heartbeat")


def test_events_health_needs_identity_and_readable_project(events_app):
    client = TestClient(events_app)
    path = "/umrs/knowledge/projects/proj-b/events/health"
    assert client.get(path).status_code == 401
    assert client.get(path, headers=ALICE).status_code == 403


def test_events_health_hides_other_projects_counts(events_app):
    _seed_listener("proj-b")
    client = TestClient(events_app)
    own = client.get("/umrs/knowledge/projects/proj-a/events/health", headers=ALICE)
    assert own.status_code == 200
    assert own.json()["project_clients"] == 0
    assert own.json()["total_clients"] is None
    admin = client.get("/umrs/knowledge/projects/proj-b/events/health", headers=ROOT)
    assert admin.json()["total_clients"] == 1
    assert admin.json()["project_clients"] == 1


# ── web dashboard ────────────────────────────────────────────────────────────


def _job(job_id: str, project: str, user: str):
    from umrs.jobs.models import JobStatus, JobType

    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        job_id=job_id,
        project_id=project,
        user_id=user,
        status=JobStatus.RUNNING,
        job_type=next(iter(JobType)),
        ttl_seconds=None,
        fan_out_child_job_ids=[],
        created_at=now,
        updated_at=now,
        active_attempt_id=None,
    )


class _FakeJobStore:
    def __init__(self, jobs):
        self.jobs = {j.job_id: j for j in jobs}
        self.attempt_reads: list[str] = []

    def list_jobs(self, project_id=None, limit=100, offset=0, user_id=None, **_filters):
        rows = [
            j
            for j in self.jobs.values()
            if (not project_id or j.project_id == project_id)
            and (user_id is None or j.user_id == user_id)
        ]
        return rows[offset : offset + limit]

    def get_job(self, job_id):
        from umrs.jobs.store import JobNotFoundError

        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        return self.jobs[job_id]

    def get_attempts_for_job(self, job_id):
        self.attempt_reads.append(job_id)
        return []


def _model(model_id: str, scope: str, project: str | None, owner: str):
    from umrs.model_registry.models import ModelScope, ModelStatus, ModelType

    return SimpleNamespace(
        model_id=model_id,
        name=model_id,
        scope=ModelScope(scope),
        project_id=project,
        owner_id=owner,
        model_type=next(iter(ModelType)),
        status=next(iter(ModelStatus)),
        version=None,
        created_at=None,
    )


class _FakeModelStore:
    """Mirrors ModelStore: the ACL applies only when a ctx is passed."""

    def __init__(self, models):
        self.models = {m.model_id: m for m in models}

    @staticmethod
    def _readable(model, ctx) -> bool:
        if ctx is None or ctx.is_admin or model.scope.value == "system":
            return True
        if model.scope.value == "project":
            return model.project_id in ctx.project_memberships
        return model.owner_id == ctx.user_id

    def get_model(self, model_id, ctx=None):
        from umrs.model_registry.store.errors import ModelNotFoundError, PermissionDeniedError

        model = self.models.get(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)
        if not self._readable(model, ctx):
            raise PermissionDeniedError(ctx.user_id, model_id, "read", "denied")
        return model

    def get_model_versions(self, *_args, **_kwargs):
        return SimpleNamespace(versions=[], latest=None)

    def list_models(self, ctx, limit=100, offset=0, **_filters):
        visible = [m for m in self.models.values() if self._readable(m, ctx)]
        return visible[offset : offset + limit]


_NOTIFICATIONS = [
    {
        "id": 1,
        "kind": "budget",
        "title": "a stop",
        "detail": {},
        "project_id": "proj-a",
        "user_id": "alice",
    },
    {
        "id": 2,
        "kind": "budget",
        "title": "b stop",
        "detail": {},
        "project_id": "proj-b",
        "user_id": "bob",
    },
]


class _FakeNotificationStore:
    def __init__(self, _conn):
        pass

    def list(self, *, user_id=None, project_id=None, limit=100, **_filters):
        return [
            dict(r)
            for r in _NOTIFICATIONS
            if (not project_id or r["project_id"] == project_id)
            and (not user_id or r["user_id"] == user_id)
        ][:limit]


class _FakePool:
    def getconn(self):
        return object()

    def putconn(self, _conn):
        pass


@pytest.fixture
def dash(monkeypatch):
    from umrs.api import web_dashboard
    from umrs.server.workflow_execution_registry import (
        get_execution_registry,
        reset_execution_registry_for_tests,
    )

    from umrs import platform_notifications

    jobs = _FakeJobStore([_job("job-a", "proj-a", "alice"), _job("job-b", "proj-b", "bob")])
    models = _FakeModelStore(
        [
            _model("m-sys", "system", None, "root"),
            _model("m-a", "project", "proj-a", "alice"),
            _model("m-b", "project", "proj-b", "bob"),
        ]
    )
    monkeypatch.setattr(web_dashboard, "get_job_store", lambda: jobs)
    monkeypatch.setattr(web_dashboard, "get_optional_job_store", lambda: jobs)
    monkeypatch.setattr(web_dashboard, "get_optional_model_store", lambda: models)
    monkeypatch.setattr("umrs.api.models.get_model_store", lambda: models)
    monkeypatch.setattr(platform_notifications, "NotificationStore", _FakeNotificationStore)

    monkeypatch.setenv("UMRS_WORKFLOW_EXECUTION_REGISTRY", "memory")
    reset_execution_registry_for_tests()
    registry = get_execution_registry()
    registry.create(
        workflow_name="wf-alice",
        user_id="alice",
        params={"project_id": "proj-a"},
        identity_provenance="gateway_verified",
    )
    registry.create(
        workflow_name="wf-bob",
        user_id="bob",
        params={"project_id": "proj-b"},
        identity_provenance="gateway_verified",
    )

    app = FastAPI()
    app.state.umrs_db_pool = _FakePool()
    app.include_router(web_dashboard.router)
    yield app, jobs
    reset_execution_registry_for_tests()


_IDENTITY_ROUTES = [
    "/umrs/web/dashboard/overview",
    "/umrs/web/dashboard/jobs",
    "/umrs/web/dashboard/jobs/job-a",
    "/umrs/web/dashboard/models",
    "/umrs/web/dashboard/models/m-a",
    "/umrs/web/dashboard/notifications",
    "/umrs/web/dashboard/workflows",
    "/umrs/web/dashboard/connectors",
    "/umrs/web/dashboard/knowledge?project_id=proj-a",
    "/umrs/web/dashboard/sse/jobs",
    "/umrs/web/dashboard/sse/workflows",
    "/umrs/web/dashboard/sse/intake",
]


@pytest.mark.parametrize("path", _IDENTITY_ROUTES)
async def test_dashboard_routes_need_identity_in_strict_mode(dash, path):
    app, _ = dash
    status, _ = await _first_frame(app, path)
    assert status == 401


def test_dashboard_health_stays_open(dash):
    app, _ = dash
    assert TestClient(app).get("/umrs/web/dashboard/health").status_code == 200


def test_dashboard_job_list_is_the_callers_own(dash):
    app, _ = dash
    client = TestClient(app)
    own = client.get("/umrs/web/dashboard/jobs", headers=ALICE).json()
    assert [j["job_id"] for j in own["items"]] == ["job-a"]
    # A user filter cannot widen the view to another user's jobs.
    other = client.get("/umrs/web/dashboard/jobs", params={"user_id": "bob"}, headers=ALICE).json()
    assert other["items"] == []
    admin = client.get("/umrs/web/dashboard/jobs", headers=ROOT).json()
    assert {j["job_id"] for j in admin["items"]} == {"job-a", "job-b"}


def test_dashboard_foreign_job_reads_as_not_found(dash):
    app, jobs = dash
    client = TestClient(app)
    resp = client.get("/umrs/web/dashboard/jobs/job-b", headers=ALICE)
    assert resp.status_code == 404
    assert "job-b" not in jobs.attempt_reads
    assert client.get("/umrs/web/dashboard/jobs/job-a", headers=ALICE).status_code == 200
    assert client.get("/umrs/web/dashboard/jobs/job-b", headers=ROOT).status_code == 200


def test_dashboard_foreign_model_reads_as_not_found(dash):
    app, _ = dash
    client = TestClient(app)
    assert client.get("/umrs/web/dashboard/models/m-b", headers=ALICE).status_code == 404
    assert client.get("/umrs/web/dashboard/models/m-a", headers=ALICE).status_code == 200
    assert client.get("/umrs/web/dashboard/models/m-sys", headers=ALICE).status_code == 200
    assert client.get("/umrs/web/dashboard/models/m-b", headers=ROOT).status_code == 200


def test_dashboard_model_list_hides_other_projects(dash):
    app, _ = dash
    body = TestClient(app).get("/umrs/web/dashboard/models", headers=ALICE).json()
    assert {m["model_id"] for m in body["items"]} == {"m-sys", "m-a"}


def test_dashboard_notifications_are_the_callers_projects(dash):
    app, _ = dash
    client = TestClient(app)
    own = client.get("/umrs/web/dashboard/notifications", headers=ALICE).json()
    assert [n["details"]["project_id"] for n in own["notifications"]] == ["proj-a"]
    admin = client.get("/umrs/web/dashboard/notifications", headers=ROOT).json()
    assert admin["total"] == 2


def test_dashboard_knowledge_panel_of_another_project_is_refused(dash):
    app, _ = dash
    app.state.umrs_db_pool = None  # degraded payload: the check must come first
    client = TestClient(app)
    resp = client.get(
        "/umrs/web/dashboard/knowledge", params={"project_id": "proj-b"}, headers=ALICE
    )
    assert resp.status_code == 403
    own = client.get(
        "/umrs/web/dashboard/knowledge", params={"project_id": "proj-a"}, headers=ALICE
    )
    assert own.status_code == 200


def test_dashboard_workflows_panel_is_the_callers_own(dash):
    app, _ = dash
    client = TestClient(app)
    own = client.get("/umrs/web/dashboard/workflows", headers=ALICE).json()
    assert [r["workflow_name"] for r in own["recent"]] == ["wf-alice"]
    assert own["counts"] == {"submitted": 1}
    admin = client.get("/umrs/web/dashboard/workflows", headers=ROOT).json()
    assert {r["workflow_name"] for r in admin["recent"]} == {"wf-alice", "wf-bob"}


def test_dashboard_overview_counts_only_visible_rows(dash):
    app, _ = dash
    body = TestClient(app).get("/umrs/web/dashboard/overview", headers=ALICE).json()
    assert body["workflows"]["total"] == 1
    assert body["total_models"] == 2  # system + own project


async def test_dashboard_workflow_snapshot_is_the_callers_own(dash):
    app, _ = dash
    status, body = await _first_frame(app, "/umrs/web/dashboard/sse/workflows", ALICE)
    assert status == 200
    header, data = body.decode().split("\n\n", 1)[0].split("\n", 1)
    assert header == "event: workflows-snapshot"
    payload = json.loads(data.removeprefix("data: ").strip())
    assert [r["workflow_name"] for r in payload["recent"]] == ["wf-alice"]


def test_dashboard_connector_sync_rows_are_admin_only(dash, monkeypatch):
    from umrs.connectors.sync_state import InMemorySyncStateStore, SyncStateRecord

    store = InMemorySyncStateStore()
    store.put(SyncStateRecord(name="lab-slack", connector_type="slack", last_status="ok"))
    monkeypatch.setattr("umrs.connectors.sync_state.build_default_store", lambda: store)
    app, _ = dash
    client = TestClient(app)
    assert client.get("/umrs/web/dashboard/connectors", headers=ALICE).json()["connectors"] == {}
    admin = client.get("/umrs/web/dashboard/connectors", headers=ROOT).json()
    assert set(admin["connectors"]) == {"lab-slack"}
    overview = client.get("/umrs/web/dashboard/overview", headers=ALICE).json()
    assert overview["connectors"]["sync"] == {}


# ── forged admin claims and unsigned memberships ─────────────────────────────

# X-Verified-Is-Admin flipped to "true" over a signature made for a non-admin,
# and the same flag with no signature at all: both must be refused (401),
# never honoured as an admin view.
_FORGED_ADMIN = {
    "X-User-ID": "alice",
    "X-Verified-Project-ID": "proj-a",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": sign_claims(SECRET, "alice", "proj-a", False),
}
_UNSIGNED_ADMIN = {
    "X-User-ID": "alice",
    "X-Verified-Project-ID": "proj-a",
    "X-Verified-Is-Admin": "true",
}


@pytest.mark.parametrize("headers", [_FORGED_ADMIN, _UNSIGNED_ADMIN], ids=["forged", "unsigned"])
@pytest.mark.parametrize(
    "path", ["/umrs/web/dashboard/jobs/job-b", "/umrs/web/dashboard/connectors"]
)
def test_dashboard_forged_admin_flag_is_refused(dash, headers, path):
    app, jobs = dash
    assert TestClient(app).get(path, headers=headers).status_code == 401
    assert "job-b" not in jobs.attempt_reads


@pytest.mark.parametrize("headers", [_FORGED_ADMIN, _UNSIGNED_ADMIN], ids=["forged", "unsigned"])
def test_events_health_forged_admin_flag_is_refused(events_app, headers):
    _seed_listener("proj-b")
    resp = TestClient(events_app).get(
        "/umrs/knowledge/projects/proj-b/events/health", headers=headers
    )
    assert resp.status_code == 401


@pytest.mark.parametrize("headers", [_FORGED_ADMIN, _UNSIGNED_ADMIN], ids=["forged", "unsigned"])
async def test_events_stream_forged_admin_flag_is_refused(events_app, headers):
    from umrs.knowledge import events_sse

    status, _ = await _first_frame(events_app, "/umrs/knowledge/projects/proj-b/events", headers)
    assert status == 401
    assert events_sse._clients == {}


def test_dashboard_unsigned_memberships_do_not_widen_model_access(dash):
    """X-Verified-Project-Memberships is not covered by the gateway signature,
    so the dashboard's model ACL uses the signed project only (deliberately
    stricter than GET /umrs/models)."""
    app, _ = dash
    client = TestClient(app)
    widened = {**ALICE, "X-Verified-Project-Memberships": "proj-a,proj-b"}
    assert client.get("/umrs/web/dashboard/models/m-b", headers=widened).status_code == 404
    body = client.get("/umrs/web/dashboard/models", headers=widened).json()
    assert {m["model_id"] for m in body["items"]} == {"m-sys", "m-a"}
