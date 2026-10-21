"""Cycle-2 umrs-follow lane (N2, 2026-10-21): a foreign id reads like an unknown one.

Found live by the umrs-api lane (gateway -> auth sidecar -> umrs, two
projects): bob in ``pb`` asking for alice's ``pa`` job or model got a 403,
a random id a 404 -- an existence oracle. The H-1 / ADR-086 existence-hiding
rule (already applied to projects, traces and A2L) says a foreign id answers
exactly like an unknown one: same status, same body, same error code. Admins
keep their view.

Each case below asks one route twice as bob -- once with alice's real id,
once with a random id -- and asserts the two answers are indistinguishable
once the id itself is masked out of the body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import jobs as jobs_api
from umrs.api import provenance as provenance_api
from umrs.api import traces as traces_api
from umrs.api.jobs import CallerIdentity
from umrs.api.models import get_model_store
from umrs.api.models import router as models_router
from umrs.jobs.store import JobNotFoundError
from umrs.model_registry.store import ModelStore
from umrs.model_registry.store.acl import AccessContext
from umrs.model_registry.store.errors import (
    HiddenModelError,
    ModelNotFoundError,
    PermissionDeniedError,
)

from r4_contracts.handshake_hmac import sign_claims, sign_memberships

JOB_ID = "job-alice00000001"
RANDOM_JOB = "job-zzzzzzzzzzzz"
MODEL_ID = "mdl-alice0000001"
RANDOM_MODEL = "mdl-zzzzzzzzzzzz"
SECRET = "c2-existence-secret"
NOW = datetime(2026, 10, 21, 12, 0, 0)

ALICE = CallerIdentity(user_id="alice", project_id="pa")
BOB = CallerIdentity(user_id="bob", project_id="pb")
ADMIN = CallerIdentity(user_id="root", project_id="ops", is_admin=True)


def _shape(resp: Any, ident: str) -> tuple[int, str, str]:
    """What a caller can observe, with the asked-for id masked out."""
    return (
        resp.status_code,
        resp.text.replace(ident, "<id>"),
        resp.headers.get("x-platform-error-code", ""),
    )


# ── jobs, provenance, traces ───────────────────────────────────────────────


class _Job:
    job_id = JOB_ID
    project_id = "pa"
    user_id = "alice"
    spec: dict[str, Any] = {}
    commit_attempt_id = None


class _Attempt:
    attempt_id = "att-1"
    attempt_no = 1


class _JobStore:
    """One job (alice's); every other id is unknown."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_job(self, job_id: str) -> _Job:
        self.calls.append(f"get_job:{job_id}")
        if job_id != JOB_ID:
            raise JobNotFoundError(job_id)
        return _Job()

    def get_attempts_for_job(self, job_id: str) -> list[_Attempt]:
        self.calls.append(f"attempts:{job_id}")
        return [_Attempt()]

    def get_attempt_manifest(self, _attempt_id: str) -> dict[str, Any]:
        return {"environment": {}}

    def get_latest_attempt_manifest(self, _job_id: str) -> dict[str, Any]:
        return {"environment": {}}


class _TraceStore:
    def get_traces_for_job(self, _job_id: str) -> list[Any]:
        return []


def _jobs_client(store: _JobStore, caller: CallerIdentity) -> TestClient:
    app = FastAPI()
    for module in (jobs_api, provenance_api, traces_api):
        app.include_router(module.router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    app.dependency_overrides[traces_api.get_trace_store] = _TraceStore
    return TestClient(app)


JOB_ROUTES = [
    ("GET", "/umrs/jobs/{id}", None),
    ("GET", "/umrs/jobs/{id}/manifest", None),
    ("GET", "/umrs/jobs/{id}/manifests", None),
    ("POST", "/umrs/jobs/{id}/cancel", None),
    ("POST", "/umrs/jobs/{id}/attempts/att-1/credentials", None),
    ("POST", "/umrs/jobs/{id}/attempts/att-1/credentials/sts", None),
    ("POST", "/umrs/jobs/{id}/attempts/att-1/presign", {"key": "k", "method": "get"}),
    ("POST", "/umrs/jobs/{id}/reproduce", None),
    ("GET", "/umrs/provenance/{id}", None),
    ("GET", "/umrs/provenance/{id}/lineage", None),
    ("GET", "/umrs/traces/job/{id}", None),
    ("POST", "/umrs/traces", {"job_id": "{id}", "step_name": "s", "step_type": "tool"}),
]


def _fill(value: Any, ident: str) -> Any:
    if isinstance(value, str):
        return value.replace("{id}", ident)
    if isinstance(value, dict):
        return {k: _fill(v, ident) for k, v in value.items()}
    return value


def _ask(client: TestClient, method: str, path: str, body: Any, ident: str) -> Any:
    return client.request(method, _fill(path, ident), json=_fill(body, ident))


@pytest.mark.parametrize(("method", "path", "body"), JOB_ROUTES)
def test_foreign_job_reads_like_an_unknown_job(method: str, path: str, body: Any) -> None:
    store = _JobStore()
    client = _jobs_client(store, BOB)
    foreign = _ask(client, method, path, body, JOB_ID)
    unknown = _ask(client, method, path, body, RANDOM_JOB)
    assert foreign.status_code == 404, foreign.text
    assert _shape(foreign, JOB_ID) == _shape(unknown, RANDOM_JOB)
    # No work beyond the lookup happens for the foreign id.
    assert [c for c in store.calls if not c.startswith("get_job:")] == []


def _diff_refusal(job_id: str) -> tuple[int, str]:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        provenance_api.diff_provenance(
            job_a=job_id, job_b=job_id, store=_JobStore(), caller=BOB, trace_store=_TraceStore()
        )
    return exc.value.status_code, str(exc.value.detail).replace(job_id, "<id>")


def test_provenance_diff_hides_a_foreign_job() -> None:
    """Called directly (the route order is pinned separately in
    test_c2_n7_provenance_diff_route.py; /diff now resolves over HTTP)."""
    assert _diff_refusal(JOB_ID) == _diff_refusal(RANDOM_JOB) == (404, "Job not found: <id>")


@pytest.mark.parametrize("caller", [ALICE, ADMIN], ids=["owner", "admin"])
def test_owner_and_admin_still_read_the_job(caller: CallerIdentity) -> None:
    client = _jobs_client(_JobStore(), caller)
    resp = client.get(f"/umrs/jobs/{JOB_ID}/manifests")
    assert resp.status_code == 200, resp.text
    assert resp.json()["job_id"] == JOB_ID


# ── models ─────────────────────────────────────────────────────────────────


def _model_row(model_id: str) -> tuple:
    return (
        model_id, "m1", "project", "alice", "pa", "/shared/projects/pa/models/m1",
        "llm", None, None, {}, "available", None, "compose", NOW, NOW, {}, {},
    )


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._params: tuple = ()
        self._sql = ""

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._sql = " ".join(sql.split())
        self._params = tuple(params)
        self._conn.statements.append(self._sql)

    def fetchone(self) -> tuple | None:
        if "FROM model_catalog_entries" in self._sql and MODEL_ID in self._params:
            return _model_row(MODEL_ID)
        return None

    def fetchall(self) -> list[tuple]:
        return []


class _Conn:
    closed = 0

    def __init__(self) -> None:
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    flag = "true" if admin else "false"
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": flag,
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
        "X-Verified-Project-Memberships": project,
        "X-Verified-Memberships-HMAC": sign_memberships(SECRET, user, project, admin, [project]),
    }


@pytest.fixture
def model_client(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, _Conn]:
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    conn = _Conn()
    app = FastAPI()
    app.include_router(models_router)
    app.dependency_overrides[get_model_store] = lambda: ModelStore(conn)
    return TestClient(app), conn


MODEL_ROUTES = [
    ("GET", "/umrs/models/{id}", None),
    ("PATCH", "/umrs/models/{id}", {"description": "x"}),
    ("DELETE", "/umrs/models/{id}", None),
    ("POST", "/umrs/models/{id}/detect?dry_run=true", None),
    ("POST", "/umrs/models/{id}/detect?dry_run=false", None),
    ("POST", "/umrs/models/{id}/publish", {"target_project_id": "pb"}),
    ("GET", "/umrs/models/{id}/path", None),
    ("GET", "/umrs/models/{id}/versions", None),
    ("POST", "/umrs/models/{id}/versions", {"version": "9", "storage_ref": "/x"}),
    ("POST", "/umrs/models/{id}/versions/1.0/activate", None),
    ("POST", "/umrs/models/{id}/versions/1.0/deactivate", None),
    ("PATCH", "/umrs/models/{id}/versions/1.0", {"active": False}),
]


@pytest.mark.parametrize(("method", "path", "body"), MODEL_ROUTES)
def test_foreign_model_reads_like_an_unknown_model(
    model_client: tuple[TestClient, _Conn], method: str, path: str, body: Any
) -> None:
    client, conn = model_client
    bob = _headers("bob", "pb")
    foreign = client.request(method, _fill(path, MODEL_ID), json=body, headers=bob)
    unknown = client.request(method, _fill(path, RANDOM_MODEL), json=body, headers=bob)
    assert foreign.status_code == 404, foreign.text
    assert _shape(foreign, MODEL_ID) == _shape(unknown, RANDOM_MODEL)
    assert not [s for s in conn.statements if s.startswith(("INSERT", "UPDATE", "DELETE"))]


def test_admin_still_reads_a_foreign_model(model_client: tuple[TestClient, _Conn]) -> None:
    client, _conn = model_client
    resp = client.get(f"/umrs/models/{MODEL_ID}", headers=_headers("root", "ops", admin=True))
    assert resp.status_code == 200, resp.text
    assert resp.json()["model_id"] == MODEL_ID


def test_hidden_model_error_is_both_not_found_and_a_refusal() -> None:
    """Store callers that catch either exception keep working; str() is the
    unknown-id message exactly."""
    err = HiddenModelError(MODEL_ID, "bob")
    assert isinstance(err, ModelNotFoundError)
    assert isinstance(err, PermissionDeniedError)
    assert str(err) == str(ModelNotFoundError(MODEL_ID))


def test_store_name_lookup_hides_a_foreign_model() -> None:
    """get_model_by_name names the looked-up key, as for an unknown name."""
    from umrs.model_registry.models import ModelScope

    class _NameCursor(_Cursor):
        def fetchone(self) -> tuple | None:
            return _model_row(MODEL_ID) if "FROM model_catalog_entries" in self._sql else None

    class _NameConn(_Conn):
        def cursor(self) -> _Cursor:
            return _NameCursor(self)

    bob = AccessContext(user_id="bob", project_memberships={"pb"})
    with pytest.raises(ModelNotFoundError) as hidden:
        ModelStore(_NameConn()).get_model_by_name("m1", ModelScope.PROJECT, "pa", ctx=bob)
    with pytest.raises(ModelNotFoundError) as unknown:
        ModelStore(_Conn()).get_model_by_name("m1", ModelScope.PROJECT, "pa", ctx=bob)
    assert str(hidden.value) == str(unknown.value) == "Model not found: project:m1"


def test_project_member_still_gets_403_on_a_visible_system_model() -> None:
    """A model the caller CAN see is not hidden: refusing a write there is a
    plain 403 (the existence is no secret)."""
    from umrs.model_registry.models import ModelEntry

    acl_ctx = AccessContext(user_id="bob", project_memberships={"pb"})
    store = ModelStore(_Conn())
    system = ModelEntry.from_row(
        (MODEL_ID, "m1", "system", None, None, "/shared/models/system/m1", "llm", None,
         None, {}, "available", None, "compose", NOW, NOW, {}, {})
    )
    with pytest.raises(PermissionDeniedError) as exc:
        store.acl.enforce_write(system, acl_ctx)
    assert not isinstance(exc.value, ModelNotFoundError)
