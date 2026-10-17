"""H-1 remainder, projects lane (hardening plan H-1, 2026-10-18).

Existence disclosures and an unchecked project segment:

1. ``POST /umrs/projects`` for an id already taken returned that project's
   row (owner, data class, ask policy) to any caller, while
   ``GET /umrs/projects/{id}`` hides it as 404. A caller who may not read the
   project now gets ``projects.conflict`` "project id is taken" without it.
2. ``GET /umrs/circles/{project_id}``: an unknown project answered 404 but
   another existing project 200 with an empty list. A caller who cannot read
   the project now gets 404 for both.
3. ``PUT /umrs/projects/{id}/budget``: a non-owner got 403 while GET hides
   the project as 404. A caller who cannot read it now gets 404; a reader who
   is not owner/admin keeps 403.
4. ``PUT /umrs/buckets/{bucket}/access`` audited under the bucket's project
   segment, a free experiment label any caller picks at create time (ADR-079
   D-2). It now audits under the actor's verified project; the label rides in
   the details. Create keeps the free label.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-projects-secret"
CODE = "x-platform-error-code"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


# Bogus HMAC, admin flag true, a foreign project: must never pass.
FORGED = {
    "X-User-ID": "alice",
    "X-Verified-Project-ID": "proj-b",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": "0" * 64,
}


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


class _Pool:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def getconn(self):
        return self.conn

    def putconn(self, _conn) -> None:
        pass


def _app(router, pool: Any = None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if pool is not None:
        app.state.umrs_db_pool = pool
    return TestClient(app)


# ── projects: create + budget ────────────────────────────────────────────────


@pytest.fixture
def projects():
    from test_api_projects import _BudgetConn
    from umrs.api import projects as mod

    conn = _BudgetConn()
    store = mod.ProjectStore(conn)
    store.create("proj-b", name="Secret B", owner_id="bob")
    store.set_budget("proj-b", max_gpu_hours=8.0, max_wall_clock_hours=3.0)
    return _app(mod.router, _Pool(conn)), conn


def _create(client, pid: str, headers: dict[str, str]):
    return client.post("/umrs/projects", json={"project_id": pid}, headers=headers)


def test_taken_id_is_a_conflict_without_the_row(projects):
    client, conn = projects
    resp = _create(client, "proj-b", _headers("alice", "proj-a"))
    assert resp.status_code == 409
    assert resp.headers[CODE] == "projects.conflict"
    assert "project id is taken" in resp.text
    assert "bob" not in resp.text and "Secret B" not in resp.text
    assert conn.rows["proj-b"][2] == "bob"


def test_forged_tuple_creates_nothing_and_reads_nothing(projects):
    client, _ = projects
    resp = _create(client, "proj-b", FORGED)
    assert resp.status_code == 401
    assert "bob" not in resp.text


@pytest.mark.parametrize(
    "headers",
    [
        _headers("bob", "proj-x"),  # owner re-posts (idempotent ensure-shape)
        _headers("carol", "proj-b"),  # verified into the project
        _headers("root", "proj-a", admin=True),  # admin
    ],
)
def test_readers_still_get_the_existing_row(projects, headers):
    client, _ = projects
    resp = _create(client, "proj-b", headers)
    assert resp.status_code == 201
    assert resp.json()["project"]["owner_id"] == "bob"


def test_a_free_id_is_created_for_the_caller(projects):
    client, conn = projects
    resp = _create(client, "proj-new", _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert resp.json()["project"]["owner_id"] == "alice"
    assert conn.rows["proj-new"][2] == "alice"


def _put_budget(client, pid: str, headers: dict[str, str]):
    return client.put(
        f"/umrs/projects/{pid}/budget",
        json={"max_gpu_hours": 99.0, "max_wall_clock_hours": 99.0},
        headers=headers,
    )


@pytest.mark.parametrize("pid", ["proj-b", "proj-ghost"])
def test_budget_put_by_a_non_reader_reads_as_not_found(projects, pid):
    client, conn = projects
    resp = _put_budget(client, pid, _headers("alice", "proj-a"))
    assert resp.status_code == 404
    assert resp.headers[CODE] == "projects.not_found"
    assert conn.budgets["proj-b"][1] == 8.0


def test_budget_put_by_a_reader_who_is_not_owner_is_forbidden(projects):
    client, conn = projects
    resp = _put_budget(client, "proj-b", _headers("carol", "proj-b"))
    assert resp.status_code == 403
    assert conn.budgets["proj-b"][1] == 8.0


def test_budget_put_forged_tuple_is_rejected(projects):
    client, conn = projects
    assert _put_budget(client, "proj-b", FORGED).status_code == 401
    assert conn.budgets["proj-b"][1] == 8.0


@pytest.mark.parametrize(
    "headers", [_headers("bob", "proj-x"), _headers("root", "proj-a", admin=True)]
)
def test_budget_put_by_owner_or_admin(projects, headers):
    client, _ = projects
    resp = _put_budget(client, "proj-b", headers)
    assert resp.status_code == 200
    assert resp.json()["budget"]["max_gpu_hours"] == 99.0


# ── circles: project listing ─────────────────────────────────────────────────


def _circle(conn, pid: str, cid: str, owner: str, status: str = "active") -> None:
    conn.circles[(pid, cid)] = (pid, cid, cid, owner, status, None, None, None)


@pytest.fixture
def circles():
    from test_api_circles import _Pool as CirclesPool
    from umrs.api import circles as mod

    pool = CirclesPool()
    conn = pool._conn
    conn.projects["proj-b"] = ("proj-b", "bob", "active")
    _circle(conn, "proj-b", "team", "erin")
    conn.members_add("proj-b", "team", "erin", "admin", "erin")
    _circle(conn, "proj-b", "gina", "gina", status="retired")
    conn.members_add("proj-b", "gina", "gina", "admin", "gina")
    return _app(mod.router, pool)


@pytest.mark.parametrize("pid", ["proj-b", "proj-ghost"])
def test_unreadable_project_listing_is_not_found_either_way(circles, pid):
    resp = circles.get(f"/umrs/circles/{pid}", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 404
    assert resp.headers[CODE] == "circles.not_found"
    assert "team" not in resp.text


def test_unreadable_listing_with_retired_rows_is_not_found(circles):
    resp = circles.get(
        "/umrs/circles/proj-b",
        params={"include_retired": "true"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 404


def test_listing_forged_tuple_is_rejected(circles):
    resp = circles.get("/umrs/circles/proj-b", headers=FORGED)
    assert resp.status_code == 401
    assert "team" not in resp.text


def test_caller_verified_into_the_project_sees_own_rows(circles):
    resp = circles.get("/umrs/circles/proj-b", headers=_headers("carol", "proj-b"))
    assert resp.status_code == 200
    assert resp.json()["circles"] == []


def test_owner_of_a_retired_circle_still_lists_it(circles):
    headers = _headers("gina", "proj-a")
    retired = circles.get(
        "/umrs/circles/proj-b", params={"include_retired": "true"}, headers=headers
    )
    assert retired.status_code == 200
    assert [c["circle_id"] for c in retired.json()["circles"]] == ["gina"]
    assert circles.get("/umrs/circles/proj-b", headers=headers).status_code == 404


@pytest.mark.parametrize(
    "headers",
    [
        _headers("erin", "proj-a"),  # affiliated
        _headers("bob", "proj-a"),  # project owner
        _headers("root", "proj-a", admin=True),  # admin
    ],
)
def test_privileged_callers_see_the_roster(circles, headers):
    resp = circles.get("/umrs/circles/proj-b", headers=headers)
    assert resp.status_code == 200
    assert [c["circle_id"] for c in resp.json()["circles"]] == ["team"]


# ── buckets: the project segment ─────────────────────────────────────────────


@pytest.fixture
def buckets(monkeypatch):
    from test_personal_buckets import _FakeS3, _FakeS3Config, _FakeStore
    from umrs.api import buckets as mod
    from umrs.server import step_staging
    from umrs.shared.storage import user_buckets as ub

    store, s3 = _FakeStore(), _FakeS3()
    ub.set_user_bucket_store(store)  # type: ignore[arg-type]
    monkeypatch.setattr(step_staging, "_bucket_registry", None)
    monkeypatch.setattr(mod, "get_s3_config", lambda: _FakeS3Config(s3))
    yield _app(mod.router), store, s3
    ub.set_user_bucket_store(None)


def _bucket(client, project: str, headers: dict[str, str]):
    return client.post(
        "/umrs/buckets", json={"project": project, "alias": "videos"}, headers=headers
    )


def test_bucket_project_segment_is_a_free_label(buckets):
    """ADR-079 D-2: the segment is the researcher's project / experiment name
    (e.g. ``opossum``), not a platform project, so it is not checked."""
    client, store, _ = buckets
    resp = _bucket(client, "opossum", _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert store.rows["alice.opossum.videos"].owner_id == "alice"


def test_bucket_forged_tuple_is_rejected(buckets):
    client, store, s3 = buckets
    assert _bucket(client, "proj-b", FORGED).status_code == 401
    assert store.rows == {} and s3.buckets == set()


def test_access_change_audits_under_the_callers_verified_project(buckets, monkeypatch):
    """Fails on the old code: the audit row went to the bucket's free label
    (here another project's id), so any caller could file audit rows under
    any project."""
    from umrs.api import audit_events

    client, _, _ = buckets
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(audit_events, "emit_audit", lambda *_a, **kw: seen.append(kw))
    alice = _headers("alice", "proj-a")
    assert _bucket(client, "proj-b", alice).status_code == 201
    resp = client.put(
        "/umrs/buckets/alice.proj-b.videos/access", json={"access": []}, headers=alice
    )
    assert resp.status_code == 200
    assert [kw["project_id"] for kw in seen] == ["proj-a"]
    assert seen[0]["details"]["label"] == "proj-b"
