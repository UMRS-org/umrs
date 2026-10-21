"""Cycle-2 umrs-api lane (2026-10-21): model version writes need write access.

Found live (gateway -> auth sidecar -> umrs, two projects): bob in project
``pb`` cannot read alice's project-``pa`` model (403), yet
``POST /umrs/models/{id}/versions`` returned 201 and made bob's
``storage_ref`` the model's latest version; ``POST /{id}/detect`` returned
the model's name and storage path to him (and, without ``dry_run``, would
rewrite its capabilities). ``ModelStore.create_version`` looked the model up
without the caller's context, the activate/deactivate paths checked read
access only, ``detect`` never passed the caller at all, and a version's
``storage_ref`` skipped the scope path check that model creation applies.

The store here is the real ``ModelStore`` (real ACL, real path validator)
over a scripted connection, so the SQL each call would run is visible.

2026-10-21 (umrs-follow, N2): bob cannot even read the model, so every
refusal now reads as not found (404, like an unknown id), never a 403.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.models import get_model_store, router
from umrs.model_registry.store import ModelStore
from umrs.model_registry.store.acl import AccessContext
from umrs.model_registry.store.errors import PathNotAllowedError, PermissionDeniedError

from r4_contracts.handshake_hmac import sign_claims, sign_memberships

SECRET = "c2-model-version-secret"
MODEL_ID = "mdl-alice0000001"
NOW = datetime(2026, 10, 21, 12, 0, 0)


def _model_row(scope: str = "project", project_id: str | None = "pa") -> tuple:
    owner = None if scope == "system" else "alice"
    ref = f"/shared/projects/{project_id}/models/m1" if project_id else "/shared/models/system/m1"
    return (
        MODEL_ID, "m1", scope, owner, project_id, ref, "llm", None, None,
        {}, "available", None, "compose", NOW, NOW, {}, {},
    )


def _release_row(version: str = "1.0") -> tuple:
    return (
        "ver-0001", MODEL_ID, version, "/shared/projects/pa/models/good", None,
        True, False, "manual", "released", NOW, NOW, None,
    )


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._last = ""

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        del params  # the script answers by statement, not by bound values
        self._last = " ".join(sql.split())
        self._conn.statements.append(self._last)

    def fetchone(self) -> tuple | None:
        if "FROM model_catalog_entries" in self._last:
            return self._conn.model_row
        return None

    def fetchall(self) -> list[tuple]:
        if "FROM model_releases" in self._last:
            return [_release_row()]
        return []


class _Conn:
    closed = 0

    def __init__(self, model_row: tuple) -> None:
        self.model_row = model_row
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def writes(self) -> list[str]:
        return [s for s in self.statements if s.startswith(("INSERT", "UPDATE"))]


def _ctx(user: str, project: str) -> AccessContext:
    return AccessContext(user_id=user, is_admin=False, project_memberships={project})


ALICE = _ctx("alice", "pa")
BOB = _ctx("bob", "pb")


@pytest.fixture
def conn() -> _Conn:
    return _Conn(_model_row())


@pytest.fixture
def store(conn: _Conn) -> ModelStore:
    return ModelStore(conn)


# ── store layer ────────────────────────────────────────────────────────────


def test_foreign_caller_cannot_create_a_version(store: ModelStore, conn: _Conn) -> None:
    with pytest.raises(PermissionDeniedError):
        store.create_version(MODEL_ID, "9.9", "/shared/projects/pb/models/evil", ctx=BOB)
    assert conn.writes() == []


@pytest.mark.parametrize("method", ["activate_version", "deactivate_version"])
def test_foreign_caller_cannot_flip_a_version(store: ModelStore, conn: _Conn, method: str) -> None:
    with pytest.raises(PermissionDeniedError):
        getattr(store, method)(MODEL_ID, "1.0", BOB)
    assert conn.writes() == []


@pytest.mark.parametrize("method", ["activate_version", "deactivate_version"])
def test_system_model_reader_cannot_flip_its_versions(method: str) -> None:
    """System models are readable by everyone and writable by admins only."""
    conn = _Conn(_model_row(scope="system", project_id=None))
    with pytest.raises(PermissionDeniedError):
        getattr(ModelStore(conn), method)(MODEL_ID, "1.0", BOB)
    assert conn.writes() == []


def test_version_storage_ref_must_stay_in_the_model_scope(store: ModelStore, conn: _Conn) -> None:
    with pytest.raises(PathNotAllowedError):
        store.create_version(MODEL_ID, "2.0", "/shared/users/bob/models/secret", ctx=ALICE)
    assert conn.writes() == []


def test_owner_still_creates_and_flips_versions(store: ModelStore, conn: _Conn) -> None:
    ver = store.create_version(MODEL_ID, "2.0", "/shared/projects/pa/models/v2", ctx=ALICE)
    assert ver.version == "2.0"
    store.deactivate_version(MODEL_ID, "1.0", ALICE)
    store.activate_version(MODEL_ID, "1.0", ALICE)
    assert any(s.startswith("INSERT INTO model_releases") for s in conn.writes())


# ── HTTP layer (strict mode, signed identity) ──────────────────────────────


def _headers(user: str, project: str) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
        "X-Verified-Project-Memberships": project,
        "X-Verified-Memberships-HMAC": sign_memberships(SECRET, user, project, False, [project]),
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, store: ModelStore) -> TestClient:
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_model_store] = lambda: store
    return TestClient(app)


def test_http_foreign_version_post_is_refused(client: TestClient, conn: _Conn) -> None:
    r = client.post(
        f"/umrs/models/{MODEL_ID}/versions",
        json={"version": "9.9", "storage_ref": "/shared/projects/pb/models/evil"},
        headers=_headers("bob", "pb"),
    )
    assert r.status_code == 404
    assert conn.writes() == []


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/versions", None),
        ("POST", "/versions/1.0/activate", None),
        ("POST", "/versions/1.0/deactivate", None),
        ("PATCH", "/versions/1.0", {"active": False}),
    ],
)
def test_http_foreign_version_routes_are_404_not_500(
    client: TestClient, conn: _Conn, method: str, path: str, body: dict | None
) -> None:
    r = client.request(
        method, f"/umrs/models/{MODEL_ID}{path}", json=body, headers=_headers("bob", "pb")
    )
    assert r.status_code == 404
    assert conn.writes() == []


def test_http_version_outside_scope_is_400(client: TestClient, conn: _Conn) -> None:
    r = client.post(
        f"/umrs/models/{MODEL_ID}/versions",
        json={"version": "2.0", "storage_ref": "/etc"},
        headers=_headers("alice", "pa"),
    )
    assert r.status_code == 400
    assert conn.writes() == []


@pytest.mark.parametrize("dry_run", ["true", "false"])
def test_http_foreign_detect_is_refused(client: TestClient, conn: _Conn, dry_run: str) -> None:
    r = client.post(
        f"/umrs/models/{MODEL_ID}/detect",
        params={"dry_run": dry_run},
        headers=_headers("bob", "pb"),
    )
    assert r.status_code == 404
    assert "/shared/projects/pa" not in r.text
    assert conn.writes() == []


def test_http_owner_creates_a_version(client: TestClient) -> None:
    r = client.post(
        f"/umrs/models/{MODEL_ID}/versions",
        json={"version": "2.0", "storage_ref": "/shared/projects/pa/models/v2"},
        headers=_headers("alice", "pa"),
    )
    assert r.status_code == 201, r.text
