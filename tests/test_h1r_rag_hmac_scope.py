"""H-1 (hardening plan, 2026-10-18), rag lane: every /umrs/rag route reads
the caller only through the gateway HMAC.

``rag._get_caller_identity`` (deleted 2026-10-18; ``_verified_caller`` is the
one reader) resolved the tuple without ``verify_hmac`` and
``_require_rag_scope``, ``_query_via_spine``, ``put_corpus_access`` and
``rag_retrieve.is_admin`` read ``X-Verified-Is-Admin`` raw. In strict mode a
forged tuple (mallory, victim project, admin true, bogus HMAC) passed the
scope check, so a direct sidecar caller could upload into the victim project
and retrieve its passages, ambient rows included. A forged tuple is now a 401
on every route; admin comes from the signed flag only.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-rag-secret"

FORGED = {
    "X-User-ID": "mallory",
    "X-Verified-Project-ID": "victim",
    "X-Verified-Is-Admin": "true",
    "X-Verified-HMAC": "bogus",
}


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _admin_flag_flipped() -> dict[str, str]:
    """A member's valid signature with the admin flag flipped to true."""
    headers = _headers("mallory", "mallory-lab")
    headers["X-Verified-Is-Admin"] = "true"
    return headers


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


class _CorpusStore:
    def __init__(self) -> None:
        self.rows: dict[str, list[str]] = {}

    def get_access(self, project_id: str) -> list[str]:
        return self.rows.get(project_id, [])

    def set_access(self, project_id: str, access: list[str], set_by: str) -> None:
        self.rows[project_id] = list(access)
        self.set_by = set_by


@pytest.fixture
def seen(monkeypatch):
    """Fake every store the rag routes reach; record what they were handed."""
    record: dict = {}
    store = _CorpusStore()
    monkeypatch.setattr(rag_api, "get_corpus_access_store", lambda: store)
    monkeypatch.setattr(rag_api, "shared_corpus_readable", lambda *_a: False)
    monkeypatch.setattr("umrs.api.audit_events.emit_audit", lambda *_a, **_k: None)

    async def central(_request, file, project_id):
        record["upload"] = project_id
        return rag_api.UploadResponse(document_id="j1", filename=file.filename, status="ok")

    def run(_request, _db, spec, _graphs, _exclude):
        record["retrieve"] = (spec.project_id, spec.include_ambient)
        return {"passages": []}

    def fetch(_request, _db, project_id, node_id, _window, _max):
        record["passage"] = (project_id, node_id, rag_retrieve.is_admin(_request))
        return {"source_id": node_id}

    monkeypatch.setattr(rag_api, "_upload_via_central", central)
    monkeypatch.setattr(rag_retrieve, "run_retrieve", run)
    monkeypatch.setattr(rag_retrieve, "fetch_passage", fetch)
    record["store"] = store
    return record


@pytest.fixture
def client(seen) -> TestClient:
    app = FastAPI()
    app.include_router(rag_api.router)
    app.dependency_overrides[get_registry] = EndpointRegistry
    app.state.umrs_db_pool = None
    return TestClient(app)


def _retrieve(client: TestClient, headers: dict, project: str | None = None):
    client.app.state.umrs_db_pool = object()
    params = {"q": "q"} | ({"project_id": project} if project else {})
    return client.get("/umrs/rag/retrieve", params=params, headers=headers)


# ── the helpers ─────────────────────────────────────────────────────────────


def _request(headers: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(headers={k.lower(): v for k, v in headers.items()})


@pytest.mark.parametrize("headers", [FORGED, _admin_flag_flipped()])
def test_scope_refuses_a_forged_tuple(headers):
    with pytest.raises(HTTPException) as exc:
        rag_api._require_rag_scope(_request(headers), "victim", write=True)
    assert exc.value.status_code == 401


@pytest.mark.parametrize("headers", [FORGED, _admin_flag_flipped()])
def test_verified_caller_verifies_the_hmac(headers):
    """The one identity reader of the rag routes (the dead
    ``_get_caller_identity`` was deleted, 2026-10-18)."""
    with pytest.raises(HTTPException) as exc:
        rag_api._verified_caller(_request(headers))
    assert exc.value.status_code == 401


def test_verified_caller_takes_the_signed_tuple():
    caller = rag_api._verified_caller(_request(_headers("root", "ops", admin=True)))
    assert (caller.user_id, caller.project_id, caller.is_admin) == ("root", "ops", True)


def test_get_caller_identity_is_gone():
    assert not hasattr(rag_api, "_get_caller_identity")


@pytest.mark.parametrize("headers", [FORGED, _admin_flag_flipped()])
def test_is_admin_never_honours_an_unsigned_flag(headers):
    try:
        admin = rag_retrieve.is_admin(_request(headers))
    except HTTPException as exc:
        assert exc.status_code == 401
    else:
        assert admin is False


def test_is_admin_honours_the_signed_flag():
    assert rag_retrieve.is_admin(_request(_headers("root", "ops", admin=True))) is True
    assert rag_retrieve.is_admin(_request(_headers("u1", "lab"))) is False


@pytest.mark.parametrize("headers", [{}, {"X-Verified-Is-Admin": "true"}])
def test_is_admin_is_false_without_a_tuple_in_strict_mode(headers):
    """No tuple at all is not an admin, not a 401: rejecting an anonymous
    caller stays ``_require_rag_scope``'s job (review, 2026-10-18)."""
    assert rag_retrieve.is_admin(_request(headers)) is False


# ── every route: a forged tuple is a 401 ────────────────────────────────────

_ROUTES = [
    ("post", "/umrs/rag/upload", {"files": {"file": ("a.txt", b"x", "text/plain")}}),
    ("get", "/umrs/rag/retrieve", {"params": {"q": "q", "project_id": "victim"}}),
    ("get", "/umrs/rag/passages/n1", {"params": {"project_id": "victim"}}),
    ("get", "/umrs/rag/corpus-access", {"params": {"project_id": "victim"}}),
    ("put", "/umrs/rag/corpus-access", {"json": {"project_id": "victim", "access": ["system"]}}),
    ("post", "/umrs/rag/query", {"json": {"query": "q", "project_id": "victim"}}),
    ("get", "/umrs/rag/documents", {"params": {"project_id": "victim"}}),
    ("get", "/umrs/rag/documents/d1", {}),
    ("delete", "/umrs/rag/documents/d1", {}),
    ("get", "/umrs/rag/stats", {}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), _ROUTES)
@pytest.mark.parametrize("headers", [FORGED, _admin_flag_flipped()], ids=["forged", "flipped"])
def test_every_rag_route_refuses_a_forged_tuple(client, seen, method, path, kwargs, headers):
    client.app.state.umrs_db_pool = object()
    response = client.request(method.upper(), path, headers=headers, **kwargs)
    assert response.status_code == 401, (path, response.status_code, response.text)
    assert "upload" not in seen and "retrieve" not in seen and "passage" not in seen
    assert seen["store"].rows == {}


# ── positive controls: the signed tuple still works ─────────────────────────


def test_a_signed_member_uploads_into_their_own_project(client, seen):
    response = client.post(
        "/umrs/rag/upload",
        files={"file": ("a.txt", b"x", "text/plain")},
        headers=_headers("alice", "lab"),
    )
    assert response.status_code == 200, response.text
    assert seen["upload"] == "lab"


def test_a_signed_member_retrieves_without_ambient_rows(client, seen):
    response = _retrieve(client, _headers("alice", "lab"))
    assert response.status_code == 200, response.text
    assert seen["retrieve"] == ("lab", False)


def test_a_signed_member_cannot_name_a_foreign_project(client, seen):
    response = _retrieve(client, _headers("alice", "lab"), project="victim")
    assert response.status_code == 403
    assert "retrieve" not in seen


def test_a_signed_admin_reads_any_project_with_ambient_rows(client, seen):
    response = _retrieve(client, _headers("root", "ops", admin=True), project="victim")
    assert response.status_code == 200, response.text
    assert seen["retrieve"] == ("victim", True)


def test_a_signed_admin_fetches_a_passage_as_admin(client, seen):
    client.app.state.umrs_db_pool = object()
    response = client.get(
        "/umrs/rag/passages/n1",
        params={"project_id": "victim"},
        headers=_headers("root", "ops", admin=True),
    )
    assert response.status_code == 200, response.text
    assert seen["passage"] == ("victim", "n1", True)


def test_only_a_signed_admin_shares_a_corpus_with_system(client, seen):
    body = {"project_id": "lab", "access": ["system"]}
    member = client.put("/umrs/rag/corpus-access", json=body, headers=_headers("alice", "lab"))
    assert member.status_code == 403
    admin = client.put(
        "/umrs/rag/corpus-access", json=body, headers=_headers("root", "ops", admin=True)
    )
    assert admin.status_code == 200, admin.text
    assert seen["store"].rows == {"lab": ["system"]}


def test_the_spine_query_reads_ambient_rows_only_for_a_signed_admin(client, monkeypatch):
    calls: list[bool] = []

    def answer(_request, _pool, _body, _project, admin):
        calls.append(admin)
        return SimpleNamespace(passages=[], answer="", reason="none", model="")

    monkeypatch.setattr(rag_api, "_answer_in_scope", answer)
    client.app.state.umrs_db_pool = object()
    body = {"query": "q", "project_id": "lab"}
    assert (
        client.post("/umrs/rag/query", json=body, headers=_headers("alice", "lab")).status_code
        == 200
    )
    admin = _headers("root", "ops", admin=True)
    assert client.post("/umrs/rag/query", json=body, headers=admin).status_code == 200
    assert calls == [False, True]
