"""H-1 item 6 (hardening plan, 2026-10-17): routes that honoured a requested
project without a check.

- intake: ``body.project_id`` won over the caller even in strict mode, and the
  manifest-id routes took no identity (presigned PUTs into any project).
- skills: a project-scope skill was registered into any ``body.project_id``.
- companion workers: the durable job went to any ``body.project_id``.
- pipelines: the project came from the raw, unsigned ``X-Project-ID`` header
  (the CRUD is retired since 2026-10-18; it still resolves the caller first).
- chat completions: the raw body ``project_id`` became the Assistant
  Service's RAG scope.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-item6-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _client(router) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = MagicMock()
    return TestClient(app)


# ── intake ───────────────────────────────────────────────────────────────────


@pytest.fixture
def intake(monkeypatch):
    from umrs.api import intake_uploads
    from umrs.data_catalog.intake_uploads import InMemoryIntakeUploadStore

    monkeypatch.setattr(intake_uploads, "_store", InMemoryIntakeUploadStore())
    return _client(intake_uploads.router)


def _manifest(client, headers, project=None):
    body = {"target_key": "data/a.bin", "expected_parts": 2}
    if project is not None:
        body["project_id"] = project
    return client.post("/umrs/uploads/manifests", json=body, headers=headers)


def test_intake_manifest_for_another_project_is_refused(intake):
    assert _manifest(intake, _headers("alice", "proj-a"), project="proj-b").status_code == 403


def test_intake_manifest_binds_the_callers_project(intake):
    resp = _manifest(intake, _headers("alice", "proj-a"))
    assert resp.status_code == 200
    assert resp.json()["manifest"]["project_id"] == "proj-a"


def test_another_projects_manifest_reads_as_not_found(intake):
    created = _manifest(intake, _headers("bob", "proj-b")).json()["manifest"]
    mid = created["manifest_id"]
    alice = _headers("alice", "proj-a")
    assert intake.get(f"/umrs/uploads/manifests/{mid}", headers=alice).status_code == 404
    assert intake.post(f"/umrs/uploads/manifests/{mid}/parts/1/presign", headers=alice).status_code == 404
    record = intake.post(
        f"/umrs/uploads/manifests/{mid}/parts/1", json={"etag": "e", "size_bytes": 1}, headers=alice
    )
    assert record.status_code == 404
    assert intake.post(f"/umrs/uploads/manifests/{mid}/complete", headers=alice).status_code == 404
    assert intake.get(f"/umrs/uploads/manifests/{mid}", headers=_headers("bob", "proj-b")).status_code == 200


def test_manifest_routes_need_identity(intake):
    created = _manifest(intake, _headers("bob", "proj-b")).json()["manifest"]
    assert intake.get(f"/umrs/uploads/manifests/{created['manifest_id']}").status_code == 401


# ── skills ───────────────────────────────────────────────────────────────────


def test_project_skill_for_another_project_is_refused():
    from umrs.api import skills

    client = _client(skills.router)
    resp = client.post(
        "/umrs/skills",
        json={"manifest": {}, "skill_md": "", "scope": "project", "project_id": "proj-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403


# ── companion workers ────────────────────────────────────────────────────────


def test_companion_worker_for_another_project_is_refused():
    from umrs.api import companion

    client = _client(companion.router)
    resp = client.post(
        "/umrs/companion/workers",
        json={"owner_user_id": "alice", "project_id": "proj-b", "job_id": "j-1"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403


# ── pipelines ────────────────────────────────────────────────────────────────


@pytest.fixture
def pipelines():
    """The pipelines CRUD is retired (H-10.1 follow-up, 2026-10-18; 410), but
    it still resolves the verified caller first."""
    from umrs.api import pipelines as mod

    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


def test_pipelines_answer_gone_to_a_verified_caller(pipelines):
    headers = {**_headers("alice", "proj-a"), "X-Project-ID": "proj-b"}
    assert pipelines.get("/umrs/pipelines", headers=headers).status_code == 410


def test_pipelines_need_identity_in_strict_mode(pipelines):
    assert pipelines.get("/umrs/pipelines", headers={"X-Project-ID": "proj-b"}).status_code == 401


# ── chat completions ─────────────────────────────────────────────────────────


def test_chat_rag_scope_of_another_project_is_refused():
    from umrs.api import chat

    client = _client(chat.router)
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        json={"model": "m", "messages": [], "project_id": "proj-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
