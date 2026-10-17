"""H-1 (hardening plan, 2026-10-17): operator restore, the files ``bucket``
parameter and the endpoint health details.

- ``POST /umrs/knowledge/restore`` names a directory on the UMRS host, may
  rewrite its manifest and retargets rows into any project; the operator
  check let an anonymous caller through in mvp mode. It is admin-only in
  every mode now.
- ``/umrs/files/presign``, ``/{file_id}/metadata``, ``DELETE /{file_id}`` and
  ``/exists`` honoured a caller-named ``bucket`` without the ownership check
  the other bucket routes (download, list, upload) apply.
- ``GET /umrs/endpoints/health?include_details=true`` listed every project's
  endpoints (host, port, model, errors) to any caller.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from botocore.exceptions import ClientError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-admin-identity-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── bundle restore: admin-only in every mode ─────────────────────────────────


@pytest.fixture
def restore(monkeypatch, tmp_path):
    import umrs.api._asyncpg_pool as pool_mod
    import umrs.knowledge.bundle_import as bundle_import
    from umrs.api.bundle_restore import router

    engine = MagicMock()
    engine.import_project = AsyncMock(return_value=SimpleNamespace(to_dict=lambda: {"nodes": 1}))
    monkeypatch.setattr(pool_mod, "knowledge_asyncpg_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(bundle_import, "KnowledgeImportEngine", lambda **_kwargs: engine)
    (tmp_path / "manifest.json").write_text(json.dumps({"project_id": "proj-a"}))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False), engine, str(tmp_path)


def test_anonymous_restore_is_refused_in_mvp_mode(monkeypatch, restore):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    client, engine, bundle_dir = restore
    resp = client.post(
        "/umrs/knowledge/restore",
        json={"bundle_dir": bundle_dir, "project_id": "someone-elses-project"},
    )
    assert resp.status_code == 403
    engine.import_project.assert_not_called()


def test_mvp_admin_still_restores(monkeypatch, restore):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    client, engine, bundle_dir = restore
    resp = client.post(
        "/umrs/knowledge/restore",
        json={"bundle_dir": bundle_dir},
        headers={"X-User-ID": "ops", "X-Verified-Project-ID": "ops", "X-Verified-Is-Admin": "true"},
    )
    assert resp.status_code == 200, resp.text
    engine.import_project.assert_awaited_once()


def test_strict_forged_admin_restore_is_refused(strict, restore):
    client, engine, bundle_dir = restore
    headers = _headers("alice", "proj-a")
    headers["X-Verified-Is-Admin"] = "true"
    resp = client.post("/umrs/knowledge/restore", json={"bundle_dir": bundle_dir}, headers=headers)
    assert resp.status_code in (401, 403)
    engine.import_project.assert_not_called()


def test_strict_signed_admin_restores(strict, restore):
    client, engine, bundle_dir = restore
    resp = client.post(
        "/umrs/knowledge/restore",
        json={"bundle_dir": bundle_dir},
        headers=_headers("root", "ops", admin=True),
    )
    assert resp.status_code == 200, resp.text
    engine.import_project.assert_awaited_once()


# ── files: a caller-named bucket must be one the caller owns ─────────────────


class _S3:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Bucket, Key))
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def delete_object(self, Bucket, Key):
        self.calls.append(("delete", Bucket, Key))


class _Config:
    default_ttl = 60
    max_ttl = 600
    presign_endpoint = None

    def __init__(self, bucket: str, s3: _S3) -> None:
        self.bucket = bucket
        self._s3 = s3

    def get_client(self):
        return self._s3


@pytest.fixture
def files(strict, monkeypatch):
    from umrs.api import files as files_mod
    from umrs.server import step_staging

    s3 = _S3()
    project_config = _Config("proj-store", s3)
    monkeypatch.setattr(files_mod, "get_s3_config", lambda: _Config("r4-data", s3))
    monkeypatch.setattr(files_mod, "generate_presigned_put", lambda *_a, **_k: "http://put")
    monkeypatch.setattr(files_mod, "generate_presigned_get", lambda *_a, **_k: "http://get")
    monkeypatch.setattr(
        step_staging, "bucket_owned_by", lambda bucket, uid: (bucket, uid) == ("bob-bucket", "bob")
    )
    monkeypatch.delenv("S3_DOWNLOAD_EXTRA_BUCKETS", raising=False)
    app = FastAPI()
    app.include_router(files_mod.router)
    app.dependency_overrides[files_mod.get_project_s3_config] = lambda: project_config
    return TestClient(app), s3


def _presign(client, bucket, user="alice", project="proj-a"):
    body = {"operation": "upload", "key": "data/x.bin"}
    if bucket is not None:
        body["bucket"] = bucket
    return client.post("/umrs/files/presign", json=body, headers=_headers(user, project))


def test_presign_into_a_foreign_bucket_is_refused(files):
    client, _s3 = files
    resp = _presign(client, "bob-bucket")
    assert resp.status_code == 403


def test_presign_batch_into_a_foreign_bucket_is_refused(files):
    client, _s3 = files
    resp = client.post(
        "/umrs/files/presign/batch",
        json={"requests": [{"operation": "upload", "key": "x", "bucket": "bob-bucket"}]},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 200
    assert resp.json()["results"] == []
    assert len(resp.json()["errors"]) == 1


def test_metadata_in_a_foreign_bucket_is_refused(files):
    client, s3 = files
    resp = client.get(
        "/umrs/files/x.bin/metadata",
        params={"bucket": "bob-bucket"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
    assert not s3.calls


def test_delete_in_a_foreign_bucket_is_refused(files):
    client, s3 = files
    resp = client.delete(
        "/umrs/files/x.bin", params={"bucket": "bob-bucket"}, headers=_headers("alice", "proj-a")
    )
    assert resp.status_code == 403
    assert not s3.calls


def test_exists_in_a_foreign_bucket_is_refused(files):
    client, s3 = files
    resp = client.post(
        "/umrs/files/exists",
        params={"bucket": "bob-bucket"},
        json=["x.bin"],
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
    assert not s3.calls


def test_owner_presigns_into_own_bucket(files):
    client, _s3 = files
    resp = _presign(client, "bob-bucket", user="bob")
    assert resp.status_code == 200, resp.text
    assert resp.json()["bucket"] == "bob-bucket"


@pytest.mark.parametrize("bucket", [None, "proj-store", "proj-store-output", "r4-data"])
def test_namespace_buckets_need_no_ownership(files, bucket):
    """The project's own store bucket, its -output pair and the server default
    the gateway's download ACL names keep working (key stays project-prefixed)."""
    client, _s3 = files
    resp = _presign(client, bucket)
    assert resp.status_code == 200, resp.text
    assert resp.json()["key"] == "proj-a/data/x.bin"


def test_delete_in_the_project_bucket_still_works(files):
    client, s3 = files
    resp = client.delete("/umrs/files/x.bin", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 204
    assert s3.calls == [("delete", "proj-store", "proj-a/x.bin")]


# ── endpoints health: details are caller-scoped ──────────────────────────────


@pytest.fixture
def health(strict):
    from umrs.api.endpoints import router
    from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

    registry = EndpointRegistry()
    for project, job in (("proj-a", "job-a"), ("proj-b", "job-b")):
        registry.register(
            project_id=project, model="m", job_id=job, host="node", port=8000, backend="slurm"
        )
        registry.mark_ready(job)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _detail_projects(resp) -> set[str]:
    return {ep["project_id"] for ep in resp.json()["endpoints"]}


def test_health_details_hide_other_projects(health):
    resp = health.get(
        "/umrs/endpoints/health",
        params={"include_details": "true"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 200, resp.text
    assert _detail_projects(resp) == {"proj-a"}


def test_health_details_need_identity_in_strict_mode(health):
    resp = health.get("/umrs/endpoints/health", params={"include_details": "true"})
    assert resp.status_code == 401


def test_health_details_reject_a_forged_admin_flag(health):
    headers = _headers("alice", "proj-a")
    headers["X-Verified-Is-Admin"] = "true"
    resp = health.get("/umrs/endpoints/health", params={"include_details": "true"}, headers=headers)
    assert resp.status_code == 401


def test_admin_sees_every_projects_health_details(health):
    resp = health.get(
        "/umrs/endpoints/health",
        params={"include_details": "true"},
        headers=_headers("root", "ops", admin=True),
    )
    assert resp.status_code == 200, resp.text
    assert _detail_projects(resp) == {"proj-a", "proj-b"}


def test_health_summary_stays_open_without_identity(health):
    """Monitors (mcp-servers platform health, ops probes) call it bare."""
    resp = health.get("/umrs/endpoints/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_endpoints"] == 2
    assert resp.json()["endpoints"] == []


# Reviewer follow-up (2026-10-18): mvp mode. "Anonymous" there means anyone
# who can reach the port, so the cross-project detail dump is admin-only in
# every mode, like the sibling ``GET /umrs/endpoints`` listing.


@pytest.fixture
def mvp_health(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
    from umrs.api.endpoints import router
    from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

    registry = EndpointRegistry()
    for project, job in (("proj-a", "job-a"), ("proj-b", "job-b")):
        registry.register(
            project_id=project, model="m", job_id=job, host="node", port=8000, backend="slurm"
        )
        registry.mark_ready(job)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def test_mvp_anonymous_health_details_list_no_project(mvp_health):
    resp = mvp_health.get("/umrs/endpoints/health", params={"include_details": "true"})
    assert resp.status_code == 403, resp.text
    assert "proj-a" not in resp.text and "proj-b" not in resp.text


def test_mvp_legacy_project_sees_its_own_health_details(mvp_health):
    resp = mvp_health.get(
        "/umrs/endpoints/health",
        params={"include_details": "true"},
        headers={"X-Project-ID": "proj-a"},
    )
    assert resp.status_code == 200, resp.text
    assert _detail_projects(resp) == {"proj-a"}


def test_mvp_admin_sees_every_projects_health_details(mvp_health):
    headers = {"X-User-ID": "root", "X-Verified-Project-ID": "ops", "X-Verified-Is-Admin": "true"}
    resp = mvp_health.get(
        "/umrs/endpoints/health", params={"include_details": "true"}, headers=headers
    )
    assert resp.status_code == 200, resp.text
    assert _detail_projects(resp) == {"proj-a", "proj-b"}


def test_mvp_anonymous_health_summary_stays_open(mvp_health):
    resp = mvp_health.get("/umrs/endpoints/health")
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_endpoints"] == 2
