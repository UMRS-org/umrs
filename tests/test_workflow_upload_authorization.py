"""UMRS workflow-upload authorization (HMAC-covered workflow_author/admin)."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from r4_contracts.handshake_hmac import sign_claims, sign_claims_v2
from umrs.api.workflows import router

SECRET = "test-secret"


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    yield


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=catalog_file,
    ):
        yield catalog_file


def _zip_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "workflow.json",
            json.dumps(
                {
                    "name": "ct-analysis",
                    "version": "1.0",
                    "steps": [{"name": "s", "entrypoint": "m:S"}],
                }
            ),
        )
        zf.writestr("steps.py", "class S: pass\n")
    return buffer.getvalue()


def _headers(*, user="alice", project="proj-a", admin=False, author=False, version=2):
    headers = {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-Workflow-Author": "true" if author else "false",
    }
    if version == 1:
        headers["X-Verified-HMAC"] = sign_claims(SECRET, user, project, admin)
    else:
        headers["X-Verified-HMAC"] = sign_claims_v2(
            SECRET, user, project, admin, author
        )
    return headers


def _upload(client, headers, *, scope="user"):
    return client.post(
        "/umrs/workflows/upload",
        headers=headers,
        data={"user_id": "alice", "workflow_name": "ct-analysis", "scope": scope},
        files={"file": ("ct-analysis.zip", _zip_bytes(), "application/zip")},
    )


def test_workflow_author_can_upload_user_scope(client):
    with (
        patch(
            "umrs.api.workflows.handle_upload_workflow",
            return_value=(True, "/tmp/dest", None),
        ) as mocked,
        patch(
            "umrs.api.workflows._load_extracted_workflow_json",
            return_value=("1.0", {"description": ""}),
        ),
        patch("umrs.api.workflows._register_uploaded_workflow"),
    ):
        response = _upload(client, _headers(author=True))
    assert response.status_code == 202
    assert response.json()["accepted"] is True
    mocked.assert_called_once()


def test_admin_can_upload_system_scope(client):
    with (
        patch(
            "umrs.api.workflows.handle_upload_workflow",
            return_value=(True, "/tmp/dest", None),
        ),
        patch(
            "umrs.api.workflows._load_extracted_workflow_json",
            return_value=("1.0", {"description": ""}),
        ),
        patch("umrs.api.workflows._register_uploaded_workflow"),
    ):
        response = _upload(
            client,
            _headers(user="root", project="system", admin=True),
            scope="system",
        )
    assert response.status_code == 202


def test_ordinary_user_denied(client):
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(client, _headers(author=False))
    assert response.status_code == 403
    assert "workflow_author" in response.json()["detail"]
    mocked.assert_not_called()


def test_author_cannot_upload_system_scope(client):
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(client, _headers(author=True), scope="system")
    assert response.status_code == 403
    assert "admin" in response.json()["detail"]
    mocked.assert_not_called()


def test_invalid_scope_rejected_before_archive(client):
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(client, _headers(author=True), scope="project")
    assert response.status_code == 400
    mocked.assert_not_called()


def test_spoofed_author_header_on_v2_hmac_rejected(client):
    headers = _headers(author=False)
    headers["X-Verified-Workflow-Author"] = "true"
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(client, headers)
    assert response.status_code == 401
    mocked.assert_not_called()


def test_v1_spoofed_author_header_does_not_grant_upload(client):
    headers = _headers(author=True, version=1)
    headers["X-Verified-Workflow-Author"] = "true"
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(client, headers)
    assert response.status_code == 403
    mocked.assert_not_called()


def test_v1_admin_can_still_upload(client):
    with (
        patch(
            "umrs.api.workflows.handle_upload_workflow",
            return_value=(True, "/tmp/dest", None),
        ),
        patch(
            "umrs.api.workflows._load_extracted_workflow_json",
            return_value=("1.0", {"description": ""}),
        ),
        patch("umrs.api.workflows._register_uploaded_workflow"),
    ):
        response = _upload(
            client, _headers(user="root", project="system", admin=True, version=1)
        )
    assert response.status_code == 202


def test_direct_unsigned_request_rejected(client):
    with patch("umrs.api.workflows.handle_upload_workflow") as mocked:
        response = _upload(
            client,
            {
                "X-User-ID": "alice",
                "X-Verified-Project-ID": "proj-a",
                "X-Verified-Workflow-Author": "true",
            },
        )
    assert response.status_code == 401
    mocked.assert_not_called()
