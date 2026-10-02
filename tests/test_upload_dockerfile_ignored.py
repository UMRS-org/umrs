"""An uploaded Dockerfile is never built on the server host (ADR-074 D-7).

The upload is accepted; the Dockerfile is ignored with a named notice, and
nothing rewrites the workflow's ``routing_hints.image``.
"""

from __future__ import annotations

import json
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.server.file_handling import DOCKERFILE_IGNORED_NOTICE, handle_upload_workflow

_WORKFLOW = {
    "name": "dock",
    "version": "1.0",
    "description": "One-step upload fixture.",
    "source": {"type": "local_path", "path": "."},
    "steps": [{"name": "s1", "entrypoint": "steps:StepOne", "depends_on": []}],
}


def _zip(tmp_path: Path, *, dockerfile: bool) -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("workflow.json", json.dumps(_WORKFLOW))
        zf.writestr("steps.py", "class StepOne: pass\n")
        if dockerfile:
            zf.writestr("Dockerfile", "FROM alpine\nRUN touch /pwned\n")
    path = tmp_path / "upload.zip"
    path.write_bytes(buffer.getvalue())
    return str(path)


@pytest.fixture(autouse=True)
def _no_host_process(tmp_path, monkeypatch):
    """Any process the upload tried to start (a podman build) fails the test."""
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))

    def _refuse(*args, **_kw):
        raise AssertionError(f"upload started a host process: {args!r}")

    monkeypatch.setattr(subprocess, "run", _refuse)
    monkeypatch.setattr(subprocess, "Popen", _refuse)


def test_dockerfile_is_ignored_with_a_notice(tmp_path):
    ok, dest, notice = handle_upload_workflow("alice", _zip(tmp_path, dockerfile=True), "dock")
    assert ok, notice
    assert notice == DOCKERFILE_IGNORED_NOTICE
    workflow = json.loads(Path(dest, "workflow.json").read_text())
    assert "routing_hints" not in workflow["steps"][0]


def test_no_dockerfile_no_notice(tmp_path):
    ok, _dest, notice = handle_upload_workflow("alice", _zip(tmp_path, dockerfile=False), "dock")
    assert ok and notice is None


def test_http_upload_reports_the_notice(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
    from r4_contracts.handshake_hmac import sign_claims

    app = FastAPI()
    app.include_router(router)
    headers = {
        "X-User-ID": "alice",
        "X-Verified-Project-ID": "project-a",
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": sign_claims("test-secret", "alice", "project-a", True),
    }
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=str(tmp_path / "catalog.json"),
    ):
        response = TestClient(app).post(
            "/umrs/workflows/upload",
            headers=headers,
            data={"workflow_name": "dock"},
            files={"file": ("dock.zip", Path(_zip(tmp_path, dockerfile=True)).read_bytes())},
        )
    assert response.status_code == 202, response.text
    assert DOCKERFILE_IGNORED_NOTICE in response.json()["message"]
