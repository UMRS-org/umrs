"""An upload's workflow_name is one directory under its owner's tree (ADR-074 D-2/D-8).

An absolute name or ``..`` let an upload write its files wherever the server
may write: a lazily imported umrs module (uploaded code in the server) or
another owner's workflow tree (their next run executes it). Containment is a
path test, not a prefix test: ``wf2`` is not inside ``wf``.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.server.file_handling import handle_upload_workflow, workflow_name_refusal

_WORKFLOW = {
    "name": "wf",
    "version": "1.0",
    "source": {"type": "local_path", "path": "."},
    "steps": [{"name": "s1", "entrypoint": "steps:StepOne"}],
}


def _zip(tmp_path: Path, members: dict[str, str] | None = None, workflow=None) -> str:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("workflow.json", json.dumps(workflow or _WORKFLOW))
        for name, text in (members or {"steps.py": "class StepOne: pass\n"}).items():
            zf.writestr(name, text)
    path = tmp_path / f"upload-{len(list(tmp_path.glob('upload-*')))}.zip"
    path.write_bytes(buffer.getvalue())
    return str(path)


@pytest.fixture
def state(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("UMRS_WORKFLOW_STATE_DIR", str(tmp_path / "state"))
    return tmp_path / "state"


@pytest.mark.parametrize(
    "name",
    ["{target}", "../../bob/wf", "../bob", ".", "..", "", "a/b", "-x", "x" * 129, "wf\n"],
)
def test_a_name_that_is_not_one_directory_is_refused(tmp_path, state, name):
    target = tmp_path / "server" / "umrs" / "server"
    target.mkdir(parents=True)
    ok, dest, message = handle_upload_workflow("alice", _zip(tmp_path), name.format(target=target))
    assert not ok and dest is None
    assert "Invalid workflow_name" in message
    assert list(target.iterdir()) == []
    assert not (state / "extracted" / "bob").exists()


def test_a_plain_name_extracts_under_the_owner(tmp_path, state):
    ok, dest, _ = handle_upload_workflow("alice", _zip(tmp_path), "demo08_yield-eval.v2")
    assert ok
    assert Path(dest) == (state / "extracted" / "alice" / "demo08_yield-eval.v2").resolve()


def test_a_zip_member_in_a_sibling_directory_is_refused(tmp_path, state):
    members = {"steps.py": "", "../wf2/steps.py": "print('owned')\n"}
    ok, _dest, message = handle_upload_workflow("alice", _zip(tmp_path, members), "wf")
    assert not ok and "unsafe path: ../wf2/steps.py" in message
    assert not (state / "extracted" / "alice" / "wf2").exists()


def test_a_source_path_in_a_sibling_directory_is_refused(tmp_path, state):
    (state / "extracted" / "alice" / "wf2").mkdir(parents=True)
    workflow = {**_WORKFLOW, "source": {"type": "local_path", "path": "../wf2"}}
    ok, _dest, message = handle_upload_workflow("alice", _zip(tmp_path, workflow=workflow), "wf")
    assert not ok and "source path escapes" in message


def test_the_name_rule():
    assert workflow_name_refusal("video-ai-yolo") is None
    assert workflow_name_refusal("r4-complex-dag-demo") is None
    assert workflow_name_refusal(None) is not None
    assert workflow_name_refusal("my workflow") is not None


def test_http_upload_refuses_a_traversing_name(tmp_path, state, monkeypatch):
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
            data={"workflow_name": "../bob/wf"},
            files={"file": ("wf.zip", Path(_zip(tmp_path)).read_bytes())},
        )
    assert response.status_code == 400, response.text
    assert "Invalid workflow_name" in response.text
    assert not (state / "extracted" / "bob").exists()
