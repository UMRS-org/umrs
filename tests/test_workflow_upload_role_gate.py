"""ADR-027: workflow upload role gate tests (P0).

Gate semantics (header-driven, strict-or-verified-identity requests):
- ``scope=system`` → admin required (aligns with ModelScope/SkillScope).
- Other uploads → ``workflow_author`` role claim or admin required.
- Every decision (allow + deny) emits a ``workflow_upload`` audit event.
"""

from __future__ import annotations

import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# splice removed (ADR-053): pytest auto-inserts tests/ for sibling helpers; umrs editable

from umrs.api import audit_events, workflows  # noqa: E402


@pytest.fixture(autouse=True)
def _identity_gate_bypass(monkeypatch: pytest.MonkeyPatch):
    """The Wave-6 identity gate itself is covered elsewhere; isolate the
    role gate."""
    monkeypatch.setattr(workflows, "_require_verified_workflow_identity", lambda *_a, **_k: None)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(workflows.router)
    monkeypatch.setattr(workflows, "_check_workflow_quota", lambda *_a, **_k: None)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    return TestClient(app)


def _zip_bytes() -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("workflow.json", '{"version": "0.1", "description": "t"}')
    return buf.getvalue()


def _post(client, headers: dict, scope: str = "user"):
    return client.post(
        "/umrs/workflows/upload",
        headers=headers,
        data={"scope": scope, "workflow_name": "wf-t"},
        files={"file": ("wf.zip", _zip_bytes(), "application/zip")},
    )


BASE = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}
AUTHOR = {**BASE, "X-Verified-Workflow-Author": "true"}
ADMIN = {**BASE, "X-Verified-Is-Admin": "true"}


def _audits() -> list[dict]:
    return [e for e in audit_events._audit_buffer if e.get("event") == "workflow_upload"]


def _stub_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workflows, "handle_upload_workflow", lambda *_a, **_k: (True, "/tmp/wf", "")
    )
    monkeypatch.setattr(
        workflows,
        "_load_extracted_workflow_json",
        lambda *_a, **_k: ("0.1", {"description": "t"}),
    )
    monkeypatch.setattr(workflows, "_register_uploaded_workflow", lambda **_k: None)


def test_no_role_upload_denied_and_audited(client):
    resp = _post(client, BASE)
    assert resp.status_code == 403
    assert "workflow_author" in resp.json()["detail"]
    events = _audits()
    assert len(events) == 1
    assert events[0]["outcome"] == "denied"
    assert events[0]["actor"] == "alice"


def test_system_scope_requires_admin_even_for_author(client):
    resp = _post(client, AUTHOR, scope="system")
    assert resp.status_code == 403
    assert "admin" in resp.json()["detail"]
    assert _audits()[-1]["details"]["reason"] == "system scope requires admin"


def test_explicit_not_author_denied(client):
    # An explicit author=false claim without admin cannot mint uploads.
    resp = _post(client, {**BASE, "X-Verified-Workflow-Author": "false"})
    assert resp.status_code == 403
    assert _audits()[-1]["outcome"] == "denied"


def test_author_upload_succeeds_and_audits_allow_nonstrict(client, monkeypatch):
    # Compat mode: the presented role header is trusted as-is (sig carrier
    # absent); strict-mode signature coverage is tested separately below.
    monkeypatch.setattr(workflows, "_strict_auth_mode", lambda: False)
    _stub_pipeline(monkeypatch)
    resp = _post(client, AUTHOR)
    assert resp.status_code == 202, resp.text
    events = _audits()
    assert events[-1]["outcome"] == "success"
    assert events[-1]["scope"] == "user"
    assert events[-1]["details"]["workflow"] == "wf-t"


def test_admin_system_upload_succeeds_and_audits_allow(client, monkeypatch):
    _stub_pipeline(monkeypatch)
    resp = _post(client, ADMIN, scope="system")
    assert resp.status_code == 202, resp.text
    events = _audits()
    assert events[-1]["outcome"] == "success"
    assert events[-1]["scope"] == "system"


STRICT_SECRET = "sig-secret-027"


def test_strict_author_sig_valid_passes(client, monkeypatch):
    from r4_contracts.handshake_hmac import sign_workflow_author_claim

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(workflows, "_strict_auth_mode", lambda: True)
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", STRICT_SECRET)
    sig = sign_workflow_author_claim(STRICT_SECRET, "alice", "proj-a", True)
    resp = _post(client, {**AUTHOR, "X-Verified-Workflow-Author-HMAC": sig})
    assert resp.status_code == 202, resp.text
    assert _audits()[-1]["outcome"] == "success"


def test_strict_author_sig_forged_denied(client, monkeypatch):
    monkeypatch.setattr(workflows, "_strict_auth_mode", lambda: True)
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", STRICT_SECRET)
    resp = _post(client, {**AUTHOR, "X-Verified-Workflow-Author-HMAC": "0" * 64})
    assert resp.status_code == 403
    assert _audits()[-1]["outcome"] == "denied"


def test_strict_admin_without_author_claim_passes(client, monkeypatch):
    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(workflows, "_strict_auth_mode", lambda: True)
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", STRICT_SECRET)
    resp = _post(client, ADMIN)
    assert resp.status_code == 202, resp.text
