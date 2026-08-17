"""Sidecar uses the shared r4-contracts HMAC signer (no inlined twin)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from r4_contracts.handshake_hmac import HEADER_WORKFLOW_AUTHOR, sign_claims_v2


def _load_sidecar():
    sidecar_path = Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "main.py"
    assert sidecar_path.is_file(), f"expected sidecar at {sidecar_path}"
    asyncpg = types.ModuleType("asyncpg")
    asyncpg.Pool = object  # type: ignore[attr-defined]
    sys.modules["asyncpg"] = asyncpg
    jose = types.ModuleType("jose")
    jose.JWTError = Exception  # type: ignore[attr-defined]
    jose_jwt = types.ModuleType("jose.jwt")
    jose.jwt = jose_jwt  # type: ignore[attr-defined]
    sys.modules["jose"] = jose
    sys.modules["jose.jwt"] = jose_jwt
    spec = importlib.util.spec_from_file_location("_sidecar_main", sidecar_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_sidecar_normalizes_admin_role_case_in_hmac(monkeypatch):
    sidecar = _load_sidecar()
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s3crt")
    headers = sidecar._verified_project_headers(
        {
            "user_id": "root",
            "project_id": "system",
            "roles": ["Admin"],
        }
    )
    assert headers["X-Verified-Is-Admin"] == "true"
    assert headers["X-Verified-HMAC"] == sign_claims_v2(
        "s3crt", "root", "system", True, False
    )


def test_sidecar_emits_v2_hmac_covering_workflow_author(monkeypatch):
    sidecar = _load_sidecar()
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s3crt")
    headers = sidecar._verified_project_headers(
        {
            "user_id": "alice",
            "project_id": "proj-a",
            "roles": ["workflow_author"],
        }
    )
    assert headers[HEADER_WORKFLOW_AUTHOR] == "true"
    assert headers["X-Verified-HMAC"] == sign_claims_v2(
        "s3crt", "alice", "proj-a", False, True
    )


def test_sidecar_maps_groups_when_roles_empty(monkeypatch):
    sidecar = _load_sidecar()
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s3crt")
    claims = sidecar._extract_sidecar_claims(
        {
            "sub": "alice",
            "project_id": "proj-a",
            "roles": [],
            "groups": ["workflow_author"],
            "exp": 9999999999,
        }
    )
    headers = sidecar._verified_project_headers(claims)
    assert headers[HEADER_WORKFLOW_AUTHOR] == "true"


def test_sidecar_audit_uses_shared_admin_role_helper():
    source = Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "main.py"
    text = source.read_text(encoding="utf-8")
    assert "has_admin_role" in text
    assert '"admin" not in (claims.get("roles")' not in text


def test_sidecar_has_no_inlined_signer():
    sidecar = _load_sidecar()
    assert not hasattr(sidecar, "_sign_identity_hmac")
    source = Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "main.py"
    text = source.read_text(encoding="utf-8")
    assert "_sign_identity_hmac" not in text
    assert "sign_claims_v2" in text
