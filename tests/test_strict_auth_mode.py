"""Tests for UMRS strict auth mode fallback behavior."""

from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_endpoint_project_id_rejects_header_in_strict_mode(monkeypatch):
    """Strict mode disables MVP X-Project-ID trust."""
    from umrs.api import endpoints

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    with pytest.raises(HTTPException) as exc_info:
        endpoints.get_project_id(x_project_id="project-a")

    assert exc_info.value.status_code == 401
    assert "fallbacks are disabled" in exc_info.value.detail


def test_endpoint_project_id_accepts_verified_header_in_strict_mode(monkeypatch):
    """Strict mode accepts the Gateway-issued verified project context.

    Wave-3 note: acceptance requires the HMAC-signed identity tuple — plain
    unsigned verified headers are rejected to block direct-sidecar spoofing.
    """
    from r4_contracts.handshake_hmac import sign_claims
    from umrs.api import endpoints

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")

    assert (
        endpoints.get_project_id(
            x_user_id="gateway",
            x_verified_project_id="project-a",
            x_verified_hmac=sign_claims("test-secret", "gateway", "project-a", False),
        )
        == "project-a"
    )


def test_endpoint_optional_project_id_returns_none_in_strict_mode(monkeypatch):
    """Admin helpers should not synthesize project IDs in strict mode."""
    from umrs.api import endpoints

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    assert endpoints.get_project_id_optional(x_project_id="project-a") is None


def test_files_project_id_rejects_bearer_default_in_strict_mode(monkeypatch):
    """Strict mode disables the default-project bearer fallback."""
    from umrs.api import files

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    with pytest.raises(HTTPException) as exc_info:
        files.get_project_id(authorization="Bearer token")

    assert exc_info.value.status_code == 401
    assert "fallbacks are disabled" in exc_info.value.detail


def test_files_project_id_accepts_verified_header_in_strict_mode(monkeypatch):
    """Strict mode accepts verified project context for file APIs."""
    from r4_contracts.handshake_hmac import sign_claims
    from umrs.api import files

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")

    assert (
        files.get_project_id(
            x_user_id="gateway",
            x_verified_project_id="project-a",
            x_verified_hmac=sign_claims("test-secret", "gateway", "project-a", False),
        )
        == "project-a"
    )


def test_mvp_project_id_header_still_supported_by_default(monkeypatch):
    """Development mode keeps the existing explicit-header behavior."""
    from umrs.api import endpoints, files

    monkeypatch.delenv("UMRS_AUTH_MODE", raising=False)

    assert endpoints.get_project_id(x_project_id="project-a") == "project-a"
    assert files.get_project_id(x_project_id="project-a") == "project-a"
