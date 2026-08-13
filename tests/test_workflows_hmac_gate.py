"""Wave-6 F1 — workflow routers must not trust unsigned gateway headers.

Before this fix ``_require_verified_workflow_identity`` accepted presence of
``X-User-ID`` + ``X-Verified-Project-ID`` as sufficient under strict mode,
``_verified_workflow_params`` accepted self-issued
``X-Verified-Project-Memberships``/``X-Verified-Is-Admin`` for cross-project
authorization, and the trigger route minted ``_identity_verified=True`` into
the durable workflow payload when both headers were merely present. All of
those are exactly the trust-marker failures wave-3/4 eliminated everywhere
else. The regression suite below pins the corrected behavior.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from r4_contracts.handshake_hmac import sign_claims
from umrs.api import workflows
from umrs.api.workflows import (
    _require_verified_workflow_identity,
    _verified_workflow_params,
)


def _secret_env(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "wave6-secret")
    return "wave6-secret"


class TestRequireVerifiedWorkflowIdentity:
    def test_strict_presence_only_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _secret_env(monkeypatch)
        with pytest.raises(HTTPException) as exc_info:
            _require_verified_workflow_identity("victim", "victim-project", None, None)
        assert exc_info.value.status_code == 401
        assert "signature rejected" in str(exc_info.value.detail)

    def test_strict_forged_hmac_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _secret_env(monkeypatch)
        forged = sign_claims("wrong-secret", "victim", "victim-project", False)
        with pytest.raises(HTTPException) as exc_info:
            _require_verified_workflow_identity("victim", "victim-project", "false", forged)
        assert exc_info.value.status_code == 401

    def test_strict_valid_hmac_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret = _secret_env(monkeypatch)
        good = sign_claims(secret, "alice", "alice-proj", False)
        # Must not raise.
        _require_verified_workflow_identity("alice", "alice-proj", "false", good)

    def test_strict_valid_admin_hmac_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret = _secret_env(monkeypatch)
        good = sign_claims(secret, "root", "system", True)
        _require_verified_workflow_identity("root", "system", "true", good)

    def test_mvp_mode_presence_still_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        _require_verified_workflow_identity("u", "p", None, None)


class TestVerifiedWorkflowParamsMembershipsStrict:
    def test_strict_memberships_header_not_trusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _secret_env(monkeypatch)
        # Even with a forged memberships header naming the victim project,
        # strict mode must refuse it: memberships are not part of the v1
        # signature tuple.
        with pytest.raises(HTTPException) as exc_info:
            _verified_workflow_params(
                {"project_id": "victim-proj"},
                "attacker-proj",
                "victim-proj,attacker-proj",
                "false",
            )
        assert exc_info.value.status_code == 403

    def test_strict_own_primary_project_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _secret_env(monkeypatch)
        params = _verified_workflow_params(
            {"project_id": "alice-proj"},
            "alice-proj",
            None,
            "false",
        )
        assert params["project_id"] == "alice-proj"

    def test_mvp_mode_memberships_still_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        params = _verified_workflow_params(
            {"project_id": "team-proj"},
            "alice-proj",
            "team-proj",
            "false",
        )
        assert params["project_id"] == "team-proj"


class TestWorkflowsHandlersCarryHmac:
    def test_handlers_accept_hmac_header_params(self) -> None:
        import inspect

        for name in (
            "list_workflows",
            "list_workflow_triggers",
            "describe_workflow",
            "upload_workflow",
            "trigger_workflow",
        ):
            handler = getattr(workflows, name)
            sig = inspect.signature(handler)
            assert "x_verified_hmac" in sig.parameters, name
            assert "x_verified_is_admin" in sig.parameters, name
