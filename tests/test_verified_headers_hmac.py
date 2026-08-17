"""HMAC handshake contract tests (live-spoof-class closure).

Pre-fix PoC (2026-08-13, live): X-User-ID + X-Verified-Project-ID (+ optional
Is-Admin) headers alone returned 200 on /umrs/endpoints with no JWT at all —
a data0-local shell could mint admin. The handshake gate must reject forged
headers in strict mode and accept only gateway-issued signatures in
authenticated deployments.
"""

from __future__ import annotations

import uuid

import pytest

from r4_contracts.handshake_hmac import sign_claims, verify_claims


class TestHandshakeContract:
    def test_sign_verify_roundtrip(self):
        mac = sign_claims("s3cret", "alice", "proj-a", True)
        assert verify_claims("s3cret", mac, "alice", "proj-a", True)

    def test_tampered_user_rejected(self):
        mac = sign_claims("s3cret", "alice", "proj-a", True)
        assert not verify_claims("s3cret", mac, "attacker", "proj-a", True)

    def test_tampered_project_rejected(self):
        mac = sign_claims("s3cret", "alice", "proj-a", True)
        assert not verify_claims("s3cret", mac, "alice", "proj-b", True)

    def test_tampered_admin_flag_rejected(self):
        mac = sign_claims("s3cret", "alice", "proj-a", False)
        assert not verify_claims("s3cret", mac, "alice", "proj-a", True)

    def test_wrong_secret_rejected(self):
        mac = sign_claims("s3cret", "alice", "proj-a", True)
        assert not verify_claims("wrong", mac, "alice", "proj-a", True)

    def test_missing_inputs_never_trusted(self):
        assert not verify_claims(None, "x", "a", "p", False)
        assert not verify_claims("s", None, "a", "p", False)
        assert not verify_claims("s", "x", "", "p", False)


class TestStrictResolver:
    def _resolver_module(self):
        from umrs.api import endpoints

        return endpoints

    def test_signed_headers_accepted(self, monkeypatch):
        endpoints = self._resolver_module()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims("test-secret", "alice", "proj-a", False)
        result = endpoints._resolve_authenticated_project_id(
            None,
            "alice",
            "proj-a",
            None,
            x_verified_is_admin=None,
            x_verified_hmac=mac,
        )
        assert result == "proj-a"

    def test_unsigned_headers_rejected(self, monkeypatch):
        from fastapi import HTTPException

        endpoints = self._resolver_module()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        with pytest.raises(HTTPException) as exc:
            endpoints._resolve_authenticated_project_id(
                None,
                "attacker",
                "proj-a",
                None,
                x_verified_is_admin=None,
                x_verified_hmac=None,
            )
        assert exc.value.status_code == 401

    def test_forged_signature_rejected(self, monkeypatch):
        from fastapi import HTTPException

        endpoints = self._resolver_module()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        forged = uuid.uuid4().hex + uuid.uuid4().hex[:32]
        with pytest.raises(HTTPException):
            endpoints._resolve_authenticated_project_id(
                None,
                "attacker",
                "proj-a",
                None,
                x_verified_is_admin=None,
                x_verified_hmac=forged,
            )

    def test_secret_absent_strict_never_trusts_headers(self, monkeypatch):
        from fastapi import HTTPException

        endpoints = self._resolver_module()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.delenv("UMRS_HANDSHAKE_HMAC_SECRET", raising=False)
        with pytest.raises(HTTPException):
            endpoints._resolve_authenticated_project_id(
                None,
                "alice",
                "proj-a",
                None,
                x_verified_is_admin=None,
                x_verified_hmac="anything",
            )

    def test_admin_flag_flip_invalidates(self, monkeypatch):
        endpoints = self._resolver_module()
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims("test-secret", "alice", "proj-a", False)
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            endpoints._resolve_authenticated_project_id(
                None,
                "alice",
                "proj-a",
                None,
                x_verified_is_admin="true",
                x_verified_hmac=mac,
            )


class TestCallerAdmin:
    def test_admin_requires_signed_admin_tuple(self, monkeypatch):
        from umrs.api import endpoints

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims("test-secret", "alice", "proj-a", True)
        assert endpoints._caller_is_admin(
            None, "alice", "true", x_verified_project_id="proj-a", x_verified_hmac=mac
        )
        # non-admin signed but claiming admin → denied
        mac2 = sign_claims("test-secret", "alice", "proj-a", False)
        assert not endpoints._caller_is_admin(
            None, "alice", "true", x_verified_project_id="proj-a", x_verified_hmac=mac2
        )


class TestWorkflowAuthorClaim:
    def test_v2_author_tamper_rejected(self, monkeypatch):
        from fastapi import HTTPException
        from r4_contracts.handshake_hmac import sign_claims_v2
        from umrs.api._verified_headers import resolve_caller

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims_v2("test-secret", "alice", "proj-a", False, False)
        with pytest.raises(HTTPException) as exc:
            resolve_caller("alice", "proj-a", "false", mac, "true")
        assert exc.value.status_code == 401

    def test_v2_author_accepted(self, monkeypatch):
        from r4_contracts.handshake_hmac import sign_claims_v2
        from umrs.api._verified_headers import resolve_caller

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims_v2("test-secret", "alice", "proj-a", False, True)
        identity = resolve_caller("alice", "proj-a", "false", mac, "true")
        assert identity is not None
        assert identity.is_workflow_author is True

    def test_v1_does_not_grant_author(self, monkeypatch):
        from umrs.api._verified_headers import resolve_caller

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
        mac = sign_claims("test-secret", "alice", "proj-a", False)
        identity = resolve_caller("alice", "proj-a", "false", mac, "true")
        assert identity is not None
        assert identity.is_workflow_author is False
