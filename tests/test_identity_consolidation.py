"""R1 (drift remediation 2026-08-22): canonical identity resolution.

Matrix-tests for umrs.api._identity plus per-surface parity head-checks
(chat / llm / rag / skills / scope adapter). All module-specific behavior
verified vs the pre-consolidation semantics:

- verified (X-Verified-Project-ID) wins over legacy (X-Project-ID)
- strict mode requires verified project (401)
- HMAC required only when ``verify_hmac=True``
- query-param identity logs one deprecation warning per feature
"""

from __future__ import annotations

import logging

import pytest
from fastapi import HTTPException, Request
from umrs.api import _identity


def _request(headers: dict[str, str] | None = None) -> Request:
    hdrs = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(scope={"type": "http", "headers": hdrs})


def _status(exc: HTTPException) -> int:
    return exc.status_code


@pytest.fixture(autouse=True)
def _mvp_mode(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    yield
    _identity._LEGACY_PARAM_WARN_ONCE.clear()


@pytest.fixture
def _strict_mode(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    yield
    _identity._LEGACY_PARAM_WARN_ONCE.clear()


class TestVerifiedTier:
    def test_verified_project_read(self):
        identity = _identity.request_identity(
            _request(
                {
                    "x-user-id": "alice",
                    "x-verified-project-id": "lab1",
                    "x-project-id": "attacker",
                }
            )
        )
        assert (identity.user_id, identity.project_id) == ("alice", "lab1")
        assert identity.anonymous is False
        assert identity.legacy is False

    def test_admin_flag_parsed(self):
        identity = _identity.request_identity(
            _request(
                {
                    "x-verified-project-id": "lab1",
                    "x-verified-is-admin": "true",
                }
            )
        )
        assert identity.is_admin is True

    @pytest.mark.usefixtures("_strict_mode")
    def test_verify_hmac_rejects_missing_signature(self):
        with pytest.raises(HTTPException) as err:
            _identity.request_identity(
                _request({"x-verified-project-id": "lab1", "x-user-id": "a"}),
                verify_hmac=True,
            )
        assert _status(err.value) == 401
        assert "signature rejected" in err.value.detail

    def test_verify_hmac_off_by_default(self):
        identity = _identity.request_identity(
            _request({"x-verified-project-id": "lab1"}),
        )
        assert identity.project_id == "lab1"


class TestStrictMode:
    def test_missing_verified_rejected(self, _strict_mode):
        with pytest.raises(HTTPException) as err:
            _identity.request_identity(_request({"x-project-id": "lab1", "x-user-id": "a"}))
        assert _status(err.value) == 401
        assert "Verified-Project-ID" in err.value.detail

    def test_verified_satisfied(self, _strict_mode):
        identity = _identity.request_identity(_request({"x-verified-project-id": "lab1"}))
        assert identity.project_id == "lab1"

    def test_strict_optional_needs_verified(self, _strict_mode):
        with pytest.raises(HTTPException):
            _identity.request_identity(_request({}), require_project=False)


class TestLegacyAndAnonymousTiers:
    def test_legacy_project_adapter(self):
        identity = _identity.request_identity(
            _request({"x-project-id": "lab1", "x-user-id": "bob"})
        )
        assert identity.project_id == "lab1"
        assert identity.legacy is True

    def test_missing_project_rejected_when_required(self):
        with pytest.raises(HTTPException) as err:
            _identity.request_identity(_request({}))
        assert _status(err.value) == 401

    def test_anonymous_when_optional(self):
        identity = _identity.request_identity(_request({}), require_project=False)
        assert identity.anonymous is True
        assert identity.project_id == ""


class TestScopeAdapter:
    def test_verified_maps_to_scope(self):
        from umrs.knowledge.access_scope import CallerScope

        scope = _identity.request_caller(
            _request(
                {
                    "x-user-id": "alice",
                    "x-verified-project-id": "lab1",
                    "x-verified-is-admin": "1",
                }
            )
        )
        assert scope == CallerScope(user_id="alice", project_id="lab1", is_admin=True)

    def test_legacy_and_anonymous_shapes(self):
        scope = _identity.request_caller(_request({"x-project-id": "lab1"}))
        assert scope.project_id == "lab1"
        assert scope.user_id == ""

        anon = _identity.request_caller(_request({}))
        assert anon.anonymous is True


class TestTupleAdapter:
    def test_verified_identity_tuple(self):
        assert _identity.verified_identity(
            _request({"x-user-id": "a", "x-verified-project-id": "p"})
        ) == ("a", "p")


class TestQueryParamDeprecation:
    def test_warns_once_per_feature(self, caplog):
        with caplog.at_level(logging.WARNING, logger="umrs.api.identity"):
            _identity.warn_legacy_params_once("test.feature")
            _identity.warn_legacy_params_once("test.feature")
        hits = [r for r in caplog.records if "DEPRECATED" in r.message]
        assert len(hits) == 1

    def test_different_features_each_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="umrs.api.identity"):
            _identity.warn_legacy_params_once("feature.a")
            _identity.warn_legacy_params_once("feature.b")
        assert len(caplog.records) == 2


class TestSurfaceParity:
    """Head-checks that each surface delegates to the canonical resolver."""

    def test_chat_delegates(self):
        from umrs.api import chat

        assert chat._get_caller_identity(
            _request({"x-user-id": "a", "x-verified-project-id": "p"})
        ) == ("a", "p")

    def test_llm_delegates(self):
        from umrs.api import llm

        # H-1 (2026-10-18): llm._get_caller_identity became _verified_caller,
        # which returns the full verified identity (the admin flag included).
        identity = llm._verified_caller(_request({"x-user-id": "a", "x-verified-project-id": "p"}))
        assert (identity.user_id, identity.project_id) == ("a", "p")

    def test_rag_delegates(self):
        from umrs.api import rag

        assert rag._get_caller_identity(
            _request({"x-user-id": "a", "x-verified-project-id": "p"})
        ) == ("a", "p")

    def test_skills_delegates(self):
        from umrs.api.skills import _caller_identity

        identity = _caller_identity(
            _request(
                {
                    "x-user-id": "a",
                    "x-verified-project-id": "p",
                    "x-verified-is-admin": "yes",
                }
            )
        )
        assert (identity.user_id, identity.project_id, identity.is_admin) == (
            "a",
            "p",
            True,
        )

    def test_jobs_scope_compat_properties(self):
        from umrs.api.jobs import CallerIdentity

        admin = CallerIdentity(user_id="root", project_id="lab", is_admin=True)
        assert admin.is_anonymous is False
        assert admin.personal_project == "user:root"
        anon = CallerIdentity(user_id="", project_id="")
        assert anon.is_anonymous is True
