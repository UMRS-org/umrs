"""Typed error taxonomy parity (G-P12-5, WP-16).

The zmq_sessions family is the first taxonomy consumer: every error it
surfaces carries the code on ``x-platform-error-code`` while the detail
body stays byte-identical to the legacy stringly form. These pins assert
both halves — the header is typed, the body is unchanged.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from r4_contracts.errors import ERROR_CODE_HEADER

from umrs.api.zmq_sessions import router, set_session_curve_provider
from umrs.core.session_curve_provider import SessionCredentialsProvider


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def provider():
    p = SessionCredentialsProvider(max_ttl_seconds=86400, max_per_user=5)
    set_session_curve_provider(p)
    yield p
    set_session_curve_provider(None)


@pytest.fixture()
def client(provider):
    return TestClient(_app())


def _override_user(user_id: str):
    from umrs.api.zmq_sessions import _require_user_id

    async def _mock():
        return user_id

    return _mock


def _override_identity(user_id: str):
    from umrs.api.zmq_sessions import _JWTIdentity

    async def _mock():
        return _JWTIdentity(
            user_id=user_id,
            email=f"{user_id}@test.local",
            projects=["test-project"],
            roles=["user"],
        )

    return _mock


class TestHeaderParity:
    def test_missing_auth_header_is_typed(self):
        # No dependency overrides: the raw _extract path runs.
        resp = TestClient(_app()).get("/umrs/zmq/session")
        assert resp.headers.get(ERROR_CODE_HEADER) == "auth.missing_header"
        assert resp.json()["detail"] == "Missing Authorization header"

    def test_bad_scheme_is_typed(self):
        resp = TestClient(_app()).get(
            "/umrs/zmq/session", headers={"Authorization": "token abc"}
        )
        assert resp.headers.get(ERROR_CODE_HEADER) == "auth.bad_scheme"
        assert resp.json()["detail"] == "Authorization header must be 'Bearer <token>'"

    def test_revoke_unknown_session_is_typed_not_found(self, client, provider):
        app = client.app
        from umrs.api.zmq_sessions import _require_user_id

        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.delete("/umrs/zmq/session/s-404")
        assert resp.status_code == 404
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.not_found"
        assert resp.json()["detail"] == "Session s-404 not found"
        app.dependency_overrides.clear()

    def test_revoke_other_users_session_is_typed_forbidden(self, client, provider):
        app = client.app
        from umrs.api.zmq_sessions import _extract_jwt_identity, _require_user_id

        # Owner registers, stranger revokes.
        app.dependency_overrides[_extract_jwt_identity] = _override_identity("bob")
        app.dependency_overrides[_require_user_id] = _override_user("bob")
        reg = client.post(
            "/umrs/zmq/session",
            json={
                "client_public_key": "A" * 40,
                "ttl_seconds": 300,
            },
        )
        assert reg.status_code == 201, reg.text
        session_id = reg.json()["session_id"]
        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.delete(f"/umrs/zmq/session/{session_id}")
        assert resp.status_code == 403
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.forbidden"
        assert resp.json()["detail"] == "Cannot revoke another user's session"
        app.dependency_overrides.clear()

    def test_refresh_unknown_session_is_typed(self, client):
        app = client.app
        from umrs.api.zmq_sessions import _require_user_id

        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.post(
            "/umrs/zmq/session/s-9/refresh", json={"ttl_seconds": 600}
        )
        assert resp.status_code == 404
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.not_found"
        assert resp.json()["detail"] == "Session s-9 not found"
        app.dependency_overrides.clear()


class TestProviderUnavailable:
    def test_register_503_carries_typed_header(self):
        """Dependency-direct probe: shared provider unset -> typed 503."""
        from fastapi import HTTPException

        from umrs.api.zmq_sessions import get_session_curve_provider

        set_session_curve_provider(None)
        with pytest.raises(HTTPException) as caught:
            get_session_curve_provider()
        exc = caught.value
        assert exc.status_code == 503
        assert exc.headers.get(ERROR_CODE_HEADER) == "provider.unavailable"
        assert "provider not available" in exc.detail


# ---------------------------------------------------------------------------
# WP-16 round 2 — research_sessions family becomes typed (additive header)
# ---------------------------------------------------------------------------


def test_contract_gains_research_session_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    assert codes["session.store_unavailable"] == 503
    assert codes["session.archive_cascade_failed"] == 500
    assert codes["session.not_found"] == 404


def test_research_session_404_is_typed_and_byte_identical():
    import asyncio
    from unittest.mock import MagicMock

    import pytest

    import umrs.api.research_sessions as rs

    store = MagicMock()
    store.get.return_value = None
    prev_store = rs._store
    rs._store = store
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(rs.get_session("nope-1", caller=None))
    finally:
        rs._store = prev_store

    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.detail == "Session nope-1 not found"  # byte-identical, no drift
    headers = getattr(exc, "headers", {}) or {}
    assert headers.get("x-platform-error-code") == "session.not_found"


def test_contract_files_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "files.traversal": 400,
        "files.filename_required": 400,
        "files.upload_failed": 500,
        "files.presign_failed": 500,
        "files.not_found": 404,
        "files.job_not_found": 404,
        "files.auth_unavailable": 503,
        "files.metadata_failed": 500,
        "files.delete_failed": 500,
        "files.download_failed": 500,
        "files.bucket_unreachable": 500,
        "files.bucket_denied": 403,
        "auth.verified_project_mismatch": 401,
        "auth.verified_project_required": 401,
        "auth.project_required": 401,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_files_bucket_denied_carries_typed_header(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import pytest

    import umrs.api.files as files_api

    config = SimpleNamespace(bucket="umrs-files")
    with pytest.raises(Exception) as excinfo:
        asyncio.run(
            files_api.download_file_stream(
                path="ns/key.txt",
                bucket="other-bucket",
                project_id="proj-1",
                config=config,
            )
        )
    exc = excinfo.value
    assert getattr(exc, "status_code", None) == 403, f"{exc!r}"
    headers = getattr(exc, "headers", {}) or {}
    assert headers.get("x-platform-error-code") == "files.bucket_denied"
    assert "Bucket not in download allowlist: other-bucket" == getattr(exc, "detail", "")


def test_contract_jobs_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "jobs.not_found": 404,
        "jobs.attempt_not_found": 404,
        "jobs.conflict": 409,
        "jobs.forbidden": 403,
        "jobs.bad_request": 400,
        "jobs.unprocessable": 422,
        "jobs.unavailable": 503,
        "jobs.manifest_missing": 404,
        "auth.gateway_signature_rejected": 401,
        "auth.gateway_identity_required": 401,
        "auth.caller_identity_required": 401,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_jobs_404_carries_typed_header_byte_identical():
    from unittest.mock import MagicMock

    import pytest

    import umrs.api.jobs as jobs_api
    from umrs.jobs.store import JobNotFoundError

    store = MagicMock()
    store.get_job.side_effect = JobNotFoundError("nope")
    caller = MagicMock()
    import asyncio

    with pytest.raises(Exception) as excinfo:
        asyncio.run(jobs_api.get_job("job-nothing", caller=caller, store=store))
    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.headers.get("x-platform-error-code") == "jobs.not_found"
    assert exc.detail == "Job not found: job-nothing"


def test_contract_pipelines_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "pipelines.not_found": 404,
        "pipelines.forbidden": 403,
        "pipelines.conflict": 409,
        "pipelines.bad_spec": 400,
        "pipelines.rag_profile_not_found": 404,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_ops_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "ops.governance_unavailable": 503,
        "ops.reload_unavailable": 503,
        "ops.not_found": 404,
        "ops.bad_request": 400,
        "ops.unprocessable": 422,
        "auth.admin_required": 403,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_workflows_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "workflows.not_found": 404,
        "workflows.bad_request": 400,
        "workflows.forbidden": 403,
        "workflows.conflict": 409,
        "workflows.failure": 500,
        "workflows.bad_gateway": 502,
        "workflows.upload_too_large": 413,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_models_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "models.not_found": 404,
        "models.version_not_found": 404,
        "models.forbidden": 403,
        "models.conflict": 409,
        "models.bad_request": 400,
        "models.unavailable": 503,
        "models.failure": 500,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_models_404_carries_typed_header_byte_identical():
    import asyncio
    from unittest.mock import MagicMock

    import pytest

    import umrs.api.models as models_api
    from umrs.model_registry.store.crud import ModelNotFoundError

    store = MagicMock()
    store.get_model.side_effect = ModelNotFoundError("nope")
    ctx = MagicMock(user_id="u", memberships={"proj-1"})

    with pytest.raises(Exception) as excinfo:
        result = models_api.get_model("nope", ctx=ctx, store=store)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    exc = excinfo.value
    assert exc.status_code == 404
    headers = getattr(exc, "headers", {}) or {}
    assert headers.get("x-platform-error-code") == "models.not_found"
    assert exc.detail == "Model not found: nope"


def test_contract_endpoints_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "endpoints.not_found": 404,
        "endpoints.forbidden": 403,
        "endpoints.conflict": 409,
        "endpoints.bad_request": 400,
        "endpoints.unavailable": 503,
        "endpoints.failure": 500,
        "endpoints.not_implemented": 501,
        "endpoints.bad_gateway": 502,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_loops_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "loops.not_found": 404,
        "loops.forbidden": 403,
        "loops.conflict": 409,
        "loops.bad_request": 400,
        "loops.unprocessable": 422,
        "loops.unavailable": 503,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_companion_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "companion.not_found": 404,
        "companion.forbidden": 403,
        "companion.unauthorized": 401,
        "companion.conflict": 409,
        "companion.gone": 410,
        "companion.unavailable": 503,
        "companion.unprocessable": 422,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_rag_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "rag.unavailable": 503,
        "rag.forbidden": 403,
        "rag.not_found": 404,
        "rag.bad_request": 400,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_artifacts_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "artifacts.not_found": 404,
        "artifacts.unavailable": 503,
        "artifacts.bad_request": 400,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_mcp_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "mcp.not_found": 404,
        "mcp.unavailable": 503,
        "mcp.unprocessable": 422,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_knowledge2_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "knowledge2.unknown_graph": 400,
        "knowledge2.graph_not_active": 400,
        "knowledge2.bad_request": 400,
        "knowledge2.unavailable": 503,
        "knowledge2.failure": 500,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_ragservice_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "ragservice.unavailable": 500,
        "ragservice.not_found": 404,
        "ragservice.bad_request": 400,
        "ragservice.failure": 500,
    }
    for code, status in expected.items():
        assert codes[code] == status, code


def test_contract_usersources_family_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    expected = {
        "usersources.unavailable": 503,
        "usersources.unauthorized": 401,
        "usersources.bad_request": 400,
        "usersources.failure": 500,
        "usersources.not_found": 404,
    }
    for code, status in expected.items():
        assert codes[code] == status, code
