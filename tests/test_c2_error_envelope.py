"""Cycle-2 umrs-api lane (2026-10-21): the served error envelope keeps the
exception's headers and its structured detail.

Found live against the real sidecar app: every ``raise_typed`` error lost its
``x-platform-error-code`` header (the r4-contracts typed code that r4's
``_http.py`` and ``venue_rail.py`` read), because the app-wide
``structured_http_exception_handler`` rebuilt the response without
``exc.headers`` (Retry-After and WWW-Authenticate went the same way). A dict
``detail`` became its Python repr inside ``message``: the 409 for a duplicate
job read ``"{'message': ..., 'existing_job_id': 'job-…'}"``, so no client
could read the existing job id. Router-level tests never saw it: they mount
routers without the app's handler.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from umrs.api.error_responses import structured_http_exception_handler

from r4_contracts.handshake_hmac import sign_claims

SECRET = "c2-error-envelope-secret"


@pytest.fixture
def sidecar_client(monkeypatch: pytest.MonkeyPatch):
    """The real sidecar app (its handlers and routers), no database."""
    from umrs.jobs.service import get_job_service, set_job_service

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("GATEWAY_ENVIRONMENT", "test")
    prior = get_job_service()
    set_job_service(None)
    with patch("umrs.server.http_sidecar._build_db_pool", lambda _app: None):
        from umrs.server import http_sidecar

        app = http_sidecar._create_app()
    yield TestClient(app)
    set_job_service(prior)


def test_typed_code_header_survives_the_real_app(sidecar_client: TestClient) -> None:
    r = sidecar_client.get("/umrs/jobs")
    assert r.status_code == 401
    assert r.headers.get("x-platform-error-code"), dict(r.headers)
    assert r.json()["error_id"] == "auth.unauthorized"


def test_dict_detail_is_structured_not_a_repr(sidecar_client: TestClient) -> None:
    headers = {
        "X-User-ID": "alice",
        "X-Verified-Project-ID": "pa",
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, "alice", "pa", False),
    }
    r = sidecar_client.post(
        "/umrs/k2/ingest",
        json={"sources": [], "project_id": "pb"},
        headers=headers,
    )
    assert r.status_code == 409
    body = r.json()
    assert not body["message"].startswith("{"), body
    assert "does not match" in body["message"]
    assert body["details"]["error"] == "project_mismatch"
    assert body["details"]["verified_project_id"] == "pa"


def _app_raising(exc: HTTPException) -> TestClient:
    app = FastAPI()
    app.add_exception_handler(HTTPException, structured_http_exception_handler)

    @app.get("/boom")
    def boom() -> None:
        raise exc

    return TestClient(app)


def test_retry_after_and_auth_headers_pass_through() -> None:
    client = _app_raising(
        HTTPException(429, "slow down", headers={"Retry-After": "7", "WWW-Authenticate": "Bearer"})
    )
    r = client.get("/boom")
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["message"] == "slow down"


def test_dict_detail_without_message_keeps_every_field() -> None:
    r = _app_raising(HTTPException(409, {"existing_job_id": "job-1"})).get("/boom")
    assert r.status_code == 409
    assert r.json()["details"] == {"existing_job_id": "job-1"}
    assert "job-1" in r.json()["message"]
