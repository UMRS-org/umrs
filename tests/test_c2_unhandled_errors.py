"""Cycle-2 umrs-api lane (2026-10-21): an exception no route handles still
leaves the real app as the typed JSON envelope, with an honest status.

Found live (gateway -> sidecar -> umrs, Postgres stopped mid-run): every
store-backed route (/umrs/jobs, models, knowledge/graphs, projects, skills,
loops, ...) answered a plain-text ``Internal Server Error`` 500 instead of a
typed 503, while ``/readiness`` already knew the database was down. A NUL
character in a JSON body (``"a\\u0000b"``) reached Postgres on ~10 routes
(evidence, knowledge/graphs, scientific-catalogs, traces, mcp/call, ...)
and came back as the same untyped 500 instead of a 400.
"""

from __future__ import annotations

from unittest.mock import patch

import psycopg2
import psycopg2.pool
import pytest
from fastapi.testclient import TestClient
from umrs.api.jobs import get_job_store

from r4_contracts.handshake_hmac import sign_claims

SECRET = "c2-unhandled-secret"
HEADERS = {
    "X-User-ID": "alice",
    "X-Verified-Project-ID": "pa",
    "X-Verified-Is-Admin": "false",
    "X-Verified-HMAC": sign_claims(SECRET, "alice", "pa", False),
}


class _RaisingStore:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def list_jobs(self, **_kwargs: object) -> list:
        raise self._exc


@pytest.fixture
def make_client(monkeypatch: pytest.MonkeyPatch):
    from umrs.jobs.service import get_job_service, set_job_service

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.setenv("GATEWAY_ENVIRONMENT", "test")
    prior = get_job_service()
    set_job_service(None)
    with patch("umrs.server.http_sidecar._build_db_pool", lambda _app: None):
        from umrs.server import http_sidecar

        app = http_sidecar._create_app()

    def _client(exc: BaseException) -> TestClient:
        app.dependency_overrides[get_job_store] = lambda: _RaisingStore(exc)
        return TestClient(app, raise_server_exceptions=False)

    yield _client
    set_job_service(prior)


def _typed(r) -> dict:
    assert r.headers["content-type"].startswith("application/json"), r.text
    body = r.json()
    assert body["error_id"] and body["message"] and body["request_id"]
    assert "Traceback" not in r.text and 'File "' not in r.text
    return body


@pytest.mark.parametrize(
    "exc",
    [
        psycopg2.OperationalError(
            'connection to server at "127.0.0.1", port 5432 failed: Connection refused'
        ),
        psycopg2.InterfaceError("connection already closed"),
        psycopg2.pool.PoolError("connection pool exhausted"),
    ],
)
def test_database_outage_is_a_typed_503(make_client, exc: BaseException) -> None:
    r = make_client(exc).get("/umrs/jobs", headers=HEADERS)
    assert r.status_code == 503
    body = _typed(r)
    assert body["error_id"] == "service.unavailable"
    assert "127.0.0.1" not in body["message"]  # no connection details leak
    assert r.headers.get("retry-after")


def test_wrapped_database_outage_is_a_typed_503(make_client) -> None:
    try:
        raise psycopg2.OperationalError("server closed the connection unexpectedly")
    except psycopg2.OperationalError as inner:
        wrapped = RuntimeError("store call failed")
        wrapped.__cause__ = inner
    r = make_client(wrapped).get("/umrs/jobs", headers=HEADERS)
    assert r.status_code == 503


def test_nul_character_is_a_typed_400(make_client) -> None:
    exc = ValueError("A string literal cannot contain NUL (0x00) characters.")
    r = make_client(exc).get("/umrs/jobs", headers=HEADERS)
    assert r.status_code == 400
    assert _typed(r)["error_id"] == "request.bad_request"


def test_any_other_exception_is_a_typed_500(make_client) -> None:
    r = make_client(KeyError("secret-internal-key")).get("/umrs/jobs", headers=HEADERS)
    assert r.status_code == 500
    body = _typed(r)
    assert body["error_id"] == "server.internal_error"
    assert "secret-internal-key" not in r.text
