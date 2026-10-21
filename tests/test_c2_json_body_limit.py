"""Cycle-2 umrs-api lane (2026-10-21): JSON request bodies have a size cap.

Found live through the gateway: the JSON API locations inherit nginx's
unlimited ``client_max_body_size`` and FastAPI parses a JSON body whole; one
authenticated 300 MB ``POST /gw/jobs`` took the UMRS process from 238 MB to
a 1.16 GB peak RSS. The real app now answers a typed 413 above
``UMRS_MAX_JSON_BODY_BYTES`` (default 64 MiB); uploads (multipart,
octet-stream) are not JSON and keep their own limits.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from umrs.server.body_limit import JsonBodyLimitMiddleware, max_json_body_bytes

LIMIT = 1024


@pytest.fixture
def sidecar(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from umrs.jobs.service import get_job_service, set_job_service

    monkeypatch.setenv("UMRS_MAX_JSON_BODY_BYTES", str(LIMIT))
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("GATEWAY_ENVIRONMENT", "test")
    prior = get_job_service()
    set_job_service(None)
    with patch("umrs.server.http_sidecar._build_db_pool", lambda _app: None):
        from umrs.server import http_sidecar

        app = http_sidecar._create_app()
    yield TestClient(app)
    set_job_service(prior)


def test_declared_oversize_json_is_413_in_the_real_app(sidecar: TestClient) -> None:
    r = sidecar.post(
        "/umrs/jobs",
        content=b'{"x":"' + b"A" * (LIMIT * 4) + b'"}',
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 413
    assert r.json()["error_id"] == "request.too_large"


def test_json_under_the_cap_reaches_the_route(sidecar: TestClient) -> None:
    r = sidecar.post("/umrs/jobs", json={"x": "small"})
    assert r.status_code != 413  # the route's own answer (here: no identity)


def test_uploads_are_not_capped(sidecar: TestClient) -> None:
    r = sidecar.post(
        "/umrs/files/upload",
        files={"file": ("big.bin", b"B" * (LIMIT * 8), "application/octet-stream")},
    )
    assert r.status_code != 413


def _echo_app() -> FastAPI:
    app = FastAPI()

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        return {"n": len(await request.body())}

    app.add_middleware(JsonBodyLimitMiddleware)
    return app


def test_streamed_oversize_json_is_413(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Content-Length (chunked): the cap applies while the body streams."""
    monkeypatch.setenv("UMRS_MAX_JSON_BODY_BYTES", str(LIMIT))

    def chunks():
        for _ in range(8):
            yield b"A" * LIMIT

    r = TestClient(_echo_app()).post(
        "/echo", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 413


def test_default_cap_is_64_mib(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_MAX_JSON_BODY_BYTES", raising=False)
    assert max_json_body_bytes() == 64 * 1024 * 1024
    monkeypatch.setenv("UMRS_MAX_JSON_BODY_BYTES", "garbage")
    assert max_json_body_bytes() == 64 * 1024 * 1024
