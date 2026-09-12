"""WP-16 r2 — RFC 7807 problem+json projection pins (fail-without-change)."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from umrs.api.problem_json import (
    PROBLEM_MEDIA_TYPE,
    install_problem_json_handler,
    problem_body,
)

from r4_contracts.errors import ERROR_CODE_HEADER, PlatformError


class TestProblemBody:
    def test_rfc7807_field_shape(self):
        error = PlatformError(code="session.store_unavailable", message="store down", http_status=503)
        body = problem_body(error)
        assert body["type"] == "https://platform.dev/errors/session"
        assert body["title"] == "Session error"
        assert body["status"] == 503
        assert body["detail"] == "store down"
        assert body["code"] == "session.store_unavailable"

    def test_unprefixed_code_falls_back_platform(self):
        error = PlatformError(code="platform.internal", message="x", http_status=500)
        body = problem_body(error)
        assert body["type"].endswith("/platform")


class TestInstalledHandler:
    def _app(self):
        app = FastAPI()
        install_problem_json_handler(app)

        @app.get("/typed")
        def _typed():
            from umrs.api.typed_errors import raise_typed

            raise_typed(PlatformError.jobs("not_found", "Job not found: job-7"))

        @app.get("/legacy")
        def _legacy():
            raise HTTPException(status_code=404, detail="plain-keep")

        return app

    def test_typed_error_emits_problem_json(self):
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            response = client.get("/typed")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
        assert response.headers[ERROR_CODE_HEADER] == "jobs.not_found"
        body = response.json()
        assert body["type"] == "https://platform.dev/errors/jobs"
        assert body["title"] == "Jobs error"
        assert body["status"] == 404
        assert body["detail"] == "Job not found: job-7"
        assert body["code"] == "jobs.not_found"

    def test_legacy_shape_byte_identical(self):
        with TestClient(self._app(), raise_server_exceptions=False) as client:
            response = client.get("/legacy")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"detail": "plain-keep"}

    def test_env_flag_default_on_and_opt_out(self, monkeypatch):
        from umrs.api import problem_json as p

        monkeypatch.delenv("UMRS_PROBLEM_JSON", raising=False)
        assert p.enabled_by_env() is True
        monkeypatch.setenv("UMRS_PROBLEM_JSON", "0")
        assert p.enabled_by_env() is False
        monkeypatch.setenv("UMRS_PROBLEM_JSON", "1")
        assert p.enabled_by_env() is True


def test_env_default_is_on_with_opt_out(monkeypatch):
    from umrs.api import problem_json as p

    monkeypatch.delenv("UMRS_PROBLEM_JSON", raising=False)
    assert p.enabled_by_env() is True
    monkeypatch.setenv("UMRS_PROBLEM_JSON", "0")
    assert p.enabled_by_env() is False
    monkeypatch.setenv("UMRS_PROBLEM_JSON", "off")
    assert p.enabled_by_env() is False
    monkeypatch.setenv("UMRS_PROBLEM_JSON", "1")
    assert p.enabled_by_env() is True
