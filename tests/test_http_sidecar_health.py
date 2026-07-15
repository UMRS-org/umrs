"""Tests for the HTTP sidecar health and readiness endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sidecar_app():
    """Create a test instance of the HTTP sidecar app."""
    with patch.dict("os.environ", {"GATEWAY_ENVIRONMENT": "test"}, clear=False):
        from umrs.server.http_sidecar import _create_app

        app = _create_app()
    return app


@pytest.fixture
def client(sidecar_app):
    return TestClient(sidecar_app)


class TestHealthEndpoint:
    """Tests for GET /health (liveness probe)."""

    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "components" in data
        assert "uptime_seconds" in data


class TestReadinessEndpoint:
    """Tests for GET /readiness (readiness probe)."""

    def test_readiness_degraded_without_db(self, client):
        """Without DB pool configured, readiness returns 503."""
        response = client.get("/readiness")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert "components" in data

    def test_readiness_ok_with_stores(self, sidecar_app):
        """With job store, db pool, and durable jobs available, readiness is ok."""
        from umrs.api.jobs import set_job_store

        mock_store = MagicMock()
        mock_store.count_by_status.return_value = 0
        set_job_store(mock_store)

        # Also give the app a db_pool
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        sidecar_app.state.umrs_db_pool = mock_pool

        mock_svc = MagicMock()
        mock_svc._started = True

        try:
            with (
                patch("umrs.jobs.service.get_job_service", return_value=mock_svc),
                patch("umrs.server.durable_job_runtime.get_runtime", return_value=None),
            ):
                client = TestClient(sidecar_app)
                response = client.get("/readiness")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["components"]["postgres"]["status"] == "ok"
            assert data["components"]["job_store"]["status"] == "ok"
            assert data["components"]["durable_jobs"]["status"] == "ok"
        finally:
            # Clean up global state
            set_job_store(None)  # type: ignore[arg-type]
