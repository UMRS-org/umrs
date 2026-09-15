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
        assert response.json() == {"status": "ok"}


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
        """With job store available, readiness should report ok."""
        from umrs.api.jobs import set_job_store

        mock_store = MagicMock()
        set_job_store(mock_store)

        # Also give the app a db_pool
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        sidecar_app.state.umrs_db_pool = mock_pool

        try:
            client = TestClient(sidecar_app)
            response = client.get("/readiness")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert data["components"]["database"] == "ok"
            assert data["components"]["job_store"] == "ok"
        finally:
            # Clean up global state
            set_job_store(None)  # type: ignore[arg-type]
