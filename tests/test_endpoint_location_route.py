"""Tests for admin PUT /umrs/endpoints/{job_id}/location route."""

import os
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import endpoints as endpoints_module
from umrs.api.endpoints import router, get_registry


class TestLocationRoute:
    """Tests for the admin location update route."""

    @pytest.fixture
    def app(self, monkeypatch):
        """Create test app with registry override."""
        from umrs.orchestrator.endpoint_registry import EndpointRegistry

        registry = EndpointRegistry()
        registry.register(
            project_id="test-project",
            model="test-model",
            job_id="job-loc",
            host="compute-01",
            port=11434,
            backend="slurm",
        )

        def _get_registry() -> EndpointRegistry:
            return registry

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_registry] = _get_registry
        return app, registry

    def test_location_updates_host_port(self, app, monkeypatch):
        """PUT /location should update host, port, and metadata."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        test_app, registry = app
        with TestClient(test_app) as client:
            response = client.put(
                "/umrs/endpoints/job-loc/location",
                json={"host": "compute-02", "port": 11435, "metadata": {"updated": "true"}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["host"] == "compute-02"
        assert data["port"] == 11435
        assert data["status"] == "provisioning"
        assert data["metadata"]["updated"] == "true"

        # Verify persistence in registry
        ep = registry.get_by_job_id("job-loc")
        assert ep.host == "compute-02"
        assert ep.port == 11435

    def test_location_partial_update(self, app, monkeypatch):
        """PUT /location should only update provided fields."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        test_app, registry = app
        with TestClient(test_app) as client:
            response = client.put(
                "/umrs/endpoints/job-loc/location",
                json={"host": "compute-03"},
            )

        assert response.status_code == 200
        assert response.json()["host"] == "compute-03"
        assert response.json()["port"] == 11434

    def test_location_not_found(self, app, monkeypatch):
        """PUT /location for non-existent job should return 404."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        test_app, registry = app
        with TestClient(test_app) as client:
            response = client.put(
                "/umrs/endpoints/nonexistent/location",
                json={"host": "compute-01"},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_location_requires_strict_auth(self, app, monkeypatch):
        """Route should return 403 when UMRS_AUTH_MODE != strict."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        test_app, registry = app
        with TestClient(test_app) as client:
            response = client.put(
                "/umrs/endpoints/job-loc/location",
                json={"host": "compute-02"},
            )

        assert response.status_code == 403
        assert "strict" in response.json()["detail"].lower()
