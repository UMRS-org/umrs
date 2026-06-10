"""Security regression tests for UMRS endpoint lifecycle routes."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry
from umrs.orchestrator.service import ProvisionResult


class _StubOrchestrationService:
    def __init__(self):
        self.requests = []
        self.deprovisions = []

    async def provision(self, request):
        self.requests.append(request)
        return ProvisionResult(success=True, job_id="job-123")

    async def deprovision(self, job_id, *, force=False):
        self.deprovisions.append((job_id, force))
        return True


def _new_registry():
    return EndpointRegistry()


def _app_with_registry(registry=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = (lambda: registry) if registry else _new_registry
    return app


def _patch_catalog(monkeypatch):
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwen3-8b",
            "huggingface_id": "Qwen/Qwen3-8B",
            "model_type": "llm",
        },
    )


def test_provision_endpoint_rejects_system_tier_for_non_admin(monkeypatch):
    service = _StubOrchestrationService()
    _patch_catalog(monkeypatch)
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = _app_with_registry()
    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            headers={"X-Project-ID": "project-a"},
            json={"model": "qwen3-8b", "tier": "system"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 403
    assert service.requests == []


def test_provision_endpoint_rejects_denied_mount_source(monkeypatch):
    service = _StubOrchestrationService()
    _patch_catalog(monkeypatch)
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = _app_with_registry()
    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={
                "project_id": "project-a",
                "model": "qwen3-8b",
                "mounts": {
                    "project_data": {"source": "/etc/passwd", "target": "/project"},
                },
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 422
    assert service.requests == []


def test_provision_endpoint_rejects_denied_model_path(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = _app_with_registry()
    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={
                "project_id": "project-a",
                "model": "qwen3-8b",
                "model_path": "/etc/passwd",
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "Invalid model_path" in response.json()["detail"]
    assert service.requests == []


def test_get_endpoint_hides_cross_project_job_id():
    registry = EndpointRegistry()
    registry.register(
        project_id="victim-project",
        model="qwen3-8b",
        job_id="job-123",
        host="pending",
        port=11434,
        backend="slurm",
    )

    app = _app_with_registry(registry)
    with TestClient(app) as client:
        attacker_response = client.get(
            "/umrs/endpoints/job-123",
            headers={"X-Project-ID": "attacker-project"},
        )
        owner_response = client.get(
            "/umrs/endpoints/job-123",
            headers={"X-Project-ID": "victim-project"},
        )

    app.dependency_overrides.clear()

    assert attacker_response.status_code == 404
    assert owner_response.status_code == 200
    assert owner_response.json()["project_id"] == "victim-project"


def test_deprovision_endpoint_hides_cross_project_job_id(monkeypatch):
    service = _StubOrchestrationService()
    registry = EndpointRegistry()
    registry.register(
        project_id="victim-project",
        model="qwen3-8b",
        job_id="job-123",
        host="pending",
        port=11434,
        backend="slurm",
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = _app_with_registry(registry)
    with TestClient(app) as client:
        attacker_response = client.delete(
            "/umrs/endpoints/job-123/deprovision",
            headers={"X-Project-ID": "attacker-project"},
        )
        owner_response = client.delete(
            "/umrs/endpoints/job-123/deprovision",
            headers={"X-Project-ID": "victim-project"},
        )

    app.dependency_overrides.clear()

    assert attacker_response.status_code == 404
    assert owner_response.status_code == 204
    assert service.deprovisions == [("job-123", False)]


def test_deprovision_returns_202_when_backend_still_stopping(monkeypatch):
    class _StoppingService:
        async def deprovision(self, job_id, *, force=False):
            return False

    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-777",
        host="pending",
        port=11434,
        backend="slurm",
    )
    registry.update_metadata("job-777", {"cleanup_status": "running"})

    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: _StoppingService(),
    )

    app = _app_with_registry(registry)
    with TestClient(app) as client:
        response = client.delete(
            "/umrs/endpoints/job-777/deprovision?force=true",
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "stopping"
    assert body["job_id"] == "job-777"


def test_deprovision_returns_500_on_unexpected_failure(monkeypatch):
    class _FailingService:
        async def deprovision(self, job_id, *, force=False):
            return False

    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-888",
        host="pending",
        port=11434,
        backend="slurm",
    )

    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: _FailingService(),
    )

    app = _app_with_registry(registry)
    with TestClient(app) as client:
        response = client.delete(
            "/umrs/endpoints/job-888/deprovision?force=true",
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 500