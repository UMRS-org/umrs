"""Route-level tests for UMRS endpoint provisioning path resolution."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry
from umrs.orchestrator.service import ProvisionResult


class _StubOrchestrationService:
    def __init__(self):
        self.requests = []

    async def provision(self, request):
        self.requests.append(request)
        return ProvisionResult(success=True, job_id="job-123")


def test_provision_endpoint_resolves_model_path_from_catalog(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwen3-8b",
            "huggingface_id": "Qwen/Qwen3-8B",
            "model_type": "llm",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: EndpointRegistry()

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "qwen3-8b"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].model_path == "/shared/models/qwen3-8b"
    assert service.requests[0].metadata["readiness_probe"] == "chat"
    assert service.requests[0].metadata["served_model_name"] == "Qwen/Qwen3-8B"


def test_provision_endpoint_rejects_unresolved_model_without_model_path(monkeypatch):
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: None,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: EndpointRegistry()

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "unknown-model"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "Unable to resolve model path" in response.json()["detail"]


def test_get_endpoint_exposes_progress_detail_from_registry_metadata():
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-123",
        host="pending",
        port=11434,
        backend="slurm",
        metadata={
            "detail": "starting container / loading model",
            "job_status": "starting",
        },
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.get("/umrs/endpoints/job-123")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["detail"] == "starting container / loading model"
    assert data["job_status"] == "starting"