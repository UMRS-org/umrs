"""Route-level tests for UMRS active endpoint discovery."""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import endpoints as endpoints_module
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry
from umrs.orchestrator import service as service_module


def _register_ready_endpoint(
    registry: EndpointRegistry,
    *,
    project_id: str,
    model: str,
    job_id: str,
    served_model_name: str | None = None,
) -> None:
    metadata = {}
    if served_model_name:
        metadata["served_model_name"] = served_model_name

    registry.register(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host="node-a",
        port=8001,
        backend="slurm-podman",
        metadata=metadata,
    )
    registry.mark_ready(job_id)


def test_active_endpoints_wildcard_returns_all_ready_endpoints():
    registry = EndpointRegistry()
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-1",
        served_model_name="Qwen/Qwen3-8B",
    )
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-30b-a3b",
        job_id="job-2",
    )
    _register_ready_endpoint(
        registry,
        project_id="project-b",
        model="other-model",
        job_id="job-3",
    )

    def _get_registry() -> EndpointRegistry:
        return registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _get_registry

    with TestClient(app) as client:
        response = client.get(
            "/umrs/endpoints/active",
            params={"model": "*"},
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert {endpoint["model"] for endpoint in payload["endpoints"]} == {
        "qwen3-8b",
        "qwen3-30b-a3b",
    }


def test_active_endpoints_default_legacy_records_to_user_tier():
    registry = EndpointRegistry()
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-1",
    )

    def _get_registry() -> EndpointRegistry:
        return registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _get_registry

    with TestClient(app) as client:
        response = client.get(
            "/umrs/endpoints/active",
            params={"model": "*"},
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["endpoints"][0]["owner_type"] == "user"
    assert payload["endpoints"][0]["tier"] == "user"


def test_active_endpoints_served_model_name_lookup_still_matches_single_endpoint():
    registry = EndpointRegistry()
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-1",
        served_model_name="Qwen/Qwen3-8B",
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.get(
            "/umrs/endpoints/active",
            params={"model": "Qwen/Qwen3-8B"},
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["endpoints"][0]["job_id"] == "job-1"


def test_health_route_is_not_shadowed_by_job_id_route():
    registry = EndpointRegistry()

    def _get_registry() -> EndpointRegistry:
        return registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _get_registry

    with TestClient(app) as client:
        response = client.get("/umrs/endpoints/health")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["registry_ok"] is True


def test_active_endpoints_prune_stale_registry_ghosts(monkeypatch):
    registry = EndpointRegistry()
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-live",
    )
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="bonsai-8b",
        job_id="job-ghost",
    )

    async def _fake_backend_job_liveness(service, job_id: str) -> bool | None:
        return job_id == "job-live"

    monkeypatch.setattr(
        endpoints_module,
        "_backend_job_liveness",
        _fake_backend_job_liveness,
    )
    def _get_service():
        return SimpleNamespace()

    monkeypatch.setattr(
        service_module,
        "get_orchestration_service",
        _get_service,
    )

    def _get_registry() -> EndpointRegistry:
        return registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _get_registry

    with TestClient(app) as client:
        response = client.get(
            "/umrs/endpoints/active",
            params={"model": "*"},
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["endpoints"][0]["job_id"] == "job-live"
    assert registry.get_by_job_id("job-ghost") is None


def test_active_endpoints_specific_model_drops_stale_ready_entry(monkeypatch):
    registry = EndpointRegistry()
    _register_ready_endpoint(
        registry,
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-ghost",
    )

    async def _fake_backend_job_liveness(service, job_id: str) -> bool | None:
        return False

    monkeypatch.setattr(
        endpoints_module,
        "_backend_job_liveness",
        _fake_backend_job_liveness,
    )
    def _get_service():
        return SimpleNamespace()

    monkeypatch.setattr(
        service_module,
        "get_orchestration_service",
        _get_service,
    )

    def _get_registry() -> EndpointRegistry:
        return registry

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _get_registry

    with TestClient(app) as client:
        response = client.get(
            "/umrs/endpoints/active",
            params={"model": "qwen3-8b"},
            headers={"X-Project-ID": "project-a"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 0
    assert payload["endpoints"] == []
    assert registry.get_by_job_id("job-ghost") is None