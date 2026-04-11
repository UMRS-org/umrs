"""Route-level tests for UMRS active endpoint discovery."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


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

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry

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