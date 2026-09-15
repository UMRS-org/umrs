from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import umrs.api.vision as vision_api
from umrs.api.vision import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus


class _S3Config:
    bucket = "umrs-files"
    default_ttl = 3600
    max_ttl = 86400

    def get_client(self):
        return object()


def _registry_with_vision_endpoint() -> EndpointRegistry:
    registry = EndpointRegistry()
    endpoint = registry.register(
        project_id="project-a",
        model="sam3",
        job_id="job-vision",
        host="node-a",
        port=3001,
        backend="slurm",
        metadata={
            "model_type": "vision",
            "serving_backend": "bentoml",
            "service_kind": "bentoml_vision",
            "service_url": "http://node-a:3001",
            "vision_adapter": "sam3",
        },
    )
    endpoint.status = EndpointStatus.READY
    return registry


def test_submit_inference_generates_internal_presigned_urls(monkeypatch):
    registry = _registry_with_vision_endpoint()
    seen = {}

    async def fake_request_endpoint(endpoint, method, path, *, json_payload=None, params=None):
        seen["endpoint"] = endpoint
        seen["method"] = method
        seen["path"] = path
        seen["payload"] = json_payload
        return httpx.Response(200, json={"task_id": "task-123", "status": "submitted"})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request_endpoint)
    monkeypatch.setattr(vision_api, "generate_presigned_get", lambda *args, **kwargs: "get-url")
    monkeypatch.setattr(
        vision_api,
        "generate_presigned_put",
        lambda _client, _bucket, key, **kwargs: f"put-url:{key}",
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[vision_api.get_registry] = lambda: registry
    app.dependency_overrides[vision_api.get_s3_config] = _S3Config

    with TestClient(app) as client:
        response = client.post(
            "/umrs/vision/infer/submit",
            headers={"X-Project-ID": "project-a", "X-Verified-Project-ID": "project-a"},
            json={
                "model": "sam3",
                "image_uri": "s3://umrs-files/project-a/images/cell.png",
                "inference_config": {"prompt": "cell"},
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["endpoint_id"] == "job-vision"
    assert data["adapter"] == "sam3"
    assert data["task"]["task_id"] == "task-123"
    assert {item["key"] for item in data["outputs"]} == {
        "project-a/images/cell_sam3_seg.png",
        "project-a/images/cell_sam3_results.pkl",
    }
    assert seen["method"] == "POST"
    assert seen["path"] == "/infer/submit"
    assert seen["payload"]["presigned_input_url"] == "get-url"
    assert seen["payload"]["presigned_output_urls"] == {
        "project-a/images/cell_sam3_seg.png": "put-url:project-a/images/cell_sam3_seg.png",
        "project-a/images/cell_sam3_results.pkl": "put-url:project-a/images/cell_sam3_results.pkl",
    }


def test_submit_inference_requires_ready_vision_endpoint(monkeypatch):
    registry = EndpointRegistry()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[vision_api.get_registry] = lambda: registry
    app.dependency_overrides[vision_api.get_s3_config] = _S3Config

    with TestClient(app) as client:
        response = client.post(
            "/umrs/vision/infer/submit",
            headers={"X-Project-ID": "project-a", "X-Verified-Project-ID": "project-a"},
            json={"model": "sam3", "image_uri": "project-a/images/cell.png"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 404
    assert "No ready vision endpoint" in response.json()["detail"]
