"""Tests for the capability/model_type filter on GET /umrs/endpoints/active.

The filter consults the registered model entry's ``capabilities`` block and
``model_type`` to keep only matching endpoints. Unknown capability tags
filter everything out (empty list, status 200) — same behaviour as
``model=`` matching nothing.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import endpoints as endpoints_mod
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _register(registry: EndpointRegistry, *, project_id: str, model: str, job_id: str) -> None:
    registry.register(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host="node-a",
        port=8001,
        backend="slurm-podman",
        metadata={},
    )
    registry.mark_ready(job_id)


def _patch_registry(monkeypatch, entries: dict[str, dict[str, Any]]):
    """Stub _resolve_registered_model_entry to return canned metadata."""

    def fake_resolve(model_name: str) -> dict[str, Any] | None:
        return entries.get(model_name)

    monkeypatch.setattr(endpoints_mod, "_resolve_registered_model_entry", fake_resolve)


def _build_client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def test_capabilities_filter_keeps_only_vision_capable(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="vlm-1", job_id="j-vlm")
    _register(registry, project_id="p", model="text-1", job_id="j-text")

    _patch_registry(
        monkeypatch,
        {
            "vlm-1": {
                "name": "vlm-1",
                "model_type": "vision",
                "capabilities": {
                    "supports_vision": True,
                    "supports_tool_calling": False,
                },
            },
            "text-1": {
                "name": "text-1",
                "model_type": "llm",
                "capabilities": {
                    "supports_vision": False,
                    "supports_tool_calling": True,
                },
            },
        },
    )

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "*", "capabilities": "vision"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["endpoints"][0]["model"] == "vlm-1"


def test_model_type_filter_keeps_only_llm(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="vlm-1", job_id="j-vlm")
    _register(registry, project_id="p", model="text-1", job_id="j-text")
    _patch_registry(
        monkeypatch,
        {
            "vlm-1": {"name": "vlm-1", "model_type": "vision", "capabilities": {}},
            "text-1": {"name": "text-1", "model_type": "llm", "capabilities": {}},
        },
    )

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "*", "model_type": "llm"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert {ep["model"] for ep in body["endpoints"]} == {"text-1"}


def test_unknown_capability_tag_returns_empty_with_200(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="vlm-1", job_id="j-vlm")
    _patch_registry(
        monkeypatch,
        {
            "vlm-1": {
                "name": "vlm-1",
                "model_type": "vision",
                "capabilities": {"supports_vision": True},
            },
        },
    )

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "*", "capabilities": "fancy_new_thing"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0


def test_no_filter_unchanged_behaviour(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="vlm-1", job_id="j-vlm")
    _register(registry, project_id="p", model="text-1", job_id="j-text")
    _patch_registry(monkeypatch, {})

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "*"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2


def test_gpu_tag_is_always_satisfied(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="vlm-1", job_id="j-vlm")
    _patch_registry(
        monkeypatch,
        {
            "vlm-1": {
                "name": "vlm-1",
                "model_type": "vision",
                "capabilities": {"supports_vision": True},
            },
        },
    )

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "*", "capabilities": "gpu,vision"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1


def test_specific_model_filtered_out_by_capabilities(monkeypatch):
    registry = EndpointRegistry()
    _register(registry, project_id="p", model="text-1", job_id="j-text")
    _patch_registry(
        monkeypatch,
        {
            "text-1": {
                "name": "text-1",
                "model_type": "llm",
                "capabilities": {"supports_vision": False},
            },
        },
    )

    with _build_client(registry) as client:
        r = client.get(
            "/umrs/endpoints/active",
            params={"model": "text-1", "capabilities": "vision"},
            headers={"X-Project-ID": "p"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
