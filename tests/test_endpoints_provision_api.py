"""Route-level tests for UMRS endpoint provisioning path resolution."""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus, get_registry
from umrs.orchestrator.service import ProvisionResult


class _StubOrchestrationService:
    def __init__(self):
        self.requests = []

    async def provision(self, request):
        self.requests.append(request)
        return ProvisionResult(success=True, job_id="job-123")


class _ReadyPoller:
    def __init__(self):
        self.calls = []

    async def check_ready_detail(
        self, host, port, *, probe_model=None, probe_kind="models", probe_path=None
    ):
        self.calls.append(
            {
                "host": host,
                "port": port,
                "probe_model": probe_model,
                "probe_kind": probe_kind,
                "probe_path": probe_path,
            }
        )
        return True, False, [probe_model], ""


def _new_registry():
    return EndpointRegistry()


def test_provision_endpoint_resolves_model_path_from_catalog(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwen3-8b",
            "huggingface_id": "Qwen/Qwen3-8B",
            "serving_profile": {"served_model_name": "Qwen/Qwen3-8B"},
            "model_type": "llm",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "qwen3-8b"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].model_path == "/shared/models/qwen3-8b"
    assert service.requests[0].wait_for_ready is True
    assert service.requests[0].metadata["readiness_probe"] == "chat"
    assert service.requests[0].metadata["owner_type"] == "user"
    assert service.requests[0].metadata["tier"] == "user"
    assert service.requests[0].metadata["served_model_name"] == "Qwen/Qwen3-8B"
    assert service.requests[0].env_vars["VLLM_SERVED_MODEL_NAME"] == "Qwen/Qwen3-8B"


def test_provision_endpoint_uses_model_readiness_for_embedding_model(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/e5-base-v2",
            "huggingface_id": "intfloat/e5-base-v2",
            "model_type": "embedding",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "e5-base-v2"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].metadata["readiness_probe"] == "models"


def test_provision_endpoint_uses_bentoml_defaults_for_vision_model(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/sam3",
            "huggingface_id": "facebook/sam3",
            "model_type": "vision",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "sam3"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    request = service.requests[0]
    assert request.metadata["model_type"] == "vision"
    assert request.metadata["serving_backend"] == "bentoml"
    assert request.metadata["service_kind"] == "bentoml_vision"
    assert request.metadata["readiness_probe"] == "http"
    assert request.metadata["readiness_path"] == "/readyz"
    assert request.metadata["launch_profiles"] == ["harness", "vision-bentoml"]
    assert request.env_vars["RIVAULT_EXTRA_PROFILES"] == "harness vision-bentoml"
    assert request.env_vars["VISION_REQUIRE_PRESIGNED_URLS"] == "true"
    assert request.env_vars["TRANSFORMERS_OFFLINE"] == "1"


def test_provision_endpoint_can_co_locate_vllm_for_vision_model(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/sam3",
            "model_type": "vision",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "sam3", "co_locate_vllm": True},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].metadata["launch_profiles"] == [
        "harness",
        "vllm",
        "vision-bentoml",
    ]


def test_provision_endpoint_preserves_site_litellm_db_profile(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwopus3.5-9b",
            "model_type": "llm",
            "serving_profile": {"launch_profiles": ["harness", "vllm"]},
        },
    )
    monkeypatch.setattr(
        endpoints_api,
        "_active_site_profiles",
        lambda: ["harness", "vllm", "vision-bentoml", "litellm-db"],
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "qwopus3.5-9b"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].metadata["launch_profiles"] == [
        "harness",
        "vllm",
        "litellm-db",
    ]
    assert service.requests[0].env_vars["RIVAULT_EXTRA_PROFILES"] == "harness vllm litellm-db"


def test_provision_endpoint_llm_without_profiles_excludes_vision_bentoml(monkeypatch):
    """A bare LLM model (no explicit launch_profiles) must launch the vLLM
    stack only — never the BentoML vision sidecar.

    Regression: previously LLM models resolved to an empty profile list, so
    RIVAULT_PROFILES was left unset and the Slurm wrapper fell back to a
    site-wide default that also started ``vision-bentoml`` — launching the
    wrong service stack for a text model and starving vLLM of GPU memory.
    """
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwen3-30b-a3b",
            "huggingface_id": "Qwen/Qwen3-30B-A3B",
            "model_type": "llm",
        },
    )
    monkeypatch.setattr(
        endpoints_api,
        "_active_site_profiles",
        lambda: ["harness", "vllm", "vision-bentoml", "litellm-db"],
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "Qwen3-30B-A3B"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    request = service.requests[0]
    assert request.metadata["model_type"] == "llm"
    assert request.metadata["serving_backend"] == "vllm"
    assert request.metadata["launch_profiles"] == ["harness", "vllm", "litellm-db"]
    assert request.env_vars["RIVAULT_EXTRA_PROFILES"] == "harness vllm litellm-db"
    assert "vision-bentoml" not in request.env_vars["RIVAULT_EXTRA_PROFILES"]


def test_provision_endpoint_uses_gateway_project_header(monkeypatch):
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
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            headers={"X-Project-ID": "project-from-gateway"},
            json={"model": "qwen3-8b", "models": ["qwen3-8b"], "serving_mode": "single"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].project_id == "project-from-gateway"


def test_provision_endpoint_strict_mode_requires_verified_project_header(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "t-secret")
    from r4_contracts.handshake_hmac import sign_claims

    signed_hmac = sign_claims("t-secret", "gateway-user", "verified-project", False)
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
    app.dependency_overrides[get_registry] = _new_registry
    app.dependency_overrides[endpoints_api._get_requesting_user] = lambda: None

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            headers={"X-Project-ID": "untrusted-project"},
            json={"project_id": "body-project", "model": "qwen3-8b"},
        )
        verified_response = client.post(
            "/umrs/endpoints/provision",
            headers={
                "X-User-ID": "gateway-user",
                "X-Verified-Project-ID": "verified-project",
                "X-Verified-HMAC": signed_hmac,
            },
            json={"project_id": "body-project", "model": "qwen3-8b"},
        )
        spoofed_verified_response = client.post(
            "/umrs/endpoints/provision",
            headers={"X-Verified-Project-ID": "verified-project"},
            json={"project_id": "body-project", "model": "qwen3-8b"},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 401
    assert verified_response.status_code == 202
    assert spoofed_verified_response.status_code == 401
    assert service.requests[0].project_id == "verified-project"


def test_strict_mode_verified_project_header_allows_projectless_oidc_token(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "t-secret")
    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.setattr(
        endpoints_api,
        "_jwt_claims",
        lambda authorization, required=False: {"sub": "241723", "roles": ["user"]},
    )

    project_id = endpoints_api._resolve_authenticated_project_id(
        "Bearer oidc-token",
        "241723",
        "verified-project",
        None,
        x_verified_is_admin=None,
        x_verified_hmac=sign_claims("t-secret", "241723", "verified-project", False),
    )

    assert project_id == "verified-project"


def test_provision_endpoint_forwards_wait_false(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: {
            "storage_ref": "/shared/models/qwen3-8b",
            "huggingface_id": "Qwen/Qwen3-8B",
            "serving_profile": {"served_model_name": "Qwen/Qwen3-8B"},
            "model_type": "llm",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "qwen3-8b", "wait": False},
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    assert service.requests[0].wait_for_ready is False


def test_provision_endpoint_forwards_mounts_to_service(monkeypatch):
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

    monkeypatch.setenv("EGRESS_GUARD_IMAGE", "localhost/r4-mcp:test")  # site declares a proxy

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    # H-1 (2026-10-18): a non-admin's mount sources must lie under the caller's
    # own user/project roots (they were checked only against the site-wide
    # prefixes, so /project/data of anyone passed); the caller is alice in
    # project-a and the sources are hers.
    monkeypatch.delenv("UMRS_ENDPOINT_CALLER_MOUNT_ROOTS", raising=False)
    monkeypatch.setattr(endpoints_api, "_check_requesting_user_gate", lambda _user: None)
    body = {
        "project_id": "project-a",
        "model": "qwen3-8b",
        "tier": "project",
        "mounts": {
            "mcp_servers": [{"source": "/home/alice/tools", "target": "/mcp-servers/tools"}],
            "project_data": {"source": "/projects/project-a/data", "target": "/project"},
            "rag_index": {"source": "/projects/project-a/rag-index", "target": "/rag-index"},
        },
        "egress_domains": ["api.example.com"],
    }
    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision", json=body, headers={"X-User-ID": "alice"}
        )

    app.dependency_overrides.clear()

    assert response.status_code == 202
    request = service.requests[0]
    assert request.tier == "project"
    assert request.mounts["mcp_servers"][0]["target"] == "/mcp-servers/tools"
    assert request.mounts["mcp_servers"][0]["source"].endswith("/home/alice/tools")
    assert request.mounts["project_data"] == body["mounts"]["project_data"]
    assert request.mounts["rag_index"] == body["mounts"]["rag_index"]
    assert request.egress_domains == ["api.example.com"]
    assert request.metadata["tier"] == "project"


def test_provision_endpoint_rejects_unresolved_model_without_model_path(monkeypatch):
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda model_name: None,
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

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


def test_get_endpoint_refreshes_starting_endpoint_when_ready(monkeypatch):
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="qwen3-8b",
        job_id="job-123",
        host="ng-dgx-m0-08",
        port=8001,
        backend="slurm",
        metadata={"readiness_probe": "chat"},
    )
    registry.update_status("job-123", EndpointStatus.STARTING)

    poller = _ReadyPoller()
    monkeypatch.setattr(endpoints_api, "get_readiness_poller", lambda: poller)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry

    with TestClient(app) as client:
        response = client.get("/umrs/endpoints/job-123")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["detail"] == "endpoint ready"
    assert registry.get_by_job_id("job-123").status == EndpointStatus.READY
    assert poller.calls == [
        {
            "host": "ng-dgx-m0-08",
            "port": 8001,
            "probe_model": "qwen3-8b",
            "probe_kind": "chat",
            "probe_path": None,
        }
    ]


def test_provision_endpoint_refuses_egress_without_declared_proxy_image(monkeypatch):
    """ADR-040 D4: egress_domains / custom MCP mounts need a site-declared proxy image."""
    from umrs.orchestrator.site_config import SiteConfig

    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda _model_name: {
            "storage_ref": "/shared/models/qwen3-8b",
            "huggingface_id": "Qwen/Qwen3-8B",
            "model_type": "llm",
        },
    )
    monkeypatch.setattr(
        "umrs.orchestrator.service.get_orchestration_service",
        lambda: service,
    )
    monkeypatch.delenv("EGRESS_GUARD_IMAGE", raising=False)
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.get_site_config", lambda _path=None: SiteConfig()
    )

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = _new_registry

    body = {
        "project_id": "project-a",
        "model": "qwen3-8b",
        "tier": "project",
        "egress_domains": ["api.example.com"],
    }
    with TestClient(app) as client:
        response = client.post("/umrs/endpoints/provision", json=body)

    app.dependency_overrides.clear()

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "refusing per ADR-040 D4" in detail
    assert "egress_domains=api.example.com" in detail
    assert service.requests == []  # refused before any provisioning work
