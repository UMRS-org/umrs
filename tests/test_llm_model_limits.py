"""W5/U3+U4 (2026-10-05, gap row D-11): model limits in ``/v1/models``.

Every model the LLM proxy lists carries ``context_window``,
``max_output_tokens``, ``reasoning`` and ``supports_reasoning_effort`` — the
value when the platform knows it, null when it does not — so R4 sizes its
output budget from the platform instead of a name table. Existing fields
(``max_model_len`` included) are unchanged. The values come from the
endpoint's registry metadata, recorded at provision time from the catalog
entry (``endpoint_limit_metadata``); the serving profile gains an optional
``max_output_tokens`` field (U4).
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from umrs.api import endpoints as endpoints_api
from umrs.api import llm as llm_api
from umrs.api.models import ServingProfileRequest, serving_profile_from_request
from umrs.model_registry.capabilities import endpoint_limit_metadata
from umrs.model_registry.models import ServingProfile
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry
from umrs.orchestrator.service import ProvisionResult

_HEADERS = {"x-user-id": "user-1", "x-project-id": "test-project"}
_LIMIT_KEYS = {
    "context_window",
    "max_output_tokens",
    "reasoning",
    "supports_reasoning_effort",
}


def _registry(metadata: dict) -> EndpointRegistry:
    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="glm-5.3",
        job_id="job-1",
        host="compute-01",
        port=11434,
        backend="slurm",
        metadata={"served_model_name": "zai-org/GLM-5.3", **metadata},
    )
    registry.mark_ready("job-1")
    return registry


def _list(registry: EndpointRegistry, upstream: list[dict] | Exception) -> list[dict]:
    app = FastAPI()
    app.include_router(llm_api.router)
    app.dependency_overrides[get_registry] = lambda: registry

    async def fake_get(self, url, **kwargs):
        if isinstance(upstream, Exception):
            raise upstream
        return httpx.Response(
            200,
            json={"object": "list", "data": upstream},
            request=httpx.Request("GET", str(url)),
        )

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = TestClient(app).get("/umrs/llm/v1/models", headers=_HEADERS)
    assert response.status_code == 200
    return response.json()["data"]


# ---------------------------------------------------------------------------
# /v1/models listing
# ---------------------------------------------------------------------------


_GLM_UPSTREAM = {"id": "zai-org/GLM-5.3", "object": "model", "owned_by": "vllm"}
_GLM_KNOWN = {
    "max_model_len": 202752,
    "max_output_tokens": 131072,
    "reasoning": True,
    "supports_reasoning_effort": True,
}


def test_listing_publishes_known_limits():
    [entry] = _list(_registry(_GLM_KNOWN), [dict(_GLM_UPSTREAM)])

    assert {key: entry[key] for key in _LIMIT_KEYS} == {
        "context_window": 202752,
        "max_output_tokens": 131072,
        "reasoning": True,
        "supports_reasoning_effort": True,
    }


def test_listing_keeps_existing_fields_unchanged():
    [entry] = _list(_registry(_GLM_KNOWN), [dict(_GLM_UPSTREAM)])

    # id is the catalog-name rewrite exactly as before W5.
    assert entry["id"] == "glm-5.3"
    assert entry["catalog_model"] == "glm-5.3"
    assert entry["object"] == "model"
    assert entry["owned_by"] == "vllm"
    assert "max_model_len" not in entry  # not invented when upstream omits it


def test_context_window_prefers_the_served_max_model_len_and_keeps_it():
    """vLLM reports what it actually serves; that beats the requested value."""
    upstream = {"id": "zai-org/GLM-5.3", "object": "model", "max_model_len": 65536}

    [entry] = _list(_registry({"max_model_len": 202752}), [upstream])

    assert entry["context_window"] == 65536
    assert entry["max_model_len"] == 65536


def test_unknown_limits_are_listed_as_null():
    [entry] = _list(_registry({}), [{"id": "zai-org/GLM-5.3", "object": "model"}])

    assert set(entry) >= _LIMIT_KEYS
    assert all(entry[key] is None for key in _LIMIT_KEYS)


def test_malformed_metadata_values_are_unknown_not_coerced():
    registry = _registry(
        {
            "max_model_len": "lots",
            "max_output_tokens": True,
            "reasoning": "yes",
            "supports_reasoning_effort": 1,
        }
    )

    [entry] = _list(registry, [{"id": "zai-org/GLM-5.3"}])

    assert all(entry[key] is None for key in _LIMIT_KEYS)


def test_a_value_the_upstream_already_sent_wins():
    upstream = {"id": "zai-org/GLM-5.3", "max_output_tokens": 8192, "reasoning": False}

    [entry] = _list(_registry({"max_output_tokens": 131072, "reasoning": True}), [upstream])

    assert entry["max_output_tokens"] == 8192
    assert entry["reasoning"] is False


def test_synthesized_entry_carries_the_limits_when_the_fetch_fails():
    error = httpx.ConnectError("down", request=httpx.Request("GET", "http://x"))

    [entry] = _list(_registry({"max_model_len": 202752, "reasoning": True}), error)

    assert entry["id"] == "glm-5.3"
    assert entry["context_window"] == 202752
    assert entry["reasoning"] is True
    assert entry["max_output_tokens"] is None


# ---------------------------------------------------------------------------
# endpoint_limit_metadata: catalog entry → registry metadata
# ---------------------------------------------------------------------------


def test_no_catalog_entry_records_nothing():
    assert endpoint_limit_metadata(None) == {}
    assert endpoint_limit_metadata({}) == {}


def test_non_mapping_entry_records_nothing():
    class _NotAMapping:
        def get(self, _key, _default=None):
            return 4096

    assert endpoint_limit_metadata(_NotAMapping()) == {}  # type: ignore[arg-type]


def test_serving_profile_max_output_tokens_wins_over_generation_config():
    entry = {
        "name": "qwen3-8b",
        "serving_profile": {"max_output_tokens": 16384},
        "capabilities": {"max_output_tokens": 8192},
    }
    assert endpoint_limit_metadata(entry)["max_output_tokens"] == 16384


def test_generation_config_max_output_tokens_is_the_fallback():
    entry = {"name": "qwen3-8b", "capabilities": {"max_output_tokens": 8192}}
    assert endpoint_limit_metadata(entry)["max_output_tokens"] == 8192


@pytest.mark.parametrize("junk", [0, -1, True, "abc", None])
def test_junk_max_output_tokens_is_unknown(junk):
    entry = {"name": "qwen3-8b", "serving_profile": {"max_output_tokens": junk}}
    assert "max_output_tokens" not in endpoint_limit_metadata(entry)


def test_reasoning_model_derived_from_the_registry_name_table():
    """glm-* carries reasoning-effort levels in the registry table, so it is
    a reasoning model that honours reasoning_effort even with no stored caps."""
    flags = endpoint_limit_metadata({"name": "glm-5.3", "capabilities": {}})
    assert flags == {"reasoning": True, "supports_reasoning_effort": True}


def test_undeclared_non_reasoning_entry_publishes_no_guess():
    assert endpoint_limit_metadata({"name": "qwen3-8b", "capabilities": {}}) == {}


def test_declared_non_reasoning_entry_publishes_false():
    entry = {
        "name": "qwen3-8b",
        "capabilities": {"supports_tool_calling": True, "model_family": "qwen"},
    }
    assert endpoint_limit_metadata(entry) == {
        "reasoning": False,
        "supports_reasoning_effort": False,
    }


def test_family_with_a_reasoning_parser_is_a_reasoning_model():
    """Kimi K2.7 Code thinks always and exposes no effort control."""
    entry = {"name": "Kimi-K2.7-Code", "capabilities": {"model_family": "kimi_k2"}}
    assert endpoint_limit_metadata(entry) == {
        "reasoning": True,
        "supports_reasoning_effort": False,
    }


# ---------------------------------------------------------------------------
# U4: the serving-profile field
# ---------------------------------------------------------------------------


def test_serving_profile_max_output_tokens_round_trips_and_defaults_to_none():
    assert ServingProfile().max_output_tokens is None
    assert ServingProfile().to_dict()["max_output_tokens"] is None
    profile = ServingProfile.from_dict({"max_output_tokens": 131072})
    assert profile.max_output_tokens == 131072
    assert ServingProfile.from_dict(profile.to_dict()) == profile
    # Pre-2026-10-05 rows have no key: the field stays unset.
    assert ServingProfile.from_dict({"max_model_len": 4096}).max_output_tokens is None


def test_serving_profile_request_carries_max_output_tokens():
    req = ServingProfileRequest(max_model_len=202752, max_output_tokens=131072)
    profile = serving_profile_from_request(req)
    assert profile is not None
    assert profile.max_output_tokens == 131072
    assert profile.max_model_len == 202752
    assert ServingProfileRequest().max_output_tokens is None
    with pytest.raises(ValidationError):
        ServingProfileRequest(max_output_tokens=0)


# ---------------------------------------------------------------------------
# Provision plumbing: catalog entry → endpoint metadata → listing
# ---------------------------------------------------------------------------


class _StubOrchestrationService:
    def __init__(self):
        self.requests = []

    async def provision(self, request):
        self.requests.append(request)
        return ProvisionResult(success=True, job_id="job-123")


def test_provision_records_limits_from_the_catalog_entry(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda _name: {
            "name": "glm-5.3",
            "storage_ref": "/shared/models/glm-5.3",
            "model_type": "llm",
            "serving_profile": {"max_model_len": 202752, "max_output_tokens": 131072},
            "capabilities": {"model_family": "glm"},
        },
    )
    monkeypatch.setattr("umrs.orchestrator.service.get_orchestration_service", lambda: service)
    app = FastAPI()
    app.include_router(endpoints_api.router)
    app.dependency_overrides[get_registry] = EndpointRegistry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "glm-5.3"},
        )

    assert response.status_code == 202
    sent = service.requests[0]
    assert sent.metadata["max_output_tokens"] == 131072
    assert sent.metadata["reasoning"] is True
    assert sent.metadata["supports_reasoning_effort"] is True
    # Profile values reach the launch unchanged ...
    assert sent.env_vars["VLLM_MAX_MODEL_LEN"] == "202752"
    # ... and the output budget is NOT a launch arg: the launchers have no
    # generation-config lane (U4 publishes it in /v1/models only).
    assert "VLLM_EXTRA_ARGS" not in sent.env_vars
    assert not any("GENERATION" in key for key in sent.env_vars)


def test_provision_without_catalog_limits_records_none(monkeypatch):
    service = _StubOrchestrationService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda _name: {
            "name": "qwen3-8b",
            "storage_ref": "/shared/models/qwen3-8b",
            "model_type": "llm",
            "serving_profile": {},
        },
    )
    monkeypatch.setattr("umrs.orchestrator.service.get_orchestration_service", lambda: service)
    app = FastAPI()
    app.include_router(endpoints_api.router)
    app.dependency_overrides[get_registry] = EndpointRegistry

    with TestClient(app) as client:
        response = client.post(
            "/umrs/endpoints/provision",
            json={"project_id": "project-a", "model": "qwen3-8b"},
        )

    assert response.status_code == 202
    metadata = service.requests[0].metadata
    assert not ({"max_output_tokens", "reasoning", "supports_reasoning_effort"} & set(metadata))


def test_recorded_metadata_reaches_the_listing():
    entry = {
        "name": "glm-5.3",
        "serving_profile": {"max_output_tokens": 131072},
        "capabilities": {"model_family": "glm"},
    }
    registry = _registry({"max_model_len": 202752, **endpoint_limit_metadata(entry)})

    [listed] = _list(registry, [{"id": "zai-org/GLM-5.3", "object": "model"}])

    assert {key: listed[key] for key in _LIMIT_KEYS} == {
        "context_window": 202752,
        "max_output_tokens": 131072,
        "reasoning": True,
        "supports_reasoning_effort": True,
    }


def _durable_spec():
    from umrs.orchestrator.deployment_spec import DeploymentSpec

    return DeploymentSpec(
        alias="glm-5.3",
        huggingface_id="zai-org/GLM-5.3",
        model_family="glm",
        served_names=("glm-5.3",),
        storage_ref="/models/glm-5.3",
        tp=1,
        pp=1,
        ep=False,
        parallel_strategy="tp_pp",
        gpus_per_node=1,
        node_count=1,
        tool_call_parser="glm47",
        reasoning_parser="glm45",
        kv_cache_dtype=None,
        enforce_eager=True,
        flashinfer_moe_fp16=False,
        chat_template_format="string",
        mm_encoder_tp_mode=None,
        trust_remote_code=True,
        enable_auto_tool_choice=True,
        enable_prefix_caching=True,
        max_model_len=202752,
        max_num_seqs=4,
        max_num_batched_tokens=8192,
        gpu_memory_utilization=0.9,
        partition="test-gpu",
        slurm_account=None,
        container_image="/images/test.sif",
        slurm_time_limit="96:00:00",
        slurm_cpus=8,
        slurm_memory="",
        readiness_probe="models",
        readiness_probe_model="glm-5.3",
        readiness_path=None,
        env_overrides={},
        ttl_sec=0,
        tier="user",
        registry_model_id="model-glm",
    )


def test_durable_provision_path_records_limits_from_the_catalog_entry():
    """The durable-job adapter registers endpoints itself; it records the
    same limit metadata as the provision route."""
    import asyncio
    from datetime import datetime
    from unittest.mock import AsyncMock, MagicMock

    from umrs.jobs import (
        AttemptStatus,
        Job,
        JobAttempt,
        JobStatus,
        JobType,
        compute_idempotency_key,
    )
    from umrs.jobs.adapters import EndpointProvisionBackend
    from umrs.orchestrator.deployment_spec import DeploymentSpec
    from umrs.orchestrator.preflight import PreflightResult

    orchestrator = AsyncMock()
    orchestrator.backend_type = "slurm"
    orchestrator.submit_job = AsyncMock(
        return_value=MagicMock(
            job_id="slurm-1",
            status=MagicMock(value="pending"),
            endpoint=MagicMock(node="gpu-0"),
        )
    )
    registry = MagicMock()
    backend = EndpointProvisionBackend(
        orchestrator=orchestrator,
        registry=registry,
        port_manager=MagicMock(allocate=MagicMock(return_value=11434)),
    )
    spec = {"model": "glm-5.3", "model_path": "/models/glm-5.3", "ttl_sec": 0}
    job = Job(
        job_id="job-glm",
        project_id="proj-1",
        user_id="user-1",
        job_type=JobType.ENDPOINT_PROVISION,
        status=JobStatus.RUNNING,
        spec=spec,
        idempotency_key=compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
        ),
        priority=0,
        desired_max_attempts=3,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    attempt = JobAttempt(
        attempt_id="att-glm",
        job_id="job-glm",
        attempt_no=1,
        backend="slurm",
        status=AttemptStatus.CREATED,
        umrs_instance_id="umrs-1",
        created_at=datetime.utcnow(),
    )
    catalog_entry = MagicMock(storage_ref="/models/glm-5.3")
    catalog_entry.to_dict.return_value = {
        "name": "glm-5.3",
        "serving_profile": {"max_output_tokens": 131072},
        "capabilities": {"model_family": "glm"},
    }
    site = MagicMock()
    site.scheduler.partition = "test-gpu"
    site.partition_profiles = {}

    with (
        patch.object(backend, "_determine_target_host", return_value="gpu-0"),
        patch("umrs.jobs.adapters._resolve_model_entry", return_value=catalog_entry),
        patch(
            "umrs.jobs.adapters._check_active_certification",
            return_value={"cert_id": "cert-1", "smoke_test_passed": True},
        ),
        patch("umrs.orchestrator.site_config.load_site_config", return_value=site),
        patch.object(DeploymentSpec, "from_registry", return_value=_durable_spec()),
        patch(
            "umrs.orchestrator.preflight.PreflightValidator.run_all",
            new=AsyncMock(return_value=PreflightResult(passed=True, checks=[])),
        ),
    ):
        asyncio.run(backend.submit_job(job, attempt))

    metadata = registry.register.call_args.kwargs["metadata"]
    assert metadata["max_model_len"] == 202752
    assert metadata["max_output_tokens"] == 131072
    assert metadata["reasoning"] is True
    assert metadata["supports_reasoning_effort"] is True
