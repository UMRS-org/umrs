from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from umrs.api import rag_service
from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint


class FakeRegistry:
    def __init__(self, endpoints):
        self._endpoints = list(endpoints)
        self.used = []

    def get(self, project_id, model, prefer_warm=True, min_ttl_sec=None):
        for endpoint in self._endpoints:
            if endpoint.project_id != project_id:
                continue
            if endpoint.model != model:
                continue
            if prefer_warm and not endpoint.is_usable:
                continue
            return endpoint
        return None

    def list_all(self, project_id=None, status_filter=None):
        endpoints = list(self._endpoints)
        if project_id is not None:
            endpoints = [endpoint for endpoint in endpoints if endpoint.project_id == project_id]
        if status_filter is not None:
            endpoints = [endpoint for endpoint in endpoints if endpoint.status in status_filter]
        return endpoints

    def mark_used(self, job_id):
        self.used.append(job_id)


def _ready_endpoint(project_id, model, job_id, host, port, minutes_ago=1):
    created_at = datetime.now(UTC) - timedelta(minutes=minutes_ago + 1)
    ready_at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return RegisteredEndpoint(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host=host,
        port=port,
        status=EndpointStatus.READY,
        created_at=created_at,
        ready_at=ready_at,
        backend="slurm-podman",
    )


def _starting_endpoint(project_id, model, job_id, host, port, minutes_ago=1):
    created_at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    return RegisteredEndpoint(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host=host,
        port=port,
        status=EndpointStatus.STARTING,
        created_at=created_at,
        backend="slurm-podman",
    )


def test_strip_reasoning_blocks_removes_think_sections():
    raw = "<think>internal chain</think>Final answer"

    assert rag_service._strip_reasoning_blocks(raw) == "Final answer"


def test_extract_rewritten_queries_strips_reasoning_blocks_and_quotes():
    raw = "<think>hidden</think>\n1. \"What is RiVault?\"\n- 'How does it work?'"

    assert rag_service._extract_rewritten_queries(raw, "fallback") == [
        "What is RiVault?",
        "How does it work?",
    ]


def test_build_llm_candidates_prefers_registry_chat_endpoint(monkeypatch):
    registry = FakeRegistry(
        [
            _ready_endpoint(
                project_id="rag-qa-warmup",
                model="intfloat/e5-base-v2",
                job_id="job-embed",
                host="ng-dgx-m0-08",
                port=8001,
                minutes_ago=3,
            ),
            _ready_endpoint(
                project_id="rag-qa-chat",
                model="Qwen/Qwen3-0.6B",
                job_id="job-chat",
                host="ng-dgx-m0-09",
                port=8001,
                minutes_ago=1,
            ),
        ]
    )

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("UMRS_RAG_LLM_PROJECT_ID", raising=False)
    monkeypatch.delenv("UMRS_RAG_LLM_MODEL", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-live")
    monkeypatch.setattr(rag_service, "get_registry", lambda: registry)
    monkeypatch.setattr(
        rag_service,
        "get_site_config",
        lambda: SimpleNamespace(
            inject_env=lambda overwrite=False: None,
            network=SimpleNamespace(litellm_master_key=""),
        ),
    )
    monkeypatch.setattr(
        rag_service,
        "DB_MODEL_STORE",
        {
            "text-entry": {
                "store_embeddings_flag": True,
                "model": {"model_name": "intfloat/e5-base-v2"},
            }
        },
    )

    candidates = rag_service._build_llm_candidates()

    assert candidates[0].model_name == "Qwen/Qwen3-0.6B"
    assert candidates[0].base_url == "http://ng-dgx-m0-09:8001/v1"
    assert candidates[0].job_id == "job-chat"
    assert all(candidate.model_name != "intfloat/e5-base-v2" for candidate in candidates[:-1])


def test_build_llm_candidates_uses_starting_chat_endpoint_when_ready_state_lags(monkeypatch):
    registry = FakeRegistry(
        [
            _starting_endpoint(
                project_id="rag-qa-chat",
                model="Qwen/Qwen3-0.6B",
                job_id="job-chat",
                host="ng-dgx-m0-09",
                port=8001,
                minutes_ago=1,
            ),
            _starting_endpoint(
                project_id="rag-qa-warmup",
                model="intfloat/e5-base-v2",
                job_id="job-embed",
                host="ng-dgx-m0-08",
                port=8001,
                minutes_ago=2,
            ),
        ]
    )

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("UMRS_RAG_LLM_PROJECT_ID", raising=False)
    monkeypatch.delenv("UMRS_RAG_LLM_MODEL", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-live")
    monkeypatch.setattr(rag_service, "get_registry", lambda: registry)
    monkeypatch.setattr(
        rag_service,
        "get_site_config",
        lambda: SimpleNamespace(
            inject_env=lambda overwrite=False: None,
            network=SimpleNamespace(litellm_master_key=""),
        ),
    )
    monkeypatch.setattr(
        rag_service,
        "DB_MODEL_STORE",
        {
            "text-entry": {
                "store_embeddings_flag": True,
                "model": {"model_name": "intfloat/e5-base-v2"},
            }
        },
    )

    candidates = rag_service._build_llm_candidates()

    assert candidates[0].model_name == "Qwen/Qwen3-0.6B"
    assert candidates[0].base_url == "http://ng-dgx-m0-09:8001/v1"
    assert candidates[0].source == "registry-stale-fallback"
    assert all(candidate.model_name != "intfloat/e5-base-v2" for candidate in candidates[:-1])


def test_invoke_llm_with_candidates_retries_and_marks_registry_usage(monkeypatch):
    registry = FakeRegistry([])
    candidates = [
        rag_service.ResolvedLLMEndpoint(
            model_name="bad-model",
            base_url="http://bad-endpoint/v1",
            api_key="sk-bad",
            source="registry-fallback",
            job_id="job-bad",
        ),
        rag_service.ResolvedLLMEndpoint(
            model_name="Qwen/Qwen3-0.6B",
            base_url="http://good-endpoint/v1",
            api_key="sk-good",
            source="registry-fallback",
            job_id="job-good",
        ),
    ]

    class FakeChatOpenAI:
        def __init__(self, model, base_url, api_key, temperature):
            self.base_url = base_url

        def invoke(self, messages):
            if "bad-endpoint" in self.base_url:
                raise RuntimeError("upstream failed")
            return SimpleNamespace(content="<think>hidden</think>Recovered answer")

    monkeypatch.setattr(rag_service, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(rag_service, "get_registry", lambda: registry)

    response, candidate = rag_service._invoke_llm_with_candidates(["payload"], candidates)

    assert candidate.job_id == "job-good"
    assert (
        rag_service._strip_reasoning_blocks(rag_service._response_text(response))
        == "Recovered answer"
    )
    assert registry.used == ["job-good"]
