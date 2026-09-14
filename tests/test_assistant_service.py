"""Unit tests for the UMRS Assistant Service (ADR-012).

Network is never touched: RAG retrieval and the upstream forward are exercised
through pure helpers and monkeypatched hooks.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from umrs.api import assistant_service as svc


@pytest.fixture()
def client() -> TestClient:
    return TestClient(svc.app)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_extract_last_user_query_plain_string():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest question"},
    ]
    assert svc.extract_last_user_query(messages) == "latest question"


def test_extract_last_user_query_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "part one"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "part two"},
            ],
        }
    ]
    assert svc.extract_last_user_query(messages) == "part one part two"


def test_extract_last_user_query_none_when_no_user():
    assert svc.extract_last_user_query([{"role": "system", "content": "s"}]) is None


def test_build_context_block_empty_when_no_sources():
    assert svc.build_context_block([]) == ""


def test_build_context_block_includes_numbered_sources():
    sources = [
        {"content": "alpha", "metadata": {"source": "a.md"}},
        {"content": "beta", "metadata": {"source": "b.md"}},
    ]
    block = svc.build_context_block(sources)
    assert "[1] alpha" in block
    assert "[2] beta" in block
    assert "Relevant context" in block
    assert "a.md" in block and "b.md" in block


def test_build_context_block_respects_token_budget():
    big = {"content": "x" * 10_000, "metadata": {}}
    block = svc.build_context_block([big], max_context_tokens=50)
    # Truncated (or dropped) but never exceeds a small multiple of the budget.
    assert len(block) < 10_000


def test_inject_context_merges_into_existing_system_message():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": "q"},
    ]
    out = svc.inject_context_into_messages(messages, "\nCTX")
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "base\nCTX"
    assert out[1]["content"] == "q"


def test_inject_context_prepends_system_when_absent():
    messages = [{"role": "user", "content": "q"}]
    out = svc.inject_context_into_messages(messages, "\nCTX")
    assert out[0]["role"] == "system"
    assert "\nCTX" in out[0]["content"]
    assert len(out) == 2


def test_inject_context_noop_when_block_empty():
    messages = [{"role": "user", "content": "q"}]
    assert svc.inject_context_into_messages(messages, "") == messages


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_config_defaults(monkeypatch):
    for name in (
        "ASSISTANT_UPSTREAM_BASE_URL",
        "ASSISTANT_UPSTREAM_API_KEY",
        "LITELLM_MASTER_KEY",
        "VLLM_API_KEY",
        "ASSISTANT_RAG_URL",
        "ASSISTANT_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    config = svc._resolve_config()
    assert config.upstream_base_url == "http://litellm:4000/v1"
    assert config.rag_url == "http://localhost:8081"
    assert config.default_model == "rivault-gpt-oss-static"
    assert config.upstream_api_key == ""


def test_config_prefers_litellm_master_key(monkeypatch):
    monkeypatch.delenv("ASSISTANT_UPSTREAM_API_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-live")
    monkeypatch.setenv("VLLM_API_KEY", "sk-other")
    assert svc._resolve_config().upstream_api_key == "sk-live"


# ---------------------------------------------------------------------------
# augment_messages (RAG hook monkeypatched — no network)
# ---------------------------------------------------------------------------


async def test_augment_messages_injects_when_sources_returned(monkeypatch):
    async def _fake_retrieve(*_a, **_k):
        return [{"content": "knowledge", "metadata": {}}]

    monkeypatch.setattr(svc, "retrieve_context", _fake_retrieve)
    config = svc._resolve_config()
    out = await svc.augment_messages(
        config,
        [{"role": "user", "content": "hi"}],
        project_id="p1",
        top_k=3,
        threshold=0.7,
    )
    assert out[0]["role"] == "system"
    assert "knowledge" in out[0]["content"]


async def test_augment_messages_passthrough_when_no_sources(monkeypatch):
    async def _fake_retrieve(*_a, **_k):
        return []

    monkeypatch.setattr(svc, "retrieve_context", _fake_retrieve)
    config = svc._resolve_config()
    messages = [{"role": "user", "content": "hi"}]
    assert (
        await svc.augment_messages(config, messages, project_id=None, top_k=3, threshold=0.7)
        == messages
    )


async def test_augment_messages_passthrough_when_no_query(monkeypatch):
    called = {"n": 0}

    async def _spy(*_a, **_k):
        called["n"] += 1
        return []

    monkeypatch.setattr(svc, "retrieve_context", _spy)
    config = svc._resolve_config()
    messages = [{"role": "system", "content": "s"}]
    assert (
        await svc.augment_messages(config, messages, project_id="p", top_k=3, threshold=0.7)
        == messages
    )
    assert called["n"] == 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_manifest_shape(client):
    resp = client.get("/manifest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "assistant"
    assert data["surfaces"]["chat"] == "/v1/chat/completions"
    assert data["capabilities"]["rag"] is True
    assert data["capabilities"]["project_aware"] is True


def test_chat_completions_strips_extensions_and_forwards(monkeypatch, client):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"id": "cmpl-1", "choices": []}

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, **_kwargs):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    monkeypatch.setattr(svc.httpx, "AsyncClient", FakeAsyncClient)

    async def _fake_retrieve(*_a, **_k):
        return []

    monkeypatch.setattr(svc, "retrieve_context", _fake_retrieve)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "project_id": "proj-42",
            "rag": True,
            "rag_top_k": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == "cmpl-1"
    # Assistant-only extensions must not leak to the upstream OpenAI endpoint.
    for key in ("project_id", "rag", "rag_top_k", "rag_threshold"):
        assert key not in captured["json"]
    assert captured["json"]["model"] == "rivault-gpt-oss-static"
    assert captured["url"].endswith("/chat/completions")


def test_chat_completions_upstream_failure_returns_502(monkeypatch, client):
    class FailingClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(svc.httpx, "AsyncClient", FailingClient)

    async def _fake_retrieve(*_a, **_k):
        return []

    monkeypatch.setattr(svc, "retrieve_context", _fake_retrieve)

    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Security: loopback bind by default
# ---------------------------------------------------------------------------


def test_assistant_service_defaults_to_loopback_bind():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "src/umrs/api/assistant_service.py").read_text()
    assert 'os.getenv("ASSISTANT_BIND_HOST", "127.0.0.1")' in source
    assert 'host="0.0.0.0"' not in source


# ---------------------------------------------------------------------------
# #209: /v1/models must fail honestly when the upstream probe fails
# ---------------------------------------------------------------------------


class _BoomHttpClient:
    def __init__(self, *_a, **_k):
        pass

    async def __aenter__(self):
        raise ConnectionError("upstream refused")

    async def __aexit__(self, *_a):
        return False


class _StatusHttpClient:
    def __init__(self, status_code: int, payload: dict | None = None):
        self._resp = FakeResp(status_code, payload or {})

    def __call__(self, *_a, **_k):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def get(self, *_a, **_k):
        return self._resp


class FakeResp:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_list_models_503_when_upstream_unreachable(monkeypatch, client):
    monkeypatch.setattr(svc.httpx, "AsyncClient", _BoomHttpClient)
    resp = client.get("/v1/models")
    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["degraded"] is True
    assert "configured_default_model" in error
    assert "upstream refused" in error["detail"]


def test_list_models_503_when_upstream_not_200(monkeypatch, client):
    monkeypatch.setattr(svc.httpx, "AsyncClient", _StatusHttpClient(500))
    resp = client.get("/v1/models")
    assert resp.status_code == 503
    assert "500" in resp.json()["error"]["detail"]


def test_list_models_passes_through_live_upstream(monkeypatch, client):
    payload = {"object": "list", "data": [{"id": "qwen-live", "object": "model"}]}
    monkeypatch.setattr(svc.httpx, "AsyncClient", _StatusHttpClient(200, payload))
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "qwen-live"
