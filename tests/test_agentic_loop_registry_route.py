"""The agentic loop reaches its model through the endpoint registry.

Gap D-63 / plan §3c.2: with no injected chat callable (every production build
site), the executor posted to UMRS_AGENT_LITELLM_URL, default
http://127.0.0.1:11434, which inside a container is the container itself.
On the stand-alone stack every agentic iteration failed there and was still
marked completed. An explicit UMRS_AGENT_LITELLM_URL keeps winning, as it does
for extraction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters
from umrs.knowledge import agent_executor
from umrs.knowledge.agent_executor import UmrsAgentIterationExecutor

_LLAMA = SimpleNamespace(
    model="llama3.2:3b", litellm_url="http://host.containers.internal:11434", metadata={}
)
_QWEN = SimpleNamespace(
    model="qwen", litellm_url="http://gpu-node:8000", metadata={"served_model_name": "Qwen/Q"}
)


@pytest.fixture()
def calls(monkeypatch):
    seen: list[dict] = []

    def chat(messages, *, model, base_url, api_key, max_tokens):
        seen.append({"model": model, "base_url": base_url})
        return '{"hypothesis": "h"}'

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    monkeypatch.delenv("UMRS_AGENT_LITELLM_URL", raising=False)
    monkeypatch.delenv("UMRS_AGENT_MODEL", raising=False)
    return seen


def _run(monkeypatch, candidates, **kwargs):
    asked: list[str] = []

    def lookup(project_id):
        asked.append(project_id)
        return list(candidates)

    monkeypatch.setattr(knowledge_adapters, "_registry_chat_candidates", lookup)
    executor = UmrsAgentIterationExecutor(SimpleNamespace(project_id="p-zeo"), **kwargs)
    return executor._call_llm("goal", {"nodes": [], "edges": []}, 1), asked


def test_the_projects_registry_endpoint_serves_the_turn(monkeypatch, calls):
    reply, asked = _run(monkeypatch, [_LLAMA])
    assert reply == {"hypothesis": "h"} and asked == ["p-zeo"]
    assert calls == [{"model": "llama3.2:3b", "base_url": _LLAMA.litellm_url}]


def test_a_named_model_picks_among_candidates(monkeypatch, calls):
    monkeypatch.setenv("UMRS_AGENT_MODEL", "Qwen/Q")
    _run(monkeypatch, [_LLAMA, _QWEN])
    assert calls == [{"model": "Qwen/Q", "base_url": _QWEN.litellm_url}]


def test_an_explicit_url_still_wins(monkeypatch, calls):
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://site-litellm:4000")
    _reply, asked = _run(monkeypatch, [_LLAMA])
    assert asked == []
    assert calls == [{"model": "rivault-gpt-oss-static", "base_url": "http://site-litellm:4000"}]


def test_no_endpoint_is_named_not_dialled(monkeypatch, calls):
    reply, _ = _run(monkeypatch, [])
    assert calls == []
    assert "no READY chat endpoint" in reply["_error"]


def test_an_empty_project_never_reaches_an_unfiltered_lookup(monkeypatch, calls):
    """list_all(project_id="") skips its project filter: every project's
    private endpoint would have counted as the loop's own."""
    asked: list[str] = []
    monkeypatch.setattr(
        knowledge_adapters, "_registry_chat_candidates", lambda pid: asked.append(pid) or [_LLAMA]
    )
    executor = UmrsAgentIterationExecutor(SimpleNamespace(project_id=""))
    reply = executor._call_llm("goal", {"nodes": [], "edges": []}, 1)
    assert asked == [] and calls == []
    assert "no READY chat endpoint" in reply["_error"]
