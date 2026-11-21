"""U3 S1 item 4: the site LiteLLM chat helper is public core API.

The RAG answer path, knowledge extraction and the companion turn executor
reached into ``agent_executor._default_litellm_chat``. They now call
``umrs.knowledge.llm_chat.chat_completion`` (looked up on the module, so one
monkeypatch reaches every caller), and api/llm.py shares its key resolver.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.api import llm as llm_api
from umrs.knowledge import agent_executor, llm_chat

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"
CONSUMERS = ("api/rag_spine.py", "jobs/knowledge_adapters.py")  # turn executor: retired (U3 S4a)


def test_the_private_helper_is_gone():
    assert not hasattr(agent_executor, "_default_litellm_chat")


def test_one_service_key_resolver(monkeypatch):
    assert llm_api._resolve_litellm_service_key is llm_chat.resolve_service_key
    monkeypatch.delenv("GATEWAY_LITELLM_SERVICE_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "k-master")
    assert llm_chat.resolve_service_key() == "k-master"
    monkeypatch.setenv("GATEWAY_LITELLM_SERVICE_KEY", "k-gateway")
    assert llm_chat.resolve_service_key() == "k-gateway"


@pytest.mark.parametrize(
    ("message", "text"),
    [
        ({"content": "hi"}, "hi"),
        ({"content": "", "reasoning_content": "thought"}, "thought"),
        ({"content": None, "reasoning": "r"}, "r"),
        ({"content": None}, ""),
    ],
)
def test_message_text(message, text):
    assert llm_chat.message_text(message) == text


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumers_use_the_public_helper(rel):
    tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "umrs.knowledge.agent_executor" not in modules, rel
    assert "llm_chat" in names, rel
