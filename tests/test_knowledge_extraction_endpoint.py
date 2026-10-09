"""Semantic KG extraction resolves its model through the endpoint registry.

It used to dial UMRS_AGENT_LITELLM_URL, defaulting to http://127.0.0.1:11434 —
inside a container, the container itself — so every extraction was refused,
and the job still reported degraded=False: a "complete" ingest that produced
only structural nodes. Measured 2026-10-09 on the stand-alone stack.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.orchestrator.endpoint_registry import EndpointStatus


def _ep(project, model, *, access=None, meta=None, url="http://m:11434"):
    metadata = dict(meta or {})
    if access is not None:
        metadata["access"] = access
    return SimpleNamespace(project_id=project, model=model, metadata=metadata, litellm_url=url)


class _Registry:
    def __init__(self, eps):
        self.eps = eps
        self.status_filters = []

    def list_all(self, project_id=None, status_filter=None):
        self.status_filters.append(status_filter)
        return [e for e in self.eps if project_id is None or e.project_id == project_id]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in (
        "UMRS_AGENT_LITELLM_URL",
        "UMRS_KNOWLEDGE_EXTRACT_MODEL",
        "UMRS_KNOWLEDGE_LLM_EXTRACT",
    ):
        monkeypatch.delenv(var, raising=False)


def _use(monkeypatch, eps):
    reg = _Registry(eps)
    monkeypatch.setattr("umrs.orchestrator.endpoint_registry.get_registry", lambda: reg)
    return reg


def test_explicit_url_still_wins(monkeypatch):
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", "http://site-litellm:4000")
    assert ka._extract_chat_endpoint("p") == ("http://site-litellm:4000", "gpt-oss")


def test_resolves_the_projects_ready_chat_endpoint(monkeypatch):
    reg = _use(monkeypatch, [_ep("p", "llama3.2:3b", url="http://host.containers.internal:11434")])
    assert ka._extract_chat_endpoint("p") == (
        "http://host.containers.internal:11434",
        "llama3.2:3b",
    )
    assert [EndpointStatus.READY, EndpointStatus.DEGRADED] in reg.status_filters


def test_uses_a_system_shared_endpoint_from_another_project(monkeypatch):
    _use(monkeypatch, [_ep("default", "llama3.2:3b", access=["system"])])
    assert ka._extract_chat_endpoint("other")[1] == "llama3.2:3b"


def test_never_uses_an_unshared_endpoint_of_another_project(monkeypatch):
    _use(monkeypatch, [_ep("someone-else", "private-model")])
    assert ka._extract_chat_endpoint("p") is None


def test_skips_vision_and_embedding_rows(monkeypatch):
    _use(
        monkeypatch,
        [
            _ep("p", "clip", meta={"model_type": "vision"}),
            _ep("p", "e5", meta={"service_kind": "embedding"}),
            _ep("p", "chatty"),
        ],
    )
    assert ka._extract_chat_endpoint("p")[1] == "chatty"


def test_served_model_name_is_what_is_sent(monkeypatch):
    _use(monkeypatch, [_ep("p", "catalog-name", meta={"served_model_name": "served-name"})])
    assert ka._extract_chat_endpoint("p")[1] == "served-name"


def test_preferred_model_selects_among_candidates(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_MODEL", "qwen2.5:7b")
    _use(monkeypatch, [_ep("p", "llama3.2:3b"), _ep("p", "qwen2.5:7b", url="http://q:1")])
    assert ka._extract_chat_endpoint("p") == ("http://q:1", "qwen2.5:7b")


def test_no_endpoint_is_reported_not_silent(monkeypatch):
    _use(monkeypatch, [])
    notes: list[str] = []
    assert ka._llm_extract_document("prose", project_id="p", notes=notes) is None
    assert notes and "no READY chat endpoint" in notes[0]


def test_connection_failure_is_reported(monkeypatch):
    _use(monkeypatch, [_ep("p", "m")])

    def _boom(*_a, **_k):
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr("umrs.knowledge.agent_executor._default_litellm_chat", _boom)
    notes: list[str] = []
    assert ka._llm_extract_document("prose", project_id="p", notes=notes) is None
    assert "ConnectionRefusedError" in notes[0]


def test_disabled_is_not_a_failure(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_LLM_EXTRACT", "0")
    notes: list[str] = []
    assert ka._llm_extract_document("prose", project_id="p", notes=notes) is None
    assert notes == []


def test_notes_flow_out_of_nodes_edges_for_artifact(monkeypatch):
    _use(monkeypatch, [])
    parsed = {
        k: [] for k in ("modules", "classes", "functions", "algorithms", "dependencies", "kernels")
    }
    notes: list[str] = []
    nodes, _ = ka._nodes_edges_for_artifact(
        "art-1", "ir-1", "document", b"plain prose", parsed, "p", "", notes=notes
    )
    assert len(nodes) == 1 and nodes[0].node_type == "file"
    assert notes, "a failed extraction must be visible to the caller"


@pytest.mark.parametrize(
    ("raw", "expected"), [(None, 60.0), ("180", 180.0), ("1", 5.0), ("nonsense", 60.0)]
)
def test_extract_timeout_is_configurable(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("UMRS_KNOWLEDGE_EXTRACT_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("UMRS_KNOWLEDGE_EXTRACT_TIMEOUT", raw)
    assert ka._extract_timeout() == expected
