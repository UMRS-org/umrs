"""DEC-063 / gap D-63: one gated registry resolver for every server consumer.

Before 2026-11-24 only umrs-infer's site source refused a public-passthrough
registry row for a sensitive project (ADR-062). Chat and llm discovery,
knowledge extraction, the agentic executor and perception picked rows
without the class gate, so a sensitive project's text reached a public
provider through any of them. Every test below that expects a refusal
failed on that code; the admitted cases pin that non-sensitive projects and
self-deployed rows keep working (hosted and self-hosted alike).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from umrs.api import chat as chat_api
from umrs.api import llm as llm_api
from umrs.data_catalog import perception
from umrs.infer_proxy import site_models
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import llm_chat
from umrs.knowledge.agent_executor import UmrsAgentIterationExecutor
from umrs.orchestrator import registry_route
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.shared import endpoint_locality

_PUBLIC = {"locality": "public", "model_type": "llm"}
_OPENAI = {"provider_kind": "openai", "model_type": "llm"}
_SELF = {"model_type": "llm"}


class _Cursor:
    def __init__(self, classes, asked):
        self._classes, self._asked, self._row = classes, asked, None

    def execute(self, _sql, params):
        self._asked.append(params[0])
        klass = self._classes.get(params[0])
        self._row = (klass,) if klass else None

    def fetchone(self):
        return self._row


class _Pool:
    """A projects table holding only ``data_class`` (migration 0107)."""

    def __init__(self, classes):
        self.classes, self.asked = classes, []

    def getconn(self):
        return SimpleNamespace(cursor=lambda: _Cursor(self.classes, self.asked))

    def putconn(self, _conn):
        return None


@pytest.fixture()
def pool(monkeypatch):
    fake = _Pool({"p-sens": "sensitive", "p-open": "non_sensitive"})
    monkeypatch.setattr(endpoint_locality, "_default_pool", lambda: fake)
    return fake


@pytest.fixture()
def registry(monkeypatch):
    reg = EndpointRegistry()
    monkeypatch.setattr("umrs.orchestrator.endpoint_registry.get_registry", lambda: reg)
    for var in ("UMRS_AGENT_LITELLM_URL", "UMRS_KNOWLEDGE_EXTRACT_MODEL", "UMRS_AGENT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return reg


def _row(reg, project, job, meta, model="gpt-x"):
    reg.register(project, model, job, f"{job}.host", 4000, metadata=dict(meta))
    reg.mark_ready(job)


# ── the resolver ──────────────────────────────────────────────────────────


def test_sensitive_project_is_refused_a_public_row_by_name(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    with pytest.raises(registry_route.RegistryRouteRefused) as caught:
        registry_route.resolve_registry_endpoint("p-sens", "gpt-x")
    assert caught.value.reason == "source_class_refused"
    assert "source_class_refused" in str(caught.value)


def test_public_provider_kind_counts_as_public(pool, registry):
    _row(registry, "p-sens", "j-oai", _OPENAI)
    with pytest.raises(registry_route.RegistryRouteRefused):
        registry_route.resolve_registry_endpoint("p-sens")


def test_non_sensitive_project_keeps_its_public_row(pool, registry):
    _row(registry, "p-open", "j-pub", _PUBLIC)
    assert registry_route.resolve_registry_endpoint("p-open", "gpt-x").job_id == "j-pub"


def test_self_deployed_row_is_the_sensitive_projects_own_route(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    _row(registry, "p-sens", "j-self", _SELF)
    assert registry_route.resolve_registry_endpoint("p-sens", "gpt-x").job_id == "j-self"


def test_a_system_shared_public_row_is_refused_too(pool, registry):
    _row(registry, "p-sys", "j-shared", {**_PUBLIC, "access": ["system"]})
    with pytest.raises(registry_route.RegistryRouteRefused):
        registry_route.resolve_registry_endpoint("p-sens")


def test_no_public_row_means_no_class_lookup(pool, registry):
    _row(registry, "p-sens", "j-self", _SELF)
    assert registry_route.resolve_registry_endpoint("p-sens").job_id == "j-self"
    assert pool.asked == []


def test_nothing_serving_the_model_is_none_not_a_refusal(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    assert registry_route.resolve_registry_endpoint("p-sens", "other-model") is None


# ── server consumers ──────────────────────────────────────────────────────


def test_chat_discovery_refuses_the_public_row(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    with pytest.raises(HTTPException) as caught:
        chat_api._discover_endpoint(registry, "p-sens", "gpt-x")
    assert caught.value.status_code == 503
    assert "source_class_refused" in str(caught.value.detail)


def test_chat_discovery_admits_non_sensitive_and_self_deployed(pool, registry):
    _row(registry, "p-open", "j-pub", _PUBLIC)
    _row(registry, "p-sens", "j-self", _SELF)
    assert chat_api._discover_endpoint(registry, "p-open", "gpt-x").job_id == "j-pub"
    assert chat_api._discover_endpoint(registry, "p-sens", "gpt-x").job_id == "j-self"


def test_llm_discovery_refuses_the_public_row(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    with pytest.raises(HTTPException) as caught:
        llm_api._discover_endpoint(registry, "p-sens", "gpt-x")
    assert caught.value.status_code == 503
    assert "source_class_refused" in str(caught.value.detail)


def test_llm_discovery_admits_non_sensitive_and_self_deployed(pool, registry):
    _row(registry, "p-open", "j-pub", _PUBLIC)
    _row(registry, "p-sens", "j-self", _SELF, model="local-m")
    assert llm_api._discover_endpoint(registry, "p-open", "gpt-x").job_id == "j-pub"
    assert llm_api._discover_endpoint(registry, "p-sens", "local-m").job_id == "j-self"


def test_llm_discovery_skips_a_refused_pick_for_the_self_deployed_replica(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    _row(registry, "p-sens", "j-self", _SELF)
    registry.get_by_job_id("j-self").active_requests = 3
    assert llm_api._discover_endpoint(registry, "p-sens", "gpt-x").job_id == "j-self"


def test_llm_failover_never_picks_a_public_replica(pool, registry):
    _row(registry, "p-sens", "j-self", _SELF)
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    failed = SimpleNamespace(job_id="j-self", model="gpt-x")
    assert llm_api._alternate_endpoint(registry, "p-sens", "gpt-x", None, failed) is None


def test_extraction_refuses_the_public_row(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    assert ka._extract_chat_endpoint("p-sens") is None
    _row(registry, "p-open", "j-open", _PUBLIC)
    assert ka._extract_chat_endpoint("p-open") == ("http://j-open.host:4000", "gpt-x")


def test_agent_executor_refuses_the_public_row_by_name(pool, registry, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(
        llm_chat, "chat_completion", lambda *_a, **kw: sent.append(kw["base_url"]) or "{}"
    )
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    executor = UmrsAgentIterationExecutor(SimpleNamespace(project_id="p-sens"))
    reply = executor._call_llm("goal", {"nodes": [], "edges": []}, 1)
    assert "source_class_refused" in reply["_error"] and sent == []


def test_agent_executor_uses_the_self_deployed_row(pool, registry, monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(
        llm_chat,
        "chat_completion",
        lambda *_a, **kw: sent.append(kw["base_url"]) or '{"hypothesis": "h"}',
    )
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    _row(registry, "p-sens", "j-self", _SELF)
    executor = UmrsAgentIterationExecutor(SimpleNamespace(project_id="p-sens"))
    executor._call_llm("goal", {"nodes": [], "edges": []}, 1)
    assert sent == ["http://j-self.host:4000"]


def test_perception_never_sees_the_public_row(pool, registry):
    _row(registry, "p-sens", "j-pub", _PUBLIC)
    _row(registry, "p-open", "j-open", _PUBLIC)
    assert perception._usable_endpoints("p-sens") == []
    assert [ep.job_id for ep in perception._usable_endpoints("p-open")] == ["j-open"]


def test_agent_url_exception_is_the_one_declared_override(registry, monkeypatch):
    monkeypatch.setenv("UMRS_AGENT_LITELLM_URL", " http://site-litellm:4000 ")
    assert registry_route.agent_url_exception("p-sens") == "http://site-litellm:4000"
    assert ka._extract_chat_endpoint("p-sens") == ("http://site-litellm:4000", "gpt-oss")


def test_infer_proxy_shares_the_one_public_row_predicate():
    assert site_models.is_public_row is endpoint_locality.is_public_row
