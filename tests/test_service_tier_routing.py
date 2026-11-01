"""Service-model tier routing (ADR-106 D-1, D-5; plan S1/S3/S4, 2026-11-01).

- Role aliases (``svc-*``) route to the least-loaded READY replica of any
  model whose roster entry carries the alias. That includes both models
  during an upgrade roll, and system-tier endpoints shared with the caller
  (ADR-078). The engine always receives the backing model's served name.
- DRAINING rows take no new requests, and drain is one-way: a late
  readiness probe never puts a retiring job back into selection.
- ``/umrs/llm`` retries once, on another replica, when the first fails
  before any byte reached the caller: a connect error or an upstream
  502/503/504. Other statuses are the model's answer and are returned as-is.
- A stream that never opened releases its request count. Before this
  change the count leaked, and a drain would wait on it forever.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.llm import router
from umrs.orchestrator import provision_roster
from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    get_registry,
)

ROSTER = """\
pool: test
endpoints:
  - name: svc-chat-glm53
    model: glm-5.3-flash
    aliases: [svc-chat, glm-5.3-flash-nvfp4]
  - name: deepseek-v4.1-flash-1017
    model: DeepSeek-V4.1-Flash-1017
    aliases: [svc-chat]
  - name: nimble
    model: Bespoke-Nimble-9B
    aliases: [svc-decision]
"""
HEADERS = {"x-user-id": "user-1", "x-project-id": "proj-a"}
CHAT = {"messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(autouse=True)
def roster(tmp_path, monkeypatch):
    path = tmp_path / "pool.yaml"
    path.write_text(ROSTER)
    monkeypatch.setenv(provision_roster.ROSTER_ENV_VAR, str(path))
    provision_roster._cache.clear()
    yield
    provision_roster._cache.clear()


def _add(registry, job_id, model, *, project="system", served=None, status="ready", load=0):
    metadata = {"access": ["system"]}
    if served:
        metadata["served_model_name"] = served
    registry.register(
        project_id=project,
        model=model,
        job_id=job_id,
        host=f"host-{job_id}",
        port=8001,
        backend="slurm",
        metadata=metadata,
    )
    registry.update_status(job_id, EndpointStatus(status))
    registry.get_by_job_id(job_id).active_requests = load


def _client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


class _Upstream:
    """Fake upstream: status per host; records (host, model) of each call."""

    def __init__(self, statuses: dict[str, int] | None = None, refuse: tuple[str, ...] = ()):
        self.statuses = statuses or {}
        self.refuse = refuse
        self.calls: list[tuple[str, str]] = []

    def host_of(self, url) -> str:
        return str(url).split("//", 1)[1].split(":", 1)[0]

    async def request(self, *args, **kwargs):
        import json

        host = self.host_of(kwargs.get("url") or args[-1])
        self.calls.append((host, json.loads(kwargs.get("content") or b"{}").get("model", "")))
        if host in self.refuse:
            raise httpx.ConnectError("refused")
        code = self.statuses.get(host, 200)
        return httpx.Response(code, json={"host": host}, headers={"content-type": "application/json"})


def _post(registry, upstream, model, monkeypatch):
    monkeypatch.setattr(httpx.AsyncClient, "request", upstream.request)
    return _client(registry).post(
        "/umrs/llm/v1/chat/completions", json={"model": model, **CHAT}, headers=HEADERS
    )


def test_role_alias_models_lists_every_entry_in_roster_order() -> None:
    found = {
        "svc-chat": provision_roster.role_alias_models("svc-chat"),
        "svc-decision": provision_roster.role_alias_models("svc-decision"),
        "catalog alias": provision_roster.role_alias_models("glm-5.3-flash-nvfp4"),
        "unknown": provision_roster.role_alias_models("svc-nope"),
    }
    assert found == {
        "svc-chat": ("glm-5.3-flash", "DeepSeek-V4.1-Flash-1017"),
        "svc-decision": ("Bespoke-Nimble-9B",),
        "catalog alias": (),
        "unknown": (),
    }


def test_alias_routes_to_least_loaded_shared_replica_with_served_name(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "glm", "glm-5.3-flash", served="GLM-5.3-Flash-NVFP4", load=3)
    _add(registry, "ds", "DeepSeek-V4.1-Flash-1017", load=1)
    upstream = _Upstream()
    resp = _post(registry, upstream, "svc-chat", monkeypatch)
    assert (resp.status_code, upstream.calls) == (200, [("host-ds", "DeepSeek-V4.1-Flash-1017")])


def test_alias_rewrites_to_served_name(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "glm", "glm-5.3-flash", served="GLM-5.3-Flash-NVFP4")
    upstream = _Upstream()
    _post(registry, upstream, "svc-chat", monkeypatch)
    assert upstream.calls == [("host-glm", "GLM-5.3-Flash-NVFP4")]


def test_draining_replica_takes_no_new_requests(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "old", "glm-5.3-flash", load=0)
    _add(registry, "new", "glm-5.3-flash", load=5)
    registry.mark_draining("old")
    upstream = _Upstream()
    _post(registry, upstream, "svc-chat", monkeypatch)
    assert upstream.calls == [("host-new", "glm-5.3-flash")]


def test_drain_is_one_way() -> None:
    registry = EndpointRegistry()
    _add(registry, "old", "glm-5.3-flash")
    registry.mark_draining("old")
    registry.mark_ready("old")
    registry.mark_degraded("old")
    assert registry.get_by_job_id("old").status == EndpointStatus.DRAINING


def test_only_draining_replicas_means_unavailable(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "old", "glm-5.3-flash")
    registry.mark_draining("old")
    resp = _post(registry, _Upstream(), "svc-chat", monkeypatch)
    assert resp.status_code == 503


def test_failover_on_connect_error(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash", load=0)
    _add(registry, "b", "glm-5.3-flash", load=1)
    upstream = _Upstream(refuse=("host-a",))
    resp = _post(registry, upstream, "glm-5.3-flash", monkeypatch)
    hosts = [h for h, _ in upstream.calls]
    assert (resp.status_code, resp.json(), hosts) == (200, {"host": "host-b"}, ["host-a", "host-b"])


@pytest.mark.parametrize("code", [502, 503, 504])
def test_failover_on_upstream_gateway_status(monkeypatch, code) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash", load=0)
    _add(registry, "b", "glm-5.3-flash", load=1)
    upstream = _Upstream(statuses={"host-a": code})
    resp = _post(registry, upstream, "svc-chat", monkeypatch)
    assert (resp.status_code, resp.json()) == (200, {"host": "host-b"})


def test_model_errors_are_not_retried(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash", load=0)
    _add(registry, "b", "glm-5.3-flash", load=1)
    upstream = _Upstream(statuses={"host-a": 400})
    resp = _post(registry, upstream, "glm-5.3-flash", monkeypatch)
    assert (resp.status_code, len(upstream.calls)) == (400, 1)


def test_failover_retries_once_only(monkeypatch) -> None:
    registry = EndpointRegistry()
    for job in ("a", "b", "c"):
        _add(registry, job, "glm-5.3-flash", load={"a": 0, "b": 1, "c": 2}[job])
    upstream = _Upstream(statuses={"host-a": 502, "host-b": 502})
    resp = _post(registry, upstream, "glm-5.3-flash", monkeypatch)
    assert (resp.status_code, len(upstream.calls)) == (502, 2)


def test_no_alternate_returns_the_original_failure(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash")
    upstream = _Upstream(statuses={"host-a": 503})
    resp = _post(registry, upstream, "glm-5.3-flash", monkeypatch)
    assert (resp.status_code, len(upstream.calls)) == (503, 1)


def test_unopened_stream_releases_its_request_count(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash")

    async def refuse(self, request, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)
    resp = _client(registry).post(
        "/umrs/llm/v1/chat/completions",
        json={"model": "glm-5.3-flash", "stream": True, **CHAT},
        headers=HEADERS,
    )
    assert (resp.status_code, registry.in_flight("a")) == (502, 0)


def test_failed_stream_reply_is_retried_and_released(monkeypatch) -> None:
    registry = EndpointRegistry()
    _add(registry, "a", "glm-5.3-flash", load=0)
    _add(registry, "b", "glm-5.3-flash", load=1)

    async def send(self, request, **kwargs):
        host = str(request.url).split("//", 1)[1].split(":", 1)[0]
        code = 502 if host == "host-a" else 200
        return httpx.Response(code, content=b"data: {}\n\n", headers={"content-type": "text/event-stream"})

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    resp = _client(registry).post(
        "/umrs/llm/v1/chat/completions",
        json={"model": "svc-chat", "stream": True, **CHAT},
        headers=HEADERS,
    )
    assert (resp.status_code, registry.in_flight("a"), registry.in_flight("b")) == (200, 0, 1)
