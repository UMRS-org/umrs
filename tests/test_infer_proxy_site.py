"""``umrs-infer``'s ``site`` source: EndpointRegistry rows from a sandbox
(ADR-092 slice 8b, the D-63 venue leg).

The lease's own project's READY rows only; the site's service key injected
by the proxy (never the sandbox's); the model rewritten to the served name;
a public-passthrough row refused for a sensitive project; the pushed-source
path unchanged (its public-host pin still refuses every internal upstream).
Fakes as the proxy tests use them: a fixed lease, a MockTransport upstream,
the real in-memory registry.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from test_infer_proxy import _app, _lease, _request, _stream_response
from umrs.infer_proxy.app import create_app
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.shared.site_source import SITE_SOURCE_ID, SITE_SOURCE_NAME
from umrs.venues.frontend_service import FrontendVenueService

URL = "/s/sess-1/site/v1/chat/completions"


def _registry(**rows):
    registry = EndpointRegistry()
    for job_id, row in (rows or {"ep-1": {}}).items():
        spec = {
            "project_id": "proj-a",
            "model": "Qwen/Qwen3-8B",
            "host": "10.4.0.7",
            "port": 4000,
            "metadata": {"served_model_name": "qwen3-8b"},
            **row,
        }
        status = spec.pop("status", EndpointStatus.READY)
        registry.register(job_id=job_id, backend="slurm", **spec)
        registry.update_status(job_id, status)
    return registry


def _site_app(monkeypatch, *, registry=None, lease=None, pool=None, respond=None):
    lease = lease or _lease(
        source_ids=("src-a", SITE_SOURCE_ID), project="proj-a", owner_id="alice"
    )
    monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _pool, _token: lease)
    monkeypatch.setenv("GATEWAY_LITELLM_SERVICE_KEY", "site-service-key")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(
            url=str(request.url), headers=dict(request.headers), body=json.loads(request.read())
        )
        return respond(request) if respond else _stream_response()

    broker = SimpleNamespace(audit=[])
    broker._audit_log = lambda *a: broker.audit.append(a)
    app = create_app(
        umrs_db_pool=pool,
        unseal_privkey="unused",
        sealed_store=SimpleNamespace(fetch_for_unseal=lambda _id: None),
        broker=broker,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        site_registry=registry if registry is not None else _registry(),
    )
    return app, seen, broker


def _post(app, model="qwen3-8b", url=URL, **headers):
    return _request(
        app,
        "POST",
        url,
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer tok-session", **headers},
    )


def test_a_site_model_is_served_by_the_projects_own_row(monkeypatch):
    app, seen, broker = _site_app(monkeypatch)
    res = _post(app, model="Qwen/Qwen3-8B", **{"x-api-key": "sandbox-tries", "cookie": "c=1"})
    assert res.status_code == 200, res.text
    assert {
        "url": seen["url"],
        "model": seen["body"]["model"],
        "auth": seen["headers"].get("authorization"),
        "leaked": [h for h in ("x-api-key", "cookie") if h in seen["headers"]],
        "identity": (seen["headers"]["x-project-id"], seen["headers"]["x-user-id"]),
        "audited": [a[1] for a in broker.audit],
    } == {
        "url": "http://10.4.0.7:4000/v1/chat/completions",
        "model": "qwen3-8b",  # the served name (ADR-068 rewrite)
        "auth": "Bearer site-service-key",  # the site's key, injected here only
        "leaked": [],
        "identity": ("proj-a", "alice"),
        "audited": ["infer"],
    }


def test_another_projects_row_is_never_reachable(monkeypatch):
    registry = _registry(**{"ep-9": {"project_id": "proj-b", "model": "secret-model"}})
    app, seen, _broker = _site_app(monkeypatch, registry=registry)
    res = _post(app, model="secret-model")
    assert (res.status_code, res.json()["error"], seen) == (404, "site_model_not_found", {})


def test_a_registry_answer_from_another_project_is_still_refused(monkeypatch):
    """Defence in depth: a selection that names another project's row (a
    registry bug, a shared name) is refused by the proxy's own scope check."""
    (foreign,) = _registry(**{"ep-9": {"project_id": "proj-b"}}).list_all(project_id="proj-b")
    leaky = SimpleNamespace(
        select_endpoint=lambda *_a, **_k: foreign, list_all=lambda **_k: [foreign]
    )
    app, seen, _broker = _site_app(monkeypatch, registry=leaky)
    res = _post(app, model="qwen3-8b")
    assert (res.status_code, res.json()["error"], seen) == (404, "site_model_not_found", {})


def test_an_upstream_redirect_is_never_followed(monkeypatch):
    def redirect(_request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    app, _seen, _broker = _site_app(monkeypatch, respond=redirect)
    res = _post(app)
    assert (res.status_code, res.json()["error"]) == (502, "redirect_offhost")


def test_a_row_that_is_not_ready_is_not_served(monkeypatch):
    registry = _registry(ep_2={"status": EndpointStatus.PROVISIONING, "model": "late"})
    app, _seen, _broker = _site_app(monkeypatch, registry=registry)
    assert _post(app, model="late").status_code == 404


def test_a_lease_without_the_site_source_is_refused(monkeypatch):
    app, seen, _broker = _site_app(monkeypatch, lease=_lease(project="proj-a"))
    res = _post(app)
    assert (res.status_code, res.json()["error"], seen) == (403, "source_not_in_lease", {})


def test_a_lease_without_a_project_reaches_no_row(monkeypatch):
    lease = _lease(source_ids=(SITE_SOURCE_ID,), default_source_id=SITE_SOURCE_ID)
    app, seen, _broker = _site_app(monkeypatch, lease=lease)
    res = _post(app)
    assert (res.status_code, res.json()["error"], seen) == (403, "site_scope_missing", {})


def test_only_the_model_verbs_pass(monkeypatch):
    app, seen, _broker = _site_app(monkeypatch)
    res = _post(app, url="/s/sess-1/site/v1/fine_tuning/jobs")
    assert (res.status_code, res.json()["error"], seen) == (403, "path_denied", {})
    listing = _request(
        app, "GET", "/s/sess-1/site/v1/models", headers={"authorization": "Bearer tok-session"}
    )
    assert listing.json()["data"] == [{"id": "qwen3-8b", "object": "model"}]


def test_a_public_passthrough_row_is_refused_for_a_sensitive_project(monkeypatch):
    registry = _registry(
        **{"ep-1": {}, "ep-3": {"model": "gpt-x", "metadata": {"locality": "public"}}}
    )
    monkeypatch.setattr("umrs.infer_proxy.app.data_class_for_project", lambda _p, _pr: "sensitive")
    app, seen, _broker = _site_app(monkeypatch, registry=registry)
    res = _post(app, model="gpt-x")
    assert (res.status_code, res.json()["error"], seen) == (403, "source_class_refused", {})
    # the project's self-deployed row is its own route even when sensitive
    assert _post(app, model="qwen3-8b").status_code == 200


def test_the_wrong_session_or_token_is_refused_before_the_registry(monkeypatch):
    app, seen, _broker = _site_app(monkeypatch)
    res = _post(app, url="/s/other/site/v1/chat/completions")
    assert (res.status_code, res.json()["error"], seen) == (401, "sid_mismatch", {})


def test_the_pushed_source_path_still_refuses_internal_hosts(monkeypatch):
    """Unchanged: a pushed source's upstream must be global unicast; only a
    registry row of the lease's project reaches an internal listener."""
    app, seen, _broker = _app(monkeypatch, provider_kind="openai_compat")
    monkeypatch.setattr("socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("10.4.0.7", 0))])
    res = _request(
        app,
        "POST",
        "/s/sess-1/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"authorization": "Bearer tok-session"},
    )
    assert res.status_code == 470 and res.json()["error"] == "host_denied"


def test_the_venue_start_resolves_site_without_a_pushed_credential():
    service = FrontendVenueService.__new__(FrontendVenueService)
    service._sealed = SimpleNamespace(get_meta=lambda _o, _n: None)
    names, ids, missing = service._resolve_sources("alice", [SITE_SOURCE_NAME, "gone"])
    assert (names, ids, missing) == ((SITE_SOURCE_NAME,), (SITE_SOURCE_ID,), ["gone"])


def test_the_site_source_id_is_a_fixed_uuid():
    import uuid

    parsed = uuid.UUID(SITE_SOURCE_ID)  # fits the leases' UUID[] column
    assert (str(parsed), parsed.version) == (SITE_SOURCE_ID, 5)


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
