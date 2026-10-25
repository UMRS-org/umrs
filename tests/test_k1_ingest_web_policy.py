"""Knowledge URL ingest follows the ADR-099 web policy (wave 79, 2026-10-25).

Before this wave the front's ``allowed_domains`` alone bounded URL and git
ingest, so a project or a user whose agent may not browse could still have
the server fetch any site the front lists (every front ships ``"*"``). Now
the requester's effective ADR-099 policy narrows the front list: ``none``
refuses, ``allowlist`` keeps only the ``web.domains`` the agent may browse
(squid ``dstdomain`` matching), ``public`` leaves the front list. The
network guards of ``egress_fetch`` stay on top in every mode.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.knowledge import egress_fetch, ingest_allowlist, ingest_web_policy
from umrs.knowledge.ingest_web_policy import (
    IngestRequester,
    intersect_patterns,
    squid_patterns,
)
from umrs.knowledge.sources import EgressDisabledError, SourceResolver
from umrs.session_policy import PolicyRule, PolicyRuleStore

from r4_contracts.errors import ERROR_CODE_HEADER

SITE = "test-front"
ME = IngestRequester(user_id="u-1", project_id="proj-a")


@pytest.fixture(autouse=True)
def _fresh_log_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ingest_allowlist, "_REPORTED", set())


def _site(domains: Any, name: str = SITE) -> Any:
    return SimpleNamespace(name=name, extra={"knowledge_ingest": {"allowed_domains": domains}})


def _store(*rules: tuple[str, str, str, dict[str, Any]]) -> PolicyRuleStore:
    """An in-memory rule store: ``(scope, subject, layer, doc)`` rows here."""
    store = PolicyRuleStore()
    for scope, subject, layer, doc in rules:
        store.put(PolicyRule(venue=SITE, scope=scope, subject=subject, layer=layer, doc=doc))
    return store


def _owner(doc: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    return ("project", "proj-a", "owner", doc)


def _resolver(store: Any, site: Any = None, requester: Any = ME) -> SourceResolver:
    return SourceResolver(
        site_config=site if site is not None else _site(["*"]),
        requester=requester,
        policy_store=store,
    )


# ── none: the project or the user may not browse, so nothing is fetched ──


def test_a_project_owner_none_refuses_url_ingest_naming_the_layer() -> None:
    store = _store(_owner({"web": {"allow": ["none"]}}))
    with pytest.raises(EgressDisabledError) as refused:
        _resolver(store).resolve("http_url", "https://arxiv.org/abs/1")
    text = str(refused.value)
    assert "web access is off for project 'proj-a'" in text
    assert "the project owner's rule (project:owner)" in text
    assert "r4 knowledge allowlist" in text


def test_a_user_none_binds_that_user_only() -> None:
    store = _store(("user", "u-1", "admin", {"web": {"allow": ["none"]}}))
    with pytest.raises(EgressDisabledError, match="an admin rule for this user"):
        _resolver(store).resolve("git_repo", "https://github.com/a/b.git")
    other = IngestRequester(user_id="u-2", project_id="proj-a")
    resolved = _resolver(store, requester=other).resolve("git_repo", "https://github.com/a/b.git")
    assert resolved.allow_domains == ["*"]


def test_the_project_rule_binds_that_project_only() -> None:
    store = _store(("project", "proj-a", "admin", {"web": {"allow": ["none"]}}))
    elsewhere = IngestRequester(user_id="u-1", project_id="proj-b")
    assert (
        _resolver(store, requester=elsewhere)
        .resolve("http_url", "https://arxiv.org/abs/1")
        .resolved
    )
    with pytest.raises(EgressDisabledError, match="an admin rule for this project"):
        _resolver(store).resolve("http_url", "https://arxiv.org/abs/1")


# ── allowlist: the domains the agent may browse, intersected with the front ──


def test_an_allowlist_cap_keeps_only_the_domains_the_agent_may_browse() -> None:
    store = _store(_owner({"web": {"domains": [".arxiv.org", "doi.org"]}}))
    resolved = _resolver(store).resolve("http_url", "https://export.arxiv.org/abs/1")
    assert resolved.allow_domains == ["arxiv.org", "*.arxiv.org", "doi.org"]
    with pytest.raises(EgressDisabledError) as refused:
        _resolver(store).resolve("http_url", "https://any.example/x")
    text = str(refused.value)
    assert "'any.example' is outside the web policy of project 'proj-a'" in text
    assert ".arxiv.org, doi.org" in text and "project:owner" in text


def test_the_cap_matches_like_the_browse_guard() -> None:
    """``doi.org`` is that host only; ``.arxiv.org`` is it and subdomains."""
    store = _store(_owner({"web": {"domains": [".arxiv.org", "doi.org"]}}))
    for url in ("https://arxiv.org/x", "https://a.b.arxiv.org/x", "https://doi.org/10.1/x"):
        assert _resolver(store).resolve("http_url", url).resolved, url
    for url in ("https://dx.doi.org/10.1/x", "https://notarxiv.org/x"):
        with pytest.raises(EgressDisabledError, match="outside the web policy"):
            _resolver(store).resolve("http_url", url)


def test_the_front_list_still_bounds_an_allowlist_policy() -> None:
    store = _store(_owner({"web": {"domains": [".nature.com", "arxiv.org"]}}))
    site = _site(["*.nature.com"])
    resolved = _resolver(store, site).resolve("http_url", "https://www.nature.com/a")
    assert resolved.allow_domains == ["*.nature.com"]
    with pytest.raises(EgressDisabledError, match="not on this site's ingest allowlist"):
        _resolver(store, site).resolve("http_url", "https://arxiv.org/abs/1")


def test_a_caller_list_still_only_narrows() -> None:
    store = _store(_owner({"web": {"domains": [".arxiv.org"]}}))
    resolved = _resolver(store).resolve(
        "http_url", "https://arxiv.org/abs/1", allow_domains=["arxiv.org", "evil.example"]
    )
    assert resolved.allow_domains == ["arxiv.org"]


def test_allowlist_without_a_cap_fetches_only_the_named_sites() -> None:
    """An admin left ``allowlist`` but no domain cap: the person names the
    sites per request, as at a session open; redirects stay on them."""
    store = _store(("site", "", "admin", {"web": {"allow": ["none", "allowlist"]}}))
    resolved = _resolver(store).resolve("http_url", "https://arxiv.org/abs/1")
    assert resolved.allow_domains == ["arxiv.org"]
    named = _resolver(store).resolve(
        "http_url", "https://doi.org/10.1/x", allow_domains=["doi.org", "*.nature.com"]
    )
    assert named.allow_domains == ["doi.org", "*.nature.com"]
    with pytest.raises(EgressDisabledError, match="name the site in allow_domains"):
        _resolver(store).resolve("http_url", "https://any.example/x", allow_domains=["doi.org"])


# ── public, no rules, no requester: the front list as before ──────────────


def test_public_and_no_rules_leave_the_front_list() -> None:
    assert _resolver(_store()).resolve("http_url", "https://any.example/x").allow_domains == ["*"]
    public = _store(_owner({"web": {"allow": ["none", "allowlist", "public"]}}))
    assert _resolver(public).resolve("http_url", "https://any.example/x").allow_domains == ["*"]


def test_a_resolver_without_a_requester_reads_no_rules() -> None:
    class _Unreadable(PolicyRuleStore):
        def applicable(self, **_kwargs: Any) -> list[PolicyRule]:
            raise AssertionError("no requester: no rule read")

    resolver = SourceResolver(site_config=_site(["*"]), policy_store=_Unreadable())
    assert resolver.resolve("http_url", "https://any.example/x").allow_domains == ["*"]


# ── fail closed ────────────────────────────────────────────────────────────


def test_unreadable_rules_refuse_the_ingest() -> None:
    class _Down(PolicyRuleStore):
        def applicable(self, **_kwargs: Any) -> list[PolicyRule]:
            raise ConnectionError("db down")

    with pytest.raises(EgressDisabledError, match="unreadable .ConnectionError."):
        _resolver(_Down()).resolve("http_url", "https://arxiv.org/abs/1")


def test_an_unknown_site_name_refuses_rather_than_reading_all_site_rows() -> None:
    with pytest.raises(EgressDisabledError, match="site name is unknown"):
        _resolver(_store(), _site(["*"], name="")).resolve("http_url", "https://arxiv.org/a")


# ── the job passes its owner and project ──────────────────────────────────


def test_the_ingest_job_applies_its_owners_policy(monkeypatch) -> None:
    from umrs.jobs import knowledge_adapters

    store = _store(_owner({"web": {"allow": ["none"]}}))
    monkeypatch.setattr(knowledge_adapters, "_ingest_site_config", lambda: _site(["*"]))
    monkeypatch.setattr(ingest_web_policy, "default_rule_store", lambda: store)
    job = SimpleNamespace(job_id="j", user_id="u-1", spec={})
    result = knowledge_adapters.KnowledgeIngestResult()
    with pytest.raises(EgressDisabledError, match="web access is off"):
        knowledge_adapters._ingest_one_source(
            job, None, None, "b", "proj-a", "http_url", "https://arxiv.org/a", None, result, [], []
        )


# ── the fetcher re-checks every redirect hop against the narrowed list ────


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — http.server's name
        if self.path == "/10.1/x":
            self.send_response(302)
            self.send_header("Location", "http://www.nature.com/articles/x")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", "5")
        self.end_headers()
        self.wfile.write(b"paper")

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture()
def fake_net(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    real_open = egress_fetch._open_connection
    monkeypatch.setattr(egress_fetch, "_resolve_host", lambda _h, _p: ["93.184.215.14"])
    monkeypatch.setattr(
        egress_fetch,
        "_open_connection",
        lambda _s, host, _p, _a, timeout: real_open("http", host, port, "127.0.0.1", timeout),
    )
    yield
    httpd.shutdown()
    httpd.server_close()


def test_a_redirect_outside_the_policy_is_refused(fake_net) -> None:
    capped = _store(_owner({"web": {"domains": [".doi.org"]}}))
    resolved = _resolver(capped).resolve("http_url", "http://doi.org/10.1/x")
    result = egress_fetch.fetch_url(resolved.uri, allow_domains=resolved.allow_domains)
    assert result.content is None and "www.nature.com" in result.error
    public = _resolver(_store()).resolve("http_url", "http://doi.org/10.1/x")
    assert (
        egress_fetch.fetch_url(public.uri, allow_domains=public.allow_domains).content == b"paper"
    )


# ── pattern arithmetic ────────────────────────────────────────────────────


def test_squid_entries_become_fetcher_patterns() -> None:
    assert squid_patterns([".arxiv.org", "doi.org", "DOI.org."]) == [
        "arxiv.org",
        "*.arxiv.org",
        "doi.org",
    ]


def test_the_intersection_keeps_the_narrower_entry() -> None:
    assert intersect_patterns(["*"], ["arxiv.org"]) == ["arxiv.org"]
    assert intersect_patterns(["*.org"], ["*.arxiv.org", "x.com"]) == ["*.arxiv.org"]
    assert intersect_patterns(["lab.x.org:8443"], ["*.x.org"]) == ["lab.x.org:8443"]
    assert intersect_patterns(["*.nature.com"], ["nature.com"]) == []
    assert intersect_patterns(["*.a.org"], ["*.ba.org"]) == []


# ── the routes: GET …/ingest/allowlist shows it, POST …/ingest refuses early ──


def _client(monkeypatch, store: Any, config: Any = None) -> TestClient:
    from umrs.api import knowledge_v2
    from umrs.api.files import get_project_id

    config = config if config is not None else _site(["*"])
    monkeypatch.setattr(ingest_allowlist, "load_site_config", lambda: config)
    app = FastAPI()
    app.include_router(knowledge_v2.router)
    app.state.session_policy_store = store
    app.dependency_overrides[get_project_id] = lambda: "proj-a"
    return TestClient(app, raise_server_exceptions=False)


def test_the_allowlist_route_shows_the_callers_effective_policy(monkeypatch) -> None:
    store = _store(_owner({"web": {"domains": [".arxiv.org"]}}))
    client = _client(monkeypatch, store)
    body = client.get("/umrs/k2/ingest/allowlist", headers={"x-user-id": "u-1"}).json()
    web = body["web_policy"]
    assert web["mode"] == "allowlist" and web["domains"] == [".arxiv.org"]
    assert web["domains_set_by"] == "project:owner" and web["project_id"] == "proj-a"
    assert web["entries"] == ["arxiv.org", "*.arxiv.org"]
    assert "both this list and the site list" in web["summary"]
    assert body["entries"] == ["*"]  # the front list, unchanged


def test_the_allowlist_check_applies_the_policy(monkeypatch) -> None:
    client = _client(monkeypatch, _store(_owner({"web": {"domains": [".arxiv.org"]}})))
    check = client.get(
        "/umrs/k2/ingest/allowlist",
        params={"check": "https://any.example/x"},
        headers={"x-user-id": "u-1"},
    ).json()["check"]
    assert check["allowed"] is False
    assert "outside the web policy" in check["reason"]


def test_the_allowlist_route_reports_none_and_unreadable(monkeypatch) -> None:
    off = _client(monkeypatch, _store(_owner({"web": {"allow": ["none"]}})))
    web = off.get("/umrs/k2/ingest/allowlist").json()["web_policy"]
    assert web["mode"] == "none" and web["entries"] == []
    assert "web access is off" in web["summary"]

    class _Down(PolicyRuleStore):
        def applicable(self, **_kwargs: Any) -> list[PolicyRule]:
            raise ConnectionError("db down")

    down = (
        _client(monkeypatch, _Down())
        .get("/umrs/k2/ingest/allowlist", params={"check": "arxiv.org"})
        .json()
    )
    assert "unreadable" in down["web_policy"]["error"]
    assert down["check"]["allowed"] is False


def _queueing_client(monkeypatch, store: Any) -> tuple[TestClient, list[dict[str, Any]]]:
    from umrs.api import knowledge_v2

    specs: list[dict[str, Any]] = []

    class _Runtime:
        async def submit_job_sync(self, **kwargs: Any) -> str:
            specs.append(kwargs["spec"])
            return "job-w"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    monkeypatch.setattr(knowledge_v2, "_resolve_ingest_target", lambda *_a, **_k: "g-w")
    monkeypatch.setattr("umrs.api.governance.guard_ingest", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.enforce_graph_mutation_authorization",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.mark_proposal_executed", lambda *_a, **_k: None
    )
    return _client(monkeypatch, store), specs


def _ingest(client: TestClient, uri: str) -> Any:
    return client.post(
        "/umrs/k2/ingest",
        json={"sources": [{"kind": "http_url", "uri": uri}]},
        headers={"x-user-id": "u-1"},
    )


def test_the_ingest_route_refuses_a_policy_none_before_queueing(monkeypatch) -> None:
    client, specs = _queueing_client(monkeypatch, _store(_owner({"web": {"allow": ["none"]}})))
    resp = _ingest(client, "https://arxiv.org/a")
    assert resp.status_code == 403, resp.text
    assert resp.headers[ERROR_CODE_HEADER] == "sessionpolicy.widening_refused"
    assert "project:owner" in resp.text
    assert specs == []


def test_the_ingest_route_refuses_a_host_outside_the_cap(monkeypatch) -> None:
    client, specs = _queueing_client(monkeypatch, _store(_owner({"web": {"domains": ["doi.org"]}})))
    assert _ingest(client, "https://any.example/x").status_code == 403
    assert _ingest(client, "https://doi.org/10.1/x").status_code in (200, 202)
    assert len(specs) == 1


def test_the_ingest_route_fails_closed_when_rules_are_unreadable(monkeypatch) -> None:
    class _Down(PolicyRuleStore):
        def applicable(self, **_kwargs: Any) -> list[PolicyRule]:
            raise ConnectionError("db down")

    client, specs = _queueing_client(monkeypatch, _Down())
    resp = _ingest(client, "https://arxiv.org/a")
    assert resp.status_code == 503
    assert resp.headers[ERROR_CODE_HEADER] == "sessionpolicy.unavailable"
    assert specs == []
