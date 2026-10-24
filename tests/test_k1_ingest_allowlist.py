"""Knowledge-ingest allowlist: ``@recommended``, ``*``, the read route and the
refusal text (2026-10-25).

K1 part A made URL and repo ingest fail-closed until a front's site.yaml
sets ``extra.knowledge_ingest.allowed_domains``. These tests pin what the
entries mean (``@recommended`` expands to the packaged list, ``*`` admits
every public host and nothing else changes), that every network guard stays
under ``*``, the route users read the list from, and refusals that name the
host and the remedy instead of printing the list.

No test touches the internet: the fetch tests run a local HTTP server and a
fake resolver (as ``test_k1_governance_egress.py`` does).
"""

from __future__ import annotations

import logging
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.knowledge import egress_fetch, ingest_allowlist
from umrs.knowledge.ingest_allowlist import effective_site_allowlist, recommended_domains
from umrs.knowledge.sources import EgressDisabledError, SourceResolver

UMRS = Path(__file__).resolve().parents[1]
FRONTS = ("rccs-front", "ai4s-front", "bdrt-front", "octopus-front")


@pytest.fixture(autouse=True)
def _fresh_log_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees the once-per-process log lines afresh."""
    monkeypatch.setattr(ingest_allowlist, "_REPORTED", set())


def _site(domains: Any, name: str = "test-front") -> Any:
    return SimpleNamespace(name=name, extra={"knowledge_ingest": {"allowed_domains": domains}})


# ── the packaged recommended list ─────────────────────────────────────────


def test_the_recommended_list_loads_from_the_installed_package() -> None:
    from importlib.resources import files

    import umrs.knowledge

    packaged = files("umrs.knowledge").joinpath(ingest_allowlist.RECOMMENDED_FILE)
    assert Path(str(packaged)).parent == Path(umrs.knowledge.__file__).parent
    domains = recommended_domains()
    for needed in (
        "doi.org",
        "*.doi.org",
        "arxiv.org",
        "*.nature.com",
        "github.com",
        "www.riken.jp",
    ):
        assert needed in domains, needed


def test_a_non_editable_install_ships_the_list() -> None:
    """The images install editable; a wheel needs the package-data line."""
    text = (UMRS / "src" / "pyproject.toml").read_text()
    block = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
    assert '"umrs.knowledge"' in block and ingest_allowlist.RECOMMENDED_FILE in block


def test_every_recommended_entry_is_a_plain_host_pattern() -> None:
    pattern = re.compile(r"^(\*\.)?[a-z0-9-]+(\.[a-z0-9-]+)+$")
    domains = recommended_domains()
    assert len(domains) == len(set(domains))
    for entry in domains:
        assert pattern.match(entry), entry
    for host in ("pastebin.com", "dropbox.com", "drive.google.com"):
        assert not egress_fetch.domain_allowed(host, list(domains)), host


def test_doi_links_reach_the_publishers() -> None:
    """A DOI redirects to the publisher; every hop is checked, so each one
    must be on the list."""
    domains = list(recommended_domains())
    for host in (
        "doi.org",
        "dx.doi.org",
        "www.nature.com",
        "linkinghub.elsevier.com",
        "www.sciencedirect.com",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "pubs.acs.org",
        "journals.aps.org",
        "www.jstage.jst.go.jp",
    ):
        assert egress_fetch.domain_allowed(host, domains), host


def test_the_recommended_list_names_no_riken_intranet_host() -> None:
    """Review 2026-10-25: ``*.riken.jp`` admitted every RIKEN intranet name,
    and RIKEN's 134.160.0.0/16 is global unicast, so the address check let
    them through (data0, the front itself, included). Only RIKEN's public
    sites are listed."""
    domains = list(recommended_domains())
    assert "*.riken.jp" not in domains and "riken.jp" not in domains
    for host in ("data0.ai.r-ccs.riken.jp", "login.rikyu.r-ccs.riken.jp", "octopus.riken.jp"):
        assert not egress_fetch.domain_allowed(host, domains), host
    for host in ("www.riken.jp", "www.r-ccs.riken.jp"):
        assert egress_fetch.domain_allowed(host, domains), host


def test_the_recommended_header_names_the_doi_chains_that_fail() -> None:
    """IEEE and Springer books send https DOIs to http, RSC needs more than
    5 redirects: the guards stay, the header says so (review 2026-10-25)."""
    text = (UMRS / "src" / "umrs" / "knowledge" / ingest_allowlist.RECOMMENDED_FILE).read_text()
    for word in ("IEEE", "Springer books", "RSC", "https article URL"):
        assert word in text, word


# ── expansion: one place, refused entries dropped by name ─────────────────


def test_recommended_expands_and_combines_with_other_entries() -> None:
    entries = effective_site_allowlist(["@recommended", "lab.example.org", "arxiv.org"])
    assert entries[: len(recommended_domains())] == list(recommended_domains())
    assert entries[-1] == "lab.example.org"
    assert entries.count("arxiv.org") == 1


def test_an_unknown_token_is_refused_by_name_and_the_rest_apply(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=ingest_allowlist.__name__)
    assert effective_site_allowlist(["@recomended", "arxiv.org"]) == ["arxiv.org"]
    assert "'@recomended' refused" in caplog.text


def test_a_wildcard_port_is_refused_not_opened(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=ingest_allowlist.__name__)
    assert effective_site_allowlist(["*:8443", "arxiv.org"]) == ["arxiv.org"]
    assert "'*:8443' refused" in caplog.text
    # Handed to the fetcher directly it admits nothing either.
    assert not egress_fetch.domain_allowed("a.example.org", ["*:8443"])
    assert not egress_fetch.port_allowed("a.example.org", 8443, ["*", "*:8443"])


def test_allow_all_warns_once_at_load(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=ingest_allowlist.__name__)
    for _ in range(3):
        assert effective_site_allowlist(["*"]) == ["*"]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "every public web site" in warnings[0].getMessage()


@pytest.mark.parametrize("entry", ["*.", "*..", " * "])
def test_a_star_with_trailing_dots_is_allow_all_and_says_so(caplog, entry) -> None:
    """The fetcher strips trailing dots, so ``*.`` admits every host: it is
    reported as ``*`` (warning, ``allow_all``), never as one odd entry
    (review 2026-10-25)."""
    caplog.set_level(logging.WARNING, logger=ingest_allowlist.__name__)
    assert egress_fetch.domain_allowed("evil.example", [entry])
    assert effective_site_allowlist([entry, "arxiv.org"]) == ["*", "arxiv.org"]
    assert "every public web site" in caplog.text
    assert ingest_allowlist.allowlist_view(_site([entry]))["allow_all"] is True


def test_a_star_with_trailing_dots_and_a_port_is_refused(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=ingest_allowlist.__name__)
    assert effective_site_allowlist(["*.:8443", "arxiv.org"]) == ["arxiv.org"]
    assert "'*.:8443' refused" in caplog.text


def test_a_single_string_is_one_entry_and_junk_is_empty() -> None:
    assert effective_site_allowlist("*") == ["*"]
    assert effective_site_allowlist(None) == []
    assert effective_site_allowlist({"a": 1}) == []


# ── "*": every host name, never an address or a port ──────────────────────


def test_allow_all_admits_any_host_on_80_and_443_only() -> None:
    assert egress_fetch.domain_allowed("anything.example", ["*"])
    assert egress_fetch.port_allowed("anything.example", 443, ["*"])
    assert not egress_fetch.port_allowed("anything.example", 8443, ["*"])
    assert egress_fetch.port_allowed("docs.example.org", 8443, ["*", "docs.example.org:8443"])


def test_with_a_site_star_a_caller_list_narrows_to_itself() -> None:
    narrow = egress_fetch.intersect_allowlists
    assert narrow(None, ["*"]) == ["*"]
    assert narrow(["lab.example.org", "*.uni.example"], ["*"]) == [
        "lab.example.org",
        "*.uni.example",
    ]
    # A caller cannot open a port the site did not name.
    assert narrow(["lab.example.org:8443"], ["*"]) == []
    assert narrow(["lab.example.org:8443"], ["*", "lab.example.org:8443"]) == [
        "lab.example.org:8443"
    ]
    # And a caller "*" never widens a named list.
    assert narrow(["*"], ["arxiv.org"]) == []


def test_the_resolver_uses_the_expanded_list() -> None:
    resolver = SourceResolver(site_config=_site(["@recommended"]))
    resolved = resolver.resolve("http_url", "https://doi.org/10.1038/x")
    assert "*.nature.com" in resolved.allow_domains
    assert "@recommended" not in resolved.allow_domains
    star = SourceResolver(site_config=_site(["*"])).resolve(
        "git_repo", "https://code.example.edu/lab/tool.git"
    )
    assert star.allow_domains == ["*"]


def test_the_resolver_carries_the_site_denied_networks() -> None:
    config = _site(["*"])
    config.extra["knowledge_ingest"]["denied_networks"] = ["134.160.0.0/16", "2001:db8::/32"]
    resolved = SourceResolver(site_config=config).resolve("http_url", "https://any.example/x")
    assert resolved.deny_networks == ["134.160.0.0/16", "2001:db8::/32"]


def test_the_ingest_job_hands_the_denied_networks_to_the_fetcher(monkeypatch) -> None:
    from umrs.jobs import knowledge_adapters
    from umrs.knowledge.sources import ResolvedSource

    seen: dict[str, Any] = {}

    def _fetch(url: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return egress_fetch.FetchResult(url=url, content=None, status_code=None, error="x")

    monkeypatch.setattr(egress_fetch, "fetch_url", _fetch)
    resolved = ResolvedSource(
        kind="http_url",
        uri="https://any.example/x",
        allow_egress=True,
        allow_domains=["*"],
        resolved=True,
        deny_networks=["134.160.0.0/16"],
    )
    result = knowledge_adapters.KnowledgeIngestResult()
    knowledge_adapters._fetch_and_store_remote(
        "job", "att", None, "proj", resolved, "art", result, [], []
    )
    assert seen == {"allow_domains": ["*"], "deny_networks": ["134.160.0.0/16"]}
    assert result.failed_items[0]["error"] == "x"


def test_a_bad_denied_network_is_logged_and_the_rest_apply(caplog) -> None:
    caplog.set_level(logging.ERROR, logger=ingest_allowlist.__name__)
    config = _site(["*"])
    config.extra["knowledge_ingest"]["denied_networks"] = ["134.160.0.0/16", "intranet"]
    assert ingest_allowlist.site_denied_networks(config) == ["134.160.0.0/16"]
    assert "'intranet' is not a network" in caplog.text


# ── refusals name the host and the remedy, not the list ───────────────────


def test_a_host_off_the_list_is_refused_at_resolve_with_the_remedy() -> None:
    resolver = SourceResolver(site_config=_site(["@recommended"]))
    with pytest.raises(EgressDisabledError) as refused:
        resolver.resolve("http_url", "https://blog.example.net/post")
    text = str(refused.value)
    assert 'domain "blog.example.net" is not on this site\'s ingest allowlist' in text
    assert "r4 knowledge allowlist" in text
    assert "extra.knowledge_ingest.allowed_domains" in text and '"*"' in text
    assert "arxiv.org" not in text  # the list itself is not printed


def test_a_caller_list_outside_the_site_list_says_so() -> None:
    resolver = SourceResolver(site_config=_site(["arxiv.org"]))
    with pytest.raises(EgressDisabledError) as refused:
        resolver.resolve("http_url", "https://arxiv.org/abs/1", allow_domains=["evil.example"])
    assert "allow_domains admits none" in str(refused.value)


# ── the fetcher under "*" and "@recommended" (local server, fake DNS) ─────


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}

    def do_GET(self) -> None:  # noqa: N802 — http.server's name
        status, headers, body = self.routes.get(self.path, (404, {}, b"missing"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return


_PUBLIC = "93.184.215.14"
_RIKEN_INTRANET = "134.160.189.136"  # global unicast, inside RIKEN
_DNS = {
    "doi.org": [_PUBLIC],
    "www.nature.com": [_PUBLIC],
    "any.example": [_PUBLIC],
    "intranet.example": ["10.1.2.3"],
    "data0.ai.r-ccs.riken.jp": [_RIKEN_INTRANET],
    "mixed.example": [_PUBLIC, _RIKEN_INTRANET],
}
_DENY = ["134.160.0.0/16"]


@pytest.fixture()
def fake_net(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    real_open = egress_fetch._open_connection

    def _resolve(host: str, _port: int) -> list[str]:
        if host not in _DNS:
            raise OSError(f"unknown test host {host}")
        return list(_DNS[host])

    def _open(_scheme: str, host: str, _port: int, address: str, timeout: float) -> Any:
        assert address == _PUBLIC, f"connect attempted to {address}"
        return real_open("http", host, port, "127.0.0.1", timeout)

    monkeypatch.setattr(egress_fetch, "_resolve_host", _resolve)
    monkeypatch.setattr(egress_fetch, "_open_connection", _open)
    yield port
    httpd.shutdown()
    httpd.server_close()


def test_a_doi_redirect_to_the_publisher_is_followed(fake_net) -> None:
    _Handler.routes = {
        "/10.1038/x": (
            302,
            {"Location": "http://www.nature.com/articles/x", "Content-Length": "0"},
            b"",
        ),
        "/articles/x": (200, {"Content-Length": "5"}, b"paper"),
    }
    domains = effective_site_allowlist(["@recommended"])
    result = egress_fetch.fetch_url("http://doi.org/10.1038/x", allow_domains=domains)
    assert result.content == b"paper", result.error


def test_allow_all_fetches_a_public_host(fake_net) -> None:
    _Handler.routes = {"/doc": (200, {"Content-Length": "2"}, b"ok")}
    result = egress_fetch.fetch_url("http://any.example/doc", allow_domains=["*"])
    assert result.content == b"ok", result.error


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://intranet.example/x", "private"),
        ("http://127.0.0.1/x", "loopback"),
        ("http://169.254.169.254/latest", "metadata"),
        ("http://any.example:8080/doc", "port 8080"),
        ("ftp://any.example/x", "scheme"),
    ],
)
def test_allow_all_keeps_every_network_guard(fake_net, url, expected) -> None:
    result = egress_fetch.fetch_url(url, allow_domains=["*"])
    assert result.content is None and expected in result.error


def test_allow_all_still_checks_every_redirect_hop(fake_net) -> None:
    _Handler.routes = {
        "/start": (302, {"Location": "http://intranet.example/x", "Content-Length": "0"}, b""),
    }
    result = egress_fetch.fetch_url("http://any.example/start", allow_domains=["*"])
    assert result.content is None and "private" in result.error


@pytest.mark.parametrize(
    "url",
    [
        "http://data0.ai.r-ccs.riken.jp/admin",
        "http://mixed.example/doc",
        f"http://{_RIKEN_INTRANET}/admin",
        f"http://[::ffff:{_RIKEN_INTRANET}]/admin",
    ],
)
def test_a_denied_network_refuses_a_global_address(fake_net, url) -> None:
    """RIKEN's intranet is global unicast: under ``*`` the global check
    alone admitted it; the site's denied_networks refuse it on every
    resolved address (review 2026-10-25)."""
    assert egress_fetch.address_refusal(_RIKEN_INTRANET) == ""
    result = egress_fetch.fetch_url(url, allow_domains=["*"], deny_networks=_DENY)
    assert result.content is None and "denied network" in result.error, result.error


def test_a_redirect_into_a_denied_network_is_refused(fake_net) -> None:
    _Handler.routes = {
        "/start": (
            302,
            {"Location": "http://data0.ai.r-ccs.riken.jp/x", "Content-Length": "0"},
            b"",
        ),
    }
    result = egress_fetch.fetch_url(
        "http://any.example/start", allow_domains=["*"], deny_networks=_DENY
    )
    assert result.content is None and "denied network" in result.error


def test_denied_networks_leave_other_public_hosts_alone(fake_net) -> None:
    _Handler.routes = {"/doc": (200, {"Content-Length": "2"}, b"ok")}
    result = egress_fetch.fetch_url(
        "http://any.example/doc", allow_domains=["*"], deny_networks=_DENY + ["2001:db8::/32"]
    )
    assert result.content == b"ok", result.error


def test_an_unparseable_denied_network_refuses_the_fetch(fake_net) -> None:
    result = egress_fetch.fetch_url(
        "http://any.example/doc", allow_domains=["*"], deny_networks=["not-a-net"]
    )
    assert result.content is None and "unparseable denied network" in result.error


def test_the_fetch_refusal_names_the_host_and_the_remedy(fake_net) -> None:
    _Handler.routes = {
        "/start": (301, {"Location": "http://any.example/x", "Content-Length": "0"}, b""),
    }
    result = egress_fetch.fetch_url("http://doi.org/start", allow_domains=["doi.org", "arxiv.org"])
    assert result.content is None
    assert 'domain "any.example" is not on this site\'s ingest allowlist' in result.error
    assert "r4 knowledge allowlist" in result.error
    assert "arxiv.org" not in result.error


# ── GET /umrs/k2/ingest/allowlist ─────────────────────────────────────────


def _route_client(monkeypatch, config: Any, *, override: bool = True) -> Any:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import knowledge_v2
    from umrs.api.files import get_project_id

    monkeypatch.setattr(ingest_allowlist, "load_site_config", lambda: config)
    app = FastAPI()
    app.include_router(knowledge_v2.router)
    if override:
        app.dependency_overrides[get_project_id] = lambda: "proj-a"
    return TestClient(app, raise_server_exceptions=False)


def test_the_route_answers_the_expanded_list(monkeypatch) -> None:
    client = _route_client(monkeypatch, _site(["@recommended", "lab.example.org"], "ai4s"))
    body = client.get("/umrs/k2/ingest/allowlist").json()
    assert body["allow_all"] is False
    assert body["configured"] == ["@recommended", "lab.example.org"]
    assert "doi.org" in body["entries"] and body["entries"][-1] == "lab.example.org"
    assert body["site"] == "ai4s"
    assert "umrs/deploy/ai4s-front/site.yaml" in body["how_to_change"]
    assert "'*'" in body["how_to_change"]


def test_the_route_reports_allow_all_and_checks_a_url(monkeypatch) -> None:
    client = _route_client(monkeypatch, _site(["*"], "rccs-front"))
    body = client.get("/umrs/k2/ingest/allowlist", params={"check": "any.example/x"}).json()
    assert body["allow_all"] is True and body["entries"] == ["*"]
    assert body["check"]["allowed"] is True and body["check"]["host"] == "any.example"
    refused = client.get(
        "/umrs/k2/ingest/allowlist", params={"check": "https://any.example:8443/x"}
    ).json()["check"]
    assert refused["allowed"] is False and "port 8443" in refused["reason"]


def test_the_route_reports_the_denied_networks(monkeypatch) -> None:
    config = _site(["*"])
    config.extra["knowledge_ingest"]["denied_networks"] = ["134.160.0.0/16"]
    body = _route_client(monkeypatch, config).get("/umrs/k2/ingest/allowlist").json()
    assert body["denied_networks"] == ["134.160.0.0/16"]
    assert "denied_networks stay blocked" in body["how_to_change"]


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("ftp://arxiv.org/pub/x", "scheme 'ftp' is not fetched (http/https only)"),
        ("git+https://github.com/a/b.git", "scheme 'git+https' is not fetched"),
        ("git@github.com:a/b.git", "use the https clone URL"),
    ],
)
def test_check_refuses_what_the_fetcher_refuses_by_scheme(url, reason) -> None:
    """``--check`` said "allowed" for ftp:// and gave a port error for the
    scp git form; the fetcher takes http and https only (review 2026-10-25)."""
    verdict = ingest_allowlist.check_url(url, ["*"])
    assert verdict["allowed"] is False and reason in verdict["reason"]


def test_check_still_admits_http_https_and_bare_hosts() -> None:
    for url in ("https://arxiv.org/abs/1", "http://arxiv.org/abs/1", "arxiv.org/abs/1"):
        assert ingest_allowlist.check_url(url, ["arxiv.org"])["allowed"] is True, url
    assert ingest_allowlist.check_url("http://arxiv.org/x", ["arxiv.org"])["port"] == 80


def test_the_route_without_a_site_config_is_empty(monkeypatch) -> None:
    body = _route_client(monkeypatch, None).get("/umrs/k2/ingest/allowlist").json()
    assert body["entries"] == [] and body["allow_all"] is False


def test_the_route_needs_a_caller(monkeypatch) -> None:
    monkeypatch.delenv("UMRS_AUTH_MODE", raising=False)
    client = _route_client(monkeypatch, _site(["*"]), override=False)
    assert client.get("/umrs/k2/ingest/allowlist").status_code in (401, 403)


# ── the four fronts ───────────────────────────────────────────────────────


@pytest.mark.parametrize("front", FRONTS)
def test_each_front_allows_every_public_site(front) -> None:
    """Maintainer decision 2026-10-25: the fronts allow "*" (the address,
    denied-network, port, redirect and cap guards stay)."""
    from umrs.orchestrator.site_config import load_site_config

    config = load_site_config(str(UMRS / "deploy" / front / "site.yaml"))
    assert ingest_allowlist.configured_entries(config) == ["*"]
    assert ingest_allowlist.site_allowlist(config) == ["*"]
    assert config.extra.get("ssh_target"), "the rest of extra is kept"


@pytest.mark.parametrize("front", FRONTS)
def test_each_front_denies_the_riken_network(front) -> None:
    """Every front runs on umrs-front inside RIKEN (134.160.0.0/16, global
    unicast): its own network is denied, data0 included."""
    from umrs.orchestrator.site_config import load_site_config

    config = load_site_config(str(UMRS / "deploy" / front / "site.yaml"))
    denied = ingest_allowlist.site_denied_networks(config)
    assert denied == ["134.160.0.0/16"]
    networks, _ = egress_fetch.parse_networks(denied)
    assert "denied network" in egress_fetch.address_refusal(_RIKEN_INTRANET, networks)


@pytest.mark.parametrize("front", FRONTS)
def test_each_front_says_to_quote_the_entries(front) -> None:
    """An unquoted ``*.example.org`` is a YAML alias and breaks the whole
    site.yaml (review 2026-10-25): the comment says to quote."""
    text = (UMRS / "deploy" / front / "site.yaml").read_text()
    assert "Quote every entry" in text


def test_an_unquoted_star_entry_breaks_yaml_as_the_docs_warn() -> None:
    import yaml

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("allowed_domains:\n  - *.uni.example\n")
    assert yaml.safe_load('allowed_domains: ["*"]') == {"allowed_domains": ["*"]}


# ── announced at server start ─────────────────────────────────────────────


def test_the_allowlist_is_announced_at_start(caplog) -> None:
    """A misspelt token reached the log only at the first ingest (review
    2026-10-25); the server now expands the list when it starts."""
    caplog.set_level(logging.INFO, logger=ingest_allowlist.__name__)
    config = _site(["@recomended", "arxiv.org"])
    config.extra["knowledge_ingest"]["denied_networks"] = ["134.160.0.0/16"]
    assert ingest_allowlist.announce_site_allowlist(config) == ["arxiv.org"]
    assert "'@recomended' refused" in caplog.text
    assert "1 entries; denied networks: 134.160.0.0/16" in caplog.text


def test_the_sidecar_announces_the_allowlist_when_it_starts(monkeypatch, caplog) -> None:
    import inspect

    from umrs.server import http_sidecar

    caplog.set_level(logging.WARNING, logger=ingest_allowlist.__name__)
    monkeypatch.setattr(ingest_allowlist, "load_site_config", lambda: _site(["*"]))
    http_sidecar._announce_ingest_allowlist()
    assert "every public web site" in caplog.text
    assert "_announce_ingest_allowlist()" in inspect.getsource(http_sidecar._create_app)


def test_a_failing_announcement_never_stops_the_server(monkeypatch) -> None:
    from umrs.server import http_sidecar

    def _boom() -> Any:
        raise RuntimeError("no config")

    monkeypatch.setattr(ingest_allowlist, "load_site_config", _boom)
    http_sidecar._announce_ingest_allowlist()
