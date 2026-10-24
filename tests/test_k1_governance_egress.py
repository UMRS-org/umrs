"""K1.1 (knowledge-sharing plan, 2026-10-24): governance and egress holes.

Each test fails on the code before K1.1:

* (a) ``/knowledge/upload`` put no ``origin_actor`` in its job spec, so an
  approved agent upload landed ``established``;
* (b) a research-loop finding upserted with no status (dataclass default
  ``established``) and no graph;
* (c) K-1 (ADR-084 D-2): entities and relations an LLM extracted from a
  human's upload landed ``established``;
* (d) the caller's ``allow_domains`` replaced the site allowlist (``[]`` from
  the HTTP route disabled it, any list widened it); ``fetch_url`` used
  ``urlopen``: no private-address block, redirects followed after one check,
  body read whole before the cap;
* (e) ``external/search`` sent query text out for any verified project.

No test touches the internet: the SSRF tests run a local HTTP server and a
fake resolver that maps test names to addresses.
"""

from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.knowledge import egress_fetch
from umrs.knowledge.finding_ingest import InMemoryFindingSink, mint_iteration
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop
from umrs.knowledge.sources import EgressDisabledError, SourceResolver

# ── (a) agent upload carries its actor to the understand chain ────────────


def _upload_app(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> FastAPI:
    from umrs.api.files import get_project_id
    from umrs.api.knowledge_upload import router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_project_id] = lambda: "proj-k1"

    class _S3:
        def upload_fileobj(self, fileobj: Any, *_args: Any, **_kwargs: Any) -> None:
            fileobj.read()

    config = SimpleNamespace(bucket="umrs-files", get_client=lambda: _S3())
    monkeypatch.setattr("umrs.api.knowledge_upload.get_s3_config", lambda: config)
    monkeypatch.setattr(
        "umrs.api.knowledge_upload._resolve_ingest_target", lambda *_a, **_k: "g-k1"
    )
    monkeypatch.setattr(
        "umrs.api.knowledge_upload.enforce_graph_mutation_authorization",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("umrs.api.knowledge_upload.guard_ingest", lambda *_a, **_k: None)

    class _Runtime:
        async def submit_job_sync(self, **kwargs: Any) -> str:
            calls.append(kwargs)
            return "job-k1"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    return app


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-r4-actor": "agent", "x-proposal-id": "prop-1"}, "agent"),
        ({}, "user"),
    ],
)
def test_upload_spec_carries_the_request_actor(monkeypatch, headers, expected) -> None:
    calls: list[dict[str, Any]] = []
    client = TestClient(_upload_app(monkeypatch, calls), raise_server_exceptions=False)
    resp = client.post(
        "/umrs/knowledge/upload",
        files={"file": ("note.md", io.BytesIO(b"# note"), "text/markdown")},
        headers={"x-user-id": "u-1", **headers},
    )
    assert resp.status_code == 202, resp.text
    assert calls[-1]["spec"]["origin_actor"] == expected


def test_agent_origin_lands_every_understood_row_proposed(monkeypatch) -> None:
    """The understand chain reads the spec's actor: an agent upload's rows
    (document node included) are proposed; confidence claims nothing."""
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_llm_extract_document", lambda *_a, **_k: None)
    nodes, edges = ka._nodes_edges_for_artifact(
        "art-a", "ir-a", "document", b"plain prose", {}, "proj-k1", "g-k1"
    )
    nodes, edges = ka._as_proposed_tier(nodes, edges)
    assert nodes and all(n.status == "proposed" and n.confidence == 0.0 for n in nodes)


# ── (c) K-1: LLM extraction from a human upload is a judgement ────────────


_LLM_PAYLOAD = {
    "nodes": [
        {"id": "catalyst", "name": "Catalyst B", "type": "entity"},
        {"id": "yield", "name": "Yield", "type": "concept"},
    ],
    "edges": [{"source": "catalyst", "target": "yield", "relation": "raises"}],
}


def _human_upload_rows(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Understand a human's prose upload whose entities a model extracts."""
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_llm_extract_document", lambda *_a, **_k: _LLM_PAYLOAD)
    return ka._nodes_edges_for_artifact(
        "art-h", "ir-h", "document", b"Catalyst B raises yield.", {}, "proj-k1", "g-k1"
    )


def _tiers(rows: list[Any]) -> set[tuple[str, float]]:
    return {(row.status, row.confidence) for row in rows}


def test_llm_entities_from_a_human_upload_land_proposed(monkeypatch) -> None:
    nodes, _edges = _human_upload_rows(monkeypatch)
    semantic = [n for n in nodes if n.node_id.startswith("node-sem-")]
    assert len(semantic) == 2
    assert _tiers(semantic) == {("proposed", 0.0)}


def test_llm_relations_from_a_human_upload_land_proposed(monkeypatch) -> None:
    _nodes, edges = _human_upload_rows(monkeypatch)
    semantic = [e for e in edges if (e.content or {}).get("semantic")]
    assert semantic
    assert {e.status for e in semantic} == {"proposed"}


def test_the_document_node_keeps_the_human_tier(monkeypatch) -> None:
    nodes, _edges = _human_upload_rows(monkeypatch)
    structural = [n for n in nodes if not n.node_id.startswith("node-sem-")]
    assert structural
    assert {n.status for n in structural} == {"established"}


def test_author_written_graph_json_keeps_the_human_tier(monkeypatch) -> None:
    """A graph the author wrote (W3 graph-JSON) is the human's content, not a
    model's judgement: it stays established."""
    import json

    from umrs.jobs import knowledge_adapters as ka

    def _no_llm(*_a: Any, **_k: Any) -> None:
        raise AssertionError("graph-JSON must not reach the LLM extractor")

    monkeypatch.setattr(ka, "_llm_extract_document", _no_llm)
    text = json.dumps(_LLM_PAYLOAD).encode()
    nodes, _edges = ka._nodes_edges_for_artifact(
        "art-j", "ir-j", "document", text, {}, "proj-k1", "g-k1"
    )
    semantic = [n for n in nodes if n.node_id.startswith("node-sem-")]
    if not semantic:
        pytest.skip("this payload shape is not detected as graph-JSON")
    assert all(n.status == "established" for n in semantic)


# ── (b) loop findings land proposed in the loop's graph ───────────────────


def _loop(metadata: dict[str, Any] | None = None) -> ResearchLoop:
    return ResearchLoop(
        loop_id="loop-k1",
        project_id="proj-k1",
        research_goal="g",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PUBLISHING,
        metadata=dict(metadata or {}),
    )


def _iteration() -> LoopIteration:
    return LoopIteration(
        iteration_id="it-k1",
        loop_id="loop-k1",
        iteration_number=1,
        phase=LoopPhase.MEMORY_UPDATE,
        findings=[
            {
                "finding_id": "finding-k1",
                "statement": "Batch-3 yield 85%",
                "confidence": 0.9,
                "evidence_refs": [{"artifact_id": "art-e"}],
            }
        ],
        status="completed",
    )


def test_loop_finding_rows_are_proposed_in_the_default_graph() -> None:
    sink = InMemoryFindingSink()
    outcome = mint_iteration(_loop(), _iteration(), sink=sink)
    assert outcome.minted == ["finding-k1"]
    rows = [*sink.nodes.values(), *sink.edges.values()]
    assert rows
    for row in rows:
        assert row["status"] == "proposed"
        assert row["confidence"] == 0.0
        assert row["graph_id"] == default_graph_id("proj-k1")


def test_loop_finding_honours_a_loop_graph() -> None:
    sink = InMemoryFindingSink()
    mint_iteration(_loop({"graph_id": "g-loop"}), _iteration(), sink=sink)
    assert {row["graph_id"] for row in sink.nodes.values()} == {"g-loop"}


def test_postgres_sink_builds_proposed_store_rows() -> None:
    """The production sink feeds the dicts to KnowledgeNode/KnowledgeEdge:
    the tier keys must be dataclass fields (a typo would raise here)."""
    from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode

    sink = InMemoryFindingSink()
    mint_iteration(_loop(), _iteration(), sink=sink)
    for node in sink.nodes.values():
        assert KnowledgeNode(**node).status == "proposed"
    for edge in sink.edges.values():
        assert KnowledgeEdge(**edge).status == "proposed"


# ── (d) the site allowlist always applies ─────────────────────────────────


def _site(domains: list[str]) -> Any:
    return SimpleNamespace(extra={"knowledge_ingest": {"allowed_domains": domains}})


def test_no_caller_list_uses_the_site_list() -> None:
    resolved = SourceResolver(site_config=_site(["arxiv.org"])).resolve(
        "http_url", "https://arxiv.org/abs/1", allow_domains=None
    )
    assert resolved.allow_domains == ["arxiv.org"]


def test_caller_list_narrows_never_widens() -> None:
    resolver = SourceResolver(site_config=_site(["arxiv.org", "*.github.com"]))
    narrowed = resolver.resolve(
        "http_url",
        "https://raw.github.com/x",
        allow_domains=["raw.github.com", "evil.example"],
    )
    assert narrowed.allow_domains == ["raw.github.com"]
    with pytest.raises(EgressDisabledError):
        resolver.resolve("http_url", "https://evil.example/", allow_domains=["evil.example"])


def test_metadata_address_in_caller_list_is_refused() -> None:
    resolver = SourceResolver(site_config=_site(["arxiv.org"]))
    with pytest.raises(EgressDisabledError):
        resolver.resolve(
            "http_url", "http://169.254.169.254/latest", allow_domains=["169.254.169.254"]
        )


def test_empty_site_list_refuses_even_with_a_caller_list() -> None:
    with pytest.raises(EgressDisabledError):
        SourceResolver(site_config=_site([])).resolve(
            "http_url", "https://arxiv.org/", allow_domains=["arxiv.org"]
        )


def test_intersection_rules() -> None:
    intersect_allowlists = egress_fetch.intersect_allowlists
    assert intersect_allowlists(None, ["a.org"]) == ["a.org"]
    assert intersect_allowlists([], ["a.org"]) == []
    assert intersect_allowlists(["x.a.org", "b.org"], ["*.a.org"]) == ["x.a.org"]


def test_http_route_carries_no_list_as_none(monkeypatch) -> None:
    """``allow_domains=[]`` (or absent) on POST /k2/ingest reaches the job as
    None — the site list — never as the empty list that disabled it."""
    from umrs.api import knowledge_v2
    from umrs.api.files import get_project_id

    specs: list[dict[str, Any]] = []

    class _Runtime:
        async def submit_job_sync(self, **kwargs: Any) -> str:
            specs.append(kwargs["spec"])
            return "job-k1"

    monkeypatch.setattr("umrs.server.durable_job_runtime.get_runtime", lambda: _Runtime())
    monkeypatch.setattr(knowledge_v2, "_resolve_ingest_target", lambda *_a, **_k: "g-k1")
    monkeypatch.setattr("umrs.api.governance.guard_ingest", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.enforce_graph_mutation_authorization",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "umrs.api._knowledge_mutation_guard.mark_proposal_executed", lambda *_a, **_k: None
    )
    app = FastAPI()
    app.include_router(knowledge_v2.router)
    app.dependency_overrides[get_project_id] = lambda: "proj-k1"
    client = TestClient(app, raise_server_exceptions=False)
    for body_extra in ({"allow_domains": []}, {}):
        resp = client.post(
            "/umrs/k2/ingest",
            json={"sources": [{"kind": "http_url", "uri": "https://arxiv.org/a"}], **body_extra},
            headers={"x-user-id": "u-1"},
        )
        assert resp.status_code in (200, 202), resp.text
    assert [s["allow_domains"] for s in specs] == [None, None]


# ── (d) the fetcher: addresses, every redirect hop, streamed cap ──────────


def fetch_url(url: str, **kwargs: Any) -> Any:
    return egress_fetch.fetch_url(url, **kwargs)


class _Handler(BaseHTTPRequestHandler):
    routes: dict[str, tuple[int, dict[str, str], bytes]] = {}

    def do_GET(self) -> None:  # noqa: N802 — http.server's name
        status, headers, body = self.routes.get(self.path, (404, {}, b"missing"))
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        if "Content-Length" not in headers:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


#: Test names → addresses. 93.184.215.14 is a global address; the connect
#: hook below sends it to the local server, so nothing leaves the host.
_PUBLIC = "93.184.215.14"
_DNS = {"docs.example.org": [_PUBLIC], "intranet.example.org": ["10.1.2.3"]}


@pytest.fixture()
def fake_net(monkeypatch, server):
    """Resolve test names without DNS; connect the fake public address to
    the local server. The address CHECK stays the production one."""
    real_open = getattr(egress_fetch, "_open_connection", None)

    def _resolve(host: str, _port: int) -> list[str]:
        if host not in _DNS:
            raise OSError(f"unknown test host {host}")
        return list(_DNS[host])

    def _open(scheme: str, host: str, port: int, address: str, timeout: int) -> Any:
        assert address == _PUBLIC, f"connect attempted to {address}"
        return real_open("http", host, server, "127.0.0.1", timeout)

    monkeypatch.setattr(egress_fetch, "_resolve_host", _resolve, raising=False)
    monkeypatch.setattr(egress_fetch, "_open_connection", _open, raising=False)
    return server


def test_allowed_public_name_is_fetched(fake_net) -> None:
    _Handler.routes = {"/doc": (200, {"Content-Length": "5"}, b"hello")}
    result = fetch_url("http://docs.example.org/doc", allow_domains=["docs.example.org"])
    assert result.error == "" and result.content == b"hello"


def test_loopback_literal_is_refused_even_when_listed(server) -> None:
    _Handler.routes = {"/secret": (200, {"Content-Length": "6"}, b"secret")}
    result = fetch_url(f"http://127.0.0.1:{server}/secret", allow_domains=["127.0.0.1"])
    assert result.content is None
    assert "loopback" in result.error


def test_metadata_literal_is_refused() -> None:
    result = fetch_url(
        "http://169.254.169.254/latest/meta-data", allow_domains=["169.254.169.254"]
    )
    assert result.content is None
    assert "metadata" in result.error


def test_name_resolving_to_a_private_address_is_refused(fake_net) -> None:
    result = fetch_url(
        "http://intranet.example.org/x", allow_domains=["intranet.example.org"]
    )
    assert result.content is None
    assert "private" in result.error


def test_redirect_to_loopback_is_refused(fake_net) -> None:
    _Handler.routes = {
        "/start": (302, {"Location": f"http://127.0.0.1:{fake_net}/secret", "Content-Length": "0"}, b""),
        "/secret": (200, {"Content-Length": "6"}, b"secret"),
    }
    result = fetch_url(
        "http://docs.example.org/start", allow_domains=["docs.example.org", "127.0.0.1"]
    )
    assert result.content is None
    assert "loopback" in result.error


def test_redirect_off_the_allowlist_is_refused(fake_net) -> None:
    _Handler.routes = {
        "/start": (301, {"Location": "http://elsewhere.example/x", "Content-Length": "0"}, b""),
    }
    result = fetch_url("http://docs.example.org/start", allow_domains=["docs.example.org"])
    assert result.content is None
    assert "not in allowlist" in result.error


def test_redirect_to_a_private_name_is_refused(fake_net) -> None:
    _Handler.routes = {
        "/start": (307, {"Location": "http://intranet.example.org/x", "Content-Length": "0"}, b""),
    }
    result = fetch_url(
        "http://docs.example.org/start",
        allow_domains=["docs.example.org", "intranet.example.org"],
    )
    assert result.content is None
    assert "private" in result.error


def test_allowed_redirect_is_followed(fake_net) -> None:
    _Handler.routes = {
        "/start": (302, {"Location": "/final", "Content-Length": "0"}, b""),
        "/final": (200, {"Content-Length": "2"}, b"ok"),
    }
    result = fetch_url("http://docs.example.org/start", allow_domains=["docs.example.org"])
    assert result.content == b"ok"


def test_body_over_the_cap_is_refused_while_streaming(fake_net, monkeypatch) -> None:
    """No Content-Length: the cap must hold on the streamed bytes."""
    monkeypatch.setattr(egress_fetch, "_EGRESS_MAX_BYTES", 1024)
    _Handler.routes = {"/big": (200, {}, b"x" * 5000)}
    result = fetch_url("http://docs.example.org/big", allow_domains=["docs.example.org"])
    assert result.content is None
    assert "too large" in result.error


def test_non_http_scheme_is_refused() -> None:
    result = fetch_url("ftp://docs.example.org/x", allow_domains=["docs.example.org"])
    assert result.content is None
    assert "scheme" in result.error


# ── (e) external/search: site flag off by default, data_class gate ────────


_SECRET = "k1-external-search-secret"


def _search_client(monkeypatch, data_class: str) -> TestClient:
    from umrs.api import knowledge as knowledge_module

    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", _SECRET)
    monkeypatch.setattr(
        "umrs.shared.project_data_class.data_class_for_project",
        lambda _pool, _project: data_class,
    )
    app = FastAPI()
    app.include_router(knowledge_module.router)
    return TestClient(
        app,
        headers={
            "X-User-ID": "alice",
            "X-Verified-Project-ID": "proj-a",
            "X-Verified-Is-Admin": "false",
            "X-Verified-HMAC": sign_claims(_SECRET, "alice", "proj-a", False),
        },
    )


def _searcher_must_not_run(monkeypatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("query text left the site")

    monkeypatch.setattr("umrs.knowledge.external.ExternalKnowledgeSearcher", _boom)


def test_external_search_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("UMRS_KNOWLEDGE_EXTERNAL_SEARCH", raising=False)
    _searcher_must_not_run(monkeypatch)
    resp = _search_client(monkeypatch, "non_sensitive").post(
        "/umrs/knowledge/external/search", json={"query": "q"}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "external_search_disabled"


@pytest.mark.parametrize("data_class", ["sensitive", ""])
def test_external_search_refused_for_a_restricted_project(monkeypatch, data_class) -> None:
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTERNAL_SEARCH", "1")
    _searcher_must_not_run(monkeypatch)
    resp = _search_client(monkeypatch, data_class).post(
        "/umrs/knowledge/external/search", json={"query": "q"}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "external_search_restricted"


def test_external_search_runs_when_on_for_a_non_sensitive_project(monkeypatch) -> None:
    monkeypatch.setenv("UMRS_KNOWLEDGE_EXTERNAL_SEARCH", "1")

    class _Searcher:
        def search_all(self, *_a: Any, **_k: Any) -> list[Any]:
            return []

    monkeypatch.setattr("umrs.knowledge.external.ExternalKnowledgeSearcher", _Searcher)
    resp = _search_client(monkeypatch, "non_sensitive").post(
        "/umrs/knowledge/external/search", json={"query": "q"}
    )
    assert resp.status_code == 200, resp.text
