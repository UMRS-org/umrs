"""``GET /umrs/rag/retrieve`` and ``GET /umrs/rag/passages/{node_id}`` (ADR-093 D-2, C1).

The retrieve verb: scored, cut and judged passages with no model generation
— so ``search_docs`` stops paying for a 3B answer it discards — reachable as
``/gw/rag/*`` and over the D-42 read-only rail because it is GET.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve, rag_spine
from umrs.knowledge.retrieval import calibrate, pipeline
from umrs.knowledge.retrieval import hydrate as hydrate_mod

from r4_contracts.retrieval import RetrievalSufficiency, RetrievedPassage, RetrieveResult

_RESULT = RetrieveResult(
    query="q",
    passages=(
        RetrievedPassage(
            rank=1, source_id="node-chunk-g-a-0", artifact_id="a", source_name="a.md", text="t"
        ),
    ),
    sufficiency=RetrievalSufficiency(status="sufficient", basis="z_null", value=5.0, threshold=3.0),
    searched=40,
    space="e5-base-v2",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("UMRS_RAG_FINAL_K_MAX", "UMRS_RAG_POOL", "UMRS_RAG_PER_ARTIFACT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    calibrate.clear_cache()


@pytest.fixture
def client(monkeypatch):
    seen = {"scope": [], "specs": []}

    def scope(request, target=None, *, write=False):
        seen["scope"].append((target, write))
        return "u1", "p-verified"

    def retrieve(pool, spec, **kwargs):
        seen["specs"].append(spec)
        seen["kwargs"] = kwargs
        return _RESULT

    monkeypatch.setattr(rag_api, "_require_rag_scope", scope)
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "hidden_graphs", lambda _req, _db, _project: ())
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: "ENCODER")
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: "FETCH")
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = object()
    return TestClient(app), seen


# ── the retrieve verb ───────────────────────────────────────────────────────


def test_the_response_is_the_retrieve_result(client):
    http, seen = client
    response = http.get("/umrs/rag/retrieve", params={"q": "what slows moisture?"})
    assert response.status_code == 200 and response.json() == _RESULT.to_dict()
    assert seen["scope"] == [(None, False)]  # read scope, the verified project
    assert seen["kwargs"] == {"encode": "ENCODER", "rerank": None, "fetch_bytes": "FETCH"}


def test_defaults_come_from_the_site(client, monkeypatch):
    http, seen = client
    monkeypatch.setenv("UMRS_RAG_FINAL_K_MAX", "3")
    http.get("/umrs/rag/retrieve", params={"q": "q"})
    spec = seen["specs"][0]
    assert (spec.project_id, spec.k_max, spec.pool, spec.per_artifact) == ("p-verified", 3, 50, 2)
    assert (spec.lexical, spec.rerank, spec.merge, spec.cutoff, spec.threshold) == (
        "auto",
        "auto",
        "auto",
        "adaptive",
        None,
    )
    assert spec.graph_ids is None and spec.exclude_graph_ids == ()


def test_every_knob_reaches_the_spec(client):
    http, seen = client
    params = [
        ("q", "first"),
        ("q", "second"),
        ("project_id", "p2"),
        ("k_max", "6"),
        ("pool", "80"),
        ("per_artifact", "3"),
        ("lexical", "off"),
        ("rerank", "on"),
        ("merge", "on"),
        ("cutoff", "fixed"),
        ("trace", "1"),
    ]
    assert http.get("/umrs/rag/retrieve", params=params).status_code == 200
    spec = seen["specs"][0]
    assert spec.queries == ("first", "second") and spec.project_id == "p2"
    assert (spec.k_max, spec.pool, spec.per_artifact) == (6, 80, 3)
    assert (spec.lexical, spec.rerank, spec.merge, spec.cutoff, spec.trace) == (
        "off",
        "on",
        "on",
        "fixed",
        True,
    )
    assert seen["scope"] == [("p2", False)]


def test_a_threshold_is_the_legacy_mode_with_k_five(client):
    http, seen = client
    http.get("/umrs/rag/retrieve", params={"q": "q", "threshold": "0.7"})
    assert (seen["specs"][0].threshold, seen["specs"][0].k_max) == (0.7, 5)


@pytest.mark.parametrize(
    ("params", "status"),
    [
        ([("q", "a"), ("q", "b"), ("q", "c"), ("q", "d")], 400),
        ([("q", "x" * 2049)], 400),
        ([("q", "   ")], 400),
        ([], 422),
        ([("q", "a"), ("lexical", "maybe")], 422),
        ([("q", "a"), ("k_max", "13")], 422),
        ([("q", "a"), ("pool", "5")], 422),
        ([("q", "a"), ("cutoff", "smart")], 422),
    ],
)
def test_bad_requests_are_refused(client, params, status):
    http, seen = client
    assert http.get("/umrs/rag/retrieve", params=params).status_code == status
    assert seen["specs"] == []


def test_no_database_is_unavailable(client):
    http, _ = client
    http.app.state.umrs_db_pool = None
    assert http.get("/umrs/rag/retrieve", params={"q": "q"}).status_code == 503


def test_named_graphs_resolve_under_kg05_and_exclusions_win(client, monkeypatch):
    from umrs.api import _access, knowledge_query

    http, seen = client
    asked = {}

    def selected(registry, project_id, graphs_csv, exclude_csv, readable):
        asked.update(project=project_id, graphs=graphs_csv, exclude=exclude_csv)
        candidates = [SimpleNamespace(graph_id="g-ok"), SimpleNamespace(graph_id="g-denied")]
        return [g for g in candidates if readable(g)], []

    monkeypatch.setattr(knowledge_query, "_resolve_selected_graphs", selected)
    monkeypatch.setattr(knowledge_query, "_resolve_exclusions", lambda _r, _csv: ({"g-x"}, ["x"]))
    monkeypatch.setattr(
        _access, "graph_meta_readable", lambda _req, graph, _scope: graph.graph_id == "g-ok"
    )
    monkeypatch.setattr(
        "umrs.knowledge.graph_registry.GraphRegistryStore", lambda _pool: "REGISTRY"
    )
    http.get(
        "/umrs/rag/retrieve",
        params={"q": "q", "graphs": "scifact-eval,secret", "exclude_graphs": "x"},
    )
    spec = seen["specs"][0]
    assert spec.graph_ids == ("g-ok",) and spec.exclude_graph_ids == ("g-x",)
    assert asked == {"project": "p-verified", "graphs": "scifact-eval,secret", "exclude": "x"}


def test_no_graphs_named_is_the_whole_project_without_a_registry_write(client, monkeypatch):
    http, seen = client

    def no_registry(_pool):
        raise AssertionError("a read without graphs must not touch the registry")

    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", no_registry)
    assert http.get("/umrs/rag/retrieve", params={"q": "q"}).status_code == 200
    assert seen["specs"][0].graph_ids is None


# ── graphs the caller may not read (KG-05 on the whole-project scope) ─────────


class _Registry:
    """Read-only registry stand-in: ``ensure_default_graph`` would be a write."""

    graphs = {
        "g-u2-personal": SimpleNamespace(
            graph_id="g-u2-personal", kind="personal", project_id=None, owner_id="u2", access=[]
        ),
        "g-shared": SimpleNamespace(
            graph_id="g-shared", kind="research", project_id="p1", owner_id="u2", access=[]
        ),
    }

    def __init__(self, _pool):
        pass

    def get_graph(self, graph_id):
        return self.graphs.get(graph_id)

    def ensure_default_graph(self, *_a, **_k):
        raise AssertionError("a read path must not write the default graph")


_MEMBER = {"x-user-id": "u1", "x-verified-project-id": "p1"}


def test_no_graphs_named_leaves_out_another_members_personal_graph(monkeypatch):
    """Review of ADR-093 (2026-10-10): naming u2's personal graph was refused,
    leaving ``graphs`` out read it (``n.project_id = p1`` admits every graph)."""
    seen = {}

    def retrieve(_db, spec, **_kwargs):
        seen["spec"] = spec
        return _RESULT

    pool = _Pool([("SELECT DISTINCT n.graph_id", [("g-u2-personal",), ("g-shared",), ("g-x",)])])
    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", _Registry)
    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u1", "p1"))
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: "ENCODER")
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = pool
    response = TestClient(app, headers=_MEMBER).get("/umrs/rag/retrieve", params={"q": "q"})
    assert response.status_code == 200, response.text
    assert seen["spec"].graph_ids is None  # still the whole project ...
    assert seen["spec"].exclude_graph_ids == ("g-u2-personal",)  # ... minus u2's graph
    assert pool.calls[0][1] == ("p1",)


def test_the_owner_reads_their_own_personal_graph(monkeypatch):
    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", _Registry)
    pool = _Pool([("SELECT DISTINCT n.graph_id", [("g-u2-personal",)])])
    request = SimpleNamespace(headers={"x-user-id": "u2", "x-verified-project-id": "p1"})
    assert rag_retrieve.hidden_graphs(request, pool, "p1") == ()


@pytest.mark.parametrize(
    ("graph_id", "hidden"),
    [
        ("g-u2-personal", True),
        ("g-shared", False),  # a project graph: the project rule
        ("g-unregistered", False),  # unknown to the registry: the project rule
        ("", False),  # a legacy row
    ],
)
def test_only_a_registered_graph_the_caller_may_not_read_is_hidden(monkeypatch, graph_id, hidden):
    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", _Registry)
    request = SimpleNamespace(headers=_MEMBER)
    assert rag_retrieve.graph_hidden(request, object(), graph_id, "p1") is hidden


def test_the_default_graph_is_never_hidden_and_needs_no_registry(monkeypatch):
    from umrs.knowledge.graph_registry import default_graph_id

    def no_registry(_pool):
        raise AssertionError("the default graph is the project's own")

    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", no_registry)
    request = SimpleNamespace(headers=_MEMBER)
    assert not rag_retrieve.graph_hidden(request, object(), default_graph_id("p1"), "p1")


def test_both_routes_are_the_rag_routers_own_and_declare_their_auth():
    """The route inventory audits a router's own routes only."""
    routes = {r.path: r for r in rag_api.router.routes}
    for path in ("/umrs/rag/retrieve", "/umrs/rag/passages/{node_id}"):
        assert routes[path].methods == {"GET"}
        assert "# auth: verified" in (routes[path].endpoint.__doc__ or "")


# ── receipt 3 through the real route ────────────────────────────────────────


class _Pool:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def getconn(self):
        return self

    def putconn(self, _conn):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self.rows = next((rows for marker, rows in self.answers if marker in sql), [])

    def fetchall(self):
        return list(self.rows)


class _Response:
    def __init__(self, body):
        self._body, self.status_code = body, 200

    def json(self):
        return self._body

    def raise_for_status(self):
        return None


def test_the_route_never_calls_a_chat_model(monkeypatch):
    from umrs.knowledge import agent_executor

    urls = []

    def post(url, **kwargs):
        urls.append(str(url))
        texts = kwargs["json"].get("texts") or [kwargs["json"].get("text")]
        return _Response({"model": "/models/e5-base-v2", "vectors": [[1.0, 0.0] for _ in texts]})

    def chat(*_a, **_k):
        raise AssertionError("chat call on the retrieve path")

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    monkeypatch.setenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", "http://rag:8003")
    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u", "p1"))
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    content = json.dumps(
        {"kind": "chunk", "artifact_id": "a", "chunk_index": 0, "text": "Alumina slows moisture."}
    )
    pool = _Pool(
        [
            ("<=>", [("node-chunk-g-a-0", content, 0.1, 3)]),
            ("LEFT JOIN", [("node-chunk-g-a-0", "g", content, None, "a.md", None, None)]),
        ]
    )
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = pool
    body = (
        TestClient(app)
        .get("/umrs/rag/retrieve", params={"q": "alumina moisture", "rerank": "on", "merge": "on"})
        .json()
    )
    assert [p["source_id"] for p in body["passages"]] == ["node-chunk-g-a-0"]
    assert body["sufficiency"]["status"] == "sufficient"  # the lexical rule: fewer than 30 vectors
    assert urls and all(u.endswith("/rag/embed") for u in urls)


# ── the fetch handle ────────────────────────────────────────────────────────

_DOC = "First sentence of the note. Second sentence follows here."


def _chunk_content(index, start, end):
    md5 = hashlib.md5(_DOC[start:end].encode()).hexdigest()
    return json.dumps(
        {
            "kind": "chunk",
            "artifact_id": "a",
            "chunk_index": index,
            "chunk_start": start,
            "chunk_end": end,
            "chunk_md5": md5,
        }
    )


def _passage_client(
    monkeypatch,
    *,
    owner="p1",
    visibility="project",
    admin=False,
    readable=False,
    hidden=False,
    neighbour_owner=None,
    graph="g",
):
    scope = {  # node -> (project, graph, visibility, artifact): what the scoped SELECT reads
        "node-chunk-g-a-0": (owner, graph, visibility, "a"),
        "node-chunk-g-a-1": (neighbour_owner or owner, graph, visibility, "a"),
    }
    rows = {
        "node-chunk-g-a-0": (
            "node-chunk-g-a-0",
            "g",
            _chunk_content(0, 0, 27),
            "ref",
            "note.md",
            None,
            None,
        ),
        "node-chunk-g-a-1": (
            "node-chunk-g-a-1",
            "g",
            _chunk_content(1, 28, len(_DOC)),
            "ref",
            "note.md",
            None,
            None,
        ),
    }

    class Pool(_Pool):
        def execute(self, sql, params=None):
            self.calls.append((sql, params))
            if "n.node_id = %s" in sql:
                self.rows = [(owner, graph, visibility, "0", "a")] if params[0] in rows else []
            elif "IS NOT DISTINCT FROM" in sql:
                ids, *centre = params
                self.rows = [(n,) for n in ids if list(scope.get(n, ())) == centre]
            elif "LEFT JOIN" in sql:
                self.rows = [rows[n] for n in params[0] if n in rows]
            else:
                self.rows = []

    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u", "p1"))
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: _DOC.encode())
    monkeypatch.setattr(
        rag_retrieve, "_graph_readable", readable if callable(readable) else lambda *_a: readable
    )
    monkeypatch.setattr(rag_retrieve, "graph_hidden", lambda *_a: hidden)
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = Pool([])
    # H-1 (2026-10-18): admin is the verified tuple's flag; a bare
    # X-Verified-Is-Admin with no identity was the unsigned-admin defect.
    admin_tuple = {
        "x-user-id": "root",
        "x-verified-project-id": "p1",
        "x-verified-is-admin": "true",
    }
    return TestClient(app, headers=admin_tuple if admin else {})


def test_a_visible_passage_comes_back_with_its_full_text(monkeypatch):
    body = _passage_client(monkeypatch).get("/umrs/rag/passages/node-chunk-g-a-0").json()
    assert body == {
        "source_id": "node-chunk-g-a-0",
        "artifact_id": "a",
        "source_name": "note.md",
        "text": "First sentence of the note.",
        "span": [0, 27],
        "window": 0,
        "chunks": [0, 0],
        "text_source": "artifact",
    }


def test_a_window_adds_the_neighbours_and_max_chars_caps_the_text(monkeypatch):
    http = _passage_client(monkeypatch)
    body = http.get("/umrs/rag/passages/node-chunk-g-a-0", params={"window": 1}).json()
    assert body["text"] == _DOC and body["span"] == [0, len(_DOC)] and body["chunks"] == [0, 1]
    capped = http.get(
        "/umrs/rag/passages/node-chunk-g-a-0", params={"window": 1, "max_chars": 10}
    ).json()
    assert capped["text"] == _DOC[:10]
    assert http.get("/umrs/rag/passages/node-chunk-g-a-0", params={"window": 3}).status_code == 422


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"visibility": "ambient"}, 404),
        ({"visibility": "ambient", "admin": True}, 200),
        ({"owner": "p-other"}, 404),
        ({"owner": "p-other", "readable": True}, 200),
        ({"owner": "p-other", "visibility": "system"}, 200),
    ],
)
def test_the_handle_uses_the_retrieve_verbs_access_predicate(monkeypatch, kwargs, status):
    http = _passage_client(monkeypatch, **kwargs)
    assert http.get("/umrs/rag/passages/node-chunk-g-a-0").status_code == status


def test_an_unknown_node_is_not_found(monkeypatch):
    assert _passage_client(monkeypatch).get("/umrs/rag/passages/nope").status_code == 404


def test_a_window_never_stitches_a_chunk_of_another_project(monkeypatch):
    """Review of ADR-093 (2026-10-10): neighbour ids are string arithmetic on
    the centre's id and were hydrated with no predicate, so a node imported
    under a chosen id in the caller's project read another project's chunks."""
    http = _passage_client(monkeypatch, neighbour_owner="p-victim")
    response = http.get("/umrs/rag/passages/node-chunk-g-a-0", params={"window": 1})
    body = response.json()
    assert response.status_code == 200
    assert body["text"] == "First sentence of the note." and body["chunks"] == [0, 0]
    pool = http.app.state.umrs_db_pool
    hydrated = [params[0] for sql, params in pool.calls if "LEFT JOIN" in sql]
    assert hydrated == [["node-chunk-g-a-0"]]  # the foreign chunk is never read


def test_the_window_reads_neighbours_under_the_centres_own_scope(monkeypatch):
    http = _passage_client(monkeypatch)
    http.get("/umrs/rag/passages/node-chunk-g-a-0", params={"window": 1})
    [(sql, params)] = [c for c in http.app.state.umrs_db_pool.calls if "IS NOT DISTINCT" in c[0]]
    assert params == (["node-chunk-g-a-1"], "p1", "g", "project", "a")
    for column in ("project_id", "graph_id", "visibility", "content->>'artifact_id'"):
        assert f"n.{column} IS NOT DISTINCT FROM %s" in sql


def test_a_project_row_of_a_graph_the_caller_may_not_read_is_not_found(monkeypatch):
    """The handle admits what the verb admits: the caller's project minus the
    graphs it may not read (another member's personal graph)."""
    http = _passage_client(monkeypatch, hidden=True)
    assert http.get("/umrs/rag/passages/node-chunk-g-a-0").status_code == 404


# Review of ADR-093 (fix-regressions-6): since core-10 the verb reads the
# legacy (graph_id '') rows of every project whose DEFAULT graph is selected,
# a shared one of another project included; the handle checked the graph ''
# (never readable) and 404'd a passage the verb had just returned.


def _default_graph_reader(project):
    from umrs.knowledge.graph_registry import default_graph_id

    shared = default_graph_id(project)
    return shared, lambda _request, _db, graph_id: graph_id == shared


def test_another_projects_legacy_row_is_its_default_graphs():
    from umrs.knowledge.graph_registry import default_graph_id

    assert rag_retrieve.row_graph_id("pB", "") == default_graph_id("pB")
    assert rag_retrieve.row_graph_id("pB", "g-named") == "g-named"
    assert rag_retrieve.row_graph_id("", "") == ""


def test_the_handle_serves_a_shared_default_graphs_legacy_row(monkeypatch):
    shared, readable = _default_graph_reader("p-other")
    http = _passage_client(monkeypatch, owner="p-other", graph="", readable=readable)
    response = http.get("/umrs/rag/passages/node-chunk-g-a-0")
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "First sentence of the note."
    _shared, only_p_third = _default_graph_reader("p-third")
    http = _passage_client(monkeypatch, owner="p-other", graph="", readable=only_p_third)
    assert http.get("/umrs/rag/passages/node-chunk-g-a-0").status_code == 404


def test_node_visible_asks_the_registry_for_the_owners_default_graph(monkeypatch):
    from umrs.api import _access

    shared, _readable = _default_graph_reader("pB")
    asked = []

    class Registry:
        def __init__(self, _pool):
            pass

        def get_graph(self, graph_id):
            asked.append(graph_id)
            return SimpleNamespace(graph_id=graph_id) if graph_id == shared else None

    monkeypatch.setattr("umrs.knowledge.graph_registry.GraphRegistryStore", Registry)
    monkeypatch.setattr(_access, "graph_meta_readable", lambda _req, _graph, _scope: True)
    request = SimpleNamespace(headers=_MEMBER)
    assert rag_retrieve.node_visible(request, object(), "pA", ("pB", "", "project", "0", "a"))
    assert not rag_retrieve.node_visible(request, object(), "pA", ("pC", "", "project", "0", "a"))
    assert asked[0] == shared


def test_the_verbs_encoder_carries_the_nulls_encoding_probe(monkeypatch):
    """fix-regressions-7: the null cache sees the query encoding through the
    probe the verb's encoder adds to each batch (no extra call)."""
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8003")
    sent = []

    def post(url, json=None, **_kwargs):
        sent.append(json["texts"])
        vectors = [[1.0, float(i)] for i in range(len(json["texts"]))]
        return httpx.Response(
            200, json={"model": "e5", "vectors": vectors}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(httpx, "post", post)
    encode = rag_retrieve.batch_encoder()
    assert isinstance(encode, calibrate.FingerprintedEncoder)
    model, vectors, _reason = encode(["q"])
    assert sent == [["q", calibrate.ENCODING_PROBE]] and vectors == [[1.0, 0.0]]
    assert encode.encoding_probe == [1.0, 1.0]
