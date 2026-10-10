"""`POST /umrs/rag/query` answers from the knowledge spine (ADR-027 D3/P2).

The route proxied only to the per-experience RAG store, which uploads stopped
feeding in WP-15: an uploaded document could be searched but never answered
from — "No documents retrieved" on the stand-alone stack (2026-10-09).

ADR-093 (2026-10-10): retrieval runs through the retrieval core
(``umrs.knowledge.retrieval``), whose hydration these tests pin; the compose
leg renders the shared context pack (question last). The pack, the
post-check and the scorer abstention are pinned in ``test_rag_compose_pack``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve, rag_spine
from umrs.api.rag_spine import SpineAnswer
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.types import Candidate

from r4_contracts.retrieval import RetrievalSufficiency, RetrievedPassage, RetrieveResult

_DOC = "  # Perovskite note\n\nEncapsulation with alumina slows moisture ingress. Dr. Tanaka measured 1200 h.\n"
_TEXT = _DOC.strip()  # what the embed job chunked (decode + strip)
_CHUNK = _TEXT[0:80]
_MD5 = hashlib.md5(_CHUNK.encode()).hexdigest()
_REF = "projects/p1/knowledge/artifacts/abc"


def _content(**overrides):
    content = {
        "kind": "chunk",
        "artifact_id": "art-1",
        "chunk_index": 0,
        "chunk_start": 0,
        "chunk_end": 80,
        "chunk_md5": _MD5,
        "excerpt": _CHUNK[:20],
    }
    content.update(overrides)
    return json.dumps(content)


class _Pool:
    """Answers the hydrate statements: the node rows, the chunk counts and
    the search_text probe (absent unless ``search_text`` is set)."""

    def __init__(self, rows, *, search_text=False):
        self.rows = rows
        self.search_text = search_text
        self.sql = []

    def _answer(self, sql):
        if "information_schema" in sql:
            return [(1,)] if self.search_text else []
        if "GROUP BY" in sql:
            return []
        return self.rows

    def getconn(self):
        pool = self

        class _Cur:
            rows: list = []

            def execute(self, sql, params=None):
                pool.sql.append((sql, params))
                self.rows = pool._answer(sql)

            def fetchall(self):
                return self.rows

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        class _Conn:
            def cursor(self):
                return _Cur()

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return _Conn()

    def putconn(self, _conn):
        pass


@pytest.fixture(autouse=True)
def _fresh_probe(monkeypatch):
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)


def _row(node_id, content, ref=_REF, name="perovskite-note.md", staged=None, title=None):
    return (node_id, "g-1", content, ref, name, staged, title)


def _hydrated_candidates(pool, ids, fetch):
    candidates = [Candidate(node_id=i) for i in ids]
    hydrate_mod.hydrate(pool, candidates, fetch)
    return candidates


# ── hydration (moved to umrs.knowledge.retrieval.hydrate, ADR-093) ─────────


def test_full_chunk_text_comes_from_the_artifact_md5_checked():
    pool = _Pool([_row("node-chunk-art-1-0", _content())])
    fetched = []
    [c] = _hydrated_candidates(
        pool, ["node-chunk-art-1-0"], lambda ref: fetched.append(ref) or _DOC.encode()
    )
    assert c.text == _CHUNK and c.text_source == "artifact"
    assert c.source_name == "perovskite-note.md" and c.artifact_id == "art-1"
    assert c.span == (0, 80) and fetched == [_REF]


def test_md5_mismatch_falls_back_to_the_stored_excerpt():
    pool = _Pool([_row("node-chunk-art-1-0", _content(chunk_md5="0" * 32), name="")])
    [c] = _hydrated_candidates(pool, ["node-chunk-art-1-0"], lambda _ref: _DOC.encode())
    assert (c.text, c.text_source) == (_CHUNK[:20], "excerpt")


def test_a_passage_with_no_text_is_left_empty_for_the_caller_to_drop():
    """Legacy-canonized chunk nodes carry offsets only and no artifact link."""
    content = json.dumps({"kind": "chunk", "chunk_start": 0, "chunk_end": 9, "chunk_md5": "x"})
    pool = _Pool([_row("node-legacy-chunk0", content, ref=None, name=None)])
    [c] = _hydrated_candidates(pool, ["node-legacy-chunk0"], lambda _r: None)
    assert (c.text, c.text_source) == ("", "")


def test_name_falls_back_to_the_staged_upload_key():
    staged = "s3://r4-data/knowledge-uploads/p1/u/zeolite-note.md"
    pool = _Pool([_row("node-chunk-art-1-0", _content(), name=None, staged=staged)])
    [c] = _hydrated_candidates(pool, ["node-chunk-art-1-0"], lambda _r: _DOC.encode())
    assert c.source_name == "zeolite-note.md"


def test_each_artifact_is_fetched_once():
    rows = [_row("n0", _content()), _row("n1", _content())]
    fetched = []
    _hydrated_candidates(_Pool(rows), ["n0", "n1"], lambda r: fetched.append(r) or _DOC.encode())
    assert fetched == [_REF]


def test_a_canonized_legacy_chunk_answers_from_its_stored_text():
    content = json.dumps(
        {"kind": "chunk", "chunk_start": 0, "chunk_end": 9, "text": "legacy passage"}
    )
    pool = _Pool([_row("node-legacy-chunk0", content, ref=None, name=None)])
    [c] = _hydrated_candidates(pool, ["node-legacy-chunk0"], lambda _r: None)
    assert (c.text, c.text_source) == ("legacy passage", "node")


def test_search_text_is_used_when_its_md5_matches_and_no_object_is_fetched():
    pool = _Pool([(*_row("node-chunk-art-1-0", _content()), _CHUNK)], search_text=True)
    fetched = []
    [c] = _hydrated_candidates(pool, ["node-chunk-art-1-0"], lambda r: fetched.append(r) or b"")
    assert (c.text, c.text_source) == (_CHUNK, "search_text") and fetched == []
    assert "n.search_text" in pool.sql[1][0]


def test_search_text_that_does_not_match_its_md5_is_never_served():
    pool = _Pool([(*_row("node-chunk-art-1-0", _content()), "tampered text")], search_text=True)
    [c] = _hydrated_candidates(pool, ["node-chunk-art-1-0"], lambda _r: _DOC.encode())
    assert (c.text, c.text_source) == (_CHUNK, "artifact")


def test_no_search_text_column_reads_without_it():
    pool = _Pool([_row("node-chunk-art-1-0", _content())])
    _hydrated_candidates(pool, ["node-chunk-art-1-0"], lambda _r: _DOC.encode())
    assert all("search_text" not in sql for sql, _ in pool.sql if "LEFT JOIN" in sql)


# ── generation ──────────────────────────────────────────────────────────────


def _hydrated(text=_CHUNK):
    return [
        RetrievedPassage(
            rank=1,
            source_id="node-chunk-art-1-0",
            artifact_id="art-1",
            source_name="perovskite-note.md",
            text=text,
            scores={"dense": 0.8},
        )
    ]


def test_answer_uses_the_registry_model_and_the_pack_question_last(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge import agent_executor

    seen = {}
    monkeypatch.setattr(
        ka, "_extract_chat_endpoint", lambda _project: ("http://ollama:11434", "llama3.2:3b")
    )

    def chat(messages, **kwargs):
        seen.update(messages=messages, **kwargs)
        return "<think>hmm</think>Alumina encapsulation slows it [1]."

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    answer, model, reason, cmap = rag_spine.compose_answer(
        "what slows moisture?", _hydrated(), "p1"
    )
    assert (answer, model, reason) == ("Alumina encapsulation slows it [1].", "llama3.2:3b", "")
    assert cmap == {"[1]": "node-chunk-art-1-0"}
    assert seen["model"] == "llama3.2:3b" and seen["base_url"] == "http://ollama:11434"
    user = seen["messages"][1]["content"]
    assert "[1] perovskite-note.md\n" + _CHUNK in user
    assert user.startswith("<passages") and "Question: what slows moisture?" in user
    assert user.index("</passages>") < user.index("Question:")


def test_no_chat_endpoint_is_a_reason(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: None)
    answer, model, reason, _cmap = rag_spine.compose_answer("q", _hydrated(), "p1")
    assert (answer, model, reason) == ("", "", "no READY chat endpoint in the registry")


def test_chat_failure_is_a_reason_naming_the_model(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge import agent_executor

    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: ("http://x", "m"))

    def boom(*_a, **_k):
        raise TimeoutError("slow")

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", boom)
    answer, model, reason, _cmap = rag_spine.compose_answer("q", _hydrated(), "p1")
    assert answer == "" and model == "m" and "TimeoutError" in reason


# ── the route ───────────────────────────────────────────────────────────────


def _client(monkeypatch, *, facade, spine: SpineAnswer, hidden=()):
    calls = {"spine": 0, "endpoint": 0, "kwargs": []}
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", facade)
    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u", "p1"))

    def fake_spine(*_a, **kwargs):
        calls["spine"] += 1
        calls["kwargs"].append(kwargs)
        return spine

    async def fake_endpoint(body, registry, project_id):
        calls["endpoint"] += 1
        return rag_api.RAGQueryResponse(answer="from the endpoint corpus")

    monkeypatch.setattr(rag_spine, "answer_from_spine", fake_spine)
    monkeypatch.setattr(rag_api, "_query_via_endpoint", fake_endpoint)
    monkeypatch.setattr(rag_retrieve, "hidden_graphs", lambda *_a: hidden)
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = object()
    app.dependency_overrides[rag_api.get_registry] = lambda: object()
    return TestClient(app), calls


_BODY = {"query": "what slows moisture?", "project_id": "p1"}


def test_auto_answers_from_the_spine_and_names_the_venue(monkeypatch):
    spine = SpineAnswer(answer="Alumina [1].", passages=_hydrated(), model="llama3.2:3b")
    client, calls = _client(monkeypatch, facade="auto", spine=spine)
    body = client.post("/umrs/rag/query", json=_BODY).json()
    assert body["answer"] == "Alumina [1]."
    source = body["sources"][0]
    assert source["content"] == _CHUNK and source["score"] == 0.8
    assert source["metadata"]["corpus_venue"] == "spine"
    assert source["metadata"]["source_name"] == "perovskite-note.md"
    assert (calls["spine"], calls["endpoint"]) == (1, 0)


def test_auto_falls_back_to_the_endpoint_corpus_when_the_spine_has_nothing(monkeypatch):
    client, calls = _client(monkeypatch, facade="auto", spine=SpineAnswer(reason="no vectors"))
    assert client.post("/umrs/rag/query", json=_BODY).json()["answer"] == "from the endpoint corpus"
    assert (calls["spine"], calls["endpoint"]) == (1, 1)


def test_spine_mode_says_why_there_is_no_answer(monkeypatch):
    client, calls = _client(monkeypatch, facade="spine", spine=SpineAnswer(reason="no vectors"))
    answer = client.post("/umrs/rag/query", json=_BODY).json()["answer"]
    assert answer == "No knowledge in this project answers this (no vectors)."
    assert calls["endpoint"] == 0


def test_generation_failure_still_returns_the_passages(monkeypatch):
    spine = SpineAnswer(passages=_hydrated(), model="m", reason="chat model 'm' failed: x")
    client, _ = _client(monkeypatch, facade="auto", spine=spine)
    body = client.post("/umrs/rag/query", json=_BODY).json()
    assert body["answer"] == "Could not generate an answer: chat model 'm' failed: x"
    assert len(body["sources"]) == 1


def test_endpoint_mode_is_the_old_proxy(monkeypatch):
    client, calls = _client(monkeypatch, facade="endpoint", spine=SpineAnswer())
    client.post("/umrs/rag/query", json=_BODY)
    assert (calls["spine"], calls["endpoint"]) == (0, 1)


def test_threshold_defaults_to_none_and_a_float_keeps_legacy_mode(monkeypatch):
    """ADR-093 D-2: the implicit 0.7 is gone; a float sent is the legacy cut."""
    client, calls = _client(monkeypatch, facade="spine", spine=SpineAnswer(reason="x"))
    client.post("/umrs/rag/query", json=_BODY)
    client.post("/umrs/rag/query", json={**_BODY, "threshold": 0.7})
    client.post("/umrs/rag/query", json={**_BODY, "top_k": 15})
    first, legacy, explicit_k = calls["kwargs"]
    assert (first["threshold"], first["k_max"]) == (None, None)
    assert (legacy["threshold"], legacy["k_max"]) == (0.7, None)
    assert explicit_k["k_max"] == 12  # adaptive k_max is capped at 12


# R4's RAGQuery.to_dict and the eval harness's E-B arm serialize top_k=5 on
# every call, whether or not anyone chose it.
_R4_BODY = {**_BODY, "top_k": 5, "include_sources": True}


@pytest.mark.parametrize(
    ("extra", "k_max"),
    [
        ({}, None),  # the legacy default: the site's UMRS_RAG_FINAL_K_MAX
        ({"top_k": 3}, 3),  # a chosen k is the adaptive cut's k_max
        ({"top_k": 15}, 12),  # ... at most 12
        ({"threshold": 0.7}, 5),  # threshold mode keeps top_k as sent
    ],
)
def test_the_legacy_default_top_k_leaves_the_site_k_max(monkeypatch, extra, k_max):
    """Review of ADR-093 (2026-10-10): every R4 and E-B call ran at k_max 5,
    never at the site's tuned UMRS_RAG_FINAL_K_MAX (default 4)."""
    client, calls = _client(monkeypatch, facade="spine", spine=SpineAnswer(reason="x"))
    client.post("/umrs/rag/query", json={**_R4_BODY, **extra})
    assert calls["kwargs"][0]["k_max"] == k_max


def test_the_query_leaves_out_graphs_the_caller_may_not_read(monkeypatch):
    """The compose path reads the retrieve verb's whole-project scope: the
    project minus another member's personal graph (KG-05)."""
    client, calls = _client(
        monkeypatch, facade="spine", spine=SpineAnswer(reason="x"), hidden=("g-u2-personal",)
    )
    client.post("/umrs/rag/query", json=_BODY)
    assert calls["kwargs"][0]["exclude_graph_ids"] == ("g-u2-personal",)


def test_the_exclusions_reach_the_retrieval_spec(monkeypatch):
    from umrs.knowledge.retrieval import pipeline

    seen = {}

    def retrieve(_pool, spec, **_kwargs):
        seen["spec"] = spec
        return RetrieveResult(query="q", sufficiency=RetrievalSufficiency(status="insufficient"))

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    rag_spine.answer_from_spine(
        object(), "q", "p1", encode=lambda _t: ("m", None, "x"), exclude_graph_ids=["g-u2"]
    )
    assert seen["spec"].exclude_graph_ids == ("g-u2",) and seen["spec"].graph_ids is None


@pytest.mark.parametrize("value", ["", "bogus"])
def test_unknown_facade_value_means_spine(monkeypatch, value):
    """Default since 2026-10-09: legacy corpora are disposable (maintainer)."""
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", value)
    assert rag_api._query_facade() == "spine"


def test_unset_facade_is_spine(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_QUERY_FACADE", raising=False)
    assert rag_api._query_facade() == "spine"


def test_the_endpoint_corpus_never_receives_a_null_threshold(monkeypatch):
    """The mini-RiVault service types threshold as a float: an unset one is
    omitted so it keeps its own 0.7 instead of answering 422."""
    sent = {}

    class _Response:
        status_code = 200
        text = ""

        def json(self):
            return {"answer": "a", "sources": []}

    class _AsyncClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def post(self, _url, json=None, **_kwargs):
            sent.update(json)
            return _Response()

    monkeypatch.setattr(rag_api.httpx, "AsyncClient", _AsyncClient)
    monkeypatch.setattr(rag_api, "_discover_rag_endpoint", lambda *_a: "http://rag")
    body = rag_api.RAGQueryRequest(query="q", project_id="p1")
    asyncio.run(rag_api._query_via_endpoint(body, object(), "p1"))
    assert sent["query"] == "q" and "threshold" not in sent
