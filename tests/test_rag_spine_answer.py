"""`POST /umrs/rag/query` answers from the knowledge spine (ADR-027 D3/P2).

The route proxied only to the per-experience RAG store, which uploads stopped
feeding in WP-15: an uploaded document could be searched but never answered
from — "No documents retrieved" on the stand-alone stack (2026-10-09).
"""

from __future__ import annotations

import hashlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_spine
from umrs.api.rag_spine import SpineAnswer

_DOC = "  # Perovskite note\n\nEncapsulation with alumina slows moisture ingress. Dr. Tanaka measured 1200 h.\n"
_TEXT = _DOC.strip()  # what the embed job chunked (decode + strip)
_CHUNK = _TEXT[0:80]
_MD5 = hashlib.md5(_CHUNK.encode()).hexdigest()
_REF = "projects/p1/knowledge/artifacts/abc"


def _content(**overrides):
    content = {
        "kind": "chunk",
        "artifact_id": "art-1",
        "chunk_start": 0,
        "chunk_end": 80,
        "chunk_md5": _MD5,
        "excerpt": _CHUNK[:20],
    }
    content.update(overrides)
    return json.dumps(content)


class _Pool:
    def __init__(self, rows):
        self.rows = rows
        self.sql = []

    def getconn(self):
        pool = self

        class _Cur:
            def execute(self, sql, params=None):
                pool.sql.append((sql, params))

            def fetchall(self):
                return pool.rows

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


def _passage(node_id="node-chunk-art-1-0", score=0.8):
    return {
        "title": "chunk 0 of art-1",
        "content": "short",
        "score": score,
        "source_id": node_id,
        "metadata": {"lane": "similarity"},
    }


# ── hydration ───────────────────────────────────────────────────────────────


def test_full_chunk_text_comes_from_the_artifact_md5_checked():
    pool = _Pool([("node-chunk-art-1-0", _content(), _REF, "perovskite-note.md", None)])
    fetched = []
    kept = rag_spine.attach_texts(
        pool, [_passage()], lambda ref: fetched.append(ref) or _DOC.encode()
    )
    assert kept[0]["text"] == _CHUNK
    assert kept[0]["metadata"]["text_source"] == "artifact"
    assert kept[0]["source_name"] == "perovskite-note.md"
    assert fetched == [_REF]


def test_md5_mismatch_falls_back_to_the_stored_excerpt():
    pool = _Pool([("node-chunk-art-1-0", _content(chunk_md5="0" * 32), _REF, "", None)])
    kept = rag_spine.attach_texts(pool, [_passage()], lambda _ref: _DOC.encode())
    assert kept[0]["text"] == _CHUNK[:20]
    assert kept[0]["metadata"]["text_source"] == "excerpt"


def test_a_passage_with_no_text_is_dropped_not_padded():
    """Legacy-canonized chunk nodes carry offsets only and no artifact link."""
    content = json.dumps({"kind": "chunk", "chunk_start": 0, "chunk_end": 9, "chunk_md5": "x"})
    pool = _Pool([("node-legacy-chunk0", content, None, None, None)])
    assert rag_spine.attach_texts(pool, [_passage("node-legacy-chunk0")], lambda _r: None) == []


def test_name_falls_back_to_the_staged_upload_key():
    staged = "s3://r4-data/knowledge-uploads/p1/u/zeolite-note.md"
    pool = _Pool([("node-chunk-art-1-0", _content(), _REF, None, staged)])
    kept = rag_spine.attach_texts(pool, [_passage()], lambda _r: _DOC.encode())
    assert kept[0]["source_name"] == "zeolite-note.md"


def test_each_artifact_is_fetched_once():
    rows = [
        ("n0", _content(), _REF, "a.md", None),
        ("n1", _content(), _REF, "a.md", None),
    ]
    fetched = []
    rag_spine.attach_texts(
        _Pool(rows), [_passage("n0"), _passage("n1")], lambda r: fetched.append(r) or _DOC.encode()
    )
    assert fetched == [_REF]


# ── retrieval ───────────────────────────────────────────────────────────────


class _Backend:
    def __init__(self, hits, lane=None):
        self.hits = hits
        self.lane_report = lane

    def search_embeddings(self, _query, **kwargs):
        return self.hits[: kwargs["limit"]]


def _with_backend(monkeypatch, backend):
    from umrs.api import knowledge_query as kq

    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081")
    monkeypatch.setattr(kq, "_embeddings_backend_for", lambda *_a, **_k: (backend, None))


def test_retrieval_needs_a_query_encoder(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_DIRECT_URL", raising=False)
    monkeypatch.delenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", raising=False)
    assert rag_spine.retrieve_passages(object(), "q", "p1", top_k=5, threshold=0.7) == (
        [],
        "no query encoder configured",
    )


def test_retrieval_keeps_passages_at_or_above_the_threshold(monkeypatch):
    _with_backend(monkeypatch, _Backend([_passage(score=0.8), _passage("n2", score=0.6)]))
    kept, reason = rag_spine.retrieve_passages(object(), "q", "p1", top_k=5, threshold=0.7)
    assert [p["source_id"] for p in kept] == ["node-chunk-art-1-0"] and reason == ""


def test_retrieval_says_why_nothing_qualified(monkeypatch):
    _with_backend(monkeypatch, _Backend([_passage(score=0.5)]))
    assert rag_spine.retrieve_passages(object(), "q", "p1", top_k=5, threshold=0.7)[1] == (
        "no passage scored at or above 0.7"
    )
    _with_backend(monkeypatch, _Backend([], lane={"lane": "unavailable", "reason": "encoder down"}))
    assert rag_spine.retrieve_passages(object(), "q", "p1", top_k=5, threshold=0.7)[1] == (
        "encoder down"
    )


# ── generation ──────────────────────────────────────────────────────────────


def _hydrated():
    passage = _passage()
    passage.update(text=_CHUNK, source_name="perovskite-note.md")
    return [passage]


def test_answer_uses_the_registry_model_and_numbered_passages(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge import agent_executor

    seen = {}
    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: ("http://ollama:11434", "llama3.2:3b"))

    def chat(messages, **kwargs):
        seen.update(messages=messages, **kwargs)
        return "<think>hmm</think>Alumina encapsulation slows it [1]."

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    answer, model, reason = rag_spine.compose_answer("what slows moisture?", _hydrated(), "p1")
    assert (answer, model, reason) == ("Alumina encapsulation slows it [1].", "llama3.2:3b", "")
    assert seen["model"] == "llama3.2:3b" and seen["base_url"] == "http://ollama:11434"
    user = seen["messages"][1]["content"]
    assert "[1] perovskite-note.md\n" + _CHUNK in user
    assert user.startswith("Question: what slows moisture?")


def test_no_chat_endpoint_is_a_reason(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka

    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: None)
    assert rag_spine.compose_answer("q", _hydrated(), "p1") == (
        "",
        "",
        "no READY chat endpoint in the registry",
    )


def test_chat_failure_is_a_reason_naming_the_model(monkeypatch):
    from umrs.jobs import knowledge_adapters as ka
    from umrs.knowledge import agent_executor

    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _project: ("http://x", "m"))

    def boom(*_a, **_k):
        raise TimeoutError("slow")

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", boom)
    answer, model, reason = rag_spine.compose_answer("q", _hydrated(), "p1")
    assert answer == "" and model == "m" and "TimeoutError" in reason


# ── the route ───────────────────────────────────────────────────────────────


def _client(monkeypatch, *, facade, spine: SpineAnswer):
    calls = {"spine": 0, "endpoint": 0}
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", facade)
    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u", "p1"))

    def fake_spine(*_a, **_k):
        calls["spine"] += 1
        return spine

    async def fake_endpoint(body, registry, project_id):
        calls["endpoint"] += 1
        return rag_api.RAGQueryResponse(answer="from the endpoint corpus")

    monkeypatch.setattr(rag_spine, "answer_from_spine", fake_spine)
    monkeypatch.setattr(rag_api, "_query_via_endpoint", fake_endpoint)
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
    assert source["content"] == _CHUNK
    assert source["metadata"]["corpus_venue"] == "spine"
    assert source["metadata"]["source_name"] == "perovskite-note.md"
    assert calls == {"spine": 1, "endpoint": 0}


def test_auto_falls_back_to_the_endpoint_corpus_when_the_spine_has_nothing(monkeypatch):
    client, calls = _client(monkeypatch, facade="auto", spine=SpineAnswer(reason="no vectors"))
    assert client.post("/umrs/rag/query", json=_BODY).json()["answer"] == "from the endpoint corpus"
    assert calls == {"spine": 1, "endpoint": 1}


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
    assert calls == {"spine": 0, "endpoint": 1}


@pytest.mark.parametrize("value", ["", "bogus"])
def test_unknown_facade_value_means_auto(monkeypatch, value):
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", value)
    assert rag_api._query_facade() == "auto"
