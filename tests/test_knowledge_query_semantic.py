"""K3 vector lane: semantic search through the RAG service's query encoder.

``GET /umrs/knowledge/query`` searched the embeddings lane with a literal
ILIKE over model name and node label — "moisture damage" found nothing in a
note about "moisture ingress" (stand-alone stack, 2026-10-09). The facade
never loads a model (R6.1); it now asks the RAG service (``POST /rag/embed``)
for the query vector and ranks only vectors from that model's space
(ADR-083 D-2). With no encoder configured the lane stays the labelled table
reader; an encoder outage is reported, never answered lexically (D-4); a
space with no vectors is no evidence (D-5).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
from umrs.api import knowledge_query as kq

_WHEN = datetime(2026, 10, 9, tzinfo=timezone.utc)
_CHUNK = json.dumps({"kind": "chunk", "excerpt": "Encapsulation slows moisture ingress."})
_SIM_ROW = ("node-chunk-a-0", "/models/e5-base-v2", 768, _WHEN, "chunk 0 of a", "p1", 0.12, _CHUNK)
_TABLE_ROW = ("node-chunk-a-0", "/models/e5-base-v2", 768, _WHEN, "chunk 0 of a", "p1")


class _Cur:
    def __init__(self, rows_for):
        self.rows_for = rows_for
        self.queries: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        self._rows = self.rows_for(sql)

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Pool:
    def __init__(self, rows_for):
        self.cur = _Cur(rows_for)

    def getconn(self):
        pool = self

        class _Conn:
            def cursor(self):
                return pool.cur

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        return _Conn()

    def putconn(self, _conn):
        pass


def _rows(sim=None, table=None):
    return lambda sql: (sim or []) if "<=>" in sql else (table or [])


def _encoder(model="/models/e5-base-v2", vector=(0.1, 0.2), reason=""):
    calls = []

    def encode(query):
        calls.append(query)
        return model, (list(vector) if vector else None), reason

    encode.calls = calls
    return encode


# ── the lane ────────────────────────────────────────────────────────────────


def test_encoder_vector_runs_the_similarity_lane_in_one_space():
    pool = _Pool(_rows(sim=[_SIM_ROW]))
    enc = _encoder()
    backend = kq._EmbeddingsSearchBackend(pool, encoder=enc)
    hits = backend.search_embeddings("moisture damage", project_id="p1", limit=5)

    assert enc.calls == ["moisture damage"]
    sql, params = pool.cur.queries[0]
    assert "<=> %s::vector" in sql and "(e.model_name = %s OR e.model_name LIKE %s)" in sql
    assert params == ("[0.1,0.2]", "p1", "/models/e5-base-v2", "%/e5-base-v2", 5)
    assert hits[0]["metadata"]["lane"] == "similarity"
    assert hits[0]["score"] == pytest.approx(0.88)
    assert backend.lane_report == {"lane": "similarity", "model": "/models/e5-base-v2"}


def test_similarity_hit_carries_the_matched_text():
    backend = kq._EmbeddingsSearchBackend(_Pool(_rows(sim=[_SIM_ROW])), encoder=_encoder())
    hit = backend.search_embeddings("q", project_id="p1")[0]
    assert hit["content"] == "Encapsulation slows moisture ingress."


def test_a_vector_without_its_space_is_refused():
    """ADR-083 receipt 4: a vector query without a space filter fails a test."""
    backend = kq._EmbeddingsSearchBackend(_Pool(_rows()))
    with pytest.raises(ValueError, match="model_name"):
        backend.search_embeddings("q", project_id="p1", query_embedding=[0.1])


def test_no_encoder_keeps_the_labelled_table_reader():
    pool = _Pool(_rows(table=[_TABLE_ROW]))
    backend = kq._EmbeddingsSearchBackend(pool)
    hits = backend.search_embeddings("e5", project_id="p1")
    assert all(h["metadata"]["lane"] == "table-reader" for h in hits)
    assert "<=>" not in pool.cur.queries[0][0]
    assert backend.lane_report == {"lane": "table-reader", "reason": "no query encoder"}


def test_encoder_outage_is_unavailable_not_a_lexical_answer():
    """ADR-083 D-4: never a lexical match labelled "embeddings" when the encoder is down."""
    pool = _Pool(_rows(table=[_TABLE_ROW]))
    enc = _encoder(vector=None, model="", reason="query encoder unavailable (ConnectError)")
    backend = kq._EmbeddingsSearchBackend(pool, encoder=enc)
    assert backend.search_embeddings("q", project_id="p1") == []
    assert pool.cur.queries == []
    assert backend.lane_report == {
        "lane": "unavailable",
        "reason": "query encoder unavailable (ConnectError)",
    }


def test_empty_space_is_no_evidence_and_names_the_space():
    """Encoder model differs from the model that wrote the rows: nothing in its space."""
    pool = _Pool(_rows(sim=[], table=[_TABLE_ROW]))
    backend = kq._EmbeddingsSearchBackend(pool, encoder=_encoder(model="intfloat/e5-base-v2"))
    assert backend.search_embeddings("q", project_id="p1") == []
    assert len(pool.cur.queries) == 1  # no lexical second pass
    assert backend.lane_report["lane"] == "similarity"
    assert "'intfloat/e5-base-v2'" in backend.lane_report["reason"]


# ── the encoder client ──────────────────────────────────────────────────────


def test_encode_query_without_a_configured_service(monkeypatch):
    monkeypatch.delenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", raising=False)
    monkeypatch.delenv("UMRS_RAG_DIRECT_URL", raising=False)
    assert kq._encode_query("q") == ("", None, "no query encoder configured")


def test_encode_query_asks_the_rag_service(monkeypatch):
    monkeypatch.delenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", raising=False)
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081/")
    seen = {}

    def fake_post(url, json=None, timeout=None):  # noqa: A002 — httpx signature
        seen.update(url=url, json=json, timeout=timeout)
        return httpx.Response(
            200,
            json={"model": "/models/e5-base-v2", "dimension": 2, "vector": [0.5, 0.25]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(kq.httpx, "post", fake_post)
    assert kq._encode_query("moisture") == ("/models/e5-base-v2", [0.5, 0.25], "")
    assert seen["url"] == "http://rag:8081/rag/embed"
    assert seen["json"] == {"text": "moisture"}


def test_explicit_encoder_url_wins(monkeypatch):
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081")
    monkeypatch.setenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", "http://enc:9000")
    assert kq._query_encoder_url() == "http://enc:9000"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, json={"detail": "Not Found"}),
        httpx.Response(200, json={"dimension": 2}),
        httpx.Response(200, content=b"not json"),
    ],
)
def test_encoder_errors_become_a_reason(monkeypatch, response):
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081")
    monkeypatch.delenv("UMRS_KNOWLEDGE_QUERY_ENCODER_URL", raising=False)
    response.request = httpx.Request("POST", "http://rag:8081/rag/embed")
    monkeypatch.setattr(kq.httpx, "post", lambda *_a, **_k: response)
    model, vector, reason = kq._encode_query("q")
    assert vector is None and reason.startswith("query encoder unavailable")


def test_unreachable_encoder_becomes_a_reason(monkeypatch):
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081")

    def refuse(*_a, **_k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(kq.httpx, "post", refuse)
    assert kq._encode_query("q") == ("", None, "query encoder unavailable (ConnectError)")


# ── the RAG service route ───────────────────────────────────────────────────


class _FakeEmbeddings:
    model_name = "/models/e5-base-v2"

    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


def test_rag_embed_route_returns_model_and_vector(monkeypatch):
    from umrs.api import rag_service

    fake = _FakeEmbeddings()
    monkeypatch.setattr(rag_service, "EMBEDDINGS", fake)
    assert any(getattr(r, "path", "") == "/rag/embed" for r in rag_service.app.routes)
    response = rag_service.embed_query(rag_service.EmbedRequest(text="moisture"))
    assert response.model_dump() == {
        "model": "/models/e5-base-v2",
        "dimension": 3,
        "vector": [0.1, 0.2, 0.3],
    }
    # embed_query is the prefixing path (e5 "query: " is added inside it).
    assert fake.queries == ["moisture"]


def test_rag_embed_route_without_a_model(monkeypatch):
    from fastapi import HTTPException
    from umrs.api import rag_service

    monkeypatch.setattr(rag_service, "EMBEDDINGS", None)
    with pytest.raises(HTTPException) as exc_info:
        rag_service.embed_query(rag_service.EmbedRequest(text="q"))
    assert exc_info.value.status_code >= 500


def test_prefixed_embedder_adds_the_e5_query_prefix(monkeypatch):
    """The vector must be a *query* vector: chunks were embedded as passages."""
    from umrs.api import rag_service

    seen = []
    monkeypatch.setattr(
        rag_service.SentenceTransformerEmbeddings,
        "embed_query",
        lambda _self, text: seen.append(text) or [0.0],
    )
    emb = rag_service.PrefixedSentenceTransformerEmbeddings.__new__(
        rag_service.PrefixedSentenceTransformerEmbeddings
    )
    object.__setattr__(emb, "model_name", "/models/e5-base-v2")
    emb.embed_query("moisture")
    assert seen == ["query: moisture"]


def test_a_space_is_the_model_basename():
    """The bundled model, a copy staged elsewhere and the hub id are one space."""
    from umrs.knowledge.text_embedder import embedding_space

    names = {"/opt/umrs/models/e5-base-v2", "/models/e5-base-v2/", "intfloat/e5-base-v2"}
    assert {embedding_space(n) for n in names} == {"e5-base-v2"}


def test_like_wildcards_in_a_model_name_are_escaped():
    pool = _Pool(_rows(sim=[_SIM_ROW]))
    backend = kq._EmbeddingsSearchBackend(pool, encoder=_encoder(model="/m/all_MiniLM%v2"))
    backend.search_embeddings("q", project_id="p1")
    assert pool.cur.queries[0][1][3] == "%/all\\_MiniLM\\%v2"

