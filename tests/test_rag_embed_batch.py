"""POST /rag/embed: the batch form, and one query prefix per query (ADR-093 C8).

The retrieve verb encodes up to three query variants and the 30 null probes;
``{"texts": [...]}`` does that in one call. ``{"text"}`` is unchanged.

The prefix defect, found 2026-10-10: langchain's ``embed_query`` delegates to
``self.embed_documents``, which this service overrides to add the e5
``passage:`` prefix. Every query reached the model as ``passage: query: ...``.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from umrs.api import rag_service

E5 = "/opt/umrs/models/e5-base-v2"


class _Client:
    """The SentenceTransformer under the embedder: records what it encodes."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        self.calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts])


def _embedder(model_name: str = E5) -> tuple[object, _Client]:
    client = _Client()
    embedder = rag_service.PrefixedSentenceTransformerEmbeddings.model_construct(
        client=client,
        model_name=model_name,
        encode_kwargs={},
        multi_process=False,
        show_progress=False,
    )
    return embedder, client


# ── one prefix per query ────────────────────────────────────────────────────


def test_a_query_carries_the_e5_query_prefix_exactly_once():
    embedder, client = _embedder()
    embedder.embed_query("moisture sorption")
    assert client.calls == [["query: moisture sorption"]]


def test_a_batch_of_queries_is_one_pass_with_the_query_prefix():
    embedder, client = _embedder()
    vectors = embedder.embed_queries(["a b", "c"])
    assert client.calls == [["query: a b", "query: c"]]
    assert vectors == [[10.0, 1.0], [8.0, 1.0]]


def test_documents_still_carry_the_passage_prefix():
    embedder, client = _embedder()
    embedder.embed_documents(["chunk"])
    embedder.embed_query("q")
    embedder.embed_documents(["after"])
    assert client.calls == [["passage: chunk"], ["query: q"], ["passage: after"]]


def test_a_model_without_instruction_prefixes_gets_none():
    embedder, client = _embedder("/models/all-MiniLM-L6-v2")
    embedder.embed_query("q")
    embedder.embed_queries(["r"])
    assert client.calls == [["q"], ["r"]]


# ── the route ───────────────────────────────────────────────────────────────


@pytest.fixture
def http(monkeypatch):
    embedder, client = _embedder()
    monkeypatch.setattr(rag_service, "EMBEDDINGS", embedder)
    return TestClient(rag_service.app), client


def test_the_batch_form_answers_model_vectors_and_dim(http):
    api, client = http
    response = api.post("/rag/embed", json={"texts": ["a", "bb", "ccc"]})
    assert response.status_code == 200
    assert response.json() == {
        "model": E5,
        "vectors": [[8.0, 1.0], [9.0, 1.0], [10.0, 1.0]],
        "dim": 2,
    }
    assert len(client.calls) == 1


def test_the_single_form_is_unchanged(http):
    api, _client = http
    response = api.post("/rag/embed", json={"text": "a"})
    assert response.status_code == 200
    assert response.json() == {"model": E5, "dimension": 2, "vector": [8.0, 1.0]}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": "a", "texts": ["b"]},
        {"texts": []},
        {"texts": ["x"] * 65},
        {"texts": ["ok", ""]},
        {"texts": ["y" * 2049]},
    ],
)
def test_malformed_requests_are_422(http, body):
    api, _client = http
    assert api.post("/rag/embed", json=body).status_code == 422


def test_an_embedder_without_a_batch_method_is_called_per_query(monkeypatch):
    seen: list[str] = []

    class _Single:
        model_name = E5

        def embed_query(self, text):
            seen.append(text)
            return [1.0, 2.0, 3.0]

    monkeypatch.setattr(rag_service, "EMBEDDINGS", _Single())
    response = rag_service.embed_query(rag_service.EmbedRequest(texts=["a", "b"]))
    assert response.model_dump() == {
        "model": E5,
        "vectors": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        "dim": 3,
    }
    assert seen == ["a", "b"]


# ── the B0 encoding, kept as an eval-only switch (UMRS_RAG_QUERY_PREFIX) ─────


class _HeadEmbedder(rag_service.SentenceTransformerEmbeddings):
    """The embedder B0 ran (umrs 809387a1), verbatim: langchain's
    ``embed_query`` calls ``self.embed_documents``, which adds the passage
    prefix to the already query-prefixed text."""

    def embed_query(self, text):
        text = rag_service.add_query_prefix(text, self.model_name)
        return super().embed_query(text)

    def embed_documents(self, texts):
        texts = rag_service.add_document_prefix(texts, self.model_name)
        return super().embed_documents(texts)


def _head(model_name: str = E5) -> tuple[object, _Client]:
    client = _Client()
    embedder = _HeadEmbedder.model_construct(
        client=client,
        model_name=model_name,
        encode_kwargs={},
        multi_process=False,
        show_progress=False,
    )
    return embedder, client


@pytest.fixture
def b0(monkeypatch):
    monkeypatch.setattr(rag_service, "QUERY_PREFIX", "b0")


def _head_string(query: str, model_name: str) -> str:
    """The one string the B0 embedder sent the model for ``query``."""
    reference, seen = _head(model_name)
    reference.embed_query(query)
    [[text]] = seen.calls
    return text


@pytest.mark.parametrize("model_name", [E5, "/models/all-MiniLM-L6-v2"])
@pytest.mark.usefixtures("b0")
def test_b0_mode_encodes_every_query_exactly_as_the_b0_tree_did(model_name):
    """B0's facade sent one ``{"text"}`` per query; ``b0`` encodes each query,
    alone or in a batch, as that embedder did."""
    embedder, client = _embedder(model_name)
    queries = ["moisture sorption", "IL-6\nlevels", "query: already"]
    for query in queries:
        embedder.embed_query(query)
    embedder.embed_queries(queries)
    expected = [_head_string(query, model_name) for query in queries]
    assert client.calls == [[text] for text in expected] + [expected]


@pytest.mark.usefixtures("b0")
def test_b0_mode_sends_passage_query_for_one_and_for_many():
    embedder, client = _embedder()
    embedder.embed_query("x")
    embedder.embed_queries(["y", "z"])
    embedder.embed_documents(["chunk"])
    assert client.calls == [
        ["passage: query: x"],
        ["passage: query: y", "passage: query: z"],
        ["passage: chunk"],
    ]


@pytest.mark.parametrize(
    ("mode", "single", "batch"),
    [
        ("fixed", [["query: q"]], [["query: a", "query: b"]]),
        ("b0", [["passage: query: q"]], [["passage: query: a", "passage: query: b"]]),
    ],
)
def test_the_route_pins_the_string_the_model_sees(http, monkeypatch, mode, single, batch):
    api, client = http
    monkeypatch.setattr(rag_service, "QUERY_PREFIX", mode)
    assert api.post("/rag/embed", json={"text": "q"}).status_code == 200
    assert client.calls == single
    client.calls.clear()
    assert api.post("/rag/embed", json={"texts": ["a", "b"]}).status_code == 200
    assert client.calls == batch


@pytest.mark.parametrize(
    ("raw", "mode"),
    [(None, "fixed"), ("", "fixed"), ("fixed", "fixed"), (" B0 ", "b0"), ("b0", "b0")],
)
def test_the_switch_defaults_to_the_fix(monkeypatch, raw, mode):
    if raw is None:
        monkeypatch.delenv("UMRS_RAG_QUERY_PREFIX", raising=False)
    else:
        monkeypatch.setenv("UMRS_RAG_QUERY_PREFIX", raw)
    assert rag_service.query_prefix_mode() == mode


def test_an_unknown_switch_value_refuses_to_start(monkeypatch):
    """A typo must not silently measure the other encoder."""
    monkeypatch.setenv("UMRS_RAG_QUERY_PREFIX", "legacy")
    with pytest.raises(ValueError, match="UMRS_RAG_QUERY_PREFIX"):
        rag_service._init_query_prefix()


def test_startup_applies_the_switch_and_says_so(monkeypatch, caplog):
    monkeypatch.setattr(rag_service, "QUERY_PREFIX", "fixed")
    monkeypatch.setenv("UMRS_RAG_QUERY_PREFIX", "b0")
    with caplog.at_level("WARNING", logger=rag_service.logger.name):
        rag_service._init_query_prefix()
    assert rag_service.QUERY_PREFIX == "b0"
    assert "passage: query:" in caplog.text


@pytest.mark.parametrize("mode", ["fixed", "b0"])
def test_stats_name_the_encoding_the_model_actually_sees(monkeypatch, mode):
    embedder, client = _embedder()
    monkeypatch.setattr(rag_service, "EMBEDDINGS", embedder)
    monkeypatch.setattr(rag_service, "QUERY_PREFIX", mode)
    stats = rag_service._embedder_stats()
    embedder.embed_query("{q}")
    assert stats["query_prefix"] == mode
    assert stats["query_encoding"] == client.calls[-1][0]


# ── /rag/upload: a document carries the passage prefix once ─────────────────


def test_an_uploaded_chunk_carries_the_passage_prefix_once(monkeypatch):
    embedder, client = _embedder()
    monkeypatch.setattr(rag_service, "EMBEDDINGS", embedder)
    chunks, vectors = rag_service._chunk_and_embed("Perovskite films degrade.")
    assert [c["text"] for c in chunks] == ["Perovskite films degrade."]
    assert client.calls == [["passage: Perovskite films degrade."]]
    assert len(vectors) == 1
