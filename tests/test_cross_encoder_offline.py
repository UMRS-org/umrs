"""The cross-encoder loads only a staged directory and returns raw logits (ADR-093 C8).

Before 2026-10-10 ``CrossEncoderReranker`` defaulted to a hub id. Under
``HF_HUB_OFFLINE=1`` that can never load, so every request retried the load,
logged an error and silently returned the input order. It also returned
sigmoid scores, which leaves the cut nothing to calibrate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from umrs.rag import search
from umrs.rag.search import (
    CrossEncoderReranker,
    RerankerUnavailable,
    SearchResult,
    reranker_unavailable_reason,
    staged_revision,
)

HUB_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _staged(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    return path


class _Spy:
    """Stands in for the sentence-transformers load; records every call."""

    def __init__(self, result=None, error: Exception | None = None):
        self.calls: list[tuple] = []
        self.result = result
        self.error = error

    def __call__(self, *args):
        self.calls.append(args)
        if self.error is not None:
            raise self.error
        return self.result


class _Model:
    def __init__(self, logits):
        self.logits = logits
        self.kwargs: dict = {}

    def predict(self, pairs, **kwargs):
        self.kwargs = kwargs
        return self.logits[: len(pairs)]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_RERANKER_MODEL", raising=False)
    monkeypatch.delenv("UMRS_RAG_RERANKER_THREADS", raising=False)
    monkeypatch.setattr(search, "_set_torch_threads", lambda _threads: None)


# ── offline: only a staged directory loads ──────────────────────────────────


def test_a_hub_id_is_refused_by_name_and_never_loaded(monkeypatch):
    spy = _Spy(result=_Model([1.0]))
    monkeypatch.setattr(search, "_load_cross_encoder", spy)
    reranker = CrossEncoderReranker(HUB_ID)
    with pytest.raises(RerankerUnavailable, match="hub id"):
        reranker.score("q", ["p"])
    assert spy.calls == []
    assert HUB_ID in reranker.unavailable_reason()


def test_no_setting_means_no_reranker_not_a_default_hub_id():
    reranker = CrossEncoderReranker()
    assert reranker.model_name == ""
    assert "not configured" in reranker.unavailable_reason()


def test_the_setting_is_read_from_the_environment(monkeypatch, tmp_path):
    staged = _staged(tmp_path / "ms-marco-MiniLM-L6-v2")
    monkeypatch.setenv("UMRS_RAG_RERANKER_MODEL", f" {staged} ")
    reranker = CrossEncoderReranker()
    assert reranker.model_name == str(staged)
    assert reranker.unavailable_reason() is None


def test_a_directory_without_weights_is_not_staged(tmp_path):
    partial = tmp_path / "half"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    assert "not staged" in reranker_unavailable_reason(str(partial))
    assert "does not exist" in reranker_unavailable_reason(str(tmp_path / "absent"))


def test_a_load_failure_is_cached_not_retried(monkeypatch, tmp_path):
    spy = _Spy(error=OSError("corrupt safetensors header"))
    monkeypatch.setattr(search, "_load_cross_encoder", spy)
    reranker = CrossEncoderReranker(str(_staged(tmp_path / "m")))
    for _ in range(3):
        with pytest.raises(RerankerUnavailable, match="load failed: OSError"):
            reranker.score("q", ["p"])
    assert len(spy.calls) == 1
    assert reranker.load_error.startswith("load failed")


# ── the documented runtime parameters ───────────────────────────────────────


def test_scoring_runs_at_the_documented_parameters(monkeypatch, tmp_path):
    model = _Model([2.5, -7.0])
    spy = _Spy(result=model)
    threads: list[int] = []
    monkeypatch.setattr(search, "_load_cross_encoder", spy)
    monkeypatch.setattr(search, "_set_torch_threads", threads.append)
    monkeypatch.setenv("UMRS_RAG_RERANKER_THREADS", "3")
    staged = _staged(tmp_path / "m")
    reranker = CrossEncoderReranker(str(staged))
    assert reranker.score("q", ["a", "b"]) == [2.5, -7.0]
    assert spy.calls == [(str(staged), "cpu", 384)]
    assert model.kwargs["batch_size"] == 16
    assert threads == [3]


def test_threads_default_to_the_rag_containers_two_cpus(monkeypatch):
    assert CrossEncoderReranker("").threads == 2
    monkeypatch.setenv("UMRS_RAG_RERANKER_THREADS", "not-a-number")
    assert CrossEncoderReranker("").threads == 2


def test_the_revision_comes_from_the_staging_metadata(tmp_path):
    staged = _staged(tmp_path / "m")
    assert staged_revision(staged) == "unknown"
    meta = staged / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text('ce0a1b2c\n"etag"\n1760000000.0\n')
    assert CrossEncoderReranker(str(staged)).revision == "ce0a1b2c"


# ── raw logits from a real cross-encoder ────────────────────────────────────


def _tiny_cross_encoder(path: Path) -> object:
    """A one-layer BERT cross-encoder built locally; no network, ~100 kB."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("sentence_transformers")
    path.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "what", "is", "a", "cell"]
    (path / "vocab.txt").write_text("\n".join(words) + "\n")
    transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=len(words),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
    )
    model = transformers.BertForSequenceClassification(config)
    model.classifier.bias.data.fill_(5.0)  # logits near 5; a sigmoid gives 0.99
    model.save_pretrained(path, safe_serialization=True)
    return model.eval()


def test_a_staged_cross_encoder_returns_raw_logits_not_probabilities(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    model = _tiny_cross_encoder(tmp_path / "tiny-ce")
    reranker = CrossEncoderReranker(str(tmp_path / "tiny-ce"))
    logits = reranker.score("what is a cell", ["a cell", "what"])
    tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(
        tmp_path / "tiny-ce"
    )
    batch = tokenizer(["what is a cell"] * 2, ["a cell", "what"], padding=True, return_tensors="pt")
    with torch.no_grad():
        expected = model(**batch).logits.reshape(-1).tolist()
    assert logits == pytest.approx(expected, abs=1e-4)
    assert all(value > 1.0 for value in logits)  # not squashed into (0, 1)
    limit = getattr(reranker._model, "max_seq_length", None)  # sentence-transformers >= 6
    assert (limit if limit is not None else reranker._model.max_length) == 384


# ── the legacy /rag/search and /rag/qa path ─────────────────────────────────


def test_legacy_rerank_keeps_the_order_but_says_why(caplog):
    results = [SearchResult(document_id=d, chunk_id=d, content=d, score=0.0) for d in "abc"]
    with caplog.at_level(logging.WARNING, logger=search.logger.name):
        out = CrossEncoderReranker(HUB_ID).rerank("q", results, top_k=2)
    assert [r.document_id for r in out] == ["a", "b"]
    assert all(r.metadata["rerank"].startswith("unavailable: refused") for r in results)
    assert caplog.text.count("rerank skipped") == 1


def test_legacy_rerank_sorts_signed_logits_with_unscored_last():
    results = [SearchResult(document_id=d, chunk_id=d, content=d, score=0.0) for d in "abc"]
    reranker = CrossEncoderReranker("")
    reranker._model = _Model([-3.0, -1.0])  # the third result gets no score
    out = reranker.rerank("q", results)
    assert [r.document_id for r in out] == ["b", "a", "c"]
    assert out[-1].rerank_score is None
