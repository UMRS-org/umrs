"""POST /rag/rerank and the reranker line of /rag/stats (ADR-093 C8, D-7).

The rag service scores; UMRS orders and cuts. The route returns raw logits
with the model and revision that produced them, or 503 ``reranker
unavailable: <reason>``, never the input order in disguise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from umrs.api import rag_service
from umrs.rag import search
from umrs.rag.search import CrossEncoderReranker

REVISION = "ce0a1b2c3d4e"


def _staged(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"weights")
    meta = path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    (meta / "config.json.metadata").write_text(f'{REVISION}\n"etag"\n1760000000.0\n')
    return path


class _Model:
    def __init__(self, logits=None, error: Exception | None = None):
        self.logits = logits or []
        self.error = error
        self.pairs: list = []

    def predict(self, pairs, **_kwargs):
        if self.error is not None:
            raise self.error
        self.pairs = list(pairs)
        return self.logits[: len(pairs)]


def _loaded(tmp_path: Path, model: _Model) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(str(_staged(tmp_path / "ms-marco-MiniLM-L6-v2")))
    reranker._model = model
    return reranker


@pytest.fixture
def client():
    return TestClient(rag_service.app)  # no `with`: the startup hook stays unrun


def _use(monkeypatch, reranker) -> None:
    monkeypatch.setattr(rag_service, "RERANKER", reranker)


# ── the route ───────────────────────────────────────────────────────────────


def test_rerank_returns_raw_logits_with_model_and_revision(monkeypatch, tmp_path, client):
    reranker = _loaded(tmp_path, _Model([3.25, -4.5]))
    _use(monkeypatch, reranker)
    response = client.post("/rag/rerank", json={"query": "q", "passages": ["a", "b"]})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == reranker.model_name
    assert body["revision"] == REVISION
    assert body["logits"] == [3.25, -4.5]
    assert isinstance(body["ms"], float)


def test_long_passages_are_truncated_before_scoring(monkeypatch, tmp_path, client):
    model = _Model([1.0])
    _use(monkeypatch, _loaded(tmp_path, model))
    client.post("/rag/rerank", json={"query": "q", "passages": ["x" * 9000]})
    assert len(model.pairs[0][1]) == 4000


@pytest.mark.parametrize(
    ("model_name", "reason"),
    [
        ("", "not configured"),
        ("cross-encoder/ms-marco-MiniLM-L6-v2", "hub id"),
        ("/models/never-staged", "does not exist"),
    ],
)
def test_an_unstaged_reranker_answers_503_with_the_reason(monkeypatch, client, model_name, reason):
    _use(monkeypatch, CrossEncoderReranker(model_name))
    response = client.post("/rag/rerank", json={"query": "q", "passages": ["a"]})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail.startswith("reranker unavailable: ") and reason in detail


def test_an_empty_probe_still_tells_the_truth(monkeypatch, client):
    _use(monkeypatch, CrossEncoderReranker(""))
    response = client.post("/rag/rerank", json={"query": "q", "passages": []})
    assert response.status_code == 503


def test_before_startup_the_route_says_so(monkeypatch, client):
    _use(monkeypatch, None)
    response = client.post("/rag/rerank", json={"query": "q", "passages": ["a"]})
    assert response.status_code == 503
    assert "service not started" in response.json()["detail"]


def test_a_scoring_failure_is_an_outage_not_a_500(monkeypatch, tmp_path, client):
    _use(monkeypatch, _loaded(tmp_path, _Model(error=RuntimeError("out of memory"))))
    response = client.post("/rag/rerank", json={"query": "q", "passages": ["a"]})
    assert response.status_code == 503
    assert "scoring failed: RuntimeError" in response.json()["detail"]


def test_more_than_32_passages_is_refused(monkeypatch, tmp_path, client):
    _use(monkeypatch, _loaded(tmp_path, _Model([0.0] * 40)))
    response = client.post("/rag/rerank", json={"query": "q", "passages": ["p"] * 33})
    assert response.status_code == 422


# ── /rag/stats ──────────────────────────────────────────────────────────────


def test_stats_name_the_model_and_revision_when_staged(monkeypatch, tmp_path):
    reranker = _loaded(tmp_path, _Model([0.0]))
    _use(monkeypatch, reranker)
    assert rag_service._reranker_stats() == {
        "reranker": {"model": reranker.model_name, "revision": REVISION}
    }


def test_stats_say_unavailable_not_staged_with_the_reason(monkeypatch):
    _use(monkeypatch, CrossEncoderReranker(""))
    stats = rag_service._reranker_stats()
    assert stats["reranker"] == "unavailable (not staged)"
    assert "not configured" in stats["reranker_reason"]


def test_stats_say_when_the_load_failed(monkeypatch, tmp_path):
    reranker = CrossEncoderReranker(str(_staged(tmp_path / "m")))
    monkeypatch.setattr(search, "_load_cross_encoder", _raise_os_error)
    with pytest.raises(search.RerankerUnavailable):
        reranker.score("q", ["p"])
    _use(monkeypatch, reranker)
    assert rag_service._reranker_stats()["reranker"] == "unavailable (load failed)"


def _raise_os_error(*_args):
    raise OSError("bad header")


def test_the_stats_route_carries_the_reranker_line(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    _use(monkeypatch, CrossEncoderReranker(""))
    stats = rag_service.rag_stats(project_id=None)
    assert stats["reranker"] == "unavailable (not staged)"


# ── startup ─────────────────────────────────────────────────────────────────


def test_startup_builds_no_reranker_from_an_unset_variable(monkeypatch):
    monkeypatch.delenv("UMRS_RAG_RERANKER_MODEL", raising=False)
    reranker = rag_service._init_reranker()
    assert reranker.model_name == ""
    assert reranker._model is None


def test_startup_warms_a_staged_reranker(monkeypatch, tmp_path):
    loads: list = []
    monkeypatch.setenv("UMRS_RAG_RERANKER_MODEL", str(_staged(tmp_path / "m")))
    monkeypatch.setattr(search, "_set_torch_threads", lambda _threads: None)
    monkeypatch.setattr(
        search, "_load_cross_encoder", lambda *args: loads.append(args) or _Model([1.0])
    )
    reranker = rag_service._init_reranker()
    assert len(loads) == 1 and reranker.unavailable_reason() is None
