"""The rerank lane of the retrieval core (ADR-093 C8, D-7).

Pool: the top 30 fused candidates, at most 3 per artifact. Order: by logit
(``replace``) or ``1/(10+r_ce) + 0.5/(10+r_fused)`` (``interpolate``). An
outage is a lane report with a reason, never the input order relabelled.
"""

from __future__ import annotations

import httpx
import pytest
from umrs.knowledge.retrieval import rerank as rr
from umrs.knowledge.retrieval.rerank import (
    http_reranker,
    rerank_enabled,
    rerank_lane,
    reranker_from_env,
    select_pool,
)
from umrs.knowledge.retrieval.types import Candidate


def _cands(spec: str) -> list[Candidate]:
    """``"a1 a2 b1"`` -> candidates n0.. of artifacts a, a, b, with text."""
    return [
        Candidate(node_id=f"n{i}", artifact_id=tag[0], text=f"text {tag}")
        for i, tag in enumerate(spec.split())
    ]


class _Fake:
    def __init__(self, logits=None, info=None, error: Exception | None = None):
        self.logits = logits
        self.info = (
            info if info is not None else {"model": "/models/ms", "revision": "r1", "ms": 4.0}
        )
        self.error = error
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, query, passages):
        self.calls.append((query, list(passages)))
        if self.error is not None:
            raise self.error
        return self.logits, self.info


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("UMRS_RAG_RERANK_MODE", "UMRS_RAG_RERANK_DEFAULT", "UMRS_RAG_RERANKER_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("UMRS_RAG_DIRECT_URL", raising=False)


# ── the pool ────────────────────────────────────────────────────────────────


def test_the_pool_takes_at_most_three_per_artifact_within_the_limit():
    cands = _cands("a1 a2 a3 a4 b1 c1 c2")
    pool, rest = select_pool(cands, limit=5, per_artifact=3)
    assert [c.node_id for c in pool] == ["n0", "n1", "n2", "n4", "n5"]
    assert [c.node_id for c in rest] == ["n3", "n6"]


def test_candidates_without_an_artifact_count_one_each():
    cands = [Candidate(node_id=f"x{i}") for i in range(4)]
    pool, _rest = select_pool(cands, limit=30, per_artifact=1)
    assert len(pool) == 4


# ── ordering ────────────────────────────────────────────────────────────────


def test_replace_orders_the_pool_by_logit_and_appends_the_unscored_rest():
    cands = _cands("a1 b1 c1 a2 a3 a4")
    fake = _Fake(logits=[-2.0, 5.0, 0.5, 1.0, -9.0])
    result = rerank_lane("what is it", cands, fake, limit=30, per_artifact=3)
    assert result.status == "ok"
    assert [c.node_id for c in result.candidates] == ["n1", "n3", "n2", "n0", "n4", "n5"]
    assert result.candidates[0].rerank_logit == 5.0
    assert result.candidates[0].lane_ranks["rerank"] == 1
    assert result.candidates[-1].rerank_logit is None  # a4: over the per-artifact cap
    assert fake.calls[0][1] == ["text a1", "text b1", "text c1", "text a2", "text a3"]
    assert result.report() == {
        "status": "ok",
        "n": 6,
        "model": "/models/ms",
        "revision": "r1",
        "mode": "replace",
        "scored": 5,
        "model_ms": 4.0,
    }


def test_interpolate_blends_the_cross_encoder_rank_with_the_fused_rank():
    # r_ce: n2=1, n1=2, n0=3. Blend: n0 1/13+0.5/11=.1224, n1 1/12+0.5/12=.125,
    # n2 1/11+0.5/13=.1294 -> n2, n1, n0.
    cands = _cands("a1 b1 c1")
    result = rerank_lane("q", cands, _Fake(logits=[0.0, 1.0, 2.0]), mode="interpolate")
    assert [c.node_id for c in result.candidates] == ["n2", "n1", "n0"]
    assert result.info["mode"] == "interpolate"


def test_interpolate_keeps_a_fused_leader_that_replace_would_demote():
    # The CE's favourite is the 12th fused candidate: 1/11 + 0.5/22 = .114
    # loses to the fused leader at CE rank 2: 1/12 + 0.5/11 = .129.
    logits = [0.0] * 11 + [0.1]
    tags = " ".join(f"{chr(97 + i)}1" for i in range(12))
    blended = rerank_lane("q", _cands(tags), _Fake(logits=logits), mode="interpolate")
    replaced = rerank_lane("q", _cands(tags), _Fake(logits=logits), mode="replace")
    assert blended.candidates[0].node_id == "n0"
    assert replaced.candidates[0].node_id == "n11"


def test_the_mode_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("UMRS_RAG_RERANK_MODE", "interpolate")
    result = rerank_lane("q", _cands("a1"), _Fake(logits=[1.0]))
    assert result.info["mode"] == "interpolate"
    monkeypatch.setenv("UMRS_RAG_RERANK_MODE", "bogus")
    assert rerank_lane("q", _cands("a1"), _Fake(logits=[1.0])).info["mode"] == "replace"


def test_an_unhydrated_candidate_falls_back_to_its_excerpt():
    cand = Candidate(node_id="n0", content={"excerpt": "the excerpt"})
    fake = _Fake(logits=[1.0])
    rerank_lane("q", [cand], fake)
    assert fake.calls[0][1] == ["the excerpt"]


# ── never silent ────────────────────────────────────────────────────────────


def test_no_reranker_is_unavailable_with_a_reason_and_no_candidates():
    result = rerank_lane("q", _cands("a1 b1"), None)
    assert (result.status, result.candidates) == ("unavailable", [])
    assert result.reason == "no reranker configured"


def test_an_unstaged_model_is_reported_by_the_reason_the_service_gave():
    fake = _Fake(logits=None, info={"reason": "reranker unavailable: not configured"})
    result = rerank_lane("q", _cands("a1 b1"), fake)
    assert result.status == "unavailable"
    assert result.reason == "reranker unavailable: not configured"
    assert result.candidates == []


@pytest.mark.parametrize(
    ("fake", "reason"),
    [
        (_Fake(error=httpx.ConnectError("refused")), "reranker failed: ConnectError"),
        (_Fake(logits=[1.0]), "returned 1 logits for 2 passages"),
    ],
)
def test_a_failed_or_short_answer_is_unavailable(fake, reason):
    cands = _cands("a1 b1")
    result = rerank_lane("q", cands, fake)
    assert result.status == "unavailable" and reason in result.reason
    assert all(c.rerank_logit is None for c in cands)


def test_a_mostly_non_ascii_query_turns_the_english_model_off():
    fake = _Fake(logits=[1.0])
    result = rerank_lane("細胞の水分吸着とは", _cands("a1"), fake)
    assert result.status == "off" and "English-only" in result.reason
    assert fake.calls == []
    assert rerank_lane("BRCA1 の役割 in cancer", _cands("a1"), fake).status == "ok"


def test_auto_follows_the_site_default_which_is_off(monkeypatch):
    assert (rerank_enabled("auto"), rerank_enabled("on"), rerank_enabled("off")) == (
        False,
        True,
        False,
    )
    monkeypatch.setenv("UMRS_RAG_RERANK_DEFAULT", "on")
    assert rerank_enabled("auto") is True


# ── the HTTP reranker ───────────────────────────────────────────────────────


class _Transport:
    """Fakes POST /rag/rerank; records the bodies it was sent."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.status = status
        self.body = body
        self.bodies: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        sent = json.loads(request.content)
        self.bodies.append(sent)
        body = self.body
        if body is None:
            logits = [float(len(p)) for p in sent["passages"]]
            body = {"model": "/models/ms", "revision": "r1", "logits": logits, "ms": 2.5}
        return httpx.Response(self.status, json=body)


@pytest.fixture
def transport(monkeypatch):
    fake = _Transport()
    real_client = httpx.Client

    def client(**kwargs):
        return real_client(transport=httpx.MockTransport(fake.handler), **kwargs)

    monkeypatch.setattr(rr.httpx, "Client", client)
    return fake


def test_http_reranker_posts_query_and_passages(transport):
    logits, info = http_reranker("http://rag:8081/", 5.0)("q", ["a", "bbb"])
    assert logits == [1.0, 3.0]
    assert info == {"model": "/models/ms", "revision": "r1", "ms": 2.5}
    assert transport.bodies == [{"query": "q", "passages": ["a", "bbb"]}]


def test_http_reranker_splits_more_than_32_passages(transport):
    logits, info = http_reranker("http://rag:8081", 5.0)("q", ["p"] * 40)
    assert [len(b["passages"]) for b in transport.bodies] == [32, 8]
    assert len(logits) == 40 and info["ms"] == 5.0


def test_http_reranker_passes_the_services_503_reason_through(transport):
    transport.status = 503
    transport.body = {
        "detail": "reranker unavailable: not configured (UMRS_RAG_RERANKER_MODEL is empty)"
    }
    logits, info = http_reranker("http://rag:8081", 5.0)("q", ["a"])
    assert logits is None
    assert info["reason"].startswith("reranker unavailable: not configured")


def test_http_reranker_reports_an_unreachable_service(monkeypatch):
    def refuse(_request):
        raise httpx.ConnectError("connection refused")

    real_client = httpx.Client
    monkeypatch.setattr(
        rr.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(refuse), **kw),
    )
    logits, info = http_reranker("http://rag:8081", 5.0)("q", ["a"])
    assert (logits, info) == (None, {"reason": "reranker unreachable (ConnectError)"})


def test_the_reranker_url_defaults_to_the_rag_service(monkeypatch):
    assert reranker_from_env() is None
    monkeypatch.setenv("UMRS_RAG_DIRECT_URL", "http://rag:8081")
    assert callable(reranker_from_env())
