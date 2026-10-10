"""The cross-encoder rerank lane of the retrieval core (ADR-093 D-7, C8).

The model runs in the rag service (``POST /rag/rerank``), which returns raw
logits; this module picks the pool, orders it and reports the lane. The
sigmoid and the cut belong to ``cutoff.py``.

Pool: the top ``limit`` (30) candidates in fused order, at most
``per_artifact`` (3) per artifact. Order (``UMRS_RAG_RERANK_MODE``):

- ``replace`` (default): by logit;
- ``interpolate``: by ``1/(10 + r_ce) + 0.5/(10 + r_fused)``.

The tune split picks the default mode. Candidates outside the pool follow
the pool in their fused order, with ``rerank_logit`` left ``None``.

When the model is not staged, the call fails, or the query is mostly
non-ASCII (MiniLM is English-only), the lane says so (``unavailable`` or
``off``, with the reason) and returns no candidates: the input order is never
handed back as "reranked".

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any

import httpx

from umrs.knowledge.retrieval.types import Candidate, LaneResult, Reranker

LANE = "rerank"
RERANK_MODES: tuple[str, ...] = ("replace", "interpolate")
MODE_ENV = "UMRS_RAG_RERANK_MODE"
DEFAULT_ENV = "UMRS_RAG_RERANK_DEFAULT"
URL_ENV = "UMRS_RAG_RERANKER_URL"
TIMEOUT_ENV = "UMRS_RAG_RERANKER_TIMEOUT"
#: The rag service caps one request at 32 passages.
REQUEST_PASSAGES_MAX = 32
#: 384 tokens of English is under 2,000 characters; sending more is waste.
PASSAGE_CHARS = 2000
#: Interpolation constants (C8): rank offsets and the fused rank's weight.
INTERPOLATE_K = 10
INTERPOLATE_FUSED_WEIGHT = 0.5
#: Above this share of non-ASCII letters, the English-only model stays off.
NON_ASCII_MAX = 0.3


def rerank_enabled(mode: str) -> bool:
    """``on``/``off`` as asked; ``auto`` follows ``UMRS_RAG_RERANK_DEFAULT``
    (``off`` until the S4 gate passes). Staging is the reranker's to report:
    an unstaged model makes the lane ``unavailable``, not silently off."""
    if mode in ("on", "off"):
        return mode == "on"
    return os.environ.get(DEFAULT_ENV, "off").strip().lower() == "on"


def rerank_mode(asked: str | None = None) -> str:
    """``asked`` when it names a mode, else ``UMRS_RAG_RERANK_MODE``, else
    ``replace``."""
    if asked in RERANK_MODES:
        return asked
    mode = os.environ.get(MODE_ENV, "replace").strip().lower()
    return mode if mode in RERANK_MODES else "replace"


def mostly_non_ascii(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    foreign = sum(1 for ch in letters if not ch.isascii())
    return foreign / len(letters) > NON_ASCII_MAX


def passage_text(candidate: Candidate) -> str:
    """The hydrated text, else the node's own text or excerpt."""
    if candidate.text.strip():
        return candidate.text[:PASSAGE_CHARS]
    for key in ("text", "excerpt"):
        value = candidate.content.get(key)
        if isinstance(value, str) and value.strip():
            return value[:PASSAGE_CHARS]
    return ""


def select_pool(
    candidates: Sequence[Candidate], *, limit: int, per_artifact: int
) -> tuple[list[Candidate], list[Candidate]]:
    """``(pool, rest)``: the first ``limit`` in input order with at most
    ``per_artifact`` per artifact; everything else keeps its order in ``rest``."""
    pool: list[Candidate] = []
    rest: list[Candidate] = []
    taken: dict[str, int] = {}
    for candidate in candidates:
        key = candidate.artifact_id or candidate.node_id
        if len(pool) < limit and taken.get(key, 0) < per_artifact:
            taken[key] = taken.get(key, 0) + 1
            pool.append(candidate)
        else:
            rest.append(candidate)
    return pool, rest


def order_pool(
    pool: list[Candidate], logits: Sequence[float], mode: str
) -> list[Candidate]:
    """Set the logits and ranks, and return the pool in rerank order."""
    by_logit = sorted(range(len(pool)), key=lambda i: (-logits[i], i))
    ce_rank = {index: rank for rank, index in enumerate(by_logit, start=1)}
    for index, candidate in enumerate(pool):
        candidate.rerank_logit = float(logits[index])
        candidate.lane_scores[LANE] = float(logits[index])
        candidate.lane_ranks[LANE] = ce_rank[index]
    if mode != "interpolate":
        return [pool[i] for i in by_logit]

    def blended(index: int) -> tuple[float, int]:
        score = 1.0 / (INTERPOLATE_K + ce_rank[index])
        score += INTERPOLATE_FUSED_WEIGHT / (INTERPOLATE_K + index + 1)
        return -score, ce_rank[index]

    return [pool[i] for i in sorted(range(len(pool)), key=blended)]


def _call(
    reranker: Reranker, query: str, passages: list[str]
) -> tuple[list[float] | None, dict[str, Any]]:
    """The reranker's answer, with every failure turned into a reason."""
    try:
        logits, info = reranker(query, passages)
    except Exception as exc:  # noqa: BLE001 - a lane reports, it never raises
        return None, {"reason": f"reranker failed: {type(exc).__name__}: {exc}"}
    info = dict(info or {})
    if logits is not None and len(logits) != len(passages):
        info["reason"] = (
            f"reranker returned {len(logits)} logits for {len(passages)} passages"
        )
        return None, info
    return logits, info


def _skip(status: str, reason: str, started: float) -> LaneResult:
    return LaneResult(
        name=LANE,
        status=status,
        reason=reason,
        ms=(time.perf_counter() - started) * 1000.0,
    )


def rerank_lane(
    query: str,
    candidates: list[Candidate],
    reranker: Reranker | None,
    *,
    limit: int = 30,
    per_artifact: int = 3,
    mode: str | None = None,
) -> LaneResult:
    """Score the pool with the cross-encoder and return it in rerank order,
    followed by the unscored rest in fused order."""
    started = time.perf_counter()
    if reranker is None:
        return _skip("unavailable", "no reranker configured", started)
    if mostly_non_ascii(query):
        return _skip(
            "off", "query is mostly non-ASCII; the reranker is English-only", started
        )
    mode = rerank_mode(mode)
    pool, rest = select_pool(candidates, limit=limit, per_artifact=per_artifact)
    if not pool:
        return LaneResult(name=LANE, info={"mode": mode, "scored": 0})
    logits, info = _call(reranker, query, [passage_text(c) for c in pool])
    if logits is None:
        return _skip(
            "unavailable", str(info.get("reason") or "reranker unavailable"), started
        )
    ordered = order_pool(pool, logits, mode)
    return LaneResult(
        name=LANE,
        candidates=ordered + rest,
        ms=(time.perf_counter() - started) * 1000.0,
        info=_report(info, mode, len(pool)),
    )


def _report(info: dict[str, Any], mode: str, scored: int) -> dict[str, Any]:
    """The lane's trace entry: which model and revision scored how many."""
    report = {key: info[key] for key in ("model", "revision") if key in info}
    report.update({"mode": mode, "scored": scored, "model_ms": info.get("ms")})
    return report


# ── the rag service's reranker over HTTP ─────────────────────────────────────


def _post_batch(
    client: httpx.Client, url: str, query: str, passages: list[str]
) -> tuple[list[float] | None, dict[str, Any]]:
    try:
        response = client.post(url, json={"query": query, "passages": passages})
    except httpx.HTTPError as exc:
        return None, {"reason": f"reranker unreachable ({type(exc).__name__})"}
    if response.status_code != 200:
        return None, {"reason": _error_reason(response)}
    try:
        body = response.json()
        logits = [float(value) for value in body["logits"]]
    except (ValueError, KeyError, TypeError) as exc:
        return None, {
            "reason": f"reranker answered malformed JSON ({type(exc).__name__})"
        }
    info = {key: body.get(key) for key in ("model", "revision", "ms")}
    return logits, info


def _error_reason(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, str) and detail:
        return detail
    return f"reranker answered HTTP {response.status_code}"


def http_reranker(base_url: str, timeout: float) -> Reranker:
    """A :data:`Reranker` over the rag service's ``POST /rag/rerank``.

    Never raises: an outage comes back as ``(None, {"reason": ...})``, which
    the lane reports as ``unavailable``. More than 32 passages go in batches.
    """
    url = f"{base_url.rstrip('/')}/rag/rerank"

    def rerank(
        query: str, passages: Sequence[str]
    ) -> tuple[list[float] | None, dict[str, Any]]:
        logits: list[float] = []
        info: dict[str, Any] = {}
        ms = 0.0
        with httpx.Client(timeout=timeout) as client:
            for start in range(0, len(passages), REQUEST_PASSAGES_MAX):
                batch = list(passages[start : start + REQUEST_PASSAGES_MAX])
                part, info = _post_batch(client, url, query, batch)
                if part is None:
                    return None, info
                logits.extend(part)
                ms += float(info.get("ms") or 0.0)
        return logits, {**info, "ms": round(ms, 1)}

    return rerank


def reranker_from_env() -> Reranker | None:
    """``UMRS_RAG_RERANKER_URL``, else ``UMRS_RAG_DIRECT_URL`` (the rag
    service); ``None`` when neither is set. Timeout
    ``UMRS_RAG_RERANKER_TIMEOUT`` (default 10 s; the first call after a boot
    may still be warming)."""
    base = (
        os.environ.get(URL_ENV) or os.environ.get("UMRS_RAG_DIRECT_URL") or ""
    ).strip()
    if not base:
        return None
    try:
        timeout = max(0.5, float(os.environ.get(TIMEOUT_ENV, "10")))
    except ValueError:
        timeout = 10.0
    return http_reranker(base, timeout)
