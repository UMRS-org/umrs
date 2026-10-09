"""Answer a RAG question from the knowledge spine (ADR-027 D3, P2 leg).

``POST /umrs/rag/query`` proxied only to the per-experience RAG store, which
uploads stopped feeding once ingest became the one write verb (WP-15): an
uploaded document could be searched but never answered from (stand-alone
stack, 2026-10-09). This path:

1. retrieves through the K3 vector lane — the query embedded by the RAG
   service (R6.1), ranked in one embedding space (ADR-083);
2. rebuilds each passage's full chunk text from its artifact, checked against
   the chunk node's md5 — else the stored excerpt; a passage with neither is
   dropped, never passed on as a placeholder;
3. asks the chat model the endpoint registry resolves (gap D-63), the same
   resolution semantic extraction uses.

Everything here is synchronous (DB, S3, one chat call); the route runs it in a
worker thread.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

logger = logging.getLogger(__name__)

_ANSWER_SYSTEM = (
    "You answer a scientist's question using only the numbered context passages. "
    "Cite the passages you use as [1], [2]. If the passages do not contain the "
    "answer, say that the knowledge base does not answer it — never invent facts."
)
_ANSWER_MAX_TOKENS = 800
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_PASSAGE_SQL = (
    "SELECT n.node_id, n.content::text, a.source_ref, "
    "a.metadata->>'source_name', a.metadata->>'staged_from' "
    "FROM knowledge_nodes n "
    "LEFT JOIN knowledge_artifacts a ON a.artifact_id = n.content->>'artifact_id' "
    "WHERE n.node_id = ANY(%s)"
)

FetchBytes = Callable[[str], "bytes | None"]


@dataclass
class SpineAnswer:
    """An answer and the passages it was given, or the reason there is none."""

    answer: str = ""
    passages: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    reason: str = ""


def retrieve_passages(
    pool: Any,
    query: str,
    project_id: str,
    *,
    top_k: int,
    threshold: float,
    include_ambient: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """K3 similarity hits at or above ``threshold``, or ``([], reason)``."""
    from .knowledge_query import (
        _embeddings_backend_for,
        _encode_query,
        _query_encoder_url,
    )

    if not _query_encoder_url():
        return [], "no query encoder configured"
    backend, reason = _embeddings_backend_for(
        pool, include_ambient=include_ambient, encoder=_encode_query
    )
    if backend is None:
        return [], reason or "embeddings unavailable"
    hits = backend.search_embeddings(query, project_id=project_id, limit=top_k)
    kept = [hit for hit in hits if hit["score"] >= threshold]
    return kept, "" if kept else _no_passage_reason(backend, bool(hits), threshold)


def _no_passage_reason(backend: Any, had_hits: bool, threshold: float) -> str:
    if had_hits:
        return f"no passage scored at or above {threshold}"
    return (backend.lane_report or {}).get("reason") or "no vectors for this project"


def attach_texts(
    pool: Any, passages: list[dict[str, Any]], fetch_bytes: FetchBytes
) -> list[dict[str, Any]]:
    """Give each passage its ``text`` and ``source_name``; drop text-less ones.

    ``metadata.text_source`` says where the text came from: ``artifact`` (the
    full chunk, md5-verified) or ``excerpt`` (the first 500 characters the
    chunk node stores).
    """
    rows = _passage_rows(pool, [p["source_id"] for p in passages])
    documents: dict[str, str | None] = {}
    kept: list[dict[str, Any]] = []
    for passage in passages:
        content, source_ref, name = rows.get(passage["source_id"], ({}, "", ""))
        text = _artifact_slice(content, source_ref, documents, fetch_bytes)
        source = "artifact" if text else "excerpt"
        text = text or str(content.get("excerpt") or "")
        if not text:
            continue
        passage["text"] = text
        passage["source_name"] = name or passage["title"]
        passage["metadata"]["text_source"] = source
        kept.append(passage)
    return kept


def compose_answer(
    query: str, passages: list[dict[str, Any]], project_id: str
) -> tuple[str, str, str]:
    """``(answer, model, reason)`` from the registry-resolved chat model."""
    from umrs.jobs.knowledge_adapters import _extract_chat_endpoint, _extract_timeout
    from umrs.knowledge.agent_executor import _default_litellm_chat

    target = _extract_chat_endpoint(project_id)
    if target is None:
        return "", "", "no READY chat endpoint in the registry"
    base_url, model = target
    context = "\n\n".join(
        f"[{i}] {p['source_name']}\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    try:
        reply = _default_litellm_chat(
            [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
            model=model,
            base_url=base_url,
            api_key=None,
            max_tokens=_ANSWER_MAX_TOKENS,
            timeout=_extract_timeout(),
        )
    except Exception as exc:  # noqa: BLE001 — reported to the caller as the reason
        return "", model, f"chat model {model!r} failed: {type(exc).__name__}: {exc}"
    return _THINK_RE.sub("", reply).strip(), model, ""


def answer_from_spine(
    pool: Any,
    query: str,
    project_id: str,
    *,
    top_k: int,
    threshold: float,
    include_ambient: bool = False,
    fetch_bytes: FetchBytes | None = None,
) -> SpineAnswer:
    """Retrieve, hydrate, answer. ``reason`` is set whenever ``answer`` is empty."""
    passages, reason = retrieve_passages(
        pool,
        query,
        project_id,
        top_k=top_k,
        threshold=threshold,
        include_ambient=include_ambient,
    )
    if passages:
        passages = attach_texts(pool, passages, fetch_bytes or _artifact_fetcher())
        reason = "" if passages else "the matching passages have no stored text"
    if not passages:
        return SpineAnswer(reason=reason)
    answer, model, reason = compose_answer(query, passages, project_id)
    return SpineAnswer(answer=answer, passages=passages, model=model, reason=reason)


# ── helpers ──────────────────────────────────────────────────────────────────


def _passage_rows(pool: Any, node_ids: list[str]) -> dict[str, tuple[dict, str, str]]:
    """node_id → (node content, artifact source_ref, artifact display name)."""
    with pool.getconn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(_PASSAGE_SQL, (node_ids,))
                rows = cur.fetchall()
        finally:
            pool.putconn(conn)
    return {
        node_id: (_as_dict(content), ref or "", name or _basename(staged))
        for node_id, content, ref, name, staged in rows
    }


def _as_dict(content_text: str | None) -> dict[str, Any]:
    try:
        value = json.loads(content_text or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _basename(uri: str | None) -> str:
    return PurePosixPath(uri).name if uri else ""


def _chunk_span(content: dict[str, Any]) -> tuple[int, int, str] | None:
    start, end, md5 = content.get("chunk_start"), content.get("chunk_end"), content.get("chunk_md5")
    if isinstance(start, int) and isinstance(end, int) and md5:
        return start, end, str(md5)
    return None


def _artifact_slice(
    content: dict[str, Any],
    source_ref: str,
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes,
) -> str | None:
    """The chunk's full text from its artifact, only when its md5 matches."""
    span = _chunk_span(content)
    if span is None or not source_ref:
        return None
    if source_ref not in documents:
        raw = fetch_bytes(source_ref)
        # The decoding the embed job chunked (knowledge_adapters._artifact_text).
        documents[source_ref] = raw.decode("utf-8", errors="ignore").strip() if raw else None
    document = documents[source_ref]
    if document is None:
        return None
    start, end, md5 = span
    piece = document[start:end]
    digest = hashlib.md5(piece.encode("utf-8"), usedforsecurity=False).hexdigest()
    return piece if digest == md5 else None


def _artifact_fetcher() -> FetchBytes:
    """Fetch artifact bytes with the server's storage client, built on first use."""
    state: dict[str, Any] = {}

    def fetch(source_ref: str) -> bytes | None:
        from umrs.jobs.knowledge_adapters import _build_s3_client, _fetch_artifact_bytes

        if "client" not in state:
            state["client"], state["bucket"] = _build_s3_client({})
        return _fetch_artifact_bytes(state["client"], state["bucket"], source_ref)

    return fetch
