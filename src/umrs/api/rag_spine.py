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
import threading
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
    full chunk, md5-verified), ``node`` (the chunk text the node stores — legacy
    canonized corpora) or ``excerpt`` (the first 500 characters).
    """
    rows = _passage_rows(pool, [p["source_id"] for p in passages])
    documents: dict[str, str | None] = {}
    kept: list[dict[str, Any]] = []
    for passage in passages:
        content, source_ref, name = rows.get(passage["source_id"], ({}, "", ""))
        text, source = _passage_text(content, source_ref, documents, fetch_bytes)
        if not text:
            continue
        passage["text"] = text
        passage["source_name"] = name or passage["title"]
        passage["artifact_id"] = str(content.get("artifact_id") or "")
        passage["source_ref"] = source_ref
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
    conn = pool.getconn()
    try:
        with conn:
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


def _passage_text(
    content: dict[str, Any],
    source_ref: str,
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes,
) -> tuple[str, str]:
    """``(text, source)``: the verified artifact slice, else the node's text,
    else its excerpt; ``("", "")`` when the node has none."""
    text = _artifact_slice(content, source_ref, documents, fetch_bytes)
    if text:
        return text, "artifact"
    for key, source in (("text", "node"), ("excerpt", "excerpt")):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value, source
    return "", ""


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


# ── The research loop's evidence store ───────────────────────────────────────

_ARTIFACT_ROWS_SQL = (
    "SELECT k.artifact_id, k.kind, k.source_ref, k.metadata->>'source_name', "
    "(SELECT n.content->>'excerpt' FROM knowledge_nodes n "
    " WHERE n.node_type = 'file' AND n.node_id LIKE 'node-file-%%' || k.artifact_id "
    " LIMIT 1) "
    "FROM knowledge_artifacts k WHERE k.project_id = %s"
)


_KNOWLEDGE_POOL: Any = None
_KNOWLEDGE_POOL_LOCK = threading.Lock()


def _knowledge_pool() -> Any:
    """The loop store's Postgres pool, built on first use, one per process.

    The knowledge database (``_resolve_knowledge_dsn``), which a site may
    split from the jobs database. ``minconn=0``: a returned connection is
    closed, so none goes stale across a Postgres restart; at most four open.
    """
    global _KNOWLEDGE_POOL
    with _KNOWLEDGE_POOL_LOCK:
        if _KNOWLEDGE_POOL is None:
            from psycopg2.pool import ThreadedConnectionPool

            from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

            _KNOWLEDGE_POOL = ThreadedConnectionPool(0, 4, _resolve_knowledge_dsn())
        return _KNOWLEDGE_POOL


class SpineKnowledgeStore:
    """Evidence for the research loop from the knowledge spine (ADR-027 D3).

    The server-side loop engine was built with no knowledge store, so in real
    mode every iteration found no evidence and stopped at the gate ("refusing
    to publish an unevidenced finding") in every deployment (plan §3c.2,
    reproduced on the stand-alone stack 2026-10-09). The engine binds this to
    each loop's project with ``for_project``.
    """

    def __init__(
        self,
        project_id: str = "",
        *,
        pool: Callable[[], Any] = _knowledge_pool,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> None:
        self._project_id = project_id
        self._pool = pool
        self._top_k = top_k
        self._threshold = threshold

    def for_project(self, project_id: str) -> SpineKnowledgeStore:
        return SpineKnowledgeStore(
            project_id, pool=self._pool, top_k=self._top_k, threshold=self._threshold
        )

    def search_artifacts(self, query: str, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """Semantic passages for ``query`` (the K3 lane), as loop artifacts."""
        pool = self._pool()
        passages, _reason = retrieve_passages(
            pool, query, self._project_id, top_k=self._top_k, threshold=self._threshold
        )
        if passages:
            passages = attach_texts(pool, passages, _artifact_fetcher())
        return [
            {
                "artifact_id": p["artifact_id"] or p["source_id"],
                "content": p["text"],
                "filename": p["source_name"],
                "summary": p["text"][:200],
                "s3_key": p["source_ref"] or None,
            }
            for p in passages
        ]

    def list_artifacts(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        return self._artifact_rows("ORDER BY k.created_at DESC LIMIT 200", ())

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        rows = self._artifact_rows("AND k.artifact_id = %s", (artifact_id,))
        return rows[0] if rows else None

    def _artifact_rows(self, tail: str, params: tuple) -> list[dict[str, Any]]:
        pool = self._pool()
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"{_ARTIFACT_ROWS_SQL} {tail}", (self._project_id, *params))
                rows = cur.fetchall()
            conn.rollback()  # read-only; return the connection idle
        finally:
            pool.putconn(conn)
        return [
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "content": excerpt or "",
                "filename": name or "",
                "summary": (excerpt or "")[:200],
                "s3_key": source_ref or None,
            }
            for artifact_id, kind, source_ref, name, excerpt in rows
        ]

