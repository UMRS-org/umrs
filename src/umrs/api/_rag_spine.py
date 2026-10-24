"""The rag facade's document listing and corpus statistics, from the spine
(knowledge plan K1.3, 2026-10-24).

Uploads have fed only the knowledge spine since WP-15 (``POST /rag/upload``
hands them to the central ingest), yet ``GET /rag/documents`` and ``GET
/rag/stats`` still asked the endpoint corpus, which answered ``[]`` or
counted its own ``rag_chunks``: after a successful upload ``r4 knowledge
docs list`` was empty and ``stats`` said 0. With the central facade (the
default, ``UMRS_RAG_UPLOAD_FACADE``) both now answer from the spine, through
the graph-readable predicate (``_artifact_scope``), so they list exactly
what ``GET /k2/artifacts`` lists and never another member's personal-graph
file. ``UMRS_RAG_UPLOAD_FACADE=endpoint`` keeps the old proxy.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Request

#: Most documents ``GET /rag/documents`` lists (newest contribution first).
SPINE_DOCUMENTS_MAX = 500


def upload_facade() -> str:
    """``central`` (the default) or ``endpoint`` (``UMRS_RAG_UPLOAD_FACADE``)."""
    value = os.environ.get("UMRS_RAG_UPLOAD_FACADE", "central").strip().lower()
    return "endpoint" if value == "endpoint" else "central"


def spine_answers() -> bool:
    """True when the listings answer from the spine (the central facade)."""
    return upload_facade() != "endpoint"


def _users(contributions: list[dict[str, Any]], key: str) -> list[str]:
    """The distinct non-empty *key* values of the contributions (sorted)."""
    return sorted({str(c.get(key) or "") for c in contributions} - {""})


def _filename(row: dict[str, Any], contributions: list[dict[str, Any]]) -> str:
    """The first contribution title, else the artifact's own name."""
    from umrs.knowledge.contributions import title_for

    for contribution in contributions:
        if contribution.get("title"):
            return str(contribution["title"])
    meta = row.get("metadata")
    return title_for(
        meta if isinstance(meta, dict) else {}, str(row.get("source_ref") or "")
    )


def _document(row: dict[str, Any], chunks: int) -> dict[str, Any]:
    """One readable artifact as a rag document row."""
    contributions = list(row.get("contributions") or [])
    return {
        "document_id": str(row["artifact_id"]),
        "filename": _filename(row, contributions),
        "project_id": str(row.get("project_id") or ""),
        "chunks": chunks,
        "metadata": {
            "kind": row.get("kind") or "",
            "size_bytes": row.get("size_bytes") or 0,
            "status": row.get("status") or "",
            "embedding_status": row.get("embedding_status") or "",
            "added_at": str(row.get("last_added") or row.get("created_at") or ""),
            "contributors": _users(contributions, "user_id"),
            "graphs": _users(contributions, "graph_id"),
            "source": "spine",
        },
    }


async def spine_documents(
    request: Request, project_id: str, viewer: str = ""
) -> list[dict[str, Any]]:
    """The project's active documents this caller may read, newest first.
    *viewer* is the verified caller (``_require_rag_scope``): it decides
    which /save metadata shows (``contributions.public_metadata``)."""
    from umrs.knowledge.contributions import (
        ArtifactFilters,
        artifact_chunk_counts,
        list_readable_artifacts,
    )

    from ._artifact_scope import readable_graphs_for
    from ._asyncpg_pool import knowledge_asyncpg_pool

    graphs = await readable_graphs_for(request, project_id)
    pool = await knowledge_asyncpg_pool(request)
    rows, _total = await list_readable_artifacts(
        pool,
        project_id,
        graphs,
        ArtifactFilters(),
        page=1,
        page_size=SPINE_DOCUMENTS_MAX,
        viewer=viewer,
    )
    counts = await artifact_chunk_counts(
        pool, project_id, [str(r["artifact_id"]) for r in rows], graphs
    )
    return [_document(row, counts.get(str(row["artifact_id"]), 0)) for row in rows]


async def spine_document(
    request: Request, project_id: str, document_id: str, viewer: str = ""
) -> dict[str, Any] | None:
    """One document :func:`spine_documents` would list, or None (not
    readable, or not in the project): ``GET /rag/documents/{id}`` answers
    the ids the listing gives (review of K1 part B, 2026-10-24)."""
    from umrs.knowledge.contributions import artifact_chunk_counts, readable_artifact

    from ._artifact_scope import readable_graphs_for
    from ._asyncpg_pool import knowledge_asyncpg_pool

    graphs = await readable_graphs_for(request, project_id)
    pool = await knowledge_asyncpg_pool(request)
    row = await readable_artifact(pool, project_id, graphs, document_id, viewer)
    if row is None:
        return None
    counts = await artifact_chunk_counts(pool, project_id, [document_id], graphs)
    return _document(row, counts.get(document_id, 0))


async def spine_stats(request: Request, project_id: str) -> dict[str, Any]:
    """Corpus statistics over the documents :func:`spine_documents` lists."""
    from umrs.knowledge.contributions import readable_corpus_stats

    from ._artifact_scope import readable_graphs_for
    from ._asyncpg_pool import knowledge_asyncpg_pool

    graphs = await readable_graphs_for(request, project_id)
    pool = await knowledge_asyncpg_pool(request)
    stats = await readable_corpus_stats(pool, project_id, graphs)
    documents = stats["documents"]
    return {
        **stats,
        "project_id": project_id,
        "projects": 1,
        "avg_chunks_per_doc": round(stats["chunks"] / documents, 1) if documents else 0,
        "pending": documents - stats["embedded"],
        "status": "ok",
        "source": "spine",
    }


__all__ = [
    "SPINE_DOCUMENTS_MAX",
    "spine_answers",
    "spine_document",
    "spine_documents",
    "spine_stats",
    "upload_facade",
]
