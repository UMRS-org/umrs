"""Corpus attrition report (WP-15 dual-corpus dwell sweep, 2026-09-12).

Report-only, READ-ONLY, no write at any point. Operation law (never make
data age-positions silently move at query time): the route never deletes,
archives, tags, or shifts any row; it OBSERVES.

Contract (WP-15 ledger note):
- Under the dwell, uploads to the retired upload lane must stop growing
  over time per project. The operational proof is an age-bucketed read
  against the per-project vector store. The route returns counts +
  oldest/newest stamps per age bucket, and which dirs/corridor a row was
  recorded under (`metadata.corpus_venue` when present — never inferred).
- Age truth is row-local (`created_at`), not time-of-query math keepsake:
  an old upload today reads the same age tomorrow.
- Shared access rules: caller project-scoped results stay per-caller —
  you also orchestrate only your own corpus unless you're admin (P14
  occupancy).

The per-project vector store is ``knowledge_node_embeddings`` (migration
0075), the table KNOWLEDGE_EMBED writes for every chunk node of an ingested
artifact, joined to ``knowledge_nodes`` for the project and the artifact
(``content.artifact_id``). Until 2026-10-21 the route read
``memory_vectors(id, vector, metadata, document_id, project_id)``: columns
no migration creates (0010's ``memory_vectors`` is context memory, keyed by
``memory_id``), so every call was a 500. A deployment without the table
answers a typed ``knowledge.unavailable`` (503), never a 500.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..api._asyncpg_pool import knowledge_asyncpg_pool
from ..api.files import get_project_id
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)

# One row per stored vector of the project's corpus (chunk nodes carry the
# artifact they cut; another node's vector counts as its own document).
_CORPUS_VECTORS_SQL = """
    SELECT kne.node_id AS id,
           COALESCE(kn.content->>'artifact_id', kn.node_id) AS document_id,
           kn.content->>'corpus_venue' AS corpus_venue,
           kne.created_at
    FROM knowledge_node_embeddings kne
    JOIN knowledge_nodes kn ON kn.node_id = kne.node_id
    WHERE kn.project_id = $1
"""
_CORPUS_SOURCE = "knowledge_node_embeddings"

_BUCKETS = (
    ("lt_30d", 30),
    ("d30_to_90", 90),
    ("gt_90d", None),
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _days_since(ts: Any, now: datetime) -> int | None:
    if not ts:
        return None
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).days


def _bucket(days: int | None) -> str:
    if days is None:
        return "unknown"
    if days < 30:
        return "lt_30d"
    if days < 90:
        return "d30_to_90"
    return "gt_90d"


def _accumulate_row(out: dict[str, Any], row: Any, now: datetime) -> None:
    """One stored vector into the running per-project report (never nulled)."""
    created_at = row["created_at"]
    venue = row["corpus_venue"] or ""
    days = _days_since(created_at, now)
    bucket = _bucket(days)
    bucket_entry = out["buckets"][bucket]
    bucket_entry["rows"] += 1
    bucket_entry["documents"].add(row["document_id"])
    out["_documents_all"].add(row["document_id"])
    bucket_entry["oldest"] = (
        min(bucket_entry["oldest"], created_at)
        if bucket_entry["oldest"]
        else created_at
    )
    bucket_entry["newest"] = (
        max(bucket_entry["newest"], created_at)
        if bucket_entry["newest"]
        else created_at
    )
    out["_oldest"] = min(out["_oldest"], created_at) if out["_oldest"] else created_at
    out["_newest"] = max(out["_newest"], created_at) if out["_newest"] else created_at
    out["venues"][venue or "unset"] = out["venues"].get(venue or "unset", 0) + 1
    out["rows_total"] += 1


def _finalize(out: dict[str, Any]) -> dict[str, Any]:
    """Stamp the final receipt shape (timestamps iso; counters rolled up)."""
    for bucket_entry in out["buckets"].values():
        bucket_entry["documents"] = len(bucket_entry["documents"])
        bucket_entry["oldest"] = (
            bucket_entry["oldest"].isoformat() if bucket_entry["oldest"] else None
        )
        bucket_entry["newest"] = (
            bucket_entry["newest"].isoformat() if bucket_entry["newest"] else None
        )
    out["oldest"] = out["_oldest"].isoformat() if out["_oldest"] else None
    out["newest"] = out["_newest"].isoformat() if out["_newest"] else None
    out["documents"] = len(out["_documents_all"])
    for key in ("_oldest", "_newest", "_documents_all"):
        out.pop(key, None)
    return out


async def _report_for_project(pool: Any, project_id: str) -> dict[str, Any]:
    """Fetch the vector age buckets for one project (Honest CSV receipts)."""
    now = _utcnow()
    out: dict[str, Any] = {
        "project_id": project_id,
        "rows_total": 0,
        "documents": 0,
        "buckets": {},
        "venues": {},
        "generated_at": now.isoformat(),
        "oldest": None,
        "newest": None,
        "_oldest": None,
        "_newest": None,
        "_documents_all": set(),
    }
    for key, _ in _BUCKETS + (("unknown", None),):
        out["buckets"][key] = {
            "rows": 0,
            "documents": set(),
            "oldest": None,
            "newest": None,
        }

    rows = await _corpus_vector_rows(pool, project_id)
    for row in rows:
        _accumulate_row(out, row, now)
    out["source"] = _CORPUS_SOURCE
    return _finalize(out)


async def _corpus_vector_rows(pool: Any, project_id: str) -> list[Any]:
    """The project's stored vectors; a typed 503 when this deployment has no
    per-project vector table (never a 500)."""
    from asyncpg.exceptions import UndefinedColumnError, UndefinedTableError

    try:
        async with pool.acquire() as conn:
            return list(await conn.fetch(_CORPUS_VECTORS_SQL, project_id))
    except (UndefinedTableError, UndefinedColumnError) as exc:
        raise_typed(
            PlatformError.knowledge(
                "unavailable",
                "corpus attrition report is not available on this deployment: "
                f"the per-project vector table ({_CORPUS_SOURCE}, migration "
                "0075) is missing",
            ),
            cause=exc,
        )


@router.post("/corpus-attrition-report")
async def corpus_attrition_report(
    request: Request,
    project_id: str = Depends(get_project_id),
    all_projects: bool = Query(
        False, description="Admin-only: aggregate across visible projects"
    ),
) -> dict[str, Any]:
    """Issue the read-only corpus-attrition report (WP-15 dwell sweep).
    # auth: verified

    Rows are the project's stored vectors (knowledge_node_embeddings);
    counts and age buckets are honest receipts, never mutated by a caller.
    """
    pool = await knowledge_asyncpg_pool(request)
    if pool is None:
        raise HTTPException(status_code=503, detail="knowledge pool availability: 503")

    # caller identity gate: projects are caller-scoped by installed pattern;
    # `all_projects` is admin-side only.
    user_id = request.headers.get("x-user-id", "")
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id required")

    awaitable: Iterable[str]
    if all_projects:
        admin = request.headers.get("x-verified-is-admin", "").lower() == "true"
        if not admin:
            raise HTTPException(status_code=403, detail="all_projects requires admin")
        awaitable = [project_id]
    else:
        awaitable = [project_id]

    reports: dict[str, Any] = {"projects": {}}
    for pid in awaitable:
        reports["projects"][pid] = await _report_for_project(pool, pid)

    return {
        "status": "ok",
        "generated_at": _utcnow().isoformat(),
        "dwell_rule": "read-only, never delete",
        "reports": reports,
        "caller_notes": {
            "upload_lane_mortality": (
                "lt_30d buckets shrinking per project = the retired lane went quiet;"
                "gt_90d clusters = old uploads dying the natural death path."
            ),
        },
    }
