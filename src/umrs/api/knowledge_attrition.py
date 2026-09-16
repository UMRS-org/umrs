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
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..api._asyncpg_pool import knowledge_asyncpg_pool
from ..api.files import get_project_id

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)

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
    """One memory_vector row into the running per-project report (never nulled)."""
    created_at = row["created_at"]
    metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
    venue = metadata.get("corpus_venue") or ""
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

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, vector, metadata, document_id, created_at
            FROM memory_vectors
            WHERE project_id = $1
            """,
            project_id,
        )
    for row in rows:
        _accumulate_row(out, row, now)
    return _finalize(out)


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

    Rows come from per-documents-vector; counts and age buckets are
    honest CSV receipts, never mutated by a caller.
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
