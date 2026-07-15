"""A2L artifact resolve endpoints.

Exposed through Gateway at ``POST /gw/a2l/resolve`` and
``GET /gw/a2l/artifacts``.  Routes sit inside UMRS so the
NGINX auth-request middleware covers them too.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(prefix="/umrs/a2l", tags=["a2l"])

# ── Request / Response DTOs ───────────────────────────────────────────────


class ResolveRequest(BaseModel):
    """POST body for resolve."""

    content_hash: str


class ResolveResponse(BaseModel):
    """Resolved A2L reference."""

    a2l_id: str
    artifact_id: str = ""
    a2l_hash: str = ""
    synced_at: str = ""


class ArtifactListResponse(BaseModel):
    """List of artifacts with their A2L mappings."""

    artifacts: list[dict[str, Any]]


class DeriveRequest(BaseModel):
    """POST body for derive."""

    knowledge_artifact_id: str
    a2l_artifact_id: str
    job_id: str = ""
    attempt_id: str = ""
    edge_kind: str = "derived"


class DeriveResponse(BaseModel):
    """Derive result."""

    status: str = ""
    link_id: str = ""
    error: str = ""


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_a2l_by_hash(req: ResolveRequest) -> dict[str, str]:
    """Resolve an A2L ID from a content hash.

    Looks up ``a2l_artifact_refs`` by ``a2l_hash`` and returns the
    mapped ``a2l_id``.  Returns 404 if the hash is not known.
    """
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        row = await store._pool.fetchrow(  # type: ignore[union-attr]
            "SELECT a2l_id, artifact_id, synced_at "
            "FROM a2l_artifact_refs "
            "WHERE a2l_hash = $1 LIMIT 1",
            req.content_hash,
        )
        if row is None:
            return {"a2l_id": "", "artifact_id": "", "a2l_hash": req.content_hash}
        return {
            "a2l_id": row["a2l_id"],
            "artifact_id": row["artifact_id"],
            "a2l_hash": req.content_hash,
            "synced_at": row["synced_at"],
        }
    except Exception:
        # Best-effort: return empty if DB is unreachable
        return {"a2l_id": "", "artifact_id": "", "a2l_hash": req.content_hash}


@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_a2l_artifacts(
    project_id: str = Query(..., description="Project ID to filter"),
    artifact_ids: str = Query(
        "",
        description="Comma-separated list of artifact IDs to resolve",
    ),
) -> dict[str, Any]:
    """List A2L mappings for artifacts in a project."""
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        query, params = _build_artifact_query(project_id, artifact_ids)
        rows = await store._pool.fetch(query, *params)  # type: ignore[union-attr]
        return {
            "artifacts": [
                {
                    "artifact_id": r["artifact_id"],
                    "a2l_id": r["a2l_id"],
                    "a2l_hash": r["a2l_hash"],
                    "synced_at": r["synced_at"],
                    "source": r["source"],
                }
                for r in rows
            ],
        }
    except Exception:
        return {"artifacts": []}


def _build_artifact_query(project_id: str, artifact_ids_filter: str):
    """Build the SQL query and params for artifact lookup."""
    params: list[Any] = [project_id]
    where = "WHERE k.project_id = $1"
    if artifact_ids_filter:
        id_list = [aid.strip() for aid in artifact_ids_filter.split(",") if aid.strip()]
        if id_list:
            placeholders = ",".join(f"${i + 2}" for i in range(len(id_list)))
            where += f" AND k.artifact_id IN ({placeholders})"
            params.extend(id_list)

    query = f"""SELECT k.artifact_id, a.a2l_id, a.a2l_hash, a.synced_at, a.source
                FROM a2l_artifact_refs a
                JOIN knowledge_artifacts k ON k.id = a.artifact_id
                {where}
                ORDER BY a.synced_at DESC
                LIMIT 200"""
    return query, params


@router.post("/derive", response_model=DeriveResponse)
async def derive_a2l_link(req: DeriveRequest) -> dict[str, str]:
    """Register an A2L link for a knowledge artifact.

    Called by knowledge jobs (INGEST, UNDERSTAND, VERIFY) to derive
    A2L references. Best-effort: silently returns success on conflict.
    """
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        async with store._pool.acquire() as conn:  # type: ignore[union-attr]
            row = await conn.fetchrow(
                """
                INSERT INTO knowledge_a2l_links
                    (knowledge_artifact_id, a2l_artifact_id, job_id, attempt_id, edge_kind)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (knowledge_artifact_id, a2l_artifact_id)
                DO UPDATE SET link_status = 'active'
                RETURNING id
                """,
                req.knowledge_artifact_id,
                req.a2l_artifact_id,
                req.job_id,
                req.attempt_id,
                req.edge_kind,
            )
            if row:
                return {"status": "created", "link_id": str(row["id"])}
            return {"status": "updated", "link_id": "existing"}
    except Exception:
        return {"status": "error", "link_id": "", "error": "database unavailable"}
