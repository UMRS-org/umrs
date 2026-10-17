"""A2L artifact resolve endpoints.

Exposed through Gateway at ``POST /gw/a2l/resolve`` and
``GET /gw/a2l/artifacts``.  Routes sit inside UMRS so the
NGINX auth-request middleware covers them too.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from umrs.api._scope_identity import request_caller_scope
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.knowledge.access_scope import CallerScope, project_readable, project_writable

router = APIRouter(prefix="/umrs/a2l", tags=["a2l"])

# ── Request / Response DTOs ───────────────────────────────────────────────


class ResolveRequest(BaseModel):
    """POST body for resolve."""

    content_hash: str
    project_id: str | None = None  # default: the caller's verified project


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


# ── Scope (hardening plan H-1 item 5 / A.2, 2026-10-18) ──────────────────
# These routes took no identity: ``/artifacts`` listed any named project,
# ``/resolve`` mapped a content hash across every project and ``/derive``
# linked any knowledge artifact to any A2L artifact. They now resolve the
# gateway-verified caller (401 without it in strict mode), read only projects
# the caller may read, and link only a knowledge artifact the caller may
# write; another project's object reads as not found. Dev (mvp) mode keeps
# the anonymous caller permissive for a NAMED project, as everywhere in
# access_scope; a caller with no verified project must now name one (the
# old unscoped hash lookup across every project is the defect, so it is not
# kept as an mvp fallback).


def _scoped_project(request: Request, requested: str | None) -> str:
    """The named (or the caller's own) project when the caller may read it."""
    caller = request_caller_scope(request)
    project = (requested or "").strip() or caller.project_id
    if not project:
        raise_typed(
            PlatformError.a2l(
                "bad_request", "project_id is required (no verified project)"
            )
        )
    if not project_readable(project, caller):
        raise_typed(
            PlatformError.a2l("forbidden", f"project {project!r} is not readable")
        )
    return project


def _knowledge_store() -> Any:
    from umrs.knowledge.store import get_store

    return get_store()


def _text(value: Any) -> str:
    """Timestamps and UUIDs from asyncpg as the response's strings."""
    if value is None:
        return ""
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("/resolve", response_model=ResolveResponse)
async def resolve_a2l_by_hash(req: ResolveRequest, request: Request) -> dict[str, str]:
    """Resolve an A2L ID from a content hash within one readable project.

    Looks up ``a2l_artifact_refs`` by ``a2l_hash`` among the project's
    knowledge artifacts; an unknown (or another project's) hash answers
    empty ids.
    """
    project = _scoped_project(request, req.project_id)
    empty = {"a2l_id": "", "artifact_id": "", "a2l_hash": req.content_hash}
    try:
        row = await _knowledge_store()._pool.fetchrow(  # type: ignore[union-attr]
            "SELECT a.a2l_id, a.artifact_id::text AS artifact_id, a.synced_at "
            "FROM a2l_artifact_refs a "
            "JOIN knowledge_artifacts k ON k.id = a.artifact_id "
            "WHERE a.a2l_hash = $1 AND k.project_id = $2 LIMIT 1",
            req.content_hash,
            project,
        )
    except Exception:
        # Best-effort: return empty if DB is unreachable
        return empty
    if row is None:
        return empty
    return {
        "a2l_id": row["a2l_id"],
        "artifact_id": _text(row["artifact_id"]),
        "a2l_hash": req.content_hash,
        "synced_at": _text(row["synced_at"]),
    }


@router.get("/artifacts", response_model=ArtifactListResponse)
async def list_a2l_artifacts(
    request: Request,
    project_id: str | None = Query(
        None, description="Project ID to filter (default: the caller's)"
    ),
    artifact_ids: str = Query(
        "",
        description="Comma-separated list of artifact IDs to resolve",
    ),
) -> dict[str, Any]:
    """List A2L mappings for artifacts in a project the caller may read."""
    project = _scoped_project(request, project_id)
    try:
        query, params = _build_artifact_query(project, artifact_ids)
        rows = await _knowledge_store()._pool.fetch(query, *params)  # type: ignore[union-attr]
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


async def _linkable_knowledge_id(
    conn: Any, req: DeriveRequest, caller: CallerScope
) -> str | None:
    """The knowledge artifact's row id when the caller may write it and may
    read the A2L artifact (when that is a conserved ledger row); else None."""
    row = await conn.fetchrow(
        "SELECT id::text AS id, project_id FROM knowledge_artifacts "
        "WHERE id::text = $1 OR artifact_id = $1 LIMIT 1",
        req.knowledge_artifact_id,
    )
    if row is None or not project_writable(row["project_id"] or "", caller):
        return None
    ledger = await conn.fetchrow(
        "SELECT project_id FROM a2l_artifacts WHERE id = $1 AND deleted_at IS NULL",
        req.a2l_artifact_id,
    )
    if ledger is not None and not project_readable(ledger["project_id"] or "", caller):
        return None
    return str(row["id"])


async def _insert_link(
    conn: Any, knowledge_id: str, req: DeriveRequest
) -> dict[str, str]:
    row = await conn.fetchrow(
        """
        INSERT INTO knowledge_a2l_links
            (knowledge_artifact_id, a2l_artifact_id, job_id, attempt_id, edge_kind)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (knowledge_artifact_id, a2l_artifact_id)
        DO UPDATE SET link_status = 'active'
        RETURNING id
        """,
        knowledge_id,
        req.a2l_artifact_id,
        req.job_id,
        req.attempt_id,
        req.edge_kind,
    )
    if row:
        return {"status": "created", "link_id": str(row["id"])}
    return {"status": "updated", "link_id": "existing"}


@router.post("/derive", response_model=DeriveResponse)
async def derive_a2l_link(req: DeriveRequest, request: Request) -> dict[str, str]:
    """Register an A2L link for a knowledge artifact the caller may write.

    Called by knowledge jobs (INGEST, UNDERSTAND, VERIFY) to derive
    A2L references. Best-effort: silently returns success on conflict.
    Another project's knowledge (or ledger) artifact answers 404.
    """
    caller = request_caller_scope(request)
    try:
        async with _knowledge_store()._pool.acquire() as conn:  # type: ignore[union-attr]
            knowledge_id = await _linkable_knowledge_id(conn, req, caller)
            result = (
                await _insert_link(conn, knowledge_id, req) if knowledge_id else None
            )
    except Exception:
        return {"status": "error", "link_id": "", "error": "database unavailable"}
    if result is None:
        raise_typed(PlatformError.a2l("not_found", "knowledge artifact not found"))
    return result
