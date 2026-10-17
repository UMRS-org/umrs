"""Lineage query surface for knowledge artifacts (TASK-052).

Joins knowledge_a2l_links with A2L artifact refs to expose the full
provenance chain from knowledge artifacts to A2L identity space.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])


class LineageNode(BaseModel):
    """A node in the lineage chain."""

    artifact_id: str
    artifact_type: str  # "knowledge" | "a2l"
    title: str = ""
    edge_kind: str = ""
    created_at: str = ""


class LineageResponse(BaseModel):
    """Lineage query result."""

    artifact_id: str
    ancestors: list[LineageNode] = []
    descendants: list[LineageNode] = []


class LineageRequest(BaseModel):
    """CLI request body for lineage query."""

    artifact_id: str
    depth: int = 3  # max chain depth


# ── Endpoints ─────────────────────────────────────────────────────────────
#
# Scope (hardening plan H-1 item 5, 2026-10-18): the route took no identity
# and ``?project_id`` never reached the SQL, so any caller read any project's
# provenance chain. The project is now the caller's readable one (named or
# their own; 403 otherwise, 401 without identity in strict mode) and every
# query binds it: an artifact of another project has no chain here, exactly
# like an unknown one (no disclosure). Descendants stay inside that project.


@router.get("/lineage/{artifact_id}", response_model=LineageResponse)
async def get_artifact_lineage(
    artifact_id: str,
    request: Request,
    project_id: str | None = Query(
        None, description="Project to scope the query (default: the caller's)"
    ),
    depth: int = Query(3, ge=1, le=10, description="Max chain depth"),
) -> dict[str, Any]:
    """Query lineage for a knowledge artifact of a project the caller reads.

    Returns ancestors (A2L → knowledge) and descendants (knowledge → downstream).
    Joins knowledge_a2l_links with a2l_artifact_refs for full provenance.
    """
    from umrs.api.knowledge import _readable_project

    project = _readable_project(request, project_id, "Knowledge lineage")
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        return await _query_lineage(store, artifact_id, project, depth)
    except Exception:
        return _empty(artifact_id)


def _empty(artifact_id: str) -> dict[str, Any]:
    return {"artifact_id": artifact_id, "ancestors": [], "descendants": []}


def _text(value: Any) -> str:
    """Timestamps and UUIDs from asyncpg as the response's strings."""
    if value is None:
        return ""
    iso = getattr(value, "isoformat", None)
    return iso() if callable(iso) else str(value)


async def _query_lineage(
    store: Any,
    artifact_id: str,
    project_id: str,
    depth: int,
) -> dict[str, Any]:
    """Execute the lineage query inside ``project_id``."""
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        root = await conn.fetchrow(
            """
            SELECT id::text AS id, project_id
            FROM knowledge_artifacts
            WHERE (id::text = $1 OR artifact_id = $1) AND project_id = $2
            LIMIT 1
            """,
            artifact_id,
            project_id,
        )
        if root is None:
            return _empty(artifact_id)
        ancestors = await _ancestors(conn, str(root["id"]), project_id, depth)
        a2l_ids = [a["artifact_id"] for a in ancestors if a["artifact_type"] == "a2l"]
        descendants = await _descendants(
            conn, str(root["id"]), a2l_ids, project_id, depth
        )
    return {
        "artifact_id": artifact_id,
        "ancestors": ancestors,
        "descendants": descendants,
    }


async def _ancestors(
    conn: Any, root_id: str, project_id: str, depth: int
) -> list[dict[str, Any]]:
    """A2L refs linked to the root knowledge artifact."""
    rows = await conn.fetch(
        """
        SELECT l.a2l_artifact_id, l.edge_kind, l.derived_at, ref.a2l_id
        FROM knowledge_a2l_links l
        JOIN knowledge_artifacts ka ON ka.id = l.knowledge_artifact_id
        LEFT JOIN a2l_artifact_refs ref
            ON ref.artifact_id = l.knowledge_artifact_id
           AND ref.a2l_id = l.a2l_artifact_id
        WHERE l.knowledge_artifact_id::text = $1 AND ka.project_id = $2
        LIMIT $3
        """,
        root_id,
        project_id,
        depth,
    )
    return [
        {
            "artifact_id": row["a2l_artifact_id"] or root_id,
            "artifact_type": "a2l" if row["a2l_artifact_id"] else "knowledge",
            "title": row["a2l_id"] or row["a2l_artifact_id"] or root_id[:24],
            "edge_kind": row["edge_kind"] or "",
            "created_at": _text(row["derived_at"]),
        }
        for row in rows
    ]


async def _descendants(
    conn: Any, root_id: str, a2l_ids: list[str], project_id: str, depth: int
) -> list[dict[str, Any]]:
    """Other knowledge artifacts of the same project linked to the same A2L refs."""
    if not a2l_ids:
        return []
    rows = await conn.fetch(
        """
        SELECT l.knowledge_artifact_id::text AS knowledge_artifact_id,
               l.edge_kind,
               l.derived_at
        FROM knowledge_a2l_links l
        JOIN knowledge_artifacts ka ON ka.id = l.knowledge_artifact_id
        WHERE l.a2l_artifact_id = ANY($1::text[])
          AND l.knowledge_artifact_id::text != $2
          AND ka.project_id = $3
        LIMIT $4
        """,
        a2l_ids,
        root_id,
        project_id,
        depth,
    )
    return [
        {
            "artifact_id": row["knowledge_artifact_id"],
            "artifact_type": "knowledge",
            "title": row["knowledge_artifact_id"][:24],
            "edge_kind": row["edge_kind"] or "",
            "created_at": _text(row["derived_at"]),
        }
        for row in rows
    ]
