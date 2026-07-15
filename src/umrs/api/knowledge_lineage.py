"""Lineage query surface for knowledge artifacts (TASK-052).

Joins knowledge_a2l_links with A2L artifact refs to expose the full
provenance chain from knowledge artifacts to A2L identity space.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
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


@router.get("/lineage/{artifact_id}", response_model=LineageResponse)
async def get_artifact_lineage(
    artifact_id: str,
    project_id: str = Query(..., description="Project ID to scope query"),
    depth: int = Query(3, ge=1, le=10, description="Max chain depth"),
) -> dict[str, Any]:
    """Query lineage for a knowledge artifact.

    Returns ancestors (A2L → knowledge) and descendants (knowledge → downstream).
    Joins knowledge_a2l_links with a2l_artifact_refs for full provenance.
    """
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        result = await _query_lineage(store, artifact_id, project_id, depth)
        return result
    except Exception:
        return {
            "artifact_id": artifact_id,
            "ancestors": [],
            "descendants": [],
        }


async def _query_lineage(
    store: Any,
    artifact_id: str,
    project_id: str,
    depth: int,
) -> dict[str, Any]:
    """Execute the lineage query."""
    async with store._pool.acquire() as conn:  # type: ignore[union-attr]
        # Get ancestors (A2L refs linked to this knowledge artifact)
        rows = await conn.fetch(
            """
            SELECT k.id AS artifact_id,
                   a.a2l_artifact_id,
                   a.edge_kind,
                   a.derived_at,
                   ref.a2l_id,
                   ref.a2l_hash
            FROM knowledge_a2l_links k
            LEFT JOIN a2l_artifact_refs ref
                ON k.a2l_artifact_id = ref.a2l_artifact_id
            WHERE k.knowledge_artifact_id = $1
            LIMIT $2
            """,
            artifact_id,
            depth,
        )

        ancestors: list[dict[str, Any]] = []
        for row in rows:
            ancestors.append({
                "artifact_id": row["a2l_artifact_id"] or artifact_id,
                "artifact_type": "a2l" if row["a2l_artifact_id"] else "knowledge",
                "title": row["a2l_id"] or artifact_id[:24],
                "edge_kind": row["edge_kind"],
                "created_at": row["derived_at"],
            })

        # Get descendants (other knowledge artifacts linked to same A2L refs)
        a2l_ids = [r["a2l_artifact_id"] for r in rows if r["a2l_artifact_id"]]
        descendants: list[dict[str, Any]] = []
        if a2l_ids:
            placeholders = ",".join(f"${i + 2}" for i in range(len(a2l_ids)))
            desc_rows = await conn.fetch(
                f"""
                SELECT k.knowledge_artifact_id AS artifact_id,
                       k.edge_kind,
                       k.derived_at
                FROM knowledge_a2l_links k
                WHERE k.a2l_artifact_id IN ({placeholders})
                  AND k.knowledge_artifact_id != $1
                LIMIT $3
                """,
                artifact_id,
                *a2l_ids,
                depth,
            )
            for row in desc_rows:
                descendants.append({
                    "artifact_id": row["knowledge_artifact_id"],
                    "artifact_type": "knowledge",
                    "title": row["knowledge_artifact_id"][:24],
                    "edge_kind": row["edge_kind"],
                    "created_at": row["derived_at"],
                })

        return {
            "artifact_id": artifact_id,
            "ancestors": ancestors,
            "descendants": descendants,
        }
