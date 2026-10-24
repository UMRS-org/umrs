"""Artifact contributions: who added an artifact to which graph (K1.2).

Migration 0129. Artifacts are deduplicated per project, so the artifact row
cannot say who added it or to which graph; one contribution row per
(artifact, graph, contributor) can. Every artifact insert site records one,
including when the insert returned an existing (deduplicated) artifact.

The rows are also how artifact paths apply the graph-readable predicate the
retrieve verb uses (``api.rag_retrieve.graph_hidden``): an artifact is
readable through a contribution in a graph the caller may read, so a member
never lists another member's personal-graph artifact. An artifact with no
contribution row is listed nowhere (fail closed); the backfill
(:mod:`umrs.knowledge.contribution_backfill`) gives legacy artifacts theirs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Rows written for an artifact nobody can be named for. An attributed row
#: for the same (artifact, graph) replaces it.
PLACEHOLDER_ORIGIN = "unknown"


@dataclass(frozen=True)
class Contribution:
    """One (artifact, graph, contributor) row."""

    artifact_id: str
    graph_id: str
    project_id: str
    user_id: str = ""
    origin_actor: str = PLACEHOLDER_ORIGIN
    title: str = ""

    def params(self) -> tuple[str, ...]:
        return (
            self.artifact_id,
            self.graph_id,
            self.project_id,
            self.user_id,
            self.origin_actor or PLACEHOLDER_ORIGIN,
            self.title[:500],
        )


@dataclass(frozen=True)
class ArtifactFilters:
    """``/k2/artifacts`` filters; ``None`` = no filter (status defaults active)."""

    kind: str | None = None
    status: str | None = "active"
    created_by: str | None = None
    since: datetime | None = None
    graph_id: str | None = None


def origin_for(spec_origin: Any, user_id: str) -> str:
    """The origin a write records: the request's actor when it named one,
    else ``user`` for an attributed write and ``unknown`` otherwise."""
    named = str(spec_origin or "").strip().lower()
    if named:
        return named
    return "user" if user_id else PLACEHOLDER_ORIGIN


def title_for(metadata: Any, fallback: str = "") -> str:
    """A human title from artifact metadata (filename first), else *fallback*."""
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("filename", "source_name", "title", "name"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return fallback.rsplit("/", 1)[-1]


_INSERT_SQL = """INSERT INTO knowledge_artifact_contributions
       (artifact_id, graph_id, project_id, user_id, origin_actor, title)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (artifact_id, graph_id, user_id) DO NOTHING"""

_DROP_PLACEHOLDER_SQL = """DELETE FROM knowledge_artifact_contributions
 WHERE artifact_id = $1 AND graph_id = $2 AND user_id = ''
   AND origin_actor = 'unknown'"""


async def record_contribution(conn: Any, contribution: Contribution) -> None:
    """Write one contribution on an asyncpg connection (idempotent). An
    attributed row drops the untraced placeholder for the same graph."""
    await conn.execute(_INSERT_SQL, *contribution.params())
    if contribution.user_id:
        await conn.execute(
            _DROP_PLACEHOLDER_SQL, contribution.artifact_id, contribution.graph_id
        )


async def record_with_pool(pool: Any, contribution: Contribution) -> bool:
    """Best-effort write through an asyncpg pool; False (logged) on failure.

    A failed write never fails the artifact's own write: the artifact is
    then listed nowhere until the backfill or a later contribution names it
    (fail closed), and the warning says which."""
    try:
        async with pool.acquire() as conn:
            await record_contribution(conn, contribution)
        return True
    except Exception as exc:  # noqa: BLE001 — attribution never fails the write
        logger.warning(
            "contribution not recorded: artifact=%s graph=%s user=%s: %s",
            contribution.artifact_id,
            contribution.graph_id,
            contribution.user_id,
            exc,
        )
        return False


_SYNC_INSERT_SQL = """INSERT INTO knowledge_artifact_contributions
       (artifact_id, graph_id, project_id, user_id, origin_actor, title)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (artifact_id, graph_id, user_id) DO NOTHING"""

_SYNC_DROP_PLACEHOLDER_SQL = """DELETE FROM knowledge_artifact_contributions
 WHERE artifact_id = %s AND graph_id = %s AND user_id = ''
   AND origin_actor = 'unknown'"""


def record_contribution_sync(cur: Any, contribution: Contribution) -> None:
    """The same write on a psycopg cursor (the caller's transaction)."""
    cur.execute(_SYNC_INSERT_SQL, contribution.params())
    if contribution.user_id:
        cur.execute(
            _SYNC_DROP_PLACEHOLDER_SQL, (contribution.artifact_id, contribution.graph_id)
        )


# ── reads through the graph-readable predicate ──────────────────────────────

_READABLE_CTE = """WITH readable AS (
    SELECT c.artifact_id, c.graph_id, c.user_id, c.origin_actor, c.title,
           c.created_at
      FROM knowledge_artifact_contributions c
     WHERE c.project_id = $1 AND NOT (c.graph_id = ANY($2::text[]))
), matched AS (
    SELECT r.artifact_id, MAX(r.created_at) AS last_added
      FROM readable r
     WHERE ($3::text IS NULL OR r.user_id = $3)
       AND ($4::text IS NULL OR r.graph_id = $4)
       AND ($5::timestamptz IS NULL OR r.created_at >= $5)
     GROUP BY r.artifact_id
)
"""

_MATCHED_ARTIFACTS = """FROM matched m
  JOIN knowledge_artifacts a ON a.artifact_id = m.artifact_id
 WHERE a.project_id = $1
   AND ($6::text IS NULL OR a.kind = $6)
   AND ($7::text IS NULL OR a.status = $7)"""

_LIST_SQL = (
    _READABLE_CTE
    + """SELECT a.artifact_id, a.content_hash, a.kind, a.project_id, a.source_ref,
       a.size_bytes, a.mime_type, a.status, a.metadata, a.created_at,
       a.updated_at, m.last_added
"""
    + _MATCHED_ARTIFACTS
    + "\n ORDER BY m.last_added DESC, a.artifact_id\n LIMIT $8 OFFSET $9"
)

_COUNT_SQL = _READABLE_CTE + "SELECT COUNT(*) " + _MATCHED_ARTIFACTS

_CONTRIBUTIONS_SQL = """SELECT artifact_id, graph_id, user_id, origin_actor, title,
       created_at
  FROM knowledge_artifact_contributions
 WHERE project_id = $1 AND NOT (graph_id = ANY($2::text[]))
   AND artifact_id = ANY($3::text[])
 ORDER BY artifact_id, created_at"""

_READABLE_IDS_SQL = """SELECT DISTINCT artifact_id
  FROM knowledge_artifact_contributions
 WHERE project_id = $1 AND NOT (graph_id = ANY($2::text[]))
   AND ($3::text[] IS NULL OR artifact_id = ANY($3::text[]))"""


def _filter_params(filters: ArtifactFilters) -> tuple[Any, ...]:
    return (
        filters.created_by or None,
        filters.graph_id or None,
        filters.since,
        filters.kind or None,
        filters.status or None,
    )


async def list_readable_artifacts(
    pool: Any,
    project_id: str,
    hidden_graph_ids: tuple[str, ...],
    filters: ArtifactFilters,
    *,
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[dict[str, Any]], int]:
    """A page of the project's readable artifacts, newest contribution first,
    each with its readable contributions; plus the filtered total."""
    head = (project_id, list(hidden_graph_ids), *_filter_params(filters))
    async with pool.acquire() as conn:
        total = int(await conn.fetchval(_COUNT_SQL, *head) or 0)
        rows = await conn.fetch(_LIST_SQL, *head, page_size, (page - 1) * page_size)
        ids = [str(r["artifact_id"]) for r in rows]
        found = await conn.fetch(
            _CONTRIBUTIONS_SQL, project_id, list(hidden_graph_ids), ids
        )
    by_artifact: dict[str, list[dict[str, Any]]] = {aid: [] for aid in ids}
    for row in found:
        entry = dict(row)
        by_artifact[str(entry.pop("artifact_id"))].append(entry)
    return [{**dict(r), "contributions": by_artifact[str(r["artifact_id"])]} for r in rows], total


async def readable_artifact_ids(
    pool: Any,
    project_id: str,
    hidden_graph_ids: tuple[str, ...],
    artifact_ids: list[str] | None = None,
) -> set[str]:
    """The project's artifacts readable through a contribution in a graph not
    hidden from the caller (optionally only among *artifact_ids*)."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            _READABLE_IDS_SQL, project_id, list(hidden_graph_ids), artifact_ids
        )
    return {str(r["artifact_id"]) for r in rows}
