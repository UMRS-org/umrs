"""Delete one knowledge artifact and everything derived from it (ADR-093 MD-16).

Decided by the maintainer 2026-10-11. The verb behind
``DELETE /umrs/knowledge/artifacts/{id}`` and ``DELETE /umrs/rag/documents/{id}``.

ONE transaction, scoped to the artifact's project, removes:

- the artifact row (``knowledge_artifacts``), locked first (``FOR UPDATE``);
- its nodes: every node of the project whose ``content.artifact_id`` is the
  artifact (understand's file/class/function nodes, the extractor's semantic
  nodes, the embed writer's chunks), every node derived from one of its IR
  rows (``source_ir``), the produced_by provenance node keyed by the artifact
  id, and the chunks rag.upload minted, which carry only their
  ``document_id`` and are reached through the ``part_of`` edge into the
  artifact's file node;
- the vectors of those nodes (``knowledge_node_embeddings``);
- every edge touching those nodes, in both directions;
- its IR rows (``knowledge_ir``) and the verification findings naming it.

``knowledge_events`` rows are kept and detached (``artifact_id`` set NULL,
nullable since migration 0122): the audit trail outlives the artifact.

The object-store bytes go AFTER the commit, best-effort, one outcome per
object: a failed object delete is named in the report and the database delete
stands. An object another artifact row still references (content-hash keys
are shared by rows holding the same bytes) is kept and said so.

An unknown id, or an id of another project, raises :class:`ArtifactNotFound`:
there is no cross-project delete, and deleting an already-deleted id is the
same 404 (idempotent by refusal, never a second "deleted").

The SQL is driver-neutral (``$n`` placeholders) and runs over a small
:class:`Tx` protocol: :class:`AsyncpgTx` for the UMRS server's knowledge
pool, :class:`Psycopg2Tx` for the stand-alone RAG service's sync pool. Every
statement carries an ``/* artifact_delete:<step> */`` tag.

No umrs.api or umrs.server imports (foundation boundary).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Where content-addressed bytes live (``ArtifactStorage.key``).
CONTENT_PREFIX = "projects/{project_id}/knowledge/artifacts/"
#: Where ``POST /umrs/knowledge/upload`` stages the uploaded bytes.
STAGED_PREFIX = "knowledge-uploads/{project_id}/"
#: The artifact id rag.upload mints for a rag document id.
RAG_ARTIFACT_PREFIX = "artifact-rag-"

LOCK_SQL = (
    "/* artifact_delete:lock_artifact */ "
    "SELECT artifact_id, content_hash, source_ref, metadata FROM knowledge_artifacts "
    "WHERE artifact_id = $1 AND project_id = $2 FOR UPDATE"
)
#: One arm per provenance, so each reads its own index (0122's
#: ``(project_id, content->>'artifact_id')`` for the first).
NODES_SQL = (
    "/* artifact_delete:select_nodes */ "
    "WITH own AS (SELECT node_id, node_type FROM knowledge_nodes "
    "WHERE project_id = $2 AND content->>'artifact_id' = $1 "
    "UNION SELECT node_id, node_type FROM knowledge_nodes WHERE project_id = $2 "
    "AND source_ir IN (SELECT ir_id FROM knowledge_ir WHERE artifact_id = $1) "
    "UNION SELECT node_id, node_type FROM knowledge_nodes WHERE project_id = $2 "
    "AND node_id = $1 AND label = 'artifact:' || $1) "
    "SELECT node_id FROM own "
    "UNION SELECT n.node_id FROM knowledge_nodes n "
    "JOIN knowledge_edges e ON e.source_id = n.node_id AND e.relation_type = 'part_of' "
    "JOIN own f ON f.node_id = e.target_id AND f.node_type = 'file' "
    "WHERE n.project_id = $2 AND n.content->>'kind' = 'chunk'"
)
#: (count key, statement, takes the project as ``$2``); ``$1`` is the node-id
#: array.
NODE_STEPS: tuple[tuple[str, str, bool], ...] = (
    (
        "embeddings",
        "/* artifact_delete:delete_embeddings */ "
        "DELETE FROM knowledge_node_embeddings WHERE node_id = ANY($1::text[])",
        False,
    ),
    (
        "edges",
        "/* artifact_delete:delete_edges */ DELETE FROM knowledge_edges "
        "WHERE source_id = ANY($1::text[]) OR target_id = ANY($1::text[])",
        False,
    ),
    (
        "nodes",
        "/* artifact_delete:delete_nodes */ DELETE FROM knowledge_nodes "
        "WHERE node_id = ANY($1::text[]) AND project_id = $2",
        True,
    ),
)
#: (count key, statement); ``$1`` is the artifact id.
ARTIFACT_STEPS: tuple[tuple[str, str], ...] = (
    (
        "findings",
        "/* artifact_delete:delete_findings */ "
        "DELETE FROM verification_findings WHERE artifact_id = $1",
    ),
    (
        "ir",
        "/* artifact_delete:delete_ir */ DELETE FROM knowledge_ir WHERE artifact_id = $1",
    ),
    (
        "events_detached",
        "/* artifact_delete:detach_events */ "
        "UPDATE knowledge_events SET artifact_id = NULL WHERE artifact_id = $1",
    ),
)
DELETE_ARTIFACT_SQL = (
    "/* artifact_delete:delete_artifact */ "
    "DELETE FROM knowledge_artifacts WHERE artifact_id = $1 AND project_id = $2"
)
#: Rows (any project, any status) still pointing at an object: by its
#: reference, as a staged source, or by holding the same content-addressed
#: bytes in the project (``$2`` is '' for a staged object).
OBJECT_REFS_SQL = (
    "/* artifact_delete:object_refs */ SELECT count(*) FROM knowledge_artifacts "
    "WHERE source_ref = $1 OR metadata->>'staged_from' = $1 "
    "OR ($2 <> '' AND project_id = $3 AND content_hash = $2)"
)
RESOLVE_SQL = (
    "/* artifact_delete:resolve_document */ "
    "SELECT artifact_id FROM knowledge_artifacts WHERE project_id = $2 "
    "AND (artifact_id = $1 OR artifact_id = 'artifact-rag-' || $1 "
    "OR metadata->>'document_id' = $1) "
    "ORDER BY (artifact_id = $1) DESC, (artifact_id = 'artifact-rag-' || $1) DESC, "
    "artifact_id LIMIT 2"
)


class ArtifactNotFound(LookupError):
    """No artifact of that id in the project (unknown, foreign or already gone)."""

    def __init__(self, artifact_id: str, project_id: str) -> None:
        super().__init__(f"knowledge artifact {artifact_id!r} not found in project {project_id!r}")
        self.artifact_id = artifact_id
        self.project_id = project_id


class Tx(Protocol):
    """The two calls the delete makes; ``$n`` placeholders."""

    async def fetch(self, sql: str, *args: Any) -> list[Sequence[Any]]: ...

    async def execute(self, sql: str, *args: Any) -> int: ...


class ObjectStore(Protocol):
    """Deletes one object; raises on failure."""

    #: The bucket content-addressed keys live in.
    content_bucket: str
    #: Buckets the platform stages uploads in (a staged object elsewhere is kept).
    buckets: frozenset[str]

    def delete(self, bucket: str, key: str) -> None: ...


@dataclass(frozen=True)
class ObjectTarget:
    """One object the artifact's bytes live in (``bucket`` None = content bucket)."""

    ref: str
    key: str
    bucket: str | None = None
    content_hash: str = ""


@dataclass
class ArtifactDeleteReport:
    artifact_id: str
    project_id: str
    deleted: bool = True
    counts: dict[str, int] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "project_id": self.project_id,
            "deleted": self.deleted,
            "counts": dict(self.counts),
            "objects": [dict(o) for o in self.objects],
            "notes": list(self.notes),
        }


# ── drivers ──────────────────────────────────────────────────────────────────


def _status_count(status: Any) -> int:
    """Row count from an asyncpg command tag (``DELETE 3``, ``UPDATE 0``)."""
    tail = str(status or "").rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else 0


class AsyncpgTx:
    """:class:`Tx` over an asyncpg connection (the caller holds the transaction)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, sql: str, *args: Any) -> list[Sequence[Any]]:
        return list(await self._conn.fetch(sql, *args))

    async def execute(self, sql: str, *args: Any) -> int:
        return _status_count(await self._conn.execute(sql, *args))


_PLACEHOLDER = re.compile(r"\$(\d+)")


def to_pyformat(sql: str, args: Sequence[Any]) -> tuple[str, tuple[Any, ...]]:
    """``$n`` SQL and its arguments as psycopg2 ``%s`` SQL and parameters."""
    order = [int(n) - 1 for n in _PLACEHOLDER.findall(sql)]
    return _PLACEHOLDER.sub("%s", sql.replace("%", "%%")), tuple(args[i] for i in order)


class Psycopg2Tx:
    """:class:`Tx` over a psycopg2 cursor (the caller holds the transaction)."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    async def fetch(self, sql: str, *args: Any) -> list[Sequence[Any]]:
        self._cur.execute(*to_pyformat(sql, args))
        return list(self._cur.fetchall())

    async def execute(self, sql: str, *args: Any) -> int:
        self._cur.execute(*to_pyformat(sql, args))
        return max(int(getattr(self._cur, "rowcount", 0) or 0), 0)


# ── the transaction ──────────────────────────────────────────────────────────


def _metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _s3_target(ref: str, project_id: str) -> ObjectTarget | None:
    """A staged upload ``s3://bucket/knowledge-uploads/<project>/...``."""
    bucket, _, key = ref[len("s3://") :].partition("/")
    if bucket and key.startswith(STAGED_PREFIX.format(project_id=project_id)):
        return ObjectTarget(ref=ref, key=key, bucket=bucket)
    return None


def object_targets(row: Sequence[Any], project_id: str) -> list[ObjectTarget]:
    """The platform-owned objects an artifact row points at, in the project.

    Its content-addressed key, and the staged upload it was copied from. Any
    other reference (``rag://`` ids, URLs, another project's keys) is no
    object of this artifact and is never deleted.
    """
    _aid, content_hash, source_ref, raw_meta = row[0], row[1], row[2], row[3]
    refs = [str(source_ref or ""), str(_metadata(raw_meta).get("staged_from") or "")]
    targets: list[ObjectTarget] = []
    for ref in dict.fromkeys(r for r in refs if r):
        if ref.startswith(CONTENT_PREFIX.format(project_id=project_id)):
            targets.append(ObjectTarget(ref=ref, key=ref, content_hash=str(content_hash or "")))
        elif ref.startswith("s3://"):
            target = _s3_target(ref, project_id)
            if target is not None:
                targets.append(target)
    return targets


def _notes(row: Sequence[Any]) -> list[str]:
    meta = _metadata(row[3])
    if meta.get("origin") != "rag.upload":
        return []
    return [
        f"rag document {meta.get('document_id') or '?'}: rows in the endpoint corpus "
        "(the mini-RiVault RAG index) are not part of the knowledge spine and are "
        "not deleted; the spine and its retrieval no longer hold the document"
    ]


async def delete_rows(
    tx: Tx, artifact_id: str, project_id: str
) -> tuple[ArtifactDeleteReport, list[ObjectTarget]]:
    """Every row of the artifact, inside the caller's transaction.

    Raises :class:`ArtifactNotFound` before writing anything when the project
    holds no row of that id.
    """
    rows = await tx.fetch(LOCK_SQL, artifact_id, project_id)
    if not rows:
        raise ArtifactNotFound(artifact_id, project_id)
    node_ids = sorted({str(r[0]) for r in await tx.fetch(NODES_SQL, artifact_id, project_id)})
    counts: dict[str, int] = {}
    for key, sql, scoped in NODE_STEPS:
        args = (node_ids, project_id) if scoped else (node_ids,)
        counts[key] = await tx.execute(sql, *args) if node_ids else 0
    for key, sql in ARTIFACT_STEPS:
        counts[key] = await tx.execute(sql, artifact_id)
    counts["artifacts"] = await tx.execute(DELETE_ARTIFACT_SQL, artifact_id, project_id)
    if counts["artifacts"] != 1:  # the lock held it: anything else is a fault
        raise RuntimeError(f"artifact {artifact_id}: {counts['artifacts']} rows deleted, not 1")
    report = ArtifactDeleteReport(
        artifact_id=artifact_id, project_id=project_id, counts=counts, notes=_notes(rows[0])
    )
    return report, object_targets(rows[0], project_id)


# ── after the commit: the objects ────────────────────────────────────────────


async def _references(tx: Tx, target: ObjectTarget, project_id: str) -> int:
    ref = target.ref if target.bucket else target.key
    rows = await tx.fetch(OBJECT_REFS_SQL, ref, target.content_hash, project_id)
    return int(rows[0][0]) if rows else 0


def _outcome(target: ObjectTarget, bucket: str, outcome: str, detail: str = "") -> dict[str, Any]:
    entry: dict[str, Any] = {"ref": target.ref, "bucket": bucket, "key": target.key}
    entry["outcome"] = outcome
    if detail:
        entry["detail"] = detail
    return entry


async def _delete_object(
    tx: Tx, target: ObjectTarget, project_id: str, store: ObjectStore | None
) -> dict[str, Any]:
    if store is None:
        return _outcome(target, target.bucket or "", "not_attempted", "object store unavailable")
    bucket = target.bucket or store.content_bucket
    if target.bucket and target.bucket not in store.buckets:
        return _outcome(target, bucket, "kept", "not a platform staging bucket")
    shared = await _references(tx, target, project_id)
    if shared:
        return _outcome(target, bucket, "kept_shared", f"{shared} other artifact row(s)")
    try:
        await asyncio.to_thread(store.delete, bucket, target.key)
    except Exception as exc:  # noqa: BLE001 — named in the report; the DB delete stands
        logger.warning("artifact object delete failed: %s/%s: %s", bucket, target.key, exc)
        return _outcome(target, bucket, "failed", str(exc) or type(exc).__name__)
    return _outcome(target, bucket, "deleted")


async def delete_objects(
    tx: Tx,
    report: ArtifactDeleteReport,
    targets: list[ObjectTarget],
    store_factory: Callable[[], ObjectStore | None],
) -> None:
    """Best-effort object deletes after the commit, one outcome each.

    The reference check reads the committed state, so the row just deleted no
    longer counts and any other row holding the bytes keeps them.
    """
    if not targets:
        return
    try:
        store = await asyncio.to_thread(store_factory)  # may probe the store
    except Exception as exc:  # noqa: BLE001 — the objects are reported, not the delete failed
        logger.warning("object store unavailable for artifact delete: %s", exc)
        store = None
    for target in targets:
        report.objects.append(await _delete_object(tx, target, report.project_id, store))


# ── runners ──────────────────────────────────────────────────────────────────


async def delete_artifact(
    pool: Any,
    artifact_id: str,
    project_id: str,
    store_factory: Callable[[], ObjectStore | None],
) -> ArtifactDeleteReport:
    """The delete over an asyncpg pool: one transaction, then the objects."""
    async with pool.acquire() as conn:
        tx = AsyncpgTx(conn)
        async with conn.transaction():
            report, targets = await delete_rows(tx, artifact_id, project_id)
        await delete_objects(tx, report, targets, store_factory)
    return report


def delete_artifact_sync(
    pool: Any,
    artifact_id: str,
    project_id: str,
    store_factory: Callable[[], ObjectStore | None],
) -> ArtifactDeleteReport:
    """The delete over a psycopg2 pool (the stand-alone RAG service)."""
    conn = pool.getconn()
    try:
        # One transaction: ``with conn`` commits on success, rolls back on error.
        with conn, conn.cursor() as cur:
            report, targets = asyncio.run(delete_rows(Psycopg2Tx(cur), artifact_id, project_id))
        with conn, conn.cursor() as cur:
            asyncio.run(delete_objects(Psycopg2Tx(cur), report, targets, store_factory))
    finally:
        pool.putconn(conn)
    return report


async def resolve_document(tx: Tx, document_id: str, project_id: str) -> str | None:
    """The artifact a rag document id names in the project, or None.

    A rag document id is the artifact's own id, the id rag.upload minted
    (``artifact-rag-<document_id>``), or a ``metadata.document_id`` that one
    artifact alone carries.
    """
    if not document_id:
        return None
    ids = [str(r[0]) for r in await tx.fetch(RESOLVE_SQL, document_id, project_id)]
    if not ids:
        return None
    if ids[0] in (document_id, RAG_ARTIFACT_PREFIX + document_id) or len(ids) == 1:
        return ids[0]
    return None


def resolve_document_sync(pool: Any, document_id: str, project_id: str) -> str | None:
    """:func:`resolve_document` over a psycopg2 pool (read-only)."""
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            return asyncio.run(resolve_document(Psycopg2Tx(cur), document_id, project_id))
    finally:
        pool.putconn(conn)


__all__ = [
    "ArtifactDeleteReport",
    "ArtifactNotFound",
    "AsyncpgTx",
    "ObjectStore",
    "ObjectTarget",
    "Psycopg2Tx",
    "Tx",
    "delete_artifact",
    "delete_artifact_sync",
    "delete_objects",
    "delete_rows",
    "object_targets",
    "resolve_document",
    "resolve_document_sync",
    "to_pyformat",
]
