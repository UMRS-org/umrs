"""Knowledge Postgres store module.

CRUD operations for knowledge persistence tables using asyncpg-pool pattern.
No umrs.api or umrs.server imports — strict boundary.

Key operations:
  - insert_artifact: insert-dedup returning the id that holds the bytes
  - insert_ir: write-once (ON CONFLICT DO NOTHING)
  - upsert_node / upsert_edge: node/edge upsert (``conn=`` joins a
    caller's transaction)
  - truncate_nodes / truncate_edges: truncate-and-recompile
  - list_artifacts: paged list with kind/status filters
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .graph_registry import default_graph_id

logger = logging.getLogger(__name__)

# Schema version — must match migration 0055_knowledge_store.sql
_KNOWLEDGE_SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ADR-116 D-8 (KG-06, the part that would undo a review): a re-upsert keeps
# the stored row's ``review`` (written by establish_*) and ``aliases_reviewed``
# (written when a same_as edge is established) unless the new content sets
# them. Done in the statement, so the merge is atomic with the write. Only an
# object content is merged; any other JSON value replaces the stored one as
# before.
_KEPT_CONTENT_KEYS = ("review", "aliases_reviewed")
_MERGED_CONTENT_SQL = (
    "CASE WHEN jsonb_typeof(EXCLUDED.content) = 'object'"
    " AND jsonb_typeof(knowledge_nodes.content) = 'object'"
    " THEN EXCLUDED.content || (SELECT coalesce(jsonb_object_agg(k.key,"
    " knowledge_nodes.content -> k.key), '{}'::jsonb)"
    " FROM unnest(ARRAY[" + ", ".join(f"'{key}'" for key in _KEPT_CONTENT_KEYS) + "])"
    " AS k(key) WHERE knowledge_nodes.content ? k.key AND NOT EXCLUDED.content ? k.key)"
    " ELSE EXCLUDED.content END"
)

_UPSERT_NODE_SQL = f"""INSERT INTO knowledge_nodes
   (node_id, project_id, label, node_type, content, content_hash,
    source_ir, created_at, updated_at, graph_id, status, confidence)
   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
   ON CONFLICT (node_id) DO UPDATE
   SET content = {_MERGED_CONTENT_SQL}, content_hash = $6, source_ir = $7, updated_at = $9"""

# ADR-093 D-6 (0120): a chunk's search_text is the new text when one is
# given; without one the stored text survives only while the content it was
# verified against does (chunk_md5, and a sent-v1 header with its length):
# the column contract is md5(search_text[header_chars:]) = chunk_md5, and a
# re-chunk under the same node id must not leave the lexical lanes ranking
# words the chunk no longer holds (review finding data-deploy-5, 2026-10-10).
_UPSERT_NODE_TEXT_SQL = f"""INSERT INTO knowledge_nodes
   (node_id, project_id, label, node_type, content, content_hash,
    source_ir, created_at, updated_at, graph_id, status, confidence, search_text)
   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
   ON CONFLICT (node_id) DO UPDATE
   SET content = {_MERGED_CONTENT_SQL}, content_hash = $6, source_ir = $7, updated_at = $9,
       search_text = CASE
           WHEN $13::text IS NOT NULL THEN $13::text
           WHEN knowledge_nodes.content->'chunk_md5' IS NOT DISTINCT FROM EXCLUDED.content->'chunk_md5'
            AND knowledge_nodes.content->'header' IS NOT DISTINCT FROM EXCLUDED.content->'header'
            AND knowledge_nodes.content->'header_chars' IS NOT DISTINCT FROM EXCLUDED.content->'header_chars'
           THEN knowledge_nodes.search_text
       END"""

_SEARCH_TEXT_PROBE_SQL = (
    "SELECT EXISTS (SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('knowledge_nodes') AND attname = 'search_text' "
    "AND NOT attisdropped)"
)
# A present column is remembered per pool for good; an absent one is
# re-probed after this many seconds (migration 0120 applied while running).
_SEARCH_TEXT_RECHECK_S = 60.0
# id(pool) -> (pool, present, checked). asyncpg's Pool has __slots__ without
# __weakref__, so a weak-keyed memo never held one and every chunk write
# re-probed (review finding data-deploy-8). The entry holds the pool itself,
# so its id cannot be reused while the entry lives; closed pools (a job's
# pool) are dropped on the next store, and the memo is bounded.
_SEARCH_TEXT_MEMO_MAX = 64
_search_text_probes: dict[int, tuple[Any, bool, float]] = {}


def _known_search_text(pool: Any) -> bool | None:
    """The remembered probe for ``pool``, or None when it must be (re)run."""
    entry = _search_text_probes.get(id(pool))
    if entry is None or entry[0] is not pool:
        return None
    _held, present, checked = entry
    if present or time.monotonic() - checked < _SEARCH_TEXT_RECHECK_S:
        return present
    return None


def _pool_closing(pool: Any) -> bool:
    is_closing = getattr(pool, "is_closing", None)
    return bool(is_closing()) if callable(is_closing) else False


def _remember_search_text(pool: Any, present: bool) -> None:
    """Store one probe; drop closed pools and the oldest beyond the bound."""
    for key, entry in list(_search_text_probes.items()):
        if _pool_closing(entry[0]):
            del _search_text_probes[key]
    _search_text_probes.pop(id(pool), None)
    while len(_search_text_probes) >= _SEARCH_TEXT_MEMO_MAX:
        del _search_text_probes[next(iter(_search_text_probes))]
    _search_text_probes[id(pool)] = (pool, present, time.monotonic())


async def _has_search_text(pool: Any, conn: Any) -> bool:
    """Does ``knowledge_nodes.search_text`` (migration 0120) exist?"""
    known = _known_search_text(pool)
    if known is not None:
        return known
    present = (await conn.fetchval(_SEARCH_TEXT_PROBE_SQL)) is True
    _remember_search_text(pool, present)
    if not present:
        logger.info(
            "knowledge_nodes.search_text absent (migration 0120): nodes written without it"
        )
    return present


def _carries_search_text(node: KnowledgeNode) -> bool:
    """Chunk rows own the column; other nodes never touch it (no probe)."""
    content = node.content if isinstance(node.content, dict) else {}
    return node.search_text is not None or content.get("kind") == "chunk"


@dataclass(frozen=True)
class KnowledgeArtifact:
    """Row from knowledge_artifacts table."""

    artifact_id: str
    content_hash: str
    kind: str  # document, code, table, image, audio, video, embedding
    project_id: str
    source_ref: str = ""
    size_bytes: int = 0
    mime_type: str = ""
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class KnowledgeIRRow:
    """Row from knowledge_ir table."""

    ir_id: str
    artifact_id: str
    ir_type: str  # code, document, schema, catalog, embedding
    version: int = 1
    content: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    status: str = "current"
    created_at: datetime | None = None


@dataclass(frozen=True)
class KnowledgeNode:
    """Row from knowledge_nodes table.

    ``graph_id`` scopes the node to one registered knowledge graph
    (migration 0069). Legacy rows carry ``''`` and are treated as members
    of their project's default graph at read time.
    """

    node_id: str
    project_id: str
    label: str
    node_type: str  # class, function, file, module, concept, entity, relationship
    content: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    source_ir: str | None = None
    # D-35 two-tier knowledge (0106): 'proposed' rows are agent/loop-written
    # pending review; 'established' = human-promoted or vetted-chain ingest.
    status: str = "established"
    confidence: float = 0.0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    graph_id: str = ""
    # ADR-093 D-6 (0120): a chunk's full body for the lexical lanes; its md5
    # is ``content['chunk_md5']``. None = not given (an upsert keeps the
    # stored text while the chunk's md5 and header are unchanged).
    search_text: str | None = None


@dataclass(frozen=True)
class KnowledgeEdge:
    """Row from knowledge_edges table (graph-scoped since migration 0069)."""

    edge_id: str
    project_id: str
    source_id: str
    target_id: str
    relation_type: str
    content: dict[str, Any] = field(default_factory=dict)
    # D-35 two-tier knowledge (0106): same tiers as nodes.
    status: str = "established"
    confidence: float = 0.0
    created_at: datetime | None = None
    graph_id: str = ""


# ── D-35 promotion helpers ────────────────────────────────────────────────────

#: Relation of an entity-resolution edge (ADR-116 D-7).
SAME_AS_RELATION = "same_as"
#: Most partner labels one node keeps in ``content.aliases_reviewed``.
ALIASES_REVIEWED_MAX = 50

_EDGE_FOR_PROMOTION_SQL = (
    "SELECT status, content, relation_type, source_id, target_id, project_id "
    "FROM knowledge_edges WHERE edge_id = $1 FOR UPDATE"
)
# The row lock a node promotion holds until its UPDATE commits: a re-ingest
# upsert of the same node waits, then merges its content over the review
# (D-8), instead of being overwritten by the content read before it.
_NODE_FOR_PROMOTION_SQL = (
    "SELECT status FROM knowledge_nodes WHERE node_id = $1 FOR UPDATE"
)
_REVIEWED_ARRAY = (
    "CASE WHEN jsonb_typeof(n.content->'aliases_reviewed') = 'array' "
    "THEN n.content->'aliases_reviewed' ELSE '[]'::jsonb END"
)
# Each endpoint of an established same_as edge gets the other's label in
# ``content.aliases_reviewed``: appended once (an exact duplicate or the
# node's own label is skipped), at most ALIASES_REVIEWED_MAX entries. One
# statement over both directions; a non-object content is left alone. Both
# nodes must be in the edge's project ($3): an imported edge may name any
# node id, and this must never write into, or read a label from, another
# project.
_COPY_REVIEWED_ALIASES_SQL = (
    "UPDATE knowledge_nodes n SET content = jsonb_set(n.content, '{aliases_reviewed}', "
    f"{_REVIEWED_ARRAY} || to_jsonb(p.label)), updated_at = now() "
    "FROM (VALUES ($1::text, $2::text), ($2::text, $1::text)) AS pair(me, partner) "
    "JOIN knowledge_nodes p ON p.node_id = pair.partner AND p.project_id = $3 "
    "WHERE n.node_id = pair.me AND n.project_id = $3 "
    "AND jsonb_typeof(n.content) = 'object' "
    "AND p.label <> '' AND lower(p.label) <> lower(n.label) "
    f"AND NOT ({_REVIEWED_ARRAY} ? p.label) "
    f"AND jsonb_array_length({_REVIEWED_ARRAY}) < {ALIASES_REVIEWED_MAX}"
)


def _promotion_is_noop(
    row: Any, evidence_refs: list[str] | None, confidence: float | None
) -> bool:
    """An established row promoted again with nothing new to record."""
    return row["status"] == "established" and not evidence_refs and confidence is None


def _reviewed_content(
    content: Any, established_by: str, evidence_refs: list[str] | None
) -> dict[str, Any]:
    """``content`` (the row's fetched value, stamped in place) with
    ``review`` set: the reviewer and the merged, deduplicated evidence refs."""
    merged = content if isinstance(content, dict) else {}
    review = merged.setdefault("review", {})
    review["established_by"] = established_by
    if evidence_refs:
        prior = list(review.get("evidence_refs") or [])
        review["evidence_refs"] = sorted(set(prior) | set(evidence_refs))
    return merged


def _promotion_sets(
    merged: dict[str, Any], confidence: float | None
) -> tuple[str, list[Any]]:
    """The ``SET`` list of a promotion and its parameters (``$2`` onward)."""
    updates = ["status='established'", "content = $2"]
    params: list[Any] = [merged]
    if confidence is not None:
        updates.append("confidence = $3")
        params.append(max(0.0, min(1.0, float(confidence))))
    return ", ".join(updates), params


# ── entity-resolution reads (ADR-116 D-7) ─────────────────────────────────────


@dataclass(frozen=True)
class SemanticNodeRow:
    """What entity resolution reads of one extracted semantic node."""

    node_id: str
    label: str
    artifact_id: str
    semantic_type: str
    aliases: tuple[str, ...] = ()


def _alias_array(key: str) -> str:
    return (
        f"CASE WHEN jsonb_typeof(content->'{key}') = 'array' "
        f"THEN content->'{key}' ELSE '[]'::jsonb END"
    )


_SEMANTIC_NODES_SQL = (
    "SELECT node_id, label, coalesce(content->>'artifact_id', '') AS artifact_id, "
    "content->>'semantic_type' AS semantic_type, "
    "ARRAY(SELECT jsonb_array_elements_text("
    f"{_alias_array('aliases')} || {_alias_array('aliases_reviewed')})) AS aliases "
    "FROM knowledge_nodes WHERE project_id = $1 AND graph_id = ANY($2::text[]) "
    "AND node_id LIKE 'node-sem-%' AND jsonb_typeof(content) = 'object' "
    "AND content ? 'semantic_type' AND content->>'kind' IS DISTINCT FROM 'chunk' "
    "AND ($3::text[] IS NULL OR node_id = ANY($3::text[])) "
    "ORDER BY node_id LIMIT $4"
)
_SAME_AS_PAIRS_SQL = (
    "SELECT source_id, target_id FROM knowledge_edges "
    "WHERE relation_type = 'same_as' "
    "AND (source_id = ANY($1::text[]) OR target_id = ANY($1::text[]))"
)


def graph_members(project_id: str, graph_id: str) -> list[str]:
    """The ``graph_id`` values a graph's rows carry: a named graph its id;
    the project's default graph its id and the legacy ``''``."""
    if not graph_id or graph_id == default_graph_id(project_id):
        return ["", default_graph_id(project_id)]
    return [graph_id]


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _semantic_row(row: Any) -> SemanticNodeRow:
    return SemanticNodeRow(
        node_id=str(row["node_id"]),
        label=str(row["label"] or ""),
        artifact_id=str(row["artifact_id"] or ""),
        semantic_type=str(row["semantic_type"] or ""),
        aliases=tuple(str(a) for a in row["aliases"] or () if a),
    )


#: Postgres SQLSTATE of a unique violation (asyncpg ``UniqueViolationError``).
_UNIQUE_VIOLATION = "23505"


def is_unique_violation(exc: BaseException) -> bool:
    """A unique-constraint violation, read from its SQLSTATE (no driver import)."""
    return getattr(exc, "sqlstate", None) == _UNIQUE_VIOLATION


class ArtifactWriteError(RuntimeError):
    """An artifact row that was not written, named (ADR-093 D-11, wave F):
    the caller must not send its id down the chain."""


_INSERT_ARTIFACT_SQL = """INSERT INTO knowledge_artifacts
   (artifact_id, content_hash, kind, project_id, source_ref,
    size_bytes, mime_type, status, metadata, created_at, updated_at)
   VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8, $9, $10)
   ON CONFLICT (artifact_id) DO UPDATE
   SET updated_at = $10, status = 'active'
   WHERE knowledge_artifacts.project_id = EXCLUDED.project_id"""


class KnowledgeStore:
    """Knowledge Postgres store using asyncpg pool.

    Provides CRUD operations backed by migration 0055 tables.
    """

    # Authoritative ontology for knowledge_nodes.node_type (matches the live
    # DB CHECK constraint ``knowledge_nodes_node_type_check``). Semantic roles
    # (job / job_failure / provenance) live in labels; the DB category is one
    # of these values only.
    ALLOWED_NODE_TYPES: frozenset[str] = frozenset(
        {"class", "function", "file", "module", "concept", "entity", "relationship"}
    )

    def __init__(self, pool) -> None:  # type: ignore[misc]  # asyncpg Pool
        """Initialize store with an asyncpg connection pool."""
        self._pool = pool

    async def active_artifact_for(
        self, content_hash: str, project_id: str
    ) -> str | None:
        """The project's active artifact holding these bytes (per-project dedup)."""
        async with self._pool.acquire() as conn:
            return await conn.fetchval(
                """SELECT artifact_id FROM knowledge_artifacts
                   WHERE content_hash = $1 AND project_id = $2 AND status = 'active'
                   LIMIT 1""",
                content_hash,
                project_id,
            )

    @contextlib.asynccontextmanager
    async def _using(self, conn: Any = None) -> AsyncIterator[Any]:
        """``conn`` when the caller holds one (its transaction), else a pool
        connection of our own."""
        if conn is not None:
            yield conn
            return
        async with self._pool.acquire() as own:
            yield own

    async def insert_artifact(
        self,
        artifact_id: str,
        content_hash: str,
        kind: str,
        project_id: str,
        source_ref: str = "",
        size_bytes: int = 0,
        mime_type: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert (or reactivate) artifact metadata; return the artifact id
        that holds these bytes in the project.

        That is ``artifact_id`` itself when it was inserted or reactivated,
        or the stored id when the project already holds the same bytes
        (per-project dedup, audit B8). Bytes without a hash are never matched:
        ``''`` is no identity (ADR-093 D-11, wave F: a second hashless upload
        was "deduped" against the first and went down the chain with no row).
        A row that could not be written raises :class:`ArtifactWriteError`
        by name — never a silent False. Two uploads of the same bytes racing
        past the probe resolve to the stored id.
        """
        if content_hash:
            existing = await self.active_artifact_for(content_hash, project_id)
            if existing:
                logger.debug(
                    "Artifact deduped: hash=%s (exists as %s)", content_hash, existing
                )
                return existing
        # NOTE: knowledge_artifacts.created_at/updated_at are TEXT columns
        # on live deployments (0064-shim migration shape). asyncpg rejects
        # datetime values for them — persist as ISO-8601 strings.
        ts = _utcnow().isoformat()
        params = (artifact_id, content_hash, kind, project_id, source_ref)
        try:
            async with self._pool.acquire() as conn:
                status = await conn.execute(
                    _INSERT_ARTIFACT_SQL,
                    *params,
                    size_bytes,
                    mime_type,
                    metadata or {},
                    ts,
                    ts,
                )
        except Exception as exc:
            if not (content_hash and is_unique_violation(exc)):
                raise
            return await self._stored_after_race(artifact_id, content_hash, project_id)
        if str(status).strip().endswith(" 0"):
            raise ArtifactWriteError(
                f"artifact {artifact_id} exists in another project: no row written "
                f"for project {project_id}"
            )
        logger.debug("Artifact inserted: %s (hash=%s)", artifact_id, content_hash)
        return artifact_id

    async def _stored_after_race(
        self, artifact_id: str, content_hash: str, project_id: str
    ) -> str:
        """The id a concurrent insert of the same bytes stored, else a named
        failure."""
        stored = await self.active_artifact_for(content_hash, project_id)
        if not stored:
            raise ArtifactWriteError(
                f"artifact {artifact_id}: unique violation on its bytes, yet no "
                f"active row holds hash {content_hash} in project {project_id}"
            )
        return stored

    async def insert_ir(
        self,
        ir_id: str,
        artifact_id: str,
        ir_type: str,
        content: dict[str, Any],
        content_hash: str,
        version: int = 1,
    ) -> bool:
        """Insert IR write-once (ON CONFLICT DO NOTHING for idempotency).

        Returns True if inserted (new), False if already exists.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """INSERT INTO knowledge_ir
                       (ir_id, artifact_id, version, ir_type, content, content_hash, status)
                       VALUES ($1, $2, $3, $4, $5, $6, 'current')
                       ON CONFLICT (ir_id) DO NOTHING""",
                    ir_id,
                    artifact_id,
                    version,
                    ir_type,
                    content,
                    content_hash,
                )
                inserted = result != "INSERT 0 0"
                return inserted
        except Exception as exc:
            # Only a duplicate is an expected shape. A foreign-key violation
            # (no artifact row) fails the caller's item by name: logging it
            # as "likely duplicate" let understand build nodes on a missing
            # IR (ADR-093 D-11, wave F).
            if not is_unique_violation(exc):
                raise
            logger.warning("IR insert was a duplicate: %s: %s", ir_id, exc)
            return False

    async def upsert_node(
        self,
        node: KnowledgeNode,
        *,
        conn: Any = None,
    ) -> None:
        """Upsert a knowledge graph node (graph-scoped; first-write wins graph).

        D-35 (0106): status/confidence ride the row — an agent/loop-written
        node upserted with ``status='proposed'`` stays proposed; the ON
        CONFLICT update never silently promotes (status is only written at
        INSERT; promotion is the human ``establish_node`` seam).

        ADR-093 D-6 (0120): a node given ``search_text`` stores it. A chunk
        upserted without text keeps its stored text only while ``chunk_md5``
        and the header it was verified against are unchanged; otherwise the
        column goes NULL, for the backfill to refill from verified text.
        Non-chunk nodes never touch the column. Without migration 0120 the
        row is written without it (probed once per pool). ``conn`` joins the
        caller's transaction.
        """
        async with self._using(conn) as db:
            ts = _utcnow()
            params: list[Any] = [
                node.node_id,
                node.project_id,
                node.label,
                node.node_type,
                node.content,
                node.content_hash,
                node.source_ir,
                node.created_at or ts,
                ts,
                node.graph_id,
                node.status,
                node.confidence,
            ]
            if _carries_search_text(node) and await _has_search_text(self._pool, db):
                await db.execute(_UPSERT_NODE_TEXT_SQL, *params, node.search_text)
            else:
                await db.execute(_UPSERT_NODE_SQL, *params)

    async def upsert_edge(
        self,
        edge: KnowledgeEdge,
        *,
        conn: Any = None,
    ) -> None:
        """Upsert a knowledge graph edge (graph-scoped; D-35 tiers as nodes).
        ``conn`` joins the caller's transaction."""
        async with self._using(conn) as db:
            await db.execute(
                """INSERT INTO knowledge_edges
                   (edge_id, project_id, source_id, target_id, relation_type, content,
                    graph_id, status, confidence)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   ON CONFLICT (edge_id) DO UPDATE
                   SET content = $6""",
                edge.edge_id,
                edge.project_id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.content,
                edge.graph_id,
                edge.status,
                edge.confidence,
            )

    async def establish_node(
        self,
        node_id: str,
        *,
        established_by: str,
        evidence_refs: list[str] | None = None,
        confidence: float | None = None,
    ) -> bool:
        """D-35 promotion: proposed → established (human decision, 0106).

        Returns False when the row is absent; promoting an already-
        established row is an idempotent no-op returning True. The
        promotion records the human's evidence refs (merged, deduped) and
        optionally the settled confidence (only when provided — an absent
        value keeps the extraction confidence, never invented). The row is
        locked from the read to the write (one transaction), so a concurrent
        re-ingest is not overwritten with the content read before it.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(_NODE_FOR_PROMOTION_SQL, node_id)
            if row is None:
                return False
            if _promotion_is_noop(row, evidence_refs, confidence):
                return True  # idempotent no-op, review untouched
            content = await conn.fetchval(
                "SELECT content FROM knowledge_nodes WHERE node_id = $1", node_id
            )
            merged = _reviewed_content(content, established_by, evidence_refs)
            sets, params = _promotion_sets(merged, confidence)
            sets += ", updated_at=now()"
            await conn.execute(
                f"UPDATE knowledge_nodes SET {sets} WHERE node_id=$1",
                node_id,
                *params,
            )
            return True

    async def establish_edge(
        self,
        edge_id: str,
        *,
        established_by: str,
        evidence_refs: list[str] | None = None,
        confidence: float | None = None,
    ) -> bool:
        """D-35 promotion for edges (same semantics as establish_node).

        ``knowledge_edges`` has no ``updated_at`` column (0055), so the
        promotion writes none (ADR-116 D-8: writing it failed every edge
        promotion on Postgres with UndefinedColumn). Promoting a ``same_as``
        edge also appends each endpoint's label to the other node's
        ``content.aliases_reviewed`` in the same transaction (ADR-116 D-7),
        which the graph retrieval lane matches beside ``content.aliases``.
        """
        async with self._pool.acquire() as conn, conn.transaction():
            row = await conn.fetchrow(_EDGE_FOR_PROMOTION_SQL, edge_id)
            if row is None:
                return False
            if _promotion_is_noop(row, evidence_refs, confidence):
                return True  # idempotent no-op, review untouched
            merged = _reviewed_content(
                row.get("content"), established_by, evidence_refs
            )
            sets, params = _promotion_sets(merged, confidence)
            await conn.execute(
                f"UPDATE knowledge_edges SET {sets} WHERE edge_id=$1",
                edge_id,
                *params,
            )
            if row.get("relation_type") == SAME_AS_RELATION:
                await conn.execute(
                    _COPY_REVIEWED_ALIASES_SQL,
                    row.get("source_id"),
                    row.get("target_id"),
                    row.get("project_id"),
                )
            return True

    async def truncate_nodes(
        self,
        project_id: str,
        source_ir: str | None = None,
        graph_id: str | None = None,
    ) -> int:
        """Delete nodes, optionally filtered by project + source_ir + graph.

        When ``graph_id`` is given, legacy rows (``graph_id = ''``) are
        included only when the truncated graph is the project's default
        graph. Returns number of rows deleted.
        """
        async with self._pool.acquire() as conn:
            if graph_id is not None:
                is_default = graph_id == default_graph_id(project_id)
                if source_ir:
                    result = await conn.execute(
                        "DELETE FROM knowledge_nodes "
                        "WHERE project_id = $1 AND source_ir = $2 "
                        "AND (graph_id = $3 OR (graph_id = '' AND $4))",
                        project_id,
                        source_ir,
                        graph_id,
                        is_default,
                    )
                else:
                    result = await conn.execute(
                        "DELETE FROM knowledge_nodes "
                        "WHERE project_id = $1 "
                        "AND (graph_id = $2 OR (graph_id = '' AND $3))",
                        project_id,
                        graph_id,
                        is_default,
                    )
            elif source_ir:
                result = await conn.execute(
                    "DELETE FROM knowledge_nodes WHERE project_id = $1 AND source_ir = $2",
                    project_id,
                    source_ir,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM knowledge_nodes WHERE project_id = $1",
                    project_id,
                )
            return int(result.split()[-1])

    async def truncate_edges(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        *,
        project_id: str | None = None,
        graph_id: str | None = None,
    ) -> int:
        """Delete edges with optional source/target/project/graph filters.

        Returns number of rows deleted.

        Without filters: deletes ALL edges (full truncate for recompile).
        With source_id: deletes edges where source node is removed.
        With target_id: deletes edges where target node is removed.
        With graph_id (requires project_id): graph-scoped delete with the
        legacy '' fallback when truncating the project's default graph.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if graph_id is not None:
            if project_id is None:
                raise ValueError("project_id is required when graph_id is given")
            is_default = graph_id == default_graph_id(project_id)
            params.extend([graph_id, is_default])
            conditions.append(
                f"(graph_id = ${len(params) - 1} OR (graph_id = '' AND ${len(params)}))"
            )
        if project_id is not None:
            params.append(project_id)
            conditions.append(f"project_id = ${len(params)}")
        if source_id is not None:
            params.append(source_id)
            conditions.append(f"source_id = ${len(params)}")
        if target_id is not None:
            params.append(target_id)
            conditions.append(f"target_id = ${len(params)}")
        where = " AND ".join(conditions) if conditions else "TRUE"
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"DELETE FROM knowledge_edges WHERE {where}",
                *params,
            )
            return int(result.split()[-1])

    async def benchmark_node_status(
        self, project_id: str, node_ids: list[str]
    ) -> dict[str, str | None]:
        """D-44 domain benchmark: node_id → row status (None = absent).

        The scoring base for catalog-quality measurement — mechanically
        real (node existence + two-tier status), no LLM in the loop.
        """
        if not node_ids:
            return {}
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT node_id, status FROM knowledge_nodes"
                " WHERE project_id = $1 AND node_id = ANY($2)",
                project_id,
                node_ids,
            )
        found = {r["node_id"]: r["status"] for r in rows}
        return {nid: found.get(nid) for nid in node_ids}

    async def count_graph_nodes(self, project_id: str) -> int:
        """Row tally of knowledge_nodes within the project (used by the
        readiness-coverage metric; table-family deterministic by design)."""
        if not project_id:
            return 0
        async with self._pool.acquire() as conn:
            val = await conn.fetchval(
                "SELECT count(*) FROM knowledge_nodes WHERE project_id = $1",
                project_id,
            )
        return int(val or 0)

    async def rebind_artifact_source(
        self, project_id: str, artifact_id: str, source_ref: str
    ) -> bool:
        """Point one artifact at its object in THIS installation's storage
        (raw-materials import); ids and hashes are untouched."""
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                """UPDATE knowledge_artifacts SET source_ref = $3
                   WHERE project_id = $1 AND artifact_id = $2""",
                project_id,
                artifact_id,
                source_ref,
            )
        return str(status).endswith(" 1")

    async def artifacts_by_canonical_document(
        self, project_id: str, document_ids: list[str]
    ) -> dict[str, list[str]]:
        """Active artifacts whose carried canonical identity (ADR-032 #1,
        ``metadata.canonical_identity.document_id``) is one of *document_ids*."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT artifact_id,
                          metadata->'canonical_identity'->>'document_id' AS document_id
                   FROM knowledge_artifacts
                   WHERE project_id = $1 AND status = 'active'
                     AND metadata->'canonical_identity'->>'document_id' = ANY($2::text[])""",
                project_id,
                document_ids,
            )
        found: dict[str, list[str]] = {}
        for row in rows:
            found.setdefault(str(row["document_id"]), []).append(
                str(row["artifact_id"])
            )
        return found

    async def artifacts_missing_identity(
        self, project_id: str | None, after: str, limit: int
    ) -> list[dict[str, Any]]:
        """Active artifacts with no carried canonical identity (ADR-032 #1),
        keyset-paged by artifact_id (``after`` = last id seen, "" = start)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT artifact_id, content_hash, source_ref, size_bytes
                   FROM knowledge_artifacts
                   WHERE status = 'active'
                     AND NOT (COALESCE(metadata, '{}'::jsonb) ? 'canonical_identity')
                     AND ($1::text IS NULL OR project_id = $1)
                     AND artifact_id > $2
                   ORDER BY artifact_id
                   LIMIT $3""",
                project_id,
                after,
                limit,
            )
        return [dict(row) for row in rows]

    async def stamp_canonical_identity(
        self, artifact_id: str, block: dict[str, str]
    ) -> bool:
        """Add the identity block to one artifact's metadata; never overwrites
        an existing block (True = stamped)."""
        async with self._pool.acquire() as conn:
            status = await conn.execute(
                """UPDATE knowledge_artifacts
                   SET metadata = COALESCE(metadata, '{}'::jsonb)
                       || jsonb_build_object('canonical_identity', $2::jsonb)
                   WHERE artifact_id = $1
                     AND NOT (COALESCE(metadata, '{}'::jsonb) ? 'canonical_identity')""",
                artifact_id,
                block,
            )
        return str(status).endswith(" 1")

    async def list_artifacts(
        self,
        project_id: str,
        kind: str | None = None,
        status: str = "active",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[KnowledgeArtifact], int]:
        """Paged list of artifacts with optional filters.

        Returns (artifact_list, total_count).
        """
        async with self._pool.acquire() as conn:
            conditions = ["project_id = $1", "status = $2"]
            params: list[Any] = [project_id, status]
            param_idx = 3

            if kind:
                conditions.append(f"kind = ${param_idx}")
                params.append(kind)
                param_idx += 1

            where = " AND ".join(conditions)
            offset = (page - 1) * page_size

            # Get total count
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM knowledge_artifacts WHERE {where}",
                *params,
            )
            total = int(count_row["count"]) if count_row else 0

            # Get paged results
            rows = await conn.fetch(
                f"""SELECT artifact_id, content_hash, kind, project_id,
                          source_ref, size_bytes, mime_type, status,
                          metadata, created_at, updated_at
                   FROM knowledge_artifacts
                   WHERE {where}
                   ORDER BY created_at DESC, artifact_id
                   LIMIT ${param_idx} OFFSET ${param_idx + 1}""",
                *params,
                page_size,
                offset,
            )

            artifacts = []
            for row in rows:
                artifacts.append(KnowledgeArtifact(**row))

            return artifacts, total

    async def get_artifact(self, artifact_id: str) -> KnowledgeArtifact | None:
        """Get a single artifact by ID, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT artifact_id, content_hash, kind, project_id,
                          source_ref, size_bytes, mime_type, status,
                          metadata, created_at, updated_at
                   FROM knowledge_artifacts
                   WHERE artifact_id = $1""",
                artifact_id,
            )
            return KnowledgeArtifact(**row) if row else None

    async def get_ir(self, ir_id: str) -> KnowledgeIRRow | None:
        """Get a single IR by ID, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT ir_id, artifact_id, version, ir_type,
                          content, content_hash, status, created_at
                   FROM knowledge_ir
                   WHERE ir_id = $1""",
                ir_id,
            )
            return KnowledgeIRRow(**row) if row else None

    async def list_ir_by_ids(self, ir_ids: list[str]) -> list[KnowledgeIRRow]:
        """Get multiple IRs by their ids.

        Returns rows in the same order as *ir_ids*, skipping missing ids.
        """
        if not ir_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT ir_id, artifact_id, version, ir_type,
                          content, content_hash, status, created_at
                   FROM knowledge_ir
                   WHERE ir_id = ANY($1)""",
                ir_ids,
            )
            return [KnowledgeIRRow(**dict(r)) for r in rows]

    async def get_schema_version(self) -> int:
        """Return the current knowledge schema version."""
        return _KNOWLEDGE_SCHEMA_VERSION

    async def list_irs(
        self,
        project_id: str,
        ir_type: str | None = None,
        *,
        status: str | None = "current",
    ) -> list[KnowledgeIRRow]:
        """Return a project's IRs, oldest first; optional ``ir_type``.

        ``status`` defaults to ``current``: a re-extraction marks the
        artifact's earlier IRs ``superseded`` (ADR-116) and drops the rows
        they wrote, so the bundle export, the dashboard and the verifiers
        read only the IR each artifact stands on. ``status=None`` returns
        every IR, whatever its status (the behaviour before 2026-11-12).
        """
        params: list[Any] = [project_id]
        clauses = ""
        if ir_type:
            params.append(ir_type)
            clauses += f" AND ir_type = ${len(params)}"
        if status is not None:
            params.append(status)
            clauses += f" AND COALESCE(status, 'current') = ${len(params)}"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT ir_id, artifact_id, version, ir_type, "
                "content, content_hash, status, created_at "
                "FROM knowledge_ir WHERE artifact_id IN ("
                "SELECT artifact_id FROM knowledge_artifacts WHERE project_id = $1)"
                f"{clauses} ORDER BY created_at",
                *params,
            )
            return [KnowledgeIRRow(**dict(r)) for r in rows]

    async def list_nodes(
        self,
        project_id: str,
        graph_id: str | None = None,
    ) -> list[KnowledgeNode]:
        """Return all nodes for a project, optionally one graph only.

        Graph filtering uses the legacy fallback: rows with ``graph_id = ''``
        count as members of the project's default graph.
        """
        params: list[Any] = [project_id]
        graph_clause = ""
        if graph_id is not None:
            is_default = graph_id == default_graph_id(project_id)
            params.extend([graph_id, is_default])
            graph_clause = " AND (graph_id = $2 OR (graph_id = '' AND $3))"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT node_id, project_id, label, node_type,
                          content, content_hash, source_ir,
                          created_at, updated_at, graph_id
                   FROM knowledge_nodes
                   WHERE project_id = $1{graph_clause}
                   ORDER BY created_at""",
                *params,
            )
            return [KnowledgeNode(**dict(r)) for r in rows]

    async def walk_closure(
        self,
        project_id: str,
        graph_id: str,
        start_node_id: str,
        relation_type: str,
        direction: str = "outgoing",
        max_depth: int = 64,
    ) -> list[KnowledgeEdge]:
        """Transitive closure over ONE canonical relation (ADR-049 Leg C).

        Recursive CTE with a bind-arg depth guard and a path-membership
        cycle guard — termination is a property of the SQL, never of
        traversal luck. Result rows are the traversed edges (source/target
        carry canonical ids; ``content['walk_depth']`` records the depth of
        the target hop), ordered deterministic (depth, edge_id).

        The caller is the canonical view (worker-routed pool); this method
        is never combinable with the sat-collapse names (SUPPORTS /
        CONTRADICTS family) — the view fails those closed upstream.
        """
        if direction not in ("outgoing", "incoming"):
            raise ValueError(f"direction must be outgoing|incoming, got {direction!r}")
        max_depth = max(1, min(int(max_depth), 512))
        is_default = graph_id == default_graph_id(project_id)
        graph_clause = " AND (graph_id = $5 OR (graph_id = '' AND $6))"
        if direction == "outgoing":
            sql = f"""
                WITH RECURSIVE walk AS (
                  SELECT e.target_id AS node_id, e.edge_id, 1 AS depth,
                         ARRAY[$3::text, e.target_id]::text[] AS path
                  FROM knowledge_edges e
                  WHERE e.project_id = $1 AND e.source_id = $3
                    AND e.relation_type = $4{graph_clause}
                  UNION
                  SELECT e.target_id, e.edge_id, w.depth + 1, w.path || e.target_id
                  FROM knowledge_edges e
                  JOIN walk w ON e.source_id = w.node_id
                  WHERE e.project_id = $1 AND e.relation_type = $4{graph_clause}
                    AND w.depth < $2
                    AND NOT (e.target_id = ANY(w.path))
                ),
                dedup AS (
                  SELECT edge_id, MIN(depth) AS depth
                  FROM walk
                  GROUP BY edge_id
                )
                SELECT d.depth, e2.edge_id, e2.project_id, e2.source_id,
                       e2.target_id, e2.relation_type, e2.content,
                       e2.created_at, e2.graph_id
                FROM dedup d
                JOIN knowledge_edges e2 ON e2.edge_id = d.edge_id
                ORDER BY d.depth, e2.edge_id
            """
        else:
            sql = f"""
                WITH RECURSIVE walk AS (
                  SELECT e.source_id AS node_id, e.edge_id, 1 AS depth,
                         ARRAY[$3::text, e.source_id]::text[] AS path
                  FROM knowledge_edges e
                  WHERE e.project_id = $1 AND e.target_id = $3
                    AND e.relation_type = $4{graph_clause}
                  UNION
                  SELECT e.source_id, e.edge_id, w.depth + 1, w.path || e.source_id
                  FROM knowledge_edges e
                  JOIN walk w ON e.target_id = w.node_id
                  WHERE e.project_id = $1 AND e.relation_type = $4{graph_clause}
                    AND w.depth < $2
                    AND NOT (e.source_id = ANY(w.path))
                ),
                dedup AS (
                  SELECT edge_id, MIN(depth) AS depth
                  FROM walk
                  GROUP BY edge_id
                )
                SELECT d.depth, e2.edge_id, e2.project_id, e2.source_id,
                       e2.target_id, e2.relation_type, e2.content,
                       e2.created_at, e2.graph_id
                FROM dedup d
                JOIN knowledge_edges e2 ON e2.edge_id = d.edge_id
                ORDER BY d.depth, e2.edge_id
            """
        params: list[Any] = [
            project_id,
            max_depth,
            start_node_id,
            relation_type,
            graph_id,
            is_default,
        ]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        edges = []
        for row in rows:
            payload = dict(row)
            depth = payload.pop("depth")
            content = dict(payload.get("content") or {})
            content["walk_depth"] = depth
            payload["content"] = content
            edges.append(KnowledgeEdge(**payload))
        return edges

    async def list_edges(
        self,
        project_id: str,
        graph_id: str | None = None,
    ) -> list[KnowledgeEdge]:
        """Return all edges for a project, optionally one graph only.

        Same legacy '' fallback as :meth:`list_nodes`.
        """
        params: list[Any] = [project_id]
        graph_clause = ""
        if graph_id is not None:
            is_default = graph_id == default_graph_id(project_id)
            params.extend([graph_id, is_default])
            graph_clause = " AND (graph_id = $2 OR (graph_id = '' AND $3))"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""SELECT edge_id, project_id, source_id, target_id,
                          relation_type, content, created_at, graph_id
                   FROM knowledge_edges
                   WHERE project_id = $1{graph_clause}
                   ORDER BY created_at""",
                *params,
            )
            return [KnowledgeEdge(**dict(r)) for r in rows]

    async def insert_event(
        self,
        event_type: str,
        job_id: str,
        attempt_id: str,
        artifact_id: str,
        source_ref: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Insert a knowledge event into knowledge_events table.

        Returns the event_id of the inserted row.
        """
        import uuid

        from .events import _utcnow  # avoid top-level import loop

        event_id = uuid.uuid4().hex
        ts = _utcnow()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO knowledge_events
                   (event_id, event_type, job_id, attempt_id, artifact_id,
                    source_ref, details, timestamp, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                event_id,
                event_type,
                job_id,
                attempt_id,
                artifact_id,
                source_ref,
                details or {},
                ts,
                ts,
            )
        return event_id

    async def list_events(
        self,
        job_id: str,
        artifact_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """List events for a job, optionally filtered by artifact and type.

        Returns list of dicts with keys: event_id, event_type, job_id,
        attempt_id, artifact_id, source_ref, details, timestamp.
        """
        async with self._pool.acquire() as conn:
            conditions = ["job_id = $1"]
            params: list[Any] = [job_id]
            param_idx = 2

            if artifact_id:
                conditions.append(f"artifact_id = ${param_idx}")
                params.append(artifact_id)
                param_idx += 1

            if event_type:
                conditions.append(f"event_type = ${param_idx}")
                params.append(event_type)
                param_idx += 1

            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""SELECT event_id, event_type, job_id, attempt_id,
                           artifact_id, source_ref, details, timestamp
                    FROM knowledge_events
                    WHERE {where}
                    ORDER BY timestamp""",
                *params,
            )
            return [dict(r) for r in rows]

    # ── Catalog helpers ───────────────────────────────────────────────────────

    async def list_catalog(
        self,
        project_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """List catalog entries (knowledge_artifacts with kind='catalog') for a project.

        Returns (entries, total_count).
        """
        async with self._pool.acquire() as conn:
            where = "project_id = $1 AND kind = 'catalog' AND status = 'active'"
            offset = (page - 1) * page_size

            total_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM knowledge_artifacts WHERE {where}",
                project_id,
            )
            total = int(total_row["count"]) if total_row else 0

            rows = await conn.fetch(
                f"""SELECT artifact_id, content_hash, kind, project_id,
                           source_ref, size_bytes, mime_type, status,
                           metadata, created_at, updated_at
                    FROM knowledge_artifacts
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT ${2} OFFSET ${3}""",
                project_id,
                page_size,
                offset,
            )
            entries: list[dict[str, Any]] = [dict(r) for r in rows]
            return entries, total

    async def list_embedding_vectors(
        self,
        project_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return embedding vectors for a project (memory_vectors table).

        Returns list of dicts with keys: id, vector, metadata, document_id, created_at.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, vector, metadata, document_id, created_at
                   FROM memory_vectors
                   WHERE project_id = $1
                   ORDER BY id
                   LIMIT $2""",
                project_id,
                limit,
            )
            return [dict(r) for r in rows]

    async def count_embedding_vectors(
        self,
        project_id: str,
    ) -> int:
        """Return total count of embedding vectors for a project."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM memory_vectors WHERE project_id = $1",
                project_id,
            )
            return int(row["count"]) if row else 0

    async def list_embedding_vectors_paged(
        self,
        project_id: str,
        cursor: str,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paged fetch of memory_vectors using id-based cursor.

        Returns (rows, total_count).  Rows have keys: id, vector, metadata,
        document_id, created_at.
        """
        async with self._pool.acquire() as conn:
            total_row = await conn.fetchrow(
                "SELECT COUNT(*) FROM memory_vectors WHERE project_id = $1",
                project_id,
            )
            total = int(total_row["count"]) if total_row else 0

            rows = await conn.fetch(
                """SELECT id, vector, metadata, document_id, created_at
                   FROM memory_vectors
                   WHERE project_id = $1 AND id > $2
                   ORDER BY id
                   LIMIT $3""",
                project_id,
                cursor,
                page_size,
            )
            return [dict(r) for r in rows], total

    # ── Node-keyed embeddings (ADR-027 G5 / P1b, migration 0075) ────────
    #
    # Additive foundation for the P2 facade switch: these rows are the
    # canonical graph's own vectors — NOT the per-project rag chunk tables
    # (Gen1) and NOT memory_vectors (context memory). R6.1 ownership assigns
    # the write path later; the storage contract lands here first so both
    # candidate writers can adopt it without a further schema change.

    @staticmethod
    def _vector_literal(embedding: list[float]) -> str:
        """pgvector text form '[f1,f2,...]' — cast server-side ($n::vector)."""
        return "[" + ",".join(repr(float(v)) for v in embedding) + "]"

    async def upsert_node_embedding(
        self,
        node_id: str,
        model_name: str,
        embedding: list[float],
        *,
        content_hash: str = "",
        conn: Any = None,
    ) -> None:
        """Insert or refresh the node-keyed vector row for one model. ``conn``
        joins the caller's transaction."""
        sql = """
            INSERT INTO knowledge_node_embeddings
                (node_id, model_name, dimension, embedding, content_hash)
            VALUES ($1, $2, $3, $4::vector, $5)
            ON CONFLICT (node_id, model_name) DO UPDATE SET
                dimension    = EXCLUDED.dimension,
                embedding    = EXCLUDED.embedding,
                content_hash = EXCLUDED.content_hash,
                updated_at   = now()
            """
        async with self._using(conn) as db:
            await db.execute(
                sql,
                node_id,
                model_name,
                len(embedding),
                self._vector_literal(embedding),
                content_hash,
            )

    async def nearest_node_ids(
        self,
        embedding: list[float],
        *,
        project_id: str,
        model_name: str,
        limit: int = 20,
        graph_id: str | None = None,
    ) -> list[tuple[str, float]]:
        """Cosine-nearest canonical nodes (project + model scoped).

        ``graph_id`` (optional) keeps the nodes of that graph only; the
        project's default graph also holds its legacy ``graph_id = ''`` rows.

        Exact scan ORDER BY ``embedding <=> :v`` — no ANN index on the
        untyped-dimension column (requires fixed dim). Expected table size at
        this stage (chunk-capped uploads) is small; a typed shadow column +
        hnsw index is the follow-up when a canonical model is pinned (0075
        notes)."""
        sql = """
            SELECT e.node_id, e.embedding <=> $4::vector AS distance
              FROM knowledge_node_embeddings e
              JOIN knowledge_nodes n ON n.node_id = e.node_id
             WHERE e.model_name = $1
               AND n.project_id = $2
               AND ($5::text[] IS NULL OR n.graph_id = ANY($5::text[]))
             ORDER BY distance ASC
             LIMIT $3
            """
        graphs = None if graph_id is None else graph_members(project_id, graph_id)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                model_name,
                project_id,
                limit,
                self._vector_literal(embedding),
                graphs,
            )
            return [(str(r["node_id"]), float(r["distance"])) for r in rows]

    async def nearest_label_neighbours(
        self,
        queries: list[tuple[str, str, list[float]]],
        *,
        project_id: str,
        model_name: str,
        limit: int,
        graph_id: str | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Entity resolution's neighbour search (ADR-116 D-7): for each
        ``(node_id, artifact_id, vector)`` query, the ``limit`` nearest
        vectors of ``model_name`` in the project (and graph), leaving out the
        node itself and the nodes of its own artifact BEFORE the limit, so
        close nodes of the same document cannot crowd out a cross-document
        match. One statement for all queries (a LATERAL scan each)."""
        if not queries:
            return {}
        sql = """
            SELECT q.node_id AS query_id, m.node_id, m.distance
              FROM unnest($1::text[], $2::text[], $3::text[]) AS q(node_id, artifact_id, vec)
             CROSS JOIN LATERAL (
                SELECT e.node_id, e.embedding <=> q.vec::vector AS distance
                  FROM knowledge_node_embeddings e
                  JOIN knowledge_nodes n ON n.node_id = e.node_id
                 WHERE e.model_name = $4
                   AND n.project_id = $5
                   AND ($6::text[] IS NULL OR n.graph_id = ANY($6::text[]))
                   AND n.node_id <> q.node_id
                   AND (n.content->>'artifact_id') IS DISTINCT FROM q.artifact_id
                 ORDER BY distance ASC
                 LIMIT $7
             ) m
            """
        graphs = None if graph_id is None else graph_members(project_id, graph_id)
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                sql,
                [q[0] for q in queries],
                [q[1] for q in queries],
                [self._vector_literal(q[2]) for q in queries],
                model_name,
                project_id,
                graphs,
                limit,
            )
        found: dict[str, list[tuple[str, float]]] = {q[0]: [] for q in queries}
        for r in rows:
            found[str(r["query_id"])].append((str(r["node_id"]), float(r["distance"])))
        return found

    # ── Entity resolution reads (ADR-116 D-7) ────────────────────────────

    async def list_semantic_nodes(
        self,
        project_id: str,
        *,
        graph_id: str,
        node_ids: list[str] | None = None,
        limit: int = 50_000,
    ) -> list[SemanticNodeRow]:
        """The extracted semantic nodes (``node-sem-*`` with
        ``content.semantic_type``, never a chunk) of one project and graph,
        ``node_ids`` only when given, at most ``limit`` in node-id order.
        Aliases are ``content.aliases`` and ``content.aliases_reviewed``."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                _SEMANTIC_NODES_SQL,
                project_id,
                graph_members(project_id, graph_id),
                node_ids,
                limit,
            )
        return [_semantic_row(r) for r in rows]

    async def same_as_pairs(self, node_ids: list[str]) -> set[tuple[str, str]]:
        """Sorted ``(a, b)`` node pairs already linked by a ``same_as`` edge
        (either tier), for pairs touching any of ``node_ids``."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(_SAME_AS_PAIRS_SQL, node_ids)
        return {_sorted_pair(str(r["source_id"]), str(r["target_id"])) for r in rows}

    # ── Knowledge↔A2L Links ──────────────────────────────────────────────

    async def insert_knowledge_a2l_link(
        self,
        knowledge_artifact_id: str,
        a2l_artifact_id: str,
        job_id: str = "",
        attempt_id: str = "",
        edge_kind: str = "derived",
    ) -> bool:
        """Register a knowledge artifact → A2L link (ON CONFLICT DO NOTHING)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO knowledge_a2l_links
                    (knowledge_artifact_id, a2l_artifact_id, job_id, attempt_id, edge_kind)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (knowledge_artifact_id, a2l_artifact_id)
                DO NOTHING
                RETURNING id
                """,
                knowledge_artifact_id,
                a2l_artifact_id,
                job_id,
                attempt_id,
                edge_kind,
            )
            return row is not None

    # ── A2L Bridge ─────────────────────────────────────────────────────────

    async def insert_a2l_ref(
        self,
        artifact_id: str,
        a2l_hash: str,
        a2l_id: str,
        source: str = "auto",
    ) -> bool:
        """Persist an A2L→UMRS artifact mapping (ON CONFLICT DO NOTHING)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO a2l_artifact_refs (artifact_id, a2l_hash, a2l_id, source)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (artifact_id, a2l_hash) DO NOTHING
                RETURNING id
                """,
                artifact_id,
                a2l_hash,
                a2l_id,
                source,
            )
            return row is not None

    async def get_artifact_content(self, artifact_id: str) -> bytes | None:
        """Fetch raw artifact content from S3 via the artifact's source_ref.

        Returns None if the artifact has no content or source_ref is empty.
        """
        artifact = await self.get_artifact(artifact_id)
        if not artifact or not artifact.source_ref:
            return None
        # source_ref contains the S3 presigned URL or local path
        # This method is intentionally minimal — content is stored in S3,
        # not in the artifact row.
        return None


# ── Global store singleton ────────────────────────────────────────────────────
# TODO: This is a temporary fix. Define proper store factory.

_pool: Any = None
_instance: KnowledgeStore | None = None


def get_store() -> KnowledgeStore:
    """Return the global KnowledgeStore singleton.

    Uses a simple pool-based singleton pattern. In production,
    this should be replaced with a proper dependency injection
    or factory pattern.
    """
    global _instance
    if _pool is None:
        raise RuntimeError(
            "Knowledge store pool not initialized. "
            "Call build_knowledge_pool(uri) first."
        )
    if _instance is None:
        _instance = KnowledgeStore(_pool)
    return _instance


def set_pool(pool: Any) -> None:
    """Install an externally-built pool as the store's global.

    Used by canonical_graph_view's store-worker when the pool must be
    owned by a dedicated background loop instead of the caller's loop
    (build_knowledge_pool always binds the caller's loop).
    """
    global _pool, _instance
    _pool = pool
    _instance = None


def build_knowledge_pool(uri: str, **kwargs: Any) -> Any:
    """Create the asyncpg pool for the knowledge store and set the global."""
    import asyncio

    import asyncpg  # type: ignore[import-untyped]

    from umrs.shared.pg_settings import asyncpg_pool_kwargs

    global _pool
    _pool = asyncio.get_event_loop().run_until_complete(
        asyncpg.create_pool(dsn=uri, **asyncpg_pool_kwargs("knowledge-store", **kwargs))
    )
    return _pool


def reset_store() -> None:
    """Reset the store singleton for testing (pool is preserved)."""
    global _instance
    _instance = None


def reset_knowledge_pool() -> None:
    """Close and drop the global knowledge pool (test isolation).

    The pool binds to the event loop that built it
    (``build_knowledge_pool``); pytest-asyncio spins one loop per test, so
    a surviving pool crashes the next test with cross-loop asyncpg errors
    ('attached to a different loop' / 'another operation is in progress' —
    CI self-hosted 2026-09-13). Teardown between tests rebuilds under the
    live loop.
    """
    global _pool, _instance
    _instance = None
    pool, _pool = _pool, None
    # Delegate worker-owned pools to the worker's own shutdown.
    try:
        from .canonical_graph_view import _reset_store_worker_for_tests

        _reset_store_worker_for_tests()
    except Exception:  # noqa: BLE001, S110 — teardown must not raise
        pass
    if pool is None:
        return
    try:
        import asyncio

        loop = asyncio.get_event_loop_policy().get_event_loop()
        if not loop.is_closed():
            loop.run_until_complete(pool.close())
    except Exception:  # noqa: BLE001, S110 — teardown must not raise
        pass
