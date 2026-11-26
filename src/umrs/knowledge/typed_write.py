"""Typed writes through the knowledge store (ADR-081 D-4/D-5/D-9, KN-S7).

The store's thin methods (``KnowledgeStore.upsert_node`` / ``upsert_edge``
for typed rows, ``KnowledgeStore.upsert_typed`` for a batch) call into this
module. A row is *typed* when the writer gives a class (``node_class``,
``attrs`` or ``natural_key`` on a node; ``relation_class`` on an edge) or
writes through ``upsert_typed``. Untyped legacy writes keep the exact
pre-W2a statements (they migrate writer by writer in later slices; the CI
insert fence keeps new raw INSERTs typed).

Order of one typed write: resolve the graph's :class:`TypingContext` (cached
per store), run the pure ``prepare_row`` / ``prepare_edge``, refuse the
whole write with :class:`OntologyViolationError` on an enforce-mode graph
with any violation (nothing written), else write the rows, count the gaps
and (batches) append the ``knowledge_write_batches`` row and bump
``graph_version`` — all in the caller's transaction.

When the registry is not seeded (a database before the seed migration) the
rows are written untyped; attrs then land in ``content["undeclared"]`` so
nothing a writer gave is lost.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from .edge_typing import prepare_edge
from .graph_registry import default_graph_id
from .node_typing import prepare_row
from .term_registry import SnapshotCache, record_batch, record_gaps
from .term_types import (
    ORIGIN_ACTORS,
    UNDECLARED_KEY,
    EdgeDraft,
    Gap,
    NodeDraft,
    OntologyViolationError,
    PreparedEdge,
    PreparedNode,
    TypingContext,
    Violation,
)

if TYPE_CHECKING:
    from .store import KnowledgeEdge, KnowledgeNode

__all__ = [
    "TYPED_COLUMNS_PROBE_SQL",
    "WriteBatch",
    "WriteBatchResult",
    "is_typed_edge",
    "is_typed_node",
]

#: Does migration 0145 (``knowledge_nodes.node_class``) exist?
TYPED_COLUMNS_PROBE_SQL = (
    "SELECT EXISTS (SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('knowledge_nodes') AND attname = 'node_class' "
    "AND NOT attisdropped)"
)

# D-6 (W2a part): a typed re-upsert merges attrs, keeps a stored class unless
# the new one is a real class (not the warn fallback), keeps the facet with
# the class, and never touches status (still INSERT-only, D-35).
_KEEP_CLASS = "EXCLUDED.node_class IS NULL OR EXCLUDED.node_class = 'r4:Unclassified'"
# ADR-093 D-6 (0120), the same rule as store._UPSERT_NODE_TEXT_SQL: a stored
# search_text survives a re-upsert without one only while the chunk it was
# verified against (chunk_md5, header, header_chars) is unchanged; a re-chunk
# clears it (review 2026-11-27: a coalesce kept the old words).
_KEEP_SEARCH_TEXT = " AND ".join(
    f"knowledge_nodes.content->'{key}' IS NOT DISTINCT FROM EXCLUDED.content->'{key}'"
    for key in ("chunk_md5", "header", "header_chars")
)
UPSERT_TYPED_NODE_SQL = f"""INSERT INTO knowledge_nodes
   (node_id, project_id, label, node_type, content, content_hash,
    source_ir, created_at, updated_at, graph_id, status, confidence,
    node_class, attrs, natural_key, search_text)
   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
   ON CONFLICT (node_id) DO UPDATE
   SET content = CASE WHEN jsonb_typeof(EXCLUDED.content) = 'object'
                       AND jsonb_typeof(knowledge_nodes.content) = 'object'
                      THEN knowledge_nodes.content || EXCLUDED.content
                      ELSE EXCLUDED.content END,
       content_hash = $6, source_ir = $7, updated_at = $9,
       node_class = CASE WHEN {_KEEP_CLASS}
                         THEN coalesce(knowledge_nodes.node_class, EXCLUDED.node_class)
                         ELSE EXCLUDED.node_class END,
       node_type = CASE WHEN ({_KEEP_CLASS}) AND knowledge_nodes.node_class IS NOT NULL
                        THEN knowledge_nodes.node_type ELSE EXCLUDED.node_type END,
       label = CASE WHEN EXCLUDED.label <> '' THEN EXCLUDED.label
                    ELSE knowledge_nodes.label END,
       attrs = knowledge_nodes.attrs || EXCLUDED.attrs,
       natural_key = coalesce(EXCLUDED.natural_key, knowledge_nodes.natural_key),
       search_text = CASE WHEN $16::text IS NOT NULL THEN $16::text
                          WHEN {_KEEP_SEARCH_TEXT} THEN knowledge_nodes.search_text END"""

UPSERT_TYPED_EDGE_SQL = """INSERT INTO knowledge_edges
   (edge_id, project_id, source_id, target_id, relation_type, content,
    graph_id, status, confidence, relation_class)
   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
   ON CONFLICT (edge_id) DO UPDATE
   SET content = $6, relation_class = EXCLUDED.relation_class"""

# Endpoint classes for the relation domain/range check: the caller's project
# only, so another project's row never informs (or leaks into) a decision.
NODE_CLASSES_SQL = (
    "SELECT node_id, node_class FROM knowledge_nodes "
    "WHERE node_id = ANY($1::text[]) AND project_id = $2"
)


@dataclass(frozen=True)
class WriteBatch:
    """One audited typed write into one graph (ADR-081 D-9).

    Every node and edge must name ``project_id`` and either ``graph_id`` or
    ``''`` (which means this batch's graph); anything else is refused with
    ValueError before a write. ``origin_actor`` is one of ``ORIGIN_ACTORS``:
    the actor that triggered the write (KN-1), not the writer module.
    """

    graph_id: str
    project_id: str
    writer: str
    origin_actor: str
    nodes: Sequence[KnowledgeNode] = ()
    edges: Sequence[KnowledgeEdge] = ()
    batch_id: str = ""


@dataclass(frozen=True)
class WriteBatchResult:
    """What a typed batch wrote and found."""

    batch_id: str
    ontology_ref: str
    mode: str
    counts: Mapping[str, int]
    violations: tuple[Violation, ...] = ()
    gaps: int = 0
    missing_required: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def is_typed_node(node: KnowledgeNode) -> bool:
    """The writer gave a class, attrs or a natural key."""
    return bool(node.node_class or node.attrs or node.natural_key)


def is_typed_edge(edge: KnowledgeEdge) -> bool:
    """The writer gave a relation class."""
    return bool(edge.relation_class)


def node_graph(node: KnowledgeNode) -> str:
    """The graph a node's typing resolves against ('' = project default)."""
    return node.graph_id or default_graph_id(node.project_id)


def _draft(node: KnowledgeNode) -> NodeDraft:
    return NodeDraft(
        node_id=node.node_id,
        node_type=node.node_type,
        node_class=node.node_class,
        attrs=dict(node.attrs),
        content=dict(node.content) if isinstance(node.content, Mapping) else {},
    )


def _edge_draft(edge: KnowledgeEdge) -> EdgeDraft:
    return EdgeDraft(
        edge_id=edge.edge_id,
        relation_type=edge.relation_type,
        relation_class=edge.relation_class,
        content=dict(edge.content) if isinstance(edge.content, Mapping) else {},
    )


def untyped_fallback(node: KnowledgeNode) -> KnowledgeNode:
    """The node for an unseeded registry: no class, attrs kept in content."""
    if not node.attrs or not isinstance(node.content, Mapping):
        return replace(node, node_class=None, attrs={}, natural_key=None)
    content = dict(node.content)
    held = content.get(UNDECLARED_KEY)
    content[UNDECLARED_KEY] = {
        **(held if isinstance(held, Mapping) else {}),
        **node.attrs,
    }
    return replace(node, node_class=None, attrs={}, natural_key=None, content=content)


def typed_node_params(node: KnowledgeNode, prepared: PreparedNode) -> list[Any]:
    """Parameters of :data:`UPSERT_TYPED_NODE_SQL`."""
    ts = datetime.now(timezone.utc)
    return [
        node.node_id,
        node.project_id,
        node.label,
        prepared.node_type,
        dict(prepared.content),
        node.content_hash,
        node.source_ir,
        node.created_at or ts,
        ts,
        node.graph_id,
        node.status,
        node.confidence,
        prepared.node_class,
        dict(prepared.attrs),
        node.natural_key,
        node.search_text,
    ]


def typed_edge_params(edge: KnowledgeEdge, prepared: PreparedEdge) -> list[Any]:
    """Parameters of :data:`UPSERT_TYPED_EDGE_SQL`."""
    return [
        edge.edge_id,
        edge.project_id,
        edge.source_id,
        edge.target_id,
        edge.relation_type,
        dict(prepared.content),
        edge.graph_id,
        edge.status,
        edge.confidence,
        prepared.relation_class,
    ]


def refuse_if_enforcing(ctx: TypingContext, violations: Iterable[Violation]) -> None:
    """Raise :class:`OntologyViolationError` for an enforce-mode graph."""
    found = tuple(violations)
    if ctx.enforcing and found:
        raise OntologyViolationError(found)


async def context_for(
    cache: SnapshotCache, db: Any, *, project_id: str, graph_id: str
) -> TypingContext | None:
    """The graph's typing context, or None when the registry is not seeded."""
    snapshot = await cache.fetch(db, project_id=project_id, graph_id=graph_id)
    return None if snapshot is None else TypingContext.from_snapshot(snapshot)


async def endpoint_classes(
    db: Any, project_id: str, node_ids: Iterable[str], known: Mapping[str, str]
) -> dict[str, str]:
    """``node_id -> node_class`` for edge endpoints: ``known`` first (rows of
    the same batch), then the project's stored rows."""
    wanted = sorted({n for n in node_ids if n not in known})
    found = dict(known)
    if wanted:
        for row in await db.fetch(NODE_CLASSES_SQL, wanted, project_id):
            if row["node_class"]:
                found[row["node_id"]] = row["node_class"]
    return found


async def write_typed_node(
    db: Any, cache: SnapshotCache, node: KnowledgeNode, *, writer: str
) -> PreparedNode | None:
    """One typed node (``upsert_node`` path). None = registry not seeded and
    the caller must write :func:`untyped_fallback` itself."""
    ctx = await context_for(
        cache, db, project_id=node.project_id, graph_id=node_graph(node)
    )
    if ctx is None:
        return None
    prepared = prepare_row(ctx, _draft(node))
    refuse_if_enforcing(ctx, prepared.violations)
    await db.execute(UPSERT_TYPED_NODE_SQL, *typed_node_params(node, prepared))
    await record_gaps(db, graph_id=node_graph(node), writer=writer, gaps=prepared.gaps)
    return prepared


async def write_typed_edge(
    db: Any, cache: SnapshotCache, edge: KnowledgeEdge, *, writer: str
) -> PreparedEdge | None:
    """One typed edge (``upsert_edge`` path); None as :func:`write_typed_node`."""
    graph_id = edge.graph_id or default_graph_id(edge.project_id)
    ctx = await context_for(cache, db, project_id=edge.project_id, graph_id=graph_id)
    if ctx is None:
        return None
    classes = await endpoint_classes(
        db, edge.project_id, (edge.source_id, edge.target_id), {}
    )
    prepared = prepare_edge(
        ctx,
        _edge_draft(edge),
        source_class=classes.get(edge.source_id),
        target_class=classes.get(edge.target_id),
    )
    refuse_if_enforcing(ctx, prepared.violations)
    await db.execute(UPSERT_TYPED_EDGE_SQL, *typed_edge_params(edge, prepared))
    await record_gaps(db, graph_id=graph_id, writer=writer, gaps=prepared.gaps)
    return prepared


def check_batch(batch: WriteBatch) -> None:
    """Refuse a malformed batch by name (ValueError) before any write."""
    if batch.origin_actor not in ORIGIN_ACTORS:
        raise ValueError(f"unknown origin_actor {batch.origin_actor!r}")
    if not batch.graph_id or not batch.project_id or not batch.writer:
        raise ValueError("a write batch names graph_id, project_id and writer")
    rows: list[Any] = [*batch.nodes, *batch.edges]
    for row in rows:
        if row.project_id != batch.project_id or row.graph_id not in (
            "",
            batch.graph_id,
        ):
            raise ValueError(
                f"row outside the batch graph: {getattr(row, 'node_id', '') or row.edge_id}"
            )


def _in_graph(row: Any, graph_id: str) -> Any:
    return row if row.graph_id else replace(row, graph_id=graph_id)


@dataclass
class _Prepared:
    nodes: list[tuple[KnowledgeNode, PreparedNode]] = field(default_factory=list)
    edges: list[tuple[KnowledgeEdge, PreparedEdge]] = field(default_factory=list)

    def _results(self) -> list[PreparedNode | PreparedEdge]:
        return [p for _, p in self.nodes] + [p for _, p in self.edges]

    def violations(self) -> tuple[Violation, ...]:
        return tuple(v for p in self._results() for v in p.violations)

    def gaps(self) -> tuple[Gap, ...]:
        return tuple(g for p in self._results() for g in p.gaps)


async def _prepare_batch(db: Any, ctx: TypingContext, batch: WriteBatch) -> _Prepared:
    out = _Prepared()
    for node in batch.nodes:
        row = _in_graph(node, batch.graph_id)
        out.nodes.append((row, prepare_row(ctx, _draft(row))))
    known = {row.node_id: p.node_class for row, p in out.nodes}
    ends = [n for e in batch.edges for n in (e.source_id, e.target_id)]
    classes = await endpoint_classes(db, batch.project_id, ends, known)
    for edge in batch.edges:
        row = _in_graph(edge, batch.graph_id)
        prepared = prepare_edge(
            ctx,
            _edge_draft(row),
            source_class=classes.get(row.source_id),
            target_class=classes.get(row.target_id),
        )
        out.edges.append((row, prepared))
    return out


async def write_batch(
    db: Any, cache: SnapshotCache, batch: WriteBatch
) -> WriteBatchResult | None:
    """The typed batch (module docstring) in the caller's transaction. None
    = registry not seeded; the caller writes the rows untyped."""
    check_batch(batch)
    ctx = await context_for(
        cache, db, project_id=batch.project_id, graph_id=batch.graph_id
    )
    if ctx is None:
        return None
    prepared = await _prepare_batch(db, ctx, batch)
    refuse_if_enforcing(ctx, prepared.violations())
    for node, p in prepared.nodes:
        await db.execute(UPSERT_TYPED_NODE_SQL, *typed_node_params(node, p))
    for edge, pe in prepared.edges:
        await db.execute(UPSERT_TYPED_EDGE_SQL, *typed_edge_params(edge, pe))
    gaps = await record_gaps(
        db, graph_id=batch.graph_id, writer=batch.writer, gaps=prepared.gaps()
    )
    result = WriteBatchResult(
        batch_id=batch.batch_id or f"kwb-{uuid.uuid4().hex}",
        ontology_ref=ctx.ontology_ref,
        mode=ctx.mode,
        counts={"nodes": len(prepared.nodes), "edges": len(prepared.edges)},
        violations=prepared.violations(),
        gaps=gaps,
        missing_required={
            n.node_id: p.missing_required
            for n, p in prepared.nodes
            if p.missing_required
        },
    )
    await record_batch(
        db,
        batch_id=result.batch_id,
        graph_id=batch.graph_id,
        project_id=batch.project_id,
        writer=batch.writer,
        origin_actor=batch.origin_actor,
        ontology_ref=result.ontology_ref,
        mode=result.mode,
        counts=result.counts,
        violations=result.violations,
    )
    return result
