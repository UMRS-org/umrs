"""CanonicalGraphView — 9-method graph protocol over the canonical store.

Migration surface for ADR-026: exposes the same ``KnowledgeGraphAPI``
surface the deprecated ``sat_kg_api.InMemoryKnowledgeGraph`` has, but every
read is sourced from canonical ``knowledge_nodes/edges`` rows (0055) of one
(project_id, graph_id) and every write goes to the canonical
``KnowledgeStore`` (upsert_node/upsert_edge) — no sat tables involved.

Semantics mirrored exactly from the in-memory reference implementation:
- edges dedupe on (source, target, relation) with in-place update
- get_related is direction-agnostic (outgoing targets + incoming sources)
- get_neighbors is direction-agnostic
- find_path is undirected BFS, depth-bounded
- add_edge auto-creates unknown endpoint nodes as node_type="unknown"

Write durability: writes schedule ``KnowledgeStore.upsert_*`` against the
shared asyncpg pool. When a running event loop exists the write is
scheduled as a task (fire-and-forget, matching the historic sat in-memory
timing contract); otherwise it blocks via ``asyncio.run`` (engine/job
contexts, deterministic for tests).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from umrs.knowledge.graph_types import (
    KnowledgeEdge,
    KnowledgeNode,
    RelationType,
)

from .ontology import canonical_relation, canonical_to_sat
from .store import KnowledgeEdge as CanonicalEdge
from .store import KnowledgeNode as CanonicalNode
from .store import KnowledgeStore

logger = logging.getLogger(__name__)


def _run_mutation(coro: Any) -> None:
    """Schedule the store write; block when no event loop is running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        asyncio.run(coro)
    else:
        task = loop.create_task(coro)
        # Preserve visibility for debugging (errors surface via callback).
        task.add_done_callback(
            lambda t: (
                logger.warning(
                    "canonical graph mutation failed: %s", t.exception(), exc_info=True
                )
                if t.exception()
                else None
            )
        )


def _resolve_pool(request: Any | None) -> Any | None:
    """Resolve the shared api-side asyncpg pool, if caller passed a request."""
    if request is None:
        return None
    app_state = getattr(getattr(request, "app", None), "state", None)
    return getattr(app_state, "umrs_knowledge_asyncpg_pool", None)


def _maybe_knowledge_store() -> KnowledgeStore | None:
    """Resolve a durable KnowledgeStore when the knowledge DSN is configured.

    Reuses an already-initialized singleton when present (api contexts);
    otherwise builds one synchronously (engine/job contexts — must not be
    called inside a running event loop).
    """
    try:
        from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

        dsn = _resolve_knowledge_dsn()
        if not dsn:
            return None
        from .store import build_knowledge_pool, get_store  # type: ignore[attr-defined]

        try:
            return get_store()
        except RuntimeError:
            build_knowledge_pool(dsn)
            return get_store()
    except Exception:
        logger.debug("canonical store unavailable; mirror-only view", exc_info=True)
        return None


def default_view(
    project_id: str = "default",
    graph_id: str | None = None,
    *,
    store: KnowledgeStore | None = None,
) -> "CanonicalGraphView":
    """View over the project's default graph; store attached when resolvable."""
    from .graph_registry import default_graph_id

    gid = graph_id or default_graph_id(project_id)
    resolved_store = store if store is not None else _maybe_knowledge_store()
    return CanonicalGraphView(
        store=resolved_store,
        project_id=project_id,
        graph_id=gid,
    )


class CanonicalGraphView:
    """Canonical-store-backed implementation of the graph API protocol."""

    def __init__(
        self,
        *,
        store: KnowledgeStore | None,
        project_id: str,
        graph_id: str,
    ) -> None:
        self._store = store
        self._project_id = project_id
        self._graph_id = graph_id
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, KnowledgeEdge] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def graph_id(self) -> str:
        return self._graph_id

    # ── loading ─────────────────────────────────────────────────────────

    async def refresh(self) -> "CanonicalGraphView":
        """Load canonical rows into the in-memory mirror."""
        if self._store is None:
            return self
        nodes = await self._store.list_nodes(self._project_id, self._graph_id)
        edges = await self._store.list_edges(self._project_id, self._graph_id)
        for row in nodes:
            self._nodes[row.node_id] = _view_node(row)
        for row in edges:
            self._track_edge(_view_edge(row))
        return self

    @classmethod
    def in_project(
        cls,
        store: KnowledgeStore | None,
        project_id: str,
        graph_id: str,
    ) -> "CanonicalGraphView":
        return cls(store=store, project_id=project_id, graph_id=graph_id)

    # ── mutations ───────────────────────────────────────────────────────

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> None:
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.title = title or node.title
            node.metadata.update(metadata or {})
        else:
            node = KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                title=title,
                metadata=metadata or {},
            )
            self._nodes[node_id] = node
            self._outgoing.setdefault(node_id, [])
            self._incoming.setdefault(node_id, [])
        if self._store is not None:
            _run_mutation(
                self._store.upsert_node(
                    CanonicalNode(
                        node_id=node_id,
                        project_id=self._project_id,
                        label=title or node_id,
                        node_type=_canonical_node_type(node_type),
                        content=dict(metadata or {}),
                        graph_id=self._graph_id,
                    )
                )
            )

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> str:
        if source_id not in self._nodes:
            self.add_node(node_id=source_id, node_type="unknown")
        if target_id not in self._nodes:
            self.add_node(node_id=target_id, node_type="unknown")

        # Dedupe on (source, target, relation) like the sat reference impl.
        for edge_id in self._outgoing.get(source_id, []):
            edge = self._edges[edge_id]
            if edge.target_id == target_id and edge.relation_type == relation_type:
                edge.weight = weight
                edge.confidence = confidence
                edge.metadata.update(metadata or {})
                self._durable_write_edge(edge, relation_type)
                return edge.edge_id

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._track_edge(edge)
        self._durable_write_edge(edge, relation_type)
        return edge.edge_id

    def _track_edge(self, edge: KnowledgeEdge) -> None:
        self._edges[edge.edge_id] = edge
        self._outgoing.setdefault(edge.source_id, []).append(edge.edge_id)
        self._incoming.setdefault(edge.target_id, []).append(edge.edge_id)

    def _durable_write_edge(
        self, edge: KnowledgeEdge, relation_type: RelationType
    ) -> None:
        if self._store is None:
            return
        canonical_rel, inverse = canonical_relation(relation_type.value.upper())
        src_id, dst_id = edge.source_id, edge.target_id
        if inverse:
            src_id, dst_id = dst_id, src_id
        content = {
            "weight": edge.weight,
            "confidence": edge.confidence,
            "origin": "sat",
            "sat_relation": relation_type.value.upper(),
            "sat_inverse": bool(inverse),
            **(edge.metadata or {}),
        }
        _run_mutation(
            self._store.upsert_edge(
                CanonicalEdge(
                    edge_id=edge.edge_id,
                    project_id=self._project_id,
                    source_id=src_id,
                    target_id=dst_id,
                    relation_type=canonical_rel,
                    content=content,
                    graph_id=self._graph_id,
                )
            )
        )

    # ── reads (identical semantics to the sat reference implementation) ──

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        return [
            self._edges[eid]
            for eid in self._outgoing.get(node_id, [])
            if eid in self._edges
        ]

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        return [
            self._edges[eid]
            for eid in self._incoming.get(node_id, [])
            if eid in self._edges
        ]

    def get_neighbors(self, node_id: str) -> list[str]:
        neighbors: set[str] = set()
        for edge in self.get_outgoing(node_id):
            neighbors.add(edge.target_id)
        for edge in self.get_incoming(node_id):
            neighbors.add(edge.source_id)
        return list(neighbors)

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        related: list[str] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.target_id)
        for edge in self.get_incoming(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.source_id)
        return related

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        if source_id == target_id:
            return [source_id]
        visited: set[str] = {source_id}
        queue: list[tuple[str, list[str]]] = [(source_id, [source_id])]
        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor in self.get_neighbors(current):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def query(
        self,
        *,
        node_type: str | None = None,
        relation_type: RelationType | None = None,
    ) -> list[KnowledgeNode]:
        if node_type is not None:
            return [n for n in self._nodes.values() if n.node_type == node_type]
        if relation_type is not None:
            matching: set[str] = set()
            for edge in self._edges.values():
                if edge.relation_type == relation_type:
                    matching.add(edge.source_id)
                    matching.add(edge.target_id)
            return [self._nodes[nid] for nid in matching if nid in self._nodes]
        return list(self._nodes.values())


# ── row → view adapters ──────────────────────────────────────────────────


def _canonical_node_type(node_type: str) -> str:
    """Canonical node_type column check (DB CHECK constraint buckets)."""
    if node_type in KnowledgeStore.ALLOWED_NODE_TYPES:
        return node_type
    return "entity"


def _view_node(row: CanonicalNode) -> KnowledgeNode:
    content = row.content or {}
    return KnowledgeNode(
        node_id=row.node_id,
        node_type=row.node_type,
        title=str(row.label or row.node_id),
        metadata=dict(content),
    )


_PROVENANCE_KEYS = {"weight", "confidence", "origin", "sat_relation", "sat_inverse"}


def _view_edge(row: CanonicalEdge) -> KnowledgeEdge:
    content = row.content or {}
    sat_name = content.get("sat_relation")
    if sat_name:
        sat_name = str(sat_name).lower()
    else:
        sat_name = canonical_to_sat(row.relation_type, None).lower()
    try:
        rel = RelationType(sat_name)
    except ValueError:
        rel = RelationType.RELATED_TO
    src_id, dst_id = row.source_id, row.target_id
    if content.get("sat_inverse"):
        src_id, dst_id = dst_id, src_id
    weight = float(content.get("weight", 1.0))
    confidence = float(content.get("confidence", 1.0))
    metadata = {k: v for k, v in content.items() if k not in _PROVENANCE_KEYS}
    return KnowledgeEdge(
        edge_id=row.edge_id,
        source_id=src_id,
        target_id=dst_id,
        relation_type=rel,
        weight=weight,
        confidence=confidence,
        metadata=metadata,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )
