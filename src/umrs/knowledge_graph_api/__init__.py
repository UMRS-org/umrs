"""Knowledge graph API — typed relationship graph between scientific objects.

Phase 1, Step 2.3 of the RC1 migration plan.

UMRS owns the knowledge graph. R4 consumes this API.

This module provides a typed relationship graph for scientific knowledge
objects. The graph supports:

  - Typed edges (CITES, IMPLEMENTS, USES, REFERENCES, SUPPORTS, CONTRADICTS,
    DERIVES_FROM, PRODUCES, ANSWERS, VERIFIES, REFUTES, ASSUMES, RELATED_TO)
  - Graph traversal (BFS, shortest path)
  - Neighbor discovery
  - Relationship queries
  - Edge deduplication
  - Edge confidence scores

Note: This is a new API-level module (`knowledge_graph_api`) that provides
the public interface for the knowledge graph. The existing `knowledge_graph/`
module contains the skeletal implementation.

DEPRECATED (ADR-026, 2026-08-15): the ``sat_kg_*`` tables and this API face
are retained for read-compatibility and the latent team-engine consumer only.
New knowledge-graph work uses the canonical store (``umrs.knowledge`` +
knowledge graph registry, migration 0069). Do not extend this module.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Any

logger = logging.getLogger(__name__)

# ADR-026 import-time deprecation hook: loud in production (FutureWarning
# displays by default), silent under pytest and when explicitly quiesced.
if os.environ.get("UMRS_SAT_KG_QUIET") != "1" and "pytest" not in sys.modules:
    warnings.warn(
        "umrs.knowledge_graph_api (sat_kg_* tables) is DEPRECATED per ADR-026. "
        "New knowledge-graph work must use the canonical store "
        "(umrs.knowledge.store + graph registry, migration 0069). "
        "Set UMRS_SAT_KG_QUIET=1 to silence.",
        FutureWarning,
        stacklevel=2,
    )

__all__ = [
    "InMemoryKnowledgeGraph",
    "KnowledgeEdge",
    "KnowledgeGraphAPI",
    "KnowledgeNode",
    "RelationType",
    "build_default_graph",
]


# DTO home (ADR-026 wave B): the shared graph type vocabulary lives in
# ``umrs.knowledge.graph_types`` — this facade re-exports them until wave C.
from umrs.knowledge.graph_types import (
    KnowledgeEdge,
    KnowledgeNode,
    RelationType,
)


class KnowledgeGraphAPI:
    """Protocol for the knowledge graph API.

    UMRS provides an implementation of this API. R4 consumes it.
    """

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        raise NotImplementedError

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        raise NotImplementedError

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        raise NotImplementedError

    def get_neighbors(self, node_id: str) -> list[str]:
        raise NotImplementedError

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        raise NotImplementedError

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        raise NotImplementedError

    def query(
        self, *, node_type: str | None = None, relation_type: RelationType | None = None
    ) -> list[KnowledgeNode]:
        raise NotImplementedError


# ── In-Memory Implementation ───────────────────────────────────────────────


class InMemoryKnowledgeGraph(KnowledgeGraphAPI):
    """In-memory implementation of the knowledge graph API.

    This is the reference implementation. The Postgres-backed implementation
    will follow the same interface.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, KnowledgeEdge] = {}
        self._outgoing: dict[str, list[str]] = {}  # node_id -> [edge_id, ...]
        self._incoming: dict[str, list[str]] = {}  # node_id -> [edge_id, ...]

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Add a node to the graph."""
        if node_id in self._nodes:
            # Update existing node
            node = self._nodes[node_id]
            node.title = title or node.title
            node.metadata.update(metadata or {})
        else:
            self._nodes[node_id] = KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                title=title,
                metadata=metadata or {},
            )
            self._outgoing[node_id] = []
            self._incoming[node_id] = []

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
        """Add an edge to the graph.

        Returns the edge_id. If an edge with the same source, target, and
        relation_type already exists, it is updated.
        """
        # Ensure both nodes exist
        if source_id not in self._nodes:
            self.add_node(node_id=source_id, node_type="unknown")
        if target_id not in self._nodes:
            self.add_node(node_id=target_id, node_type="unknown")

        # Check for existing edge (deduplication)
        for edge_id in self._outgoing.get(source_id, []):
            edge = self._edges[edge_id]
            if edge.target_id == target_id and edge.relation_type == relation_type:
                # Update existing edge
                edge.weight = weight
                edge.confidence = confidence
                edge.metadata.update(metadata or {})
                return edge.edge_id

        # Create new edge
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._edges[edge.edge_id] = edge
        self._outgoing.setdefault(source_id, []).append(edge.edge_id)
        self._incoming.setdefault(target_id, []).append(edge.edge_id)
        return edge.edge_id

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all outgoing edges from a node."""
        edge_ids = self._outgoing.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all incoming edges to a node."""
        edge_ids = self._incoming.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbor node IDs (both outgoing and incoming)."""
        neighbors: set[str] = set()
        for edge in self.get_outgoing(node_id):
            neighbors.add(edge.target_id)
        for edge in self.get_incoming(node_id):
            neighbors.add(edge.source_id)
        return list(neighbors)

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        """Get all node IDs related to the given node with the given relation type."""
        related: list[str] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.target_id)
        for edge in self.get_incoming(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.source_id)
        return related

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find a path between two nodes using BFS.

        Returns a list of node IDs forming a path, or None if no path found.
        """
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
        """Query the graph for nodes matching the given criteria."""
        if node_type is not None:
            return [
                node for node in self._nodes.values() if node.node_type == node_type
            ]
        if relation_type is not None:
            # Return all nodes that have an edge of the given type
            matching_node_ids: set[str] = set()
            for edge in self._edges.values():
                if edge.relation_type == relation_type:
                    matching_node_ids.add(edge.source_id)
                    matching_node_ids.add(edge.target_id)
            return [self._nodes[nid] for nid in matching_node_ids if nid in self._nodes]
        return list(self._nodes.values())


# ── Default Graph Factory ───────────────────────────────────────────────


def build_default_graph(db_pool: Any = None) -> Any:
    """Build the default knowledge graph backend.

    Selection via ``UMRS_KNOWLEDGE_GRAPH_STORE_BACKEND`` (default ``auto``):

      - ``pg``     — force Postgres (raises if unavailable)
      - ``memory`` — force the in-memory reference graph
      - ``auto``   — Postgres when a pool is provided or the environment
                     points at a database, otherwise in-memory

    Postgres uses ``sat_kg_nodes`` / ``sat_kg_edges`` (migration 0060).
    """
    import os as _os

    from umrs.shared.pg import resolve_satellite_pool

    backend = _os.environ.get("UMRS_KNOWLEDGE_GRAPH_STORE_BACKEND", "auto").lower()
    if backend == "memory":
        return InMemoryKnowledgeGraph()
    if backend not in ("auto", "pg"):
        logger.warning("Unknown store backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Knowledge graph")
    if pool is None:
        return InMemoryKnowledgeGraph()

    from .pg_store import PgKnowledgeGraph

    return PgKnowledgeGraph(pool)
