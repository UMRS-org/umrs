"""Postgres-backed knowledge graph (durable K1 implementation).

Implements :class:`KnowledgeGraphAPI` against the ``sat_kg_nodes`` /
``sat_kg_edges`` tables (migration 0060). The in-memory graph remains the
reference implementation and test fake.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from contextlib import contextmanager
from typing import Any

from . import KnowledgeEdge, KnowledgeNode, RelationType

logger = logging.getLogger(__name__)

_NODE_COLUMNS = "node_id, node_type, title, metadata"
_EDGE_COLUMNS = (
    "edge_id, source_id, target_id, relation_type, weight, confidence, "
    "metadata, created_at"
)


def _row_to_node(row: dict[str, Any]) -> KnowledgeNode:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    return KnowledgeNode.from_dict(data)


def _row_to_edge(row: dict[str, Any]) -> KnowledgeEdge:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    return KnowledgeEdge.from_dict(data)


class PgKnowledgeGraph:
    """Durable typed-relationship graph backed by Postgres."""

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self):
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    # ── writes ───────────────────────────────────────────────────────────

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Upsert a node: title only replaced when non-empty, metadata merged."""
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sat_kg_nodes (node_id, node_type, title, metadata) "
                "VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT (node_id) DO UPDATE SET "
                "title = CASE WHEN excluded.title != '' "
                "THEN excluded.title ELSE sat_kg_nodes.title END, "
                "metadata = sat_kg_nodes.metadata || excluded.metadata",
                (node_id, node_type, title, json.dumps(metadata or {})),
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
        """Add (or update a matching) edge; returns the edge_id.

        One logical edge per (source, target, relation_type) triple — a
        second add updates weight/confidence and merges metadata, matching
        the in-memory dedup semantics. Missing endpoint nodes are created
        with node_type ``unknown``.
        """
        with self._cursor() as cur:
            self._ensure_node(cur, source_id)
            self._ensure_node(cur, target_id)
            cur.execute(
                "SELECT edge_id FROM sat_kg_edges "
                "WHERE source_id = %s AND target_id = %s AND relation_type = %s",
                (source_id, target_id, relation_type.value),
            )
            row = cur.fetchone()
            if row:
                edge_id = str(row["edge_id"])
                cur.execute(
                    "UPDATE sat_kg_edges SET weight = %s, confidence = %s, "
                    "metadata = sat_kg_edges.metadata || %s::jsonb "
                    "WHERE edge_id = %s",
                    (weight, confidence, json.dumps(metadata or {}), edge_id),
                )
                return edge_id
            edge = KnowledgeEdge(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
                metadata=metadata or {},
            )
            cur.execute(
                f"INSERT INTO sat_kg_edges ({_EDGE_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation_type.value,
                    edge.weight,
                    edge.confidence,
                    json.dumps(edge.metadata),
                    edge.created_at,
                ),
            )
            return edge.edge_id

    @staticmethod
    def _ensure_node(cur, node_id: str) -> None:
        cur.execute(
            "INSERT INTO sat_kg_nodes (node_id, node_type) VALUES (%s, 'unknown') "
            "ON CONFLICT (node_id) DO NOTHING",
            (node_id,),
        )

    # ── reads ────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_NODE_COLUMNS} FROM sat_kg_nodes WHERE node_id = %s",
                (node_id,),
            )
            row = cur.fetchone()
        return _row_to_node(row) if row else None

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        """All outgoing edges from a node."""
        return self._edges_from("source_id", node_id)

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        """All incoming edges to a node."""
        return self._edges_from("target_id", node_id)

    def _edges_from(self, column: str, node_id: str) -> list[KnowledgeEdge]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_EDGE_COLUMNS} FROM sat_kg_edges WHERE {column} = %s "
                "ORDER BY created_at ASC",
                (node_id,),
            )
            return [_row_to_edge(row) for row in cur.fetchall()]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Neighbor node IDs in both directions."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT target_id AS n FROM sat_kg_edges WHERE source_id = %s "
                "UNION "
                "SELECT source_id AS n FROM sat_kg_edges WHERE target_id = %s",
                (node_id, node_id),
            )
            return [str(row["n"]) for row in cur.fetchall()]

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        """Node IDs related with the given relation type (both directions)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT target_id AS n FROM sat_kg_edges "
                "WHERE source_id = %s AND relation_type = %s "
                "UNION "
                "SELECT source_id AS n FROM sat_kg_edges "
                "WHERE target_id = %s AND relation_type = %s",
                (node_id, relation_type.value, node_id, relation_type.value),
            )
            return [str(row["n"]) for row in cur.fetchall()]

    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 5
    ) -> list[str] | None:
        """BFS shortest path between two nodes, or None when unreachable.

        Per-level adjacency is loaded from Postgres (bounded by max_depth).
        """
        if source_id == target_id:
            return [source_id]
        if self.get_node(source_id) is None or self.get_node(target_id) is None:
            return None
        visited = {source_id}
        queue: deque[tuple[str, list[str]]] = deque([(source_id, [source_id])])
        while queue:
            current, path = queue.popleft()
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
        """Query nodes by type, by edge relation type, or list all."""
        with self._cursor() as cur:
            if node_type is not None:
                cur.execute(
                    f"SELECT {_NODE_COLUMNS} FROM sat_kg_nodes WHERE node_type = %s",
                    (node_type,),
                )
            elif relation_type is not None:
                cur.execute(
                    f"SELECT {_NODE_COLUMNS} FROM sat_kg_nodes "
                    "WHERE node_id IN ("
                    "SELECT source_id FROM sat_kg_edges WHERE relation_type = %s "
                    "UNION "
                    "SELECT target_id FROM sat_kg_edges WHERE relation_type = %s"
                    ")",
                    (relation_type.value, relation_type.value),
                )
            else:
                cur.execute(f"SELECT {_NODE_COLUMNS} FROM sat_kg_nodes")
            return [_row_to_node(row) for row in cur.fetchall()]
