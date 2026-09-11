"""Knowledge graph registry + graph-extension proposals (sync, psycopg2).

Phase 1/3 of the multi-graph capability
(docs/history/knowledge-graph-convergence-2026-08-14.md).

This module holds the *registry* (first-class graph identity) and the
*proposal* gate (staged, approval-required graph extensions). Storage
diversity note: ``KnowledgeStore`` (asyncpg) remains the write/derived-graph
authority for ``knowledge_nodes``/``knowledge_edges``; this module is sync
because the durable-jobs pool (psycopg2 ``ThreadedConnectionPool``) is the
sync resource available in API workers and matches the ``search_api`` sync
backend port used by the K3 facade.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server``.
Only stdlib + logging imports (mirrored on the canonical store pool rule).
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

GRAPH_STATUSES: frozenset[str] = frozenset({"active", "archived"})
PROPOSAL_STATUSES: frozenset[str] = frozenset(
    {"pending", "approved", "declined", "executed"}
)


def default_graph_id(project_id: str) -> str:
    """Deterministic default-graph id for a project.

    Pure function — no registry row lookup needed for the write path.
    Existing projects automatically project to this graph (backward compat).
    """
    return "g-" + hashlib.sha256(f"project:{project_id}".encode()).hexdigest()[:16]


def graph_id_is_default(graph_id: str, project_id: Optional[str]) -> bool:
    """True when ``graph_id`` is the deterministic default for ``project_id``."""
    if not project_id:
        return False
    return graph_id == default_graph_id(project_id)


def _iso(value: Any) -> Optional[str]:
    """Serialize a datetime-ish column value; None/str pass through safely."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _row_to_graph(row: dict[str, Any]) -> "KnowledgeGraphMeta":
    data = dict(row)
    for key in ("provenance",):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return KnowledgeGraphMeta(**data)


def _row_to_proposal(row: dict[str, Any]) -> "GraphExtensionProposal":
    data = dict(row)
    for key in ("missing_information", "source_artifacts", "proposed_operations", "execution"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return GraphExtensionProposal(**data)


@dataclass(frozen=True)
class KnowledgeGraphMeta:
    """Row from the knowledge_graphs registry table."""

    graph_id: str
    name: str
    description: str = ""
    owner_id: str = ""
    project_id: Optional[str] = None
    kind: str = "project"
    status: str = "active"
    ontology_id: Optional[str] = None
    ontology_version: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "kind": self.kind,
            "status": self.status,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "provenance": self.provenance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass(frozen=True)
class GraphExtensionProposal:
    """Row from knowledge_graph_proposals — staged mutation request."""

    proposal_id: str
    target_graph_id: str
    reason: str = ""
    missing_information: list[Any] = field(default_factory=list)
    source_artifacts: list[Any] = field(default_factory=list)
    proposed_operations: list[Any] = field(default_factory=list)
    created_by: str = ""
    session_id: str = ""
    status: str = "pending"
    decided_by: str = ""
    decided_at: Optional[datetime] = None
    execution: Optional[dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "target_graph_id": self.target_graph_id,
            "reason": self.reason,
            "missing_information": self.missing_information,
            "source_artifacts": self.source_artifacts,
            "proposed_operations": self.proposed_operations,
            "created_by": self.created_by,
            "session_id": self.session_id,
            "status": self.status,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "execution": self.execution,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class _SyncPoolStore:
    """Shared psycopg2 pool plumbing (canonical store pool, mirrors the registry's contract)."""

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)


_GRAPH_COLUMNS = (
    "graph_id, name, description, owner_id, project_id, kind, status, "
    "ontology_id, ontology_version, provenance, created_at, updated_at"
)


class GraphRegistryStore(_SyncPoolStore):
    """Registry of first-class knowledge graphs (migration 0069)."""

    def ensure_default_graph(self, project_id: str, owner_id: str = "") -> KnowledgeGraphMeta:
        """Idempotently materialize the default project graph row."""
        gid = default_graph_id(project_id)
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge_graphs "
                "(graph_id, name, description, owner_id, project_id, kind) "
                "VALUES (%s, %s, %s, %s, %s, 'project') "
                "ON CONFLICT (graph_id) DO NOTHING",
                (gid, project_id, f"Default knowledge graph for project {project_id}",
                 owner_id, project_id),
            )
        graph = self.get_graph(gid)
        assert graph is not None  # insert-on-conflict guarantees presence
        return graph

    def create_graph(
        self,
        name: str,
        *,
        description: str = "",
        owner_id: str = "",
        project_id: Optional[str] = None,
        kind: str = "research",
        ontology_id: Optional[str] = None,
        ontology_version: Optional[str] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> KnowledgeGraphMeta:
        """Create a named, independent knowledge graph."""
        graph_id = "g-" + uuid.uuid4().hex[:16]
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge_graphs "
                "(graph_id, name, description, owner_id, project_id, kind, "
                " ontology_id, ontology_version, provenance) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                (
                    graph_id, name, description, owner_id, project_id, kind,
                    ontology_id, ontology_version, json.dumps(provenance or {}),
                ),
            )
        graph = self.get_graph(graph_id)
        assert graph is not None
        return graph

    def get_graph(self, graph_id: str) -> Optional[KnowledgeGraphMeta]:
        """Fetch one graph by id; None when absent."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_GRAPH_COLUMNS} FROM knowledge_graphs WHERE graph_id = %s",
                (graph_id,),
            )
            row = cur.fetchone()
        return _row_to_graph(row) if row else None

    def resolve_graph(self, identifier: str) -> Optional[KnowledgeGraphMeta]:
        """Resolve by exact id, else exact case-insensitive name (active first)."""
        graph = self.get_graph(identifier)
        if graph is not None:
            return graph
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_GRAPH_COLUMNS} FROM knowledge_graphs "
                "WHERE lower(name) = lower(%s) "
                "ORDER BY CASE WHEN status = 'active' THEN 0 ELSE 1 END, created_at "
                "LIMIT 1",
                (identifier,),
            )
            row = cur.fetchone()
        return _row_to_graph(row) if row else None

    def graph_content(
        self,
        graph_id: str,
        *,
        node_limit: int = 200,
        edge_limit: int = 400,
    ) -> dict[str, Any]:
        """Bounded node+edge read for one registered graph (visualization).

        Legacy fallback: rows with ``graph_id = ''`` count as members of the
        owning project's default graph (same rule KnowledgeStore.list_nodes
        applies), so default graphs of pre-0069 projects still render.
        """
        meta = self.resolve_graph(graph_id)
        if meta is None:
            raise KeyError(f"Graph {graph_id} not found")
        legacy_default = bool(
            meta.project_id and graph_id_is_default(meta.graph_id, meta.project_id)
        )
        node_rows = self._content_nodes(graph_id, legacy_default, meta.project_id, node_limit)
        edge_rows = self._content_edges(graph_id, legacy_default, meta.project_id, edge_limit)
        return {
            "graph_id": meta.graph_id,
            "name": meta.name,
            "kind": meta.kind,
            "project_id": meta.project_id,
            "owner_id": meta.owner_id,
            "nodes": [_content_node_dict(row) for row in node_rows],
            "edges": [_content_edge_dict(row) for row in edge_rows],
            "truncated": len(node_rows) == node_limit,
        }

    def _content_nodes(
        self, graph_id: str, legacy_default: bool, project_id: Optional[str], limit: int
    ) -> list[Any]:
        """Graph-scope (+ legacy default-fallback) node rows for graph_content."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT node_id, project_id, label, node_type, created_at, graph_id "
                "FROM knowledge_nodes "
                "WHERE graph_id = %s OR (%s AND graph_id = '' AND project_id = %s) "
                "ORDER BY created_at DESC LIMIT %s",
                (graph_id, legacy_default, project_id, limit),
            )
            return cur.fetchall()

    def _content_edges(
        self, graph_id: str, legacy_default: bool, project_id: Optional[str], limit: int
    ) -> list[Any]:
        """Graph-scope (+ legacy default-fallback) edge rows for graph_content."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT edge_id, source_id, target_id, relation_type, created_at, graph_id "
                "FROM knowledge_edges "
                "WHERE graph_id = %s OR (%s AND graph_id = '' AND project_id = %s) "
                "ORDER BY created_at DESC LIMIT %s",
                (graph_id, legacy_default, project_id, limit),
            )
            return cur.fetchall()

    def list_graphs(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> list[KnowledgeGraphMeta]:
        """List graphs with optional project/status filters."""
        conditions, params = ["1=1"], []
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if status:
            conditions.append("status = %s")
            params.append(status)
        where = " AND ".join(conditions)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_GRAPH_COLUMNS} FROM knowledge_graphs "
                f"WHERE {where} ORDER BY created_at LIMIT %s",
                (*params, limit),
            )
            rows = cur.fetchall()
        return [_row_to_graph(row) for row in rows]

    def search_graphs(
        self,
        query: str,
        *,
        project_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[KnowledgeGraphMeta]:
        """Active graphs whose name/description match the query text."""
        pattern = f"%{query}%"
        base = (
            f"SELECT {_GRAPH_COLUMNS} FROM knowledge_graphs "
            "WHERE status = 'active' "
            "AND (name ILIKE %s OR description ILIKE %s OR graph_id ILIKE %s)"
        )
        with self._cursor() as cur:
            if project_id:
                cur.execute(
                    base + " AND project_id = %s ORDER BY created_at LIMIT %s",
                    (pattern, pattern, pattern, project_id, limit),
                )
            else:
                cur.execute(
                    base + " ORDER BY created_at LIMIT %s",
                    (pattern, pattern, pattern, limit),
                )
            rows = cur.fetchall()
        return [_row_to_graph(row) for row in rows]

    def update_graph(
        self,
        graph_id: str,
        *,
        description: Optional[str] = None,
        kind: Optional[str] = None,
        ontology_id: Optional[str] = None,
        ontology_version: Optional[str] = None,
    ) -> Optional[KnowledgeGraphMeta]:
        """Patch mutable metadata; returns the updated row (None if absent)."""
        sets, params = [], []
        for column, value in (
            ("description", description),
            ("kind", kind),
            ("ontology_id", ontology_id),
            ("ontology_version", ontology_version),
        ):
            if value is not None:
                sets.append(f"{column} = %s")
                params.append(value)
        if sets:
            with self._cursor() as cur:
                cur.execute(
                    f"UPDATE knowledge_graphs SET {', '.join(sets)}, "
                    "updated_at = now() WHERE graph_id = %s",
                    (*params, graph_id),
                )
        return self.get_graph(graph_id)

    def set_status(self, graph_id: str, status: str) -> Optional[KnowledgeGraphMeta]:
        """Lifecycle transition (active|archived). Hard delete intentionally absent."""
        if status not in GRAPH_STATUSES:
            raise ValueError(f"invalid graph status {status!r}")
        with self._cursor() as cur:
            cur.execute(
                "UPDATE knowledge_graphs SET status = %s, updated_at = now() "
                "WHERE graph_id = %s",
                (status, graph_id),
            )
        return self.get_graph(graph_id)


_PROPOSAL_COLUMNS = (
    "proposal_id, target_graph_id, reason, missing_information, source_artifacts, "
    "proposed_operations, created_by, session_id, status, decided_by, decided_at, "
    "execution, created_at, updated_at"
)


class ProposalStore(_SyncPoolStore):
    """Store for staged, approval-gated graph-extension proposals."""

    def _get(self, cur: Any, proposal_id: str) -> Optional[GraphExtensionProposal]:
        cur.execute(
            f"SELECT {_PROPOSAL_COLUMNS} FROM knowledge_graph_proposals "
            "WHERE proposal_id = %s",
            (proposal_id,),
        )
        row = cur.fetchone()
        return _row_to_proposal(row) if row else None

    def create_proposal(
        self,
        target_graph_id: str,
        *,
        reason: str = "",
        missing_information: Optional[list[Any]] = None,
        source_artifacts: Optional[list[Any]] = None,
        proposed_operations: Optional[list[Any]] = None,
        created_by: str = "",
        session_id: str = "",
    ) -> GraphExtensionProposal:
        """Stage a new pending proposal."""
        proposal_id = "gp-" + uuid.uuid4().hex[:16]
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge_graph_proposals "
                "(proposal_id, target_graph_id, reason, missing_information, "
                " source_artifacts, proposed_operations, created_by, session_id) "
                "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)",
                (
                    proposal_id, target_graph_id, reason,
                    json.dumps(missing_information or []),
                    json.dumps(source_artifacts or []),
                    json.dumps(proposed_operations or []),
                    created_by, session_id,
                ),
            )
            proposal = self._get(cur, proposal_id)
        assert proposal is not None
        return proposal

    def get_proposal(self, proposal_id: str) -> Optional[GraphExtensionProposal]:
        """Fetch one proposal; None when absent."""
        with self._cursor() as cur:
            return self._get(cur, proposal_id)

    def list_proposals(
        self,
        *,
        target_graph_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list[GraphExtensionProposal]:
        """List proposals with optional target/status filters."""
        conditions, params = ["1=1"], []
        if target_graph_id:
            conditions.append("target_graph_id = %s")
            params.append(target_graph_id)
        if status:
            conditions.append("status = %s")
            params.append(status)
        where = " AND ".join(conditions)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_PROPOSAL_COLUMNS} FROM knowledge_graph_proposals "
                f"WHERE {where} ORDER BY created_at DESC LIMIT %s",
                (*params, limit),
            )
            rows = cur.fetchall()
        return [_row_to_proposal(row) for row in rows]

    def decide(
        self,
        proposal_id: str,
        *,
        approved: bool,
        decided_by: str,
    ) -> Optional[GraphExtensionProposal]:
        """Persist an approval decision (pending-only).

        Returns the updated proposal, or the current row unchanged when it
        was already decided (callers surface 409 for mismatched re-decisions).
        """
        new_status = "approved" if approved else "declined"
        with self._cursor() as cur:
            cur.execute(
                "UPDATE knowledge_graph_proposals "
                "SET status = %s, decided_by = %s, decided_at = now(), "
                "    updated_at = now() "
                "WHERE proposal_id = %s AND status = 'pending'",
                (new_status, decided_by, proposal_id),
            )
            return self._get(cur, proposal_id)

    def mark_executed(
        self,
        proposal_id: str,
        execution: dict[str, Any],
    ) -> Optional[GraphExtensionProposal]:
        """Record execution provenance after an approved mutation ran."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE knowledge_graph_proposals "
                "SET status = 'executed', execution = %s::jsonb, updated_at = now() "
                "WHERE proposal_id = %s AND status = 'approved'",
                (json.dumps(execution), proposal_id),
            )
            return self._get(cur, proposal_id)


def _content_node_dict(row: Any) -> dict[str, Any]:
    """Wire-shape node for graph_content (id/name/type aliases + provenance)."""
    return {
        "id": str(row["node_id"]),
        "node_id": str(row["node_id"]),
        "name": str(row.get("label") or row["node_id"]),
        "label": str(row.get("label") or row["node_id"]),
        "type": str(row.get("node_type") or "knowledge"),
        "project_id": str(row.get("project_id") or ""),
        "graph_id": str(row.get("graph_id") or ""),
        "created_at": _iso(row.get("created_at")),
    }


def _content_edge_dict(row: Any) -> dict[str, Any]:
    """Wire-shape edge for graph_content (from/to/label/type aliases)."""
    return {
        "from": str(row["source_id"]),
        "to": str(row["target_id"]),
        "label": str(row.get("relation_type") or "related"),
        "type": str(row.get("relation_type") or "related"),
        "graph_id": str(row.get("graph_id") or ""),
        "created_at": _iso(row.get("created_at")),
    }
