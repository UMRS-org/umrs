"""PgGraphSink — Postgres-backed knowledge graph persistence sink.

Phase 1 of the knowledge persistence layer.  PgGraphSink implements the
KnowledgeSink protocol so the KnowledgeRouter can compile derived indexes
into Postgres tables (knowledge_nodes / knowledge_edges).

Determinism guarantee:
  Two generate() calls with the same set of IRs produce *identical row sets*
  (same node_ids, edge_ids, content_hash values).  Timestamps (created_at)
  may differ because the store uses ``datetime.now(timezone.utc)`` inside
  upsert_node — this does not affect graph topology.

CRITICAL: No imports from ``umrs.api`` or ``umrs.server``.
Only stdlib + knowledge/store.py + knowledge/ir.py imports.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from umrs.metrics import METRICS

from .graph_registry import default_graph_id
from .ir import KnowledgeIR
from .store import KnowledgeEdge, KnowledgeIRRow, KnowledgeNode, KnowledgeStore

logger = logging.getLogger(__name__)


# ── Protocol ────────────────────────────────────────────────────────────────


@runtime_checkable
class KnowledgeSink(Protocol):
    """A derived index that can be rebuilt from canonical ``KnowledgeIR`` records.

    The ``generate`` method performs a *truncate-and-recompile* over Postgres
    nodes/edges.  It is the database-backed analogue of the
    ``LocalGraphSink.compile`` method that writes JSON.
    """

    name: str

    def generate(
        self,
        store: KnowledgeStore,
        project_id: str,
        ir_ids: list[str],
    ) -> int:
        """Truncate project nodes/edges, recompile from IRs, return node count.

        *Deterministic*: two runs with identical IRs → identical row sets
        (node_ids, edge_ids, content values).

        Args:
            store: Postgres-backed KnowledgeStore.
            project_id: Scopes the truncate and upsert to one project.
            ir_ids: Canonical IR ids to load and compile from.

        Returns:
            Number of nodes created/upserted.
        """
        ...


# ── Implementation ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RegenerateResult:
    """Outcome of a Postgres graph regeneration."""

    sink: str
    nodes: int
    edges: int

    def to_dict(self) -> dict[str, Any]:
        return {"sink": self.sink, "nodes": self.nodes, "edges": self.edges}


class PgGraphSink:
    """Postgres-backed knowledge-graph sink.

    Truncates derived nodes/edges for a project, loads the canonical IRs,
    extracts nodes/edges from IR content, and upserts them into Postgres.
    """

    name = "pg_graph"

    def __init__(self, store: KnowledgeStore) -> None:
        """Initialize with a Postgres-backed KnowledgeStore."""
        self._store = store

    async def generate(
        self,
        store: Optional[KnowledgeStore] = None,
        project_id: Optional[str] = None,
        ir_ids: Optional[list[str]] = None,
        graph_id: Optional[str] = None,
    ) -> int:
        """Truncate and recompile the Postgres knowledge graph.

        Args:
            store: KnowledgeStore (defaults to the one from __init__).
            project_id: Project scope (defaults to the one from __init__).
            ir_ids: IR ids to load (defaults to the list from __init__).
            graph_id: Target graph (defaults to the project's deterministic
                default graph — no registry lookup needed on the write path).

        Returns:
            Number of nodes created/upserted.
        """
        t0 = time.monotonic()
        outcome = "ok"
        nodes_written = 0
        try:
            store = self._resolve_store(store, project_id, ir_ids)
            gid = graph_id or default_graph_id(project_id)
            nodes_written = await self._regenerate(store, project_id, gid, ir_ids)
            return nodes_written
        except Exception:
            outcome = "error"
            raise
        finally:
            # X1 metrics: failure-isolated — metric outage never breaks generate().
            try:
                METRICS.graph_sink_generate(
                    self.name, outcome, time.monotonic() - t0, nodes_written
                )
            except Exception:  # noqa: BLE001
                pass

    def _resolve_store(
        self,
        store: Optional[KnowledgeStore],
        project_id: Optional[str],
        ir_ids: Optional[list[str]],
    ) -> KnowledgeStore:
        """Validate generate() inputs and resolve the store."""
        if project_id is None or ir_ids is None:
            raise ValueError("project_id and ir_ids are required for generate()")
        return store or self._store

    async def _regenerate(
        self,
        store: KnowledgeStore,
        project_id: str,
        graph_id: str,
        ir_ids: list[str],
    ) -> int:
        """Truncate graph-scope rows and recompile nodes/edges from IRs."""
        # 1. Truncate derived nodes/edges for the graph
        await store.truncate_nodes(project_id, graph_id=graph_id)
        await store.truncate_edges(project_id=project_id, graph_id=graph_id)

        # 2. Load canonical IRs by id
        irs = await store.list_ir_by_ids(ir_ids)
        if not irs:
            logger.info("No IRs found for project %s — no nodes created", project_id)
            return 0

        # 3. Extract nodes/edges from IRs
        nodes_written = 0
        for ir in irs:
            await self._upsert_node(store, ir, project_id, graph_id)
            nodes_written += 1
            await self._upsert_edges(store, ir, project_id, graph_id)
        return nodes_written

    async def _upsert_node(
        self,
        store: KnowledgeStore,
        ir: KnowledgeIRRow,
        project_id: str,
        graph_id: str,
    ) -> None:
        """Create and upsert a single knowledge-graph node from one IR."""
        content = ir.content or {}
        extracted = content.get("extracted", {}) or {}
        artifact_id = content.get("artifact_id", "")
        artifact_type = content.get("artifact_type", "")
        filename = extracted.get("filename", "")

        node_id = _node_id_from_ir(ir)
        node_type = _classify_type(artifact_type, extracted)
        label = _extract_label(extracted)
        node = KnowledgeNode(
            node_id=node_id,
            project_id=project_id,
            label=label,
            node_type=node_type,
            content={
                "artifact_id": artifact_id,
                "artifact_type": artifact_type,
                "filename": filename,
            },
            content_hash=_content_hash_for_node(ir, node_type, label),
            source_ir=ir.ir_id,
            graph_id=graph_id,
        )
        await store.upsert_node(node)

    async def _upsert_edges(
        self,
        store: KnowledgeStore,
        ir: KnowledgeIRRow,
        project_id: str,
        graph_id: str,
    ) -> None:
        """Create and upsert edges from one IR's extracted knowledge."""
        node_id = _node_id_from_ir(ir)
        content = ir.content or {}
        extracted = content.get("extracted", {}) or {}

        # Edges from dependencies
        for dep_name in extracted.get("dependencies", []) or []:
            edge = KnowledgeEdge(
                edge_id=_dep_edge_id(ir, dep_name),
                project_id=project_id,
                source_id=node_id,
                target_id=_node_id_for_dep(dep_name, ""),
                relation_type="depends_on",
                content={"dep_name": dep_name, "ir_id": ir.ir_id},
                graph_id=graph_id,
            )
            await store.upsert_edge(edge)

        # Edges from citations
        for citation in extracted.get("citations", []) or []:
            edge = KnowledgeEdge(
                edge_id=_cite_edge_id(ir, citation),
                project_id=project_id,
                source_id=node_id,
                target_id=_node_id_for_dep(citation, "paper"),
                relation_type="cites",
                content={"citation": citation, "ir_id": ir.ir_id},
                graph_id=graph_id,
            )
            await store.upsert_edge(edge)

        # Edges from implements (code → algorithm)
        for algo in extracted.get("algorithms", []) or []:
            edge = KnowledgeEdge(
                edge_id=_impl_edge_id(ir, algo),
                project_id=project_id,
                source_id=node_id,
                target_id=_algo_node_id(algo),
                relation_type="implements",
                content={"algorithm": algo, "ir_id": ir.ir_id},
                graph_id=graph_id,
            )
            await store.upsert_edge(edge)


# ── Deterministic id generators ────────────────────────────────────────────


def _node_id_from_ir(ir: KnowledgeIR) -> str:
    """Deterministic node id from ir_id — one node per IR."""
    return f"node:{ir.ir_id}"


def _content_hash_for_node(ir: KnowledgeIR, node_type: str, label: str) -> str:
    """Content hash for node dedup (type + label + ir_id)."""
    seed = f"{ir.ir_id}:{node_type}:{label}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _dep_edge_id(ir: KnowledgeIR, dep_name: str) -> str:
    return f"edge:{ir.ir_id}:depends_on:{dep_name}"


def _cite_edge_id(ir: KnowledgeIR, citation: str) -> str:
    return f"edge:{ir.ir_id}:cites:{citation}"


def _impl_edge_id(ir: KnowledgeIR, algo: str) -> str:
    return f"edge:{ir.ir_id}:implements:{algo}"


def _node_id_for_dep(dep_name: str, dep_type: str) -> str:
    """Derived node id for a dependency (no IR record needed)."""
    return f"dep:{dep_type}:{dep_name}"


def _algo_node_id(algo: str) -> str:
    return f"algo:{algo}"


# ── Content extractors ─────────────────────────────────────────────────────


def _classify_type(artifact_type: str, extracted: dict[str, Any]) -> str:
    """Map artifact_type → knowledge graph node_type."""
    mapping = {
        "code": "class",
        "paper": "concept",
        "notebook": "entity",
        "document": "concept",
        "schema": "class",
    }
    return mapping.get(artifact_type, "entity")


def _extract_label(extracted: dict[str, Any]) -> str:
    """Derive a human-readable label from extracted knowledge."""
    if not extracted:
        return "unnamed"
    return extracted.get("title", extracted.get("filename", "unnamed"))
