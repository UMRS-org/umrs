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
import threading
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
    """Run the store write on the view's pool-owning worker loop.

    The knowledge asyncpg pool is loop-bound; a mutation scheduled on any
    OTHER loop (asyncio.run() per write from sync contexts; a pytest-asyncio
    test loop; a different api request-loop) crashes cross-loop
    (CI 2026-09-13 class: 'attached to a different loop' / 'another
    operation is in progress'). All mutations ride the same dedicated
    worker thread the pool was built on. Semantic split preserved from the
    pre-worker design: sync callers BLOCK until the write lands (their
    premises assert immediately afterwards); async callers fire-and-forget
    with a warning callback (the api request loop never parks on it).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Sync context — block so the write is visible to the caller.
        _store_worker().submit_blocking(coro)
        return
    _store_worker().submit(coro)


class _StoreWorker:
    """Daemon background thread owning the store loop + pool lifetime.

    One per process. The knowledge pool is created lazily inside the
    worker's loop and every mutation is dispatched back to it via
    run_coroutine_threadsafe — order-safe from sync and async callers
    alike.
    """

    def __init__(self) -> None:
        import threading

        self._loop = asyncio.new_event_loop()
        self._pool: Any = None
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="canonical-graph-store-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def build_pool(self, dsn: str) -> Any:
        """Create the asyncpg pool INSIDE the worker loop (idempotent)."""
        if self._pool is not None:
            return self._pool

        import asyncpg  # type: ignore[import-untyped]

        import json as _json

        async def _json_codecs(conn: Any) -> None:
            # jsonb codec registration self-contained here (foundation
            # module must not import umrs.api) — without it asyncpg refuses
            # dict params on content columns.
            for typename in ("json", "jsonb"):
                await conn.set_type_codec(
                    typename,
                    encoder=_json.dumps,
                    decoder=_json.loads,
                    schema="pg_catalog",
                    format="text",
                )

        async def _create() -> Any:
            # asyncpg.create_pool returns a PoolInit (awaitable, NOT a
            # coroutine object) — wrap it: run_coroutine_threadsafe demands
            # an actual coroutine.
            return await asyncpg.create_pool(dsn=dsn, init=_json_codecs)

        self._pool = asyncio.run_coroutine_threadsafe(_create(), self._loop).result(
            timeout=10
        )
        return self._pool

    def submit(self, coro: Any) -> None:
        """Fire-and-forget a mutation on the worker loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _report(fut: Any) -> None:
            exc = fut.exception()
            if exc is not None:
                logger.warning("canonical graph mutation failed: %s", exc, exc_info=True)

        future.add_done_callback(_report)

    def submit_blocking(self, coro: Any, timeout: float = 10.0) -> None:
        """Run a mutation on the worker loop and block the (sync) caller."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.result(timeout=timeout)

    def result_value(self, coro: Any, timeout: float = 15.0) -> Any:
        """Run a read on the worker loop and return its value to the caller."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        """Tear down pool + loop (test isolation)."""
        if self._pool is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._pool.close(), self._loop).result(
                    timeout=5
                )
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
            self._pool = None
        self._loop.call_soon_threadsafe(self._loop.stop)


_worker: "_StoreWorker | None" = None
_worker_lock = threading.Lock()


def _store_worker() -> _StoreWorker:
    global _worker
    with _worker_lock:
        if _worker is None or not _worker.loop.is_running():
            _worker = _StoreWorker()
        return _worker


def _reset_store_worker_for_tests() -> None:
    """Drop the worker (test isolation between loop-lifetime-sensitive suites)."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.shutdown()
            _worker = None


def _resolve_pool(request: Any | None) -> Any | None:
    """Resolve the shared api-side asyncpg pool, if caller passed a request."""
    if request is None:
        return None
    app_state = getattr(getattr(request, "app", None), "state", None)
    return getattr(app_state, "umrs_knowledge_asyncpg_pool", None)


def _maybe_knowledge_store() -> KnowledgeStore | None:
    """Resolve a durable KnowledgeStore when the knowledge DSN is configured.

    The pool is built inside the dedicated store-worker thread's loop
    (`_store_worker`), so mutations dispatched back to that loop stay on
    the pool's owning loop from EVERY caller context (sync, api, pytest).
    """
    try:
        from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

        dsn = _resolve_knowledge_dsn()
        if not dsn:
            return None
        from .store import get_store  # type: ignore[attr-defined]

        try:
            return get_store()
        except RuntimeError:
            _worker_pool = _store_worker().build_pool(dsn)
            from .store import set_pool

            set_pool(_worker_pool)
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


def in_memory_view(
    project_id: str = "default",
    graph_id: str | None = None,
    *,
    ontology: Any | None = None,
) -> "CanonicalGraphView":
    """Explicitly store-free view — NEVER resolves a durable store.

    Complement to :func:`default_view`, whose ``store=None`` argument means
    "auto-resolve". Premise/unit tests want the old InMemoryKnowledgeGraph
    semantics unconditionally even on hosts where the knowledge DSN points
    at a live Postgres (CI self-hosted class 2026-09-13: adapter suites
    auto-resolved a pool and died with cross-loop asyncpg errors).
    """
    from .graph_registry import default_graph_id

    gid = graph_id or default_graph_id(project_id)
    return CanonicalGraphView(
        store=None, project_id=project_id, graph_id=gid, ontology=ontology
    )


class CanonicalGraphView:
    """Canonical-store-backed implementation of the graph API protocol."""

    def __init__(
        self,
        *,
        store: KnowledgeStore | None,
        project_id: str,
        graph_id: str,
        ontology: Any | None = None,
    ) -> None:
        # The typed contract this graph answers to (ADR-049 Leg B): closure
        # verbs enforce its transitive_relations declaration; responses
        # expose it to query surfaces. DEFAULT_ONTOLOGY when unbound.
        from .ontology import DEFAULT_ONTOLOGY

        self._ontology = ontology if ontology is not None else DEFAULT_ONTOLOGY
        self._store = store
        self._project_id = project_id
        self._graph_id = graph_id
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, KnowledgeEdge] = {}
        self._outgoing: dict[str, list[str]] = {}
        self._incoming: dict[str, list[str]] = {}

    @property
    def ontology(self) -> Any:
        """The typed contract bound to this view (never None)."""
        return self._ontology

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
        """Directly-linked ids, inverse-expanded (G-P1-6 follow-up).

        Exact-name matches first (legacy semantics). For declared inverse
        pairs (ANSWERED_BY/PRODUCED_BY/REFERENCED_BY/DESCRIBED_BY/
        IMPLEMENTED_BY) the canonical rows are also scanned in the flipped
        role: ask all n's `ANSWERED_BY` and you get exactly the sources
        whose `answers` rows point at n. Sat collapsing families are not
        expanded (block per Leg C's fail-closed decision).
        """
        related: list[str] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.target_id)
        for edge in self.get_incoming(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.source_id)
        forward_name = self._inverse_pair_forward(relation_type)
        if forward_name is not None:
            # sat-inverse columns read the canonical rows in the FLIPPED
            # role: n `ANSWERED_BY` = sources of canonical rows into n.
            for edge in self.get_incoming(node_id):
                if edge.relation_type == forward_name:
                    related.append(edge.source_id)
        return self._dedupe_preserve(related)

    @staticmethod
    def _dedupe_preserve(ids: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for node in ids:
            if node not in seen:
                seen.add(node)
                out.append(node)
        return out

    def _inverse_pair_forward(self, relation_type: RelationType) -> str | None:
        """Canonical forward name when `relation_type` is a declared inverse
        tail (ANSWERED_BY family), else None."""
        from .ontology import canonical_relation, canonical_to_sat

        requested_name = (
            relation_type if isinstance(relation_type, str)
            else getattr(relation_type, "name", str(relation_type))
        )
        canonical, inverse = canonical_relation(requested_name)
        if not inverse:
            return None
        return canonical_to_sat(canonical, None)

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

    # -- closure inference (ADR-049 Leg C) -----------------------------------

    # Sat relation names whose canonical target is ambiguous (two sat names
    # collapse to the same canonical relation): closure over them can never
    # be claimed sat-faithful, only canonical — so they fail CLOSED here.
    _WALK_FAIL_CLOSED_SAT: tuple[str, ...] = (
        "SUPPORTS",
        "EVIDENCE_FOR",
        "CONTRADICTS",
        "EVIDENCE_AGAINST",
    )

    def _walk_resolve(
        self, relation_type: Any, direction: str
    ) -> tuple[str, bool, str, str]:
        """Project (sat|canonical relation, direction) onto (sat_name,
        canonical_name, effective_direction) with the fail-closed contract."""
        sat_name = relation_type if isinstance(relation_type, str) else (
            getattr(relation_type, "name", None) or str(relation_type)
        )
        if sat_name == "related_to":
            return sat_name, "related_to", direction
        if sat_name in self._WALK_FAIL_CLOSED_SAT:
            raise ValueError(
                f"walk over collapsed sat relation {sat_name!r} refused — "
                "canonical shares with another sat relation; query the "
                "canonical layer with content['sat_relation'] filters instead"
            )
        from .ontology import canonical_relation

        canonical_name, inverse = canonical_relation(sat_name)
        effective_direction = (
            "incoming" if (direction == "incoming") != bool(inverse) else "outgoing"
        )
        return sat_name, canonical_name, effective_direction

    def walk(
        self,
        start_id: str,
        relation_type: Any,
        *,
        direction: str = "outgoing",
        max_depth: int = 64,
    ) -> list[tuple[str, int]]:
        """Transitive closure over ONE declared relation (ADR-049 Leg C).

        Returns ``[(node_id, depth), ...]`` ordered (depth, node_id),
        deterministic, cycle-safe by construction (path membership), and
        depth-capped by contract. Inferred paths are RELATIONAL read
        results only — nothing about the durable rows changes.

        - relation_type is the view language (sat enum name or canonical).
        - Inverse-flagged sat relations walk the canonical edge in the
          flipped direction, per the ADR-026 projection.
        - Collapsed sat names (SUPPORTS/EVIDENCE_FOR/CONTRADICTS family)
          fail CLOSED: no sat-masquerade of an ambiguous closure.
        - Memory-backed views BFS over the mirror; store-backed views run
          a recursive CTE through the store (worker-routed pool — the pool
          stays on its own loop).
        """
        if direction not in ("outgoing", "incoming"):
            raise ValueError(f"direction must be outgoing|incoming, not {direction!r}")
        sat_name, canonical_name, effective_direction = self._walk_resolve(
            relation_type, direction
        )
        # ADR-049 Leg B: closure runs only over relations the bound
        # ontology DECLARES transitive — an undeclared closure is refused,
        # not silently traversed (query-time ontological projection).
        if canonical_name not in self._ontology.transitive_relations:
            raise ValueError(
                f"walk over {canonical_name!r} refused: ontology "
                f"'{self._ontology.ontology_id}' v{self._ontology.version} "
                f"does not declare it transitive "
                f"(transitive_relations={sorted(self._ontology.transitive_relations)})"
            )
        if self._store is None:
            return self._walk_memory(start_id, sat_name, direction, max_depth)
        edges = _store_worker().result_value(
            self._store.walk_closure(
                self._project_id,
                self._graph_id,
                start_id,
                canonical_name,
                effective_direction,
                max_depth,
            )
        )
        return sorted(
            {
                (
                    e.target_id if effective_direction == "outgoing" else e.source_id,
                    int((e.content or {}).get("walk_depth") or 0),
                )
                for e in edges
            },
            key=lambda row: (row[1], row[0]),
        )

    def _walk_successors(self, node: str, sat_name: str, direction: str) -> list[str]:
        edges = self.get_outgoing(node) if direction == "outgoing" else self.get_incoming(node)

        def _match(relation: Any) -> bool:
            # sat enums carry NAMES; stored rows may be raw strings.
            return relation == sat_name or getattr(relation, "name", None) == sat_name

        if direction == "outgoing":
            return [e.target_id for e in edges if _match(e.relation_type)]
        return [e.source_id for e in edges if _match(e.relation_type)]

    def _walk_memory(
        self,
        start_id: str,
        sat_name: str,
        direction: str,
        max_depth: int,
    ) -> list[tuple[str, int]]:
        """Cycle-safe BFS over the in-memory mirror (sat edge language)."""
        seen: dict[str, int] = {}
        frontier = [(start_id, 0)]
        while frontier:
            node, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for nid in self._walk_successors(node, sat_name, direction):
                if nid == start_id or nid in seen:
                    continue  # cycle guard — every visited node counted once
                seen[nid] = depth + 1
                frontier.append((nid, depth + 1))
        return sorted(seen.items(), key=lambda row: (row[1], row[0]))

    def _subtype_descendants(self, parent_type: str) -> set[str]:
        """{parent_type} ∪ all transitive subtype descendants (G-P1-6)."""
        children = {c for c, p in self._ontology.subtype_of if p == parent_type}
        seen = {parent_type} | children
        frontier = list(children)
        while frontier:
            child = frontier.pop()
            for c, p in self._ontology.subtype_of:
                if p == child and c not in seen:
                    seen.add(c)
                    frontier.append(c)
        return seen

    def query(
        self,
        *,
        node_type: str | None = None,
        relation_type: RelationType | None = None,
    ) -> list[KnowledgeNode]:
        # G-P1-9: traversal results project through the bound ontology's
        # entity vocabulary WHEN the graph carries a NON-default ontology —
        # the default ontology's 7 CHECK-bucket types intentionally do not
        # restrict sat semantic labels (paper/code/hypothesis/etc.), so
        # default-bound views stay no-op (premise law of the whole suite).
        from .ontology import DEFAULT_ONTOLOGY

        def _projected(nodes):
            if self._ontology.ontology_id == DEFAULT_ONTOLOGY.ontology_id:
                return nodes
            return [
                n for n in nodes if n.node_type in self._ontology.entity_types
            ]

        if node_type is not None:
            # G-P1-6 subtype expansion: querying a parent includes its
            # declared descendants transitively.
            types = self._subtype_descendants(node_type)
            return _projected([n for n in self._nodes.values() if n.node_type in types])
        if relation_type is not None:
            matching: set[str] = set()
            for edge in self._edges.values():
                if edge.relation_type == relation_type:
                    matching.add(edge.source_id)
                    matching.add(edge.target_id)
            return _projected([self._nodes[nid] for nid in matching if nid in self._nodes])
        return _projected(list(self._nodes.values()))


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
