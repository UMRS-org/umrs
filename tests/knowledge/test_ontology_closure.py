"""ADR-049 Leg C pins: transitive closure over declared relations.

Contracts locked:

1. ``CanonicalGraphView.walk`` over an in-memory mirror is deterministic
   (depth-then-id order), cycle-safe by construction (a cycle back to the
   start or a self-loop never repeats a node), depth-capped, and honors
   the direction argument for both outgoing and incoming walks.
2. Collapsed sat names (SUPPORTS / EVIDENCE_FOR / CONTRADICTS /
   EVIDENCE_AGAINST) fail CLOSED — a sat-masquerade closure over an
   ambiguous canonical relation is refused, never silently "canonical-ish".
3. The store-side recursive CTE shape carries the termination guards as
   SQL (bind-arg depth limit, path-membership cycle guard, graph scoping),
   checked statically so an undisciplined edit can't regress termination.
4. Ontology declares ``transitive_relations`` additively; serialized
   payloads without the field deserialize to the v1 default.
"""

from __future__ import annotations

import inspect

import pytest

from umrs.knowledge.canonical_graph_view import (
    CanonicalGraphView,
    _store_worker,
    in_memory_view,
)
from umrs.knowledge.ontology import Ontology, ontology_from_dict

# ADR-049 Leg B: closure traverses only ontology-DECLARED transitive
# relations; the premise suite binds an explicit permissive ontology so
# each walk pins the engine, not the default declaration.
_PERMISSIVE = Ontology(
    ontology_id="premise-permissive",
    version="1",
    entity_types=frozenset({"concept", "entity", "class", "function", "file",
                            "module", "relationship"}),
    relation_types=frozenset({"cites", "uses", "depends_on", "related_to"}),
    transitive_relations=frozenset({"cites", "uses", "depends_on"}),
)


def _view() -> CanonicalGraphView:
    return in_memory_view(project_id="proj-c", graph_id="g-c", ontology=_PERMISSIVE)


def _link(view: CanonicalGraphView, a: str, b: str, rel: str = "USES") -> None:
    view.add_edge(source_id=a, target_id=b, relation_type=rel)


class TestMemoryWalk:
    def test_linear_chain_depths(self) -> None:
        v = _view()
        _link(v, "a", "b", "CITES")
        _link(v, "b", "c", "CITES")
        _link(v, "c", "d", "CITES")
        assert v.walk("a", "CITES") == [("b", 1), ("c", 2), ("d", 3)]

    def test_cycle_terminates_and_counts_once(self) -> None:
        v = _view()
        _link(v, "a", "b", "DEPENDS_ON")
        _link(v, "b", "c", "DEPENDS_ON")
        _link(v, "c", "a", "DEPENDS_ON")  # cycle back to start
        _link(v, "c", "c", "DEPENDS_ON")  # self-loop
        got = v.walk("a", "DEPENDS_ON")
        assert got == [("b", 1), ("c", 2)]
        assert [n for n, _ in got].count("a") == 0

    def test_depth_cap(self) -> None:
        v = _view()
        for i in range(10):
            _link(v, f"n{i}", f"n{i + 1}", "CITES")
        assert v.walk("n0", "CITES", max_depth=3) == [("n1", 1), ("n2", 2), ("n3", 3)]

    def test_incoming_direction(self) -> None:
        v = _view()
        _link(v, "lib", "app", "USES")
        _link(v, "lib2", "app", "USES")
        _link(v, "core", "lib", "USES")
        assert v.walk("app", "USES", direction="incoming") == [
            ("lib", 1), ("lib2", 1), ("core", 2),
        ]

    def test_diamond_shared_ancestor_once(self) -> None:
        v = _view()
        _link(v, "a", "b", "CITES")
        _link(v, "a", "c", "CITES")
        _link(v, "b", "d", "CITES")
        _link(v, "c", "d", "CITES")
        got = v.walk("a", "CITES")
        assert [n for n, _ in got].count("d") == 1
        assert got == [("b", 1), ("c", 1), ("d", 2)]

    def test_unknown_relation_empty(self) -> None:
        v = _view()
        _link(v, "a", "b", "CITES")
        assert v.walk("a", "USES") == []

    def test_bad_direction_refused(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _view().walk("a", "CITES", direction="sideways")

    def test_collapsed_sat_names_fail_closed(self) -> None:
        for collapsed in ("SUPPORTS", "EVIDENCE_FOR", "CONTRADICTS", "EVIDENCE_AGAINST"):
            with pytest.raises(ValueError, match="collapsed"):
                _view().walk("a", collapsed)


class TestStoreCteShape:
    """Static contract on the recursive CTE the store emits (Leg C §3)."""

    def _sql_parts(self):
        from umrs.knowledge.store import KnowledgeStore

        src = inspect.getsource(KnowledgeStore.walk_closure)
        return src

    def test_depth_guard_is_bind_arg(self) -> None:
        assert "w.depth < $2" in self._sql_parts()

    def test_cycle_guard_is_path_membership(self) -> None:
        src = self._sql_parts()
        assert "NOT (e.target_id = ANY(w.path))" in src
        assert "NOT (e.source_id = ANY(w.path))" in src

    def test_recursive_marker_and_graph_scope(self) -> None:
        src = self._sql_parts()
        assert "WITH RECURSIVE walk AS" in src
        assert src.count("graph_id") >= 4

    def test_both_direction_queries_kept(self) -> None:
        src = self._sql_parts()
        assert "e.source_id = w.node_id" in src  # outgoing recursion
        assert "e.target_id = w.node_id" in src  # incoming recursion

    def test_depth_reported_in_content(self) -> None:
        src = self._sql_parts()
        assert "walk_depth" in src


class TestLegBProjection:
    def test_walk_refused_when_ontology_does_not_declare_transitive(self) -> None:
        """Leg B (ADR-049): closure over an undeclared relation refuses."""
        v = in_memory_view(project_id="proj-c", graph_id="g-c")  # DEFAULT_ONTOLOGY
        v.add_edge(source_id="a", target_id="b", relation_type="CITES")
        with pytest.raises(ValueError, match="does not declare it transitive"):
            v.walk("a", "CITES")

    def test_walk_passes_when_declared_on_default(self) -> None:
        """depends_on IS declared on the default ontology (v1 default)."""
        v = in_memory_view(project_id="proj-c", graph_id="g-c")
        v.add_edge(source_id="a", target_id="b", relation_type="DEPENDS_ON")
        v.add_edge(source_id="b", target_id="c", relation_type="DEPENDS_ON")
        assert v.walk("a", "DEPENDS_ON") == [("b", 1), ("c", 2)]

    def test_view_exposes_bound_ontology(self) -> None:
        from umrs.knowledge.ontology import DEFAULT_ONTOLOGY

        v = in_memory_view(project_id="p", graph_id="g", ontology=_PERMISSIVE)
        assert v.ontology is _PERMISSIVE
        assert in_memory_view(project_id="p2").ontology is DEFAULT_ONTOLOGY

    def test_custom_transitive_set_executes_walk(self) -> None:
        """A custom ontology granting 'uses' transitive traverses it."""
        custom = Ontology(
            ontology_id="custom-uses",
            version="1",
            entity_types=frozenset({"concept"}),
            relation_types=frozenset({"uses"}),
            transitive_relations=frozenset({"uses"}),
        )
        v = in_memory_view(project_id="p", graph_id="g", ontology=custom)
        v.add_edge(source_id="a", target_id="b", relation_type="USES")
        assert v.walk("a", "USES") == [("b", 1)]
        # ...but the same custom graph's DEFAULT undeclared dependency stays refused.
        with pytest.raises(ValueError, match="does not declare it transitive"):
            v.walk("a", "DEPENDS_ON")


class TestLivePgClosure:
    def test_live_pg_walk_matches_memory_semantics(self) -> None:
        """Store-backed walk vs the in-memory projection (live DSN rehearsal).

        Skips when no working DSN; otherwise seeds a graph and asserts the
        recursive-CTE result equals the memory BFS result row-for-row.
        """
        import os

        dsn = os.environ.get("DATABASE_URL") or ""
        if not dsn:
            pytest.skip("live PG rehearsal needs DATABASE_URL")
        worker = _store_worker()
        pool = worker.build_pool(dsn)
        import datetime

        from umrs.knowledge.store import KnowledgeEdge as StoreEdge
        from umrs.knowledge.store import KnowledgeNode as StoreNode
        from umrs.knowledge.store import KnowledgeStore

        store = KnowledgeStore(pool)
        gid, pid = "g-pin-legb", "proj-pin-legb"
        now = datetime.datetime.now(datetime.timezone.utc)

        from umrs.knowledge.canonical_graph_view import _run_mutation

        def w(coro):
            _run_mutation(coro)

        for nid in ("a", "b", "c", "d"):
            w(store.upsert_node(StoreNode(node_id=nid, project_id=pid, node_type="concept", label=nid, content={}, content_hash=nid, created_at=now, graph_id=gid)))
        for i, (s, t) in enumerate([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]):
            w(store.upsert_edge(StoreEdge(edge_id=f"p{i}", project_id=pid, source_id=s, target_id=t, relation_type="depends_on", content={}, created_at=now, graph_id=gid)))
        store_view = CanonicalGraphView(store=store, project_id=pid, graph_id=gid)
        got = store_view.walk("a", "DEPENDS_ON")
        assert got == [("b", 1), ("c", 2), ("d", 3)]


class TestOntologyTransitiveDeclaration:
    def test_default_field_and_roundtrip(self) -> None:
        onto = Ontology(
            ontology_id="x", version="1",
            entity_types=frozenset({"concept"}), relation_types=frozenset({"cites"}),
        )
        assert onto.transitive_relations == frozenset({"depends_on"})
        back = ontology_from_dict(onto.to_dict())
        assert back.transitive_relations == frozenset({"depends_on"})

    def test_serialized_absence_maps_to_default(self) -> None:
        back = ontology_from_dict(
            {
                "ontology_id": "x",
                "version": "1",
                "entity_types": ["concept"],
                "relation_types": ["cites"],
            }
        )
        assert back.transitive_relations == frozenset({"depends_on"})

    def test_explicit_transitive_set(self) -> None:
        back = ontology_from_dict(
            {
                "ontology_id": "x",
                "version": "1",
                "entity_types": ["concept"],
                "relation_types": ["cites", "part_of"],
                "transitive_relations": ["cites", "part_of"],
            }
        )
        assert back.transitive_relations == frozenset({"cites", "part_of"})
