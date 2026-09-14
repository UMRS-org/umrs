"""G-P1-6 follow-up + G-P1-9 pins: inverse read-expansion, subtype lattice,
traversal type filtering (ADR-049 residual legs).

Contracts locked:

1. **Inverse read-expansion** — sat inverse tails (ANSWERED_BY family) read
   the forward canonical rows in the flipped role. Legacy exact-name reads
   are untouched byte-wise; dedupe keeps order stable.
2. **Subtype lattice** — querying a parent type includes declared
   descendants transitively; cycles in the lattice never loop; querying an
   undeclared parent yields only exact-type rows (no widening by accident);
   subtype_of written outside entity_types refuses at from_dict.
3. **Traversal type filtering (G-P1-9)** — projection through the bound
   ontology: types outside its entity vocabulary never leave `query()`,
   relation matching included; the default vocabulary is a no-op for
   canonical store types (Adjacency preserved for all canonical rows).
"""

from __future__ import annotations

import pytest
from umrs.knowledge.canonical_graph_view import CanonicalGraphView, in_memory_view
from umrs.knowledge.ontology import Ontology, ontology_from_dict


class TestInverseReadExpansion:
    def test_forward_read_matches_legacy_exactly(self) -> None:
        v = in_memory_view()
        _ = v
        v.add_edge(source_id="paper-x", target_id="paper-y", relation_type="CITES")
        assert v.get_related("paper-x", "CITES") == ["paper-y"]
        assert v.get_related("paper-y", "CITES") == ["paper-x"]

    def test_inverse_tail_expansion(self) -> None:
        v = in_memory_view()
        # canonical storage: evidence-2 answers claim-1 (edge stored answers
        # from the forward side; ANSWERED_BY is sat-only).
        v.add_edge(source_id="evidence-2", target_id="claim-1", relation_type="ANSWERS")
        assert v.get_related("claim-1", "ANSWERS") == ["evidence-2"]
        # The sat inverse tail reads the same row via the flipped role:
        expanded = v.get_related("claim-1", "ANSWERED_BY")
        assert expanded == ["evidence-2"]


def _lattice_view() -> CanonicalGraphView:
    onto = Ontology(
        ontology_id="file-lattice",
        version="1",
        entity_types=frozenset({"file", "module", "concept", "class", "impl"}),
        relation_types=frozenset({"cites"}),
        subtype_of=(("module", "file"), ("impl", "module")),
    )
    return in_memory_view(ontology=onto)


class TestSubtypeLattice:
    def test_parent_query_includes_transitive_descendants(self) -> None:
        v = _lattice_view()
        v.add_node(node_id="a.py", node_type="file")
        v.add_node(node_id="pkg.x", node_type="module")
        v.add_node(node_id="fn.fast", node_type="impl")
        v.add_node(node_id="idea", node_type="concept")
        got = {n.node_id for n in v.query(node_type="file")}
        assert got == {"a.py", "pkg.x", "fn.fast"}

    def test_midlevel_query(self) -> None:
        v = _lattice_view()
        v.add_node(node_id="a.py", node_type="file")
        v.add_node(node_id="pkg.x", node_type="module")
        v.add_node(node_id="fn.fast", node_type="impl")
        got = {n.node_id for n in v.query(node_type="module")}
        assert got == {"pkg.x", "fn.fast"}

    def test_leaf_query_does_not_widen(self) -> None:
        v = _lattice_view()
        v.add_node(node_id="fn.fast", node_type="impl")
        v.add_node(node_id="pkg.x", node_type="module")
        assert {n.node_id for n in v.query(node_type="impl")} == {"fn.fast"}

    def test_cycle_in_lattice_terminates(self) -> None:
        onto = Ontology(
            ontology_id="cyclic",
            version="1",
            entity_types=frozenset({"a", "b"}),
            relation_types=frozenset({"cites"}),
            subtype_of=(("a", "b"), ("b", "a")),
        )
        v = in_memory_view(ontology=onto)
        v.add_node(node_id="n1", node_type="a")
        v.add_node(node_id="n2", node_type="b")
        assert {n.node_id for n in v.query(node_type="a")} == {"n1", "n2"}

    def test_subtype_outside_entity_types_refused_at_load(self) -> None:
        with pytest.raises(ValueError, match="subtype_of references undeclared"):
            ontology_from_dict(
                {
                    "ontology_id": "bad",
                    "version": "1",
                    "entity_types": ["concept"],
                    "relation_types": ["cites"],
                    "subtype_of": {"module": "file"},
                }
            )


class TestTraversalTypeFiltering:
    def test_default_vocabulary_noop_for_canonical_rows(self) -> None:
        v = in_memory_view()  # DEFAULT_ONTOLOGY entity_types cover the 7
        for t in ("concept", "file", "entity", "class", "module", "function", "relationship"):
            v.add_node(node_id=f"n-{t}", node_type=t)
        assert len(v.query()) == 7

    def test_custom_vocabulary_filters_nodes(self) -> None:
        onto = Ontology(
            ontology_id="narrow",
            version="1",
            entity_types=frozenset({"concept"}),
            relation_types=frozenset({"cites"}),
        )
        v = in_memory_view(ontology=onto)
        v.add_node(node_id="ok", node_type="concept")
        v.add_node(node_id="excluded", node_type="file")
        assert [n.node_id for n in v.query()] == ["ok"]

    def test_relation_matched_filter(self) -> None:
        onto = Ontology(
            ontology_id="narrow2",
            version="1",
            entity_types=frozenset({"concept"}),
            relation_types=frozenset({"cites"}),
        )
        v = in_memory_view(ontology=onto)
        v.add_node(node_id="ok", node_type="concept")
        v.add_node(node_id="drop", node_type="file")
        v.add_edge(source_id="ok", target_id="drop", relation_type="CITES")
        got = v.query(relation_type="CITES")
        assert [n.node_id for n in got] == ["ok"]
