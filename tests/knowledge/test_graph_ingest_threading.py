"""Named-graph ingestion: node/edge id scoping + spec threading (0069)."""

from __future__ import annotations

from umrs.jobs.knowledge_adapters import _build_edges, _build_nodes
from umrs.knowledge.graph_registry import default_graph_id

_PARSED = {"classes": ["Solver"], "functions": ["run_shot"]}


class TestBuildNodesGraphScoping:
    def test_default_graph_keeps_legacy_ids(self):
        nodes = _build_nodes("art-1", "ir-1", _PARSED, "proj-a")
        ids = [n.node_id for n in nodes]
        assert "node-file-art-1" in ids
        assert any(i.startswith("node-class-Solver") for i in ids)
        assert all(n.graph_id == "" for n in nodes)

    def test_explicit_default_graph_id_keeps_legacy_ids(self):
        default = default_graph_id("proj-a")
        nodes = _build_nodes("art-1", "ir-1", _PARSED, "proj-a", graph_id=default)
        assert "node-file-art-1" in [n.node_id for n in nodes]

    def test_named_graph_scopes_node_ids_and_carries_graph(self):
        nodes = _build_nodes("art-1", "ir-1", _PARSED, "proj-a", graph_id="g-exp")
        ids = [n.node_id for n in nodes]
        assert "node-file-g-exp-art-1" in ids
        assert any(i.startswith("node-class-g-exp-") for i in ids)
        assert all(n.graph_id == "g-exp" for n in nodes)

    def test_two_graphs_can_hold_same_artifact(self):
        default_nodes = _build_nodes("art-1", "ir-1", _PARSED, "proj-a")
        named_nodes = _build_nodes("art-1", "ir-1", _PARSED, "proj-a", graph_id="g-exp")
        assert {n.node_id for n in default_nodes}.isdisjoint({n.node_id for n in named_nodes})


class TestBuildEdgesGraphScoping:
    def test_edges_inherit_node_graph(self):
        parsed = {"classes": ["Solver"], "functions": []}
        nodes = _build_nodes("art-1", "ir-1", parsed, "proj-a", graph_id="g-exp")
        edges = _build_edges(nodes)
        assert len(edges) == 1
        assert edges[0].graph_id == "g-exp"
        assert edges[0].source_id == "node-file-g-exp-art-1"

    def test_default_edges_stay_legacy(self):
        parsed = {"classes": ["Solver"], "functions": []}
        nodes = _build_nodes("art-1", "ir-1", parsed, "proj-a")
        for edge in _build_edges(nodes):
            assert edge.graph_id == ""
