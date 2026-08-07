"""Regression tests for #207: UMRS loop executor must surface the real
project knowledge graph (nodes + edges) instead of silently injecting
an empty digest into every LLM loop context."""

from __future__ import annotations

from types import SimpleNamespace

from umrs.knowledge.agent_executor import UmrsAgentIterationExecutor
from umrs.knowledge.link import (
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    RelationType,
)


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(node_id="a", node_type="paper", title="MERMAIDs paper"))
    graph.add_node(KnowledgeNode(node_id="b", node_type="dataset", title="seismic data"))
    graph.add_edge(
        KnowledgeEdge(
            source_id="a",
            target_id="b",
            relation_type=RelationType.DERIVES_FROM,
        )
    )
    return graph


def test_load_graph_surfaces_nodes_and_edges() -> None:
    executor = UmrsAgentIterationExecutor(
        project=SimpleNamespace(knowledge_graph=_graph()),
        chat=lambda **_: "{}",
    )
    digest = executor._load_graph()

    ids = {n["id"] for n in digest["nodes"]}
    assert ids == {"a", "b"}
    types = {n["type"] for n in digest["nodes"]}
    assert "paper" in types and "dataset" in types
    assert digest["edges"] == [{"from": "a", "to": "b", "type": "derives_from"}]
    assert "note" not in digest


def test_load_graph_no_graph_reports_note() -> None:
    executor = UmrsAgentIterationExecutor(project=SimpleNamespace(), chat=lambda **_: "{}")
    digest = executor._load_graph()
    assert digest["nodes"] == [] and digest["edges"] == []
    assert digest.get("note") == "knowledge graph unavailable"
