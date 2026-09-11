"""Graph-centric reasoning — Knowledge Graph as the navigation hub.

M3 Phase 2: Knowledge Graph–Centric Reasoning.

The Knowledge Graph becomes the navigation layer for all reasoning.
When the planner or verification chat needs to reason, they go through
the graph first, which expands to:

  1. Knowledge Graph    — what is related?
  2. Evidence           — what supports this?
  3. Scientific Memory  — what has been done?
  4. Experiments        — what has been tested?
  5. Catalog           — what artifacts exist?
  6. RAG               — what is in the literature?
  7. LLM               — generate plan/answer

This module provides the GraphNavigator that wraps the KnowledgeGraph
API and provides high-level reasoning methods.

Key principle: The graph is the hub. All reasoning navigates through
the graph. This prevents reverting to `vector search → LLM`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from umrs.knowledge.canonical_graph_view import CanonicalGraphView, default_view
from umrs.knowledge_graph_api import (
    InMemoryKnowledgeGraph,
    KnowledgeEdge,
    KnowledgeNode,
    RelationType,
)

__all__ = [
    "RelatedObjects",
    "EvidenceChain",
    "ExperimentChain",
    "GraphNavigator",
    "GraphReasoningContext",
]


# ── Result Models ─────────────────────────────────────────────────────────


@dataclass
class RelatedObjects:
    """Objects related to a query through the knowledge graph.

    Attributes:
        papers: Related paper IDs
        code: Related code IDs
        notebooks: Related notebook IDs
        datasets: Related dataset IDs
        hypotheses: Related hypothesis IDs
        experiments: Related experiment IDs
        findings: Related finding IDs
        figures: Related figure IDs
        equations: Related equation IDs
    """

    papers: list[str] = field(default_factory=list)
    code: list[str] = field(default_factory=list)
    notebooks: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)

    def total(self) -> int:
        """Total number of related objects."""
        return (
            len(self.papers)
            + len(self.code)
            + len(self.notebooks)
            + len(self.datasets)
            + len(self.hypotheses)
            + len(self.experiments)
            + len(self.findings)
            + len(self.figures)
            + len(self.equations)
        )


@dataclass
class EvidenceChain:
    """Chain of evidence supporting or contradicting a claim.

    Attributes:
        claim_id: The claim (hypothesis, conclusion, finding)
        supporting: Evidence IDs that support the claim
        contradicting: Evidence IDs that contradict the claim
        graph_path: Path through the knowledge graph
    """

    claim_id: str = ""
    supporting: list[str] = field(default_factory=list)
    contradicting: list[str] = field(default_factory=list)
    graph_path: list[str] = field(default_factory=list)


@dataclass
class ExperimentChain:
    """Chain of experiments testing a hypothesis.

    Attributes:
        hypothesis_id: The hypothesis being tested
        experiments: Experiment IDs that test this hypothesis
        findings: Finding IDs from these experiments
        graph_path: Path through the knowledge graph
    """

    hypothesis_id: str = ""
    experiments: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    graph_path: list[str] = field(default_factory=list)


@dataclass
class GraphReasoningContext:
    """Complete reasoning context from the knowledge graph.

    This is what the planner and verification chat receive when they
    ask the graph navigator to reason about a query.

    Attributes:
        query: The original query
        related: Related objects organized by type
        evidence_chains: Evidence chains for each related claim
        experiment_chains: Experiment chains for each related hypothesis
        graph_paths: Paths through the knowledge graph for explainability
    """

    query: str = ""
    related: RelatedObjects = field(default_factory=RelatedObjects)
    evidence_chains: list[EvidenceChain] = field(default_factory=list)
    experiment_chains: list[ExperimentChain] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)


# ── Graph Navigator ─────────────────────────────────────────────────────


class GraphNavigator:
    """Navigate the knowledge graph for reasoning.

    The navigator wraps the KnowledgeGraph API and provides high-level
    reasoning methods that the planner and verification chat can use.

    Priority order (M3 Decision 4):
      1. Knowledge Graph    — what is related?
      2. Evidence           — what supports this?
      3. Scientific Memory  — what has been done?
      4. Experiments        — what has been tested?
      5. Catalog           — what artifacts exist?
      6. RAG               — what is in the literature?
      7. LLM               — generate plan/answer
    """

    def __init__(
        self, graph: InMemoryKnowledgeGraph | CanonicalGraphView | None = None
    ) -> None:
        # ADR-026 exit wave A: default to the canonical-store-backed view
        # (semantics identical to InMemoryKnowledgeGraph when store=None —
        # in-memory dicts, edge dedupe, direction-agnostic reads).
        self._graph = graph or default_view(store=None)

    @property
    def graph(self) -> InMemoryKnowledgeGraph | CanonicalGraphView:
        """The underlying knowledge graph."""
        return self._graph

    # ── Graph Navigation Methods ────────────────────────────────────────

    def find_related(self, object_id: str) -> RelatedObjects:
        """Find all objects related to the given object through the graph.

        This navigates the graph to find related papers, code, notebooks,
        datasets, hypotheses, experiments, findings, figures, and equations.

        Args:
            object_id: The object to find related objects for

        Returns:
            RelatedObjects with IDs organized by type
        """
        related = RelatedObjects()
        neighbors = self._graph.get_neighbors(object_id)

        for neighbor_id in neighbors:
            node = self._graph.get_node(neighbor_id)
            if node is None:
                continue
            self._categorize_node(node, related)

        return related

    def find_related_by_type(
        self,
        object_id: str,
        node_type: str,
    ) -> list[str]:
        """Find all related objects of a specific type.

        Args:
            object_id: The object to find related objects for
            node_type: The type of objects to find (paper, code, etc.)

        Returns:
            List of object IDs of the specified type
        """
        neighbors = self._graph.get_neighbors(object_id)
        result = []
        for neighbor_id in neighbors:
            node = self._graph.get_node(neighbor_id)
            if node and node.node_type == node_type:
                result.append(neighbor_id)
        return result

    def find_related_by_relation(
        self,
        object_id: str,
        relation_type: RelationType,
    ) -> list[str]:
        """Find all objects related by a specific relation type.

        Args:
            object_id: The object to find related objects for
            relation_type: The relation type (CITES, IMPLEMENTS, etc.)

        Returns:
            List of object IDs related by the specified relation type
        """
        return self._graph.get_related(object_id, relation_type)

    # ── Evidence Chain Methods ──────────────────────────────────────────

    def find_evidence_chain(self, claim_id: str) -> EvidenceChain:
        """Find the chain of evidence supporting or contradicting a claim.

        Args:
            claim_id: The claim (hypothesis, conclusion, finding) ID

        Returns:
            EvidenceChain with supporting and contradicting evidence IDs
        """
        chain = EvidenceChain(claim_id=claim_id)

        # Find evidence that SUPPORTS this claim
        supporting = self._graph.get_related(claim_id, RelationType.SUPPORTS)
        chain.supporting = supporting

        # Find evidence that CONTRADICTS this claim
        contradicting = self._graph.get_related(
            claim_id, RelationType.CONTRADICTS
        )
        chain.contradicting = contradicting

        return chain

    # ── Experiment Chain Methods ────────────────────────────────────────

    def find_experiment_chain(
        self,
        hypothesis_id: str,
    ) -> ExperimentChain:
        """Find the chain of experiments testing a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            ExperimentChain with experiment and finding IDs
        """
        chain = ExperimentChain(hypothesis_id=hypothesis_id)

        # Find experiments that test this hypothesis
        # (hypothesis ← TESTS ← experiment, or hypothesis ← VALIDATES ← experiment)
        experiments = self._graph.get_related(
            hypothesis_id, RelationType.RELATED_TO
        )
        # Filter to only experiment-type nodes
        for exp_id in experiments:
            node = self._graph.get_node(exp_id)
            if node and node.node_type == "experiment":
                chain.experiments.append(exp_id)

        # Find findings from these experiments
        for exp_id in chain.experiments:
            findings = self._graph.get_related(exp_id, RelationType.PRODUCES)
            chain.findings.extend(findings)

        return chain

    # ── Explainability Methods ──────────────────────────────────────────

    def explain_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find and explain the path between two objects in the graph.

        This is used for explainability — when a user asks "Why?",
        the system can show the path through the knowledge graph.

        Args:
            source_id: The source object ID
            target_id: The target object ID
            max_depth: Maximum search depth

        Returns:
            List of object IDs forming a path, or None if no path found
        """
        return self._graph.find_path(source_id, target_id, max_depth=max_depth)

    def explain_edges(
        self,
        source_id: str,
        target_id: str,
    ) -> list[KnowledgeEdge]:
        """Find all edges between two objects.

        Args:
            source_id: The source object ID
            target_id: The target object ID

        Returns:
            List of edges between the two objects
        """
        edges = []
        for edge in self._graph.get_outgoing(source_id):
            if edge.target_id == target_id:
                edges.append(edge)
        for edge in self._graph.get_incoming(target_id):
            if edge.source_id == source_id:
                edges.append(edge)
        return edges

    # ── Full Reasoning Context ──────────────────────────────────────────

    def reason(self, query: str, anchor_id: str) -> GraphReasoningContext:
        """Build a complete reasoning context from the knowledge graph.

        This is the main entry point for graph-centric reasoning. The
        planner and verification chat call this method to get a complete
        context for reasoning about a query.

        Args:
            query: The reasoning query
            anchor_id: The anchor object ID to start reasoning from

        Returns:
            GraphReasoningContext with related objects, evidence chains,
            experiment chains, and graph paths for explainability
        """
        context = GraphReasoningContext(query=query)

        # 1. Knowledge Graph — find related objects
        context.related = self.find_related(anchor_id)

        # 2. Evidence — find evidence chains for the anchor and related claims
        anchor_node = self._graph.get_node(anchor_id)
        self._fill_evidence_chains(context, anchor_id, anchor_node)

        # 3. Experiments — find experiment chains for the anchor and related hypotheses
        self._fill_experiment_chains(context, anchor_id, anchor_node)

        # 4. Graph paths — find paths between related objects for explainability
        all_related = (
            context.related.papers
            + context.related.code
            + context.related.notebooks
            + context.related.datasets
            + context.related.hypotheses
            + context.related.experiments
            + context.related.findings
        )
        for related_id in all_related[:5]:  # Limit to 5 paths
            path = self.explain_path(anchor_id, related_id)
            if path:
                context.graph_paths.append(path)

        return context

    # ── Internal Helper Methods ─────────────────────────────────────────

    def _fill_evidence_chains(
        self,
        context: GraphReasoningContext,
        anchor_id: str,
        anchor_node: KnowledgeNode | None,
    ) -> None:
        """Evidence chains for the anchor (when claim-shaped) and related."""
        if anchor_node and anchor_node.node_type in ("hypothesis", "finding", "conclusion"):
            chain = self.find_evidence_chain(anchor_id)
            context.evidence_chains.append(chain)
        for hypothesis_id in context.related.hypotheses:
            chain = self.find_evidence_chain(hypothesis_id)
            context.evidence_chains.append(chain)
        for finding_id in context.related.findings:
            chain = self.find_evidence_chain(finding_id)
            context.evidence_chains.append(chain)

    def _fill_experiment_chains(
        self,
        context: GraphReasoningContext,
        anchor_id: str,
        anchor_node: KnowledgeNode | None,
    ) -> None:
        """Experiment chains for the anchor (when a hypothesis) and related."""
        if anchor_node and anchor_node.node_type == "hypothesis":
            chain = self.find_experiment_chain(anchor_id)
            context.experiment_chains.append(chain)
        for hypothesis_id in context.related.hypotheses:
            chain = self.find_experiment_chain(hypothesis_id)
            context.experiment_chains.append(chain)

    def _categorize_node(
        self,
        node: KnowledgeNode,
        related: RelatedObjects,
    ) -> None:
        """Categorize a node into the RelatedObjects structure."""
        buckets = (
            ("paper", related.papers),
            ("code", related.code),
            ("notebook", related.notebooks),
            ("dataset", related.datasets),
            ("hypothesis", related.hypotheses),
            ("experiment", related.experiments),
            ("finding", related.findings),
            ("figure", related.figures),
            ("equation", related.equations),
        )
        for bucket_name, bucket in buckets:
            if node.node_type == bucket_name and node.node_id not in bucket:
                bucket.append(node.node_id)
                return
