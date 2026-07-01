"""Scientific Reasoning Engine — separates reasoning from orchestration.

M3 Phase 3 Addition: Scientific Reasoning Engine.

The planner decides WHAT needs to be achieved.
The reasoning engine determines WHY and HOW based on scientific knowledge.

Architecture:

  Planner (orchestration)
    ↓
  Scientific Reasoning Engine (reasoning)
    ↓
  Knowledge Graph → Evidence → Memory → Experiments → Catalog → RAG
    ↓
  Reasoning Result (with evidence, confidence, and graph paths)

The reasoning engine is reusable:
  - Planner uses it to generate hypotheses
  - Verification Chat uses it to answer questions
  - Critique agents use it to evaluate hypotheses
  - Review agents use it to assess findings
  - Hypothesis refinement uses it to improve hypotheses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from umrs.graph_reasoning import GraphNavigator, GraphReasoningContext

__all__ = [
    "ReasoningResult",
    "ScientificReasoningEngine",
]


# ── Reasoning Result ─────────────────────────────────────────────────────


@dataclass
class ReasoningResult:
    """Result of scientific reasoning.

    Attributes:
        reasoning_id: Unique identifier for this reasoning result
        query: The reasoning query
        answer: The reasoning answer (ephemeral, NOT persisted)
        evidence_ids: UMRS object IDs cited as evidence
        graph_paths: Paths through the knowledge graph
        confidence: Confidence score (0.0–1.0)
        reasoning_context: Complete reasoning context from the graph
        reasoning_steps: Step-by-step reasoning trace
        timestamp: When the reasoning was performed
    """

    reasoning_id: str = ""
    query: str = ""
    answer: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    graph_paths: list[list[str]] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_context: GraphReasoningContext | None = None
    reasoning_steps: list[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.reasoning_id:
            self.reasoning_id = f"reasoning-{uuid4().hex[:12]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_persistent_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistent storage.

        This excludes the ephemeral answer and reasoning context.
        """
        return {
            "reasoning_id": self.reasoning_id,
            "query": self.query,
            "evidence_ids": self.evidence_ids,
            "graph_paths": self.graph_paths,
            "confidence": self.confidence,
            "reasoning_steps": self.reasoning_steps,
            "timestamp": self.timestamp,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including ephemeral data."""
        data = self.to_persistent_dict()
        data["answer"] = self.answer
        return data


# ── Scientific Reasoning Engine ──────────────────────────────────────────


class ScientificReasoningEngine:
    """Engine for scientific reasoning over the knowledge graph.

    The engine is reusable across multiple subsystems:
      - Planner: generate hypotheses from graph reasoning
      - Verification Chat: answer questions from graph reasoning
      - Critique agents: evaluate hypotheses from graph reasoning
      - Review agents: assess findings from graph reasoning
      - Hypothesis refinement: improve hypotheses from graph reasoning

    The engine follows the M3 Decision 4 priority order:
      1. Knowledge Graph    — what is related?
      2. Evidence           — what supports this?
      3. Scientific Memory  — what has been done?
      4. Experiments        — what has been tested?
      5. Catalog           — what artifacts exist?
      6. RAG               — what is in the literature?
      7. LLM               — generate answer (not implemented in this engine)
    """

    def __init__(
        self,
        navigator: GraphNavigator | None = None,
    ) -> None:
        self._navigator = navigator or GraphNavigator()

    @property
    def navigator(self) -> GraphNavigator:
        """The graph navigator used for reasoning."""
        return self._navigator

    def reason(
        self,
        query: str,
        *,
        anchor_object_id: str | None = None,
    ) -> ReasoningResult:
        """Perform scientific reasoning over the knowledge graph.

        Args:
            query: The reasoning query
            anchor_object_id: The anchor object to start reasoning from
                (if None, uses the first paper in the graph)

        Returns:
            A ReasoningResult with evidence, confidence, and graph paths

        The reasoning follows the priority order:
          1. Knowledge Graph — find related objects
          2. Evidence — find evidence chains
          3. Experiments — find experiment chains
          4. Generate reasoning steps
          5. Calculate confidence
        """
        # If no anchor specified, try to find one
        if anchor_object_id is None:
            all_nodes = self._navigator.graph.query()
            for node in all_nodes:
                if node.node_type == "paper":
                    anchor_object_id = node.node_id
                    break

        if anchor_object_id is None:
            # No anchor found — return empty result
            return ReasoningResult(
                query=query,
                answer="No knowledge graph data available for reasoning.",
                confidence=0.0,
                reasoning_steps=["No anchor object found in the knowledge graph."],
            )

        # 1. Knowledge Graph — find related objects through the graph
        reasoning_context = self._navigator.reason(
            query=query,
            anchor_id=anchor_object_id,
        )

        # 2. Evidence — find evidence chains for related claims
        evidence_ids: list[str] = []
        for chain in reasoning_context.evidence_chains:
            evidence_ids.extend(chain.supporting)
            evidence_ids.extend(chain.contradicting)

        # 3. Experiments — find experiment chains for related hypotheses
        experiment_count = sum(
            len(chain.experiments) for chain in reasoning_context.experiment_chains
        )

        # 4. Generate reasoning steps
        reasoning_steps = self._generate_reasoning_steps(
            query, reasoning_context, evidence_ids, experiment_count
        )

        # 5. Calculate confidence
        confidence = self._calculate_confidence(
            reasoning_context, evidence_ids, experiment_count
        )

        # 6. Generate graph paths for explainability
        graph_paths = reasoning_context.graph_paths

        # 7. Generate answer (without LLM — use reasoning steps)
        answer = self._generate_answer(
            query, reasoning_context, reasoning_steps, confidence
        )

        return ReasoningResult(
            query=query,
            answer=answer,
            evidence_ids=evidence_ids,
            graph_paths=graph_paths,
            confidence=confidence,
            reasoning_context=reasoning_context,
            reasoning_steps=reasoning_steps,
        )

    def reason_about_hypothesis(
        self,
        hypothesis_id: str,
    ) -> ReasoningResult:
        """Reason about a specific hypothesis.

        Args:
            hypothesis_id: The hypothesis ID

        Returns:
            A ReasoningResult with evidence for/against the hypothesis
        """
        query = f"What evidence supports or contradicts hypothesis {hypothesis_id}?"

        # Find evidence chain for this hypothesis
        evidence_chain = self._navigator.find_evidence_chain(hypothesis_id)

        # Find experiment chain for this hypothesis
        experiment_chain = self._navigator.find_experiment_chain(hypothesis_id)

        # Generate reasoning steps
        reasoning_steps = [
            f"Analyzing hypothesis: {hypothesis_id}",
            f"Supporting evidence: {len(evidence_chain.supporting)} items",
            f"Contradicting evidence: {len(evidence_chain.contradicting)} items",
            f"Related experiments: {len(experiment_chain.experiments)} items",
            f"Related findings: {len(experiment_chain.findings)} items",
        ]

        # Calculate confidence
        supporting_count = len(evidence_chain.supporting)
        contradicting_count = len(evidence_chain.contradicting)
        total_evidence = supporting_count + contradicting_count
        if total_evidence > 0:
            confidence = supporting_count / total_evidence
        else:
            confidence = 0.5

        # Generate answer
        answer = (
            f"Hypothesis {hypothesis_id} has {supporting_count} supporting "
            f"and {contradicting_count} contradicting evidence items. "
            f"Confidence: {confidence:.2f}."
        )

        return ReasoningResult(
            query=query,
            answer=answer,
            evidence_ids=evidence_chain.supporting + evidence_chain.contradicting,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
        )

    def reason_about_project(
        self,
        anchor_object_id: str,
    ) -> ReasoningResult:
        """Reason about the entire project from an anchor object.

        Args:
            anchor_object_id: The anchor object (typically a paper)

        Returns:
            A ReasoningResult with a comprehensive project understanding
        """
        query = "Explain this project to a new researcher."

        # Get full reasoning context
        reasoning_context = self._navigator.reason(
            query=query,
            anchor_id=anchor_object_id,
        )

        # Generate reasoning steps
        reasoning_steps = [
            f"Starting from object: {anchor_object_id}",
            f"Found {reasoning_context.related.total()} related objects:",
            f"  - Papers: {len(reasoning_context.related.papers)}",
            f"  - Code: {len(reasoning_context.related.code)}",
            f"  - Notebooks: {len(reasoning_context.related.notebooks)}",
            f"  - Datasets: {len(reasoning_context.related.datasets)}",
            f"  - Hypotheses: {len(reasoning_context.related.hypotheses)}",
            f"  - Experiments: {len(reasoning_context.related.experiments)}",
            f"  - Findings: {len(reasoning_context.related.findings)}",
            f"  - Figures: {len(reasoning_context.related.figures)}",
            f"  - Equations: {len(reasoning_context.related.equations)}",
            f"Found {len(reasoning_context.evidence_chains)} evidence chains",
            f"Found {len(reasoning_context.experiment_chains)} experiment chains",
        ]

        # Calculate confidence
        confidence = self._calculate_confidence(
            reasoning_context,
            reasoning_context.related.papers
            + reasoning_context.related.code
            + reasoning_context.related.notebooks,
            len(reasoning_context.related.experiments),
        )

        # Generate answer
        answer = self._generate_project_summary(
            reasoning_context, reasoning_steps, confidence
        )

        return ReasoningResult(
            query=query,
            answer=answer,
            evidence_ids=reasoning_context.related.papers
            + reasoning_context.related.code
            + reasoning_context.related.notebooks
            + reasoning_context.related.datasets,
            graph_paths=reasoning_context.graph_paths,
            confidence=confidence,
            reasoning_context=reasoning_context,
            reasoning_steps=reasoning_steps,
        )

    # ── Internal Helper Methods ─────────────────────────────────────────

    def _generate_reasoning_steps(
        self,
        query: str,
        context: GraphReasoningContext,
        evidence_ids: list[str],
        experiment_count: int,
    ) -> list[str]:
        """Generate step-by-step reasoning trace."""
        steps = [
            f"Query: {query}",
            f"Step 1: Knowledge Graph — found {context.related.total()} related objects",
            f"  - Papers: {len(context.related.papers)}",
            f"  - Code: {len(context.related.code)}",
            f"  - Notebooks: {len(context.related.notebooks)}",
            f"  - Datasets: {len(context.related.datasets)}",
            f"  - Hypotheses: {len(context.related.hypotheses)}",
            f"  - Experiments: {len(context.related.experiments)}",
            f"  - Findings: {len(context.related.findings)}",
            f"  - Figures: {len(context.related.figures)}",
            f"  - Equations: {len(context.related.equations)}",
            f"Step 2: Evidence — found {len(evidence_ids)} evidence items",
            f"Step 3: Experiments — found {experiment_count} related experiments",
            f"Step 4: Graph Paths — found {len(context.graph_paths)} paths for explainability",
        ]
        return steps

    def _calculate_confidence(
        self,
        context: GraphReasoningContext,
        evidence_ids: list[str],
        experiment_count: int,
    ) -> float:
        """Calculate confidence score based on evidence and experiments."""
        # Base confidence from evidence
        evidence_factor = min(1.0, len(evidence_ids) / 10.0)

        # Factor from experiments
        experiment_factor = min(1.0, experiment_count / 5.0)

        # Factor from related objects
        related_factor = min(1.0, context.related.total() / 10.0)

        # Weighted combination
        confidence = (
            evidence_factor * 0.4
            + experiment_factor * 0.3
            + related_factor * 0.3
        )

        return min(1.0, confidence)

    def _generate_answer(
        self,
        query: str,
        context: GraphReasoningContext,
        reasoning_steps: list[str],
        confidence: float,
    ) -> str:
        """Generate a reasoning answer without LLM."""
        answer_parts = [
            f"Query: {query}",
            f"Related objects: {context.related.total()}",
            f"Evidence chains: {len(context.evidence_chains)}",
            f"Experiment chains: {len(context.experiment_chains)}",
            f"Graph paths: {len(context.graph_paths)}",
            f"Confidence: {confidence:.2f}",
        ]
        return "\n".join(answer_parts)

    def _generate_project_summary(
        self,
        context: GraphReasoningContext,
        reasoning_steps: list[str],
        confidence: float,
    ) -> str:
        """Generate a project summary."""
        summary_parts = [
            "Project Summary:",
            f"  Papers: {len(context.related.papers)}",
            f"  Code: {len(context.related.code)}",
            f"  Notebooks: {len(context.related.notebooks)}",
            f"  Datasets: {len(context.related.datasets)}",
            f"  Hypotheses: {len(context.related.hypotheses)}",
            f"  Experiments: {len(context.related.experiments)}",
            f"  Findings: {len(context.related.findings)}",
            f"  Figures: {len(context.related.figures)}",
            f"  Equations: {len(context.related.equations)}",
            f"  Evidence chains: {len(context.evidence_chains)}",
            f"  Experiment chains: {len(context.experiment_chains)}",
            f"  Graph paths: {len(context.graph_paths)}",
            f"  Confidence: {confidence:.2f}",
        ]
        return "\n".join(summary_parts)
