"""Closed-Loop Scientific Learning — experiment results feed back into knowledge.

M3 Phase 4: Closed-Loop Scientific Learning.

This is what makes R4 an autonomous scientific research platform.

The closed loop:
  1. Experiment runs
  2. Result collected → becomes a Finding (KnowledgeObject)
  3. Finding linked to Hypothesis via KnowledgeGraph edge (SUPPORTS/CONTRADICTS)
  4. Finding embedded via UMRS Embeddings API
  5. Finding registered as Evidence in UMRS EvidenceAPI
  6. Scientific Memory updated with Finding
  7. Planner queries updated Knowledge Graph + Evidence + Memory
  8. Planner generates next hypothesis
  9. Next experiment runs
  10. Repeat

Key principle: Knowledge grows continuously through the research loop.
Every experiment result enriches the knowledge base and improves the
planner's ability to generate better hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from umrs.evidence import (
    Evidence,
    EvidenceSourceType,
)
from umrs.hypothesis_lifecycle import (
    Hypothesis,
    HypothesisLifecycle,
    HypothesisStatus,
)
from umrs.knowledge.graph_types import RelationType
from umrs.scientific_memory import (
    InMemoryScientificMemoryStore,
    MemoryEntry,
    MemoryEntryType,
)
from umrs.scientific_reasoning_engine import ScientificReasoningEngine

__all__ = [
    "ClosedLoopLearning",
    "ExperimentResult",
    "Finding",
]


@dataclass
class ExperimentResult:
    """Result of an experiment run.

    Attributes:
        result_id: Unique identifier
        experiment_id: The experiment that produced this result
        hypothesis_id: The hypothesis being tested
        metrics: Performance metrics
        status: Result status (success, failure, partial)
        artifacts: List of artifact URIs
        logs: Execution logs
        created_at: When the result was collected
    """

    result_id: str = ""
    experiment_id: str = ""
    hypothesis_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    artifacts: list[str] = field(default_factory=list)
    logs: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.result_id:
            self.result_id = f"result-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Finding:
    """A research finding derived from experiment results.

    A finding is a KnowledgeObject that represents a scientific
    conclusion derived from experimental evidence.

    Attributes:
        finding_id: Unique identifier
        hypothesis_id: The hypothesis this finding relates to
        statement: The finding statement
        evidence_ids: Evidence IDs supporting this finding
        confidence: Confidence score (0.0–1.0)
        supporting: Whether this finding supports or contradicts the hypothesis
        experiment_id: The experiment that produced this finding
        created_at: When the finding was created
    """

    finding_id: str = ""
    hypothesis_id: str = ""
    statement: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.5
    supporting: bool = True
    experiment_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.finding_id:
            self.finding_id = f"finding-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class ClosedLoopLearning:
    """Closed-loop scientific learning engine.

    This module implements the closed loop:
      Experiment → Result → Finding → KnowledgeObject
      Finding → linked to Hypothesis (SUPPORTS/CONTRADICTS)
      Finding → embedded + registered as Evidence
      Finding → Scientific Memory updated
      Planner → queries updated knowledge → next hypothesis

    Usage:
        loop = ClosedLoopLearning(
            knowledge_graph=graph,
            scientific_memory=memory_store,
            hypothesis_lifecycle=hypothesis_lifecycle,
        )

        # Process experiment result
        finding = loop.process_experiment_result(result)

        # Get next hypothesis based on updated knowledge
        next_hyp = loop.generate_next_hypothesis(anchor_object_id)
    """

    def __init__(
        self,
        *,
        knowledge_graph: Any | None = None,
        scientific_memory: InMemoryScientificMemoryStore | None = None,
        hypothesis_lifecycle: HypothesisLifecycle | None = None,
        reasoning_engine: ScientificReasoningEngine | None = None,
    ) -> None:
        if knowledge_graph is None:
            from umrs.knowledge.canonical_graph_view import default_view

            # ADR-026 migration: default to the canonical-backed view
            # (durable when the knowledge DSN resolves, mirror-only otherwise).
            knowledge_graph = default_view()
        if scientific_memory is None:
            from umrs.scientific_memory import build_default_store

            scientific_memory = build_default_store()
        if hypothesis_lifecycle is None:
            from umrs.hypothesis_lifecycle import build_default_lifecycle

            hypothesis_lifecycle = build_default_lifecycle()
        self._graph = knowledge_graph
        self._memory = scientific_memory
        self._hypothesis_lifecycle = hypothesis_lifecycle
        self._reasoning_engine = reasoning_engine or ScientificReasoningEngine()

    def process_experiment_result(
        self,
        result: ExperimentResult,
    ) -> Finding:
        """Process an experiment result through the closed loop.

        This method:
          1. Creates a Finding from the experiment result
          2. Links the Finding to the Hypothesis via KnowledgeGraph
          3. Registers the Finding as Evidence
          4. Updates Scientific Memory with the Finding
          5. Updates the Hypothesis lifecycle

        Args:
            result: The experiment result to process

        Returns:
            The Finding created from the experiment result
        """
        # 1. Create a Finding from the experiment result
        supporting = result.status == "success"
        finding = Finding(
            hypothesis_id=result.hypothesis_id,
            statement=self._generate_finding_statement(result),
            confidence=self._calculate_finding_confidence(result),
            supporting=supporting,
            experiment_id=result.experiment_id,
        )

        # 2. Link the Finding to the Hypothesis via KnowledgeGraph
        self._link_finding_to_hypothesis(finding, result, supporting)

        # 3. Register the Finding as Evidence
        evidence = self._register_finding_evidence(finding, result)
        finding.evidence_ids.append(evidence.evidence_id)

        # 4. Update Scientific Memory with the Finding
        self._record_finding_in_memory(finding, result, supporting)

        # 5. Update the Hypothesis lifecycle
        self._advance_hypothesis(result, evidence, supporting)

        return finding

    def _link_finding_to_hypothesis(
        self, finding: Finding, result: ExperimentResult, supporting: bool
    ) -> None:
        """Step 2: graph edge finding -[SUPPORTS|CONTRADICTS]-> hypothesis."""
        relation = RelationType.SUPPORTS if supporting else RelationType.CONTRADICTS
        self._graph.add_edge(
            source_id=finding.finding_id,
            target_id=result.hypothesis_id,
            relation_type=relation,
            confidence=finding.confidence,
        )

    def _register_finding_evidence(
        self, finding: Finding, result: ExperimentResult
    ) -> Evidence:
        """Step 3: Evidence item referencing the experiment result."""
        return Evidence(
            project_id="",  # Set by caller
            claim_id=result.hypothesis_id,
            source_type=EvidenceSourceType.EXPERIMENT_RESULT,
            source_id=result.experiment_id,
            content=finding.statement,
            relevance_score=finding.confidence,
            citation=f"Finding {finding.finding_id} from experiment {result.experiment_id}",
        )

    def _record_finding_in_memory(
        self, finding: Finding, result: ExperimentResult, supporting: bool
    ) -> None:
        """Step 4: scientific-memory entry for the finding."""
        memory_entry = MemoryEntry(
            project_id="",
            experiment_id=result.experiment_id,
            entry_type=MemoryEntryType.ACCEPTED_FINDING
            if supporting
            else MemoryEntryType.FAILED_EXPERIMENT,
            content=finding.statement,
            summary=f"Finding from experiment {result.experiment_id}",
            confidence=finding.confidence,
        )
        self._memory.add(memory_entry)

    def _advance_hypothesis(
        self, result: ExperimentResult, evidence: Evidence, supporting: bool
    ) -> None:
        """Step 5: hypothesis lifecycle transitions driven by the evidence."""
        hyp = self._hypothesis_lifecycle.get(result.hypothesis_id)
        if not hyp:
            return
        if hyp.status == HypothesisStatus.PROPOSED:
            self._hypothesis_lifecycle.transition(
                result.hypothesis_id,
                HypothesisStatus.TESTING,
            )
        self._hypothesis_lifecycle.add_evidence(
            result.hypothesis_id,
            evidence.evidence_id,
            supports=supporting,
        )
        hyp = self._hypothesis_lifecycle.get(result.hypothesis_id)
        if not hyp or hyp.status != HypothesisStatus.TESTING:
            return
        self._resolve_testing_hypothesis(result, hyp, supporting)

    def _resolve_testing_hypothesis(
        self, result: ExperimentResult, hyp: Hypothesis, supporting: bool
    ) -> None:
        """Terminal transition once evidence confidence crosses thresholds."""
        if supporting and hyp.confidence >= 0.8:
            self._hypothesis_lifecycle.transition(
                result.hypothesis_id,
                HypothesisStatus.SUPPORTED,
            )
        elif not supporting and hyp.confidence <= 0.2:
            self._hypothesis_lifecycle.transition(
                result.hypothesis_id,
                HypothesisStatus.CONTRADICTED,
            )

    def generate_next_hypothesis(
        self,
        anchor_object_id: str | None = None,
    ) -> str:
        """Generate the next hypothesis based on updated knowledge.

        This method:
          1. Reasons over the knowledge graph to find gaps
          2. Proposes a new hypothesis to fill the gap
          3. Returns the hypothesis_id

        Args:
            anchor_object_id: The anchor object to start reasoning from

        Returns:
            The hypothesis_id of the newly proposed hypothesis
        """
        # 1. Reason over the knowledge graph
        reasoning_result = self._reasoning_engine.reason(
            query="What should we research next?",
            anchor_object_id=anchor_object_id,
        )

        # 2. Propose a new hypothesis
        statement = self._generate_hypothesis_statement(reasoning_result)
        hyp_id = self._hypothesis_lifecycle.propose(
            statement=statement,
            planner_version="closed-loop",
        )

        return hyp_id

    def _generate_finding_statement(self, result: ExperimentResult) -> str:
        """Generate a finding statement from an experiment result."""
        metrics_str = ", ".join(f"{k}={v}" for k, v in result.metrics.items())
        return (
            f"Experiment {result.experiment_id} "
            f"{'succeeded' if result.status == 'success' else 'failed'}. "
            f"Metrics: {metrics_str}"
        )

    def _calculate_finding_confidence(self, result: ExperimentResult) -> float:
        """Calculate confidence for a finding."""
        if result.status == "success":
            return 0.8
        elif result.status == "partial":
            return 0.5
        else:
            return 0.2

    def _generate_hypothesis_statement(
        self,
        reasoning_result: Any,
    ) -> str:
        """Generate a hypothesis statement from a reasoning result."""
        # Simple heuristic: use the reasoning answer
        if reasoning_result.answer:
            return f"Based on knowledge graph analysis: {reasoning_result.answer[:200]}"
        return "New hypothesis based on knowledge graph gaps"
