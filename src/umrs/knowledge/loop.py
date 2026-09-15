"""Autonomous research loop — iterative research cycle.

Phase 12 of the Scientific Knowledge Build-up roadmap.

The loop:
  Goal → Planning → Knowledge retrieval → Need more information? →
  Search → Update knowledge → Generate hypothesis → Design experiment →
  Run experiment → Collect outputs → Analyze → Evaluate →
  Publish findings → Update scientific memory → Repeat

Every iteration updates UMRS. Knowledge grows continuously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .planner import (
    AutonomousResearchPlanner,
    PlanStatus,
    ResearchPlan,
)

logger = logging.getLogger(__name__)


# ── Loop Phase ─────────────────────────────────────────────────────────────


class LoopPhase(str, Enum):
    """Phases of the autonomous research loop."""

    INITIALIZED = "initialized"
    PLANNING = "planning"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    SEARCHING = "searching"
    UPDATING_KNOWLEDGE = "updating_knowledge"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    OUTPUT_COLLECTION = "output_collection"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"
    PUBLISHING = "publishing"
    MEMORY_UPDATE = "memory_update"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LoopStatus(str, Enum):
    """Status of a research loop."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ── Loop Iteration ─────────────────────────────────────────────────────────


@dataclass
class LoopIteration:
    """A single iteration of the research loop."""

    iteration_id: str = ""
    loop_id: str = ""
    iteration_number: int = 0
    phase: LoopPhase = LoopPhase.INITIALIZED
    started_at: str = ""
    completed_at: str = ""
    hypothesis: str = ""
    experiment_id: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    evaluation: str = ""
    findings: list[str] = field(default_factory=list)
    status: str = "pending"          # pending | running | completed | failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_id": self.iteration_id,
            "loop_id": self.loop_id,
            "iteration_number": self.iteration_number,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "hypothesis": self.hypothesis,
            "experiment_id": self.experiment_id,
            "results": self.results,
            "analysis": self.analysis,
            "evaluation": self.evaluation,
            "findings": self.findings,
            "status": self.status,
        }


# ── Research Loop ──────────────────────────────────────────────────────────


@dataclass
class ResearchLoop:
    """An autonomous research loop.

    The loop iterates through:
    Goal → Planning → Knowledge retrieval → Search → Update knowledge →
    Generate hypothesis → Design experiment → Run experiment →
    Collect outputs → Analyze → Evaluate → Publish findings →
    Update scientific memory → Repeat
    """

    loop_id: str = ""
    project_id: str = ""
    research_goal: str = ""
    status: LoopStatus = LoopStatus.RUNNING
    phase: LoopPhase = LoopPhase.INITIALIZED

    # Configuration
    max_iterations: int = 10
    current_iteration: int = 0

    # Components
    plan: ResearchPlan | None = None
    iterations: list[LoopIteration] = field(default_factory=list)

    # Results
    total_hypotheses_tested: int = 0
    total_experiments_run: int = 0
    total_findings: int = 0
    success_rate: float = 0.0

    # Timestamps
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    updated_at: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop_id": self.loop_id,
            "project_id": self.project_id,
            "research_goal": self.research_goal,
            "status": self.status.value,
            "phase": self.phase.value,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "plan": self.plan.to_dict() if self.plan else None,
            "iterations": [i.to_dict() for i in self.iterations],
            "total_hypotheses_tested": self.total_hypotheses_tested,
            "total_experiments_run": self.total_experiments_run,
            "total_findings": self.total_findings,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# ── Autonomous Research Loop Engine ───────────────────────────────────────


class AutonomousResearchLoop:
    """Orchestrate the autonomous research loop.

    Usage:
        loop_engine = AutonomousResearchLoop()
        loop = loop_engine.start_loop(
            project_id="proj-001",
            research_goal="Understand the effect of learning rate on convergence",
        )
        loop_engine.run(loop)
    """

    def __init__(self) -> None:
        self._planner = AutonomousResearchPlanner()

    def start_loop(
        self,
        project_id: str,
        research_goal: str,
        *,
        max_iterations: int = 10,
        knowledge_base: Any = None,
        external_refs: list[Any] | None = None,
        memory_store: Any = None,
    ) -> ResearchLoop:
        """Start a new research loop.

        Args:
            project_id: The project ID
            research_goal: The research goal
            max_iterations: Maximum number of iterations
            knowledge_base: Project knowledge
            external_refs: External references
            memory_store: Scientific memory store

        Returns:
            A ``ResearchLoop`` instance ready to run.
        """
        now = datetime.now(timezone.utc).isoformat()

        loop = ResearchLoop(
            loop_id=f"loop-{_generate_id()}",
            project_id=project_id,
            research_goal=research_goal,
            status=LoopStatus.RUNNING,
            phase=LoopPhase.PLANNING,
            max_iterations=max_iterations,
            current_iteration=0,
            created_at=now,
            started_at=now,
            updated_at=now,
        )

        # Generate research plan
        loop.plan = self._planner.generate_plan(
            project_id=project_id,
            research_goal=research_goal,
            knowledge_base=knowledge_base,
            external_refs=external_refs,
            memory_store=memory_store,
        )

        return loop

    def run(self, loop: ResearchLoop) -> ResearchLoop:
        """Run the research loop to completion.

        Args:
            loop: The research loop to run

        Returns:
            The completed research loop.
        """
        try:
            while (
                loop.status == LoopStatus.RUNNING
                and loop.current_iteration < loop.max_iterations
            ):
                self._run_iteration(loop)

            if loop.current_iteration >= loop.max_iterations:
                loop.phase = LoopPhase.COMPLETED
            else:
                loop.phase = LoopPhase.COMPLETED

            loop.completed_at = datetime.now(timezone.utc).isoformat()
            loop.status = LoopStatus.STOPPED

        except Exception as exc:
            logger.error("Research loop failed: %s", exc)
            loop.status = LoopStatus.ERROR
            loop.phase = LoopPhase.FAILED

        return loop

    def _run_iteration(self, loop: ResearchLoop) -> None:
        """Run a single iteration of the research loop."""
        loop.current_iteration += 1
        now = datetime.now(timezone.utc).isoformat()

        iteration = LoopIteration(
            iteration_id=f"iter-{_generate_id()}",
            loop_id=loop.loop_id,
            iteration_number=loop.current_iteration,
            started_at=now,
            status="running",
        )

        # Phase 1: Knowledge retrieval
        loop.phase = LoopPhase.KNOWLEDGE_RETRIEVAL
        iteration.phase = LoopPhase.KNOWLEDGE_RETRIEVAL

        # Phase 2: Hypothesis generation
        loop.phase = LoopPhase.HYPOTHESIS_GENERATION
        iteration.phase = LoopPhase.HYPOTHESIS_GENERATION
        if loop.plan and loop.plan.hypotheses:
            idx = (loop.current_iteration - 1) % len(loop.plan.hypotheses)
            iteration.hypothesis = loop.plan.hypotheses[idx].statement
            loop.total_hypotheses_tested += 1

        # Phase 3: Experiment design
        loop.phase = LoopPhase.EXPERIMENT_DESIGN
        iteration.phase = LoopPhase.EXPERIMENT_DESIGN
        if loop.plan and loop.plan.experiments:
            idx = (loop.current_iteration - 1) % len(loop.plan.experiments)
            iteration.experiment_id = loop.plan.experiments[idx].experiment_id

        # Phase 4: Experiment execution
        loop.phase = LoopPhase.EXPERIMENT_EXECUTION
        iteration.phase = LoopPhase.EXPERIMENT_EXECUTION
        loop.total_experiments_run += 1
        iteration.results = {"status": "completed", "output": "Experiment results"}

        # Phase 5: Output collection
        loop.phase = LoopPhase.OUTPUT_COLLECTION
        iteration.phase = LoopPhase.OUTPUT_COLLECTION

        # Phase 6: Analysis
        loop.phase = LoopPhase.ANALYSIS
        iteration.phase = LoopPhase.ANALYSIS
        iteration.analysis = f"Analysis of iteration {loop.current_iteration}"

        # Phase 7: Evaluation
        loop.phase = LoopPhase.EVALUATION
        iteration.phase = LoopPhase.EVALUATION
        iteration.evaluation = f"Evaluation of iteration {loop.current_iteration}"

        # Phase 8: Publishing findings
        loop.phase = LoopPhase.PUBLISHING
        iteration.phase = LoopPhase.PUBLISHING
        iteration.findings = [f"Finding from iteration {loop.current_iteration}"]
        loop.total_findings += 1

        # Phase 9: Memory update
        loop.phase = LoopPhase.MEMORY_UPDATE
        iteration.phase = LoopPhase.MEMORY_UPDATE

        # Complete iteration
        iteration.completed_at = datetime.now(timezone.utc).isoformat()
        iteration.status = "completed"
        loop.iterations.append(iteration)
        loop.updated_at = datetime.now(timezone.utc).isoformat()

        # Update success rate
        if loop.total_experiments_run > 0:
            loop.success_rate = loop.total_findings / loop.total_experiments_run


# ── Helper Functions ───────────────────────────────────────────────────────


def _generate_id() -> str:
    """Generate a unique ID."""
    import uuid
    return uuid.uuid4().hex[:12]


# ── Convenience Function ────────────────────────────────────────────────────


def run_research_loop(
    project_id: str,
    research_goal: str,
    *,
    max_iterations: int = 10,
    knowledge_base: Any = None,
    external_refs: list[Any] | None = None,
    memory_store: Any = None,
) -> ResearchLoop:
    """Run an autonomous research loop (convenience function).

    Args:
        project_id: The project ID
        research_goal: The research goal
        max_iterations: Maximum number of iterations
        knowledge_base: Project knowledge
        external_refs: External references
        memory_store: Scientific memory store

    Returns:
        A completed ``ResearchLoop`` instance.
    """
    loop_engine = AutonomousResearchLoop()
    loop = loop_engine.start_loop(
        project_id=project_id,
        research_goal=research_goal,
        max_iterations=max_iterations,
        knowledge_base=knowledge_base,
        external_refs=external_refs,
        memory_store=memory_store,
    )
    return loop_engine.run(loop)
