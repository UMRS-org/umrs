"""ResearchProject — the central object for end-to-end scientific workflow.

RC1 Milestone M1: One object that owns the entire scientific workflow.

The ResearchProject is the single entry point that connects:
  - Knowledge ingestion (KnowledgeIngester)
  - Knowledge extraction (KnowledgeExtractor)
  - Knowledge linking (KnowledgeLinker → KnowledgeGraph)
  - Knowledge verification (KnowledgeVerifier)
  - Knowledge coverage (CoverageCalculator → CoverageDashboard)
  - External knowledge (ExternalKnowledgeSearcher)
  - Scientific memory (ScientificMemoryManager)
  - Research readiness (ResearchReadinessGate → ResearchReadinessReport)
  - Research planning (AutonomousResearchPlanner → ResearchPlan)
  - Research loop (AutonomousResearchLoop → ResearchLoop)

Usage:
    project = ResearchProject(project_id="proj-001", name="My Project")
    project.ingest_directory("./papers", "./notebooks", "./code", "./data")
    project.understand()
    answer = project.verify("Explain my project")
    report = project.check_readiness()
    plan = project.plan_research("What research should we perform next?")
    loop = project.run_research()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .coverage import (
    CoverageCalculator,
    CoverageDashboard,
)
from .external import ExternalKnowledgeSearcher
from .extract import ExtractedKnowledge, KnowledgeExtractor
from .link import KnowledgeGraph, KnowledgeLinker
from .loop import AutonomousResearchLoop, ResearchLoop
from .memory import ScientificMemoryManager, ScientificMemoryStore
from .planner import AutonomousResearchPlanner, ResearchPlan
from .readiness import (
    ResearchReadinessGate,
    ResearchReadinessReport,
)
from .verify import (
    InMemoryKnowledgeStore,
    KnowledgeStore,
    KnowledgeVerifier,
    VerificationAnswer,
)

logger = logging.getLogger(__name__)


# ── Workflow Phase ─────────────────────────────────────────────────────────


class ProjectPhase(str, Enum):
    """Lifecycle phases of a ResearchProject."""

    CREATED = "created"
    INGESTING = "ingesting"
    UNDERSTANDING = "understanding"
    READY = "ready"                       # Knowledge verified, ready for research
    PLANNING = "planning"
    RESEARCHING = "researching"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Transition Log ─────────────────────────────────────────────────────────


@dataclass
class WorkflowTransition:
    """One stage transition in the project workflow.

    Used for audit trail, debugging, and replay. Every phase change is
    logged with its inputs, outputs, duration, and success/failure.
    """

    from_phase: ProjectPhase
    to_phase: ProjectPhase
    timestamp: str = ""
    duration_seconds: float = 0.0
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "success": self.success,
            "error": self.error,
        }


# ── ResearchProject ─────────────────────────────────────────────────────────


class ResearchProject:
    """One scientific project — the center of R4.

    Owns:
      - Knowledge base (papers, code, notebooks, datasets) via KnowledgeStore
      - Knowledge graph (typed relationships) via KnowledgeGraph
      - Scientific memory (auto-registered artifacts) via ScientificMemoryManager
      - Research plan (objectives, hypotheses, experiments) via ResearchPlan
      - Research loop (autonomous research execution) via ResearchLoop
      - Coverage dashboard (knowledge completeness) via CoverageDashboard
      - Research readiness report (gate before autonomous research)

    Provides one entry point for each workflow stage:
      - `ingest_directory()` — ingest files from disk
      - `understand()` — extract knowledge and build knowledge graph
      - `verify(question)` — answer questions with evidence
      - `check_readiness()` — generate Research Readiness Report
      - `plan_research(goal)` — generate research plan
      - `run_research()` — execute research loop

    Every transition is logged for audit trail and replay.
    """

    def __init__(
        self,
        project_id: str,
        name: str = "",
        description: str = "",
        *,
        storage_path: str | Path | None = None,
    ) -> None:
        self.project_id = project_id
        self.name = name or project_id
        self.description = description
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.phase: ProjectPhase = ProjectPhase.CREATED

        # ── Components ──────────────────────────────────────────────────
        # Knowledge store (papers, code, notebooks, datasets) — in-memory
        self._knowledge_store: KnowledgeStore = InMemoryKnowledgeStore()

        # Knowledge graph (typed relationships)
        self._knowledge_graph: KnowledgeGraph = KnowledgeGraph()

        # Extracted knowledge per artifact_id
        self._extracted_knowledge: dict[str, ExtractedKnowledge] = {}

        # Scientific memory (auto-registered artifacts)
        memory_storage = (
            Path(storage_path) / "memory.json" if storage_path else None
        )
        self._memory_store = ScientificMemoryStore(storage_path=memory_storage)
        self._memory_store.load()
        self._memory_manager = ScientificMemoryManager(store=self._memory_store)

        # Coverage dashboard (knowledge completeness)
        self._coverage_dashboard: CoverageDashboard | None = None

        # Research readiness report (gate before autonomous research)
        self._readiness_report: ResearchReadinessReport | None = None

        # Research plan (objectives, hypotheses, experiments)
        self._research_plan: ResearchPlan | None = None

        # Research loop (autonomous research execution)
        self._research_loop: ResearchLoop | None = None

        # ── Engine instances (created on demand) ─────────────────────────
        self._extractor = KnowledgeExtractor()
        self._linker = KnowledgeLinker()
        self._coverage_calculator = CoverageCalculator()
        self._external_searcher = ExternalKnowledgeSearcher()
        self._readiness_gate = ResearchReadinessGate()
        self._planner = AutonomousResearchPlanner()
        self._loop_engine = AutonomousResearchLoop()

        # ── Workflow audit trail ─────────────────────────────────────────
        self._transitions: list[WorkflowTransition] = []

        # ── External knowledge references ────────────────────────────────
        self._external_refs: list[Any] = []

    # ── Public API: Workflow Stages ─────────────────────────────────────────

    def ingest_directory(self, *directories: str | Path) -> int:
        """Ingest all supported files from one or more directories.

        Walks each directory recursively and registers every recognized
        file (PDF, notebook, code, dataset, README) as an artifact in the
        knowledge store. Returns the number of artifacts ingested.

        Phase transition: CREATED → INGESTING → UNDERSTANDING (via understand())
        """
        self._transition(ProjectPhase.CREATED, ProjectPhase.INGESTING)
        start = datetime.now(timezone.utc)

        count = 0
        for directory in directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                logger.warning("Directory does not exist: %s", dir_path)
                continue
            for file_path in sorted(dir_path.rglob("*")):
                if not file_path.is_file():
                    continue
                if self._ingest_file(file_path):
                    count += 1

        self.updated_at = datetime.now(timezone.utc).isoformat()
        self._log_transition(
            ProjectPhase.INGESTING,
            ProjectPhase.INGESTING,
            start,
            inputs={"directories": [str(d) for d in directories]},
            outputs={"artifacts_ingested": count},
        )
        return count

    def understand(self) -> ExtractedKnowledge:
        """Extract structured knowledge from all artifacts and build the knowledge graph.

        For each artifact:
          1. Extract structured scientific information (objective, methodology, etc.)
          2. Store the ExtractedKnowledge for later queries

        Then:
          3. Link related artifacts into a KnowledgeGraph
          4. Calculate coverage dashboard
          5. Transition to READY phase

        Returns the last ExtractedKnowledge (or empty if no artifacts).
        """
        if self.phase == ProjectPhase.CREATED:
            # Allow understand() without ingest for empty projects
            pass
        self._transition(
            self.phase if self.phase != ProjectPhase.CREATED else ProjectPhase.INGESTING,
            ProjectPhase.UNDERSTANDING,
        )
        start = datetime.now(timezone.utc)

        artifacts = self._knowledge_store.list_artifacts()
        last_knowledge: ExtractedKnowledge | None = None

        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id", "")
            content = artifact.get("content", b"")
            if isinstance(content, str):
                content = content.encode("utf-8")
            filename = artifact.get("filename", "")
            artifact_type = artifact.get("artifact_type", "text")

            knowledge = self._extractor.extract(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                filename=filename,
                content=content,
            )
            self._extracted_knowledge[artifact_id] = knowledge
            self._knowledge_store.add_extracted_knowledge(artifact_id, knowledge.to_dict())
            last_knowledge = knowledge

            # Register extracted knowledge in scientific memory
            self._memory_manager.register_analysis_result(
                project_id=self.project_id,
                analysis=knowledge.summary or f"Extracted knowledge for {filename}",
                summary=f"Knowledge extracted from {filename}",
            )

        # Build knowledge graph from artifacts + extracted knowledge
        artifact_dicts = [dict(a, extracted_knowledge=self._extracted_knowledge.get(a.get("artifact_id", ""))) for a in artifacts]
        self._knowledge_graph = self._linker.link(artifact_dicts)

        # Calculate coverage dashboard
        coverage_artifacts = []
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id", "")
            knowledge = self._extracted_knowledge.get(artifact_id)
            coverage_artifacts.append({
                "artifact_id": artifact_id,
                "artifact_type": artifact.get("artifact_type", "text"),
                "filename": artifact.get("filename", ""),
                "metadata": artifact.get("metadata", {}),
                "extracted_knowledge": knowledge,
            })
        self._coverage_dashboard = self._coverage_calculator.calculate(coverage_artifacts)

        self.updated_at = datetime.now(timezone.utc).isoformat()
        self._log_transition(
            ProjectPhase.UNDERSTANDING,
            ProjectPhase.UNDERSTANDING,
            start,
            inputs={"artifact_count": len(artifacts)},
            outputs={
                "extracted_count": len(self._extracted_knowledge),
                "graph_nodes": len(self._knowledge_graph.nodes),
                "graph_edges": len(self._knowledge_graph.edges),
                "coverage_score": self._coverage_dashboard.coverage_score if self._coverage_dashboard else 0.0,
            },
        )
        # Transition to READY after understanding
        self._transition(ProjectPhase.UNDERSTANDING, ProjectPhase.READY)
        return last_knowledge or ExtractedKnowledge(
            artifact_id="",
            artifact_type="text",
            filename="",
        )

    def verify(self, question: str) -> VerificationAnswer:
        """Answer a knowledge verification question with evidence.

        The question is routed to the appropriate handler based on its
        content. Answers cite supporting evidence (uploaded files, notebook
        cells, source code, datasets, experiment results) and report
        confidence and missing information.

        Examples:
          - "Explain my project"
          - "What algorithms are implemented?"
          - "Which notebook produced Figure 5?"
          - "Which code implements Equation 4?"
          - "What assumptions are unsupported?"
        """
        verifier = KnowledgeVerifier(store=self._knowledge_store)
        return verifier.verify(question)

    def check_readiness(self) -> ResearchReadinessReport:
        """Generate a Research Readiness Report.

        Evaluates:
          - Knowledge completeness (0.0–1.0)
          - Knowledge confidence (0.0–1.0)
          - Missing datasets, references, implementations
          - Unknown assumptions

        Returns a ResearchReadinessReport with:
          - readiness_level: READY | PARTIALLY_READY | NOT_READY | BLOCKED
          - confidence_level: HIGH | MEDIUM | LOW | UNKNOWN
          - issues: list of ReadinessIssue
          - recommendation: human-readable recommendation
          - next_steps: list of next steps

        The user must approve this report before autonomous research begins.
        """
        if self._coverage_dashboard is None:
            self.understand()

        start = datetime.now(timezone.utc)
        self._readiness_report = self._readiness_gate.evaluate(
            project_id=self.project_id,
            coverage_dashboard=self._coverage_dashboard,
            memory_store=self._memory_store,
        )

        self._log_transition(
            ProjectPhase.READY,
            ProjectPhase.READY,
            start,
            inputs={"coverage_score": self._coverage_dashboard.coverage_score if self._coverage_dashboard else 0.0},
            outputs={
                "readiness_level": self._readiness_report.readiness_level.value,
                "confidence_level": self._readiness_report.confidence_level.value,
                "critical_issues": self._readiness_report.critical_issue_count,
            },
        )
        return self._readiness_report

    def approve_readiness(self, user_id: str = "user") -> bool:
        """User approves the Research Readiness Report.

        Autonomous research cannot begin until the user explicitly approves
        the report. Returns True if approved, False if no report exists.
        """
        if self._readiness_report is None:
            return False
        self._readiness_report.user_approved = True
        self._readiness_report.approved_at = datetime.now(timezone.utc).isoformat()
        self._readiness_report.approved_by = user_id
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def search_external(self, query: str, max_results_per_source: int = 5) -> list[Any]:
        """Search external knowledge sources (arXiv, Semantic Scholar, CrossRef).

        Stores the results as external references for later use by the
        planner. Returns a list of ExternalSearchResult objects.
        """
        results = self._external_searcher.search_all(
            query,
            max_results_per_source=max_results_per_source,
        )
        for result in results:
            for ref in result.references:
                self._external_refs.append(ref)
        return results

    def plan_research(self, research_goal: str) -> ResearchPlan:
        """Generate a research plan for the project.

        Uses:
          - Project knowledge (papers, code, notebooks, datasets)
          - External knowledge (arXiv, Semantic Scholar, CrossRef)
          - Scientific memory (auto-registered artifacts)

        Generates:
          - Research objectives (Understanding, Verification, Optimization)
          - Hypotheses with confidence and expected outcomes
          - Experiments with parameters, resources, and validation methods
          - Expected outcomes
          - Success probability

        Phase transition: READY → PLANNING → RESEARCHING (via run_research())
        """
        self._transition(ProjectPhase.READY, ProjectPhase.PLANNING)
        start = datetime.now(timezone.utc)

        self._research_plan = self._planner.generate_plan(
            project_id=self.project_id,
            research_goal=research_goal,
            knowledge_base=self._knowledge_store.list_artifacts(),
            external_refs=self._external_refs,
            memory_store=self._memory_store,
        )

        self._log_transition(
            ProjectPhase.PLANNING,
            ProjectPhase.PLANNING,
            start,
            inputs={"research_goal": research_goal},
            outputs={
                "objectives": len(self._research_plan.objectives),
                "hypotheses": len(self._research_plan.hypotheses),
                "experiments": len(self._research_plan.experiments),
                "success_probability": self._research_plan.success_probability,
            },
        )
        return self._research_plan

    def run_research(self, max_iterations: int = 10) -> ResearchLoop:
        """Execute the autonomous research loop.

        Requires:
          - Research readiness report approved (user_approved=True)
          - Research plan generated (plan_research() called)

        The loop iterates through:
          Planning → Knowledge Retrieval → Hypothesis Generation →
          Experiment Design → Execution → Output Collection →
          Analysis → Evaluation → Publishing → Memory Update → Repeat

        Every iteration updates scientific memory. Knowledge grows
        continuously.

        Phase transition: PLANNING → RESEARCHING → COMPLETED
        """
        if self._readiness_report is None or not self._readiness_report.user_approved:
            raise RuntimeError(
                "Research readiness report must be approved before running research. "
                "Call check_readiness() and approve_readiness() first."
            )
        if self._research_plan is None:
            raise RuntimeError(
                "Research plan must be generated before running research. "
                "Call plan_research() first."
            )

        self._transition(ProjectPhase.PLANNING, ProjectPhase.RESEARCHING)
        start = datetime.now(timezone.utc)

        self._research_loop = self._loop_engine.start_loop(
            project_id=self.project_id,
            research_goal=self._research_plan.title or "Autonomous research",
            max_iterations=max_iterations,
        )
        self._research_loop = self._loop_engine.run(self._research_loop)

        # Register research loop results in scientific memory
        for iteration in self._research_loop.iterations:
            self._memory_manager.register_analysis_result(
                project_id=self.project_id,
                analysis=iteration.analysis or "",
                summary=f"Analysis from iteration {iteration.iteration_number}",
            )
            for finding in iteration.findings:
                self._memory_manager.register_accepted_finding(
                    project_id=self.project_id,
                    finding=finding,
                    confidence=0.7,
                )

        # Log the research phase timing (before final transition)
        self._log_transition(
            ProjectPhase.RESEARCHING,
            ProjectPhase.RESEARCHING,
            start,
            inputs={"max_iterations": max_iterations},
            outputs={
                "iterations": len(self._research_loop.iterations),
                "hypotheses_tested": self._research_loop.total_hypotheses_tested,
                "experiments_run": self._research_loop.total_experiments_run,
                "findings": self._research_loop.total_findings,
                "success_rate": self._research_loop.success_rate,
            },
        )
        # Transition to COMPLETED (this is the last transition)
        self._transition(ProjectPhase.RESEARCHING, ProjectPhase.COMPLETED)
        return self._research_loop

    # ── Public API: Accessors ────────────────────────────────────────────────

    @property
    def knowledge_graph(self) -> KnowledgeGraph:
        return self._knowledge_graph

    @property
    def coverage_dashboard(self) -> CoverageDashboard | None:
        return self._coverage_dashboard

    @property
    def readiness_report(self) -> ResearchReadinessReport | None:
        return self._readiness_report

    @property
    def research_plan(self) -> ResearchPlan | None:
        return self._research_plan

    @property
    def research_loop(self) -> ResearchLoop | None:
        return self._research_loop

    @property
    def memory_manager(self) -> ScientificMemoryManager:
        return self._memory_manager

    @property
    def knowledge_store(self) -> KnowledgeStore:
        return self._knowledge_store

    @property
    def extracted_knowledge(self) -> dict[str, ExtractedKnowledge]:
        return self._extracted_knowledge

    @property
    def external_refs(self) -> list[Any]:
        return self._external_refs

    @property
    def transitions(self) -> list[WorkflowTransition]:
        return self._transitions

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase.value,
            "artifact_count": len(self._knowledge_store.list_artifacts()),
            "extracted_count": len(self._extracted_knowledge),
            "graph_nodes": len(self._knowledge_graph.nodes),
            "graph_edges": len(self._knowledge_graph.edges),
            "coverage_score": self._coverage_dashboard.coverage_score if self._coverage_dashboard else 0.0,
            "readiness_level": self._readiness_report.readiness_level.value if self._readiness_report else "unknown",
            "readiness_approved": self._readiness_report.user_approved if self._readiness_report else False,
            "has_research_plan": self._research_plan is not None,
            "has_research_loop": self._research_loop is not None,
            "external_ref_count": len(self._external_refs),
            "memory_entry_count": len(self._memory_store._entries),
            "transition_count": len(self._transitions),
        }

    # ── Internal: File Ingestion ─────────────────────────────────────────────

    def _ingest_file(self, file_path: Path) -> bool:
        """Ingest a single file. Returns True if recognized and ingested."""
        try:
            content = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return False

        # Determine artifact_type from filename
        suffix = file_path.suffix.lower()
        filename = file_path.name

        if suffix == ".ipynb":
            artifact_type = "notebook"
        elif suffix in {".py", ".pyx", ".cpp", ".cc", ".cu", ".cuh", ".rs", ".go", ".rb", ".js", ".ts"}:
            artifact_type = "code"
        elif suffix in {".pdf", ".pptx", ".ppt"}:
            artifact_type = "paper"
        elif suffix in {".md", ".txt", ".rst"}:
            # README.md and other text files
            if filename.lower().startswith("readme"):
                artifact_type = "text"
            else:
                artifact_type = "text"
        elif suffix in {".csv", ".tsv", ".json", ".yaml", ".yml"}:
            artifact_type = "dataset"
        elif suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp"}:
            artifact_type = "image"
        else:
            # Unknown type — skip for now
            return False

        artifact_id = f"{artifact_type}-{file_path.stem}-{hash(str(file_path)) & 0xFFFFFF:X}"

        # Add to knowledge store
        self._knowledge_store.add_artifact({
            "artifact_id": artifact_id,
            "artifact_type": artifact_type,
            "filename": filename,
            "content": content,
            "metadata": {
                "path": str(file_path),
                "size_bytes": len(content),
            },
        })
        return True

    # ── Internal: Phase Transitions ──────────────────────────────────────────

    def _transition(self, from_phase: ProjectPhase, to_phase: ProjectPhase) -> None:
        """Transition to a new phase, logging the transition."""
        if from_phase == to_phase:
            return
        self.phase = to_phase
        self._transitions.append(WorkflowTransition(
            from_phase=from_phase,
            to_phase=to_phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        logger.info("Project %s: %s → %s", self.project_id, from_phase.value, to_phase.value)

    def _log_transition(
        self,
        from_phase: ProjectPhase,
        to_phase: ProjectPhase,
        start: datetime,
        *,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Log a workflow transition with timing and I/O capture."""
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        # Find the most recent transition matching from→to and update it
        for transition in reversed(self._transitions):
            if transition.from_phase == from_phase and transition.to_phase == to_phase:
                transition.duration_seconds = duration
                transition.inputs = inputs or {}
                transition.outputs = outputs or {}
                transition.success = success
                transition.error = error
                return
        # If no matching transition, create a new one
        self._transitions.append(WorkflowTransition(
            from_phase=from_phase,
            to_phase=to_phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            inputs=inputs or {},
            outputs=outputs or {},
            success=success,
            error=error,
        ))
