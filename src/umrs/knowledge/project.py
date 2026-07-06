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
from .ir import EXTRACTOR_VERSION, KnowledgeIR, build_knowledge_ir
from .knowledge_driven_planner import KnowledgeDrivenPlanner
from .link import KnowledgeGraph, KnowledgeLinker
from .loop import AutonomousResearchLoop, ResearchLoop
from .memory import ScientificMemoryManager, ScientificMemoryStore
from .ocr import get_ocr_backend, resolve_content_with_ocr
from .planner import AutonomousResearchPlanner, ResearchPlan
from .readiness import (
    ResearchReadinessGate,
    ResearchReadinessReport,
)
from .ingest import IngesterConfig, KnowledgeIngester
from .local_ingest import (
    classify_local_file,
    upload_local_files,
    uploads_prefix,
)
from .persistence import (
    knowledge_ir_dir,
    read_all_local_compiled_skills,
    read_local_skill_reviews,
    write_local_compiled_skill,
    write_local_knowledge_ir,
    write_local_skill_reviews,
)
from .router import default_local_router
from ..data_catalog.adapter import CatalogArtifactAdapter
from .catalog_store import CatalogKnowledgeStore
from .compiled_skill import (
    CompiledSkill,
    SkillReview,
    compile_skills_from_irs,
    effective_status,
    review_decision,
)
from .verify import (
    InMemoryKnowledgeStore,
    KnowledgeStore,
    KnowledgeVerifier,
    VerificationAnswer,
)

# ── UMRS API Adapters (Phase 3) ──────────────────────────────────────────
from .adapters import (
    UMRSEmbeddingsAdapter,
    UMRSEvidenceAdapter,
    UMRSKnowledgeGraphAdapter,
    UMRSKnowledgeModelAdapter,
    UMRSScientificExperimentRegistryAdapter,
    UMRSScientificMemoryAdapter,
    UMRSSearchAdapter,
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
        s3_client: Any | None = None,
        s3_bucket: str | None = None,
        owner_id: str | None = None,
        ingester_config: IngesterConfig | None = None,
    ) -> None:
        self.project_id = project_id
        self.name = name or project_id
        self.description = description
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.phase: ProjectPhase = ProjectPhase.CREATED

        # S3-backed ingestion (upload → catalog → knowledge store bridge)
        self._s3_client = s3_client
        self._s3_bucket = s3_bucket
        self._owner_id = owner_id
        self._ingester_config = ingester_config
        self._storage_path = Path(storage_path) if storage_path else None

        # Knowledge store — catalog-backed when S3 is configured, else in-memory
        self._local_knowledge_store: InMemoryKnowledgeStore | None = None
        self._knowledge_store: KnowledgeStore = self._build_knowledge_store()

        # Knowledge graph (typed relationships) — UMRS API adapter
        self._knowledge_graph: KnowledgeGraph = KnowledgeGraph()

        # Extracted knowledge per artifact_id
        self._extracted_knowledge: dict[str, ExtractedKnowledge] = {}

        # Canonical KnowledgeIR per artifact_id (immutable persistence record)
        self._knowledge_ir: dict[str, KnowledgeIR] = {}

        # Scientific memory (auto-registered artifacts) — UMRS API adapter
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
        self._ocr_backend = get_ocr_backend()
        self._linker = KnowledgeLinker()
        self._coverage_calculator = CoverageCalculator()
        self._external_searcher = ExternalKnowledgeSearcher()
        self._readiness_gate = ResearchReadinessGate()
        self._planner = AutonomousResearchPlanner()
        self._loop_engine = AutonomousResearchLoop()

        # ── UMRS API Adapters (Phase 3) ─────────────────────────────────
        # These adapters wrap the UMRS subsystems and provide the same
        # interface as the existing in-memory implementations.
        # UMRS owns scientific knowledge. R4 orchestrates scientific research.
        self._umrs_scientific_memory = UMRSScientificMemoryAdapter()
        self._umrs_knowledge_model = UMRSKnowledgeModelAdapter()
        self._umrs_knowledge_graph = UMRSKnowledgeGraphAdapter()
        self._umrs_experiment_registry = UMRSScientificExperimentRegistryAdapter()
        self._umrs_evidence = UMRSEvidenceAdapter()
        self._umrs_search = UMRSSearchAdapter()
        self._umrs_embeddings = UMRSEmbeddingsAdapter()

        # ── Workflow audit trail ─────────────────────────────────────────
        self._transitions: list[WorkflowTransition] = []

        # ── External knowledge references ────────────────────────────────
        self._external_refs: list[Any] = []

    # ── Public API: Workflow Stages ─────────────────────────────────────────

    def ingest_directory(self, *directories: str | Path) -> int:
        """Ingest all supported files from one or more directories.

        When ``s3_client`` and ``s3_bucket`` are configured, uploads files to
        project-scoped S3, runs catalog enrichment via ``KnowledgeIngester``,
        and bridges READY catalog metadata into the knowledge store.

        Without S3 configuration, falls back to reading files directly into
        the in-memory knowledge store (local tests and offline use).

        Phase transition: CREATED → INGESTING → UNDERSTANDING (via understand())
        """
        self._transition(ProjectPhase.CREATED, ProjectPhase.INGESTING)
        start = datetime.now(timezone.utc)

        if self._s3_client is not None and self._s3_bucket:
            count = self._ingest_directory_via_catalog(*directories)
        else:
            count = self._ingest_directory_local(*directories)

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
            last_knowledge = self._understand_artifact(artifact)

        self._persist_extracted_knowledge_to_umrs()

        # Build knowledge graph from artifacts + extracted knowledge
        artifact_dicts = [dict(a, extracted_knowledge=self._extracted_knowledge.get(a.get("artifact_id", ""))) for a in artifacts]
        self._knowledge_graph = self._linker.link(artifact_dicts)
        self._sync_graph_to_store()
        self._sync_graph_to_umrs_adapter()
        self._compile_derived_indexes()

        # Calculate coverage dashboard
        coverage_artifacts = self._build_coverage_artifacts(artifacts)
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

    def _understand_artifact(self, artifact: dict[str, Any]) -> ExtractedKnowledge:
        """Extract, store, and persist knowledge for a single artifact."""
        artifact_id = artifact.get("artifact_id", "")
        content = artifact.get("content", b"")
        if not content:
            loaded = self._knowledge_store.get_artifact(artifact_id)
            if loaded is not None:
                content = loaded.get("content", b"")
        if isinstance(content, str):
            content = content.encode("utf-8")
        filename = artifact.get("filename", "")
        artifact_type = artifact.get("artifact_type", "text")

        resolved_content, ocr_engine = resolve_content_with_ocr(
            filename, content, self._ocr_backend
        )

        knowledge = self._extractor.extract(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            filename=filename,
            content=resolved_content,
        )
        self._extracted_knowledge[artifact_id] = knowledge
        self._knowledge_store.add_extracted_knowledge(artifact_id, knowledge.to_dict())
        self._persist_knowledge_ir(
            artifact_id, artifact_type, content, knowledge, ocr_engine=ocr_engine
        )

        # Register extracted knowledge in scientific memory
        self._memory_manager.register_analysis_result(
            project_id=self.project_id,
            analysis=knowledge.summary or f"Extracted knowledge for {filename}",
            summary=f"Knowledge extracted from {filename}",
        )
        return knowledge

    def _build_coverage_artifacts(self, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Assemble the artifact records used by the coverage calculator."""
        coverage_artifacts = []
        for artifact in artifacts:
            artifact_id = artifact.get("artifact_id", "")
            coverage_artifacts.append({
                "artifact_id": artifact_id,
                "artifact_type": artifact.get("artifact_type", "text"),
                "filename": artifact.get("filename", ""),
                "metadata": artifact.get("metadata", {}),
                "extracted_knowledge": self._extracted_knowledge.get(artifact_id),
            })
        return coverage_artifacts

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

        Uses the UMRS knowledge graph (when populated) via
        ``KnowledgeDrivenPlanner`` and falls back to template planning
        when the graph has no anchor nodes.

        Phase transition: READY → PLANNING → RESEARCHING (via run_research())
        """
        self._transition(ProjectPhase.READY, ProjectPhase.PLANNING)
        start = datetime.now(timezone.utc)

        self._research_plan = self._plan_with_knowledge_graph(research_goal)

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
                "source_knowledge_ids": len(self._research_plan.source_knowledge_ids),
            },
        )
        return self._research_plan

    def _plan_with_knowledge_graph(self, research_goal: str) -> ResearchPlan:
        """Plan via graph-centric reasoning when ``understand()`` populated the graph."""
        from umrs.graph_reasoning import GraphNavigator
        from umrs.scientific_reasoning_engine import ScientificReasoningEngine

        navigator = GraphNavigator(graph=self._umrs_knowledge_graph.graph)
        engine = ScientificReasoningEngine(navigator=navigator)
        planner = KnowledgeDrivenPlanner(reasoning_engine=engine)
        plan = planner.plan_research(
            project_id=self.project_id,
            research_goal=research_goal,
        )
        if not plan.source_knowledge_ids and self._knowledge_graph.nodes:
            plan.source_knowledge_ids = list(self._knowledge_graph.nodes.keys())
        return plan

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

    # ── UMRS API Adapter Accessors ───────────────────────────────────────

    @property
    def umrs_scientific_memory(self) -> UMRSScientificMemoryAdapter:
        """UMRS scientific memory adapter."""
        return self._umrs_scientific_memory

    @property
    def umrs_knowledge_model(self) -> UMRSKnowledgeModelAdapter:
        """UMRS knowledge model adapter."""
        return self._umrs_knowledge_model

    @property
    def umrs_knowledge_graph(self) -> UMRSKnowledgeGraphAdapter:
        """UMRS knowledge graph adapter."""
        return self._umrs_knowledge_graph

    @property
    def umrs_experiment_registry(self) -> UMRSScientificExperimentRegistryAdapter:
        """UMRS experiment registry adapter."""
        return self._umrs_experiment_registry

    @property
    def umrs_evidence(self) -> UMRSEvidenceAdapter:
        """UMRS evidence adapter."""
        return self._umrs_evidence

    @property
    def umrs_search(self) -> UMRSSearchAdapter:
        """UMRS search adapter."""
        return self._umrs_search

    @property
    def umrs_embeddings(self) -> UMRSEmbeddingsAdapter:
        """UMRS embeddings adapter."""
        return self._umrs_embeddings

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

    def _ingest_directory_local(self, *directories: str | Path) -> int:
        """Ingest files directly into the in-memory knowledge store."""
        from .local_ingest import iter_ingestible_files

        count = 0
        for _base_dir, file_path in iter_ingestible_files(*directories):
            if self._ingest_file(file_path):
                count += 1
        return count

    def _ingest_directory_via_catalog(self, *directories: str | Path) -> int:
        """Upload local files to S3, enrich via catalog, bridge to knowledge store."""
        assert self._s3_client is not None
        assert self._s3_bucket

        uploads = upload_local_files(
            self._s3_client,
            self._s3_bucket,
            self.project_id,
            *directories,
        )
        if not uploads:
            return 0

        ingester = KnowledgeIngester(
            self._s3_client,
            self._s3_bucket,
            self._ingester_config or IngesterConfig(),
        )
        owner = self._owner_id or self.project_id
        ingester.ingest_project(
            self.project_id,
            owner,
            prefix=f"{uploads_prefix(self.project_id)}/",
        )

        if isinstance(self._knowledge_store, CatalogKnowledgeStore):
            self._knowledge_store.refresh()
            return len(self._knowledge_store.list_artifacts())

        raise RuntimeError("S3 ingest requires CatalogKnowledgeStore")

    def _ingest_file(self, file_path: Path) -> bool:
        """Ingest a single local file into the in-memory store."""
        artifact_type = classify_local_file(file_path)
        if artifact_type is None:
            return False

        try:
            content = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Failed to read %s: %s", file_path, exc)
            return False

        filename = file_path.name
        artifact_id = f"{artifact_type}-{file_path.stem}-{hash(str(file_path)) & 0xFFFFFF:X}"

        # Add to knowledge store
        self._ensure_local_knowledge_store().add_artifact({
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

    def _ensure_local_knowledge_store(self) -> InMemoryKnowledgeStore:
        """Create the in-memory store on first use (local / no-S3 mode only)."""
        if self._local_knowledge_store is None:
            persistence_dir = (
                self._storage_path / "extracted_knowledge"
                if self._storage_path is not None
                else None
            )
            self._local_knowledge_store = InMemoryKnowledgeStore(
                persistence_dir=persistence_dir,
            )
        return self._local_knowledge_store

    def _build_knowledge_store(self) -> KnowledgeStore:
        """Return catalog-backed store when S3 is configured."""
        if self._s3_client is not None and self._s3_bucket:
            adapter = CatalogArtifactAdapter(
                self._s3_client,
                self._s3_bucket,
                self.project_id,
                prefix=f"{self.project_id}/",
            )
            return CatalogKnowledgeStore(adapter)
        return self._ensure_local_knowledge_store()

    def _sync_graph_to_store(self) -> None:
        """Push the in-memory knowledge graph to the knowledge store."""
        if not self._knowledge_graph.nodes and not self._knowledge_graph.edges:
            return
        graph = self._knowledge_graph.to_dict()
        if hasattr(self._knowledge_store, "set_knowledge_graph"):
            self._knowledge_store.set_knowledge_graph(graph)
        elif hasattr(self._knowledge_store, "knowledge_graph"):
            self._knowledge_store.knowledge_graph = graph

    def _sync_graph_to_umrs_adapter(self) -> None:
        """Mirror linker output into the UMRS knowledge graph adapter."""
        if not self._knowledge_graph.nodes and not self._knowledge_graph.edges:
            return
        self._umrs_knowledge_graph.load_from_linker_graph(self._knowledge_graph)

    def _persist_knowledge_ir(
        self,
        artifact_id: str,
        artifact_type: str,
        content: bytes,
        knowledge: ExtractedKnowledge,
        *,
        ocr_engine: str | None = None,
    ) -> None:
        """Build the canonical KnowledgeIR and persist it write-once (local mode)."""
        ir = build_knowledge_ir(
            project_id=self.project_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content=content,
            extracted=knowledge.to_dict(),
            extractor_version=EXTRACTOR_VERSION,
            ocr_engine=ocr_engine,
        )
        self._knowledge_ir[artifact_id] = ir
        if self._storage_path is None:
            return
        try:
            write_local_knowledge_ir(knowledge_ir_dir(self._storage_path), ir)
        except OSError as exc:
            logger.warning("Failed to persist KnowledgeIR for %s: %s", artifact_id, exc)

    def _compile_derived_indexes(self) -> None:
        """Compile derived indexes (knowledge graph) from the canonical IRs.

        Phase 1: the knowledge graph is a *derived* index, regenerable from the
        canonical ``KnowledgeIR`` records via the ``KnowledgeRouter``. Local mode
        only; the in-memory graph and UMRS adapter are synced separately.
        """
        if self._storage_path is None or not self._knowledge_ir:
            return
        try:
            router = default_local_router(self._storage_path)
            router.regenerate(list(self._knowledge_ir.values()))
        except OSError as exc:
            logger.warning("Failed to compile derived indexes: %s", exc)

    # ── Continuous Learning: Skill Compilation (Phase 3) ─────────────────────

    def compile_knowledge_skills(self) -> list[CompiledSkill]:
        """Compile human-reviewable procedural skills from canonical IRs.

        Skills are always produced in the ``pending_review`` state — the
        platform never auto-approves learned knowledge. Compiled skills are
        persisted write-once (local mode); recompiling the same understanding is
        idempotent because ``skill_id`` hashes the source IR lineage.
        """
        skills = compile_skills_from_irs(list(self._knowledge_ir.values()))
        if self._storage_path is not None:
            for skill in skills:
                try:
                    write_local_compiled_skill(self._storage_path, skill)
                except OSError as exc:
                    logger.warning(
                        "Failed to persist compiled skill %s: %s", skill.skill_id, exc
                    )
        return skills

    def list_compiled_skills(self) -> list[dict[str, Any]]:
        """List compiled skills with their effective (post-review) status."""
        if self._storage_path is None:
            return []
        skills = read_all_local_compiled_skills(self._storage_path)
        reviews = read_local_skill_reviews(self._storage_path)
        result: list[dict[str, Any]] = []
        for skill in skills:
            payload = skill.to_dict()
            payload["status"] = effective_status(skill, reviews)
            result.append(payload)
        return result

    def review_compiled_skill(
        self,
        skill_id: str,
        status: str,
        *,
        reviewer: str = "",
        note: str = "",
    ) -> SkillReview:
        """Record a human review decision (approve/reject) for a compiled skill.

        Review decisions are stored separately from the immutable compiled
        skills, as mutable latest-wins state.
        """
        review = review_decision(skill_id, status, reviewer=reviewer, note=note)
        if self._storage_path is not None:
            reviews = read_local_skill_reviews(self._storage_path)
            reviews[skill_id] = review
            write_local_skill_reviews(self._storage_path, reviews)
        return review

    def _persist_extracted_knowledge_to_umrs(self) -> None:
        """Write structured extraction output to the UMRS knowledge model store."""
        from umrs.knowledge_model import KnowledgeObjectType

        type_map = {
            "paper": KnowledgeObjectType.PAPER,
            "notebook": KnowledgeObjectType.NOTEBOOK,
            "code": KnowledgeObjectType.CODE,
            "dataset": KnowledgeObjectType.DATASET,
            "text": KnowledgeObjectType.REFERENCE,
        }

        for artifact_id, knowledge in self._extracted_knowledge.items():
            object_type = type_map.get(
                knowledge.artifact_type,
                KnowledgeObjectType.REFERENCE,
            )
            metadata = {
                "artifact_id": artifact_id,
                "filename": knowledge.filename,
                "artifact_type": knowledge.artifact_type,
            }
            # Lineage: link the derived knowledge_model row back to the
            # canonical KnowledgeIR record it was compiled from.
            ir = self._knowledge_ir.get(artifact_id)
            if ir is not None:
                metadata["ir_id"] = ir.ir_id
                metadata["content_hash"] = ir.content_hash
                metadata["extractor_version"] = ir.extractor_version
            self._umrs_knowledge_model.create_object(
                project_id=self.project_id,
                object_type=object_type,
                content=knowledge.to_dict(),
                metadata=metadata,
            )

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
