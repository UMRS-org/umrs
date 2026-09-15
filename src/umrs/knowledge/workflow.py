"""Pipeline Integration — one function for the full scientific workflow.

RC1 Milestone M2: One automatic pipeline connecting ingestion → extraction
→ linking → verification → planning → research → memory update.

The `run_project_workflow()` function is the single entry point that
exercises the entire scientific workflow end-to-end. It:

  1. Creates a ResearchProject
  2. Ingests all files from one or more directories
  3. Extracts structured knowledge from each artifact
  4. Builds a KnowledgeGraph linking related artifacts
  5. Calculates a CoverageDashboard
  6. Verifies knowledge (answers evidence-backed questions)
  7. Generates a Research Readiness Report
  8. (User approves the report — manual step)
  9. Plans research (generates objectives, hypotheses, experiments)
  10. Runs the autonomous research loop
  11. Updates scientific memory

Every phase transition is logged for audit trail and replay.
If any step fails, the error is captured and the project phase is set
to FAILED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .ingest import IngesterConfig
from .project import (
    ProjectPhase,
    ResearchProject,
    WorkflowTransition,
)

logger = logging.getLogger(__name__)


# ── Workflow Result ───────────────────────────────────────────────────────


@dataclass
class WorkflowResult:
    """Result of running the full project workflow.

    Contains the completed ResearchProject plus metadata about the
    workflow execution: total duration, per-phase durations, success/failure
    status, and any errors encountered.
    """

    project: ResearchProject
    success: bool = True
    error: str | None = None
    total_duration_seconds: float = 0.0
    phase_durations: dict[str, float] = field(default_factory=dict)
    artifacts_ingested: int = 0
    knowledge_extracted: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    coverage_score: float = 0.0
    readiness_level: str = "unknown"
    confidence_level: str = "unknown"
    research_objectives: int = 0
    research_hypotheses: int = 0
    research_experiments: int = 0
    research_iterations: int = 0
    research_findings: int = 0
    memory_entries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "total_duration_seconds": self.total_duration_seconds,
            "phase_durations": self.phase_durations,
            "artifacts_ingested": self.artifacts_ingested,
            "knowledge_extracted": self.knowledge_extracted,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "coverage_score": self.coverage_score,
            "readiness_level": self.readiness_level,
            "confidence_level": self.confidence_level,
            "research_objectives": self.research_objectives,
            "research_hypotheses": self.research_hypotheses,
            "research_experiments": self.research_experiments,
            "research_iterations": self.research_iterations,
            "research_findings": self.research_findings,
            "memory_entries": self.memory_entries,
            "project_summary": self.project.to_dict(),
        }


# ── Workflow Options ──────────────────────────────────────────────────────


@dataclass
class WorkflowOptions:
    """Configuration for the project workflow.

    Attributes:
        research_goal: The high-level research goal for the planner
        max_iterations: Maximum number of research loop iterations
        verify_questions: Optional list of verification questions to ask
        auto_approve: If True, automatically approve the readiness report
            (for testing; real usage requires manual user approval)
        storage_path: Optional path for persistent storage
        user_id: User ID for approval
        s3_client: Optional boto3 S3 client for catalog-backed ingestion
        s3_bucket: S3 bucket name (required with s3_client for catalog ingest)
        owner_id: Owner user ID for catalog ACL (defaults to user_id)
        ingester_config: Optional KnowledgeIngester configuration
    """

    research_goal: str = "What research should we perform next?"
    max_iterations: int = 3
    verify_questions: list[str] = field(default_factory=list)
    auto_approve: bool = False
    storage_path: str | Path | None = None
    user_id: str = "user"
    s3_client: Any | None = None
    s3_bucket: str | None = None
    owner_id: str | None = None
    ingester_config: IngesterConfig | None = None
    meta_store: Any | None = None


def _research_project_kwargs(
    *,
    project_id: str,
    name: str,
    description: str,
    storage_path: str | Path | None,
    user_id: str,
    s3_client: Any | None,
    s3_bucket: str | None,
    owner_id: str | None,
    ingester_config: IngesterConfig | None,
    meta_store: Any | None = None,
) -> dict[str, Any]:
    """Build ``ResearchProject`` constructor kwargs from workflow options."""
    return {
        "project_id": project_id,
        "name": name,
        "description": description,
        "storage_path": storage_path,
        "s3_client": s3_client,
        "s3_bucket": s3_bucket,
        "owner_id": owner_id or user_id,
        "ingester_config": ingester_config,
        "meta_store": meta_store,
    }


# ── Main Workflow Function ─────────────────────────────────────────────────


def run_project_workflow(
    project_id: str,
    *directories: str | Path,
    name: str = "",
    description: str = "",
    research_goal: str = "What research should we perform next?",
    max_iterations: int = 3,
    verify_questions: list[str] | None = None,
    auto_approve: bool = False,
    storage_path: str | Path | None = None,
    user_id: str = "user",
    s3_client: Any | None = None,
    s3_bucket: str | None = None,
    owner_id: str | None = None,
    ingester_config: IngesterConfig | None = None,
    meta_store: Any | None = None,
) -> WorkflowResult:
    """Run the full scientific workflow end-to-end.

    This is the single entry point for the R4 scientific workflow. It
    creates a ResearchProject, ingests files, extracts knowledge, builds
    a knowledge graph, verifies knowledge, checks readiness, plans
    research, and runs the autonomous research loop.

    Args:
        project_id: Unique identifier for the project
        *directories: One or more directories to ingest
        name: Human-readable project name
        description: Project description
        research_goal: The high-level research goal
        max_iterations: Maximum research loop iterations
        verify_questions: Optional list of verification questions
        auto_approve: If True, auto-approve readiness (testing only)
        storage_path: Optional path for persistent storage
        user_id: User ID for approval
        s3_client: Optional boto3 S3 client for catalog-backed ingestion
        s3_bucket: S3 bucket name (required with s3_client for catalog ingest)
        owner_id: Owner user ID for catalog ACL (defaults to user_id)
        ingester_config: Optional KnowledgeIngester configuration

    Returns:
        A WorkflowResult containing the completed project and execution
        metadata.

    The workflow stages are:
        1. CREATE — Initialize ResearchProject
        2. INGEST — Ingest all files from directories
        3. UNDERSTAND — Extract knowledge, build graph, calculate coverage
        4. VERIFY — Answer verification questions with evidence
        5. CHECK_READINESS — Generate Research Readiness Report
        6. APPROVE — User approves (or auto_approve for testing)
        7. PLAN — Generate research plan with objectives, hypotheses, experiments
        8. RESEARCH — Execute autonomous research loop
        9. MEMORY — Update scientific memory with findings

    If any step fails, the error is captured and the project phase is set
    to FAILED.
    """
    start_time = datetime.now(timezone.utc)
    result = WorkflowResult(
        project=ResearchProject(
            **_research_project_kwargs(
                project_id=project_id,
                name=name,
                description=description,
                storage_path=storage_path,
                user_id=user_id,
                s3_client=s3_client,
                s3_bucket=s3_bucket,
                owner_id=owner_id,
                ingester_config=ingester_config,
                meta_store=meta_store,
            ),
        ),
    )

    try:
        # ── Stage 1: INGEST ──────────────────────────────────────────────
        phase_start = datetime.now(timezone.utc)
        result.artifacts_ingested = result.project.ingest_directory(*directories)
        result.phase_durations["ingest"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds()
        logger.info(
            "Project %s: ingested %d artifacts in %.2fs",
            project_id,
            result.artifacts_ingested,
            result.phase_durations["ingest"],
        )

        # ── Stage 2: UNDERSTAND ─────────────────────────────────────────
        phase_start = datetime.now(timezone.utc)
        result.project.understand()
        result.phase_durations["understand"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds()
        result.knowledge_extracted = len(result.project.extracted_knowledge)
        result.graph_nodes = len(result.project.knowledge_graph.nodes)
        result.graph_edges = len(result.project.knowledge_graph.edges)
        if result.project.coverage_dashboard:
            result.coverage_score = result.project.coverage_dashboard.coverage_score
        logger.info(
            "Project %s: understood %d artifacts, graph has %d nodes/%d edges, coverage %.2f",
            project_id,
            result.knowledge_extracted,
            result.graph_nodes,
            result.graph_edges,
            result.coverage_score,
        )

        # ── Stage 3: VERIFY (optional questions) ─────────────────────────
        if verify_questions:
            phase_start = datetime.now(timezone.utc)
            for question in verify_questions:
                answer = result.project.verify(question)
                logger.info(
                    "Project %s: Q='%s' A='%s' confidence=%.2f evidence=%d",
                    project_id,
                    question[:50],
                    answer.answer[:100] if answer.answer else "(empty)",
                    answer.confidence,
                    len(answer.evidence),
                )
            result.phase_durations["verify"] = (
                datetime.now(timezone.utc) - phase_start
            ).total_seconds()

        # ── Stage 4: CHECK_READINESS ────────────────────────────────────
        phase_start = datetime.now(timezone.utc)
        report = result.project.check_readiness()
        result.phase_durations["check_readiness"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds()
        result.readiness_level = report.readiness_level.value
        result.confidence_level = report.confidence_level.value
        logger.info(
            "Project %s: readiness=%s confidence=%s critical_issues=%d",
            project_id,
            result.readiness_level,
            result.confidence_level,
            report.critical_issue_count,
        )

        # ── Stage 5: APPROVE ────────────────────────────────────────────
        if auto_approve:
            result.project.approve_readiness(user_id=user_id)
            logger.info("Project %s: auto-approved readiness", project_id)
        else:
            # In real usage, user must call approve_readiness() manually
            # For the workflow function, we auto-approve if requested
            # Otherwise, we stop here and return the project
            logger.info(
                "Project %s: readiness report generated, awaiting user approval",
                project_id,
            )
            result.total_duration_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            return result

        # ── Stage 6: PLAN ───────────────────────────────────────────────
        phase_start = datetime.now(timezone.utc)
        plan = result.project.plan_research(research_goal)
        result.phase_durations["plan"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds()
        result.research_objectives = len(plan.objectives)
        result.research_hypotheses = len(plan.hypotheses)
        result.research_experiments = len(plan.experiments)
        logger.info(
            "Project %s: planned %d objectives, %d hypotheses, %d experiments",
            project_id,
            result.research_objectives,
            result.research_hypotheses,
            result.research_experiments,
        )

        # ── Stage 7: RESEARCH ──────────────────────────────────────────
        phase_start = datetime.now(timezone.utc)
        loop = result.project.run_research(max_iterations=max_iterations)
        result.phase_durations["research"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds()
        result.research_iterations = len(loop.iterations)
        result.research_findings = loop.total_findings
        logger.info(
            "Project %s: research complete, %d iterations, %d findings",
            project_id,
            result.research_iterations,
            result.research_findings,
        )

        # ── Stage 8: MEMORY ────────────────────────────────────────────
        result.memory_entries = len(result.project.memory_manager.store._entries)
        logger.info(
            "Project %s: scientific memory has %d entries",
            project_id,
            result.memory_entries,
        )

    except Exception as exc:
        result.success = False
        result.error = str(exc)
        result.project.phase = ProjectPhase.FAILED
        logger.error("Project %s: workflow failed: %s", project_id, exc)

    result.total_duration_seconds = (
        datetime.now(timezone.utc) - start_time
    ).total_seconds()
    return result


# ── Convenience Functions ──────────────────────────────────────────────────


def create_and_ingest(
    project_id: str,
    *directories: str | Path,
    name: str = "",
    description: str = "",
    storage_path: str | Path | None = None,
    s3_client: Any | None = None,
    s3_bucket: str | None = None,
    owner_id: str | None = None,
    ingester_config: IngesterConfig | None = None,
    user_id: str = "user",
    meta_store: Any | None = None,
) -> ResearchProject:
    """Create a project and ingest files (convenience function).

    This is a shortcut for:
        project = ResearchProject(project_id, name, description, storage_path)
        project.ingest_directory(*directories)
        return project

    Args:
        project_id: Unique identifier for the project
        *directories: One or more directories to ingest
        name: Human-readable project name
        description: Project description
        storage_path: Optional path for persistent storage

    Returns:
        A ResearchProject with files ingested, ready for `understand()`.
    """
    project = ResearchProject(
        **_research_project_kwargs(
            project_id=project_id,
            name=name,
            description=description,
            storage_path=storage_path,
            user_id=user_id,
            s3_client=s3_client,
            s3_bucket=s3_bucket,
            owner_id=owner_id,
            ingester_config=ingester_config,
            meta_store=meta_store,
        ),
    )
    project.ingest_directory(*directories)
    return project


def understand_and_verify(
    project: ResearchProject,
    questions: list[str] | None = None,
) -> dict[str, Any]:
    """Understand the project and verify with questions (convenience function).

    Args:
        project: A ResearchProject with files ingested
        questions: Optional list of verification questions

    Returns:
        A dict with verification answers and project summary.
    """
    project.understand()

    answers: list[dict[str, Any]] = []
    if questions:
        for question in questions:
            answer = project.verify(question)
            answers.append({
                "question": question,
                "answer": answer.answer,
                "confidence": answer.confidence,
                "evidence_count": len(answer.evidence),
            })

    return {
        "project_summary": project.to_dict(),
        "verification_answers": answers,
    }


def full_research_cycle(
    project_id: str,
    *directories: str | Path,
    research_goal: str = "What research should we perform next?",
    max_iterations: int = 3,
    auto_approve: bool = True,
    storage_path: str | Path | None = None,
    s3_client: Any | None = None,
    s3_bucket: str | None = None,
    owner_id: str | None = None,
    ingester_config: IngesterConfig | None = None,
    user_id: str = "user",
) -> WorkflowResult:
    """Run the full research cycle (convenience function).

    This is a shortcut for `run_project_workflow()` with auto_approve=True
    by default, suitable for testing and automated pipelines.

    Args:
        project_id: Unique identifier for the project
        *directories: One or more directories to ingest
        research_goal: The high-level research goal
        max_iterations: Maximum research loop iterations
        auto_approve: If True, auto-approve readiness
        storage_path: Optional path for persistent storage

    Returns:
        A WorkflowResult containing the completed project and execution
        metadata.
    """
    return run_project_workflow(
        project_id,
        *directories,
        research_goal=research_goal,
        max_iterations=max_iterations,
        auto_approve=auto_approve,
        storage_path=storage_path,
        s3_client=s3_client,
        s3_bucket=s3_bucket,
        owner_id=owner_id,
        ingester_config=ingester_config,
        user_id=user_id,
    )
