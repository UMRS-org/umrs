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
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from r4_contracts.research import (
    EvidenceRef,
    EvidenceSource,
    Finding,
    LoopIterationRecord,
)

from . import loop_tracks
from .planner import (
    AutonomousResearchPlanner,
    ResearchPlan,
)

logger = logging.getLogger(__name__)

_LEGACY_DEMO08_WORKFLOW_NAME = "demo08_yield_eval"
# Kept for FakeWorkflowExecutionPort / docs. Real mode no longer falls back.
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
# Demo 08 / lab yield figures commonly appear as "85.0" / "4.21" near batch-3.
_BATCH3_YIELD_RE = re.compile(
    r"batch[-_ ]?3[^\n]{0,80}?(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)
_CO2_UPTAKE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:mmol|/g|co2)",
    re.IGNORECASE,
)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


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
    hypothesis_id: str = ""
    experiment_id: str = ""
    workflow_run_id: str = ""
    durable_job_id: str | None = None
    results: dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    evaluation: str = ""
    findings: list[Finding] = field(default_factory=list)
    retrieval_record_ids: list[str] = field(default_factory=list)
    # Artifact ids this iteration produced (knowledge artifacts minted from
    # its findings, workflow outputs) — the asset index derives the
    # loop → artifact lineage edge from them (ADR-036 D4, WP-05).
    artifact_ids: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | completed | failed | cancelled | awaiting_decision
    simulated: bool = False

    def to_record(self) -> LoopIterationRecord:
        """Project this engine iteration onto the public contract DTO."""
        return LoopIterationRecord(
            iteration_id=self.iteration_id,
            loop_id=self.loop_id,
            iteration_number=self.iteration_number,
            phase=self.phase.value
            if isinstance(self.phase, LoopPhase)
            else str(self.phase),
            hypothesis_id=self.hypothesis_id,
            hypothesis=self.hypothesis,
            experiment_id=self.experiment_id,
            workflow_run_id=self.workflow_run_id,
            durable_job_id=self.durable_job_id,
            retrieval_record_ids=list(self.retrieval_record_ids),
            findings=list(self.findings),
            artifact_ids=list(self.artifact_ids),
            results=dict(self.results),
            analysis=self.analysis,
            evaluation=self.evaluation,
            status=self.status,
            simulated=self.simulated,
            started_at=self.started_at,
            completed_at=self.completed_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.to_record().to_dict()


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


def _metrics_figures(metrics: dict[str, Any]) -> dict[str, Any]:
    """Quantitative figures the workflow metrics already carry."""
    return {
        key: metrics[key]
        for key in ("yield_pct", "co2_uptake_mmol_g", "batch_id")
        if metrics.get(key) is not None
    }


def _first_corpus_figure(corpus: str, *patterns: re.Pattern[str]) -> float | None:
    """First regex figure the corpus yields; None when no pattern hits."""
    for pattern in patterns:
        m = pattern.search(corpus)
        if m:
            return float(m.group(1))
    return None


def _figure_details(figures: dict[str, Any]) -> str:
    """Quantitative detail line; honest 'no quantitative figures' when absent."""
    detail = []
    if figures.get("yield_pct") is not None:
        detail.append(f"yield {figures['yield_pct']}%")
    if figures.get("co2_uptake_mmol_g") is not None:
        detail.append(f"uptake {figures['co2_uptake_mmol_g']}")
    return ", ".join(detail) if detail else "no quantitative figures"


class AutonomousResearchLoop:
    """Orchestrate the autonomous research loop.

    Usage:
        loop_engine = AutonomousResearchLoop(simulated=True)
        loop = loop_engine.start_loop(
            project_id="proj-001",
            research_goal="Understand the effect of learning rate on convergence",
        )
        loop_engine.run(loop)
    """

    def __init__(
        self,
        *,
        simulated: bool = False,
        workflow_port: Any | None = None,
        knowledge_store: Any | None = None,
        graph_navigator: Any | None = None,
        cancel_check: Callable[[], bool] | None = None,
        iteration_hook: Callable[[LoopIteration], None] | None = None,
    ) -> None:
        self._planner = AutonomousResearchPlanner()
        self._simulated = simulated
        self._workflow_port = workflow_port
        self._knowledge_store = knowledge_store
        self._graph_navigator = graph_navigator
        self._cancel_check = cancel_check
        self._iteration_hook = iteration_hook
        self._cancel_requested = False
        self._in_flight_execution_id: str | None = None
        self._retrieval_errors: list[str] = []
        # ADR-066 amendment 2 (D-34): the track this iteration serves ("" =
        # the single-branch loop). Gated actions carry it; denial closes it.
        self._track_id = ""

    def _checkpoint(self, iteration: LoopIteration) -> None:
        """Persist a mid-loop / end-of-iteration snapshot when a hook is set."""
        if self._iteration_hook is None:
            return
        try:
            self._iteration_hook(iteration)
        except Exception as exc:  # noqa: BLE001 — persistence must not kill the loop
            logger.warning("iteration_hook failed: %s", exc)

    @property
    def simulated(self) -> bool:
        return self._simulated

    def request_cancel(self) -> None:
        """Request cooperative cancellation between phases."""
        self._cancel_requested = True

    def start_loop(
        self,
        project_id: str,
        research_goal: str,
        *,
        max_iterations: int = 10,
        knowledge_base: Any = None,
        external_refs: list[Any] | None = None,
        memory_store: Any = None,
        research_plan: ResearchPlan | None = None,
        knowledge_store: Any | None = None,
        graph_navigator: Any | None = None,
    ) -> ResearchLoop:
        """Start a new research loop.

        Args:
            project_id: The project ID
            research_goal: The research goal
            max_iterations: Maximum number of iterations
            knowledge_base: Project knowledge
            external_refs: External references
            memory_store: Scientific memory store
            research_plan: Optional pre-built plan (skips template planner)
            knowledge_store: Optional catalog/in-memory store for retrieval
            graph_navigator: Optional graph navigator for related-object lookup

        Returns:
            A ``ResearchLoop`` instance ready to run.
        """
        if knowledge_store is not None:
            self._knowledge_store = knowledge_store
        if graph_navigator is not None:
            self._graph_navigator = graph_navigator

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
            metadata={"simulated": self._simulated},
        )

        if research_plan is not None:
            loop.plan = research_plan
        else:
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
                if self._should_cancel():
                    self._apply_cancel(loop, None)
                    break
                self._run_iteration(loop)

            if loop.status == LoopStatus.RUNNING:
                loop.phase = LoopPhase.COMPLETED
                loop.completed_at = datetime.now(timezone.utc).isoformat()
                loop.status = LoopStatus.STOPPED

        except Exception as exc:  # the loop's terminal ERROR state is the contract
            logger.exception("Research loop failed")
            loop.status = LoopStatus.ERROR
            loop.phase = LoopPhase.FAILED
            loop.metadata["error"] = str(exc)
            if not loop.completed_at:
                loop.completed_at = datetime.now(timezone.utc).isoformat()

        return loop

    def _should_cancel(self) -> bool:
        return bool(
            self._cancel_requested
            or (self._cancel_check is not None and self._cancel_check())
        )

    def _apply_cancel(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration | None,
    ) -> None:
        execution_id = self._in_flight_execution_id
        if execution_id and self._workflow_port is not None:
            try:
                self._workflow_port.cancel(execution_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to cancel workflow %s: %s", execution_id, exc)
        self._in_flight_execution_id = None
        loop.phase = LoopPhase.CANCELLED
        loop.status = LoopStatus.STOPPED
        loop.completed_at = datetime.now(timezone.utc).isoformat()
        loop.updated_at = loop.completed_at
        if iteration is not None:
            iteration.phase = LoopPhase.CANCELLED
            iteration.status = "cancelled"
            iteration.completed_at = loop.completed_at
            if iteration not in loop.iterations:
                loop.iterations.append(iteration)
            self._checkpoint(iteration)

    def _check_cancel_or_abort(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
    ) -> bool:
        """Return True if cancelled (caller should return from iteration)."""
        if not self._should_cancel():
            return False
        self._apply_cancel(loop, iteration)
        return True

    def _execute_experiment_phase(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
        experiment: Any,
    ) -> bool:
        """Phase 5. True = continue; False = deferred (D-34: the iteration
        finalised as awaiting_decision — the caller unwounds the loop)."""
        if self._simulated:
            loop.total_experiments_run += 1
            iteration.results = {"status": "completed", "output": "Experiment results"}
            return True
        if self._execute_experiment(loop, iteration, experiment) in (
            "deferred",
            "halted",
        ):
            return False
        # Checkpoint after trigger/wait so restart reconciliation can see
        # the in-flight durable_job_id even if the process dies next.
        self._checkpoint(iteration)
        return True

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
            simulated=self._simulated,
        )

        evidence_refs: list[EvidenceRef] = []
        excerpts: list[str] = []
        self._retrieval_errors = []

        # Phase 1: Knowledge retrieval
        loop.phase = LoopPhase.KNOWLEDGE_RETRIEVAL
        iteration.phase = LoopPhase.KNOWLEDGE_RETRIEVAL
        if self._check_cancel_or_abort(loop, iteration):
            return

        if not self._simulated:
            evidence_refs, excerpts = self._retrieve_knowledge(loop, iteration)

        # Phase 2: Searching (gap-fill)
        loop.phase = LoopPhase.SEARCHING
        iteration.phase = LoopPhase.SEARCHING
        if self._check_cancel_or_abort(loop, iteration):
            return

        if not self._simulated:
            search_refs, search_excerpts = self._search_knowledge(loop, iteration)
            evidence_refs.extend(search_refs)
            excerpts.extend(search_excerpts)

        # Phase 3: Hypothesis generation
        loop.phase = LoopPhase.HYPOTHESIS_GENERATION
        iteration.phase = LoopPhase.HYPOTHESIS_GENERATION
        if self._check_cancel_or_abort(loop, iteration):
            return

        hypothesis_id = ""
        track = None
        if loop.plan and loop.plan.hypotheses:
            ok, track = self._hypothesis_phase(loop, iteration, excerpts)
            if not ok:
                return
            hypothesis_id = iteration.hypothesis_id

        # Phase 4: Experiment design
        loop.phase = LoopPhase.EXPERIMENT_DESIGN
        iteration.phase = LoopPhase.EXPERIMENT_DESIGN
        if self._check_cancel_or_abort(loop, iteration):
            return

        experiment = self._pick_experiment(loop, iteration, track)

        # Phase 5: Experiment execution
        loop.phase = LoopPhase.EXPERIMENT_EXECUTION
        iteration.phase = LoopPhase.EXPERIMENT_EXECUTION
        if self._check_cancel_or_abort(loop, iteration):
            return

        if not self._execute_experiment_phase(loop, iteration, experiment):
            return  # D-34: awaiting_decision — iteration already finalised

        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return

        # Phase 6: Output collection
        loop.phase = LoopPhase.OUTPUT_COLLECTION
        iteration.phase = LoopPhase.OUTPUT_COLLECTION
        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return

        # Phase 7: Analysis
        loop.phase = LoopPhase.ANALYSIS
        iteration.phase = LoopPhase.ANALYSIS
        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return
        iteration.analysis = self._build_analysis(iteration, excerpts)

        # Phase 8: Evaluation
        loop.phase = LoopPhase.EVALUATION
        iteration.phase = LoopPhase.EVALUATION
        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return
        iteration.evaluation = self._build_evaluation(iteration, excerpts)

        # Phase 9: Publishing findings
        loop.phase = LoopPhase.PUBLISHING
        iteration.phase = LoopPhase.PUBLISHING
        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return

        finding_refs = list(evidence_refs)
        if self._simulated:
            finding_refs = []
        else:
            # A real finding without evidence is indistinguishable from a
            # fabricated one. Retrieval problems must fail the run, not
            # silently downgrade the finding.
            if self._retrieval_errors:
                raise RuntimeError(
                    "knowledge retrieval failed; refusing to publish findings "
                    f"without complete evidence: {'; '.join(self._retrieval_errors)}"
                )
            if not finding_refs:
                raise RuntimeError(
                    "knowledge retrieval returned no evidence; refusing to "
                    "publish an unevidenced finding in real mode"
                )
        statement = (
            f"Finding from iteration {loop.current_iteration}"
            if self._simulated
            else self._build_finding_statement(iteration, excerpts)
        )
        iteration.findings = [
            Finding(
                finding_id=f"finding-{_generate_id()}",
                hypothesis_id=hypothesis_id,
                statement=statement,
                evidence_refs=finding_refs,
                workflow_run_id=iteration.workflow_run_id,
                durable_job_id=iteration.durable_job_id,
                confidence=0.5 if self._simulated else 0.7,
                simulated=self._simulated,
            )
        ]
        loop.total_findings += 1

        # Phase 10: Memory update
        loop.phase = LoopPhase.MEMORY_UPDATE
        iteration.phase = LoopPhase.MEMORY_UPDATE
        if self._check_cancel_or_abort(loop, iteration):
            self._checkpoint(iteration)
            return

        # Complete iteration
        iteration.completed_at = datetime.now(timezone.utc).isoformat()
        iteration.status = "completed"
        self._stamp_track(iteration)
        # WP-12: per-iteration wall-clock span — observable (engine label)
        # and recorded on the iteration's results for the loop detail view.
        from umrs.jobs.timing import duration_ms
        from umrs.metrics import METRICS

        iteration_ms = duration_ms(iteration.started_at, iteration.completed_at)
        if iteration_ms is not None:
            iteration.results["timing_ms"] = iteration_ms
            METRICS.loop_iteration_seconds(
                "simulated" if self._simulated else "real", iteration_ms / 1000
            )
            # WP-12 r2: same window → OTel span (no-op without OTel hookup).
            from umrs.jobs.timing import maybe_emit_span

            maybe_emit_span(
                "umrs.loop.iteration",
                started_at=iteration.started_at,
                transitioned_at=iteration.completed_at,
                attributes={
                    "umrs.loop_id": str(iteration.loop_id),
                    "umrs.trace_id": str(
                        loop.metadata.get("trace_id") or iteration.loop_id
                    ),
                    "umrs.iteration_id": str(iteration.iteration_id),
                    "umrs.iteration_number": int(iteration.iteration_number),
                    "umrs.project_id": str(loop.project_id or ""),
                    "umrs.engine": "simulated" if self._simulated else "real",
                    "umrs.durable_job_id": str(iteration.durable_job_id or ""),
                },
            )
        loop.iterations.append(iteration)
        loop.updated_at = datetime.now(timezone.utc).isoformat()
        self._checkpoint(iteration)

        # Update success rate
        if loop.total_experiments_run > 0:
            loop.success_rate = loop.total_findings / loop.total_experiments_run

    def _retrieve_knowledge(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
    ) -> tuple[list[EvidenceRef], list[str]]:
        refs: list[EvidenceRef] = []
        excerpts: list[str] = []
        related_ids: list[str] = []

        anchors = list(loop.plan.source_knowledge_ids) if loop.plan else []
        for anchor_id in anchors:
            if anchor_id and anchor_id not in related_ids:
                related_ids.append(anchor_id)
                excerpt = self._excerpt_for_id(anchor_id) or f"anchor:{anchor_id}"
                refs.append(
                    EvidenceRef(
                        source=EvidenceSource.GRAPH,
                        excerpt=excerpt,
                    )
                )
                excerpts.append(excerpt)

            if self._graph_navigator is None:
                continue
            try:
                related = self._graph_navigator.find_related(anchor_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("find_related(%s) failed: %s", anchor_id, exc)
                self._retrieval_errors.append(f"find_related({anchor_id}): {exc}")
                continue
            for nid in _related_object_ids(related):
                if nid not in related_ids:
                    related_ids.append(nid)
                    excerpt = self._excerpt_for_id(nid) or f"related:{nid}"
                    refs.append(
                        EvidenceRef(
                            source=EvidenceSource.GRAPH,
                            excerpt=excerpt,
                        )
                    )
                    excerpts.append(excerpt)

        iteration.retrieval_record_ids = list(related_ids)
        return refs, excerpts

    def _search_knowledge(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
    ) -> tuple[list[EvidenceRef], list[str]]:
        refs: list[EvidenceRef] = []
        excerpts: list[str] = []
        if self._knowledge_store is None:
            return refs, excerpts

        query = loop.research_goal or ""
        if loop.plan and loop.plan.hypotheses:
            hyp = loop.plan.hypotheses[
                (loop.current_iteration - 1) % len(loop.plan.hypotheses)
            ]
            query = f"{query} {hyp.statement}".strip()

        artifacts: list[dict[str, Any]] = []
        try:
            artifacts = list(self._knowledge_store.search_artifacts(query) or [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("search_artifacts failed: %s", exc)
            self._retrieval_errors.append(f"search_artifacts: {exc}")

        if not artifacts:
            artifacts = self._gap_fill_from_store(query)

        for artifact in artifacts:
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id:
                continue
            excerpt = _artifact_excerpt(artifact)
            refs.append(
                EvidenceRef(
                    source=EvidenceSource.CATALOG,
                    excerpt=excerpt,
                    artifact_id=artifact_id,
                    s3_key=artifact.get("s3_key") or artifact.get("key"),
                )
            )
            excerpts.append(excerpt)
            if artifact_id not in iteration.retrieval_record_ids:
                iteration.retrieval_record_ids.append(artifact_id)

        return refs, excerpts

    def _gap_fill_from_store(self, query: str) -> list[dict[str, Any]]:
        """Match query tokens against artifact content when catalog search is empty."""
        try:
            all_artifacts = list(self._knowledge_store.list_artifacts() or [])
        except Exception as exc:  # noqa: BLE001
            self._retrieval_errors.append(f"list_artifacts: {exc}")
            return []
        tokens = set(_TOKEN_RE.findall(query.lower()))
        if not tokens:
            return all_artifacts[:5]
        matched: list[dict[str, Any]] = []
        for artifact in all_artifacts:
            haystack = " ".join(
                [
                    str(artifact.get("filename") or ""),
                    str(artifact.get("summary") or ""),
                    _artifact_excerpt(artifact),
                ]
            ).lower()
            if any(tok in haystack for tok in tokens):
                matched.append(artifact)
        return matched or all_artifacts[:5]

    def _excerpt_for_id(self, object_id: str) -> str:
        if self._knowledge_store is None:
            return ""
        try:
            artifact = self._knowledge_store.get_artifact(object_id)
        except Exception as exc:  # noqa: BLE001
            self._retrieval_errors.append(f"get_artifact({object_id}): {exc}")
            return ""
        if not artifact:
            return ""
        return _artifact_excerpt(artifact)

    def _select_hypothesis(self, loop: ResearchLoop, excerpts: list[str]) -> Any:
        assert loop.plan is not None
        hypotheses = loop.plan.hypotheses
        if not excerpts or self._simulated:
            idx = (loop.current_iteration - 1) % len(hypotheses)
            return hypotheses[idx]

        corpus = " ".join(excerpts).lower()
        best = None
        best_score = -1
        for hyp in hypotheses:
            tokens = _TOKEN_RE.findall(hyp.statement.lower())
            score = sum(1 for t in tokens if t in corpus)
            # Prefer substring overlap when the whole statement appears.
            if hyp.statement and hyp.statement.lower() in corpus:
                score += 10
            if score > best_score:
                best_score = score
                best = hyp
        if best is not None and best_score > 0:
            return best
        idx = (loop.current_iteration - 1) % len(hypotheses)
        return hypotheses[idx]

    def _workflow_gate_verdict(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
        workflow_name: str,
        params: dict[str, Any],
        physical: bool = False,
    ) -> bool | str:
        """The workflow-trigger gate's verdict.

        True runs the trigger; False denied (``_gate_denied`` ran — the
        caller unwinds via exception); ``"deferred"`` finalised the
        iteration as awaiting_decision (D-34 first leg — the loop
        continues on ungated work)."""
        gate = getattr(self, "_approval_gate", None)
        if gate is None:
            return True
        verdict = gate(
            {
                "kind": "workflow_trigger",
                "workflow_name": workflow_name,
                "params": dict(params),
                "iteration": loop.current_iteration,
                "track_id": self._track_id,
                # ADR-066 amendment 3: an instrument/robot experiment is also
                # a physical_action (the experiment declares it).
                "physical": physical,
            }
        )
        if verdict == "deferred":
            self._gate_deferred(loop, iteration, workflow_name)
            return "deferred"
        if not verdict:
            self._gate_denied(loop, iteration, workflow_name)
        return bool(verdict)

    def _execute_experiment(
        self,
        loop: ResearchLoop,
        iteration: LoopIteration,
        experiment: Any | None,
    ) -> None:
        if self._workflow_port is None:
            raise RuntimeError("workflow execution port required for real mode")

        workflow_name = None
        params: dict[str, Any] = {
            "project_id": loop.project_id,
            "iteration": loop.current_iteration,
            "hypothesis_id": iteration.hypothesis_id,
        }
        if experiment is not None:
            workflow_name = getattr(experiment, "workflow_name", None) or (
                experiment.metadata or {}
            ).get("workflow_name")
            params.update(dict(experiment.parameters or {}))
            params.setdefault("experiment_id", experiment.experiment_id)
        # Experiment parameters must not override the loop's project context
        # (Gateway verified headers / durable job project_id).
        params["project_id"] = loop.project_id
        # Loop-level default (resolved at loop start from the registry or the
        # caller's workflow_name — G-P1-3); experiments may still override.
        workflow_name = workflow_name or loop.metadata.get("workflow_name")

        if not workflow_name:
            # Real mode must not silently fall back to a demo workflow.
            raise RuntimeError(
                "real research mode requires experiment.metadata.workflow_name "
                "(or experiment.workflow_name / the loop's workflow_name) set to "
                "a registered UMRS workflow; pass workflow_name at loop start — "
                "UMRS uses the single registered workflow automatically and lists "
                "candidates otherwise (discover with `r4 workflow list`)"
            )

        # Tool-grain enforcement: an 'ask'-mode loop gates each workflow
        # trigger through the approvals registry (injected by the driver).
        verdict = self._workflow_gate_verdict(
            loop, iteration, workflow_name, params, physical=_is_physical(experiment)
        )
        if verdict is not True:
            # deferred → awaiting_decision; False → the iteration was already
            # finalised (stopped loop / closed track) — never trigger then.
            return "deferred" if verdict == "deferred" else "halted"

        execution_id = self._workflow_port.trigger(workflow_name, params)
        self._in_flight_execution_id = execution_id
        iteration.workflow_run_id = execution_id

        if self._check_cancel_or_abort(loop, iteration):
            return

        wait_payload = self._workflow_port.wait(execution_id)
        results_payload = self._workflow_port.results(execution_id)
        self._in_flight_execution_id = None

        durable_job_id = wait_payload.get("durable_job_id") or results_payload.get(
            "durable_job_id"
        )
        if not durable_job_id:
            raise RuntimeError(
                "durable job backend required for real research mode "
                "(durable_job_id missing from workflow response)"
            )

        status = str(
            results_payload.get("status") or wait_payload.get("status") or ""
        ).lower()
        if status not in {"succeeded", "completed", "success"}:
            err = (
                results_payload.get("error")
                or wait_payload.get("error")
                or status
                or "unknown"
            )
            raise RuntimeError(
                f"workflow execution failed (status={status or 'unknown'}): {err}"
            )

        iteration.durable_job_id = str(durable_job_id)
        metrics = dict(results_payload.get("metrics") or {})
        outputs = results_payload.get("outputs") or wait_payload.get("outputs") or {}
        # Promote known lab figures from outputs when the backend omitted metrics.
        for key in ("batch_id", "yield_pct", "co2_uptake_mmol_g"):
            if metrics.get(key) is None and outputs.get(key) is not None:
                metrics[key] = outputs[key]
        iteration.results = {
            "status": results_payload.get("status") or wait_payload.get("status"),
            "outputs": outputs,
            "metrics": metrics,
            "artifacts": results_payload.get("artifacts") or [],
        }
        loop.total_experiments_run += 1

    def _hypothesis_phase(
        self, loop: ResearchLoop, iteration: LoopIteration, excerpts: list[str]
    ) -> tuple[bool, dict[str, Any] | None]:
        """Single-branch selection, or (ADR-066 amendment 2) the next track.

        (False, None) = the iteration was finalised (gate deferred/denied, or
        every track is closed and the loop completed)."""
        self._track_id = ""
        if not loop_tracks.tracks_enabled(loop):
            return self._select_and_gate_hypothesis(loop, iteration, excerpts), None
        if not loop_tracks.tracks(loop) and not self._open_tracks(
            loop, iteration, excerpts
        ):
            return False, None
        track = loop_tracks.next_track(loop)
        if track is None:
            self._complete_tracks_exhausted(loop)
            return False, None
        self._track_id = str(track["track_id"])
        iteration.hypothesis = str(track["hypothesis"])
        iteration.hypothesis_id = str(track["hypothesis_id"])
        iteration.results["track_id"] = self._track_id
        loop.total_hypotheses_tested += 1
        return True, track

    def _open_tracks(
        self, loop: ResearchLoop, iteration: LoopIteration, excerpts: list[str]
    ) -> bool:
        """Open the tracks; the competing-hypotheses question is raised only
        when there are more candidates than tracks (a real choice)."""
        if loop_tracks.needs_choice(loop) and not self._competing_gate(loop, iteration):
            return False
        loop_tracks.open_tracks(loop, self._rank_hypotheses(loop, excerpts))
        return True

    def _rank_hypotheses(self, loop: ResearchLoop, excerpts: list[str]) -> list[Any]:
        """Plan hypotheses by evidence overlap (stable; plan order when
        simulated or without evidence)."""
        hypotheses = list(loop.plan.hypotheses)
        if not excerpts or self._simulated:
            return hypotheses
        corpus = " ".join(excerpts).lower()

        def score(hyp: Any) -> int:
            tokens = _TOKEN_RE.findall(hyp.statement.lower())
            bonus = 10 if hyp.statement and hyp.statement.lower() in corpus else 0
            return sum(1 for t in tokens if t in corpus) + bonus

        return sorted(hypotheses, key=score, reverse=True)

    def _pick_experiment(
        self, loop: ResearchLoop, iteration: LoopIteration, track: dict[str, Any] | None
    ) -> Any:
        """The track's own experiment sequence, else the loop-wide rotation."""
        if track is not None:
            experiment = loop_tracks.take_turn(loop, track)
        elif loop.plan and loop.plan.experiments:
            idx = (loop.current_iteration - 1) % len(loop.plan.experiments)
            experiment = loop.plan.experiments[idx]
        else:
            experiment = None
        if experiment is not None:
            iteration.experiment_id = experiment.experiment_id
        return experiment

    def _stamp_track(self, iteration: LoopIteration) -> None:
        if self._track_id:
            iteration.results["track_id"] = self._track_id

    def _close_track(self, loop: ResearchLoop, iteration: LoopIteration) -> None:
        """A decided "no" on a track's gated action closes that track only."""
        loop_tracks.close_track(loop, self._track_id)
        now = datetime.now(timezone.utc).isoformat()
        iteration.completed_at = now
        iteration.status = "track_rejected"
        iteration.results["track_id"] = self._track_id
        loop.iterations.append(iteration)
        loop.updated_at = now
        self._checkpoint(iteration)

    def _complete_tracks_exhausted(self, loop: ResearchLoop) -> None:
        """Every track is closed: the loop completes (nothing left to pursue)."""
        now = datetime.now(timezone.utc).isoformat()
        loop.metadata["tracks_exhausted"] = True
        loop.phase = LoopPhase.COMPLETED
        loop.status = LoopStatus.STOPPED
        loop.completed_at = now
        loop.updated_at = now

    def _competing_gate(self, loop: ResearchLoop, iteration: LoopIteration) -> bool:
        """The ADR-066 competing-hypotheses trigger (True = proceed)."""
        gate = getattr(self, "_approval_gate", None)
        if gate is None:
            return True
        verdict = gate(
            {
                "kind": "competing_hypotheses",
                "candidates": len(loop.plan.hypotheses),
                "iteration": loop.current_iteration,
            }
        )
        if verdict == "deferred":
            self._gate_deferred(loop, iteration, "competing-hypotheses")
            return False
        if not verdict:
            self._gate_denied(loop, iteration, "competing-hypotheses")
            return False
        return True

    def _select_and_gate_hypothesis(
        self, loop: ResearchLoop, iteration: LoopIteration, excerpts: list[str]
    ) -> bool:
        """Hypothesis selection + the ADR-066 competing-hypotheses gate.

        Several live candidates trigger the ask-the-scientist pause when the
        loop's pause profile names the class (cost of a wrong branch is a
        decision, not a step). Returns False when the gate denied (the
        iteration is finalised by ``_gate_denied``); on True the iteration
        carries hypothesis + hypothesis_id and the counter is bumped.
        """
        if len(loop.plan.hypotheses) > 1 and not self._competing_gate(loop, iteration):
            return False
        hyp = self._select_hypothesis(loop, excerpts)
        hyp = self._direction_checked(loop, iteration, hyp)
        if hyp is None:
            return False
        iteration.hypothesis = hyp.statement
        iteration.hypothesis_id = loop_tracks.hypothesis_id_of(loop, hyp)
        loop.total_hypotheses_tested += 1
        return True

    def _direction_checked(
        self, loop: ResearchLoop, iteration: LoopIteration, hyp: Any
    ) -> Any | None:
        """ADR-066 amendment 3: switching away from the previous completed
        iteration's hypothesis is a ``direction_change``. Approved → switch;
        a decided "no" → stay on the previous branch; deferred / stopped →
        None (the iteration was finalised)."""
        previous = _previous_hypothesis(loop)
        gate = getattr(self, "_approval_gate", None)
        if gate is None or previous is None or previous is hyp:
            return hyp
        verdict = gate(
            {
                "kind": "direction_change",
                "from": loop_tracks.hypothesis_id_of(loop, previous),
                "to": loop_tracks.hypothesis_id_of(loop, hyp),
                "iteration": loop.current_iteration,
            }
        )
        if verdict == "deferred":
            self._gate_deferred(loop, iteration, "direction-change")
            return None
        if verdict:
            return hyp
        if self._should_cancel() or loop.status != LoopStatus.RUNNING:
            self._apply_cancel(loop, iteration)
            return None
        iteration.results["direction_vetoed"] = loop_tracks.hypothesis_id_of(loop, hyp)
        return previous

    def _gate_deferred(
        self, loop: ResearchLoop, iteration: LoopIteration, label: str
    ) -> None:
        """D-34 first leg (ADR-066 amendment 2026-09-24): the profile-mode
        gate DEFERRED a pending named-class trigger — finalise the
        iteration as ``awaiting_decision`` (nothing fabricated: no
        experiment ran, no findings minted) and let the loop continue on
        ungated work. The durable question stays in the inbox; the next
        iteration re-requests idempotently under the same approval id and
        proceeds once decided. The runner's deferral ledger bounds spin
        (3 strikes → hard park)."""
        now = datetime.now(timezone.utc).isoformat()
        iteration.completed_at = now
        iteration.status = "awaiting_decision"
        iteration.results["deferred_gate"] = label
        loop.iterations.append(iteration)
        loop.updated_at = now
        self._checkpoint(iteration)

    def _gate_denied(
        self, loop: ResearchLoop, iteration: LoopIteration, workflow_name: str
    ) -> None:
        """A False gate is a cancellation when stop/abort was requested, else a rejection.

        Live finding (WP-07): stopping a loop parked at its gate used to end
        the loop in ERROR ("operator rejected or cancelled"), which its hosting
        job then recorded as failed. Stop is a cancellation; only a decided
        "no" is a rejection.
        """
        if self._should_cancel() or loop.status != LoopStatus.RUNNING:
            self._apply_cancel(loop, iteration)
            return
        if self._track_id:
            self._close_track(loop, iteration)
            return
        raise RuntimeError(
            f"operator rejected the tool-grain action (workflow_trigger '{workflow_name}')"
        )

    @staticmethod
    def _extract_evidence_figures(
        excerpts: list[str], results: dict[str, Any]
    ) -> dict[str, Any]:
        """Pull Demo 08-style yield / CO2 figures from evidence or workflow metrics."""
        metrics = (results or {}).get("metrics") or {}
        figures: dict[str, Any] = _metrics_figures(metrics)
        corpus = "\n".join(excerpts)
        if "yield_pct" not in figures:
            found = _first_corpus_figure(corpus, _BATCH3_YIELD_RE, _PCT_RE)
            if found is not None:
                figures["yield_pct"] = found
                figures.setdefault("batch_id", "batch-3")
        if "co2_uptake_mmol_g" not in figures:
            found = _first_corpus_figure(corpus, _CO2_UPTAKE_RE)
            if found is not None:
                figures["co2_uptake_mmol_g"] = found
        figures.setdefault("batch_id", metrics.get("batch_id") or "batch-3")
        return figures

    def _build_analysis(self, iteration: LoopIteration, excerpts: list[str]) -> str:
        if self._simulated:
            return f"Simulated analysis of iteration {iteration.iteration_number}"
        figures = self._extract_evidence_figures(excerpts, iteration.results)
        parts = [f"Iteration {iteration.iteration_number} analysis"]
        if iteration.hypothesis:
            parts.append(f"hypothesis='{iteration.hypothesis}'")
        if figures.get("yield_pct") is not None:
            parts.append(
                f"{figures.get('batch_id', 'batch-3')} yield={figures['yield_pct']}%"
            )
        if figures.get("co2_uptake_mmol_g") is not None:
            parts.append(f"CO2 uptake={figures['co2_uptake_mmol_g']} mmol/g")
        if iteration.durable_job_id:
            parts.append(f"durable_job_id={iteration.durable_job_id}")
        if excerpts:
            parts.append(f"evidence_excerpt={excerpts[0][:160]!r}")
        return "; ".join(parts)

    def _build_evaluation(self, iteration: LoopIteration, excerpts: list[str]) -> str:
        if self._simulated:
            return f"Simulated evaluation of iteration {iteration.iteration_number}"
        figures = self._extract_evidence_figures(excerpts, iteration.results)
        status = (iteration.results or {}).get("status") or "unknown"
        supporting = status in {"succeeded", "completed", "success"} and bool(figures)
        verdict = "supports" if supporting else "inconclusive"
        joined = _figure_details(figures)
        return (
            f"Evaluation ({verdict}): workflow status={status}; "
            f"evidence figures ({joined}) relative to hypothesis "
            f"{iteration.hypothesis_id or '(none)'}"
        )

    def _build_finding_statement(
        self,
        iteration: LoopIteration,
        excerpts: list[str],
    ) -> str:
        figures = self._extract_evidence_figures(excerpts, iteration.results)
        if figures.get("yield_pct") is not None:
            statement = (
                f"Batch-3 lab evidence shows yield "
                f"{figures['yield_pct']}% at CO2 uptake "
                f"{figures.get('co2_uptake_mmol_g', 'n/a')} mmol/g"
            )
            if iteration.hypothesis:
                statement = f"{statement}; tested hypothesis '{iteration.hypothesis}'"
        else:
            statement = f"Finding from iteration {iteration.iteration_number}"
            if iteration.hypothesis:
                statement = f"{statement}: tested '{iteration.hypothesis}'"
            if excerpts:
                statement = f"{statement} (evidence: {excerpts[0][:120]})"
        if iteration.workflow_run_id:
            statement = f"{statement} [workflow={iteration.workflow_run_id}]"
        if iteration.durable_job_id:
            statement = f"{statement} [durable_job_id={iteration.durable_job_id}]"
        return statement


# ── Helper Functions ───────────────────────────────────────────────────────


def _previous_hypothesis(loop: ResearchLoop) -> Any | None:
    """The plan hypothesis the last COMPLETED iteration pursued, if any."""
    for it in reversed(loop.iterations):
        if it.status != "completed" or not it.hypothesis_id:
            continue
        for hyp in loop.plan.hypotheses:
            if loop_tracks.hypothesis_id_of(loop, hyp) == it.hypothesis_id:
                return hyp
        return None
    return None


def _is_physical(experiment: Any | None) -> bool:
    """The experiment declares it drives an instrument/robot (amendment 3)."""
    metadata = getattr(experiment, "metadata", None) or {}
    return bool(metadata.get("physical_action"))


def _generate_id() -> str:
    """Generate a unique ID."""
    import uuid

    return uuid.uuid4().hex[:12]


def _related_object_ids(related: Any) -> list[str]:
    ids: list[str] = []
    for attr in (
        "papers",
        "code",
        "notebooks",
        "datasets",
        "hypotheses",
        "experiments",
        "findings",
        "figures",
        "equations",
    ):
        values = getattr(related, attr, None) or []
        for value in values:
            if value and value not in ids:
                ids.append(value)
    return ids


def _artifact_excerpt(artifact: dict[str, Any]) -> str:
    content = artifact.get("content", b"")
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
    else:
        text = str(content or "")
    text = text.strip()
    if text:
        return text[:240]
    summary = (
        artifact.get("summary")
        or artifact.get("filename")
        or artifact.get("artifact_id")
    )
    return str(summary or "")[:240]


# ── Convenience Function ────────────────────────────────────────────────────


def run_research_loop(
    project_id: str,
    research_goal: str,
    *,
    max_iterations: int = 10,
    knowledge_base: Any = None,
    external_refs: list[Any] | None = None,
    memory_store: Any = None,
    simulate: bool = False,
    workflow_port: Any | None = None,
    knowledge_store: Any | None = None,
    graph_navigator: Any | None = None,
) -> ResearchLoop:
    """Run an autonomous research loop (convenience function).

    Args:
        project_id: The project ID
        research_goal: The research goal
        max_iterations: Maximum number of iterations
        knowledge_base: Project knowledge
        external_refs: External references
        memory_store: Scientific memory store
        simulate: When True, stamp iterations/findings as simulated
        workflow_port: Required for real mode (``simulate=False``)
        knowledge_store: Optional store for catalog search
        graph_navigator: Optional navigator for graph retrieval

    Returns:
        A completed ``ResearchLoop`` instance.
    """
    if not simulate and workflow_port is None:
        raise RuntimeError("workflow execution port required for real mode")
    loop_engine = AutonomousResearchLoop(
        simulated=simulate,
        workflow_port=workflow_port,
        knowledge_store=knowledge_store,
        graph_navigator=graph_navigator,
    )
    loop = loop_engine.start_loop(
        project_id=project_id,
        research_goal=research_goal,
        max_iterations=max_iterations,
        knowledge_base=knowledge_base,
        external_refs=external_refs,
        memory_store=memory_store,
        knowledge_store=knowledge_store,
        graph_navigator=graph_navigator,
    )
    return loop_engine.run(loop)
