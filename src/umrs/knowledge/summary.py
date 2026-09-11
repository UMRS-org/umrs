"""Knowledge dashboard aggregation layer.

Computes a deterministic project-health snapshot from all durable
knowledge data sources: artifacts, documents, graph, and verification.

Derives next_actions from verification state and collects degraded[]
sources that failed partially.

No umrs.api or umrs.server imports — strict boundary.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from umrs.knowledge.store import KnowledgeStore
from umrs.knowledge.verification_store import (
    VerificationReport,
    VerificationStore,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ── enums ─────────────────────────────────────────────────────────────────────


class HealthLevel(str, enum.Enum):
    """Overall project health."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ActionType(str, enum.Enum):
    """Derived action categories."""
    RUN_LINT = "run_lint"
    RESOLVE_FINDINGS = "resolve_findings"
    APPROVE = "approve"
    REINDEX = "reindex"
    INGEST = "ingest"
    REVIEW = "review"


# ── dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class ArtifactSummary:
    """Counts by kind and status."""
    total: int = 0
    active: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)


@dataclass
class GraphSummary:
    """Graph node/edge counts."""
    node_count: int = 0
    edge_count: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class VerificationSummary:
    """Aggregated verification state."""
    total_reports: int = 0
    latest_report: Optional[VerificationReport] = None
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    skipped_checks: int = 0
    pass_rate: float = 0.0
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    findings_by_status: dict[str, int] = field(default_factory=dict)
    degraded_reports: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)


@dataclass
class NextAction:
    """A single derived action item."""
    action: str  # ActionType value
    priority: int  # 1 = highest
    description: str
    source: str  # which subsystem produced it


@dataclass
class DashboardPayload:
    """Full dashboard aggregation payload."""
    project_id: str
    timestamp: str
    health: HealthLevel
    artifacts: ArtifactSummary
    graph: GraphSummary
    verification: VerificationSummary
    next_actions: list[NextAction] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    score: float = 0.0  # 0.0 – 1.0 composite score


# ── aggregation engine ────────────────────────────────────────────────────────


class KnowledgeSummaryEngine:
    """Aggregate project health snapshot from durable sources."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        verification_store: VerificationStore,
        project_id: str,
    ) -> None:
        self._ks = knowledge_store
        self._vs = verification_store
        self._project_id = project_id

    async def compute(self) -> DashboardPayload:
        """Compute dashboard payload from all sources.

        Deterministic: iteration order does not affect results.
        """
        artifacts = await self._aggregate_artifacts()
        graph = await self._aggregate_graph()
        verification = await self._aggregate_verification()
        next_actions = self._derive_next_actions(verification, artifacts, graph)
        degraded = self._compute_degraded(verification, artifacts, graph)
        health = self._compute_health(verification, next_actions)
        score = self._compute_score(verification, artifacts, graph)

        return DashboardPayload(
            project_id=self._project_id,
            timestamp=_utcnow().isoformat(),
            health=health,
            artifacts=artifacts,
            graph=graph,
            verification=verification,
            next_actions=sorted(next_actions, key=lambda a: a.priority),
            degraded=sorted(degraded),
            score=round(score, 4),
        )

    # -- artifact aggregation -------------------------------------------------

    async def _aggregate_artifacts(self) -> ArtifactSummary:
        """Aggregate artifact counts from KnowledgeStore."""
        result = ArtifactSummary()
        try:
            artifacts, _ = await self._ks.list_artifacts(
                project_id=self._project_id, status="active"
            )
            result.total = len(artifacts)
            for art in artifacts:
                if art.status == "active":
                    result.active += 1
                kind = art.kind or "unknown"
                result.by_kind[kind] = result.by_kind.get(kind, 0) + 1
        except Exception:
            logger.warning("Artifact aggregation failed for %s", self._project_id, exc_info=True)
        return result

    # -- graph aggregation ----------------------------------------------------

    async def _aggregate_graph(self) -> GraphSummary:
        """Aggregate graph node/edge counts from KnowledgeStore."""
        result = GraphSummary()
        try:
            nodes = await self._ks.list_nodes(project_id=self._project_id)
            edges = await self._ks.list_edges(project_id=self._project_id)
            result.node_count = len(nodes)
            result.edge_count = len(edges)
            for node in nodes:
                ntype = node.node_type or "unknown"
                result.by_type[ntype] = result.by_type.get(ntype, 0) + 1
        except Exception:
            logger.warning("Graph aggregation failed for %s", self._project_id, exc_info=True)
        return result

    # -- verification aggregation ----------------------------------------------

    async def _aggregate_verification(self) -> VerificationSummary:
        """Aggregate verification reports and findings."""
        result = VerificationSummary()
        try:
            reports = await self._vs.list_reports(project_id=self._project_id)
            result.total_reports = len(reports)

            # Latest report: newest by created_at (list_reports already returns newest first)
            if reports:
                result.latest_report = reports[0]
                result.total_checks = reports[0].total_checks
                result.passed_checks = reports[0].passed_checks
                result.failed_checks = reports[0].failed_checks
                result.skipped_checks = reports[0].skipped_checks
                if reports[0].total_checks > 0:
                    result.pass_rate = round(
                        reports[0].passed_checks / reports[0].total_checks, 4
                    )
                if reports[0].degraded:
                    result.degraded_reports.append(reports[0].spec_name)
                    if reports[0].degraded_reason:
                        result.degraded_reasons.append(reports[0].degraded_reason)

            # Aggregate across ALL reports for severity/status stats
            all_findings = await self._vs.list_findings(project_id_or_report=None)  # type: ignore[arg-type]
            # Filter to project-level reports (match project_id from reports)
            report_ids = {r.report_id for r in reports}
            project_findings = [f for f in all_findings if f.report_id in report_ids]

            for finding in project_findings:
                sev = finding.severity or "unknown"
                result.findings_by_severity[sev] = result.findings_by_severity.get(sev, 0) + 1
                status = finding.status or "unknown"
                result.findings_by_status[status] = result.findings_by_status.get(status, 0) + 1
        except Exception:
            logger.warning("Verification aggregation failed for %s", self._project_id, exc_info=True)
        return result

    # -- next_actions derivation -----------------------------------------------

    @staticmethod
    def _resolve_findings_labels(
        open_finding_count: int, blocking_count: int, pass_rate: float
    ) -> list[str]:
        """Build the label list for the resolve_findings action."""
        labels = []
        if open_finding_count > 0:
            labels.append(f"{open_finding_count} open")
        if blocking_count > 0:
            labels.append(f"{blocking_count} blocking")
        if 0 < pass_rate < 0.5:
            labels.append(f"pass rate {pass_rate:.0%}")
        return labels

    def _derive_verification_actions(
        self, verification: VerificationSummary, add
    ) -> None:
        """Derive priority-1/2 actions from verification state."""
        open_finding_count = (
            verification.findings_by_status.get("open", 0)
            + verification.findings_by_status.get("acknowledged", 0)
        )
        blocking_count = verification.findings_by_severity.get("blocking", 0)
        pass_rate = verification.pass_rate

        labels = self._resolve_findings_labels(
            open_finding_count, blocking_count, pass_rate
        )
        if labels:
            add(
                ActionType.RESOLVE_FINDINGS, 1,
                f"Resolve findings: {', '.join(sorted(labels))}",
                "verification",
            )

        if 0.5 <= pass_rate < 0.8:
            add(
                ActionType.REVIEW, 2,
                f"Review knowledge quality (pass rate {pass_rate:.0%})",
                "verification",
            )
        if pass_rate >= 1.0 and open_finding_count == 0 and blocking_count == 0:
            add(
                ActionType.APPROVE, 2,
                "All checks passed — ready for approval",
                "verification",
            )
        if verification.degraded_reports:
            add(
                ActionType.RUN_LINT, 2,
                f"Run lint on {len(verification.degraded_reports)} degraded source(s)",
                "verification",
            )

    def _derive_next_actions(
        self,
        verification: VerificationSummary,
        artifacts: ArtifactSummary,
        graph: GraphSummary,
    ) -> list[NextAction]:
        """Derive next_actions from the current state.

        Derivation table:

        | Condition | next_action | priority |
        |-----------|-------------|----------|
        | open/ack findings > 0 | resolve_findings | 1 |
        | any blocking findings | resolve_findings | 1 |
        | pass_rate < 0.5 | resolve_findings | 1 |
        | pass_rate < 0.8 | review | 2 |
        | pass_rate >= 1.0 and no open | approve | 2 |
        | node_count == 0 and artifact_count > 0 | reindex | 3 |
        | artifact_count == 0 | ingest | 4 |
        | degraded_reports | run_lint | 2 |
        """
        actions: list[NextAction] = []
        seen: set[str] = set()

        def _add(action: str, priority: int, description: str, source: str) -> None:
            key = f"{action}:{priority}"
            if key not in seen:
                seen.add(key)
                actions.append(NextAction(
                    action=action,
                    priority=priority,
                    description=description,
                    source=source,
                ))

        self._derive_verification_actions(verification, _add)

        # Priority 3: Reindex if graph empty but artifacts exist
        if graph.node_count == 0 and artifacts.total > 0:
            _add(
                ActionType.REINDEX, 3,
                f"Reindex {artifacts.total} artifact(s) — graph is empty",
                "graph",
            )

        # Priority 4: Ingest if no artifacts at all
        if artifacts.total == 0:
            _add(
                ActionType.INGEST, 4,
                "No artifacts — ingest knowledge sources",
                "artifacts",
            )

        return actions

    # -- degraded sources -----------------------------------------------------

    def _compute_degraded(
        self,
        verification: VerificationSummary,
        artifacts: ArtifactSummary,
        graph: GraphSummary,
    ) -> list[str]:
        """Collect degraded sources that failed partially."""
        degraded: list[str] = []

        if verification.degraded_reports:
            for spec in verification.degraded_reports:
                reason = ""
                idx = verification.degraded_reports.index(spec)
                if idx < len(verification.degraded_reasons):
                    reason = f": {verification.degraded_reasons[idx]}"
                degraded.append(f"verification/{spec}{reason}")

        if graph.node_count == 0 and artifacts.total > 0:
            degraded.append("graph: 0 nodes with existing artifacts — indexing may have failed")

        if artifacts.total == 0:
            degraded.append("artifacts: no artifacts registered — check ingestion")

        return degraded

    # -- health level ----------------------------------------------------------

    def _compute_health(
        self,
        verification: VerificationSummary,
        next_actions: list[NextAction],
    ) -> HealthLevel:
        """Derive overall health from critical conditions."""
        blocking = verification.findings_by_severity.get("blocking", 0)
        critical_actions = [a for a in next_actions if a.priority == 1]
        degraded_sources = len(verification.degraded_reports)

        if blocking > 0 or (critical_actions and not verification.latest_report):
            return HealthLevel.CRITICAL
        if degraded_sources > 0 or len(next_actions) > 0:
            return HealthLevel.DEGRADED
        return HealthLevel.HEALTHY

    # -- composite score -------------------------------------------------------

    def _compute_score(
        self,
        verification: VerificationSummary,
        artifacts: ArtifactSummary,
        graph: GraphSummary,
    ) -> float:
        """Compute composite health score 0.0–1.0.

        Weighted factors:
          - verification pass_rate: 40%
          - artifact presence: 20%
          - graph coverage: 20%
          - findings penalty: 20%
        """
        # Verification factor (0–1)
        verif = verification.pass_rate if verification.total_checks > 0 else 0.0

        # Artifact factor: scale 0–1 based on known-good threshold
        artifact_score = min(1.0, artifacts.total / 10)  # 10+ = full marks

        # Graph factor: nodes/edges relative to artifacts
        if artifacts.total > 0:
            graph_score = min(1.0, graph.node_count / max(1, artifacts.total))
        else:
            graph_score = 1.0  # no artifacts expected

        # Findings penalty (0–1, deducted by open/binding)
        open_count = (
            verification.findings_by_status.get("open", 0)
            + verification.findings_by_status.get("acknowledged", 0)
        )
        blocking_count = verification.findings_by_severity.get("blocking", 0)
        findings_penalty = min(1.0, max(0.0, 1.0 - (open_count * 0.1) - (blocking_count * 0.3)))

        score = (
            verif * 0.40
            + artifact_score * 0.20
            + graph_score * 0.20
            + findings_penalty * 0.20
        )
        return round(min(1.0, max(0.0, score)), 4)


# ── module-level convenience --------------------------------------------------


async def compute_dashboard(
    knowledge_store: KnowledgeStore,
    verification_store: VerificationStore,
    project_id: str,
) -> DashboardPayload:
    """Compute a dashboard payload for a project.

    Convenience wrapper over KnowledgeSummaryEngine.
    """
    engine = KnowledgeSummaryEngine(knowledge_store, verification_store, project_id)
    return await engine.compute()
