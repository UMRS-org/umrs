"""Workflow outcome tracking store (MEM-D1).

Provides durable persistence for structured workflow execution outcomes,
enabling per-template and per-model analytics (success rate, duration,
failure breakdown, resource usage).

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to jobs and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_outcome_id() -> str:
    return f"wfo-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class StepMetric:
    """Per-step execution metric within a workflow run."""

    step_name: str
    status: str = ""
    duration_sec: float | None = None
    model: str | None = None
    tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "status": self.status,
            "duration_sec": self.duration_sec,
            "model": self.model,
            "tokens": self.tokens,
        }


@dataclass
class WorkflowOutcome:
    """Structured outcome record for a single workflow execution."""

    outcome_id: str
    job_id: str
    workflow_name: str
    user_id: str
    project_id: str
    status: str  # succeeded | failed | cancelled

    workflow_version: str | None = None
    execution_duration_sec: float | None = None
    step_count: int | None = None
    failed_step_name: str | None = None

    model_used: str | None = None
    model_version: str | None = None
    embedding_model: str | None = None

    total_tokens: int | None = None
    tool_call_count: int | None = None
    retrieval_doc_count: int | None = None

    resource_usage: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None
    failure_category: str | None = None  # timeout|oom|tool_error|llm_error|network|user_input

    step_metrics: list[StepMetric] = field(default_factory=list)

    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome_id": self.outcome_id,
            "job_id": self.job_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "status": self.status,
            "execution_duration_sec": self.execution_duration_sec,
            "step_count": self.step_count,
            "failed_step_name": self.failed_step_name,
            "model_used": self.model_used,
            "model_version": self.model_version,
            "embedding_model": self.embedding_model,
            "total_tokens": self.total_tokens,
            "tool_call_count": self.tool_call_count,
            "retrieval_doc_count": self.retrieval_doc_count,
            "resource_usage": self.resource_usage,
            "failure_reason": self.failure_reason,
            "failure_category": self.failure_category,
            "step_metrics": [m.to_dict() for m in self.step_metrics],
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "created_at": _iso(self.created_at),
        }


@dataclass
class OutcomeSummary:
    """Aggregated outcome summary for a workflow template."""

    workflow_name: str
    total_runs: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    success_rate: float = 0.0
    avg_duration_sec: float | None = None
    avg_tokens: float | None = None
    failure_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "total_runs": self.total_runs,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": self.success_rate,
            "avg_duration_sec": self.avg_duration_sec,
            "avg_tokens": self.avg_tokens,
            "failure_breakdown": self.failure_breakdown,
        }


VALID_STATUSES = {"succeeded", "failed", "cancelled"}
VALID_FAILURE_CATEGORIES = {
    "timeout", "oom", "tool_error", "llm_error", "network", "user_input", None,
}


class WorkflowOutcomeStore:
    """Durable workflow outcome store backed by Postgres.

    Falls back to in-memory storage when ``db_url`` is not set (dev mode).
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2  # type: ignore
                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "db_url set but psycopg2 unavailable; "
                    "workflow outcomes will use in-memory store"
                )
        self._mem: dict[str, WorkflowOutcome] = {}

    @property
    def db_enabled(self) -> bool:
        return self._db_url is not None and self._psycopg2 is not None

    def _conn(self):  # type: ignore[no-untyped-def]
        return self._psycopg2.connect(self._db_url)  # type: ignore[union-attr]

    # ── Record ──────────────────────────────────────────────

    def record_outcome(
        self,
        job_id: str,
        workflow_name: str,
        user_id: str,
        project_id: str,
        status: str,
        *,
        workflow_version: str | None = None,
        execution_duration_sec: float | None = None,
        step_count: int | None = None,
        failed_step_name: str | None = None,
        model_used: str | None = None,
        model_version: str | None = None,
        embedding_model: str | None = None,
        total_tokens: int | None = None,
        tool_call_count: int | None = None,
        retrieval_doc_count: int | None = None,
        resource_usage: dict[str, Any] | None = None,
        failure_reason: str | None = None,
        failure_category: str | None = None,
        step_metrics: list[StepMetric] | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> WorkflowOutcome:
        """Record a workflow execution outcome."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status!r}; must be one of {VALID_STATUSES}")
        if failure_category not in VALID_FAILURE_CATEGORIES:
            raise ValueError(
                f"Invalid failure_category: {failure_category!r}; "
                f"must be one of {VALID_FAILURE_CATEGORIES}"
            )

        outcome = WorkflowOutcome(
            outcome_id=_generate_outcome_id(),
            job_id=job_id,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            user_id=user_id,
            project_id=project_id,
            status=status,
            execution_duration_sec=execution_duration_sec,
            step_count=step_count,
            failed_step_name=failed_step_name,
            model_used=model_used,
            model_version=model_version,
            embedding_model=embedding_model,
            total_tokens=total_tokens,
            tool_call_count=tool_call_count,
            retrieval_doc_count=retrieval_doc_count,
            resource_usage=resource_usage or {},
            failure_reason=failure_reason,
            failure_category=failure_category,
            step_metrics=step_metrics or [],
            started_at=started_at,
            completed_at=completed_at,
        )

        if self.db_enabled:
            self._db_insert(outcome)
        else:
            self._mem[outcome.outcome_id] = outcome

        logger.info(
            "Workflow outcome recorded: %s (job=%s wf=%s status=%s)",
            outcome.outcome_id, job_id, workflow_name, status,
        )
        return outcome

    # ── Query ───────────────────────────────────────────────

    def get_outcome(self, outcome_id: str) -> WorkflowOutcome | None:
        if self.db_enabled:
            return self._db_get(outcome_id)
        return self._mem.get(outcome_id)

    def list_outcomes(
        self,
        workflow_name: str | None = None,
        model_used: str | None = None,
        user_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[WorkflowOutcome]:
        """List outcomes with optional filters."""
        if self.db_enabled:
            return self._db_list(workflow_name, model_used, user_id, status, limit)
        results = list(self._mem.values())
        if workflow_name:
            results = [o for o in results if o.workflow_name == workflow_name]
        if model_used:
            results = [o for o in results if o.model_used == model_used]
        if user_id:
            results = [o for o in results if o.user_id == user_id]
        if status:
            results = [o for o in results if o.status == status]
        results.sort(key=lambda o: o.created_at, reverse=True)
        return results[:limit]

    def summarize(self, workflow_name: str) -> OutcomeSummary:
        """Compute aggregate statistics for a workflow template."""
        if self.db_enabled:
            return self._db_summarize(workflow_name)

        outcomes = [o for o in self._mem.values() if o.workflow_name == workflow_name]
        return self._compute_summary(workflow_name, outcomes)

    def summarize_by_model(self, model_name: str) -> OutcomeSummary:
        """Compute aggregate statistics for runs using a specific model."""
        if self.db_enabled:
            return self._db_summarize_by_model(model_name)

        outcomes = [o for o in self._mem.values() if o.model_used == model_name]
        return self._compute_summary(f"model:{model_name}", outcomes)

    # ── Private: summary computation ────────────────────────

    @staticmethod
    def _compute_summary(name: str, outcomes: list[WorkflowOutcome]) -> OutcomeSummary:
        total = len(outcomes)
        if total == 0:
            return OutcomeSummary(workflow_name=name)

        succeeded = sum(1 for o in outcomes if o.status == "succeeded")
        failed = sum(1 for o in outcomes if o.status == "failed")
        cancelled = sum(1 for o in outcomes if o.status == "cancelled")

        durations = [o.execution_duration_sec for o in outcomes if o.execution_duration_sec is not None]
        tokens = [o.total_tokens for o in outcomes if o.total_tokens is not None]

        failure_cats: dict[str, int] = {}
        for o in outcomes:
            if o.status == "failed" and o.failure_category:
                failure_cats[o.failure_category] = failure_cats.get(o.failure_category, 0) + 1

        return OutcomeSummary(
            workflow_name=name,
            total_runs=total,
            succeeded=succeeded,
            failed=failed,
            cancelled=cancelled,
            success_rate=succeeded / total if total else 0.0,
            avg_duration_sec=sum(durations) / len(durations) if durations else None,
            avg_tokens=sum(tokens) / len(tokens) if tokens else None,
            failure_breakdown=failure_cats,
        )

    # ── Private: Postgres ───────────────────────────────────

    def _db_insert(self, o: WorkflowOutcome) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO workflow_outcomes
                       (outcome_id, job_id, workflow_name, workflow_version,
                        user_id, project_id, status, execution_duration_sec,
                        step_count, failed_step_name, model_used, model_version,
                        embedding_model, total_tokens, tool_call_count,
                        retrieval_doc_count, resource_usage, failure_reason,
                        failure_category, step_metrics, started_at, completed_at,
                        created_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        o.outcome_id, o.job_id, o.workflow_name, o.workflow_version,
                        o.user_id, o.project_id, o.status, o.execution_duration_sec,
                        o.step_count, o.failed_step_name, o.model_used, o.model_version,
                        o.embedding_model, o.total_tokens, o.tool_call_count,
                        o.retrieval_doc_count,
                        json.dumps(o.resource_usage),
                        o.failure_reason, o.failure_category,
                        json.dumps([m.to_dict() for m in o.step_metrics]),
                        o.started_at, o.completed_at, o.created_at,
                    ),
                )

    def _db_get(self, outcome_id: str) -> WorkflowOutcome | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM workflow_outcomes WHERE outcome_id = %s",
                    (outcome_id,),
                )
                row = cur.fetchone()
                return self._row_to_outcome(row, cur) if row else None

    def _db_list(
        self,
        workflow_name: str | None,
        model_used: str | None,
        user_id: str | None,
        status: str | None,
        limit: int,
    ) -> list[WorkflowOutcome]:
        conditions = []
        params: list[Any] = []
        if workflow_name:
            conditions.append("workflow_name = %s")
            params.append(workflow_name)
        if model_used:
            conditions.append("model_used = %s")
            params.append(model_used)
        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        if status:
            conditions.append("status = %s")
            params.append(status)

        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT * FROM workflow_outcomes WHERE {where} "
                    "ORDER BY created_at DESC LIMIT %s",
                    params,
                )
                return [self._row_to_outcome(r, cur) for r in cur.fetchall()]

    def _db_summarize(self, workflow_name: str) -> OutcomeSummary:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT
                         COUNT(*) AS total,
                         SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END),
                         SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),
                         SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END),
                         AVG(execution_duration_sec) FILTER (WHERE execution_duration_sec IS NOT NULL),
                         AVG(total_tokens) FILTER (WHERE total_tokens IS NOT NULL)
                       FROM workflow_outcomes WHERE workflow_name = %s""",
                    (workflow_name,),
                )
                row = cur.fetchone()
                total = row[0] or 0
                summary = OutcomeSummary(
                    workflow_name=workflow_name,
                    total_runs=total,
                    succeeded=row[1] or 0,
                    failed=row[2] or 0,
                    cancelled=row[3] or 0,
                    success_rate=(row[1] or 0) / total if total else 0.0,
                    avg_duration_sec=float(row[4]) if row[4] is not None else None,
                    avg_tokens=float(row[5]) if row[5] is not None else None,
                )

                # Failure breakdown
                cur.execute(
                    """SELECT failure_category, COUNT(*)
                       FROM workflow_outcomes
                       WHERE workflow_name = %s AND status = 'failed'
                         AND failure_category IS NOT NULL
                       GROUP BY failure_category""",
                    (workflow_name,),
                )
                summary.failure_breakdown = {r[0]: r[1] for r in cur.fetchall()}
                return summary

    def _db_summarize_by_model(self, model_name: str) -> OutcomeSummary:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT
                         COUNT(*) AS total,
                         SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END),
                         SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),
                         SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END),
                         AVG(execution_duration_sec) FILTER (WHERE execution_duration_sec IS NOT NULL),
                         AVG(total_tokens) FILTER (WHERE total_tokens IS NOT NULL)
                       FROM workflow_outcomes WHERE model_used = %s""",
                    (model_name,),
                )
                row = cur.fetchone()
                total = row[0] or 0
                return OutcomeSummary(
                    workflow_name=f"model:{model_name}",
                    total_runs=total,
                    succeeded=row[1] or 0,
                    failed=row[2] or 0,
                    cancelled=row[3] or 0,
                    success_rate=(row[1] or 0) / total if total else 0.0,
                    avg_duration_sec=float(row[4]) if row[4] is not None else None,
                    avg_tokens=float(row[5]) if row[5] is not None else None,
                )

    def _row_to_outcome(self, row: tuple, cur: Any) -> WorkflowOutcome:
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        resource = d.get("resource_usage", {})
        if isinstance(resource, str):
            resource = json.loads(resource)
        raw_steps = d.get("step_metrics", [])
        if isinstance(raw_steps, str):
            raw_steps = json.loads(raw_steps)
        steps = [
            StepMetric(
                step_name=s.get("step_name", ""),
                status=s.get("status", ""),
                duration_sec=s.get("duration_sec"),
                model=s.get("model"),
                tokens=s.get("tokens"),
            )
            for s in (raw_steps or [])
        ]
        return WorkflowOutcome(
            outcome_id=d["outcome_id"],
            job_id=d["job_id"],
            workflow_name=d["workflow_name"],
            workflow_version=d.get("workflow_version"),
            user_id=d["user_id"],
            project_id=d["project_id"],
            status=d["status"],
            execution_duration_sec=d.get("execution_duration_sec"),
            step_count=d.get("step_count"),
            failed_step_name=d.get("failed_step_name"),
            model_used=d.get("model_used"),
            model_version=d.get("model_version"),
            embedding_model=d.get("embedding_model"),
            total_tokens=d.get("total_tokens"),
            tool_call_count=d.get("tool_call_count"),
            retrieval_doc_count=d.get("retrieval_doc_count"),
            resource_usage=resource,
            failure_reason=d.get("failure_reason"),
            failure_category=d.get("failure_category"),
            step_metrics=steps,
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            created_at=d.get("created_at", _now()),
        )
