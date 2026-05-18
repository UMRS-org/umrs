"""Model routing feedback store (MEM-D2).

Records which model was selected for each step, the outcome, and whether
the user accepted or corrected the result.  Feeds into scoring for future
routing decisions (§4.14 Step Routing).

This module lives in ``umrs.jobs`` because it persists durable state tied
to jobs and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class RoutingFeedback:
    """Record of a model routing decision and its outcome."""

    feedback_id: str
    job_id: str
    step_name: str
    model_selected: str

    # Outcome
    outcome: str = ""  # succeeded | failed | timeout | rejected
    user_accepted: bool | None = None  # None = no feedback yet
    user_correction: str | None = None  # model the user preferred instead

    # Context for routing
    step_type: str = ""  # llm | embedding | tool | rag
    routing_reason: str = ""  # why this model was selected
    latency_ms: float | None = None
    tokens_used: int | None = None
    quality_score: float | None = None  # 0.0-1.0, user or auto rating

    project_id: str = ""
    user_id: str = ""
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "job_id": self.job_id,
            "step_name": self.step_name,
            "model_selected": self.model_selected,
            "outcome": self.outcome,
            "user_accepted": self.user_accepted,
            "user_correction": self.user_correction,
            "step_type": self.step_type,
            "routing_reason": self.routing_reason,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "quality_score": self.quality_score,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "created_at": _iso(self.created_at),
        }


@dataclass
class ModelRoutingSummary:
    """Aggregated routing statistics for a model."""

    model_name: str
    total_selections: int = 0
    succeeded: int = 0
    failed: int = 0
    user_accepted: int = 0
    user_rejected: int = 0
    user_corrected: int = 0
    avg_latency_ms: float | None = None
    avg_quality_score: float | None = None
    acceptance_rate: float = 0.0
    correction_targets: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_selections": self.total_selections,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "user_accepted": self.user_accepted,
            "user_rejected": self.user_rejected,
            "user_corrected": self.user_corrected,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_quality_score": self.avg_quality_score,
            "acceptance_rate": self.acceptance_rate,
            "correction_targets": self.correction_targets,
        }


VALID_OUTCOMES = {"succeeded", "failed", "timeout", "rejected", ""}


class RoutingFeedbackStore:
    """Durable store for model routing feedback.

    Postgres-backed with in-memory fallback for dev/test.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2
                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning("psycopg2 unavailable; using in-memory store")
        self._mem: dict[str, RoutingFeedback] = {}

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def _get_conn(self):
        return self._psycopg2.connect(self._db_url)

    # =========================================================================
    # Record feedback
    # =========================================================================

    def record_feedback(
        self,
        job_id: str,
        step_name: str,
        model_selected: str,
        *,
        outcome: str = "",
        user_accepted: bool | None = None,
        user_correction: str | None = None,
        step_type: str = "",
        routing_reason: str = "",
        latency_ms: float | None = None,
        tokens_used: int | None = None,
        quality_score: float | None = None,
        project_id: str = "",
        user_id: str = "",
    ) -> RoutingFeedback:
        """Record a model routing decision and its outcome."""
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Invalid outcome: {outcome!r}; must be one of {VALID_OUTCOMES}")

        fb = RoutingFeedback(
            feedback_id=f"rfb-{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            step_name=step_name,
            model_selected=model_selected,
            outcome=outcome,
            user_accepted=user_accepted,
            user_correction=user_correction,
            step_type=step_type,
            routing_reason=routing_reason,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            quality_score=quality_score,
            project_id=project_id,
            user_id=user_id,
        )

        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO routing_feedback
                            (feedback_id, job_id, step_name, model_selected,
                             outcome, user_accepted, user_correction,
                             step_type, routing_reason, latency_ms,
                             tokens_used, quality_score,
                             project_id, user_id, created_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            fb.feedback_id, fb.job_id, fb.step_name,
                            fb.model_selected, fb.outcome, fb.user_accepted,
                            fb.user_correction, fb.step_type, fb.routing_reason,
                            fb.latency_ms, fb.tokens_used, fb.quality_score,
                            fb.project_id, fb.user_id, fb.created_at,
                        ),
                    )
                conn.commit()
            finally:
                conn.close()
        else:
            self._mem[fb.feedback_id] = fb

        return fb

    def update_user_feedback(
        self,
        feedback_id: str,
        user_accepted: bool,
        user_correction: str | None = None,
        quality_score: float | None = None,
    ) -> bool:
        """Update user acceptance/correction for a routing decision."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE routing_feedback
                        SET user_accepted = %s,
                            user_correction = %s,
                            quality_score = COALESCE(%s, quality_score)
                        WHERE feedback_id = %s
                        """,
                        (user_accepted, user_correction, quality_score, feedback_id),
                    )
                    updated = cur.rowcount > 0
                conn.commit()
                return updated
            finally:
                conn.close()

        fb = self._mem.get(feedback_id)
        if fb:
            fb.user_accepted = user_accepted
            fb.user_correction = user_correction
            if quality_score is not None:
                fb.quality_score = quality_score
            return True
        return False

    # =========================================================================
    # Query
    # =========================================================================

    def get_feedback(self, feedback_id: str) -> RoutingFeedback | None:
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM routing_feedback WHERE feedback_id = %s", (feedback_id,))
                    row = cur.fetchone()
                    if not row:
                        return None
                    return self._row_to_feedback(cur, row)
            finally:
                conn.close()
        return self._mem.get(feedback_id)

    def list_feedback(
        self,
        model_name: str | None = None,
        job_id: str | None = None,
        step_type: str | None = None,
        limit: int = 50,
    ) -> list[RoutingFeedback]:
        """List routing feedback with optional filters."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    conditions = []
                    params: list[Any] = []
                    if model_name:
                        conditions.append("model_selected = %s")
                        params.append(model_name)
                    if job_id:
                        conditions.append("job_id = %s")
                        params.append(job_id)
                    if step_type:
                        conditions.append("step_type = %s")
                        params.append(step_type)
                    where = ""
                    if conditions:
                        where = "WHERE " + " AND ".join(conditions)
                    params.append(limit)
                    cur.execute(
                        f"SELECT * FROM routing_feedback {where} ORDER BY created_at DESC LIMIT %s",
                        params,
                    )
                    return [self._row_to_feedback(cur, r) for r in cur.fetchall()]
            finally:
                conn.close()

        results = list(self._mem.values())
        if model_name:
            results = [f for f in results if f.model_selected == model_name]
        if job_id:
            results = [f for f in results if f.job_id == job_id]
        if step_type:
            results = [f for f in results if f.step_type == step_type]
        results.sort(key=lambda f: f.created_at, reverse=True)
        return results[:limit]

    def summarize_model(self, model_name: str) -> ModelRoutingSummary:
        """Build aggregated routing summary for a model."""
        feedbacks = self.list_feedback(model_name=model_name, limit=10000)

        total = len(feedbacks)
        succeeded = sum(1 for f in feedbacks if f.outcome == "succeeded")
        failed = sum(1 for f in feedbacks if f.outcome in ("failed", "timeout"))
        accepted = sum(1 for f in feedbacks if f.user_accepted is True)
        rejected = sum(1 for f in feedbacks if f.user_accepted is False and not f.user_correction)
        corrected = sum(1 for f in feedbacks if f.user_correction is not None)

        latencies = [f.latency_ms for f in feedbacks if f.latency_ms is not None]
        scores = [f.quality_score for f in feedbacks if f.quality_score is not None]

        correction_targets: dict[str, int] = {}
        for f in feedbacks:
            if f.user_correction:
                correction_targets[f.user_correction] = correction_targets.get(f.user_correction, 0) + 1

        feedback_count = accepted + rejected + corrected
        acceptance_rate = accepted / feedback_count if feedback_count > 0 else 0.0

        return ModelRoutingSummary(
            model_name=model_name,
            total_selections=total,
            succeeded=succeeded,
            failed=failed,
            user_accepted=accepted,
            user_rejected=rejected,
            user_corrected=corrected,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else None,
            avg_quality_score=sum(scores) / len(scores) if scores else None,
            acceptance_rate=round(acceptance_rate, 4),
            correction_targets=correction_targets,
        )

    def delete_feedback(self, feedback_id: str) -> bool:
        """Delete a single feedback record."""
        if self.db_enabled:
            conn = self._get_conn()
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM routing_feedback WHERE feedback_id = %s", (feedback_id,))
                    deleted = cur.rowcount > 0
                conn.commit()
                return deleted
            finally:
                conn.close()
        return self._mem.pop(feedback_id, None) is not None

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _row_to_feedback(cur, row) -> RoutingFeedback:
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        return RoutingFeedback(
            feedback_id=d["feedback_id"],
            job_id=d["job_id"],
            step_name=d["step_name"],
            model_selected=d["model_selected"],
            outcome=d.get("outcome", ""),
            user_accepted=d.get("user_accepted"),
            user_correction=d.get("user_correction"),
            step_type=d.get("step_type", ""),
            routing_reason=d.get("routing_reason", ""),
            latency_ms=d.get("latency_ms"),
            tokens_used=d.get("tokens_used"),
            quality_score=d.get("quality_score"),
            project_id=d.get("project_id", ""),
            user_id=d.get("user_id", ""),
            created_at=d.get("created_at", _now()),
        )
