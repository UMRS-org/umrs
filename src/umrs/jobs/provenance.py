"""Provenance assembly from durable job/attempt data.

Reads completed (or in-flight) jobs and their attempts from the JobStore,
enriches them with model metadata extracted from the job spec, and returns
a :class:`r4_contracts.provenance.ProvenanceRecord`.

This module lives in ``umrs.jobs`` because it reads from the job store
and does not import from ``umrs.api`` or ``umrs.server``.

The ``build_provenance`` function is intentionally duck-typed — it reads
``.job_id``, ``.status.value``, ``.spec`` etc. from whatever objects the
caller passes, so tests can supply lightweight mock dataclasses without
triggering the ``umrs.jobs.__init__`` circular-import chain.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from r4_contracts.provenance import (
    ArtifactRef,
    AttemptSummary,
    ModelRef,
    ProvenanceRecord,
)

logger = logging.getLogger(__name__)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start and end:
        return round((end - start).total_seconds(), 3)
    return None


def _model_from_spec(spec: dict[str, Any]) -> ModelRef | None:
    """Extract model reference from job spec if present."""
    model_name = spec.get("model") or spec.get("model_name")
    model_path = spec.get("model_path")
    if not model_name and not model_path:
        return None
    return ModelRef(
        name=model_name or "",
        path=model_path,
        version=spec.get("model_version"),
        source=spec.get("model_source"),
    )


def _attempt_summary(attempt: Any) -> AttemptSummary:
    return AttemptSummary(
        attempt_id=attempt.attempt_id,
        attempt_no=attempt.attempt_no,
        backend=attempt.backend,
        backend_ref=attempt.backend_ref,
        status=attempt.status.value
        if hasattr(attempt.status, "value")
        else str(attempt.status),
        started_at=_iso(attempt.started_at),
        ended_at=_iso(attempt.ended_at),
        exit_reason=attempt.exit_reason,
        duration_seconds=_duration_seconds(attempt.started_at, attempt.ended_at),
    )


def _artifacts_from_attempts(job: Any, attempts: list[Any]) -> list[ArtifactRef]:
    """Collect artifact references from committed attempt log_uri and metrics."""
    artifacts: list[ArtifactRef] = []
    commit_id = job.commit_attempt_id

    for att in attempts:
        # Only include artifacts from the committed attempt (or all if none committed)
        if commit_id and att.attempt_id != commit_id:
            continue

        if att.log_uri:
            artifacts.append(
                ArtifactRef(uri=att.log_uri, media_type="application/x-log")
            )

        # Look for artifact refs stored in attempt metrics
        for artifact_entry in att.metrics.get("artifacts", []):
            ref = _artifact_ref(artifact_entry)
            if ref is not None:
                artifacts.append(ref)

    return artifacts


def _artifact_ref(entry: Any) -> ArtifactRef | None:
    """One ``metrics["artifacts"]`` entry as an ArtifactRef.

    Producers write the ingest descriptor shape (``source_ref``,
    ``content_hash``, ``kind``); this reader took only ``uri``, so provenance
    reported no artifact for any workflow run or ingest (plan §3c.2,
    measured 2026-10-09). Both shapes are read.
    """
    if not isinstance(entry, dict):
        return None
    uri = entry.get("uri") or entry.get("source_ref")
    if not uri:
        return None
    return ArtifactRef(
        uri=str(uri),
        media_type=str(entry.get("media_type") or entry.get("kind") or ""),
        size_bytes=entry.get("size_bytes"),
        sha256=entry.get("sha256") or entry.get("content_hash") or None,
    )


def build_provenance(job: Any, attempts: list[Any]) -> ProvenanceRecord:
    """Build a provenance record from a job and its attempts.

    This is a pure assembly function — no database access. The caller
    provides the job and attempts.
    """
    completed_at: datetime | None = None
    for att in attempts:
        if att.ended_at and (completed_at is None or att.ended_at > completed_at):
            completed_at = att.ended_at

    # Determine execution authority from backend of first attempt
    authority = attempts[0].backend if attempts else None

    # Workflow-specific fields
    spec = job.spec or {}
    workflow_name = spec.get("workflow_name")
    execution_id = spec.get("execution_id")
    parent_job_id = spec.get("parent_job_id")

    return ProvenanceRecord(
        job_id=job.job_id,
        job_type=job.job_type.value
        if hasattr(job.job_type, "value")
        else str(job.job_type),
        project_id=job.project_id,
        user_id=job.user_id,
        status=job.status.value if hasattr(job.status, "value") else str(job.status),
        spec=spec,
        created_at=_iso(job.created_at),
        completed_at=_iso(completed_at),
        total_duration_seconds=_duration_seconds(job.created_at, completed_at),
        execution_authority=authority,
        umrs_instance_id=attempts[-1].umrs_instance_id if attempts else None,
        model=_model_from_spec(spec),
        attempts=[_attempt_summary(a) for a in attempts],
        total_attempts=len(attempts),
        commit_attempt_id=job.commit_attempt_id,
        artifacts=_artifacts_from_attempts(job, attempts),
        restart_policy=(
            job.restart_policy.value
            if hasattr(job.restart_policy, "value")
            else str(job.restart_policy)
        ),
        max_attempts=job.desired_max_attempts,
        idempotency_key=job.idempotency_key,
        workflow_name=workflow_name,
        execution_id=execution_id,
        parent_job_id=parent_job_id,
    )


def get_provenance(store: Any, job_id: str) -> ProvenanceRecord:
    """Load a job and its attempts from the store and build provenance.

    ``store`` must be a :class:`umrs.jobs.store.JobStore` (or duck-typed
    equivalent with ``get_job`` and ``get_attempts_for_job`` methods).
    """
    job = store.get_job(job_id)
    attempts = store.get_attempts_for_job(job_id)
    return build_provenance(job, attempts)
