"""Durable workflow_run job adapter hooks."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from collections.abc import Callable
from typing import Any

from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)


class WorkflowRunBackend(OrchestratorBackend):
    """Adapter hook for routing workflow_run jobs through JobService."""

    def __init__(self, submitter: Callable[[Job, JobAttempt], str] | None = None):
        self._submitter = submitter

    @property
    def backend_name(self) -> str:
        return "workflow-run"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        if self._submitter is not None:
            return self._submitter(job, attempt)
        execution_id = job.spec.get("execution_id", attempt.attempt_id)
        workflow_name = job.spec.get("workflow_name")
        if not isinstance(workflow_name, str) or not workflow_name.strip():
            raise ValueError("workflow_run spec.workflow_name is required")

        from umrs.shared.workflow_execution_access import (
            _launch_workflow_from_catalog,
            durable_job_verified_project,
        )

        source_path = job.spec.get("source_path")
        if (
            not isinstance(source_path, str)
            or not source_path.strip()
            or not os.path.isdir(source_path)
        ):
            source_path = None

        _launch_workflow_from_catalog(
            workflow_name=workflow_name,
            client_id=job.user_id,
            params=job.spec.get("params", {}),
            execution_id=execution_id,
            durable_job_id=job.job_id,
            source_path=source_path,
            # ADR-080: a re-resolution admits a project share only for the
            # job's own project the server recorded as verified.
            caller_project=durable_job_verified_project(job.project_id, job.spec),
        )
        return f"workflow_run|execution_id={execution_id}"


class WorkflowRunStatusChecker:
    """Report workflow_run attempt status from the execution registry."""

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        execution_id = _execution_id_from_attempt_or_job(attempt, job)
        from umrs.shared.workflow_execution_access import get_execution_registry

        record = get_execution_registry().get(execution_id)
        if record is None:
            return AttemptCheckResult.running(
                detail=f"waiting for workflow execution {execution_id}"
            )

        data = record.to_dict()
        metrics = {
            "execution_id": execution_id,
            "progress": data.get("progress", 0.0),
            "step_count": len(data.get("steps", [])),
        }
        status = data.get("status")
        if status == "succeeded":
            artifacts = _persist_run_artifacts(record=record, job=job, attempt=attempt)
            if artifacts:
                metrics["artifacts"] = artifacts
                await _record_run_artifacts(
                    job=job, attempt=attempt, artifacts=artifacts, record=record
                )
            return AttemptCheckResult.succeeded(metrics=metrics, commit=True)
        if status == "failed":
            return AttemptCheckResult.failed(data.get("error") or "workflow failed")
        if status == "cancelled":
            return AttemptCheckResult.failed("workflow cancelled")
        return AttemptCheckResult.running(detail=f"workflow {status or 'running'}")


def _execution_id_from_attempt_or_job(attempt: JobAttempt, job: Job) -> str:
    if (
        isinstance(job.spec.get("execution_id"), str)
        and job.spec["execution_id"].strip()
    ):
        return job.spec["execution_id"].strip()
    if attempt.backend_ref and "execution_id=" in attempt.backend_ref:
        return attempt.backend_ref.rsplit("execution_id=", 1)[1]
    return attempt.attempt_id


def cancel_workflow_execution_for_job(job: Job) -> None:
    """Cooperatively cancel the in-process workflow record for a durable job."""
    execution_id = job.spec.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id.strip():
        return

    from umrs.shared.workflow_execution_access import get_execution_registry

    get_execution_registry().cancel(execution_id.strip())


# The cancel order and error policy every cancel route shares (peer review
# F2, 2026-09-26). The durable job is cancelled first: a re-dispatch racing
# the cancel then either finds it cancelled (``create_attempt`` refuses it,
# or the launch's ``durable_job_cancelled`` check does), or the registry
# cancel that follows finds the record the dispatch registered. An ended
# durable job (InvalidTransitionError) still has its run cancelled: the run
# may have left step jobs for a retry that will not come. Any other store
# error propagates and leaves the run alone, because a registry-only cancel
# is a rerun: the monitor reports the cancelled run failed and
# ``mark_attempt_failed`` requeues the still-live durable job.


def cancel_job_and_run(store: Any, job: Job, **cancel_kwargs: Any) -> None:
    """Cancel durable *job* through *store* (``cancel_kwargs`` go to
    ``cancel_job``), then a workflow_run job's registry record. An
    InvalidTransitionError is raised after the registry cancel."""
    from .store import InvalidTransitionError

    try:
        store.cancel_job(job.job_id, **cancel_kwargs)
    except InvalidTransitionError:
        _cancel_run_of(job)
        raise
    _cancel_run_of(job)


def _cancel_run_of(job: Job) -> None:
    if job.job_type == JobType.WORKFLOW_RUN:
        cancel_workflow_execution_for_job(job)


def cancel_workflow_execution(execution_id: str, *, store: Any, job: Job | None) -> Any:
    """Cancel workflow execution *execution_id*: its durable *job* when there
    is one (already ended is fine), then its registry record whatever an
    earlier lookup found, since a dispatch may have registered it since.
    Returns the registry's record, None when it has none."""
    from umrs.shared.workflow_execution_access import get_execution_registry

    from .store import InvalidTransitionError

    if job is not None and store is not None:
        with contextlib.suppress(InvalidTransitionError):
            store.cancel_job(job.job_id, reason="workflow_execution_cancelled")
    return get_execution_registry().cancel(execution_id)


async def _record_run_artifacts(
    *,
    job: Job,
    attempt: JobAttempt,
    artifacts: list[dict[str, Any]],
    record: Any,
) -> None:
    """Conserve successful run artifacts in the a2l ledger (ADR-034).

    Fire-and-forget: the active recorder (Null when unconfigured) degrades
    silently; workflow success itself is never gated on conservation.
    """
    from umrs.a2l.recorder import get_recorder

    recorder = get_recorder()
    for item in artifacts:
        uri = item.get("source_ref")
        digest = item.get("content_hash")
        if not uri or not digest:
            continue
        await recorder.record_attempt_artifact(
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            artifact_uri=uri,
            artifact_type="workflow_result",
            user_id=job.user_id or getattr(record, "user_id", "") or "unknown",
            content_hash=digest,
            project_id=job.project_id or None,
            metadata={
                "name": item.get("name", ""),
                "size_bytes": item.get("size_bytes", 0),
                "registry_id": item.get("registry_id", ""),
            },
        )


# ── Attempt-scoped artifact persistence ─────────────────────────────────────
#
# Workflow steps upload outputs into the in-memory record blob store
# (``record.results["blobs"]``); nothing durably persisted them, so the
# auto-ingest pipeline had no artifacts to consume.  On success we now:
#   1. upload each blob to S3 under ``attempt.output_prefix``
#      (``jobs/{job_id}/attempt_{n}/{key}`` — the attempt-scoped invariant),
#   2. register an Artifact row (scientific catalog) via ``ArtifactStore``,
#   3. emit ingest-shaped descriptors into ``metrics["artifacts"]`` for
#      ``jobs.knowledge_adapters._artifacts_from_attempt_metrics``.


def _resolve_s3(job: Job, s3_client: Any) -> tuple[Any, str]:
    """Return (S3 client, bucket): injected, project-bound (ADR-077) or env-built.

    A bound project whose binding cannot be read answers ``(None, bucket)``:
    persistence is skipped (logged) rather than written to another store.
    """
    from .knowledge_adapters import _resolve_bucket

    if s3_client is not None:
        return s3_client, _resolve_bucket(job.spec)
    from umrs.shared.storage.project_storage import (
        ProjectStorageUnavailable,
        project_s3_config,
    )

    try:
        bound = project_s3_config(job.project_id)
    except ProjectStorageUnavailable:
        logger.warning("storage binding unreadable for project %s", job.project_id)
        return None, _resolve_bucket(job.spec)
    if bound is not None:
        return bound.get_client(), bound.bucket
    from .knowledge_adapters import _build_s3_client

    return _build_s3_client(job.spec)


def _close_quietly(conn: Any) -> None:
    """Close a PG connection, ignoring errors."""
    if conn is None:
        return
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def _persist_run_artifacts(
    *,
    record: Any,
    job: Job,
    attempt: JobAttempt,
    s3_client: Any = None,
    artifact_store: Any = None,
) -> list[dict[str, Any]]:
    """Persist execution blobs as S3 artifacts + registry rows.

    Returns ingest-shaped descriptors (artifact_id, content_hash, kind,
    source_ref).  Empty when there is nothing to persist or persistence is
    degraded (S3 unreachable): the workflow result itself is unaffected.
    """
    blobs = (record.results or {}).get("blobs") or {}
    if not blobs:
        return []

    client, bucket = _resolve_s3(job, s3_client)
    if client is None:
        logger.warning(
            "S3 unavailable — workflow blobs not persisted for job %s", job.job_id
        )
        return []

    store, owned_conn = (
        _open_artifact_store() if artifact_store is None else (artifact_store, None)
    )
    ctx = _RunContext(job=job, attempt=attempt, record=record)
    try:
        prefix = attempt.output_prefix
        return [
            info
            for full_key, body in blobs.items()
            if (
                info := _persist_one_blob(
                    client,
                    store,
                    ctx,
                    bucket=bucket,
                    prefix=prefix,
                    full_key=full_key,
                    body=body,
                )
            )
        ]
    finally:
        _close_quietly(owned_conn)


class _RunContext:
    """Job/attempt/record bundle carried through blob persistence."""

    __slots__ = ("attempt", "job", "record")

    def __init__(self, job: Job, attempt: JobAttempt, record: Any) -> None:
        self.job = job
        self.attempt = attempt
        self.record = record


def _persist_one_blob(
    client: Any,
    store: Any,
    ctx: _RunContext,
    *,
    bucket: str,
    prefix: str,
    full_key: str,
    body: Any,
) -> dict[str, Any] | None:
    """Persist one blob: S3 put + ArtifactStore row + ingest descriptor."""
    if not isinstance(body, (bytes, bytearray)):
        return None
    payload = bytes(body)
    # Blob keys are "<bucket>/<step-key>" from the in-memory shim; keep only
    # the step key under the attempt-scoped prefix.
    key = full_key.split("/", 1)[1] if "/" in full_key else full_key
    s3_key = f"{prefix}/{key}"
    digest = hashlib.sha256(payload).hexdigest()
    try:
        client.put_object(Bucket=bucket, Key=s3_key, Body=payload)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Persist workflow blob failed for %s: %s", s3_key, exc)
        return None
    registry_id = _register_result_artifact(
        store,
        ctx,
        bucket=bucket,
        key=key,
        s3_key=s3_key,
        size=len(payload),
        digest=digest,
    )
    return {
        "artifact_id": f"wfart-{digest[:16]}",
        "registry_id": registry_id,
        "content_hash": digest,
        "kind": "s3_uri",
        "source_ref": f"s3://{bucket}/{s3_key}",
        "name": key,
        "size_bytes": len(payload),
    }


def _open_artifact_store() -> tuple[Any, Any]:
    """Open an ArtifactStore on the knowledge DSN. Returns (store, conn).

    Returns (None, None) when Postgres is unavailable — the caller treats
    registry persistence as best-effort.
    """
    try:
        import psycopg2

        from umrs.artifact_registry.store import ArtifactStore
        from umrs.shared.pg_settings import pg_connect_kwargs

        from .knowledge_adapters import _resolve_knowledge_dsn

        conn = psycopg2.connect(
            _resolve_knowledge_dsn(), **pg_connect_kwargs("workflow-run")
        )
        return ArtifactStore(lambda: conn), conn
    except Exception as exc:  # noqa: BLE001
        logger.warning("Artifact registry unavailable: %s", exc)
        return None, None


def _register_result_artifact(
    store: Any,
    ctx: _RunContext,
    *,
    bucket: str,
    key: str,
    s3_key: str,
    size: int,
    digest: str,
) -> str | None:
    """Write the artifact-registry row; returns registry UUID or None."""
    if store is None:
        return None
    job = ctx.job
    attempt = ctx.attempt
    record = ctx.record
    try:
        from umrs.artifact_registry.models import Artifact

        created = store.create(
            Artifact(
                artifact_type="result",
                name=key,
                owner_id=job.user_id or record.user_id,
                project_id=job.project_id or None,
                uri=f"s3://{bucket}/{s3_key}",
                bucket=bucket,
                s3_key=s3_key,
                size_bytes=size,
                checksum=digest,
                execution_id=record.execution_id,
                workflow_id=record.workflow_name,
                job_id=job.job_id,
                metadata={
                    "attempt_no": attempt.attempt_no,
                    "attempt_id": attempt.attempt_id,
                    "registered_by": "workflow_run_checker",
                },
            )
        )
        return str(created.id) if created.id else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Artifact registry write failed for %s: %s", s3_key, exc)
        return None
