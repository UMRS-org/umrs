"""
Durable ZMQ bridge for job tracking commands.

This handler is the CLI-facing job path used by server_umrs.py.
It delegates JOB_* commands to a JobStore-backed durable backend.

The server prefers the process-owned durable runtime so scheduler and monitor
loops share the same store as ZMQ job commands. If that runtime cannot start,
server boot may still fall back to a command-only durable backend. In either
case, there is a single durable source of truth for job state.

Supported commands (via TaskCmd):
    JOB_SUBMIT  — submit a new job
    JOB_CANCEL  — cancel a running job
    JOB_LIST    — list tracked jobs

To query running job state, use the existing STATUS command (umrs-client status)
which reads from the in-memory UMRSThreads store.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2 import sql as pg_sql

from umrs.jobs.models import JobStatus, JobType, RestartPolicy
from umrs.jobs.store import DuplicateJobError, InvalidTransitionError, JobNotFoundError, JobStore

logger = logging.getLogger(__name__)

# Valid job statuses (matching jobs/models.py JobStatus enum)
VALID_STATUSES = {"planned", "queued", "running", "completed", "failed", "cancelled"}
# Only job types that genuinely need explicit submission (not triggered by file uploads).
# - endpoint_provision: pre-warm Slurm/K8s service before uploads arrive
# - index_build: rebuild RAG index (pgvector IVFFlat re-cluster) after bulk inserts
# - cleanup: teardown endpoints explicitly (endpoint target only for now)
#
# Excluded because they have dedicated paths:
# - model_download → MODEL_DOWNLOAD ZMQ command
# - embedding_batch → PROCESS_OBJECTS (upload) + recompute/update maintenance
# - pipeline_run → deferred until steps framework is fully wired
VALID_JOB_TYPES = {"endpoint_provision", "index_build", "cleanup", "workflow_run"}
JOB_SUBMIT = "JOB_SUBMIT"
JOB_CANCEL = "JOB_CANCEL"
JOB_LIST = "JOB_LIST"


def _parse_payload(payload_str: str) -> Dict[str, Any]:
    if not payload_str:
        return {}
    return json.loads(payload_str)


def _parse_job_type(job_type: str) -> JobType:
    if not job_type:
        raise ValueError("job_type is required")
    if job_type not in VALID_JOB_TYPES:
        raise ValueError(f"Invalid job_type: {job_type}. Valid: {sorted(VALID_JOB_TYPES)}")
    return JobType(job_type)


def _parse_job_status(status: Optional[str]) -> Optional[JobStatus]:
    if not status:
        return None
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Valid: {sorted(VALID_STATUSES)}")
    return JobStatus(status)


def _parse_restart_policy(restart_policy: str) -> RestartPolicy:
    try:
        return RestartPolicy(restart_policy or RestartPolicy.ON_CRASH.value)
    except ValueError as exc:
        valid = sorted(policy.value for policy in RestartPolicy)
        raise ValueError(
            f"Invalid restart_policy: {restart_policy}. Valid: {valid}"
        ) from exc


def _require_string(spec: Dict[str, Any], key: str, label: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _validate_index_build_spec(spec: Dict[str, Any]) -> None:
    _require_string(spec, "database", "index_build spec.database")
    _require_string(spec, "table", "index_build spec.table")
    if "rebuild" in spec and not isinstance(spec["rebuild"], bool):
        raise ValueError("index_build spec.rebuild must be a boolean")


def _validate_workflow_run_spec(spec: Dict[str, Any]) -> None:
    _require_string(spec, "workflow_name", "workflow_run spec.workflow_name")
    _require_string(spec, "execution_id", "workflow_run spec.execution_id")
    if "params" in spec and not isinstance(spec["params"], dict):
        raise ValueError("workflow_run spec.params must be an object")


def _validate_cleanup_spec(spec: Dict[str, Any]) -> None:
    target = _require_string(spec, "target", "cleanup spec.target").lower()
    if target != "endpoint":
        raise ValueError("cleanup currently supports only spec.target='endpoint'")
    _require_string(spec, "job_id", "cleanup spec.job_id")


_JOB_SPEC_VALIDATORS = {
    JobType.INDEX_BUILD: _validate_index_build_spec,
    JobType.WORKFLOW_RUN: _validate_workflow_run_spec,
    JobType.CLEANUP: _validate_cleanup_spec,
}


def _validate_job_spec(job_type: JobType, spec: Any) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("spec must be a JSON object")

    validator = _JOB_SPEC_VALIDATORS.get(job_type)
    if validator is not None:
        validator(spec)
    return spec


def _serialize_attempt(attempt) -> Dict[str, Any]:
    data = attempt.to_dict()
    data.setdefault("exit_reason", None)
    return data


def _serialize_job(job, attempts: Optional[List[Any]] = None) -> Dict[str, Any]:
    data = job.to_dict()
    data["attempts"] = [_serialize_attempt(attempt) for attempt in (attempts or [])]
    return data


class DurableJobCommandBackend:
    """PostgreSQL-backed adapter for the ZMQ job command surface."""

    def __init__(self, job_store: JobStore, db_pool=None):
        self._job_store = job_store
        self._db_pool = db_pool

    def close(self) -> None:
        if self._db_pool is not None:
            self._db_pool.closeall()

    def _submit_endpoint_cleanup(self, client_id: str, job) -> str:
        cleanup_spec = {
            "target": "endpoint",
            "job_id": job.job_id,
        }

        try:
            cleanup_job_id = self._job_store.create_job(
                project_id=job.project_id,
                user_id=job.user_id,
                job_type=JobType.CLEANUP,
                spec=cleanup_spec,
                max_attempts=1,
                restart_policy=RestartPolicy.NEVER,
            )
            self._job_store.enqueue_job(cleanup_job_id)
            cleanup_job = self._job_store.get_job(cleanup_job_id)
            cleanup_attempts = self._job_store.get_attempts_for_job(cleanup_job_id)
            message = (
                f"Endpoint cleanup submitted for completed provision job {job.job_id}: "
                f"{cleanup_job_id}"
            )
        except DuplicateJobError as exc:
            cleanup_job = self._job_store.get_job(exc.existing_job_id)
            cleanup_attempts = self._job_store.get_attempts_for_job(exc.existing_job_id)
            message = (
                f"Endpoint cleanup already exists for completed provision job {job.job_id}: "
                f"{exc.existing_job_id}"
            )

        logger.info("Durable endpoint cleanup requested: %s by %s", job.job_id, client_id)
        return json.dumps(
            {
                "status": "ok",
                "message": message,
                "target_job_id": job.job_id,
                "job": _serialize_job(cleanup_job, cleanup_attempts),
            },
            default=str,
        )

    def submit(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_payload(payload_str)
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "error", "message": f"Invalid JSON: {exc}"})

        try:
            job_type = _parse_job_type((payload.get("job_type") or "").strip())
            project_id = payload.get("project_id", "default")
            spec = _validate_job_spec(job_type, payload.get("spec", {}))
            priority = int(payload.get("priority", 0))
            max_attempts = int(payload.get("max_attempts", 3))
            restart_policy = _parse_restart_policy(payload.get("restart_policy", "on_crash"))
            idempotency_key = payload.get("idempotency_key")

            job_id = self._job_store.create_job(
                project_id=project_id,
                user_id=client_id,
                job_type=job_type,
                spec=spec,
                idempotency_key=idempotency_key,
                priority=priority,
                max_attempts=max_attempts,
                restart_policy=restart_policy,
            )
            self._job_store.enqueue_job(job_id)
            job = self._job_store.get_job(job_id)
            attempts = self._job_store.get_attempts_for_job(job_id)
        except DuplicateJobError as exc:
            job = self._job_store.get_job(exc.existing_job_id)
            attempts = self._job_store.get_attempts_for_job(exc.existing_job_id)
            return json.dumps(
                {
                    "status": "ok",
                    "message": f"Job already exists: {exc.existing_job_id}",
                    "job": _serialize_job(job, attempts),
                },
                default=str,
            )
        except (InvalidTransitionError, JobNotFoundError, ValueError) as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        logger.info("Durable job submitted: %s (type=%s, user=%s)", job.job_id, job.job_type.value, client_id)
        return json.dumps(
            {
                "status": "ok",
                "message": f"Job submitted: {job.job_id}",
                "job": _serialize_job(job, attempts),
            },
            default=str,
        )

    def cancel(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_payload(payload_str)
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "error", "message": f"Invalid JSON: {exc}"})

        job_id = (payload.get("job_id") or "").strip()
        if not job_id:
            return json.dumps({"status": "error", "message": "job_id is required"})

        try:
            job = self._job_store.get_job(job_id)
        except JobNotFoundError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        try:
            self._job_store.cancel_job(job_id, reason=f"cancelled by {client_id}")
            if job.job_type == JobType.WORKFLOW_RUN:
                from umrs.jobs.workflow_run import cancel_workflow_execution_for_job

                cancel_workflow_execution_for_job(job)
        except InvalidTransitionError as exc:
            if job.job_type == JobType.ENDPOINT_PROVISION and job.status == JobStatus.COMPLETED:
                return self._submit_endpoint_cleanup(client_id, job)
            return json.dumps({"status": "error", "message": str(exc)})

        logger.info("Durable job cancelled: %s by %s", job_id, client_id)
        return json.dumps({"status": "ok", "message": f"Job {job_id} cancelled"})

    def list_jobs(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_payload(payload_str)
        except json.JSONDecodeError:
            payload = {}

        try:
            job_id = payload.get("job_id")
            status = _parse_job_status(payload.get("status"))
            job_type = None
            if payload.get("job_type"):
                job_type = _parse_job_type(payload.get("job_type"))
            project_id = payload.get("project_id")
            limit = min(int(payload.get("limit", 200)), 1000)
            offset = max(int(payload.get("offset", 0)), 0)
        except ValueError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        if job_id:
            try:
                job = self._job_store.get_job(job_id)
            except JobNotFoundError as exc:
                return json.dumps({"status": "error", "message": str(exc)})

            attempts = self._job_store.get_attempts_for_job(job.job_id)
            return json.dumps(
                {
                    "status": "ok",
                    "jobs": [_serialize_job(job, attempts)],
                    "total": 1,
                },
                default=str,
            )

        jobs = self._job_store.list_jobs(
            project_id=project_id,
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )
        serialized_jobs = []
        for job in jobs:
            attempts = self._job_store.get_attempts_for_job(job.job_id)
            serialized_jobs.append(_serialize_job(job, attempts))

        return json.dumps(
            {
                "status": "ok",
                "jobs": serialized_jobs,
                "total": len(serialized_jobs),
            },
            default=str,
        )


def build_durable_job_db_pool_from_env():
    """Bootstrap a thread-safe durable jobs connection pool from process env."""
    db_host = os.environ.get("DB_HOST", "127.0.0.1")
    db_port = os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))
    db_name = os.environ.get("UMRS_JOBS_DB", os.environ.get("POSTGRES_DB", "umrs_db"))
    db_user = os.environ.get("DB_SUPERUSER", os.environ.get("POSTGRES_USER", "postgres"))
    db_password = os.environ.get(
        "DB_SUPERUSER_PASSWORD",
        os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )

    # sslmode=disable: local container doesn't support SSL and the default
    # 'prefer' mode can fail during early startup with "error response during
    # SSL exchange".
    admin_conn = psycopg2.connect(
        dbname="postgres",
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        sslmode="disable",
    )
    admin_conn.autocommit = True
    try:
        with admin_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cur.fetchone()
            if not exists:
                cur.execute(
                    pg_sql.SQL("CREATE DATABASE {}")
                    .format(pg_sql.Identifier(db_name))
                )
    finally:
        admin_conn.close()

    db_pool = pg_pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port,
        sslmode="disable",
    )
    logger.info(
        "Durable jobs database pool ready (db=%s host=%s port=%s)",
        db_name,
        db_host,
        db_port,
    )
    return db_pool


def build_durable_job_backend_from_env() -> Optional[DurableJobCommandBackend]:
    """Best-effort bootstrap of the durable job backend from process env."""
    try:
        db_pool = build_durable_job_db_pool_from_env()
        job_store = JobStore(db_pool)
        logger.info("Durable ZMQ job backend enabled")
        return DurableJobCommandBackend(job_store, db_pool=db_pool)
    except Exception:
        logger.exception("Durable ZMQ job backend unavailable")
        return None


def _backend_unavailable_response() -> str:
    return json.dumps(
        {
            "status": "error",
            "message": "Durable job backend unavailable",
        }
    )


def handle_job_command(cmd: str, client_id: str, payload_str: str, backend: Optional[DurableJobCommandBackend] = None) -> str:
    """
    Dispatch a job tracking command and return a JSON response string.

    Called from the server main loop when a JOB_* command is received.

    Args:
        cmd: TaskCmd value (JOB_SUBMIT, JOB_CANCEL, JOB_LIST)
        client_id: Username / client identifier
        payload_str: JSON-encoded payload
        backend: Durable backend shared with the server runtime

    Returns:
        JSON string to send back to client via ZMQ
    """
    try:
        if backend is None:
            logger.error("Received %s but durable job backend is unavailable", cmd)
            return _backend_unavailable_response()

        if cmd == JOB_SUBMIT:
            return backend.submit(client_id, payload_str)
        elif cmd == JOB_CANCEL:
            return backend.cancel(client_id, payload_str)
        elif cmd == JOB_LIST:
            return backend.list_jobs(client_id, payload_str)
        else:
            return json.dumps({"status": "error", "message": f"Unknown job command: {cmd}"})
    except Exception as e:
        logger.exception("Error handling job command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
