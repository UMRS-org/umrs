"""
Compatibility ZMQ bridge for job tracking commands.

This handler is the legacy CLI-facing job path used by server_umrs.py.
It persists state in ~/.config/umrs/jobs.json so ZMQ job commands can work
without requiring the durable PostgreSQL-backed job service to be bootstrapped.

The durable implementation lives under umrs.jobs.* and is the intended long-term
authority for retries, reconciliation, and attempt state. Until server startup
wires that service into the ZMQ loop, this module remains an explicit
compatibility backend rather than a second full jobs architecture.

Supported commands (via TaskCmd):
    JOB_SUBMIT  — submit a new job
    JOB_CANCEL  — cancel a running job
    JOB_LIST    — list compatibility-tracked jobs

To query running job state, use the existing STATUS command (umrs-client status)
which reads from the in-memory UMRSThreads store.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2 import sql as pg_sql

from umrs.jobs.models import JobStatus, JobType, RestartPolicy
from umrs.jobs.store import DuplicateJobError, InvalidTransitionError, JobNotFoundError, JobStore
from umrs.shared.config import get_default_umrs_cfg_dir

logger = logging.getLogger(__name__)

_JOBS_LOCK = threading.Lock()

# Valid job statuses (matching jobs/models.py JobStatus enum)
VALID_STATUSES = {"planned", "queued", "running", "completed", "failed", "cancelled"}
# Only job types that genuinely need explicit submission (not triggered by file uploads).
# - endpoint_provision: pre-warm Slurm/K8s service before uploads arrive
# - index_build: rebuild RAG index (pgvector IVFFlat re-cluster) after bulk inserts
# - cleanup: teardown endpoints, stale S3 objects, or database tables
#
# Excluded because they have dedicated paths:
# - model_download → MODEL_DOWNLOAD ZMQ command
# - embedding_batch → PROCESS_OBJECTS (upload) + recompute/update maintenance
# - pipeline_run → deferred until steps framework is fully wired
VALID_JOB_TYPES = {"endpoint_provision", "index_build", "cleanup"}


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

    def submit(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_payload(payload_str)
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "error", "message": f"Invalid JSON: {exc}"})

        try:
            job_type = _parse_job_type((payload.get("job_type") or "").strip())
            project_id = payload.get("project_id", "default")
            spec = payload.get("spec", {})
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
            self._job_store.cancel_job(job_id, reason=f"cancelled by {client_id}")
        except (InvalidTransitionError, JobNotFoundError) as exc:
            return json.dumps({"status": "error", "message": str(exc)})

        logger.info("Durable job cancelled: %s by %s", job_id, client_id)
        return json.dumps({"status": "ok", "message": f"Job {job_id} cancelled"})

    def list_jobs(self, client_id: str, payload_str: str) -> str:
        try:
            payload = _parse_payload(payload_str)
        except json.JSONDecodeError:
            payload = {}

        try:
            status = _parse_job_status(payload.get("status"))
            job_type = None
            if payload.get("job_type"):
                job_type = _parse_job_type(payload.get("job_type"))
            project_id = payload.get("project_id")
            limit = min(int(payload.get("limit", 200)), 1000)
            offset = max(int(payload.get("offset", 0)), 0)
        except ValueError as exc:
            return json.dumps({"status": "error", "message": str(exc)})

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


def build_durable_job_backend_from_env() -> Optional[DurableJobCommandBackend]:
    """Best-effort bootstrap of the durable job backend from process env."""
    db_host = os.environ.get("DB_HOST", "127.0.0.1")
    db_port = os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))
    db_name = os.environ.get("UMRS_JOBS_DB", os.environ.get("POSTGRES_DB", "umrs_db"))
    db_user = os.environ.get("DB_SUPERUSER", os.environ.get("POSTGRES_USER", "postgres"))
    db_password = os.environ.get(
        "DB_SUPERUSER_PASSWORD",
        os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )

    try:
        admin_conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
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

        db_pool = pg_pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        job_store = JobStore(db_pool)
        logger.info(
            "Durable ZMQ job backend enabled (db=%s host=%s port=%s)",
            db_name,
            db_host,
            db_port,
        )
        return DurableJobCommandBackend(job_store, db_pool=db_pool)
    except Exception:
        logger.exception("Falling back to compatibility JSON job backend")
        return None


def _jobs_path() -> str:
    """Return path to the compatibility JSON jobs file."""
    cfg_dir = get_default_umrs_cfg_dir()
    return os.path.join(cfg_dir, "jobs.json")


def _load_jobs() -> Dict[str, Any]:
    """Load jobs from JSON file."""
    path = _jobs_path()
    if not os.path.exists(path):
        return {"jobs": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_jobs(data: Dict[str, Any]) -> None:
    """Save jobs to JSON file."""
    path = _jobs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def _generate_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


def _generate_attempt_id() -> str:
    return f"att-{uuid.uuid4().hex[:12]}"


def _handle_submit(client_id: str, payload_str: str) -> str:
    """Submit a new job."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    job_type = payload.get("job_type", "").strip()
    spec = payload.get("spec", {})
    project_id = payload.get("project_id", "default")

    if not job_type:
        return json.dumps({"status": "error", "message": "job_type is required"})
    if job_type not in VALID_JOB_TYPES:
        return json.dumps({"status": "error", "message": f"Invalid job_type: {job_type}. Valid: {sorted(VALID_JOB_TYPES)}"})

    job_id = _generate_job_id()
    now = datetime.utcnow().isoformat()

    job = {
        "job_id": job_id,
        "project_id": project_id,
        "user_id": client_id,
        "job_type": job_type,
        "spec": spec,
        "status": "queued",
        "priority": payload.get("priority", 0),
        "desired_max_attempts": payload.get("max_attempts", 3),
        "restart_policy": payload.get("restart_policy", "on_crash"),
        "keep_alive": payload.get("keep_alive", False),
        "attempts": [],
        "created_at": now,
        "updated_at": now,
    }

    # Create initial attempt
    attempt = {
        "attempt_id": _generate_attempt_id(),
        "attempt_no": 1,
        "status": "created",
        "started_at": now,
        "ended_at": None,
        "exit_reason": None,
    }
    job["attempts"].append(attempt)
    job["active_attempt_id"] = attempt["attempt_id"]

    with _JOBS_LOCK:
        data = _load_jobs()
        data["jobs"].append(job)
        _save_jobs(data)

    logger.info("Job submitted: %s (type=%s, user=%s)", job_id, job_type, client_id)
    return json.dumps({
        "status": "ok",
        "message": f"Job submitted: {job_id}",
        "job": job,
    }, default=str)


def _handle_cancel(client_id: str, payload_str: str) -> str:
    """Cancel a job."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    job_id = payload.get("job_id", "").strip()
    if not job_id:
        return json.dumps({"status": "error", "message": "job_id is required"})

    with _JOBS_LOCK:
        data = _load_jobs()
        found = False
        for job in data["jobs"]:
            if job["job_id"] == job_id:
                found = True
                if job["status"] in ("completed", "cancelled", "failed"):
                    return json.dumps({
                        "status": "error",
                        "message": f"Job {job_id} is already {job['status']}",
                    })
                job["status"] = "cancelled"
                job["updated_at"] = datetime.utcnow().isoformat()
                # Cancel active attempt
                for attempt in job.get("attempts", []):
                    if attempt["status"] in ("created", "submitted", "running"):
                        attempt["status"] = "failed"
                        attempt["ended_at"] = datetime.utcnow().isoformat()
                        attempt["exit_reason"] = "cancelled by user"
                break

        if not found:
            return json.dumps({"status": "error", "message": f"Job not found: {job_id}"})

        _save_jobs(data)

    logger.info("Job cancelled: %s by %s", job_id, client_id)
    return json.dumps({
        "status": "ok",
        "message": f"Job {job_id} cancelled",
    })


def _handle_list(client_id: str, payload_str: str) -> str:
    """List jobs with optional filters."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}

    filters = {
        "status": payload.get("status"),
        "job_type": payload.get("job_type"),
        "project_id": payload.get("project_id"),
    }
    limit = min(int(payload.get("limit", 200)), 1000)

    with _JOBS_LOCK:
        data = _load_jobs()

    jobs = data.get("jobs", [])
    for key, val in filters.items():
        if val:
            jobs = [j for j in jobs if j.get(key) == val]

    # Most recent first, limited
    jobs = sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)[:limit]

    return json.dumps({
        "status": "ok",
        "jobs": jobs,
        "total": len(jobs),
    }, default=str)


def handle_job_command(cmd: str, client_id: str, payload_str: str, backend: Optional[DurableJobCommandBackend] = None) -> str:
    """
    Dispatch a job tracking command and return a JSON response string.

    Called from the server main loop when a JOB_* command is received.

    Args:
        cmd: TaskCmd value (JOB_SUBMIT, JOB_CANCEL)
        client_id: Username / client identifier
        payload_str: JSON-encoded payload

    Returns:
        JSON string to send back to client via ZMQ
    """
    from umrs.core.TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.JOB_SUBMIT:
            if backend is not None:
                return backend.submit(client_id, payload_str)
            return _handle_submit(client_id, payload_str)
        elif cmd == TaskCmd.JOB_CANCEL:
            if backend is not None:
                return backend.cancel(client_id, payload_str)
            return _handle_cancel(client_id, payload_str)
        elif cmd == TaskCmd.JOB_LIST:
            if backend is not None:
                return backend.list_jobs(client_id, payload_str)
            return _handle_list(client_id, payload_str)
        else:
            return json.dumps({"status": "error", "message": f"Unknown job command: {cmd}"})
    except Exception as e:
        logger.exception("Error handling job command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
