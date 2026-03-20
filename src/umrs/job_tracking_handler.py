"""
Job Tracking Handler — bridges ZMQ protocol to job tracking operations.

Provides handle_job_command() for the server to call from its ZMQ loop.
Uses a JSON file store for lightweight job tracking (no PostgreSQL required).
When the full jobs/store.py is available (PostgreSQL), it can be used instead.

Supported commands (via TaskCmd):
  JOB_SUBMIT  — submit a new job
  JOB_STATUS  — get status of a job (with attempt history)
  JOB_LIST    — list jobs with optional filters
  JOB_CANCEL  — cancel a running job
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import get_default_umrs_cfg_dir

logger = logging.getLogger(__name__)

_JOBS_LOCK = threading.Lock()

# Valid job statuses (matching jobs/models.py JobStatus enum)
VALID_STATUSES = {"planned", "queued", "running", "completed", "failed", "cancelled"}
VALID_JOB_TYPES = {"index_build", "embedding_batch", "pipeline_run", "endpoint_provision", "model_download", "cleanup"}


def _jobs_path() -> str:
    """Return path to the JSON jobs file."""
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


def _handle_status(client_id: str, payload_str: str) -> str:
    """Get status of a specific job."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})

    job_id = payload.get("job_id", "").strip()
    if not job_id:
        return json.dumps({"status": "error", "message": "job_id is required"})

    with _JOBS_LOCK:
        data = _load_jobs()

    for job in data["jobs"]:
        if job["job_id"] == job_id:
            return json.dumps({
                "status": "ok",
                "job": job,
            }, default=str)

    return json.dumps({"status": "error", "message": f"Job not found: {job_id}"})


def _handle_list(client_id: str, payload_str: str) -> str:
    """List jobs with optional filters."""
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}

    with _JOBS_LOCK:
        data = _load_jobs()

    jobs = data["jobs"]

    # Apply filters
    status_filter = payload.get("status")
    if status_filter:
        jobs = [j for j in jobs if j["status"] == status_filter]

    job_type_filter = payload.get("job_type")
    if job_type_filter:
        jobs = [j for j in jobs if j["job_type"] == job_type_filter]

    user_filter = payload.get("user_id")
    if user_filter:
        jobs = [j for j in jobs if j["user_id"] == user_filter]

    # Limit
    limit = payload.get("limit", 50)
    jobs = jobs[-limit:]

    return json.dumps({
        "status": "ok",
        "count": len(jobs),
        "jobs": jobs,
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


def handle_job_command(cmd: str, client_id: str, payload_str: str) -> str:
    """
    Dispatch a job tracking command and return a JSON response string.

    Called from the server main loop when a JOB_* command is received.

    Args:
        cmd: TaskCmd value (JOB_SUBMIT, JOB_STATUS, JOB_LIST, JOB_CANCEL)
        client_id: Username / client identifier
        payload_str: JSON-encoded payload

    Returns:
        JSON string to send back to client via ZMQ
    """
    from .TaskCmdStatus import TaskCmd

    try:
        if cmd == TaskCmd.JOB_SUBMIT:
            return _handle_submit(client_id, payload_str)
        elif cmd == TaskCmd.JOB_STATUS:
            return _handle_status(client_id, payload_str)
        elif cmd == TaskCmd.JOB_LIST:
            return _handle_list(client_id, payload_str)
        elif cmd == TaskCmd.JOB_CANCEL:
            return _handle_cancel(client_id, payload_str)
        else:
            return json.dumps({"status": "error", "message": f"Unknown job command: {cmd}"})
    except Exception as e:
        logger.exception("Error handling job command %s", cmd)
        return json.dumps({"status": "error", "message": str(e)})
