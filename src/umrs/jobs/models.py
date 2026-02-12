"""
Job and Attempt models with enums.

These dataclasses mirror the database schema and provide type safety.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class JobType(Enum):
    """Types of jobs supported by UMRS."""

    INDEX_BUILD = "index_build"           # Build RAG index from dataset
    EMBEDDING_BATCH = "embedding_batch"   # Batch embedding computation
    PIPELINE_RUN = "pipeline_run"         # Execute ingestion pipeline
    ENDPOINT_PROVISION = "endpoint_provision"  # Provision Mini-RiVault
    MODEL_DOWNLOAD = "model_download"     # Download model weights to approved path
    CLEANUP = "cleanup"                   # Delete resources
    CONTEXT_RETENTION = "context_retention"     # Cleanup expired conversations
    CONVERSATION_SUMMARIZE = "conversation_summarize"  # Summarize long conversations


class JobStatus(Enum):
    """Job-level status (user intent)."""

    PLANNED = "planned"       # Created but not yet queued
    QUEUED = "queued"         # Ready to schedule
    RUNNING = "running"       # Has active attempt
    COMPLETED = "completed"   # Successfully finished
    FAILED = "failed"         # All retries exhausted
    CANCELLED = "cancelled"   # User cancelled


class AttemptStatus(Enum):
    """Attempt-level status (execution try)."""

    CREATED = "created"       # Attempt record created
    SUBMITTED = "submitted"   # Submitted to backend (Slurm/K8s/Compose)
    RUNNING = "running"       # Backend reports running
    SUCCEEDED = "succeeded"   # Completed successfully
    FAILED = "failed"         # Attempt failed (may retry)
    LOST = "lost"             # UMRS crashed, attempt orphaned


class RestartPolicy(Enum):
    """When to retry failed/lost attempts."""

    ON_CRASH = "on_crash"     # Retry on UMRS crash (default)
    ON_FAILURE = "on_failure" # Retry on any failure
    NEVER = "never"           # No retry (destructive ops)


# Default backoff settings per job type
DEFAULT_BACKOFFS: Dict[JobType, Dict[str, Any]] = {
    JobType.ENDPOINT_PROVISION: {
        "delays": [5, 10, 20],      # Fast retry
        "max_attempts": 3,
    },
    JobType.INDEX_BUILD: {
        "delays": [30, 60, 120],    # Slow retry
        "max_attempts": 5,
    },
    JobType.EMBEDDING_BATCH: {
        "delays": [30, 60, 120],
        "max_attempts": 5,
    },
    JobType.PIPELINE_RUN: {
        "delays": [15, 30, 60],     # Medium
        "max_attempts": 4,
    },
    JobType.MODEL_DOWNLOAD: {
        "delays": [60, 120],        # Slow retry for downloads
        "max_attempts": 2,          # Limited retries (network issues)
    },
    JobType.CLEANUP: {
        "delays": [],
        "max_attempts": 1,         # No retry for cleanup
    },
    JobType.CONTEXT_RETENTION: {
        "delays": [60, 120],       # Retry after delay
        "max_attempts": 3,
    },
    JobType.CONVERSATION_SUMMARIZE: {
        "delays": [30, 60, 120],   # Medium retry for LLM calls
        "max_attempts": 3,
    },
}


def generate_job_id() -> str:
    """Generate a unique job ID."""
    return f"job-{uuid.uuid4().hex[:12]}"


def generate_attempt_id() -> str:
    """Generate a unique attempt ID."""
    return f"att-{uuid.uuid4().hex[:12]}"


def compute_idempotency_key(
    project_id: str,
    user_id: str,
    job_type: JobType,
    spec: Dict[str, Any],
) -> str:
    """
    Compute stable idempotency key from job parameters.

    This prevents duplicate jobs from client retries.
    """
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    combined = f"{project_id}:{user_id}:{job_type.value}:{spec_json}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


@dataclass
class Job:
    """
    Job record (user intent).

    Immutable after creation except for:
    - status transitions
    - active_attempt_id / commit_attempt_id updates
    - timestamp updates
    """

    job_id: str
    project_id: str
    user_id: str
    job_type: JobType
    spec: Dict[str, Any]  # JSONB - job-specific parameters
    idempotency_key: str

    status: JobStatus = JobStatus.PLANNED
    priority: int = 0  # Higher = more urgent

    desired_max_attempts: int = 3
    restart_policy: RestartPolicy = RestartPolicy.ON_CRASH

    next_run_at: Optional[datetime] = None
    active_attempt_id: Optional[str] = None
    commit_attempt_id: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_state_change_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None

    def can_retry(self, current_attempt_no: int) -> bool:
        """Check if job can be retried given current attempt number."""
        if self.restart_policy == RestartPolicy.NEVER:
            return False
        return current_attempt_no < self.desired_max_attempts

    def get_backoff_delay(self, attempt_no: int) -> int:
        """Get backoff delay in seconds for retry."""
        backoff = DEFAULT_BACKOFFS.get(self.job_type, {"delays": [30, 60, 120]})
        delays = backoff["delays"]
        if not delays:
            return 0
        # Use last delay for attempts beyond array length
        idx = min(attempt_no - 1, len(delays) - 1)
        return delays[idx] if idx >= 0 else delays[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "job_type": self.job_type.value,
            "spec": self.spec,
            "idempotency_key": self.idempotency_key,
            "status": self.status.value,
            "priority": self.priority,
            "desired_max_attempts": self.desired_max_attempts,
            "restart_policy": self.restart_policy.value,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "active_attempt_id": self.active_attempt_id,
            "commit_attempt_id": self.commit_attempt_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_state_change_at": self.last_state_change_at.isoformat() if self.last_state_change_at else None,
            "last_heartbeat_at": self.last_heartbeat_at.isoformat() if self.last_heartbeat_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            project_id=data["project_id"],
            user_id=data["user_id"],
            job_type=JobType(data["job_type"]),
            spec=data["spec"],
            idempotency_key=data["idempotency_key"],
            status=JobStatus(data["status"]),
            priority=data.get("priority", 0),
            desired_max_attempts=data.get("desired_max_attempts", 3),
            restart_policy=RestartPolicy(data.get("restart_policy", "on_crash")),
            next_run_at=_parse_datetime(data.get("next_run_at")),
            active_attempt_id=data.get("active_attempt_id"),
            commit_attempt_id=data.get("commit_attempt_id"),
            created_at=_parse_datetime(data.get("created_at")) or datetime.utcnow(),
            updated_at=_parse_datetime(data.get("updated_at")) or datetime.utcnow(),
            last_state_change_at=_parse_datetime(data.get("last_state_change_at")),
            last_heartbeat_at=_parse_datetime(data.get("last_heartbeat_at")),
        )


@dataclass
class JobAttempt:
    """
    Attempt record (one execution try).

    Multiple attempts per job on retries.
    """

    attempt_id: str
    job_id: str
    attempt_no: int  # 1-based

    backend: str  # "slurm", "kubernetes", "compose"
    backend_ref: Optional[str] = None  # Slurm job ID, K8s Job name, etc.

    status: AttemptStatus = AttemptStatus.CREATED

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    exit_reason: Optional[str] = None  # Error message or success detail
    log_uri: Optional[str] = None  # s3://…/jobs/<job_id>/attempt_<n>/logs

    # Operational fields
    metrics: Dict[str, Any] = field(default_factory=dict)  # JSONB
    umrs_instance_id: Optional[str] = None  # Which UMRS started this
    scheduler_claimed_at: Optional[datetime] = None
    status_detail: Optional[str] = None  # e.g., "polling readiness"

    # Ignored attempt tracking
    ignored_at: Optional[datetime] = None
    ignore_reason: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def output_prefix(self) -> str:
        """S3 path prefix for this attempt's outputs."""
        return f"jobs/{self.job_id}/attempt_{self.attempt_no}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "attempt_id": self.attempt_id,
            "job_id": self.job_id,
            "attempt_no": self.attempt_no,
            "backend": self.backend,
            "backend_ref": self.backend_ref,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "exit_reason": self.exit_reason,
            "log_uri": self.log_uri,
            "metrics": self.metrics,
            "umrs_instance_id": self.umrs_instance_id,
            "scheduler_claimed_at": self.scheduler_claimed_at.isoformat() if self.scheduler_claimed_at else None,
            "status_detail": self.status_detail,
            "ignored_at": self.ignored_at.isoformat() if self.ignored_at else None,
            "ignore_reason": self.ignore_reason,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobAttempt":
        """Create from dictionary."""
        return cls(
            attempt_id=data["attempt_id"],
            job_id=data["job_id"],
            attempt_no=data["attempt_no"],
            backend=data["backend"],
            backend_ref=data.get("backend_ref"),
            status=AttemptStatus(data["status"]),
            started_at=_parse_datetime(data.get("started_at")),
            ended_at=_parse_datetime(data.get("ended_at")),
            exit_reason=data.get("exit_reason"),
            log_uri=data.get("log_uri"),
            metrics=data.get("metrics", {}),
            umrs_instance_id=data.get("umrs_instance_id"),
            scheduler_claimed_at=_parse_datetime(data.get("scheduler_claimed_at")),
            status_detail=data.get("status_detail"),
            ignored_at=_parse_datetime(data.get("ignored_at")),
            ignore_reason=data.get("ignore_reason"),
            created_at=_parse_datetime(data.get("created_at")) or datetime.utcnow(),
        )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    # Handle both with and without 'Z' suffix
    if value.endswith("Z"):
        value = value[:-1]
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
