"""
Durable Job Tracking System (U-701 ~ U-744).

This module implements crash-safe job tracking for UMRS:
- Jobs table: User intent (immutable spec)
- Attempts table: Execution tries (multiple per job)
- Commit pointer: Marks which attempt's outputs are current

Design Invariants:
A. Attempt-scoped writes: All outputs to s3://…/jobs/<job_id>/attempt_<n>/…
B. Commit pointer is truth: Only commit_attempt_id outputs visible
C. No concurrent attempts: Advisory lock per job
"""

from .models import (
    JobType,
    JobStatus,
    AttemptStatus,
    RestartPolicy,
    Job,
    JobAttempt,
    compute_idempotency_key,
    generate_job_id,
    generate_attempt_id,
)
from .schema import ensure_schema, get_schema_version
from .store import (
    JobStore,
    JobStoreError,
    DuplicateJobError,
    JobNotFoundError,
    AttemptNotFoundError,
    InvalidTransitionError,
)
from .reconciler import Reconciler, ReconcileResult, reconcile_on_startup
from .scheduler import Scheduler, SimpleBackendAdapter
from .monitor import (
    AttemptMonitor,
    AttemptCheckResult,
    CompositeStatusChecker,
    EndpointProvisionChecker,
)
from .service import JobService, create_job_service
from .adapters import (
    CleanupBackend,
    CleanupStatusChecker,
    EndpointProvisionBackend,
    EndpointProvisionStatusChecker,
    IndexBuildBackend,
    IndexBuildStatusChecker,
    create_backend_factory,
    create_status_checker,
)
# Note: context_jobs excluded from selective merge
from .ha import (
    HeartbeatManager,
    DistributedLockManager,
    ClusterCoordinator,
    InstanceInfo,
    HeartbeatError,
    LockError,
    LockAcquisitionError,
    LockNotHeldError,
)

__all__ = [
    # Enums
    "JobType",
    "JobStatus",
    "AttemptStatus",
    "RestartPolicy",
    # Models
    "Job",
    "JobAttempt",
    # Utilities
    "compute_idempotency_key",
    "generate_job_id",
    "generate_attempt_id",
    # Schema
    "ensure_schema",
    "get_schema_version",
    # Store
    "JobStore",
    "JobStoreError",
    "DuplicateJobError",
    "JobNotFoundError",
    "AttemptNotFoundError",
    "InvalidTransitionError",
    # Reconciler
    "Reconciler",
    "ReconcileResult",
    "reconcile_on_startup",
    # Scheduler
    "Scheduler",
    "SimpleBackendAdapter",
    # Monitor
    "AttemptMonitor",
    "AttemptCheckResult",
    "CompositeStatusChecker",
    "EndpointProvisionChecker",
    # Service
    "JobService",
    "create_job_service",
    # Adapters
    "CleanupBackend",
    "CleanupStatusChecker",
    "EndpointProvisionBackend",
    "EndpointProvisionStatusChecker",
    "IndexBuildBackend",
    "IndexBuildStatusChecker",
    "create_backend_factory",
    "create_status_checker",
    # HA Coordination (U-703, U-715)
    "HeartbeatManager",
    "DistributedLockManager",
    "ClusterCoordinator",
    "InstanceInfo",
    "HeartbeatError",
    "LockError",
    "LockAcquisitionError",
    "LockNotHeldError",
]
