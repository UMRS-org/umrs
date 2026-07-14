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
from .data_ingestion import DataIngestionBackend
from .data_ingestion import DataIngestionStatusChecker
from .knowledge_adapters import (
    KnowledgeEvent,
    KnowledgeIngestBackend,
    KnowledgeIngestResult,
    KnowledgeIngestStatusChecker,
    KnowledgeUnderstandBackend,
    KnowledgeUnderstandResult,
    KnowledgeUnderstandStatusChecker,
)

# Note: context_jobs excluded from selective merge
from .ha import (
    ClusterCoordinator,
    DistributedLockManager,
    HeartbeatError,
    HeartbeatManager,
    InstanceInfo,
    LockAcquisitionError,
    LockError,
    LockNotHeldError,
)
from .models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
    compute_idempotency_key,
    generate_attempt_id,
    generate_job_id,
)
from .monitor import (
    AttemptCheckResult,
    AttemptMonitor,
    CompositeStatusChecker,
    EndpointProvisionChecker,
)
from .reconciler import Reconciler, ReconcileResult, reconcile_on_startup
from .scheduler import Scheduler, SimpleBackendAdapter
from .schema import ensure_schema, get_schema_version
from .service import JobService, create_job_service
from .store import (
    AttemptNotFoundError,
    DuplicateJobError,
    InvalidTransitionError,
    JobNotFoundError,
    JobStore,
    JobStoreError,
)
from .workflow_run import WorkflowRunBackend

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
    "DataIngestionBackend",
    "DataIngestionStatusChecker",
    "KnowledgeIngestBackend",
    "KnowledgeIngestResult",
    "KnowledgeIngestStatusChecker",
    "KnowledgeUnderstandBackend",
    "KnowledgeUnderstandResult",
    "KnowledgeUnderstandStatusChecker",
    "KnowledgeEvent",
    "WorkflowRunBackend",
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
