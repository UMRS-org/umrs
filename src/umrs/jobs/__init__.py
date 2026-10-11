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

Exports are resolved lazily (PEP 562): importing ``umrs.jobs`` must not drag
``psycopg2`` (jobs.store) into driver containers that only need a leaf
module such as ``umrs.jobs.timing`` — first real cluster leg crashed the
sim engine at exactly that import (leg 381165).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORT_MAP: dict[str, str] = {
    # adapters
    "CleanupBackend": ".adapters",
    "CleanupStatusChecker": ".adapters",
    "EndpointProvisionBackend": ".adapters",
    "EndpointProvisionStatusChecker": ".adapters",
    "IndexBuildBackend": ".adapters",
    "IndexBuildStatusChecker": ".adapters",
    "create_backend_factory": ".adapters",
    "create_status_checker": ".adapters",
    # data_ingestion
    "DataIngestionBackend": ".data_ingestion",
    "DataIngestionStatusChecker": ".data_ingestion",
    # knowledge_adapters
    "KnowledgeEmbedBackend": ".knowledge_adapters",
    "KnowledgeEmbedResult": ".knowledge_adapters",
    "KnowledgeEmbedStatusChecker": ".knowledge_adapters",
    "KnowledgeEvent": ".knowledge_adapters",
    "KnowledgeIngestBackend": ".knowledge_adapters",
    "KnowledgeIngestResult": ".knowledge_adapters",
    "KnowledgeIngestStatusChecker": ".knowledge_adapters",
    "KnowledgeUnderstandBackend": ".knowledge_adapters",
    "KnowledgeUnderstandResult": ".knowledge_adapters",
    "KnowledgeUnderstandStatusChecker": ".knowledge_adapters",
    # ha
    "ClusterCoordinator": ".ha",
    "DistributedLockManager": ".ha",
    "HeartbeatError": ".ha",
    "HeartbeatManager": ".ha",
    "InstanceInfo": ".ha",
    "LockAcquisitionError": ".ha",
    "LockError": ".ha",
    "LockNotHeldError": ".ha",
    # models
    "AttemptStatus": ".models",
    "Job": ".models",
    "JobAttempt": ".models",
    "JobStatus": ".models",
    "JobType": ".models",
    "RestartPolicy": ".models",
    "compute_idempotency_key": ".models",
    "default_idempotency_key": ".models",
    "generate_attempt_id": ".models",
    "generate_job_id": ".models",
    # monitor
    "AttemptCheckResult": ".monitor",
    "AttemptMonitor": ".monitor",
    "CompositeStatusChecker": ".monitor",
    "EndpointProvisionChecker": ".monitor",
    # reconciler
    "Reconciler": ".reconciler",
    "ReconcileResult": ".reconciler",
    "reconcile_on_startup": ".reconciler",
    # scheduler
    "Scheduler": ".scheduler",
    "SimpleBackendAdapter": ".scheduler",
    # schema
    "ensure_schema": ".schema",
    "get_schema_version": ".schema",
    # service
    "JobService": ".service",
    "create_job_service": ".service",
    # store
    "AttemptNotFoundError": ".store",
    "DuplicateJobError": ".store",
    "InvalidTransitionError": ".store",
    "JobNotFoundError": ".store",
    "JobStore": ".store",
    "JobStoreError": ".store",
    # workflow_run
    "WorkflowRunBackend": ".workflow_run",
}


def __getattr__(name: str) -> Any:
    module = _EXPORT_MAP.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORT_MAP)
