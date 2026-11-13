"""Provenance contract DTOs shared between r4 and UMRS.

Stdlib-only dataclasses representing execution provenance for
durable workflow and job runs.  A provenance record answers:

  What inputs, model refs, code versions, job attempts, artifacts,
  and policies produced this result?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an immutable artifact produced by a job or workflow."""

    uri: str  # s3://…/jobs/<job_id>/attempt_<n>/…
    media_type: str = ""
    size_bytes: int | None = None
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "uri": self.uri,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArtifactRef:
        """Build a reference from its wire representation."""
        return cls(**value)


@dataclass
class InputSnapshot:
    """Immutable input snapshot for deterministic step replay (MEM-B5).

    Captures the exact inputs provided to a step so the step can be
    re-executed with identical data.  Small payloads are stored inline;
    large payloads reference S3.
    """

    trace_id: str
    step_name: str
    inputs_hash: str | None = None
    inputs_data: dict[str, Any] | None = None  # inline for small payloads
    inputs_ref: str | None = None  # S3 URI for large payloads
    outputs_hash: str | None = None
    outputs_data: dict[str, Any] | None = None
    outputs_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "step_name": self.step_name,
            "inputs_hash": self.inputs_hash,
            "inputs_data": self.inputs_data,
            "inputs_ref": self.inputs_ref,
            "outputs_hash": self.outputs_hash,
            "outputs_data": self.outputs_data,
            "outputs_ref": self.outputs_ref,
        }


@dataclass
class ModelRef:
    """Reference to the model used in a job."""

    name: str
    path: str | None = None
    version: str | None = None
    source: str | None = None  # "filesystem_path" | "hf_repo_ref" | "pvc_ref"


@dataclass
class AttemptSummary:
    """Compact summary of a single execution attempt."""

    attempt_id: str
    attempt_no: int
    backend: str
    backend_ref: str | None = None
    status: str = ""
    started_at: str | None = None
    ended_at: str | None = None
    exit_reason: str | None = None
    duration_seconds: float | None = None


@dataclass
class ProvenanceRecord:
    """Full provenance record for a durable job or workflow run.

    Captures the complete lineage of a computation: what ran, on what
    inputs, with which model, how many attempts, what was produced, and
    under what policies.
    """

    job_id: str
    job_type: str
    project_id: str
    user_id: str
    status: str

    # What was requested
    spec: dict[str, Any] = field(default_factory=dict)

    # Execution timeline
    created_at: str | None = None
    completed_at: str | None = None
    total_duration_seconds: float | None = None

    # Authority and runtime
    execution_authority: str | None = None  # slurm | k8s | compose
    runtime_wrapper: str | None = None
    umrs_instance_id: str | None = None

    # Model provenance
    model: ModelRef | None = None

    # Attempt history
    attempts: list[AttemptSummary] = field(default_factory=list)
    total_attempts: int = 0
    commit_attempt_id: str | None = None

    # Artifacts produced
    artifacts: list[ArtifactRef] = field(default_factory=list)

    # Policy and config snapshot
    restart_policy: str | None = None
    max_attempts: int = 0
    idempotency_key: str | None = None

    # Workflow-specific fields
    workflow_name: str | None = None
    execution_id: str | None = None
    parent_job_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dict for JSON transport."""
        result: dict[str, Any] = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "status": self.status,
            "spec": self.spec,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "execution_authority": self.execution_authority,
            "runtime_wrapper": self.runtime_wrapper,
            "umrs_instance_id": self.umrs_instance_id,
            "total_attempts": self.total_attempts,
            "commit_attempt_id": self.commit_attempt_id,
            "restart_policy": self.restart_policy,
            "max_attempts": self.max_attempts,
            "idempotency_key": self.idempotency_key,
            "workflow_name": self.workflow_name,
            "execution_id": self.execution_id,
            "parent_job_id": self.parent_job_id,
        }
        if self.model:
            result["model"] = {
                "name": self.model.name,
                "path": self.model.path,
                "version": self.model.version,
                "source": self.model.source,
            }
        result["attempts"] = [
            {
                "attempt_id": a.attempt_id,
                "attempt_no": a.attempt_no,
                "backend": a.backend,
                "backend_ref": a.backend_ref,
                "status": a.status,
                "started_at": a.started_at,
                "ended_at": a.ended_at,
                "exit_reason": a.exit_reason,
                "duration_seconds": a.duration_seconds,
            }
            for a in self.attempts
        ]
        result["artifacts"] = [
            {
                "uri": a.uri,
                "media_type": a.media_type,
                "size_bytes": a.size_bytes,
                "sha256": a.sha256,
            }
            for a in self.artifacts
        ]
        return result


@dataclass
class EnvironmentSnapshot:
    """Frozen snapshot of the execution environment at job start."""

    python_version: str = ""
    umrs_version: str = ""
    r4_version: str = ""
    site_config_hash: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    container_image: str = ""
    container_runtime: str = ""  # singularity | podman | docker
    hostname: str = ""
    gpu_devices: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "umrs_version": self.umrs_version,
            "r4_version": self.r4_version,
            "site_config_hash": self.site_config_hash,
            "env_vars": self.env_vars,
            "container_image": self.container_image,
            "container_runtime": self.container_runtime,
            "hostname": self.hostname,
            "gpu_devices": self.gpu_devices,
        }


@dataclass
class CodeVersionRef:
    """Reference to a specific code version at execution time."""

    repo_url: str = ""
    commit_sha: str = ""
    branch: str = ""
    dirty: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_url": self.repo_url,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "dirty": self.dirty,
            "tags": self.tags,
        }


@dataclass
class ExecutionManifest:
    """Full execution manifest capturing environment, code, and provenance.

    Extends the ProvenanceRecord concept with frozen environment state,
    code versions, and step-level trace references for full reproducibility.
    Used by MEM-B3 (Execution Manifest DTO) in the Operational Memory Model.
    """

    provenance: ProvenanceRecord | None = None
    environment: EnvironmentSnapshot | None = None
    code_versions: list[CodeVersionRef] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)
    pipeline_name: str = ""
    pipeline_version: str = ""
    trigger: str = ""  # manual | scheduled | api | webhook
    tags: dict[str, str] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "trigger": self.trigger,
            "tags": self.tags,
            "created_at": self.created_at,
            "trace_ids": self.trace_ids,
        }
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        if self.environment:
            result["environment"] = self.environment.to_dict()
        result["code_versions"] = [cv.to_dict() for cv in self.code_versions]
        return result


@dataclass
class CheckpointRef:
    """Reference to a durable checkpoint for step resumption (MEM-B6)."""

    checkpoint_id: str
    job_id: str
    attempt_id: str
    step_name: str
    sequence_no: int = 0
    state_uri: str | None = None
    state_size_bytes: int = 0
    iteration: int = 0
    progress: float = 0.0
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "step_name": self.step_name,
            "sequence_no": self.sequence_no,
            "state_uri": self.state_uri,
            "state_size_bytes": self.state_size_bytes,
            "iteration": self.iteration,
            "progress": self.progress,
            "created_at": self.created_at,
        }


@dataclass
class WorkflowOutcomeSummary:
    """Aggregated workflow outcome statistics (MEM-D1)."""

    workflow_name: str
    total_runs: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    success_rate: float = 0.0
    avg_duration_sec: float | None = None
    avg_tokens: float | None = None
    failure_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "total_runs": self.total_runs,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "success_rate": self.success_rate,
            "avg_duration_sec": self.avg_duration_sec,
            "avg_tokens": self.avg_tokens,
            "failure_breakdown": self.failure_breakdown,
        }


@dataclass
class RecoveredSessionRef:
    """Summary of a recovered session after crash (MEM-A4)."""

    job_id: str
    execution_id: str | None = None
    job_status: str = "unknown"
    can_retry: bool = False
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    running_steps: int = 0
    resume_from_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "job_status": self.job_status,
            "can_retry": self.can_retry,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "running_steps": self.running_steps,
            "resume_from_step": self.resume_from_step,
        }


@dataclass
class VectorVersionRef:
    """Reference to an embedding model version (MEM-C2)."""

    version_id: str
    model_name: str
    model_version: str
    embedding_dimension: int = 0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "embedding_dimension": self.embedding_dimension,
            "is_active": self.is_active,
        }


@dataclass
class KnowledgeNamespaceRef:
    """Reference to a knowledge namespace partition (MEM-C4)."""

    namespace_id: str
    name: str
    scope: str  # default | project | experiment | user
    project_id: str | None = None
    experiment_id: str | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "name": self.name,
            "scope": self.scope,
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "is_active": self.is_active,
        }


@dataclass
class WorkflowForkRef:
    """Reference to a forked workflow run (MEM-B8)."""

    new_job_id: str
    source_job_id: str
    param_overrides: dict[str, Any] = field(default_factory=dict)
    lineage_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_job_id": self.new_job_id,
            "source_job_id": self.source_job_id,
            "param_overrides": self.param_overrides,
            "lineage_chain": self.lineage_chain,
        }


@dataclass
class KnowledgeSearchHitRef:
    """Reference to a cross-experiment knowledge search hit (MEM-C5)."""

    memory_id: str
    content: str
    namespace: str
    score: float = 0.0
    source: str = ""
    project_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "namespace": self.namespace,
            "score": self.score,
            "source": self.source,
            "project_id": self.project_id,
        }


@dataclass
class RoutingFeedbackRef:
    """Routing feedback reference DTO (MEM-D2)."""

    feedback_id: str
    job_id: str = ""
    step_name: str = ""
    model_selected: str = ""
    outcome: str = ""
    user_accepted: bool | None = None
    user_correction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "job_id": self.job_id,
            "step_name": self.step_name,
            "model_selected": self.model_selected,
            "outcome": self.outcome,
            "user_accepted": self.user_accepted,
            "user_correction": self.user_correction,
        }


@dataclass
class PlanningHeuristicRef:
    """Planning heuristic reference DTO (MEM-D3)."""

    heuristic_id: str
    workflow_name: str = ""
    rule_type: str = ""
    condition: str = ""
    action: str = ""
    confidence: float = 0.0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic_id": self.heuristic_id,
            "workflow_name": self.workflow_name,
            "rule_type": self.rule_type,
            "condition": self.condition,
            "action": self.action,
            "confidence": self.confidence,
            "is_active": self.is_active,
        }


@dataclass
class ReflectiveMemoryEntryRef:
    """Reflective memory entry reference DTO (MEM-D4)."""

    entry_id: str
    tier: str = ""
    source_store: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "tier": self.tier,
            "source_store": self.source_store,
            "summary": self.summary,
        }


@dataclass
class ReplayResultRef:
    """Step replay result reference DTO (MEM-B5)."""

    replay_id: str
    source_trace_id: str = ""
    new_trace_id: str = ""
    new_job_id: str = ""
    step_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "source_trace_id": self.source_trace_id,
            "new_trace_id": self.new_trace_id,
            "new_job_id": self.new_job_id,
            "step_name": self.step_name,
        }


@dataclass
class WorkloadIdentityRef:
    """Workload identity reference DTO (SEC-P3a)."""

    identity_id: str
    job_id: str = ""
    scope: str = ""
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "job_id": self.job_id,
            "scope": self.scope,
            "is_valid": self.is_valid,
        }


@dataclass
class SecretLeaseRef:
    """Secret lease reference DTO (SEC-P3b)."""

    lease_id: str
    job_id: str = ""
    secret_name: str = ""
    scope: str = ""
    is_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "job_id": self.job_id,
            "secret_name": self.secret_name,
            "scope": self.scope,
            "is_valid": self.is_valid,
        }
