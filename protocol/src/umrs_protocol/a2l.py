"""A2L conservation contracts shared between r4, UMRS, and the a2l domain.

Stdlib-only dataclasses (ADR-005). These describe the *conserved additions*
to execution history owned by the A2L conservation domain (ADR-034):

  artifact identities (deterministic content+context hash), lineage edges,
  and agent-execution annotations — all keyed to the authoritative
  execution spine by ``job_id`` / ``attempt_id``. A2L joins against the
  spine at query time; it never copies execution state.

Wire form uses ISO-8601 strings for timestamps so DTOs stay plain and
serde-stable across HTTP and ZMQ transports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class A2lArtifactStatus(str, Enum):
    """FSM status for conserved artifacts (pending → running → completed|failed)."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Legal FSM transitions; retry = new artifact (new derive call), never a rewind.
A2L_ARTIFACT_TRANSITIONS: dict[A2lArtifactStatus, frozenset] = {
    A2lArtifactStatus.PENDING: frozenset({A2lArtifactStatus.RUNNING}),
    A2lArtifactStatus.RUNNING: frozenset({A2lArtifactStatus.COMPLETED, A2lArtifactStatus.FAILED}),
    A2lArtifactStatus.COMPLETED: frozenset(),
    A2lArtifactStatus.FAILED: frozenset(),
}


def a2l_artifact_transition_ok(current: str, target: str) -> bool:
    """Check whether an artifact status transition is legal."""
    try:
        cur = A2lArtifactStatus(current)
        tgt = A2lArtifactStatus(target)
    except ValueError:
        return False
    return tgt in A2L_ARTIFACT_TRANSITIONS.get(cur, frozenset())


class A2lExecutionStatus(str, Enum):
    """Status of one conserved agent execution annotation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ────────────────────────────────────────────────────────────────────────────
# Artifacts
# ────���───────────────────────────────────────────────────────────────────────


@dataclass
class A2lArtifactRecord:
    """One conserved artifact identity + metadata.

    ``id`` is the deterministic SHA-256 content+context hash. Operational
    linkage to the execution spine lives in ``job_id``/``attempt_id``
    (empty when not yet attributed) — never copy job state here.
    """

    id: str
    type: str
    uri: str
    user_id: str
    status: str = A2lArtifactStatus.PENDING.value
    type_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None
    hash_version: int = 2
    artifact_class: str | None = None
    # Authoritative spine keys (ADR-034 join-not-copy)
    job_id: str = ""
    attempt_id: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "uri": self.uri,
            "user_id": self.user_id,
            "status": self.status,
            "type_schema": self.type_schema,
            "metadata": self.metadata,
            "project_id": self.project_id,
            "hash_version": self.hash_version,
            "artifact_class": self.artifact_class,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lArtifactRecord:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})

    def is_terminal(self) -> bool:
        return self.status in (
            A2lArtifactStatus.COMPLETED.value,
            A2lArtifactStatus.FAILED.value,
        )


@dataclass
class A2lArtifactEdge:
    """Operational lineage edge: parent artifact → child artifact."""

    parent_id: str
    child_id: str
    relationship_type: str = "derived_from"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "relationship_type": self.relationship_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lArtifactEdge:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


@dataclass
class A2lArtifactDeriveRequest:
    """Agent write path: derive a child artifact from existing parents.

    Guarantees lineage is created atomically with the child record.
    ``job_id``/``attempt_id`` attribute the derivation to the spine.
    """

    parents: list[str]
    operation: str
    type: str
    uri: str
    user_id: str
    input_data_hash: str
    params: dict[str, Any] = field(default_factory=dict)
    type_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None
    artifact_class: str | None = None
    job_id: str = ""
    attempt_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "parents": self.parents,
            "operation": self.operation,
            "type": self.type,
            "uri": self.uri,
            "user_id": self.user_id,
            "input_data_hash": self.input_data_hash,
            "params": self.params,
            "type_schema": self.type_schema,
            "metadata": self.metadata,
            "project_id": self.project_id,
            "artifact_class": self.artifact_class,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lArtifactDeriveRequest:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


# ────────────────────────────────────────────────────────────────────────────
# Agent executions
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class A2lAgentExecutionRecord:
    """One conserved agent-execution annotation.

    Carries trace/outputs/errors — the *interpretation* of an execution,
    keyed to the authoritative spine via ``job_id``/``attempt_id``
    (ADR-034: join, never copy status/timestamps).
    """

    id: str
    agent_name: str
    user_id: str
    status: str = A2lExecutionStatus.PENDING.value
    agent_version: str = ""
    model_ref: str = ""
    prompt_hash: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    execution_trace: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None
    # Authoritative spine keys (ADR-034 join-not-copy)
    job_id: str = ""
    attempt_id: str = ""
    error: str = ""
    created_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "user_id": self.user_id,
            "status": self.status,
            "agent_version": self.agent_version,
            "model_ref": self.model_ref,
            "prompt_hash": self.prompt_hash,
            "params": self.params,
            "execution_trace": self.execution_trace,
            "project_id": self.project_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lAgentExecutionRecord:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


@dataclass
class A2lAgentExecutionCreate:
    """Create request for one agent-execution annotation."""

    agent_name: str
    user_id: str
    agent_version: str = ""
    model_ref: str = ""
    prompt_hash: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    project_id: str | None = None
    job_id: str = ""
    attempt_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "user_id": self.user_id,
            "agent_version": self.agent_version,
            "model_ref": self.model_ref,
            "prompt_hash": self.prompt_hash,
            "params": self.params,
            "project_id": self.project_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lAgentExecutionCreate:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})


@dataclass
class A2lAgentExecutionUpdate:
    """Patch an agent-execution annotation (status/trace/error)."""

    status: str | None = None
    execution_trace: dict[str, Any] | None = None
    error: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "execution_trace": self.execution_trace,
            "error": self.error,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> A2lAgentExecutionUpdate:
        return cls(**{k: v for k, v in value.items() if k in cls.__dataclass_fields__})
