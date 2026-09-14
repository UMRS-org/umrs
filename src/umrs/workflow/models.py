"""Workflow schema with per-step outputs support.

Extends the canonical workflow definition model with per-step outputs:
  [{name, path_or_ref, kind, ingest}]

Version bump with backward compatibility. Validation in upload path.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# Schema version bump — outputs section added in v2
WORKFLOW_SCHEMA_VERSION = "2.0.0"


class StepType(str, Enum):
    """Types of workflow steps."""

    MCP_CALL = "mcp_call"
    UMRS_JOB = "umrs_job"
    RAG_QUERY = "rag_query"
    LLM_PROMPT = "llm_prompt"
    SHELL_EXEC = "shell_exec"
    PYTHON_EXEC = "python_exec"
    MANUAL_GATE = "manual_gate"


class StepStatus(str, Enum):
    """Individual step execution status."""

    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""

    DRAFT = "draft"
    READY = "ready"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepOutput:
    """Output artifact from a workflow step.

    Per-step outputs allow downstream steps to reference and chain outputs,
    and optionally trigger knowledge ingestion on terminal-success.
    """

    name: str
    description: str = ""
    kind: str = "artifact"  # "artifact" | "file" | "metadata" | "signal"
    path_or_ref: str | None = None
    ingest: bool = False  # If true, auto-ingest output into knowledge store

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "path_or_ref": self.path_or_ref,
            "ingest": self.ingest,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepOutput:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            kind=data.get("kind", "artifact"),
            path_or_ref=data.get("path_or_ref"),
            ingest=data.get("ingest", False),
        )


@dataclass
class StepInput:
    """Input parameter for a workflow step or workflow definition."""

    name: str
    description: str = ""
    input_type: str = "string"
    required: bool = True
    default: Any | None = None
    source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.input_type,
            "required": self.required,
            "default": self.default,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepInput:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            input_type=data.get("type") or data.get("input_type") or "string",
            required=data.get("required", True),
            default=data.get("default"),
            source=data.get("source"),
        )


@dataclass
class WorkflowStep:
    """A single step in a workflow.

    Extends the base step model with per-step outputs (TASK-031).
    """

    id: str
    name: str
    action: str  # The action to execute
    step_type: StepType = StepType.UMRS_JOB
    inputs: list[StepInput] = field(default_factory=list)
    outputs: list[StepOutput] = field(
        default_factory=list
    )  # TASK-031: per-step outputs
    depends_on: list[str] = field(default_factory=list)
    description: str = ""
    timeout_seconds: int = 3600
    retry_count: int = 0
    retry_delay_seconds: int = 5

    # Runtime state (not persisted)
    status: StepStatus = StepStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = None

    @property
    def has_outputs(self) -> bool:
        """Check if this step declares outputs."""
        return len(self.outputs) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "step_type": self.step_type.value,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],  # TASK-031
            "depends_on": self.depends_on,
            "description": self.description,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value if self.status else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowStep:
        step_type_str = data.get("step_type", "umrs_job")
        try:
            step_type = StepType(step_type_str)
        except ValueError:
            step_type = StepType.UMRS_JOB

        return cls(
            id=data["id"],
            name=data["name"],
            action=data["action"],
            step_type=step_type,
            inputs=[StepInput.from_dict(i) for i in data.get("inputs", [])],
            outputs=[
                StepOutput.from_dict(o) for o in data.get("outputs", [])
            ],  # TASK-031
            depends_on=data.get("depends_on", []),
            description=data.get("description", ""),
            timeout_seconds=data.get("timeout_seconds", 3600),
            status=StepStatus(data["status"])
            if data.get("status")
            else StepStatus.PENDING,
        )


@dataclass
class WorkflowDefinition:
    """Workflow definition with per-step outputs support (TASK-031).

    Schema version v2 adds per-step outputs with ingest flags.
    Backward compatible: old workflows without outputs section load fine.
    """

    id: str
    name: str
    description: str = ""
    version: str = WORKFLOW_SCHEMA_VERSION
    steps: list[WorkflowStep] = field(default_factory=list)
    inputs: list[StepInput] = field(default_factory=list)
    outputs: list[StepOutput] = field(default_factory=list)  # Workflow-level outputs
    metadata: dict[str, Any] = field(default_factory=dict)
    status: WorkflowStatus = WorkflowStatus.DRAFT
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    created_by: str | None = None

    @property
    def step_outputs(self) -> dict[str, list[StepOutput]]:
        """Map step id -> its outputs for quick lookup."""
        return {step.id: step.outputs for step in self.steps if step.has_outputs}

    @property
    def needs_outputs_section(self) -> bool:
        """Check if this workflow declares any step-level outputs.

        Used by upload validation to determine if output-dependent
        processing (e.g., auto-ingestion) is needed.
        """
        return any(step.has_outputs for step in self.steps) or len(self.outputs) > 0

    def validate_outputs(self) -> list[str]:
        """Validate output references are valid.

        Returns list of validation errors (empty if valid).
        """
        errors: list[str] = []
        step_ids = {s.id for s in self.steps}

        # Check step outputs reference valid from_step sources
        for step in self.steps:
            for output in step.outputs:
                if output.from_step and output.from_step not in step_ids:
                    errors.append(
                        f"Step '{step.id}' output '{output.name}' references "
                        f"non-existent step '{output.from_step}'"
                    )
                if output.ingest and not output.path_or_ref:
                    errors.append(
                        f"Step '{step.id}' output '{output.name}' has ingest=true "
                        f"but no path_or_ref"
                    )

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "steps": [s.to_dict() for s in self.steps],
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],  # TASK-031
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDefinition:
        status_str = data.get("status", "draft")
        try:
            status = WorkflowStatus(status_str)
        except ValueError:
            status = WorkflowStatus.DRAFT

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", WORKFLOW_SCHEMA_VERSION),
            steps=[WorkflowStep.from_dict(s) for s in data.get("steps", [])],
            inputs=[StepInput.from_dict(i) for i in data.get("inputs", [])],
            outputs=[
                StepOutput.from_dict(o) for o in data.get("outputs", [])
            ],  # TASK-031
            metadata=data.get("metadata", {}),
            status=status,
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
            created_by=data.get("created_by"),
        )


@dataclass
class WorkflowExecution:
    """Runtime state for a workflow execution."""

    id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(
        default_factory=dict
    )  # output_name -> artifact_ref
    error: str | None = None
    error_step: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "step_results": self.step_results,
            "output_artifacts": self.output_artifacts,
            "error": self.error,
            "error_step": self.error_step,
        }


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO datetimes while tolerating missing values."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def generate_workflow_id() -> str:
    """Generate a new workflow ID."""
    return str(uuid.uuid4())


def generate_step_id(prefix: str = "step") -> str:
    """Generate a new step ID."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
