"""Response models for the agent-exposed workflow routes (ADR-105 D-1).

"Type before expose": an operation enters the R4 agent-tool manifest only
with a request *and* a response model. These models describe the payloads
the six workflow routes (list, describe, trigger, execution status, results,
cancel) already return; they do not change them:

- ``extra="allow"`` keeps every key a handler returns, declared or not;
- every field a branch can leave out is optional with a default, and the
  routes set ``response_model_exclude_unset=True``, so a default is never
  added to a payload that did not carry the key;
- numeric fields are ``int | float`` (pydantic's smart union keeps an int an
  int), and free-form JSON stays ``Any`` / ``dict[str, Any]``.

Tightening a type here is a contract change, made with the r4 SDK in step.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

Number = int | float


class _OpenModel(BaseModel):
    """Base for payload models: undeclared keys pass through unchanged."""

    model_config = ConfigDict(extra="allow")


class WorkflowListResponse(_OpenModel):
    """``GET /umrs/workflows`` (operation ``workflow_list``)."""

    status: str = Field(..., description="'ok'")
    count: int | None = Field(default=None, description="Number of rows")
    workflows: list[dict[str, Any]] | None = Field(
        default=None, description="Catalog rows (own, shared, system)"
    )


class WorkflowDescribeResponse(_OpenModel):
    """``GET /umrs/workflows/{workflow_name}`` (operation ``workflow_describe``)."""

    status: str = Field(..., description="'ok'")
    contract_version: str | None = None
    workflow_name: str | None = None
    version: Any = None
    versions: list[str] | None = Field(
        default=None, description="Every registered version, highest first (ADR-113)"
    )
    source_sha256: str | None = Field(
        default=None, description="Digest of this version's files (ADR-113 D-2)"
    )
    description: Any = None
    scope: str | None = None
    user_id: Any = None
    parameter_schema: dict[str, Any] | None = Field(
        default=None,
        description="Free-form JSON Schema the trigger parameters must satisfy",
    )
    steps: list[Any] | None = None
    # Passed through from the uploaded workflow.json unvalidated: a malformed
    # upload ("triggers": {}) must not turn describe into a 500.
    requirements: Any = None
    triggers: Any = None


class WorkflowTriggerResponse(_OpenModel):
    """``POST /umrs/workflows/{workflow_name}/trigger`` (operation ``workflow_run``)."""

    status: str = Field(..., description="'ok'")
    action: str | None = None
    workflow_name: str | None = None
    workflow_version: str | None = Field(
        default=None, description="The version the run pinned (ADR-113 D-4)"
    )
    source_sha256: str | None = Field(
        default=None, description="Digest of the files the run pinned"
    )
    source_path: str | None = None
    steps: Number | None = Field(default=None, description="Declared step count")
    params: dict[str, Any] | None = None
    execution_id: str | None = None
    durable_job_id: str | None = None
    durable_job: dict[str, Any] | None = None
    message: str | None = None
    intent_id: str | None = Field(
        default=None,
        description="sha256 of the signed intent of a run under your own HPC account (ADR-112)",
    )
    grant_id: str | None = Field(
        default=None, description="The standing grant the run executes under (ADR-112 wave C)"
    )
    run_spec_id: str | None = Field(
        default=None, description="sha256 of the run spec the server composed under the grant"
    )


class WorkflowExecution(_OpenModel):
    """One execution row (registry record, or derived from the durable job)."""

    execution_id: str | None = None
    workflow_name: str | None = None
    workflow_version: str | None = None
    source_sha256: str | None = None
    user_id: str | None = None
    status: str | None = None
    submitted_at: Number | None = None
    started_at: Number | None = None
    finished_at: Number | None = None
    error: Any = None
    progress: Number | None = None
    params: dict[str, Any] | None = None
    steps: list[Any] | None = None
    artifacts: list[Any] | None = None
    durable_job_id: str | None = None
    durable_job: dict[str, Any] | None = None
    cancel_requested: bool | None = None
    identity_provenance: str | None = None
    gates: dict[str, Any] | None = None


class WorkflowExecutionResponse(_OpenModel):
    """Execution status and cancel (``workflow_status``, ``workflow_cancel``)."""

    status: str = Field(..., description="'ok'")
    execution: WorkflowExecution | None = None


class WorkflowResultsBody(_OpenModel):
    """Per-step and final outputs gathered so far."""

    steps: dict[str, Any] | None = None
    final: Any = None


class WorkflowResultsResponse(_OpenModel):
    """``GET …/executions/{execution_id}/results`` (operation ``workflow_results``)."""

    status: str = Field(..., description="'ok'")
    execution_id: str | None = None
    execution_status: str | None = None
    progress: Number | None = None
    error: Any = None
    workflow_version: str | None = None
    source_sha256: str | None = None
    results: WorkflowResultsBody | None = None
    artifacts: list[Any] | None = None
    durable_job_id: str | None = None
    durable_job: dict[str, Any] | None = None


# ── ADR-105 D-4: plan -> commit ────────────────────────────────────────────


class WorkflowPlanStep(_OpenModel):
    """One step as the plan resolved it."""

    name: str = Field(..., description="Step name")
    lane: str = Field(..., description="Executor lane the step runs on")
    gate: bool = Field(..., description="Manual-approval gate (runs no code)")
    gpu: bool = Field(..., description="Whether the step gets a GPU")
    gpus: int = Field(..., description="GPUs per attempt (0 without a GPU)")
    attempt_seconds: int = Field(
        ...,
        description="Wall bound of one attempt (the HPC --time, else the step timeout)",
    )
    attempts: int = Field(..., description="1 + the step's max_retries")
    max_wall_seconds: int = Field(..., description="attempt_seconds x attempts")


class WorkflowPlanQuota(_OpenModel):
    """The caller's agent-unit budget (``api/agent_quota``); reported, not
    charged: a workflow run is not an agent unit."""

    active_units: int | None = Field(
        default=None, description="Live agent units now (null when unreadable)"
    )
    max_units: int = Field(..., description="Per-user ceiling (0 = uncapped)")


class WorkflowPlanResponse(_OpenModel):
    """``POST /umrs/workflows/{workflow_name}/plan`` (operation ``workflow_plan``).

    Nothing is stored: send ``signed`` and ``digest`` back unchanged to
    ``.../commit`` before ``expires_at``.
    """

    status: str = Field(..., description="'ok'")
    plan_id: str = Field(
        ..., description="'plan-' + digest[:24]; the run's idempotency key"
    )
    digest: str = Field(
        ..., description="Server HMAC-SHA256 over the canonical `signed`"
    )
    expires_at: str = Field(..., description="UTC ISO-8601; commit refuses after it")
    signed: dict[str, Any] = Field(..., description="The exact dict the digest covers")
    workflow_name: str
    version: Any = None
    steps: list[WorkflowPlanStep]
    lane: str = Field(
        ..., description="The run's lane (the uploaded-code lane when any)"
    )
    partition: str | None = Field(
        default=None, description="Slurm partition when a step runs on an HPC lane"
    )
    account: str | None = Field(
        default=None, description="Slurm account when a step runs on an HPC lane"
    )
    gpu_steps: int = Field(..., description="Steps that get a GPU")
    max_wall_seconds: int = Field(
        ...,
        description=(
            "Upper bound: the sum of every step's attempt bound x attempts, as "
            "if the steps ran one after another; queue wait and a crash "
            "re-dispatch of the whole run are not counted"
        ),
    )
    max_gpu_hours: float = Field(
        ...,
        description=(
            "Upper bound: sum over GPU steps of max_wall_seconds x gpus / 3600; "
            "signed, and re-checked on commit"
        ),
    )
    quota: WorkflowPlanQuota
    auto_commit_max_gpu_hours: float | None = Field(
        default=None, description="The project's auto-commit threshold (null = none)"
    )
    require_plan: bool = Field(
        ..., description="Whether the project refuses direct triggers"
    )
    auto_commit_allowed: bool = Field(
        ..., description="Threshold set and max_gpu_hours <= threshold"
    )
    requires_intent: bool = Field(
        default=False,
        description=(
            "The run executes under your own HPC account (ADR-112): it starts "
            "only from a signed intent (r4 workflow trigger --site), never "
            "from a commit"
        ),
    )
    grant_id: str | None = Field(
        default=None,
        description=(
            "The one standing grant of yours that covers this run (ADR-112 wave "
            "C), else null; a grant is not a per-run approval: plan and commit still apply"
        ),
    )


class WorkflowCommitResponse(WorkflowTriggerResponse):
    """``POST /umrs/workflows/{workflow_name}/commit`` (operation
    ``workflow_commit``): the trigger payload plus the plan it committed."""

    plan_id: str | None = None
    existing_run: bool | None = Field(
        default=None,
        description="True when this plan was already committed: the payload names that run",
    )
