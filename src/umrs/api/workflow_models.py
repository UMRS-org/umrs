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
    description: Any = None
    scope: str | None = None
    user_id: Any = None
    parameter_schema: dict[str, Any] | None = Field(
        default=None,
        description="Free-form JSON Schema the trigger parameters must satisfy",
    )
    steps: list[Any] | None = None
    requirements: list[Any] | None = None
    triggers: list[Any] | None = None


class WorkflowTriggerResponse(_OpenModel):
    """``POST /umrs/workflows/{workflow_name}/trigger`` (operation ``workflow_run``)."""

    status: str = Field(..., description="'ok'")
    action: str | None = None
    workflow_name: str | None = None
    source_path: str | None = None
    steps: Number | None = Field(default=None, description="Declared step count")
    params: dict[str, Any] | None = None
    execution_id: str | None = None
    durable_job_id: str | None = None
    durable_job: dict[str, Any] | None = None
    message: str | None = None


class WorkflowExecution(_OpenModel):
    """One execution row (registry record, or derived from the durable job)."""

    execution_id: str | None = None
    workflow_name: str | None = None
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
    results: WorkflowResultsBody | None = None
    artifacts: list[Any] | None = None
    durable_job_id: str | None = None
    durable_job: dict[str, Any] | None = None
