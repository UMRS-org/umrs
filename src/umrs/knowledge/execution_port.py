"""Workflow execution port — UMRS-owned protocol for registered workflow runs.

The research loop depends on this Protocol. The R4 adapter
(``R4ClientWorkflowExecutor``) is injected at the composition layer so UMRS
never imports top-level ``r4``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4


@runtime_checkable
class WorkflowExecutionPort(Protocol):
    """Narrow bridge to the registered workflow lifecycle."""

    def trigger(self, workflow_name: str, params: dict[str, Any]) -> str:
        """Start a workflow; return ``execution_id``."""
        ...

    def status(self, execution_id: str) -> dict[str, Any]:
        """Return current execution status payload."""
        ...

    def wait(
        self, execution_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Block until terminal; return status/results payload."""
        ...

    def results(self, execution_id: str) -> dict[str, Any]:
        """Return structured results for an execution."""
        ...

    def cancel(self, execution_id: str) -> dict[str, Any]:
        """Request cancellation; return status payload."""
        ...


@dataclass
class FakeWorkflowExecutionPort:
    """Test double that records calls and returns canned durable responses."""

    durable_job_id: str = "job-fake-durable"
    execution_id_prefix: str = "exec-fake"
    wait_payload: dict[str, Any] | None = None
    results_payload: dict[str, Any] | None = None
    status_payload: dict[str, Any] | None = None
    cancel_payload: dict[str, Any] | None = None
    require_durable: bool = True
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=list
    )
    _executions: dict[str, dict[str, Any]] = field(default_factory=dict)
    _counter: int = 0

    def trigger(self, workflow_name: str, params: dict[str, Any]) -> str:
        self.calls.append(("trigger", (workflow_name, params), {}))
        if self.require_durable and not self.durable_job_id:
            raise RuntimeError("durable job backend required for real research mode")
        self._counter += 1
        execution_id = (
            f"{self.execution_id_prefix}-{self._counter:04d}-{uuid4().hex[:6]}"
        )
        self._executions[execution_id] = {
            "execution_id": execution_id,
            "workflow_name": workflow_name,
            "params": dict(params),
            "status": "accepted",
            "durable_job_id": self.durable_job_id,
            "outputs": {"message": "fake workflow completed"},
        }
        return execution_id

    def status(self, execution_id: str) -> dict[str, Any]:
        self.calls.append(("status", (execution_id,), {}))
        if self.status_payload is not None:
            return dict(self.status_payload)
        record = self._executions.get(execution_id, {})
        return {
            "execution_id": execution_id,
            "status": record.get("status", "succeeded"),
            "durable_job_id": record.get("durable_job_id", self.durable_job_id),
            "workflow_name": record.get("workflow_name", ""),
        }

    def wait(
        self, execution_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        self.calls.append(("wait", (execution_id,), {"timeout": timeout}))
        if self.wait_payload is not None:
            return dict(self.wait_payload)
        record = self._executions.get(execution_id, {})
        payload = {
            "execution_id": execution_id,
            "status": "succeeded",
            "durable_job_id": record.get("durable_job_id", self.durable_job_id),
            "workflow_name": record.get("workflow_name", ""),
            "outputs": record.get("outputs", {"message": "fake workflow completed"}),
        }
        if self.require_durable and not payload.get("durable_job_id"):
            raise RuntimeError("durable job backend required for real research mode")
        return payload

    def results(self, execution_id: str) -> dict[str, Any]:
        self.calls.append(("results", (execution_id,), {}))
        if self.results_payload is not None:
            return dict(self.results_payload)
        record = self._executions.get(execution_id, {})
        workflow_name = record.get("workflow_name", "")
        outputs = record.get("outputs", {"message": "fake workflow completed"})
        metrics: dict[str, Any] = {}
        # Demo 08 YieldPredictor vs batch-3 CSV — canned durable lifecycle numbers.
        if workflow_name == "demo08_yield_eval":
            metrics = {
                "batch_id": "batch-3",
                "yield_pct": 85.0,
                "co2_uptake_mmol_g": 4.21,
            }
            outputs = {**outputs, **metrics}
        return {
            "execution_id": execution_id,
            "status": "succeeded",
            "durable_job_id": record.get("durable_job_id", self.durable_job_id),
            "workflow_name": workflow_name,
            "outputs": outputs,
            "artifacts": [],
            "metrics": metrics,
        }

    def cancel(self, execution_id: str) -> dict[str, Any]:
        self.calls.append(("cancel", (execution_id,), {}))
        if self.cancel_payload is not None:
            return dict(self.cancel_payload)
        if execution_id in self._executions:
            self._executions[execution_id]["status"] = "cancelled"
        return {
            "execution_id": execution_id,
            "status": "cancelled",
            "durable_job_id": self._executions.get(execution_id, {}).get(
                "durable_job_id", self.durable_job_id
            ),
        }
