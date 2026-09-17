"""HTTP-backed WorkflowExecutionPort for cluster-venue drivers (WP-05 r2).

A cluster-side engine running inside the mini-RiVault/Slurm sandbox has no
durable job store; real-mode workflow executions have to travel the
guard-allowed gateway lane exactly like an r4 client would:

    POST /gw/workflows/{workflow_name}/trigger        → execution_id
    GET  /gw/workflows/executions/{execution_id}      → status polling
    GET  /gw/workflows/executions/{execution_id}/results → structured results
    POST /gw/workflows/executions/{execution_id}/cancel  → cooperative cancel

Verification headers come from the same env the loop job injects
(``R4_LOOP_OWNER`` / ``R4_LOOP_PROJECT`` + the HMAC secret) — stdlib only
(the driver image must stay minimal).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from .execution_port import WorkflowExecutionPort
from .http_loop_store import ClusterLoopTransport, _owner_headers_from_env

logger = logging.getLogger(__name__)

_TERMINAL_STATUS = {"succeeded", "failed", "cancelled", "completed", "success"}
_SUCCESS_STATUS = {"succeeded", "completed", "success"}


def _transport_from_env() -> ClusterLoopTransport:
    base = os.environ.get("R4_GATEWAY_BASE", "").rstrip("/")
    if not base:
        raise RuntimeError("R4_GATEWAY_BASE is required for the cluster workflow port")
    return ClusterLoopTransport(
        base_url=base,
        owner_headers=_owner_headers_from_env(),
        path_prefix="/gw/loopjob/workflows",
    )


class ClusterHttpWorkflowExecutionPort(WorkflowExecutionPort):
    """WorkflowExecutionPort over the guard-allowed gateway lane (stdlib only).

    ``wait`` polls until the run lands at a terminal status and returns the
    status projection; ``results`` returns the structured results document.
    Terminal responses without a durable_job_id are raised honestly in
    ``results`` (the engine requires the linkage, and a silent missing gate
    hides this for days).
    """

    def __init__(
        self,
        transport: ClusterLoopTransport | None = None,
        *,
        poll_interval: float = 5.0,
        max_wait_seconds: float = 12 * 3600,
    ) -> None:
        self._transport = transport or _transport_from_env()
        self._poll = poll_interval
        self._max_wait = max_wait_seconds

    def trigger(self, workflow_name: str, params: dict[str, Any]) -> str:
        payload = {"user_id": os.environ.get("R4_LOOP_OWNER", ""), "params": params}
        data = self._transport.post(f"/{workflow_name}/trigger", payload)
        execution_id = str(
            data.get("execution_id")
            or data.get("executionId")
            or data.get("id")
            or ""
        ).strip()
        if not execution_id:
            raise RuntimeError(
                f"workflow trigger returned no execution id for {workflow_name}: "
                f"transport answered {str(data)[:160]}"
            )
        return execution_id

    def status(self, execution_id: str) -> dict[str, Any]:
        return self._transport.get(f"/executions/{execution_id}")

    def wait(
        self, execution_id: str, timeout: float | None = None
    ) -> dict[str, Any]:
        goal = time.monotonic() + float(timeout or self._max_wait)
        while True:
            payload = self._transport.get(f"/executions/{execution_id}")
            execution = payload.get("execution") or payload
            status = str(
                execution.get("status") or payload.get("execution_status") or ""
            ).lower()
            if status in _TERMINAL_STATUS:
                return execution
            if time.monotonic() >= goal:
                raise TimeoutError(
                    f"workflow execution {execution_id} did not reach a terminal "
                    f"status in {timeout or self._max_wait}s (last: {status!r})"
                )
            time.sleep(self._poll)

    def results(self, execution_id: str) -> dict[str, Any]:
        data = dict(self._transport.get(f"/executions/{execution_id}/results"))
        # Envelope normalization (live receipts 2026-09-17): the route answers
        # {"status": "ok", "execution_status": "succeeded", "durable_job_id": ...}
        # while the engine reads a top-level terminal `status` + durable_job_id.
        # Both variants fold into one shape here; the durable-id gate still
        # refuses honestly when the linkage is absent.
        status = str(
            data.get("execution_status") or data.get("status") or ""
        ).lower()
        if status in _TERMINAL_STATUS and not data.get("durable_job_id"):
            raise RuntimeError(
                f"workflow execution {execution_id} reports terminal status "
                f"without a durable_job_id — the engine needs the linkage to mint "
                "lineage; do not smudge this through a fabricated value"
            )
        data["status"] = status
        return data

    def cancel(self, execution_id: str) -> dict[str, Any]:
        return self._transport.post(f"/executions/{execution_id}/cancel", {})
