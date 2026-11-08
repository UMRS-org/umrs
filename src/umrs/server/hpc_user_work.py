"""The workflow side of this server's USER_WORK checks (ADR-112 D-2, A.3.4).

The identity binding (site config, admin bit, the resolver call) lives in the
foundation module :mod:`umrs.shared.hpc_user_work_access`, which the durable
job backends share. This module adds the one lane condition every workflow
entry point checks before it accepts a run: does the run use the site's
connected-HPC lane (staged steps run under the site's login)?

A staged step is user work whatever code it runs (ADR-112 D-2: "staged
steps"), so the condition is the lane, not uploaded code alone: a workflow
of built-in steps that asks for ``execution_mode: slurm`` reaches the same
login and allocation as an uploaded one (review 2026-11-08).
"""

from __future__ import annotations

import os
from typing import Any

from umrs.orchestrator.hpc_identity import HpcAccessRefused, refusal_message
from umrs.shared.hpc_user_work_access import user_work_refusal


def requests_hpc_lane(worker: str | None) -> bool:
    """Whether a run of built-in steps takes the HPC lane: the workflow's own
    lane, else the deployment's ``UMRS_STEP_EXECUTOR``, else in-process (the
    same choice as ``build_step_executor``)."""
    from umrs.server.step_trust import HPC_LANES

    chosen = worker or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc"
    return chosen.strip().lower() in HPC_LANES


def run_uses_hpc_lane(
    steps: list[dict[str, Any]], source_path: str | None, worker: str | None
) -> bool:
    """Whether a run of *steps* puts work on this site's HPC lane.

    Uploaded code goes where :func:`uploaded_code_lane` sends it; built-in
    steps go to the lane the workflow (or the deployment) asks for.
    """
    from umrs.server.step_trust import (
        HPC_LANES,
        uploaded_code_lane,
        workflow_has_uploaded_code,
    )

    if workflow_has_uploaded_code(steps, source_path):
        return uploaded_code_lane(worker) in HPC_LANES
    return requests_hpc_lane(worker)


def workflow_user_work_refusal(
    workflow_json: dict[str, Any],
    source_path: str | None,
    *,
    user_id: str | None,
    project_id: str | None,
) -> HpcAccessRefused | None:
    """The refusal for a run of *workflow_json*, or None.

    Only a run that uses the site's HPC lane is user work on the site's
    login; any other run is not checked here.
    """
    from umrs.server.workflow_runner import _executor_name_from_execution_mode

    steps = [s for s in workflow_json.get("steps") or [] if isinstance(s, dict)]
    worker = _executor_name_from_execution_mode(workflow_json.get("execution_mode"))
    if not run_uses_hpc_lane(steps, source_path, worker):
        return None
    return user_work_refusal(user_id=user_id, project_id=project_id)


def runner_refusal_text(exc: HpcAccessRefused) -> str:
    """The run record's error text for a refused user-work run."""
    return (
        "this run uses the connected HPC lane, which runs under the site's "
        f"login: {refusal_message(exc)}"
    )


__all__ = [
    "requests_hpc_lane",
    "run_uses_hpc_lane",
    "runner_refusal_text",
    "workflow_user_work_refusal",
]
