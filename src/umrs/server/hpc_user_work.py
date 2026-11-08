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

from umrs.orchestrator.hpc_identity import (
    USER_ACCOUNT,
    HpcAccessRefused,
    HpcIdentity,
    refusal_message,
)
from umrs.shared.hpc_user_work_access import resolve_user_work


def requests_hpc_lane(worker: str | None) -> bool:
    """Whether a run of built-in steps takes the HPC lane: the workflow's own
    lane, else the deployment's ``UMRS_STEP_EXECUTOR``, else in-process (the
    same choice as ``build_step_executor``)."""
    from umrs.server.step_trust import HPC_LANES

    chosen = worker or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc"
    return chosen.strip().lower() in HPC_LANES


def lane_has_gated_path(lane: str | None) -> bool:
    """Whether the HPC executor *lane* builds has a gated path (ADR-112 B2):
    only the SSH-submitting staged lane does (``slurm-remote``, or ``slurm``
    on a front end, ``UMRS_SLURM_STEP_SUBMIT=remote``; the same choice as
    ``build_step_executor``). A local ``sbatch`` lane runs as the server's
    own login, so a ``user_account`` run there refuses ``lane_not_gated``
    (I-1; fixer finding 2026-11-08)."""
    from umrs.shared.uploaded_code import load_site_env

    load_site_env()
    chosen = (lane or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc").strip().lower()
    if chosen == "slurm-remote":
        return True
    return chosen == "slurm" and os.environ.get("UMRS_SLURM_STEP_SUBMIT") == "remote"


def hpc_lane_of(
    steps: list[dict[str, Any]], source_path: str | None, worker: str | None
) -> str | None:
    """The HPC lane a run of *steps* uses here, or None (it uses none).

    Uploaded code goes where :func:`uploaded_code_lane` sends it; built-in
    steps go to the lane the workflow (or the deployment) asks for.
    """
    from umrs.server.step_trust import (
        HPC_LANES,
        uploaded_code_lane,
        workflow_has_uploaded_code,
    )

    if workflow_has_uploaded_code(steps, source_path):
        lane = uploaded_code_lane(worker)
    else:
        lane = (worker or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc").strip().lower()
    return lane if lane in HPC_LANES else None


def run_uses_hpc_lane(
    steps: list[dict[str, Any]], source_path: str | None, worker: str | None
) -> bool:
    """Whether a run of *steps* puts work on this site's HPC lane."""
    return hpc_lane_of(steps, source_path, worker) is not None


def workflow_hpc_lane(workflow_json: dict[str, Any], source_path: str | None) -> str | None:
    """The HPC lane a run of *workflow_json* uses here, or None."""
    from umrs.server.workflow_runner import _executor_name_from_execution_mode

    steps = [s for s in workflow_json.get("steps") or [] if isinstance(s, dict)]
    worker = _executor_name_from_execution_mode(workflow_json.get("execution_mode"))
    return hpc_lane_of(steps, source_path, worker)


def workflow_uses_hpc_lane(
    workflow_json: dict[str, Any], source_path: str | None
) -> bool:
    """Whether a run of *workflow_json* puts work on this site's HPC lane."""
    return workflow_hpc_lane(workflow_json, source_path) is not None


def workflow_user_work_identity(
    workflow_json: dict[str, Any],
    source_path: str | None,
    *,
    user_id: str | None,
    project_id: str | None,
) -> HpcIdentity | None:
    """The identity a run of *workflow_json* uses on this site's HPC lane
    (None: the run does not use it), or :class:`HpcAccessRefused`.

    Staged steps on the SSH-submitting lane are the one lane with a gated
    path (wave B2), so a ``user_account`` identity is returned there; its
    run then needs a signed intent (:func:`intent_refusal_text`). On a lane
    without one it refuses ``lane_not_gated`` (:func:`lane_has_gated_path`)."""
    lane = workflow_hpc_lane(workflow_json, source_path)
    if lane is None:
        return None
    return resolve_user_work(
        user_id=user_id, project_id=project_id, gated=lane_has_gated_path(lane)
    )


def workflow_user_work_refusal(
    workflow_json: dict[str, Any],
    source_path: str | None,
    *,
    user_id: str | None,
    project_id: str | None,
) -> HpcAccessRefused | None:
    """The refusal for a run of *workflow_json*, or None.

    Only a run that uses the site's HPC lane is user work on the site's
    login; any other run is not checked here. A ``user_account`` identity
    is not refused here: the run's intent decides (``signature_required``).
    """
    try:
        workflow_user_work_identity(
            workflow_json, source_path, user_id=user_id, project_id=project_id
        )
    except HpcAccessRefused as exc:
        return exc
    return None


def is_user_account(identity: HpcIdentity | None) -> bool:
    return identity is not None and identity.mode == USER_ACCOUNT


def intent_refusal_text(identity: HpcIdentity | None, has_intent: bool) -> str:
    """Why a run with (or without) a signed intent may not run under
    *identity*, or "" (plan §5 B2.0 §10.3): a ``user_account`` run needs
    one (``hpcaccess.signature_required``); any other run must carry none
    (``hpcaccess.intent_invalid``)."""
    if is_user_account(identity) and not has_intent:
        site = identity.site if identity is not None else ""
        return (
            f"Site {site} runs this workflow under your own account there: "
            f"sign the run with 'r4 workflow trigger <name> --site {site}' "
            "(hpcaccess.signature_required)"
        )
    if has_intent and not is_user_account(identity):
        return (
            "this run carries a signed intent, but it does not run under your "
            "own HPC account here (hpcaccess.intent_invalid)"
        )
    return ""


def runner_refusal_text(exc: HpcAccessRefused) -> str:
    """The run record's error text for a refused user-work run."""
    return (
        "this run uses the connected HPC lane, which runs under the site's "
        f"login: {refusal_message(exc)}"
    )


__all__ = [
    "hpc_lane_of",
    "intent_refusal_text",
    "is_user_account",
    "lane_has_gated_path",
    "requests_hpc_lane",
    "run_uses_hpc_lane",
    "runner_refusal_text",
    "workflow_user_work_identity",
    "workflow_user_work_refusal",
    "workflow_uses_hpc_lane",
]
