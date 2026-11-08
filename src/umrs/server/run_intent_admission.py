"""Admission of a workflow run under the person's own HPC account (ADR-112
B2; plan §5 B2.0 §10.3). One place for the HTTP trigger and the ZMQ
``RUN_WORKFLOW`` handler (the HTTP route reaches the handler too).

A run whose HPC-lane identity resolves to ``user_account`` starts only with
the person's signed intent (``intent`` + ``signature`` in the payload);
without one it is refused ``hpcaccess.signature_required`` (428) before
anything is created. An intent on any other run is ``intent_invalid``. An
intent whose ``run_id`` already names a run is ``intent_used`` (409). The
run then executes the intent's ``params`` under ``execution_id = run_id``.

Errors are the handler's JSON (``{"status": "error", "code", "message"}``);
the HTTP route maps the ``hpcaccess.*`` code to its typed status.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from umrs.hpc_accounts.intents import (
    IntentInvalid,
    SignedIntent,
    TriggerBinding,
    check_trigger_intent,
    decode_signed_intent,
)
from umrs.orchestrator.hpc_identity import HpcIdentity
from umrs.server.hpc_user_work import is_user_account

#: Params the platform injects (never chosen by the trigger body alone).
PLATFORM_PARAM_KEYS = ("project_id", "input_files")


def error_json(code: str, message: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "code": code, "message": message, **extra})


@dataclass(frozen=True)
class RunAdmission:
    """What the run starts with: its params and, for a gated run, its intent."""

    params: dict[str, Any]
    intent: SignedIntent | None = None

    @property
    def execution_id(self) -> str | None:
        return self.intent.run_id if self.intent is not None else None

    def new_execution_id(self) -> str:
        """The run's execution id: the intent's ``run_id``, else a fresh one."""
        return self.execution_id or f"wf-{uuid.uuid4().hex[:12]}"

    def intent_spec(self) -> dict[str, str] | None:
        return self.intent.spec() if self.intent is not None else None


def _carries_params(params: dict[str, Any]) -> bool:
    """Whether the trigger body named params of its own (beyond the
    verified ``project_id`` the server injects)."""
    return any(key != "project_id" for key in params)


def _binding(
    identity: HpcIdentity, run: dict[str, Any], params: dict[str, Any]
) -> TriggerBinding:
    return TriggerBinding(
        workflow=str(run.get("workflow") or ""),
        version=str(run.get("version") or ""),
        project=str(run.get("project") or ""),
        site=identity.site,
        principal=str(run.get("principal") or ""),
        login=identity.login,
        account=str(identity.slurm_account or ""),
        params=params if _carries_params(params) else None,
    )


def _signature_required(identity: HpcIdentity) -> str:
    site = identity.site
    return error_json(
        "hpcaccess.signature_required",
        f"Site {site} runs this workflow under your own account there: sign the "
        f"run with 'r4 workflow trigger <name> --site {site}' (nothing ran)",
        site=site,
    )


def _invalid(exc: IntentInvalid) -> str:
    return error_json(
        "hpcaccess.intent_invalid",
        f"the signed intent does not match this run: {exc}",
        reason=exc.reason,
    )


def _signed(payload: dict[str, Any]) -> bool:
    return bool(payload.get("intent") or payload.get("signature"))


def _check_project_param(intent: SignedIntent) -> None:
    params = intent.body.get("params")
    project = intent.body.get("project")
    if not isinstance(params, dict) or params.get("project_id") != project:
        raise IntentInvalid(
            "params_mismatch", "the intent's params.project_id is not its project"
        )


def admit_run(
    identity: HpcIdentity | None,
    payload: dict[str, Any],
    params: dict[str, Any],
    run: dict[str, Any],
    run_exists: Any = None,
) -> tuple[RunAdmission | None, str | None]:
    """``(admission, None)`` or ``(None, error JSON)`` for one run.

    *run* names ``workflow``, ``version``, ``project`` (verified) and
    ``principal`` (the caller). *run_exists* answers whether an execution id
    already has a run (``intent_used``)."""
    if not is_user_account(identity):
        if _signed(payload):
            return None, _invalid(IntentInvalid("not_user_account", "no gated identity"))
        return RunAdmission(params), None
    assert identity is not None
    if not _signed(payload):
        return None, _signature_required(identity)
    try:
        intent = decode_signed_intent(payload.get("intent"), payload.get("signature"))
        check_trigger_intent(intent, _binding(identity, run, params))
        _check_project_param(intent)
    except IntentInvalid as exc:
        return None, _invalid(exc)
    if run_exists is not None and run_exists(intent.run_id):
        return None, error_json(
            "hpcaccess.intent_used",
            f"run {intent.run_id} of this intent already exists: sign a new one",
        )
    return RunAdmission(dict(intent.body["params"]), intent), None


def execution_exists(execution_id: str) -> bool:
    """Whether this execution id already has a run (memory or the shared
    ``workflow_executions`` table)."""
    from umrs.server.workflow_execution_registry import get_execution_registry

    return get_execution_registry().get(execution_id) is not None


__all__ = [
    "PLATFORM_PARAM_KEYS",
    "RunAdmission",
    "admit_run",
    "error_json",
    "execution_exists",
]
