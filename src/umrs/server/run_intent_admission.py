"""Admission of a workflow run under the person's own HPC account (ADR-112
B2; plan §5 B2.0 §10.3). One place for the HTTP trigger and the ZMQ
``RUN_WORKFLOW`` handler (the HTTP route reaches the handler too).

A run whose HPC-lane identity resolves to ``user_account`` starts only with
the person's signed intent (``intent`` + ``signature`` in the payload);
without one it is refused ``hpcaccess.signature_required`` (428) before
anything is created. An intent on any other run is ``intent_invalid``. An
intent whose ``run_id`` already names a run is ``intent_used`` (409). The
run then executes the intent's ``params`` under ``execution_id = run_id``.

Wave C (plan §6 C/D.0 C4): a ``user_account`` run without an intent runs
under one of the person's standing grants (:mod:`.grant_admission`): the
trigger may name it (``grant_id``); an intent together with a ``grant_id``
is ``intent_invalid`` (``both``); a ``grant_id`` on any other run is
``grant_invalid`` (``not_covered``). Without a single covering grant the
run is still ``signature_required`` (428), naming the candidates.

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
    """What the run starts with: its params and, for a gated run, its
    intent or its grant run (wave C)."""

    params: dict[str, Any]
    intent: SignedIntent | None = None
    grant: Any = None

    @property
    def execution_id(self) -> str | None:
        if self.intent is not None:
            return self.intent.run_id
        return self.grant.run_id if self.grant is not None else None

    def new_execution_id(self) -> str:
        """The run's execution id: the intent's or run spec's ``run_id``,
        else a fresh one."""
        return self.execution_id or f"wf-{uuid.uuid4().hex[:12]}"

    def intent_spec(self) -> dict[str, str] | None:
        """The gated run's mapping the spec and the record keep: the
        intent's ``{b64, signature, id}`` or the grant run's fields."""
        if self.intent is not None:
            return self.intent.spec()
        return self.grant.spec() if self.grant is not None else None


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


def _budget_refusal(body: dict[str, Any]) -> str | None:
    from umrs.hpc_accounts.site_profile import published
    from umrs.hpc_accounts.usage_ledger import BudgetExceeded, LedgerUnavailable
    from umrs.shared.hpc_user_work_access import current_site

    from .grant_admission import check_run_budget

    try:
        check_run_budget(body, published(current_site())[0])
    except (BudgetExceeded, LedgerUnavailable) as exc:
        return error_json(exc.code, str(exc))
    return None


def _service_budget_refusal(
    identity: HpcIdentity | None,
    run: dict[str, Any],
    params: dict[str, Any],
    target: Any,
) -> str | None:
    if identity is None or target is None:
        return None
    from umrs.hpc_accounts.usage_ledger import BudgetExceeded, LedgerUnavailable

    from .grant_admission import check_service_run_budget

    try:
        check_service_run_budget(target, identity, run, params)
    except (BudgetExceeded, LedgerUnavailable) as exc:
        return error_json(exc.code, str(exc))
    return None


def _not_gated(
    identity: HpcIdentity | None,
    payload: dict[str, Any],
    params: dict[str, Any],
    run: dict[str, Any],
    target: Any,
) -> tuple[RunAdmission | None, str | None]:
    """A run on any other identity carries neither an intent nor a grant."""
    if _signed(payload):
        return None, _invalid(IntentInvalid("not_user_account", "no gated identity"))
    if payload.get("grant_id"):
        return None, error_json(
            "hpcaccess.grant_invalid",
            "this run does not run under your own HPC account: it takes no grant",
            reason="not_covered",
        )
    refused = _service_budget_refusal(identity, run, params, target)
    if refused is not None:
        return None, refused
    return RunAdmission(params), None


def _grant_admission(
    identity: HpcIdentity,
    payload: dict[str, Any],
    params: dict[str, Any],
    run: dict[str, Any],
    target: Any,
) -> tuple[RunAdmission | None, str | None]:
    """Wave C: the run under one of the person's grants (or the refusal)."""
    from .grant_admission import admit_grant, refusal_json

    if target is None:
        return None, _signature_required(identity)
    try:
        gated = admit_grant(identity, payload, params, run, target)
    except Exception as exc:
        refused = refusal_json(exc)
        if refused is None:
            raise
        return None, refused
    return RunAdmission(dict(gated.body["params"]), grant=gated), None


def admit_run(
    identity: HpcIdentity | None,
    payload: dict[str, Any],
    params: dict[str, Any],
    run: dict[str, Any],
    run_exists: Any = None,
    target: Any = None,
) -> tuple[RunAdmission | None, str | None]:
    """``(admission, None)`` or ``(None, error JSON)`` for one run.

    *run* names ``workflow``, ``version``, ``project`` (verified) and
    ``principal`` (the caller). *run_exists* answers whether an execution id
    already has a run (``intent_used``). *target* (the run's definition:
    ``entry``, ``source_path``, ``workflow_json``) lets a run without an
    intent run under a grant (wave C)."""
    if not is_user_account(identity):
        return _not_gated(identity, payload, params, run, target)
    assert identity is not None
    if not _signed(payload):
        return _grant_admission(identity, payload, params, run, target)
    if payload.get("grant_id"):
        return None, _invalid(
            IntentInvalid(
                "both", "a run carries a signed intent or a grant_id, not both"
            )
        )
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
    refused = _budget_refusal(intent.body)
    if refused is not None:
        return None, refused
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
