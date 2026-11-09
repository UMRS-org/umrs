"""Admission of a run under a standing grant (ADR-112 wave C; plan §6
C/D.0 C4 "Admission", K11, K12, K14, K19).

A run whose identity resolves to ``user_account`` and that carries no
signed intent runs under one of the person's grants:

1. **The grant.** With ``grant_id`` it must be the caller's here
   (``grant_not_found``), active and within its window (``grant_invalid``
   ``expired`` / ``cancelled``) and cover the run (``not_covered``).
   Without one, the candidates are the caller's active grants here that
   list the verified project, the version and the link's account and cover
   now (K11): exactly one is used; none or several is
   ``hpcaccess.signature_required`` (428) naming the candidates. Never a
   fallback (I-1). A loop passes the grant it recorded (K12).
2. **The run spec.** The exact B2 intent body, unsigned (``build_intent``:
   the same rules and refusals, the same HMAC ``run_id``), with
   ``max_run_seconds = min(RUN_MAX_SECONDS, expires - issued)`` of the
   grant (K19); then the gate's own ``grant_covers`` pre-check and the
   envelope size (16 MiB, K1).
3. **Budget** (D3): the whole run's estimate (Σ ``grant_cost``) against the
   project's budget, advisory (K14: each step's pending ledger write binds).

Nothing is created before all of this passed. Errors are the handler's JSON.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

#: The gate's raised ``SUBMIT_MAX_STDIN`` (K1): the whole submit envelope.
ENVELOPE_MAX_BYTES = 16 * 1024 * 1024
_ENVELOPE_ROOM = 256
#: The gate's clock slack on a grant's window.
WINDOW_SLACK = 120


class SignatureRequired(LookupError):
    """No single grant covers the run: the person signs, or names one."""

    def __init__(self, candidates: list[str]) -> None:
        super().__init__(", ".join(candidates))
        self.candidates = candidates


class GrantEnded(RuntimeError):
    """K12: the grant a loop recorded has ended; the loop stops."""

    def __init__(self, grant_id: str, detail: str = "") -> None:
        super().__init__(f"grant_ended:{grant_id}" + (f" ({detail})" if detail else ""))
        self.grant_id = grant_id


@dataclass(frozen=True)
class GrantRun:
    """A grant-backed run: the grant, the run spec and the pinned skeleton."""

    grant_id: str
    run_spec: bytes
    workflow_steps: bytes
    body: dict[str, Any]

    @property
    def run_spec_id(self) -> str:
        return hashlib.sha256(self.run_spec).hexdigest()

    @property
    def run_id(self) -> str:
        return str(self.body["run_id"])

    def spec(self) -> dict[str, str]:
        """What the durable job spec and the run record keep (0137)."""
        return {
            "grant_id": self.grant_id,
            "run_spec_b64": base64.b64encode(self.run_spec).decode("ascii"),
            "run_spec_id": self.run_spec_id,
            "workflow_steps_b64": base64.b64encode(self.workflow_steps).decode("ascii"),
        }


def _gate() -> Any:
    from r4_contracts import hpc_gate

    return hpc_gate


# ── the grant ────────────────────────────────────────────────────────────────


def grant_rows(identity: Any) -> tuple[list[dict[str, Any]], Any]:
    """The person's grants at this venue and the context they are judged in."""
    from umrs.hpc_accounts.grant_service import build_context
    from umrs.hpc_accounts.grants import grant_store_for
    from umrs.shared.hpc_user_work_access import current_site, site_account_store

    store = site_account_store(identity.site)
    rows = grant_store_for(store).list(identity.principal)
    return rows, build_context(
        identity.site, identity.principal, store, None, current_site()
    )


def _in_window(body: dict[str, Any], now: float) -> bool:
    return int(body["not_before"]) <= now + WINDOW_SLACK and now - WINDOW_SLACK <= int(
        body["expires"]
    )


def _explicit(
    rows: list[dict[str, Any]], ctx: Any, grant_id: str, now: float
) -> dict[str, Any]:
    """The named grant: the caller's here, active, in its window, not stale."""
    from umrs.hpc_accounts.grants import (
        GrantInvalid,
        GrantNotFound,
        signed_body,
        stale_reasons,
    )

    row = next((r for r in rows if r["grant_id"] == grant_id), None)
    if row is None:
        raise GrantNotFound(f"no grant {grant_id} of yours here")
    if row["status"] != "active":
        raise GrantInvalid(
            "cancelled" if row["status"] == "cancelled" else "expired", grant_id
        )
    if not _in_window(signed_body(row), now):
        raise GrantInvalid("expired", f"grant {grant_id} is outside its window")
    stale = stale_reasons(row, ctx)
    if stale:
        raise GrantInvalid("not_covered", stale[0])
    return row


def choose_grant(
    rows: list[dict[str, Any]],
    ctx: Any,
    grant_id: str,
    run: dict[str, Any],
    account: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """The grant this run uses (C4 admission step 2)."""
    from umrs.hpc_accounts.grants import candidates, signed_body

    current = time.time() if now is None else now
    if grant_id:
        return _explicit(rows, ctx, grant_id, current)
    found = [
        r
        for r in candidates(
            rows,
            ctx,
            project=str(run["project"]),
            workflow=(str(run["workflow"]), str(run["version"])),
            account=account,
        )
        if _in_window(signed_body(r), current)
    ]
    if len(found) != 1:
        raise SignatureRequired([r["grant_id"] for r in found])
    return found[0]


def covering_grant_id(identity: Any, run: dict[str, Any]) -> str | None:
    """``/plan``: the one grant that would cover the run now, else None
    (none, several, or unreadable)."""
    try:
        rows, ctx = grant_rows(identity)
        return str(
            choose_grant(rows, ctx, "", run, str(identity.slurm_account or ""))[
                "grant_id"
            ]
        )
    except Exception:  # noqa: BLE001 - advisory: the trigger decides
        return None


# ── the run spec ─────────────────────────────────────────────────────────────


def _proposal_request(
    target: Any,
    identity: Any,
    run: dict[str, Any],
    params: dict[str, Any],
    inputs: Any,
    *,
    service_pricing: bool = False,
) -> Any:
    from umrs.hpc_accounts.site_profile import published, service_pricing_profile
    from umrs.server.intent_proposal import ProposalRequest
    from umrs.shared.hpc_user_work_access import current_site

    site = current_site()
    if service_pricing:
        profile, sha = service_pricing_profile(site), ""
    else:
        profile, _data, sha = published(site)
    return ProposalRequest(
        workflow=str(run["workflow"]),
        version=str(run["version"]),
        source_sha256=str(target.entry.get("source_sha256") or ""),
        project=str(run["project"]),
        principal=str(run["principal"]),
        login=identity.login,
        account=str(identity.slurm_account or ""),
        params={**params, "project_id": str(run["project"])},
        workflow_json=target.workflow_json,
        source_path=target.source_path,
        site=site,
        profile=profile,
        profile_sha256=sha,
        listed=dict(inputs or {}),
    )


def run_max_seconds(grant: dict[str, Any]) -> int:
    """K19: ``min(RUN_MAX_SECONDS, expires - issued)`` of the grant."""
    from umrs.server.intent_proposal import RUN_MAX_SECONDS

    window = int(grant["expires"]) - int(grant["issued"])
    return max(60, min(RUN_MAX_SECONDS, window))


def compose_run_spec(
    target: Any,
    identity: Any,
    run: dict[str, Any],
    params: dict[str, Any],
    grant: dict[str, Any],
    *,
    inputs: Any = None,
    s3: Any = None,
    now: int | None = None,
) -> dict[str, Any]:
    """The run spec body: B2's ``build_intent`` with the grant's run window."""
    from umrs.server.intent_proposal import build_intent, proposal_s3

    req = _proposal_request(target, identity, run, params, inputs)
    client = s3 if s3 is not None else proposal_s3(req.params, req.project)
    body = build_intent(req, client, now=now)
    body["max_run_seconds"] = run_max_seconds(grant)
    return body


def encode_run(
    grant_id: str, body: dict[str, Any], workflow_json: dict[str, Any]
) -> GrantRun:
    """Canonical bytes of the run spec (``validate_intent``) and the skeleton."""
    from umrs.hpc_accounts.grants import skeleton_bytes
    from umrs.server.intent_proposal import encode_intent

    payload, _id = encode_intent(body)
    return GrantRun(grant_id, payload, skeleton_bytes(workflow_json), body)


def check_params(
    grant: dict[str, Any], run: dict[str, Any], params: dict[str, Any]
) -> None:
    """The caller's params against the grant's schema for this workflow
    (K9a), before anything is read: a run that could never be covered is
    refused ``not_covered`` without staging its inputs."""
    from umrs.hpc_accounts.grants import GrantInvalid

    key = (str(run["workflow"]), str(run["version"]))
    entry = next(
        (e for e in grant["workflows"] if (e["name"], e["version"]) == key), None
    )
    if entry is None:
        return  # the coverage pre-check names the workflow
    project = str(run["project"])
    problem = _gate().params_problems(
        grant["param_schema"],
        {**params, "project_id": project},
        project,
        entry["params"],
    )
    if problem:
        raise GrantInvalid("not_covered", f"params: {problem}"[:300])


def composed_run_spec(
    target: Any,
    identity: Any,
    run: dict[str, Any],
    params: dict[str, Any],
    grant: dict[str, Any],
    inputs: Any,
) -> dict[str, Any]:
    """:func:`compose_run_spec`; an input the server cannot read (staging
    refuses it) is ``not_covered`` ``inputs: ...``, a typed refusal, not a
    crash of the trigger."""
    from umrs.hpc_accounts.grants import GrantInvalid
    from umrs.server.step_executor import StepExecutionError

    try:
        return compose_run_spec(target, identity, run, params, grant, inputs=inputs)
    except StepExecutionError as exc:
        raise GrantInvalid("not_covered", f"inputs: {exc.message}"[:300]) from exc


def check_covers(grant_row: dict[str, Any], run: GrantRun) -> None:
    """The gate's own ``grant_covers`` and the envelope size (K1)."""
    from umrs.hpc_accounts.grants import GrantInvalid, signed_body

    skeleton = json.loads(run.workflow_steps)
    problem = _gate().grant_covers(signed_body(grant_row), run.body, skeleton)
    if problem:
        raise GrantInvalid("not_covered", str(problem)[:300])
    spec = run.spec()
    size = len(grant_row["payload_b64"]) + len(grant_row["signature"]) + _ENVELOPE_ROOM
    size += len(spec["run_spec_b64"]) + len(spec["workflow_steps_b64"])
    if size > ENVELOPE_MAX_BYTES:
        raise GrantInvalid(
            "too_large", f"the run's envelope exceeds {ENVELOPE_MAX_BYTES} bytes"
        )


# ── budget (advisory, K14) ───────────────────────────────────────────────────


def run_estimate_seconds(body: dict[str, Any], profile: dict[str, Any]) -> int:
    """Σ canonical K4 GPU-seconds; an unpriceable partition refuses the run."""
    from umrs.hpc_accounts.usage_ledger import LedgerUnavailable, estimate

    total = 0
    for step in body["steps"]:
        try:
            partition = profile["partitions"][step["partition"]]
            if not isinstance(partition, dict) or not partition:
                raise TypeError("invalid partition")
            res = estimate(step["cpus"], step["gpus"], step["time"], partition)
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerUnavailable(
                f"site profile partition {step.get('partition')!r} is unavailable or invalid"
            ) from exc
        total += res.gpus * res.time_s
    return total


def check_run_budget(body: dict[str, Any], profile: dict[str, Any]) -> None:
    from umrs.hpc_accounts.usage_ledger import check_budget, get_usage_ledger

    check_budget(
        get_usage_ledger(), str(body["project"]), run_estimate_seconds(body, profile)
    )


def check_service_run_budget(
    target: Any, identity: Any, run: dict[str, Any], params: dict[str, Any]
) -> None:
    """D3 only for a positively capped service-account project.

    Legacy uncapped projects do not need a pricing context. Capped projects
    use the service site's K4 resources, never gated-user requirements.
    """
    from umrs.hpc_accounts.usage_ledger import (
        LedgerUnavailable,
        budget_seconds,
        get_usage_ledger,
    )
    from umrs.server.intent_proposal import resource_estimates

    project = str(run["project"])
    if not project:
        return
    try:
        cap = get_usage_ledger().max_gpu_hours(project)
    except Exception as exc:
        raise LedgerUnavailable(
            f"the project budget is unavailable: {type(exc).__name__}"
        ) from exc
    if budget_seconds(cap) <= 0:
        return
    try:
        req = _proposal_request(
            target, identity, run, params, None, service_pricing=True
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise LedgerUnavailable(
            f"the service site's K4 pricing context is unavailable: {type(exc).__name__}"
        ) from exc
    body = {"project": project, "steps": resource_estimates(req)}
    check_run_budget(body, req.profile)


# ── the whole admission ──────────────────────────────────────────────────────


def admit_grant(
    identity: Any,
    payload: dict[str, Any],
    params: dict[str, Any],
    run: dict[str, Any],
    target: Any,
) -> GrantRun:
    """A run's grant and run spec, or the typed exception of the first miss."""
    from umrs.hpc_accounts.grants import signed_body
    from umrs.hpc_accounts.site_profile import published

    rows, ctx = grant_rows(identity)
    row = choose_grant(
        rows,
        ctx,
        str(payload.get("grant_id") or ""),
        run,
        str(identity.slurm_account or ""),
    )
    grant = signed_body(row)
    check_params(grant, run, params)
    body = composed_run_spec(
        target, identity, run, params, grant, payload.get("inputs")
    )
    gated = encode_run(row["grant_id"], body, target.workflow_json)
    check_covers(row, gated)
    check_run_budget(body, published(ctx_site())[0])
    return gated


def ctx_site() -> Any:
    from umrs.shared.hpc_user_work_access import current_site

    return current_site()


def refusal_json(exc: Exception) -> str | None:
    """The handler's error JSON for a grant admission refusal (None: not one)."""
    from umrs.hpc_accounts.grants import GrantInvalid, GrantNotFound
    from umrs.hpc_accounts.intents import IntentInvalid
    from umrs.hpc_accounts.usage_ledger import BudgetExceeded, LedgerUnavailable

    from .run_intent_admission import error_json

    if isinstance(exc, SignatureRequired):
        ids = ", ".join(exc.candidates) or "none"
        return error_json(
            "hpcaccess.signature_required",
            f"no single grant covers this run (candidates: {ids}): name one with grant_id, "
            "or sign the run with 'r4 workflow trigger <name> --site <site>' (nothing ran)",
            candidates=exc.candidates,
        )
    table: tuple[tuple[type, str], ...] = (
        (GrantInvalid, "hpcaccess.grant_invalid"),
        (GrantNotFound, "hpcaccess.grant_not_found"),
        (BudgetExceeded, "hpcaccess.budget_exceeded"),
        (LedgerUnavailable, "hpcaccess.ledger_unavailable"),
        (IntentInvalid, "hpcaccess.intent_invalid"),
    )
    for kind, code in table:
        if isinstance(exc, kind):
            return error_json(
                code, str(exc), reason=str(getattr(exc, "reason", "") or "")
            )
    return None


__all__ = [
    "ENVELOPE_MAX_BYTES",
    "GrantEnded",
    "GrantRun",
    "SignatureRequired",
    "admit_grant",
    "check_covers",
    "check_params",
    "check_service_run_budget",
    "choose_grant",
    "compose_run_spec",
    "composed_run_spec",
    "encode_run",
    "refusal_json",
    "run_estimate_seconds",
    "run_max_seconds",
]
