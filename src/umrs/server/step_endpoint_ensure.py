"""Trigger-time endpoint ensure (ADR-109 D-4).

When a workflow run starts, every step that will route through the registry
(``requires`` / ``model_type`` / ``model``, no ``api_base`` of its own) gets
its endpoint started now, so the step waits for it (bounded, see
:func:`umrs.server.step_router.resolve_waiting`) instead of failing with "no
matching active endpoint".

For each such step with registry candidates but no live or starting endpoint
the run's user may use (own project, or shared with them: ADR-078):

* a service model (a ``svc-*`` role alias, or a model whose roster entry
  carries a ``service`` block) -> the relay starts one replica
  (:func:`umrs.server.endpoint_relay.ensure_service_replica`); it stays the
  one owner of service lifetime (ADR-106 D-3);
* any other registered model -> ``provision_endpoint`` as the triggering
  user, user tier, ``ttl_sec=0``, ``wait=False``, ``access:
  [project:<run project>]``. The handler's own checks (model scope, quota,
  user gate, failure cap, recipe policy) decide; nothing here widens them.

Lifetime is unchanged: the endpoint ends on ``r4 endpoint down`` or its Slurm
walltime, never with the run. ``metadata.started_by_run`` names the run that
started it. Plans are made and executed under one process lock and re-read
the registry, so two triggers never start two endpoints for one model (the
second sees the first's PROVISIONING row). A step pinning nothing, with
several candidate models, starts nothing (the choice is the author's).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from umrs.pipeline_sdk.workflow import StepSpec
from umrs.server.step_router import StepRouter

logger = logging.getLogger(__name__)

#: Endpoint metadata naming the run whose trigger started the endpoint.
STARTED_BY_RUN_KEY = "started_by_run"

#: One ensure at a time in this process: plan + provision under the lock, so
#: a concurrent trigger sees the row the first one registered.
_ENSURE_LOCK = threading.Lock()


@dataclass(frozen=True)
class EnsureTarget:
    """One endpoint to start: *model* is a registry model or a role alias."""

    step: str
    model: str
    service_alias: bool = False


def _routed_step(spec: dict[str, Any]) -> StepSpec | None:
    """The step when the router will resolve it through the registry."""
    if spec.get("gate") or spec.get("manual_gate"):
        return None
    if (spec.get("config") or {}).get("api_base"):
        return None  # the author pinned the upstream (explicit_config)
    try:
        step = StepSpec.from_dict(spec)
    except Exception:  # noqa: BLE001 - the runner reports a malformed step
        return None
    return step if (step.requires or step.model_type or step.model) else None


def _target_model(
    step: StepSpec, names: list[str], router: StepRouter
) -> tuple[str, bool] | None:
    """``(model, is_alias)`` to start, or None when the choice is not the
    step's: a pinned model (or role alias) among the candidates, else the
    only candidate of a step that pinned nothing."""
    pinned = router.pinned_names(step)
    if step.model and any(name in pinned for name in names):
        return step.model, step.model not in names  # not a candidate: an alias
    if not step.model and len(names) == 1:
        return names[0], False
    return None


def _target_for(
    step: StepSpec, router: StepRouter, project_id: str
) -> EnsureTarget | None:
    names = [
        str(c.get("name")) for c in router.registry_candidates(step) if c.get("name")
    ]
    if not names or router.has_usable_endpoint(names, project_id):
        return None
    chosen = _target_model(step, names, router)
    if chosen is None:
        logger.info(
            "ensure: step %s names no single model of %d candidates; nothing started",
            step.name,
            len(names),
        )
        return None
    model, alias = chosen
    return EnsureTarget(step=step.name, model=model, service_alias=alias)


def plan_ensure(
    steps_spec: Iterable[dict[str, Any]], router: StepRouter, project_id: str
) -> list[EnsureTarget]:
    """The endpoints to start for these steps, one per model."""
    targets: dict[str, EnsureTarget] = {}
    for spec in steps_spec:
        step = _routed_step(spec)
        target = _target_for(step, router, project_id) if step else None
        if target is not None:
            targets.setdefault(target.model, target)
    return list(targets.values())


def _tag_started_by_run(job_id: str | None, execution_id: str) -> None:
    """Record the run on a row it started (never over an earlier record)."""
    if not job_id:
        return
    from umrs.orchestrator.endpoint_registry import get_registry

    registry = get_registry()
    row = registry.get_by_job_id(job_id)
    if row is not None and STARTED_BY_RUN_KEY not in (row.metadata or {}):
        registry.update_metadata(job_id, {STARTED_BY_RUN_KEY: execution_id})


async def _provision_user_tier(
    model: str, *, project_id: str, user_id: str
) -> dict[str, Any]:
    """The ``r4 endpoint up`` handler as the triggering user (never admin)."""
    from umrs.api.endpoints import ProvisionRequest, provision_endpoint
    from umrs.orchestrator.endpoint_registry import get_registry
    from umrs.server.endpoint_relay import provision_result

    request = ProvisionRequest(
        model=model,
        project_id=project_id,
        tier="user",
        ttl_sec=0,
        access=[f"project:{project_id}"],
        wait=False,
    )
    result = await provision_endpoint(
        request=request,
        project_id=project_id,
        is_admin=False,
        requesting_user=user_id,
        registry=get_registry(),
    )
    answer = provision_result(result)
    started = answer.get("status") == "provisioning"
    return {
        "outcome": "provisioned" if started else str(answer.get("status") or "failed"),
        "model": model,
        "job_id": answer.get("job_id") if started else None,
        "error": answer.get("error"),
    }


async def _ensure_target(
    target: EnsureTarget, *, project_id: str, user_id: str, execution_id: str
) -> dict[str, Any]:
    from umrs.orchestrator.endpoint_access import EndpointCaller
    from umrs.server.endpoint_relay import ensure_service_replica

    reason = f"run {execution_id} step {target.step}"
    caller = EndpointCaller(user_id=user_id, project_id=project_id)
    answer = await ensure_service_replica(target.model, reason=reason, caller=caller)
    if answer["outcome"] != "no_service":
        return {"step": target.step, "via": "service", **answer}
    if target.service_alias:
        # A role alias no service entry carries: nothing to start (a user-tier
        # copy would bypass the replica controller, ADR-109 alternatives).
        return {"step": target.step, "via": "none", **answer}
    answer = await _provision_user_tier(
        target.model, project_id=project_id, user_id=user_id
    )
    return {"step": target.step, "via": "user", **answer}


def _run_one(
    target: EnsureTarget, *, project_id: str, user_id: str, execution_id: str
) -> dict[str, Any]:
    try:
        result = asyncio.run(
            _ensure_target(
                target,
                project_id=project_id,
                user_id=user_id,
                execution_id=execution_id,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a refused start fails its step by name later
        logger.warning(
            "ensure: %s for step %s refused: %s", target.model, target.step, exc
        )
        return {
            "step": target.step,
            "model": target.model,
            "outcome": "refused",
            "error": str(exc),
        }
    _tag_started_by_run(result.get("job_id"), execution_id)
    return result


def ensure_step_endpoints(
    steps_spec: Iterable[dict[str, Any]],
    *,
    router: StepRouter,
    project_id: str,
    user_id: str,
    execution_id: str,
) -> list[dict[str, Any]]:
    """Start the endpoints these steps need; one result row per model.

    Runs in the run's own thread (never on the trigger path) and never waits
    for readiness: provisions submit and return.
    """
    steps = list(steps_spec)
    with _ENSURE_LOCK:
        targets = plan_ensure(steps, router, project_id)
        results = [
            _run_one(
                target,
                project_id=project_id,
                user_id=user_id,
                execution_id=execution_id,
            )
            for target in targets
        ]
    for row in results:
        logger.info("ensure for run %s: %s", execution_id, row)
    return results
