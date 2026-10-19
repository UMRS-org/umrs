"""Shareable endpoint_provision submit policy (H-1, 2026-10-19).

The HTTP POST /umrs/jobs submit (``_enforce_endpoint_host_paths``) and the
ZMQ durable-job submit (``server/job_tracking_handler``) apply ONE policy to
an ``endpoint_provision`` spec (parent or fan-out child): a non-admin may
name only a catalog model it may provision, may not set env_vars,
container_image or tier=system, and its mount sources and a host model_path
it names must lie under its own roots (``orchestrator/caller_roots.py``).

Moved here verbatim from ``api/jobs.py`` on 2026-10-19 so the ZMQ lane
enforces the same rules without importing the HTTP module back.
"""

from __future__ import annotations

from typing import Any

from ..api.typed_errors import PlatformError, raise_typed
from ..jobs.models import JobType


def _spec_targets(
    job_type: JobType, spec: Any, fan_out_jobs: Any
) -> list[tuple[Any, Any]]:
    """(job_type, spec) of the parent and of every fan-out child with a known
    type (the fan-out builder rejects an unknown one itself)."""
    targets = [(job_type, spec)]
    for child in fan_out_jobs or []:
        try:
            child_type = JobType(str((child or {}).get("job_type") or ""))
        except ValueError:
            continue
        targets.append((child_type, (child or {}).get("spec") or {}))
    return targets


def _deny_unless(condition: bool, detail: str) -> None:
    """Raise 403 for a violated scope rule (keeps call-sites above budget)."""
    if not condition:
        raise_typed(PlatformError.jobs("forbidden", detail))


def _check_endpoint_spec(spec: dict[str, Any], user_id: str, project_id: str) -> None:
    from ..orchestrator.caller_roots import foreign_sources, raw_mount_sources

    _check_endpoint_spec_recipe(spec)
    foreign = foreign_sources(
        raw_mount_sources(spec.get("mounts")), project_id, user_id
    )
    _deny_unless(
        not foreign,
        f"mount source {foreign[0] if foreign else ''} is outside your own project and "
        "user data roots (UMRS_ENDPOINT_CALLER_MOUNT_ROOTS)",
    )
    _check_endpoint_spec_model(spec, user_id, project_id)


def _check_endpoint_spec_model(
    spec: dict[str, Any], user_id: str, project_id: str
) -> None:
    from umrs.api.endpoints import (
        _authorize_model_entry_for_caller,
        _resolve_registered_model_entry,
        _validated_model_path_override,
    )

    from ..orchestrator.caller_roots import model_path_in_scope

    # The executor resolves spec.get("model", "default") itself (adapters.py,
    # with a JSON-catalog fallback when Postgres fails): check that same name
    # and refuse when no entry resolves here, or an unchecked entry runs.
    model = str(spec.get("model", "default") or "").strip()
    entry = _resolve_registered_model_entry(model) if model else None
    if entry is None:
        raise_typed(
            PlatformError.jobs(
                "bad_request",
                f"model '{model}' is not in the model catalog; register it first",
            )
        )
    _authorize_model_entry_for_caller(entry, user_id=user_id, project_id=project_id)
    raw_path = str(spec.get("model_path") or "").strip()
    if not raw_path:
        return
    resolved = _validated_model_path_override(raw_path)
    _deny_unless(
        model_path_in_scope(resolved, entry, project_id=project_id, user_id=user_id),
        f"model path {resolved} is outside your own project and user data roots "
        "(UMRS_ENDPOINT_CALLER_MOUNT_ROOTS); register the model in the catalog or "
        "use a path under your own roots",
    )


def _check_endpoint_spec_recipe(spec: dict[str, Any]) -> None:
    """env_vars, container_image and tier=system are admin-only, as on
    POST /umrs/endpoints/provision. The executor merges spec.env_vars over the
    mount env (PROJECT_DATA_PATH, RAG_INDEX_PATH, MCP_SERVERS_PATH are host
    bind sources in serve-model.slurm), so env_vars would skip the mount rule.
    The partition is left to the executor's site check, as before."""
    from types import SimpleNamespace

    from umrs.api.endpoints import (
        _enforce_provision_authorization,
        _enforce_recipe_policy,
    )

    view = SimpleNamespace(
        partition=None,
        env_vars=spec.get("env_vars"),
        container_image=spec.get("container_image"),
        tier=spec.get("tier"),
    )
    _enforce_recipe_policy(False, view)
    _enforce_provision_authorization(view, is_admin=False)


def apply_caller_roots_env(
    env_vars: dict[str, str],
    *,
    tier: str | None = None,
    is_admin: bool | None = None,
    project_id: str,
    user_id: str | None,
) -> dict[str, str]:
    """Stamp ``RIVAULT_CALLER_ROOTS_JSON`` into a launch env for non-admin
    seams (HTTP provision and the durable adapter both call here with a bare
    keyword instead of an inline branch — the fat callers stay at baseline).

    Admins (``is_admin``) and system tiers (``tier == "system"``) skip the
    gate — their launch carries no caller root list and the worker's realpath
    gate then stays off (legacy path)."""
    if is_admin is True or tier == "system":
        return env_vars
    from ..orchestrator.caller_roots import caller_roots_json

    env_vars.setdefault(
        "RIVAULT_CALLER_ROOTS_JSON", caller_roots_json(project_id, user_id or "")
    )
    return env_vars


def enforce_endpoint_policy(
    spec: dict[str, Any], user_id: str, project_id: str
) -> None:
    """One endpoint_provision spec follows the policy, non-admin callers only
    — the ZMQ lane's entry point (``server/job_tracking_handler._submit``)."""
    _check_endpoint_spec(spec, user_id, project_id)


def enforce_job_endpoint_policy(
    job_type: JobType,
    spec: dict[str, Any],
    *,
    user_id: str,
    project_id: str,
    is_admin: bool,
    fan_out_jobs: Any = None,
) -> None:
    """Every endpoint_provision target — the parent and each fan-out child —
    follows the policy (``api.jobs._enforce_endpoint_host_paths`` and the
    ZMQ submit both delegate here)."""
    if is_admin:
        return
    for target_type, target_spec in _spec_targets(job_type, spec, fan_out_jobs):
        if target_type == JobType.ENDPOINT_PROVISION and isinstance(target_spec, dict):
            _check_endpoint_spec(target_spec, user_id, project_id)


__all__ = [
    "_check_endpoint_spec",
    "_check_endpoint_spec_model",
    "_check_endpoint_spec_recipe",
    "_deny_unless",
    "_spec_targets",
    "apply_caller_roots_env",
    "enforce_endpoint_policy",
    "enforce_job_endpoint_policy",
]
