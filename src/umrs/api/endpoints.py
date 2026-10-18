"""
Endpoint Registry REST API.

Provides:
- GET /umrs/endpoints/active - Query active endpoints for Gateway discovery
- POST /umrs/endpoints - Register an endpoint (internal use)
- DELETE /umrs/endpoints/{job_id} - Remove an endpoint (internal use)
- GET /umrs/endpoints/{job_id}/status - Get endpoint status
- PUT /umrs/endpoints/{job_id}/used - Mark endpoint as used (for TTL extension)
- GET /umrs/endpoints/stats - Registry statistics

Authentication:
- project_id is extracted from Authorization header (JWT claims)
- For MVP: Accept X-Project-ID header for testing
"""

import logging
import os
from typing import Any

import httpx
from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..model_registry.capabilities import endpoint_limit_metadata
from ..orchestrator.caller_roots import (
    caller_named_host_path,
    foreign_sources,
    model_path_in_scope,
)
from ..orchestrator.endpoint_access import usable_endpoints
from ..orchestrator.endpoint_mounts import (
    egress_reasons,
    require_egress_proxy_image,
    validate_mount_source,
)
from ..orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)
from ..orchestrator.readiness import get_readiness_poller
from ..shared.auth import JWTError, resolve_jwt_secret, validate_jwt
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)


def _strict_auth_mode() -> bool:
    """Return True when MVP header/default project fallback is disabled."""
    # Default strict (audit B12): mvp is an explicit dev opt-in, never the
    # silent default — an unconfigured process must not run wide open.
    return os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"


def _current_cluster_name() -> str:
    """Cluster identity for provenance (env, then site config name)."""
    env = os.environ.get("UMRS_CLUSTER", "").strip()
    if env:
        return env
    try:
        from ..orchestrator.site_config import load_site_config

        return str(load_site_config().name)
    except Exception:
        return "unknown"


def _validate_partition_allowed(request_partition: str | None) -> None:
    """Reject partitions outside this deployment's scheduler partition list.

    The partition list is site-owned (``site.scheduler.partitions``); a caller
    may run within any LISTED partition but never invent one (avoids silent
    off-site placement and late sbatch-side failures).
    """
    part = (request_partition or "").strip()
    if not part:
        return
    try:
        from ..orchestrator.site_config import load_site_config

        site = load_site_config()
        allowed = [
            p.strip() for p in (site.scheduler.partition or "").split(",") if p.strip()
        ]
    except Exception:
        allowed = []
    if allowed and part not in allowed:
        raise_typed(
            PlatformError.endpoints(
                "bad_request",
                (
                    f"partition '{part}' not in this site's scheduler partitions "
                    f"({', '.join(allowed)})"
                ),
            )
        )


def _enforce_serving_mode_supported(request: Any) -> None:
    """Reject serving modes the runtime can validate but never materialize.

    ``multi-lora`` and ``co-located`` are part of the API contract
    (shape-validated) but have ZERO runtime consumers: the serving pipeline
    never emits a LoRA env var and never starts a second vLLM instance, so
    accepting them reported a provisioning success while silently serving a
    plain single model (audit: dead-input, high severity). Fail fast with 501
    so agents and clients can distinguish "not yet implemented" from a 400
    user error.
    """
    serving_mode = getattr(request, "serving_mode", "single") or "single"
    if serving_mode != "single":
        raise_typed(
            PlatformError.endpoints(
                "not_implemented",
                (
                    f"serving_mode='{serving_mode}' is not yet implemented by the "
                    "serving runtime — the request is validated but no runtime "
                    "consumer materializes it. Provision with serving_mode='single'."
                ),
            )
        )


def _authorize_model_entry_for_caller(
    entry: dict[str, Any] | None,
    user_id: str | None,
    project_id: str | None,
) -> None:
    """Enforce model-catalog scope on provisioning.

    The resolver looks up model entries with an admin catalog context
    (mechanics resolution only); without this check, a caller who can *name*
    someone else's user-/project-scope model could provision it (audit E).
    Enforcement is complete-identity: with full identity available we deny
    strict-scope mismatches (403). Legacy mvp deployments where identity is
    absent keep previous behavior but at WARNING visibility, not silently.
    """
    if not entry:
        return
    scope = str(entry.get("scope") or "system")
    if _scope_matches(entry, scope, user_id, project_id):
        return
    if user_id is None and project_id is None:
        logger.warning(
            "Model scope check degraded (no identity): %s scope=%s; allowing (mvp mode)",
            entry.get("name"),
            scope,
        )
        return
    raise_typed(
        PlatformError.endpoints(
            "forbidden",
            (
                f"Model '{entry.get('name')}' is {scope}-scope and not provisionable "
                "by this caller"
            ),
        ),
    )


def _scope_matches(
    entry: dict[str, Any],
    scope: str,
    user_id: str | None,
    project_id: str | None,
) -> bool:
    """Exact-scope matching rule for a catalog entry vs caller identity."""
    if scope == "system":
        return True
    if scope == "project":
        return bool(project_id) and entry.get("project_id") == project_id
    if scope == "user":
        return bool(user_id) and entry.get("owner_id") == user_id
    return False


def _enforce_recipe_policy(is_admin: bool, request: Any) -> None:
    """Reject recipe/security-controlled request fields for non-admin callers.

    Custom-model users keep full request shape; only two categories are
    admin-only by architecture decision (no arbitrary image/env injection):
      • env_vars — arbitrary environment injection
      • container_image — arbitrary serving image selection
    Mounts/executor/account policies stay with their existing validators.
    """
    _validate_partition_allowed(getattr(request, "partition", None))
    if is_admin:
        return
    if getattr(request, "env_vars", None):
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                (
                    "env_vars require admin role (recipe-controlled channel); "
                    "use a curated recipe or ask an admin to add one"
                ),
            ),
        )
    if getattr(request, "container_image", None):
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                (
                    "container_image override requires admin role "
                    "(recipe-controlled channel)"
                ),
            ),
        )


def _raise_strict_project_auth_error() -> None:
    raise_typed(
        PlatformError.auth(
            "gateway_identity_required",
            (
                "UMRS_AUTH_MODE=strict requires a verified project claim; "
                "X-Project-ID and default-project fallbacks are disabled."
            ),
        ),
    )


def _header_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _bearer_token(authorization: str | None) -> str | None:
    value = _header_str(authorization)
    if not value:
        return None
    parts = value.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1]


def _jwt_claims(
    authorization: str | None,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    token = _bearer_token(authorization)
    if not token:
        if required:
            raise_typed(
                PlatformError.auth(
                    "gateway_identity_required", "Missing Authorization bearer token"
                ),
            )
        return None
    if token.count(".") != 2:
        if required:
            raise_typed(
                PlatformError.auth(
                    "gateway_identity_required",
                    "Authorization bearer token is not a JWT",
                ),
            )
        return None
    try:
        return validate_jwt(token, resolve_jwt_secret())
    except JWTError as exc:
        raise_typed(
            PlatformError.auth("gateway_identity_required", str(exc)),
            cause=exc,
        )


def _claim_values(claims: dict[str, Any], *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        value = claims.get(name)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
        elif isinstance(value, list):
            values.extend(
                item.strip() for item in value if isinstance(item, str) and item.strip()
            )
    return values


def _claim_project_id(claims: dict[str, Any]) -> str | None:
    values = _claim_values(claims, "project_id", "project", "tenant_id", "tenant")
    return values[0] if values else None


def _claim_project_memberships(claims: dict[str, Any]) -> set[str]:
    memberships = set(
        _claim_values(
            claims,
            "project_memberships",
            "project_ids",
            "projects",
            "allowed_projects",
        )
    )
    project_id = _claim_project_id(claims)
    if project_id:
        memberships.add(project_id)
    return memberships


def _claim_is_admin(claims: dict[str, Any] | None) -> bool:
    if not claims:
        return False
    return "admin" in set(_claim_values(claims, "roles", "groups"))


def _claim_can_access_project(claims: dict[str, Any], project_id: str) -> bool:
    if _claim_is_admin(claims):
        return True
    memberships = _claim_project_memberships(claims)
    return "*" in memberships or project_id in memberships


def _forbid_if_project_mismatch(
    endpoint: RegisteredEndpoint,
    project_id: str | None,
) -> None:
    if project_id is not None and endpoint.project_id != project_id:
        raise_typed(
            PlatformError.endpoints("not_found", "Endpoint not found"),
        )


def _require_identity_for_mutation(project_id: str | None, action: str) -> None:
    """Deny-by-default for mutating lifecycle calls in strict mode.

    In strict mode, the optional project resolver returns ``None`` for an
    unauthenticated/headerless caller, and ``_forbid_if_project_mismatch``
    silently no-ops on ``None`` — the right semantic for read-only lookups
    but catastrophically wrong for mutations (audited: unauthenticated
    cross-project deprovision / TTL refresh). A mutating caller must either
    present an identity or receive 401. Non-strict (mvp) deployments keep
    their existing permissive behavior.
    """
    if project_id is None and _strict_auth_mode():
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required",
                (f"{action} requires an authenticated project identity in strict mode"),
            ),
        )


def _resolve_authenticated_project_id(
    authorization: str | None,
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_project_id: str | None,
    *,
    optional: bool = False,
    x_verified_is_admin: str | None = None,
    x_verified_hmac: str | None = None,
) -> str | None:
    strict = _strict_auth_mode()
    claims = _jwt_claims(authorization, required=False)
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    requested_project_id = verified_project_id or _header_str(x_project_id)

    if strict and verified_project_id:
        from ._verified_headers import handshake_secret, verified_headers_trusted

        # Live-PoC'd bypass class: plain-text verified headers carried no
        # gateway-origin proof. Honor them only when the sign/verify secret
        # validates over the FULL identity tuple including the admin flag.
        trusted = verified_headers_trusted(
            strict=strict,
            secret=handshake_secret(),
            presented_hmac=_header_str(x_verified_hmac),
            user_id=verified_user_id,
            project_id=verified_project_id,
            is_admin=_header_bool(x_verified_is_admin),
        )
        if verified_user_id and trusted:
            return verified_project_id
        # Verified headers without a matching signature are a spoofed claim.
        _raise_strict_project_auth_error()

    if claims:
        return _project_id_authorized_by_claims(claims, requested_project_id)

    if strict:
        if optional:
            return None
        _raise_strict_project_auth_error()

    return _legacy_project_id(authorization, x_project_id, optional=optional)


def _project_id_authorized_by_claims(
    claims: dict[str, Any],
    requested_project_id: str | None,
) -> str:
    effective_project_id = requested_project_id or _claim_project_id(claims)
    if not effective_project_id:
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required", "JWT is missing project_id claim"
            ),
        )
    if not _claim_can_access_project(claims, effective_project_id):
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                "Requested project is not allowed by JWT project memberships",
            ),
        )
    return effective_project_id


def _legacy_project_id(
    authorization: str | None,
    x_project_id: str | None,
    *,
    optional: bool,
) -> str | None:
    """MVP-only fallback (deferred R1c): raw header and silent default-project.

    Deprecated: strict mode never reaches this resolver — callers land here
    only in mvp. Emit one deprecation warning per process; removal is
    scheduled for the next release.
    """
    from ._identity import warn_legacy_params_once

    project_id = _header_str(x_project_id)
    if project_id:
        warn_legacy_params_once("endpoints.legacy_x_project_id")
        return project_id
    if authorization and authorization.startswith("Bearer "):
        warn_legacy_params_once("endpoints.legacy_bearer_default_project")
        if optional:
            return "default-project"
        logger.warning("JWT project_id extraction not implemented, using default")
        return "default-project"
    return None


def _header_bool(value: str | None) -> bool:
    normalized = _header_str(value)
    return normalized is not None and normalized.lower() in {"1", "true", "yes", "on"}


def _caller_is_admin(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> bool:
    claims = _jwt_claims(authorization, required=False)
    if claims is not None:
        return _claim_is_admin(claims)
    header_admin = _header_bool(x_verified_is_admin)
    if not header_admin:
        # Non-admin callers only need base identity, which callers resolve
        # via _resolve_authenticated_project_id.
        return False
    from ._verified_headers import handshake_secret, verify_admin_flag

    return verify_admin_flag(
        strict=_strict_auth_mode(),
        secret=handshake_secret(),
        presented_hmac=_header_str(x_verified_hmac),
        user_id=_header_str(x_user_id),
        project_id=_header_str(x_verified_project_id),
        presented_is_admin=True,
    )


def _require_internal_endpoint_writer(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> None:
    # The signed-admin-header path needs the project and HMAC too; they were
    # not passed, so only a JWT admin could pass (H-1, 2026-10-17).
    if not _strict_auth_mode():
        return
    if not _caller_is_admin(
        authorization,
        x_user_id,
        x_verified_is_admin,
        x_verified_project_id,
        x_verified_hmac,
    ):
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                "Endpoint registry mutation requires admin JWT in strict mode",
            ),
        )


def _enforce_provision_authorization(
    request: "ProvisionRequest",
    *,
    is_admin: bool,
) -> None:
    if request.tier == "system" and not is_admin:
        raise_typed(
            PlatformError.endpoints(
                "forbidden", "Only admins may provision system-tier endpoints"
            ),
        )


# H-1 (2026-10-18): mount sources were checked only against the site-wide
# prefixes (UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES: /home, /projects, ...), so a
# caller could mount another user's home or another project's data. A
# non-admin's sources must lie under the caller's own roots
# (orchestrator/caller_roots.py, shared with POST /umrs/jobs endpoint_provision
# and the upgrade route since 2026-10-18).


def _mount_sources(mounts: "EndpointMounts | None") -> list[str]:
    if mounts is None:
        return []
    specs = [*mounts.mcp_servers, mounts.project_data, mounts.rag_index]
    return [spec.source for spec in specs if spec is not None]


def _enforce_mount_scope(
    request: "ProvisionRequest",
    *,
    is_admin: bool,
    project_id: str,
    user_id: str | None,
) -> None:
    """403 unless every mount source lies under the caller's own roots."""
    sources = _mount_sources(request.mounts)
    if is_admin or not sources:
        return
    foreign = foreign_sources(sources, project_id, user_id or "")
    if foreign:
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                (
                    f"mount source {foreign[0]} is outside your own project and user "
                    "data roots (UMRS_ENDPOINT_CALLER_MOUNT_ROOTS)"
                ),
            ),
        )


# H-1 (2026-10-18, review): the model weights are a second host path bound
# into the endpoint container. A model_path override, or an absolute model
# name with no catalog entry, was checked only against the site-wide
# prefixes, so a non-admin could serve another user's or project's weights.
# A host path the caller names itself must lie under the caller's own roots;
# the authorised catalog entry's storage_ref and remote URIs (hf://, s3://)
# are not caller-named host paths. Residual: the check resolves links on
# this host at request time only; the compute-node launcher must re-check.
def _enforce_model_path_scope(
    resolved: str,
    model_entry: dict[str, Any] | None,
    *,
    is_admin: bool,
    project_id: str,
    user_id: str | None,
) -> None:
    """403 unless a caller-named host model path lies under the caller's roots."""
    if is_admin or model_path_in_scope(
        resolved, model_entry, project_id=project_id, user_id=user_id or ""
    ):
        return
    raise_typed(
        PlatformError.endpoints(
            "forbidden",
            (
                f"model path {resolved} is outside your own project and user data "
                "roots (UMRS_ENDPOINT_CALLER_MOUNT_ROOTS); register the model in "
                "the catalog or use a path under your own roots"
            ),
        ),
    )


def _derive_reasoning_parser(model_name: str) -> str:
    """Mirror serve.slurm's derive_reasoning_parser() for forced reasoning.

    Used only when the caller requests ``reasoning="on"`` so the parser is
    set explicitly even if a site disables the shell-side auto-derivation.
    Returns an empty string when the model family has no reasoning parser.
    """
    name_lc = (model_name or "").lower()
    if "glm" in name_lc:
        return "glm45"
    if "gemma" in name_lc and "4" in name_lc:
        return "gemma4"
    if "kimi" in name_lc:
        return "kimi_k2"
    return ""


router = APIRouter(prefix="/umrs/endpoints", tags=["endpoints"])

_DEFAULT_PROVISION_MODEL_PATH = "/models/rivault-gpt-oss"


# Backend job statuses considered "alive enough" to reuse a warm endpoint
# (registry says READY but the underlying job must also be in a non-terminal
# state).  Imported lazily inside the helper to avoid an import cycle.
async def _is_backend_job_alive(service: Any, job_id: str) -> bool:
    """Return True if the backend reports the job as alive.

    Used to guard against stale warm-endpoint reuse: an endpoint may be
    cached as READY in the registry while its Slurm/K8s job has already
    terminated.  Returns False on any verification error so the caller
    falls through to a fresh provisioning path (fail-safe default).
    """
    from ..orchestrator.base import JobStatus

    backend = getattr(service, "backend", None)
    if backend is None:
        return False
    try:
        info = await backend.get_job_status(job_id)
    except Exception as exc:  # noqa: BLE001 - defensive, treat as dead
        logger.warning(
            "Backend get_job_status(%s) failed during warm-endpoint check: %s",
            job_id,
            exc,
        )
        return False
    status_value = getattr(info, "status", None)
    return status_value in (
        JobStatus.PENDING,
        JobStatus.STARTING,
        JobStatus.READY,
        JobStatus.RUNNING,
    )


async def _backend_job_liveness(service: Any, job_id: str) -> bool | None:
    """Return backend liveness as alive / dead / unknown.

    ``True`` means the backend positively reports a non-terminal job.
    ``False`` means the backend positively reports a terminal job
    (``STOPPED`` or ``FAILED``).
    ``None`` means we could not verify backend state (including
    ``JobStatus.UNKNOWN``) and should preserve the registry row rather than
    deleting a potentially valid endpoint.
    """
    from ..orchestrator.base import JobStatus

    backend = getattr(service, "backend", None)
    if backend is None:
        return None

    try:
        info = await backend.get_job_status(job_id)
    except Exception as exc:  # noqa: BLE001 - preserve on verification errors
        logger.warning(
            "Backend get_job_status(%s) failed during active-endpoint verification: %s",
            job_id,
            exc,
        )
        return None

    status_value = getattr(info, "status", None)
    if status_value in (
        JobStatus.PENDING,
        JobStatus.STARTING,
        JobStatus.READY,
        JobStatus.RUNNING,
    ):
        return True
    # Only treat *positively* terminal states as dead. A ``JobStatus.UNKNOWN``
    # (info file momentarily absent, squeue miss under SSH load during a
    # concurrent provision, or an unrecognized transient Slurm state) must NOT
    # delete a healthy endpoint — return ``None`` so the caller preserves the
    # registry row rather than removing a live sibling by mistake.
    if status_value in (JobStatus.FAILED, JobStatus.STOPPED):
        return False
    return None


async def _filter_live_ready_endpoints(
    registry: EndpointRegistry,
    endpoints: list[Any],
) -> list[Any]:
    """Return only endpoints whose backing jobs are still alive.

    ``/umrs/endpoints/active`` is a discovery surface, so returning a stale
    READY record after a UMRS restart is worse than returning nothing. When we
    can initialize the orchestration service, verify every READY/DEGRADED row
    against the backend and remove ghosts from the registry backend.
    """
    if not endpoints:
        return endpoints

    from ..orchestrator.service import get_orchestration_service

    try:
        service = get_orchestration_service()
    except Exception as exc:  # noqa: BLE001 - keep discovery available
        logger.warning(
            "Could not initialize orchestration service during active endpoint verification: %s",
            exc,
        )
        return endpoints

    live_endpoints: list[Any] = []
    for endpoint in endpoints:
        is_alive = await _backend_job_liveness(service, endpoint.job_id)
        if is_alive is None:
            live_endpoints.append(endpoint)
            continue
        if is_alive:
            live_endpoints.append(endpoint)
            continue

        logger.warning(
            "Removing stale active endpoint %s/%s (job_id=%s): backend job no longer alive",
            endpoint.project_id,
            endpoint.model,
            endpoint.job_id,
        )
        registry.remove(endpoint.job_id)

    return live_endpoints


_REFRESHABLE_ENDPOINT_STATUSES = {
    EndpointStatus.PENDING,
    EndpointStatus.PROVISIONING,
    EndpointStatus.STARTING,
}


_READINESS_DETAIL_RULES = (
    (
        ("connection error", "connection refused"),
        "starting LiteLLM / waiting for service port",
    ),
    (
        ("chat probe timeout", "request timeout"),
        "loading model weights / waiting for inference warmup",
    ),
    (
        ("http 404", "http 500", "http 502", "http 503"),
        "starting LiteLLM / waiting for proxy readiness",
    ),
    (
        ("no models in response", "expected model"),
        "registering served model / loading model",
    ),
    (
        ("chat probe http", "chat probe error"),
        "endpoint reachable; waiting for inference readiness",
    ),
)


def _short_readiness_error(error: str | None, *, limit: int = 96) -> str | None:
    if not error:
        return None
    cleaned = " ".join(error.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _readiness_status_detail(endpoint: Any, error: str | None) -> str:
    target = f"{endpoint.host}:{endpoint.port}"
    if not error:
        return f"waiting for endpoint readiness ({target})"

    lowered = error.lower()
    for needles, detail in _READINESS_DETAIL_RULES:
        if any(needle in lowered for needle in needles):
            return detail

    return f"waiting for endpoint readiness ({target}): {_short_readiness_error(error, limit=72)}"


def _endpoint_can_refresh_readiness(endpoint: Any) -> bool:
    return (
        endpoint.status in _REFRESHABLE_ENDPOINT_STATUSES
        and bool(endpoint.host)
        and endpoint.host != "pending"
    )


async def _refresh_endpoint_readiness(
    registry: EndpointRegistry,
    endpoint: Any,
) -> Any:
    if not _endpoint_can_refresh_readiness(endpoint):
        return endpoint

    readiness_probe = str(
        endpoint.metadata.get("readiness_probe")
        or _resolve_readiness_probe_mode(endpoint.model)
    )
    readiness_path = endpoint.metadata.get("readiness_path")

    try:
        (
            is_ready,
            is_degraded,
            models,
            error,
        ) = await get_readiness_poller().check_ready_detail(
            endpoint.host,
            endpoint.port,
            probe_model=endpoint.metadata.get("served_model_name") or endpoint.model,
            probe_kind=readiness_probe,
            probe_path=str(readiness_path) if readiness_path else None,
        )
    except Exception as exc:  # noqa: BLE001 - status lookup must stay best-effort
        logger.warning(
            "Endpoint readiness refresh failed for %s: %s", endpoint.job_id, exc
        )
        return endpoint

    return _apply_readiness_state(
        registry, endpoint, is_ready, is_degraded, models, error
    )


def _apply_readiness_state(
    registry: EndpointRegistry,
    endpoint: Any,
    is_ready: bool,
    is_degraded: bool,
    models: Any,
    error: Any,
) -> Any:
    """Record probe outcome in registry metadata; return refreshed endpoint."""
    if is_ready:
        registry.update_metadata(
            endpoint.job_id,
            {
                "detail": "endpoint ready",
                "readiness_error": None,
                "readiness_models": models,
            },
        )
        return registry.mark_ready(endpoint.job_id) or endpoint

    if is_degraded:
        registry.update_metadata(
            endpoint.job_id,
            {
                "detail": f"degraded: {_readiness_status_detail(endpoint, error)}",
                "readiness_error": _short_readiness_error(error),
                "readiness_models": models,
            },
        )
        return registry.mark_degraded(endpoint.job_id) or endpoint

    registry.update_metadata(
        endpoint.job_id,
        {
            "detail": _readiness_status_detail(endpoint, error),
            "readiness_error": _short_readiness_error(error),
        },
    )
    return registry.get_by_job_id(endpoint.job_id) or endpoint


# =============================================================================
# Request/Response Models
# =============================================================================


class EndpointResponse(BaseModel):
    """Endpoint data for API responses."""

    project_id: str
    model: str
    job_id: str
    host: str
    port: int
    status: str
    created_at: str
    ready_at: str | None = None
    last_used_at: str | None = None
    expires_at: str | None = None
    backend: str
    error_message: str | None = None
    litellm_url: str
    owner_type: str | None = None
    owner_id: str | None = None
    access: list[str] | None = None  # ADR-078 scopes beyond the owning project
    tier: str | None = None
    served_model_name: str | None = None
    model_type: str | None = None
    serving_backend: str | None = None
    service_kind: str | None = None
    service_url: str | None = None
    invoke_base_path: str | None = None
    readiness_probe: str | None = None
    readiness_path: str | None = None
    launch_profiles: list[str] | None = None
    container_image: str | None = None
    vision_adapter: str | None = None
    vision_task: str | None = None
    output_format: str | None = None
    detail: str | None = None
    job_status: str | None = None
    max_model_len: int | None = None
    variant: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "proj-123",
                "model": "rivault-gpt-oss-static",
                "job_id": "slurm-456",
                "host": "compute-node-01",
                "port": 11434,
                "status": "ready",
                "created_at": "2024-01-15T10:30:00Z",
                "ready_at": "2024-01-15T10:31:00Z",
                "last_used_at": "2024-01-15T10:35:00Z",
                "expires_at": "2024-01-15T11:30:00Z",
                "backend": "slurm",
                "error_message": None,
                "litellm_url": "http://compute-node-01:11434/v1",
                "owner_type": "user",
                "owner_id": None,
                "tier": "user",
                "detail": "starting container / loading model",
                "job_status": "starting",
            }
        }
    )


class EndpointsListResponse(BaseModel):
    """Response for endpoint listing."""

    endpoints: list[EndpointResponse]
    total: int


class EndpointRegisterRequest(BaseModel):
    """Request to register a new endpoint."""

    project_id: str = Field(..., description="Project/tenant identifier")
    model: str = Field(..., description="Model name (e.g., rivault-gpt-oss-static)")
    job_id: str = Field(..., description="Backend job identifier")
    host: str = Field(..., description="Hostname or IP address")
    port: int = Field(..., ge=1, le=65535, description="LiteLLM port")
    backend: str = Field(
        "unknown", description="Backend type (slurm/kubernetes/compose)"
    )
    ttl_sec: int | None = Field(
        None,
        ge=0,
        description="TTL in seconds (0 = no expiry; None = server default)",
    )
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class EndpointStatusUpdate(BaseModel):
    """Request to update endpoint status."""

    status: str = Field(..., description="New status")
    error_message: str | None = Field(None, description="Error message if failed")


class LoraSpec(BaseModel):
    """LoRA adapter specification for multi-lora endpoints."""

    name: str = Field(
        ...,
        min_length=1,
        description="Adapter name (served as `model` over OpenAI API)",
    )
    source: str = Field(
        ..., min_length=1, description="Filesystem path or registered LoRA model name"
    )


class EndpointMountSpec(BaseModel):
    """Read-only host mount requested for a provisioned endpoint."""

    source: str = Field(
        ..., min_length=1, description="Host/shared filesystem source path"
    )
    target: str = Field(..., min_length=1, description="Container target path")

    @model_validator(mode="after")
    def _validate_paths(self) -> "EndpointMountSpec":
        try:
            self.source = validate_mount_source(self.source)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not self.target.startswith("/"):
            raise ValueError("mount target must be an absolute container path")
        return self


class EndpointMounts(BaseModel):
    """Mount bundle for mini-RiVault MCP and RAG side services."""

    mcp_servers: list[EndpointMountSpec] = Field(default_factory=list)
    project_data: EndpointMountSpec | None = None
    rag_index: EndpointMountSpec | None = None

    @model_validator(mode="after")
    def _validate_mount_targets(self) -> "EndpointMounts":
        for mount in self.mcp_servers:
            if not _target_under(mount.target, "/mcp-servers"):
                raise ValueError("mcp_servers targets must be under /mcp-servers")
        if self.project_data and not _target_under(
            self.project_data.target, "/project"
        ):
            raise ValueError("project_data target must be under /project")
        if self.rag_index and not _target_under(self.rag_index.target, "/rag-index"):
            raise ValueError("rag_index target must be under /rag-index")
        return self


def _target_under(target: str, root: str) -> bool:
    normalized = target.rstrip("/") or "/"
    return normalized == root or normalized.startswith(root + "/")


class ProvisionRequest(BaseModel):
    """Request to provision a new endpoint (U-202)."""

    project_id: str | None = Field(
        None,
        description=(
            "Project/tenant identifier. Optional when the Gateway provides "
            "X-Project-ID or X-Verified-Project-ID."
        ),
    )
    model: str = Field(..., description="Model name (primary; equals models[0])")
    models: list[str] | None = Field(
        None,
        description=(
            "Optional list of base models for multi-model endpoints. "
            "Defaults to ``[model]``. The first entry must equal ``model``."
        ),
    )
    serving_mode: str = Field(
        "single",
        description="One of: single, multi-lora, co-located",
    )
    serving_backend: str | None = Field(
        None,
        description="Serving backend override: vllm or bentoml",
    )
    service_kind: str | None = Field(
        None,
        description="Service API kind override: openai or bentoml_vision",
    )
    launch_profiles: list[str] | None = Field(
        None,
        description="Mini-RiVault profiles to activate for this endpoint",
    )
    co_locate_vllm: bool = Field(
        False,
        description="For BentoML vision, also launch the vLLM profile",
    )
    container_image: str | None = Field(None, description="Container image override")
    service_port: int | None = Field(
        None, ge=1, le=65535, description="External service port override"
    )
    readiness_probe: str | None = Field(
        None, description="Readiness probe override: models, chat, or http"
    )
    readiness_path: str | None = Field(
        None, description="HTTP readiness path for non-OpenAI services"
    )
    invoke_base_path: str | None = Field(
        None, description="Base path for invoking the service"
    )
    vision_adapter: str | None = Field(None, description="Vision adapter name")
    vision_task: str | None = Field(None, description="Default vision task")
    output_format: str | None = Field(None, description="Default vision result format")
    loras: list[LoraSpec] | None = Field(
        None,
        description="LoRA adapters (multi-lora mode only)",
    )
    tier: str = Field("user", description="Deployment tier: system, project, or user")
    mounts: EndpointMounts | None = Field(
        None,
        description="Read-only MCP/RAG mounts for user/project mini-RiVault instances",
    )
    egress_domains: list[str] = Field(
        default_factory=list,
        description="Domain allowlist for EgressGuard-enabled side services",
    )
    model_path: str | None = Field(
        None,
        description="Optional path to model files. If omitted, resolve from the registered model catalog.",
    )
    ttl_sec: int = Field(
        3600,
        ge=0,
        description="TTL in seconds (0 = no expiry / never auto-recycle)",
    )
    gpu_count: int = Field(1, ge=1, description="Number of GPUs per node")
    node_count: int = Field(
        1,
        ge=1,
        description="Number of Slurm nodes (>1 for multi-node pipeline parallelism)",
    )
    max_model_len: int = Field(4096, ge=512, description="Max context length")
    prefer_warm: bool = Field(True, description="Prefer reusing existing endpoint")
    access: list[str] | None = Field(
        None,
        description=(
            "ADR-078 access scopes beyond the owning project: user:<id>, "
            "project:<id>, circle:<proj>/<id> (members only) or system (admin only). "
            "Omitted = the roster recipe's access when the caller may grant "
            "it, else the owning project only."
        ),
    )
    variant: str | None = Field(
        None,
        description=(
            "Deployment variant name (e.g. '128k', 'tep-128k'). "
            "Differentiates multiple endpoints for the same model with "
            "different parameters (ctx length, node count, etc.). "
            "When set, warm-reuse only considers endpoints with the same variant."
        ),
    )
    wait: bool = Field(True, description="Wait for readiness before responding")
    partition: str | None = Field(
        None, description="Override Slurm partition (must be in site candidates)"
    )
    nodelist: str | None = Field(
        None, description="Slurm --nodelist hint (e.g. 'ai-h200-brc-00')"
    )
    parallel_strategy: str | None = Field(
        None,
        description=(
            "Multi-node parallelism layout: 'tp_pp' (default; TP=GPUs/node, "
            "PP=node count) or 'tep' (tensor+expert parallel: TP=world size, "
            "PP=1, --enable-expert-parallel). Forwarded to serve.slurm via "
            "VLLM_PARALLEL_STRATEGY."
        ),
    )
    reasoning: str | None = Field(
        None,
        description=(
            "Reasoning-parser control for thinking models: 'auto' (default; "
            "server derives the parser from the model name), 'on' (force the "
            "model's native reasoning parser), or 'off' (disable reasoning "
            "splitting). Forwarded to serve.slurm via VLLM_REASONING_PARSER."
        ),
    )
    env_vars: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra environment variables to pass to the Slurm job / mini-RiVault "
            "container. These are merged with the ServingProfile-derived defaults, "
            "with request env_vars taking precedence via setdefault semantics."
        ),
    )

    @model_validator(mode="after")
    def _validate_multi_model(self) -> "ProvisionRequest":
        self._validate_models_list()
        self._validate_serving_enums()
        self._validate_serving_mode_combinations()
        return self

    def _validate_models_list(self) -> None:
        # Normalize/validate models[]
        if self.models is None:
            self.models = [self.model]
        if not self.models:
            raise ValueError("models must be a non-empty list")
        if any(not m for m in self.models):
            raise ValueError("models entries must be non-empty strings")
        if self.models[0] != self.model:
            raise ValueError("models[0] must equal model")

    def _validate_serving_enums(self) -> None:
        self._validate_backend_enums()
        self._validate_caller_enums()

    def _validate_backend_enums(self) -> None:
        if self.serving_mode not in ("single", "multi-lora", "co-located"):
            raise ValueError(
                "serving_mode must be one of: single, multi-lora, co-located"
            )
        if self.serving_backend and self.serving_backend not in ("vllm", "bentoml"):
            raise ValueError("serving_backend must be one of: vllm, bentoml")
        if self.service_kind and self.service_kind not in ("openai", "bentoml_vision"):
            raise ValueError("service_kind must be one of: openai, bentoml_vision")
        if self.readiness_probe and self.readiness_probe not in (
            "models",
            "chat",
            "http",
            "bentoml",
        ):
            raise ValueError(
                "readiness_probe must be one of: models, chat, http, bentoml"
            )

    def _validate_caller_enums(self) -> None:
        if self.tier not in ("system", "project", "user"):
            raise ValueError("tier must be one of: system, project, user")
        if self.reasoning is not None and self.reasoning not in ("auto", "on", "off"):
            raise ValueError("reasoning must be one of: auto, on, off")

    def _validate_serving_mode_combinations(self) -> None:
        self._validate_model_count_for_mode()
        self._validate_lora_combination()

    def _validate_model_count_for_mode(self) -> None:
        if self.serving_mode == "single" and len(self.models or []) != 1:
            raise ValueError("serving_mode=single requires exactly one model")
        if self.serving_mode == "co-located" and len(self.models or []) < 2:
            raise ValueError("serving_mode=co-located requires at least 2 models")

    def _validate_lora_combination(self) -> None:
        if self.loras:
            if self.serving_mode != "multi-lora":
                raise ValueError("loras requires serving_mode=multi-lora")
            if len(self.models or []) != 1:
                raise ValueError("multi-lora requires exactly one base model")
            names = [lora.name for lora in self.loras]
            if len(set(names)) != len(names):
                raise ValueError("LoRA names must be unique")
        elif self.serving_mode == "multi-lora":
            raise ValueError("serving_mode=multi-lora requires a non-empty loras list")


class ProvisionResponse(BaseModel):
    """Response for provision request."""

    status: str = Field(..., description="Status: ready, provisioning, failed")
    job_id: str | None = Field(None, description="Job ID if provisioning started")
    endpoint: EndpointResponse | None = Field(None, description="Endpoint if ready")
    retry_after: int | None = Field(None, description="Seconds to wait before retry")
    error: str | None = Field(None, description="Error message if failed")


class RegistryStatsResponse(BaseModel):
    """Registry statistics."""

    total: int
    by_status: dict[str, int]
    by_backend: dict[str, int]
    ready_count: int


class EndpointHealthInfo(BaseModel):
    """Health information for a single endpoint."""

    job_id: str
    project_id: str
    model: str
    status: str
    host: str
    port: int
    healthy: bool
    last_health_check: str | None = None
    uptime_seconds: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Comprehensive health response (U-204)."""

    status: str = Field(..., description="Overall status: healthy, degraded, unhealthy")
    registry_ok: bool
    total_endpoints: int
    healthy_endpoints: int
    unhealthy_endpoints: int
    provisioning_endpoints: int
    endpoints: list[EndpointHealthInfo] = Field(default_factory=list)
    message: str | None = None


# =============================================================================
# Dependencies
# =============================================================================


def get_project_id(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """
    Extract project_id from request headers.

    Production: Extract from JWT claims in Authorization header.
    MVP/Testing: Accept X-Project-ID header.

    Raises:
        HTTPException: If project_id cannot be determined.
    """
    project_id = _resolve_authenticated_project_id(
        authorization,
        x_user_id,
        x_verified_project_id,
        x_project_id,
        x_verified_is_admin=x_verified_is_admin,
        x_verified_hmac=x_verified_hmac,
    )
    if project_id:
        return project_id

    raise_typed(
        PlatformError.auth(
            "gateway_identity_required",
            "Missing project_id. Provide X-Project-ID header or valid JWT.",
        ),
    )


def get_project_id_optional(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str | None:
    """Get project_id if available, None otherwise (for admin endpoints)."""
    return _resolve_authenticated_project_id(
        authorization,
        x_user_id,
        x_verified_project_id,
        x_project_id,
        optional=True,
        x_verified_is_admin=x_verified_is_admin,
        x_verified_hmac=x_verified_hmac,
    )


def _get_project_id_for_internal_scraper(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_project_id: str | None = Header(None),
) -> str | None:
    """Return project_id if the caller provides one, otherwise None.

    Used by internal observability scrapers that reach UMRS on the internal
    network without JWTs.  Unlike ``get_project_id_optional`` this never
    raises in strict mode; it simply treats an unauthenticated caller as a
    cross-project internal scraper.
    """
    try:
        return _resolve_authenticated_project_id(
            authorization,
            x_user_id,
            x_verified_project_id,
            x_project_id,
            optional=True,
        )
    except HTTPException:
        return None


def _get_requesting_user(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
) -> str | None:
    """Extract the requesting username from JWT ``sub`` or verified header.

    Returns the username string, or None if the caller is unauthenticated
    (non-strict mode).
    """
    claims = _jwt_claims(authorization, required=False)
    if claims is not None:
        return claims.get("sub") or claims.get("user_id")
    return _header_str(x_user_id)


def _resolve_provision_project_id(
    request: ProvisionRequest,
    authenticated_project_id: str | None,
) -> str:
    """Resolve the project that owns an endpoint provision request."""
    if authenticated_project_id is not None:
        if request.project_id and request.project_id != authenticated_project_id:
            logger.warning(
                "Ignoring endpoint provision body project_id=%s in favor of authenticated project_id=%s",
                request.project_id,
                authenticated_project_id,
            )
        return authenticated_project_id

    if _strict_auth_mode():
        _raise_strict_project_auth_error()

    body_project_id = _header_str(request.project_id)
    if body_project_id:
        return body_project_id

    raise_typed(
        PlatformError.auth(
            "gateway_identity_required",
            "Missing project_id. Provide X-Project-ID header or project_id in the request body.",
        ),
    )


def _resolve_served_model_name(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
) -> str | None:
    """Resolve served_model_name from model catalog metadata."""
    entry = model_entry or _resolve_registered_model_entry(model_name)
    if not entry:
        return model_name

    serving_profile = entry.get("serving_profile") or {}
    served_model_name = serving_profile.get("served_model_name")
    if served_model_name:
        return str(served_model_name)

    # Default to the catalog alias. Hugging Face IDs identify the source model,
    # not necessarily the runtime API name exposed by vLLM.
    return model_name


_SCOPE_ORDER = {"system": 0, "project": 1, "user": 2}


def _pick_first_by_scope(matches: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not matches:
        return None
    matches.sort(
        key=lambda entry: _SCOPE_ORDER.get(str(entry.get("scope", "system")), 99)
    )
    return matches[0]


def _entry_from_store(model_name: str) -> dict[str, Any] | None:
    from ..model_registry.store import AccessContext
    from .models import get_model_store

    store = get_model_store()
    matches = [
        entry.to_dict()
        for entry in store.list_models(
            AccessContext(user_id="", is_admin=True),
            limit=500,
        )
        if entry.name == model_name
    ]
    return _pick_first_by_scope(matches)


def _entry_from_json_catalog(model_name: str) -> dict[str, Any] | None:
    try:
        from umrs.server.model_registry_handler import _load_catalog

        matches = [
            entry
            for entry in _load_catalog().get("models", [])
            if entry.get("name") == model_name
        ]
        return _pick_first_by_scope(matches)
    except Exception as e:
        logger.debug(f"Failed to resolve model entry for {model_name}: {e}")
    return None


def _resolve_registered_model_entry(model_name: str) -> dict[str, Any] | None:
    """Resolve a registered model entry from the authoritative UMRS catalog."""
    try:
        return _entry_from_store(model_name)
    except ImportError:
        # Legacy file-only deployments: no store package at all. The JSON
        # fallback below is the intended substitute there.
        pass
    except Exception as e:
        # The authoritative store EXISTS but failed. Never mask an
        # authoritative-store error: log loud and return None (honest fail).
        logger.warning(
            "Model store lookup failed for %s (authoritative store error, NOT falling back to JSON catalog): %s",
            model_name,
            e,
            exc_info=True,
        )
        return None
    return _entry_from_json_catalog(model_name)


def _resolve_requested_model_path(
    model_name: str,
    model_path: str | None,
    model_entry: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the actual model storage path for endpoint provisioning."""
    normalized_path = model_path.strip() if model_path else None
    if normalized_path and normalized_path != _DEFAULT_PROVISION_MODEL_PATH:
        return _validated_model_path_override(normalized_path)

    entry = model_entry or _resolve_registered_model_entry(model_name)
    if entry and entry.get("storage_ref"):
        return entry["storage_ref"]

    if model_name.startswith("/"):
        return _validated_model_path_override(model_name)

    return None


def _validated_model_path_override(model_path: str) -> str:
    if "://" in model_path and not model_path.startswith("file://"):
        return model_path
    try:
        return validate_mount_source(model_path)
    except ValueError as exc:
        raise_typed(
            PlatformError.endpoints("bad_request", f"Invalid model_path: {exc}"),
            cause=exc,
        )


def _serving_profile(model_entry: dict[str, Any] | None) -> dict[str, Any]:
    profile = (model_entry or {}).get("serving_profile") or {}
    return profile if isinstance(profile, dict) else {}


def _resolve_model_type(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
) -> str:
    entry = model_entry or _resolve_registered_model_entry(model_name)
    return str((entry or {}).get("model_type") or "llm").lower()


def _resolve_serving_backend(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_backend = _serving_profile(entry).get("serving_backend")
    if profile_backend:
        return str(profile_backend).lower()
    if _resolve_model_type(model_name, entry) == "vision":
        return "bentoml"
    return "vllm"


def _resolve_service_kind(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_kind = _serving_profile(entry).get("service_kind")
    if profile_kind:
        return str(profile_kind)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    if backend == "bentoml" or _resolve_model_type(model_name, entry) == "vision":
        return "bentoml_vision"
    return "openai"


def _profile_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.replace(",", " ").split() if part]
    if isinstance(value, (list, tuple)):
        return [str(part) for part in value if str(part)]
    return []


def _dedupe_profiles(existing: str, new_profiles: list[str]) -> list[str]:
    """Merge *existing* extra profiles with *new_profiles*, de-duplicating."""
    result: list[str] = []
    for p in _profile_list(existing) + new_profiles:
        if p and p not in result:
            result.append(p)
    return result


def _require_egress_proxy_for_request(request: ProvisionRequest) -> None:
    """ADR-040 D4: refuse egress-requiring provisions when no proxy image is declared."""
    mounts = request.mounts
    reasons = egress_reasons(
        mcp_mounts=bool(mounts and mounts.mcp_servers),
        egress_domains=[
            str(d).strip() for d in (request.egress_domains or []) if str(d).strip()
        ],
    )
    if not reasons:
        return
    try:
        require_egress_proxy_image(reasons)
    except ValueError as exc:
        raise_typed(PlatformError.endpoints("bad_request", str(exc)), cause=exc)


def _active_site_profiles() -> list[str]:
    profiles: list[str] = []
    try:
        from ..orchestrator.site_config import get_site_config

        profiles.extend(get_site_config().runtime.profiles)
    except Exception:
        pass
    profiles.extend(_profile_list(os.environ.get("RIVAULT_PROFILES")))
    return list(dict.fromkeys(profiles))


def _with_site_support_profiles(profiles: list[str]) -> list[str]:
    resolved = list(profiles)
    if (
        resolved
        and "harness" in resolved
        and "litellm-db" not in resolved
        and "litellm-db" in _active_site_profiles()
    ):
        resolved.append("litellm-db")
    return resolved


def _resolve_launch_profiles(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: list[str] | None = None,
    co_locate_vllm: bool = False,
) -> list[str]:
    if explicit:
        return _profile_list(explicit)
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_profiles = _profile_list(_serving_profile(entry).get("launch_profiles"))
    if profile_profiles:
        return _with_site_support_profiles(profile_profiles)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    if backend == "bentoml" or _resolve_model_type(model_name, entry) == "vision":
        profiles = ["harness", "vision-bentoml"]
        if co_locate_vllm:
            profiles.insert(1, "vllm")
        return _with_site_support_profiles(profiles)
    # LLM / embedding models are served by vLLM only. Return an explicit vllm
    # profile set (instead of an empty list) so the orchestrator always sets
    # RIVAULT_PROFILES for the job. Otherwise the Slurm wrapper falls back to a
    # site-wide default that also starts the BentoML vision sidecar, launching
    # the wrong service stack for a text model and starving vLLM of GPU memory.
    return _with_site_support_profiles(["harness", "vllm"])


def _normalize_readiness_probe(value: Any) -> str | None:
    if not value:
        return None
    normalized = str(value)
    return "http" if normalized == "bentoml" else normalized


def _is_bentoml_vision_service(
    model_name: str,
    model_entry: dict[str, Any] | None,
    *,
    serving_backend: str | None = None,
    service_kind: str | None = None,
) -> bool:
    entry = model_entry or _resolve_registered_model_entry(model_name)
    backend = serving_backend or _resolve_serving_backend(model_name, entry)
    kind = service_kind or _resolve_service_kind(
        model_name, entry, serving_backend=backend
    )
    return (
        backend == "bentoml"
        or kind == "bentoml_vision"
        or _resolve_model_type(model_name, entry) == "vision"
    )


def _resolve_readiness_probe_mode(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    explicit: str | None = None,
) -> str:
    """Resolve the readiness probe mode for a provisioned endpoint."""
    explicit_probe = _normalize_readiness_probe(explicit)
    if explicit_probe:
        return explicit_probe
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_probe = _normalize_readiness_probe(
        _serving_profile(entry).get("readiness_probe")
    )
    if profile_probe:
        return profile_probe
    model_type = _resolve_model_type(model_name, entry)
    if _is_bentoml_vision_service(model_name, entry, serving_backend=serving_backend):
        return "http"
    if model_type == "embedding":
        return "models"
    return "chat"


def _resolve_readiness_path(
    model_name: str,
    model_entry: dict[str, Any] | None = None,
    *,
    serving_backend: str | None = None,
    service_kind: str | None = None,
    readiness_probe: str | None = None,
    explicit: str | None = None,
) -> str | None:
    if explicit:
        return explicit
    entry = model_entry or _resolve_registered_model_entry(model_name)
    profile_path = _serving_profile(entry).get("readiness_path")
    if profile_path:
        return str(profile_path)
    probe = readiness_probe or _resolve_readiness_probe_mode(
        model_name,
        entry,
        serving_backend=serving_backend,
    )
    if probe == "http" and _is_bentoml_vision_service(
        model_name,
        entry,
        serving_backend=serving_backend,
        service_kind=service_kind,
    ):
        return "/readyz"
    return None


def _enrich_endpoint_response(ep_dict: dict[str, Any]) -> dict[str, Any]:
    """Add served_model_name to endpoint dict if not already present."""
    if not ep_dict.get("served_model_name"):
        served = _resolve_served_model_name(ep_dict["model"])
        if served:
            ep_dict["served_model_name"] = served
    return ep_dict


# Mapping of capability tag -> (model_entry path, expected truthy value)
# ``None`` for the second slot means "boolean: any truthy value is OK".
_CAPABILITY_TAG_MAP: dict[str, Any] = {
    "vision": ("capabilities.supports_vision", None),
    "tool_calling": ("capabilities.supports_tool_calling", None),
    "structured_output": ("capabilities.supports_structured_output", None),
    "embedding": ("model_type", "embedding"),
    # ``gpu`` is treated as always-satisfied: every served model in this
    # platform runs on GPU. Kept here so workflows can declare it without
    # being filtered out.
    "gpu": (None, None),
}


def _entry_field(entry: dict[str, Any], dotted: str) -> Any:
    cur: Any = entry
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _capability_satisfied(entry: dict[str, Any] | None, tag: str) -> bool:
    """Return True if the model entry advertises the given capability tag."""
    if entry is None:
        return False
    if tag not in _CAPABILITY_TAG_MAP:
        # Unknown tag: never matches (caller gets an empty list).
        return False
    field_path, expected = _CAPABILITY_TAG_MAP[tag]
    if field_path is None:
        # Always-satisfied tag (e.g. "gpu").
        return True
    actual = _entry_field(entry, field_path)
    if expected is None:
        return bool(actual)
    return actual == expected


def _endpoint_passes_capability_filter(
    ep_dict: dict[str, Any],
    capabilities: list[str] | None,
    model_type: str | None,
) -> bool:
    """Return True if the endpoint's served model satisfies the filters."""
    if not capabilities and not model_type:
        return True
    entry = _resolve_registered_model_entry(ep_dict.get("model", ""))
    if model_type:
        actual_mt = (entry or {}).get("model_type")
        if actual_mt != model_type:
            return False
    if capabilities:
        for tag in capabilities:
            if not _capability_satisfied(entry, tag):
                return False
    return True


def _parse_capabilities_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags or None


def _is_all_models_query(model: str | None) -> bool:
    """Return True when the caller is requesting all visible models."""
    if model is None:
        return True
    return model.strip() in {"", "*"}


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/active",
    response_model=EndpointsListResponse,
    summary="Query active endpoints",
    description="""
    Query active (ready) endpoints for a project.

    This is the primary discovery endpoint for Gateway to find where to route requests.
    Returns only endpoints with status=ready and sufficient TTL remaining.

    **Usage by Gateway:**
    1. Extract project_id from incoming request JWT
    2. Call GET /umrs/endpoints/active?model=rivault-gpt-oss-static
    3. Use returned litellm_url to proxy the request
    """,
)
async def get_active_endpoints(
    model: str | None = Query(
        None, description="Filter by model name (catalog or served name)"
    ),
    capabilities: str | None = Query(
        None,
        description=(
            "Comma-separated capability tags the served model must advertise. "
            "Known tags: vision, tool_calling, structured_output, embedding, gpu."
        ),
    ),
    model_type: str | None = Query(
        None,
        description="Filter by ModelType: 'llm', 'vision', or 'embedding'.",
    ),
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
    raw_request: Request = None,
    requesting_user: str | None = Depends(_get_requesting_user),
) -> EndpointsListResponse:
    """Get active (ready) endpoints for the project (plus ADR-078 shared ones)."""
    from ._endpoint_access import caller_for

    caller = caller_for(
        raw_request, user_id=requesting_user, project_id=project_id, is_admin=False
    )

    normalized_model = model.strip() if model else None
    capability_tags = _parse_capabilities_csv(capabilities)
    normalized_model_type = model_type.strip() if model_type else None

    # If specific model requested, try to get that endpoint
    if not _is_all_models_query(normalized_model):
        endpoint = _find_named_ready(
            registry, project_id, normalized_model
        ) or registry.select_shared_endpoint(caller, normalized_model)
        if endpoint is None:
            return EndpointsListResponse(endpoints=[], total=0)
        live = await _filter_live_ready_endpoints(registry, [endpoint])
        if not live:
            return EndpointsListResponse(endpoints=[], total=0)
        ep_dict = _enrich_endpoint_response(live[0].to_dict())
        if not _endpoint_passes_capability_filter(
            ep_dict, capability_tags, normalized_model_type
        ):
            return EndpointsListResponse(endpoints=[], total=0)
        return EndpointsListResponse(
            endpoints=[EndpointResponse(**ep_dict)],
            total=1,
        )

    # List all ready endpoints for the project, then those shared with the caller
    live = [EndpointStatus.READY, EndpointStatus.DEGRADED]
    endpoints = registry.list_all(project_id=project_id, status_filter=live)
    endpoints += usable_endpoints(
        registry.list_all(status_filter=live), caller, shared_only=True
    )
    endpoints = await _filter_live_ready_endpoints(registry, endpoints)
    return _active_list_response(endpoints, capability_tags, normalized_model_type)


def _find_named_ready(
    registry: EndpointRegistry, project_id: str, normalized_model: str
):
    """Resolve single ready/DEGRADED endpoint by model or served name."""
    endpoint = registry.get(project_id, normalized_model, prefer_warm=True)
    if endpoint is not None:
        return endpoint
    # Try matching by served_model_name in metadata
    all_eps = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    for ep in all_eps:
        served = ep.metadata.get("served_model_name")
        if not served:
            served = _resolve_served_model_name(ep.model)
        if served == normalized_model:
            return ep
    return None


def _active_list_response(
    endpoints, capability_tags, normalized_model_type
) -> EndpointsListResponse:
    enriched = [_enrich_endpoint_response(e.to_dict()) for e in endpoints]
    filtered = [
        ep
        for ep in enriched
        if _endpoint_passes_capability_filter(
            ep, capability_tags, normalized_model_type
        )
    ]
    return EndpointsListResponse(
        endpoints=[EndpointResponse(**e) for e in filtered],
        total=len(filtered),
    )


@router.get(
    "",
    response_model=EndpointsListResponse,
    summary="List all endpoints",
    description="List all endpoints (for monitoring/admin). Optionally filter by project.",
)
async def list_endpoints(
    project_id: str | None = Depends(get_project_id_optional),
    status_filter: str | None = Query(None, description="Comma-separated statuses"),
    is_admin: bool = Depends(_caller_is_admin),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointsListResponse:
    """List all endpoints, optionally filtered.

    F-012: READY/DEGRADED endpoints are verified against the backend and
    ghost entries are pruned, same as the ``/active`` route.  This ensures
    ``r4 endpoint list --all`` never shows stale READY rows.
    """

    # Cross-project dump is admin-only: anonymous or non-admin callers
    # would otherwise enumerate every tenant's endpoints incl. host/port/
    # litellm_url (audited: all-tenant registry dump with no admin gate).
    if project_id is None and not is_admin:
        raise_typed(
            PlatformError.endpoints(
                "forbidden", "Cross-project endpoint listing requires admin role"
            ),
        )

    statuses = _parse_status_filter(status_filter)
    endpoints = registry.list_all(project_id=project_id, status_filter=statuses)

    # F-012: Prune ghost READY/DEGRADED endpoints at list time so the
    # user-facing ``r4 endpoint list`` output is always honest.
    endpoints = await _prune_ghost_ready(registry, endpoints)
    return EndpointsListResponse(
        endpoints=[EndpointResponse(**e.to_dict()) for e in endpoints],
        total=len(endpoints),
    )


def _parse_status_filter(status_filter: str | None) -> list[EndpointStatus] | None:
    if not status_filter:
        return None
    try:
        return [EndpointStatus(s.strip()) for s in status_filter.split(",")]
    except ValueError as e:
        raise_typed(
            PlatformError.endpoints("bad_request", f"Invalid status filter: {e}"),
            cause=e,
        )


async def _prune_ghost_ready(registry: EndpointRegistry, endpoints: list) -> list:
    ready_eps = [
        e
        for e in endpoints
        if e.status in (EndpointStatus.READY, EndpointStatus.DEGRADED)
    ]
    if not ready_eps:
        return endpoints
    live_ready = await _filter_live_ready_endpoints(registry, ready_eps)
    live_job_ids = {e.job_id for e in live_ready}
    return [
        e
        for e in endpoints
        if e.status not in (EndpointStatus.READY, EndpointStatus.DEGRADED)
        or e.job_id in live_job_ids
    ]


@router.post(
    "",
    response_model=EndpointResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register an endpoint",
    description="Register a new endpoint (internal use by UMRS orchestrator).",
)
async def register_endpoint(
    request: EndpointRegisterRequest,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Register a new endpoint."""

    endpoint = registry.register(
        project_id=request.project_id,
        model=request.model,
        job_id=request.job_id,
        host=request.host,
        port=request.port,
        backend=request.backend,
        ttl_sec=request.ttl_sec,
        metadata=request.metadata,
    )

    return EndpointResponse(**endpoint.to_dict())


@router.put(
    "/{job_id}/status",
    response_model=EndpointResponse,
    summary="Update endpoint status",
)
async def update_endpoint_status(
    job_id: str,
    request: EndpointStatusUpdate,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Update endpoint status (internal use)."""

    try:
        new_status = EndpointStatus(request.status)
    except ValueError:
        raise_typed(
            PlatformError.endpoints("bad_request", f"Invalid status: {request.status}"),
            cause=None,
        )

    endpoint = registry.update_status(job_id, new_status, request.error_message)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )

    return EndpointResponse(**endpoint.to_dict())


@router.post(
    "/{job_id}/refresh",
    response_model=EndpointResponse,
    summary="Refresh a stuck endpoint",
    description="""
    Re-resolve host/port from the backend and re-probe readiness.

    Used to recover endpoints stuck in PROVISIONING/STARTING with
    host="pending" after a _finalize_provision task died (e.g. UMRS restart).
    """,
)
async def refresh_endpoint(
    job_id: str,
    project_id: str = Depends(get_project_id),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Refresh a stuck endpoint."""
    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    from ..orchestrator.service import get_orchestration_service

    service = get_orchestration_service()
    refreshed = await service.refresh_endpoint(job_id)
    if not refreshed:
        raise_typed(
            PlatformError.endpoints(
                "not_found", f"Endpoint not found after refresh: {job_id}"
            ),
        )

    return EndpointResponse(**refreshed.to_dict())


@router.put(
    "/{job_id}/used",
    response_model=EndpointResponse,
    summary="Mark endpoint as used",
    description="Update last_used_at timestamp (for activity tracking).",
)
async def mark_endpoint_used(
    job_id: str,
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Mark endpoint as used (update last_used_at)."""

    # Identity before the lookup (no job_id existence oracle; H-1, 2026-10-18).
    _require_identity_for_mutation(project_id, "Mark endpoint used")
    existing = registry.get_by_job_id(job_id)
    if not existing:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(existing, project_id)

    endpoint = registry.mark_used(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )

    return EndpointResponse(**endpoint.to_dict())


def _metrics_headers() -> dict[str, str]:
    """LiteLLM bearer for /metrics when configured (best-effort)."""
    try:
        from ..orchestrator.site_config import get_site_config

        _key = get_site_config().network.litellm_master_key
        if _key:
            return {"Authorization": f"Bearer {_key}"}
    except Exception:  # noqa: BLE001 - best-effort; continue without auth
        pass
    return {}


async def _fetch_upstream_metrics(
    target_url: str, metrics_headers: dict[str, str]
) -> httpx.Response:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            return await client.get(target_url, headers=metrics_headers)
    except Exception as exc:  # noqa: BLE001 - best-effort proxy
        logger.warning("Failed to fetch endpoint metrics from %s: %s", target_url, exc)
        raise_typed(
            PlatformError.endpoints("bad_gateway", "Failed to fetch endpoint metrics"),
            cause=exc,
        )


@router.get(
    "/{job_id}/metrics",
    summary="Proxy endpoint /metrics for Prometheus scraping",
    description="""
    Return the /metrics exposition output of a ready endpoint.
    Prometheus uses this route as the scrape path for dynamic endpoint targets.
    """,
)
async def proxy_endpoint_metrics(
    job_id: str,
    project_id: str | None = Depends(_get_project_id_for_internal_scraper),
    is_admin: bool = Depends(_caller_is_admin),
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    """Proxy /metrics from the backing endpoint.

    The endpoint registry already tracks host/port.  Prometheus cannot reach
    the loopback-bound dynamic ports from its bridge network, so UMRS acts as
    an internal proxy and returns the raw exposition text.

    The design-internal "unauthenticated scraper" is NOT honored in strict
    mode (M10 finding): verifying LiteLLM bearer + host/port with no caller
    identity at all let any sidecar-reachable client metrics-scope a tenant's
    endpoint. Prometheus's own scrapes use the server-level /metrics route;
    this proxy requires a resolved caller project — or admin.
    """
    if project_id is None and _strict_auth_mode() and not is_admin:
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required",
                "Endpoint metrics proxy requires an authenticated project",
            ),
        )
    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    if endpoint.status not in {EndpointStatus.READY, EndpointStatus.DEGRADED}:
        raise_typed(
            PlatformError.endpoints(
                "unavailable", f"Endpoint {job_id} is {endpoint.status.value}"
            ),
        )

    target_url = f"http://{endpoint.host}:{endpoint.port}/metrics"
    # Pass LiteLLM master key if configured — LiteLLM protects /metrics with
    # bearer auth.  Fetched from site config so it matches the running stack.
    upstream = await _fetch_upstream_metrics(target_url, _metrics_headers())

    if upstream.status_code != status.HTTP_200_OK:
        raise_typed(
            PlatformError.endpoints(
                "bad_gateway", f"Endpoint metrics returned {upstream.status_code}"
            )
        )

    return Response(
        content=upstream.content,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an endpoint",
)
async def remove_endpoint(
    job_id: str,
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> None:
    """Remove an endpoint from the registry."""

    endpoint = registry.remove(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )


@router.get(
    "/stats",
    response_model=RegistryStatsResponse,
    summary="Get registry statistics",
)
async def get_registry_stats(
    _authorized: None = Depends(_require_internal_endpoint_writer),
    registry: EndpointRegistry = Depends(get_registry),
) -> RegistryStatsResponse:
    """Get registry statistics."""
    return RegistryStatsResponse(**registry.stats())


# =============================================================================
# Health check endpoints (U-204)
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Registry health check (U-204)",
    description="""
    Comprehensive health check for the endpoint registry.

    Returns:
    - Overall status (healthy/degraded/unhealthy)
    - Counts by status
    - Individual endpoint health details

    Use this to monitor the health of all active endpoints.
    """,
)
async def health_check(
    request: Request,
    include_details: bool = Query(False, description="Include per-endpoint details"),
    registry: EndpointRegistry = Depends(get_registry),
) -> HealthResponse:
    """Comprehensive health check endpoint.

    The counts stay open (monitors call it bare). Per-endpoint details
    (host, port, model, errors) listed every project's endpoints to any
    caller (H-1, 2026-10-17): they now need a caller identity (401 in strict
    mode without one); a non-admin sees its own project's endpoints, a
    signed admin sees all. Mvp anonymous callers keep the full list.
    """
    detail_project = _health_detail_project(request) if include_details else None
    all_endpoints = registry.list_all()
    counts = _health_counts(all_endpoints)
    endpoint_details = (
        [
            _endpoint_health_info(ep)
            for ep in all_endpoints
            if detail_project is None or ep.project_id == detail_project
        ]
        if include_details
        else []
    )

    # Determine overall status
    overall_status, message = _overall_health(
        total=len(all_endpoints), unhealthy=counts["unhealthy"]
    )
    return HealthResponse(
        status=overall_status,
        registry_ok=True,
        total_endpoints=len(all_endpoints),
        healthy_endpoints=counts["healthy"],
        unhealthy_endpoints=counts["unhealthy"],
        provisioning_endpoints=counts["provisioning"],
        endpoints=endpoint_details,
        message=message,
    )


def _health_detail_project(request: Request) -> str | None:
    """Project whose endpoint details the caller may see (None = all).

    ``request_caller_scope`` verifies the gateway HMAC (admin flag included)
    and 401s in strict mode without an identity; the mvp legacy
    ``X-Project-ID`` tier scopes to that project. An mvp anonymous caller
    (anyone who can reach the port) gets 403, the same admin-only rule as
    the cross-project ``GET /umrs/endpoints`` listing.
    """
    from ._scope_identity import request_caller_scope

    scope = request_caller_scope(request)
    if scope.is_admin:
        return None
    if scope.anonymous or not scope.project_id:
        raise_typed(
            PlatformError.endpoints(
                "forbidden", "Cross-project endpoint health details require admin role"
            ),
        )
    return scope.project_id


def _endpoint_is_healthy(ep: Any) -> bool:
    if ep.status in (EndpointStatus.READY, EndpointStatus.PENDING):
        return True  # PENDING: still healthy, just starting
    if ep.status in (EndpointStatus.FAILED, EndpointStatus.TERMINATED):
        return False
    return ep.is_usable


def _health_counts(endpoints: list) -> dict[str, int]:
    """Healthy / unhealthy / provisioning counts by endpoint status."""
    counts = {"healthy": 0, "unhealthy": 0, "provisioning": 0}
    for ep in endpoints:
        if ep.status == EndpointStatus.READY:
            counts["healthy"] += 1
        elif ep.status == EndpointStatus.PENDING:
            counts["provisioning"] += 1
        elif ep.status in (EndpointStatus.FAILED, EndpointStatus.TERMINATED):
            counts["unhealthy"] += 1
    return counts


def _endpoint_health_info(ep: Any) -> EndpointHealthInfo:
    """One endpoint's health detail row."""
    from datetime import datetime

    uptime = None
    if ep.ready_at:
        uptime = (datetime.utcnow() - ep.ready_at).total_seconds()
    return EndpointHealthInfo(
        job_id=ep.job_id,
        project_id=ep.project_id,
        model=ep.model,
        status=ep.status.value,
        host=ep.host,
        port=ep.port,
        healthy=_endpoint_is_healthy(ep),
        last_health_check=ep.last_used_at.isoformat() if ep.last_used_at else None,
        uptime_seconds=uptime,
        error=ep.error_message,
    )


def _overall_health(*, total: int, unhealthy: int) -> tuple[str, str | None]:
    if total == 0:
        return "healthy", "No endpoints registered"
    if unhealthy == 0:
        return "healthy", None
    if unhealthy < total:
        return "degraded", f"{unhealthy} endpoint(s) unhealthy"
    return "unhealthy", "All endpoints unhealthy"


@router.get(
    "/{job_id}",
    response_model=EndpointResponse,
    summary="Get endpoint by job_id",
)
async def get_endpoint(
    job_id: str,
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Get endpoint details by job_id."""

    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    endpoint = await _refresh_endpoint_readiness(registry, endpoint)

    return EndpointResponse(**endpoint.to_dict())


def _merge_request_env(request, env_vars: dict) -> None:
    """Request-supplied env_vars take precedence over catalog/roster defaults."""
    for _k, _v in (request.env_vars or {}).items():
        env_vars[str(_k)] = str(_v)


def _resolve_serving_scalars(request, model_entry) -> dict:
    """Straight-line resolver chain over catalog/profile/request defaults."""
    serving_profile = _serving_profile(model_entry)
    serving_backend = _resolve_serving_backend(
        request.model,
        model_entry=model_entry,
        explicit=request.serving_backend,
    )
    service_kind = _resolve_service_kind(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        explicit=request.service_kind,
    )
    readiness_probe = _resolve_readiness_probe_mode(
        request.model,
        model_entry=model_entry,
        serving_backend=serving_backend,
        explicit=request.readiness_probe,
    )
    return {
        "model_type": _resolve_model_type(request.model, model_entry),
        "serving_backend": serving_backend,
        "service_kind": service_kind,
        "launch_profiles": _resolve_launch_profiles(
            request.model,
            model_entry=model_entry,
            serving_backend=serving_backend,
            explicit=request.launch_profiles,
            co_locate_vllm=request.co_locate_vllm,
        ),
        "readiness_probe": readiness_probe,
        "readiness_path": _resolve_readiness_path(
            request.model,
            model_entry=model_entry,
            serving_backend=serving_backend,
            service_kind=service_kind,
            readiness_probe=readiness_probe,
            explicit=request.readiness_path,
        ),
        "serving_profile": serving_profile,
        "container_image": request.container_image
        or serving_profile.get("container_image"),
        "service_port": request.service_port or serving_profile.get("service_port"),
        "invoke_base_path": request.invoke_base_path
        or serving_profile.get("invoke_base_path"),
        "vision_adapter": request.vision_adapter
        or serving_profile.get("vision_adapter"),
        "vision_task": request.vision_task or serving_profile.get("vision_task"),
        "output_format": request.output_format or serving_profile.get("output_format"),
    }


def _env_setdefault_gt1(
    serving_profile: dict, env_vars: dict, key: str, env_name: str
) -> None:
    v = serving_profile.get(key)
    if v and int(v) > 1:
        env_vars.setdefault(env_name, str(v))


def _env_setdefault_present(
    serving_profile: dict, env_vars: dict, key: str, env_name: str
) -> None:
    v = serving_profile.get(key)
    if v:
        env_vars.setdefault(env_name, str(v))


def _apply_profile_vllm_env(serving_profile: dict, env_vars: dict) -> None:
    """Wire ServingProfile vLLM parameters as env-var defaults."""
    _env_setdefault_gt1(serving_profile, env_vars, "tensor_parallel_size", "VLLM_TP")
    _env_setdefault_present(
        serving_profile, env_vars, "gpu_memory_utilization", "VLLM_GPU_MEM_UTIL"
    )
    profile_dtype = serving_profile.get("dtype")
    if profile_dtype and profile_dtype != "auto":
        env_vars.setdefault("VLLM_DTYPE", str(profile_dtype))
    _env_setdefault_present(
        serving_profile, env_vars, "quantization", "VLLM_QUANTIZATION"
    )
    _env_setdefault_present(
        serving_profile, env_vars, "max_model_len", "VLLM_MAX_MODEL_LEN"
    )
    _env_setdefault_gt1(serving_profile, env_vars, "pipeline_parallel_size", "VLLM_PP")


def _apply_reasoning_env(request, env_vars: dict) -> None:
    """Reasoning-parser control: 'off' disables, 'on' forces native parser."""
    if request.reasoning == "off":
        env_vars["VLLM_REASONING_PARSER"] = "none"
    elif request.reasoning == "on":
        _forced_parser = _derive_reasoning_parser(request.model)
        if _forced_parser:
            env_vars["VLLM_REASONING_PARSER"] = _forced_parser


def _recommended_gpu(request, serving_profile: dict) -> int:
    profile_gpu_count = serving_profile.get("recommended_gpu_count")
    if request.gpu_count == 1 and profile_gpu_count and int(profile_gpu_count) > 1:
        logger.info(
            "Using recommended_gpu_count=%d from model catalog for '%s'",
            int(profile_gpu_count),
            request.model,
        )
        return int(profile_gpu_count)
    return request.gpu_count


def _recommended_nodes(request, serving_profile: dict, env_vars: dict) -> int:
    profile_node_count = serving_profile.get("recommended_node_count")
    effective_node_count = request.node_count
    if request.node_count == 1 and profile_node_count and int(profile_node_count) > 1:
        effective_node_count = int(profile_node_count)
        logger.info(
            "Using recommended_node_count=%d from model catalog for '%s'",
            effective_node_count,
            request.model,
        )
    if effective_node_count > 1:
        env_vars.setdefault("SLURM_NODES", str(effective_node_count))
    return effective_node_count


def _effective_strategy(request, serving_profile: dict, env_vars: dict) -> str | None:
    strategy = request.parallel_strategy or serving_profile.get("parallel_strategy")
    if strategy:
        env_vars.setdefault("VLLM_PARALLEL_STRATEGY", str(strategy).lower())
    return strategy


def _compute_resource_overrides(request, serving_profile: dict, env_vars: dict) -> dict:
    """GPU/nodes/max-len/strategy overrides honoring request-level defaults."""
    profile_max_model_len = serving_profile.get("max_model_len")
    effective_max_model_len = request.max_model_len
    if request.max_model_len == 4096 and profile_max_model_len:
        effective_max_model_len = int(profile_max_model_len)
    effective_parallel_strategy = _effective_strategy(
        request, serving_profile, env_vars
    )
    _apply_reasoning_env(request, env_vars)
    return {
        "gpu": _recommended_gpu(request, serving_profile),
        "max_len": effective_max_model_len,
        "node": _recommended_nodes(request, serving_profile, env_vars),
        "parallel": effective_parallel_strategy,
        "profile_max_model_len": profile_max_model_len,
    }


def _scale_partition_nodes(eff: dict, serving_plan, env_vars: dict) -> None:
    if eff["partition"] is None and serving_plan.site.partition:
        eff["partition"] = serving_plan.site.partition
    if eff["node"] == 1 and serving_plan.runtime.nodes > 1:
        eff["node"] = serving_plan.runtime.nodes
        env_vars.setdefault("SLURM_NODES", str(eff["node"]))


def _scale_gpu_len(eff: dict, serving_plan, env_vars: dict) -> None:
    if eff["gpu"] == 1 and serving_plan.runtime.gpus_per_node > 1:
        eff["gpu"] = serving_plan.runtime.gpus_per_node
    if eff["max_len"] == 4096 and serving_plan.engine.max_model_len:
        eff["max_len"] = serving_plan.engine.max_model_len
        env_vars.setdefault("VLLM_MAX_MODEL_LEN", str(eff["max_len"]))
    if not eff["parallel"] and serving_plan.runtime.strategy != "tp_pp":
        env_vars.setdefault("VLLM_PARALLEL_STRATEGY", serving_plan.runtime.strategy)


def _plan_apply_scaling(eff: dict, serving_plan, env_vars: dict) -> None:
    """Plan scaling override for request-default partition/nodes/gpus/len/strategy."""
    _scale_partition_nodes(eff, serving_plan, env_vars)
    _scale_gpu_len(eff, serving_plan, env_vars)


def _plan_apply_env(ss, serving_plan, env_vars: dict) -> None:
    for _k, _v in ss.spec_env_vars(serving_plan).items():
        env_vars.setdefault(str(_k), str(_v))
    if serving_plan.patches:
        for _k2, _v2 in ss.patches_to_env(serving_plan.patches).items():
            env_vars.setdefault(str(_k2), str(_v2))


def _maybe_apply_serving_plan(request, serving_plan, ss, env_vars, **eff) -> dict:
    """Apply plan overrides only when a curated plan was resolved."""
    if serving_plan is None:
        return eff
    return _apply_serving_plan(request, serving_plan, ss, env_vars, **eff)


def _apply_serving_plan(request, serving_plan, ss, env_vars, **eff) -> dict:
    """Apply curated serving-plan overrides (request defaults only)."""
    _plan_apply_scaling(eff, serving_plan, env_vars)
    if request.ttl_sec == 3600 and serving_plan.ttl_sec is not None:
        eff["ttl_sec"] = int(serving_plan.ttl_sec)
    _plan_apply_env(ss, serving_plan, env_vars)
    logger.info(
        "Resolved recipe for '%s' (plan=%s): partition=%s nodes=%d gpus=%d "
        "max_model_len=%d ttl_sec=%s",
        request.model,
        serving_plan.plan_hash()[:12],
        eff["partition"],
        eff["node"],
        eff["gpu"],
        eff["max_len"],
        eff["ttl_sec"],
    )
    return eff


def _optional_metadata_fields(
    metadata: dict,
    request,
    env_vars: dict,
    *,
    launch_profiles,
    readiness_path,
    service_port,
    invoke_base_path,
    container_image,
    vision_adapter,
    vision_task,
    output_format,
    serving_plan,
) -> None:
    """Optional metadata fields (set only when truthy) exactly as before."""
    _metadata_profiles_and_plan(metadata, launch_profiles, env_vars, serving_plan)
    if readiness_path:
        metadata["readiness_path"] = readiness_path
    if service_port:
        metadata["service_port"] = int(service_port)
    if invoke_base_path:
        metadata["invoke_base_path"] = str(invoke_base_path)
    if container_image:
        metadata["container_image"] = str(container_image)
    if vision_adapter:
        metadata["vision_adapter"] = str(vision_adapter)
    if vision_task:
        metadata["vision_task"] = str(vision_task)
    if output_format:
        metadata["output_format"] = str(output_format)


def _metadata_profiles_and_plan(
    metadata: dict, launch_profiles, env_vars: dict, serving_plan
) -> None:
    """launch_profiles + serving-plan metadata fields with env extras."""
    if launch_profiles:
        metadata["launch_profiles"] = launch_profiles
        # Use RIVAULT_EXTRA_PROFILES so the Slurm wrapper appends these
        # profiles to its default set ("harness vllm").
        existing_extra = env_vars.get("RIVAULT_EXTRA_PROFILES", "")
        all_extras = _dedupe_profiles(existing_extra, launch_profiles)
        if all_extras:
            env_vars["RIVAULT_EXTRA_PROFILES"] = " ".join(all_extras)
    if serving_plan is not None:
        metadata["serving_plan_hash"] = serving_plan.plan_hash()
        metadata["serving_plan"] = serving_plan.to_plan_dict()
        metadata["recipe_file"] = serving_plan.provenance.recipe_file


def _apply_vision_env(
    serving_backend, service_kind, env_vars: dict, container_image
) -> None:
    if serving_backend == "bentoml" or service_kind == "bentoml_vision":
        env_vars.setdefault("VISION_REQUIRE_PRESIGNED_URLS", "true")
        env_vars.setdefault("TRANSFORMERS_OFFLINE", "1")
        env_vars.setdefault("HF_HUB_OFFLINE", "1")
        if container_image:
            env_vars.setdefault("BENTOML_VISION_IMAGE", str(container_image))


def _apply_served_name_env(
    request, model_entry, metadata: dict, env_vars: dict
) -> str | None:
    served_model_name = _resolve_served_model_name(
        request.model,
        model_entry=model_entry,
    )
    if served_model_name and served_model_name != request.model:
        metadata["served_model_name"] = served_model_name
        env_vars["VLLM_SERVED_MODEL_NAME"] = served_model_name
    # Multi-model fields (forwarded via metadata so orchestrator backends
    # can read them without a service-layer schema change yet).
    metadata["serving_mode"] = request.serving_mode
    metadata["models"] = list(request.models or [request.model])
    if request.loras:
        metadata["loras"] = [
            {"name": lora.name, "source": lora.source} for lora in request.loras
        ]
    return served_model_name


def _record_endpoint_created(requesting_user, project_id, request, job_id) -> None:
    from .audit_events import emit_audit
    from .governance import record_resource_created

    emit_audit(
        "endpoint.created",
        user_id=requesting_user or "",
        project_id=project_id,
        details={"model": request.model, "tier": request.tier, "job_id": job_id},
    )
    record_resource_created(
        "endpoints",
        job_id,
        requesting_user or "",
        project_id=project_id,
        tier=request.tier,
        metadata={"model": request.model},
    )


def _accepted_provisioning_response(
    request,
    requesting_user,
    project_id,
    result,
    emit_audit,
    record_resource_created,
):
    """202+Retry-After path when provisioning started but is not yet ready."""
    from fastapi.responses import JSONResponse

    emit_audit(
        "endpoint.provisioning",
        user_id=requesting_user or "",
        project_id=project_id,
        details={
            "model": request.model,
            "tier": request.tier,
            "job_id": result.job_id,
        },
    )
    record_resource_created(
        "endpoints",
        result.job_id,
        requesting_user or "",
        project_id=project_id,
        tier=request.tier,
        metadata={"model": request.model},
    )
    response = ProvisionResponse(
        status="provisioning",
        job_id=result.job_id,
        retry_after=30,
    )
    return JSONResponse(
        content=response.model_dump(),
        status_code=status.HTTP_202_ACCEPTED,
        headers={"Retry-After": "30"},
    )


async def _finalize_provision(
    service,
    service_request,
    request,
    requesting_user,
    effective_project_id,
    provision_trace_id,
):
    """Call provision and translate the result (audit/record per branch)."""
    from .audit_events import emit_audit
    from .governance import record_resource_created

    try:
        result = await service.provision(service_request)
        if result.success and result.endpoint and result.endpoint.is_usable:
            # Ready immediately (unlikely for real provisioning)
            _record_endpoint_created(
                requesting_user, effective_project_id, request, result.job_id
            )
            return ProvisionResponse(
                status="ready",
                job_id=result.job_id,
                endpoint=EndpointResponse(**result.endpoint.to_dict())
                if result.endpoint
                else None,
            )
        elif result.success:
            return _accepted_provisioning_response(
                request,
                requesting_user,
                effective_project_id,
                result,
                emit_audit,
                record_resource_created,
            )
        else:
            emit_audit(
                "endpoint.provision_failed",
                user_id=requesting_user or "",
                project_id=effective_project_id,
                outcome="error",
                details={"model": request.model, "error": result.error},
            )
            return ProvisionResponse(
                status="failed",
                job_id=result.job_id,
                error=result.error,
            )
    except Exception as e:
        logger.exception(
            "provision-failed trace=%s model=%s: %s",
            provision_trace_id,
            request.model,
            e,
        )
        emit_audit(
            "endpoint.provision_failed",
            user_id=requesting_user or "",
            project_id=effective_project_id,
            outcome="error",
            details={
                "model": request.model,
                "error": str(e),
                "trace": provision_trace_id,
            },
        )
        return ProvisionResponse(
            status="failed",
            error=str(e),
        )


def _check_requesting_user_gate(requesting_user: str) -> None:
    """Validate the requesting user exists and (optionally) has a Slurm account."""
    try:
        from ..admin.user_store import UserStore, UserStoreError

        _user_store = UserStore()
        _user_record = _user_store.resolve_user(requesting_user)
        if _user_record.disabled:
            raise_typed(
                PlatformError.endpoints(
                    "forbidden", f"User '{_user_record.username}' is disabled."
                ),
            )
        _require_slurm_account(_user_record, requesting_user)
    except HTTPException:
        raise
    except UserStoreError as _use:
        # User record not found in the store — reject rather than proceed.
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                f"User '{requesting_user}' is not registered. "
                "Ask an admin to run: "
                f"umrs-server admin user add {requesting_user}",
            ),
            cause=_use,
        )
    except Exception:
        # User store infrastructure failure (I/O error, corrupt file, etc.).
        # Log and proceed to avoid blocking provisioning when the store is
        # temporarily unreachable (e.g. NFS hiccup in development).
        logger.warning(
            "User store unavailable while verifying user %s; proceeding without validation",
            requesting_user,
            exc_info=True,
        )


def _require_slurm_account(user_record, requesting_user: str) -> None:
    _site_name: str | None = None
    _require_acct: bool = False
    try:
        from ..orchestrator.site_config import load_site_config as _load_sc

        _sc = _load_sc()
        _site_name = _sc.name
        _require_acct = getattr(_sc.scheduler, "require_user_slurm_account", False)
    except FileNotFoundError:
        pass
    if not _require_acct:
        return
    if user_record.resolve_slurm_account(_site_name):
        return
    raise_typed(
        PlatformError.endpoints(
            "forbidden",
            (
                f"User '{user_record.username}' has no Slurm account configured. "
                "Ask an admin to run: "
                f"umrs-server admin user set-slurm-account {user_record.username} <account>"
            ),
        ),
    )


async def _try_warm_reuse(registry, service, project_id: str, request):
    """Prefer-warm short-circuit: ready endpoint with live backend or None."""
    if not request.prefer_warm:
        return None
    existing = registry.get(
        project_id,
        request.model,
        prefer_warm=True,
        min_ttl_sec=60,
        variant=request.variant,
    )
    if not (existing and existing.is_usable):
        return None
    # Verify the backend job is actually still alive before handing the
    # cached endpoint back (registry can be stale → ghost endpoints).
    if await _is_backend_job_alive(service, existing.job_id):
        return ProvisionResponse(
            status="ready",
            job_id=existing.job_id,
            endpoint=EndpointResponse(**existing.to_dict()),
        )
    logger.warning(
        "Stale warm endpoint detected for %s/%s (job_id=%s, variant=%s); "
        "marking failed and provisioning fresh",
        project_id,
        request.model,
        existing.job_id,
        request.variant,
    )
    registry.mark_failed(
        existing.job_id,
        "stale registry entry: backend job no longer alive",
    )
    return None


def _record_owner_and_access(
    metadata: dict, requesting_user, access: list[str]
) -> None:
    """ADR-078: who owns the row and who else may use it."""
    if requesting_user:
        metadata.setdefault("owner_id", requesting_user)
    if access:
        metadata["access"] = access


async def _access_and_warm_reuse(
    registry, raw_request, request, requesting_user, project_id, is_admin
):
    """(access list to record, warm ProvisionResponse or None)."""
    from ..orchestrator.service import get_orchestration_service
    from ._endpoint_access import caller_for, provision_access

    caller = caller_for(
        raw_request, user_id=requesting_user, project_id=project_id, is_admin=is_admin
    )
    access = provision_access(request, caller)
    service = get_orchestration_service()
    warm = await _try_warm_reuse(registry, service, project_id, request)
    if warm is None:
        warm = await _try_shared_warm_reuse(registry, service, caller, request)
    return access, warm


async def _try_shared_warm_reuse(registry, service, caller, request):
    """ADR-078: a READY endpoint another project shares with the caller.

    Same contract as the project warm path: only with ``prefer_warm``, and
    only after the backend job is verified alive.
    """
    if not request.prefer_warm:
        return None
    shared = registry.select_shared_endpoint(
        caller, request.model, variant=request.variant, min_ttl_sec=60
    )
    if shared is None or not shared.is_usable:
        return None
    if not await _is_backend_job_alive(service, shared.job_id):
        return None
    return ProvisionResponse(
        status="ready",
        job_id=shared.job_id,
        endpoint=EndpointResponse(**shared.to_dict()),
    )


def _dedupe_in_flight(registry, project_id: str, request):
    """202 + Retry-After when the same (model, variant) is already provisioning."""
    from fastapi.responses import JSONResponse

    provisioning = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.PENDING, EndpointStatus.PROVISIONING],
    )
    for ep in provisioning:
        if ep.model == request.model and ep.variant == request.variant:
            response = ProvisionResponse(
                status="provisioning",
                job_id=ep.job_id,
                retry_after=30,  # Check again in 30 seconds
            )
            return JSONResponse(
                content=response.model_dump(),
                status_code=status.HTTP_202_ACCEPTED,
                headers={"Retry-After": "30"},
            )
    return None


# =============================================================================
# Provisioning endpoints (U-202, U-203, U-205)
# =============================================================================


@router.post(
    "/provision",
    response_model=ProvisionResponse,
    responses={
        200: {"description": "Endpoint ready", "model": ProvisionResponse},
        202: {"description": "Provisioning in progress", "model": ProvisionResponse},
        503: {"description": "Service unavailable"},
    },
    summary="Provision an endpoint (U-202)",
    description="""
    Request provisioning of a new inference endpoint.

    Response codes (U-205):
    - 200 OK: Endpoint is ready and usable
    - 202 Accepted: Provisioning in progress, check Retry-After header
    - 503 Service Unavailable: Backend unavailable, retry later

    If provisioning, includes `retry_after` hint for when to check again.
    The actual HTTP response includes a `Retry-After` header when status is "provisioning".
    """,
)
async def provision_endpoint(
    request: ProvisionRequest,
    project_id: str | None = Depends(get_project_id_optional),
    is_admin: bool = Depends(_caller_is_admin),
    requesting_user: str | None = Depends(_get_requesting_user),
    registry: EndpointRegistry = Depends(get_registry),
    raw_request: Request = None,  # FastAPI injects; in-process callers may omit
):
    """Provision a new endpoint or return existing one."""
    from ..orchestrator.service import (
        ProvisionRequest as ServiceProvisionRequest,
    )
    from ..orchestrator.service import (
        get_orchestration_service,
    )

    effective_project_id = _resolve_provision_project_id(request, project_id)
    _enforce_provision_authorization(request, is_admin=is_admin)
    _enforce_mount_scope(
        request,
        is_admin=is_admin,
        project_id=effective_project_id,
        user_id=requesting_user,
    )

    # Server-side correlation trail: received before any heavy work so a
    # client-side HTTP timeout still leaves a discoverable server-side record.
    # Emit the same trace id on the failure path so operators can join the
    # two log lines without needing a request-id middleware.
    import uuid as _uuid

    provision_trace_id: str = _uuid.uuid4().hex[:12]
    logger.info(
        "provision-receive trace=%s model=%s project=%s user=%s admin=%s",
        provision_trace_id,
        request.model,
        effective_project_id,
        requesting_user or "-",
        is_admin,
    )

    # Governance: quota check before provisioning
    from .governance import check_quota

    check_quota(effective_project_id, "endpoints")

    # Validate that the requesting user is registered and (optionally) has
    # a Slurm account.  When ``require_user_slurm_account`` is False (the
    # default) users without a per-user Slurm account can still submit jobs
    # — the backend falls back to the site's master Slurm account.
    if requesting_user:
        _check_requesting_user_gate(requesting_user)

    # Honest-failure gate for unimplemented serving modes — MUST run
    # before the warm-endpoint short-circuit below, otherwise a multi-lora
    # request against an already-ready model silently returns the
    # single-model warm endpoint (audited drift, live verification).
    _enforce_serving_mode_supported(request)
    # Recipe policy + partition validation (audited drift, same class as the
    # serving-mode gate above): a request carrying env_vars/container_image
    # or an invalid partition used to slip through the warm-reuse and
    # already-provisioning dedupe returns below, silently receiving a
    # 200/202 it should have failed with. Fail fast before ANY reuse return.
    _enforce_recipe_policy(is_admin, request)

    # Check for existing warm endpoint first — the caller's project, then
    # endpoints shared with it (ADR-078); the access list is decided (and
    # refused) before any reuse.
    access, warm = await _access_and_warm_reuse(
        registry, raw_request, request, requesting_user, effective_project_id, is_admin
    )
    if warm is not None:
        return warm

    # Check if there's already a provisioning endpoint for this
    # project/model/variant combination. Multiple endpoints for the
    # same model are allowed when they have different variants, so
    # we only block on exact (model, variant) match.
    dup = _dedupe_in_flight(registry, effective_project_id, request)
    if dup is not None:
        return dup

    # Trigger new provisioning
    service = get_orchestration_service()
    model_entry = _resolve_registered_model_entry(request.model)
    _authorize_model_entry_for_caller(
        model_entry,
        user_id=requesting_user,
        project_id=effective_project_id,
    )
    resolved_model_path = _resolve_requested_model_path(
        request.model,
        request.model_path,
        model_entry=model_entry,
    )
    if not resolved_model_path:
        raise_typed(
            PlatformError.endpoints(
                "bad_request",
                (
                    f"Unable to resolve model path for '{request.model}'. "
                    "Register the model in the catalog or provide model_path explicitly."
                ),
            ),
        )
    _enforce_model_path_scope(
        resolved_model_path,
        model_entry,
        is_admin=is_admin,
        project_id=effective_project_id,
        user_id=requesting_user,
    )

    env_vars: dict[str, str] = {}
    # Merge request-supplied env_vars first so they take precedence
    # over ServingProfile/roster defaults.
    _merge_request_env(request, env_vars)
    scalars = _resolve_serving_scalars(request, model_entry)
    model_type = scalars["model_type"]
    serving_backend = scalars["serving_backend"]
    service_kind = scalars["service_kind"]
    launch_profiles = scalars["launch_profiles"]
    readiness_probe = scalars["readiness_probe"]
    readiness_path = scalars["readiness_path"]
    serving_profile = scalars["serving_profile"]
    container_image = scalars["container_image"]
    service_port = scalars["service_port"]
    invoke_base_path = scalars["invoke_base_path"]
    vision_adapter = scalars["vision_adapter"]
    vision_task = scalars["vision_task"]
    output_format = scalars["output_format"]

    # Wire ServingProfile vLLM parameters as env-var defaults.
    _apply_profile_vllm_env(serving_profile, env_vars)

    # Resource overrides (GPU/nodes/max-len/strategy/reasoning) with catalog
    # defaults applied only while the request carries field defaults.
    eff = _compute_resource_overrides(request, serving_profile, env_vars)
    effective_gpu_count = eff["gpu"]
    effective_max_model_len = eff["max_len"]
    effective_node_count = eff["node"]
    effective_parallel_strategy = eff["parallel"]

    # Per-gateway pool-roster sizing defaults. When this deployment publishes a
    # pool roster (UMRS_PROVISION_ROSTER), use it to size single-endpoint
    # provisions for known large models. This is what routes models like
    # DeepSeek-V4-Pro to the 4n4gpu-p partition with the right node/GPU/context
    # and TTL even when provisioned individually (not via `r4 pool up`) — without
    # it the request falls back to 1 node / 1 GPU and Slurm holds the job
    # PD (MaxMemPerLimit) forever. Roster values fill ONLY fields the caller
    # left at their defaults, so explicit API/CLI flags always win.
    # Canonical recipe resolver (M3): the ServingPlan is the authoritative
    # source of curated serving material. Legacy behavior is bitwise
    # equivalent (proved by umrs/tests/test_serving_spec_golden_diff.py) —
    # changing only the accessor from legacy defaults to the typed plan.
    from ..orchestrator import serving_spec as _ss
    from ..orchestrator.provision_roster import resolve_model_serving_spec

    effective_partition = request.partition
    effective_ttl_sec = request.ttl_sec
    serving_plan = resolve_model_serving_spec(
        request.model,
        request.variant,
        site=_ss.SiteSpec(
            cluster=_current_cluster_name(),
            backend="slurm-remote",
            partition=str(effective_partition or ""),
        ),
        project_id=effective_project_id,
        owner_id=requesting_user or "",
        storage_ref=resolved_model_path,
    )
    # The typed plan is authoritative for CURATED models (recipes). A generic
    # model has no curated plan — then the caller's explicit partition flag
    # must win over the site default (legacy-identical semantics from the
    # pre-M6 dispatcher where request.partition was honored). Otherwise a
    # custom model on a heterogeneous cluster is silently re-homed to the
    # partition defined by its unrelated recipe default (P1 regression class:
    # job 270440 ignored the m1 flag and queued on m0).
    eff = _maybe_apply_serving_plan(
        request,
        serving_plan,
        _ss,
        env_vars,
        partition=effective_partition,
        ttl_sec=effective_ttl_sec,
        gpu=effective_gpu_count,
        node=effective_node_count,
        max_len=effective_max_model_len,
        parallel=effective_parallel_strategy,
    )
    effective_partition = eff["partition"]
    effective_ttl_sec = eff["ttl_sec"]
    effective_gpu_count = eff["gpu"]
    effective_node_count = eff["node"]
    effective_max_model_len = eff["max_len"]

    request_metadata: dict[str, Any] = {
        "model_type": model_type,
        "serving_backend": serving_backend,
        "service_kind": service_kind,
        "readiness_probe": readiness_probe,
    }
    _optional_metadata_fields(
        request_metadata,
        request,
        env_vars,
        launch_profiles=launch_profiles,
        readiness_path=readiness_path,
        service_port=service_port,
        invoke_base_path=invoke_base_path,
        container_image=container_image,
        vision_adapter=vision_adapter,
        vision_task=vision_task,
        output_format=output_format,
        serving_plan=serving_plan,
    )
    _apply_vision_env(serving_backend, service_kind, env_vars, container_image)

    request_metadata.setdefault("owner_type", request.tier)
    request_metadata.setdefault("tier", request_metadata["owner_type"])
    _record_owner_and_access(request_metadata, requesting_user, access)
    _apply_served_name_env(request, model_entry, request_metadata, env_vars)
    # W5/U3 (2026-10-05): max_output_tokens / reasoning flags the LLM proxy's
    # /v1/models listing publishes per model (known values only).
    request_metadata.update(endpoint_limit_metadata(model_entry))

    _require_egress_proxy_for_request(request)

    # Create provision request
    service_request = ServiceProvisionRequest(
        project_id=effective_project_id,
        model=request.model,
        model_path=resolved_model_path,
        ttl_sec=effective_ttl_sec,
        gpu_count=effective_gpu_count,
        node_count=effective_node_count,
        max_model_len=effective_max_model_len,
        tier=request.tier,
        mounts=request.mounts.model_dump(exclude_none=True) if request.mounts else {},
        egress_domains=list(request.egress_domains),
        prefer_warm=request.prefer_warm,
        wait_for_ready=request.wait,
        partition_override=effective_partition,
        nodelist_hint=request.nodelist,
        min_gpu_memory_gb=serving_profile.get("min_gpu_memory_gb"),
        env_vars=env_vars,
        metadata=request_metadata,
        requesting_user=requesting_user,
        variant=request.variant,
    )

    return await _finalize_provision(
        service,
        service_request,
        request,
        requesting_user,
        effective_project_id,
        provision_trace_id,
    )


def _cancel_lineage_best_effort(endpoint: "Any | None", job_id: str) -> None:
    """ADR-058 D-2 lineage cancel: when the backend dispatch is unachieved
    (surrogate or backend-side failure), stop further attempts on the
    durable lineage too — the queued-claim race dies here.

    The attempt-side job status IS the marker (`cancel_job` holds the
    advisory lock against attempt creation; claim SQL never pulls
    CANCELLED). Never raises — the deprovision surface owns the verdict."""
    if endpoint is None:
        return
    durable_job_id = (endpoint.metadata or {}).get("durable_job_id")
    if not durable_job_id:
        return
    from ..jobs.cancel_gateway import cancel_job_best_effort

    cancelled = cancel_job_best_effort(
        durable_job_id, reason=f"endpoint_deprovision:{job_id}"
    )
    if cancelled:
        logger.info(
            "lineage cancel recorded for deprovision %s: durable_job %s cancelled",
            job_id,
            durable_job_id,
        )


def _raise_unachieved(job_id: str, backend_identity: str | None) -> None:
    """Never render a possibly-alive backend job as "deprovisioned" (D-3).

    The row's known identity is included when it exists so resolver chains
    have something real to query — surrogate rows name nothing."""
    detail = f"deprovision of {job_id} could not terminate the backend job"
    if backend_identity:
        detail += f" (backend_job_id={backend_identity})"
    else:
        detail += " (no backend identity assigned yet — pre-claim lineage)"
    raise_typed(
        PlatformError.endpoints("cancellation_unachieved", detail),
    )


def _surface_deprovision_outcome(
    outcome: "Any", registry: "Any", job_id: str
) -> "Any | None":
    """ADR-058 D-3 honest surface for the deprovision outcome.

    Returns a FastAPI response to send, or None when the outcome is GONE
    (route continues its existing audit + governance success path)."""
    from ..orchestrator.service import DeprovisionOutcome

    if outcome is DeprovisionOutcome.UNACHIEVED:
        still = registry.get_by_job_id(job_id)
        _cancel_lineage_best_effort(still, job_id)
        _raise_unachieved(job_id, still.backend_job_id if still else None)
    if outcome is DeprovisionOutcome.NOT_FOUND:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    if outcome is DeprovisionOutcome.GONE:
        return None

    # STOPPING family: deprovision was issued but the backend job is still
    # settling (e.g. Slurm COMPLETING). A job that is still tearing down is
    # not a server error — report it as "accepted, stopping" (202) so the
    # client does not surface a spurious 500. The janitor finalizes the
    # registry removal once the backend job actually exits.
    from fastapi.responses import JSONResponse

    still = registry.get_by_job_id(job_id)
    cleanup_status = (still.metadata or {}).get("cleanup_status") if still else None
    if still is not None and cleanup_status:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "job_id": job_id,
                "status": "stopping",
                "detail": (
                    "Deprovision initiated; backend job still stopping "
                    f"({cleanup_status})."
                ),
            },
        )
    raise_typed(
        PlatformError.endpoints("failure", f"Failed to deprovision endpoint: {job_id}"),
    )
    return None


class EndpointAccessUpdate(BaseModel):
    """ADR-078: the complete access list beyond the owning project."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = owning project only"
        ),
    )


@router.put(
    "/{job_id}/access",
    response_model=EndpointResponse,
    summary="Set who may use an endpoint (ADR-078)",
)
async def set_endpoint_access(
    job_id: str,
    body: EndpointAccessUpdate,
    raw_request: Request,
    project_id: str | None = Depends(get_project_id_optional),
    is_admin: bool = Depends(_caller_is_admin),
    requesting_user: str | None = Depends(_get_requesting_user),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointResponse:
    """Replace the endpoint's access list (owning project or admin only).
    # auth: verified
    """
    from ._endpoint_access import caller_for, checked_access

    # Identity before the lookup: an anonymous strict caller gets 401 for
    # every job_id, never a 404 that tells it which ids exist (H-1, 2026-10-18).
    _require_identity_for_mutation(project_id, "Access change")
    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}")
        )
    if not is_admin:
        _forbid_if_project_mismatch(endpoint, project_id)
    caller = caller_for(
        raw_request,
        user_id=requesting_user,
        project_id=project_id or endpoint.project_id,
        is_admin=is_admin,
    )
    access = checked_access(body.access, caller)
    updated = registry.update_metadata(job_id, {"access": access})
    from .audit_events import emit_audit

    emit_audit(
        "endpoint.access_changed",
        project_id=endpoint.project_id,
        details={"job_id": job_id, "access": access, "by": requesting_user or ""},
    )
    return EndpointResponse(**(updated or endpoint).to_dict())


@router.delete(
    "/{job_id}/deprovision",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deprovision an endpoint",
    description="Stop and clean up a provisioned endpoint.",
)
async def deprovision_endpoint(
    job_id: str,
    force: bool = Query(False, description="Force cleanup even if in use"),
    project_id: str | None = Depends(get_project_id_optional),
    registry: EndpointRegistry = Depends(get_registry),
) -> None:
    """Deprovision (stop and cleanup) an endpoint."""
    from ..orchestrator.service import get_orchestration_service

    # Identity before the lookup (no job_id existence oracle; H-1, 2026-10-18).
    _require_identity_for_mutation(project_id, "Deprovision")
    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(endpoint, project_id)

    service = get_orchestration_service()

    outcome = await service.deprovision(job_id, force=force)
    surfaced = _surface_deprovision_outcome(outcome, registry, job_id)
    if surfaced is not None:
        return surfaced

    from .audit_events import emit_audit

    emit_audit(
        "endpoint.deprovisioned",
        project_id=endpoint.project_id if endpoint else "",
        details={"job_id": job_id, "force": force},
    )
    from .governance import record_resource_deleted

    record_resource_deleted(
        "endpoints",
        job_id,
        project_id=endpoint.project_id if endpoint else "",
    )


# ---------------------------------------------------------------------------
# Rolling endpoint upgrade (Q-13 Phase 2)
# ---------------------------------------------------------------------------


def _registered_entry_for_path(model_path: str) -> dict[str, Any] | None:
    """The catalog entry whose storage_ref is *model_path* (the upgrade route
    names a path, not a model); a missing or failing store reads as none."""
    from .models import get_optional_model_store

    store = get_optional_model_store()
    if store is None:
        return None
    from ..model_registry.store import AccessContext

    try:
        entries = store.list_models(AccessContext(user_id="", is_admin=True), limit=500)
    except Exception as exc:  # noqa: BLE001 — fail closed: the path must then be the caller's
        logger.warning("Model store lookup for %s failed: %s", model_path, exc)
        return None
    rows = [entry.to_dict() for entry in entries]
    return _pick_first_by_scope(
        [
            row
            for row in rows
            if row.get("storage_ref") and not caller_named_host_path(model_path, row)
        ]
    )


def _upgrade_model_path(
    model_path: str,
    *,
    is_admin: bool,
    project_id: str | None,
    user_id: str | None,
) -> str:
    """The upgrade target, checked like a provision model_path override.

    Site-wide prefixes for everyone; for a non-admin, a host path it names
    must lie under its own roots unless it is the storage_ref of a catalog
    entry it may provision (H-1, 2026-10-18; it reached
    service.upgrade_endpoint unchecked). An mvp caller with no identity at
    all keeps the site-wide check only, as _authorize_model_entry_for_caller
    does.
    """
    resolved = _validated_model_path_override(model_path.strip())
    if is_admin or (project_id is None and user_id is None):
        return resolved
    caller_project, caller_user = project_id or "", user_id or ""
    if model_path_in_scope(
        resolved, None, project_id=caller_project, user_id=caller_user
    ):
        return resolved
    entry = _registered_entry_for_path(resolved)
    _authorize_model_entry_for_caller(entry, user_id=user_id, project_id=project_id)
    _enforce_model_path_scope(
        resolved, entry, is_admin=False, project_id=caller_project, user_id=user_id
    )
    return resolved


class EndpointUpgradeRequest(BaseModel):
    """Payload for POST /umrs/endpoints/{job_id}/upgrade."""

    model_config = ConfigDict(extra="forbid")

    model_path: str = Field(
        ...,
        min_length=1,
        description="New model path (must already be registered)",
    )
    timeout_sec: int = Field(
        900,
        ge=1,
        le=3600,
        description="Maximum seconds to wait for replacement health",
    )
    wait: bool = Field(
        False,
        description="If true, block until the flip converges (SyncSem)",
    )


class EndpointUpgradeResponse(BaseModel):
    """Response for a successful upgrade acceptance."""

    job_id: str
    old_model_path: str
    new_model_path: str
    status: str  # provisioning | ready | degraded | failed
    new_job_id: str | None = None
    error: str | None = None


@router.post(
    "/{job_id}/upgrade",
    response_model=EndpointUpgradeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Rolling model-version upgrade (Q-13 Phase 2)",
    description=(
        "Provision a replacement for the referenced endpoint, wait for its "
        "readiness, flip the registry record so subsequent routing lands on "
        "the new model, then retire the old backend job. The old endpoint is "
        "preserved until the new one is healthy. "
        "One upgrade per endpoint_id at a time; conflicting upgrades return "
        "HTTP 409."
    ),
)
async def upgrade_endpoint(
    job_id: str,
    request: EndpointUpgradeRequest,
    project_id: str | None = Depends(get_project_id_optional),
    is_admin: bool = Depends(_caller_is_admin),
    requesting_user: str | None = Depends(_get_requesting_user),
    registry: EndpointRegistry = Depends(get_registry),
) -> EndpointUpgradeResponse:
    """Rolling upgrade of a running endpoint.

    Security gates (preserved from existing lifecycle):
      - Resolved project identity required for mutation in strict mode,
        checked before the lookup (no job_id existence oracle)
      - Project mismatch → 404 (never leak endpoint presence cross-project)
      - model_path: site-wide prefixes, then the per-caller root rule of a
        provision model_path override for non-admins (H-1, 2026-10-18)
      - Concurrent upgrade → 409 (one flip per endpoint at a time)

    Health-failure semantics:
      - Replacement provision fails → error surfaced, old untouched
      - Replacement lands but never goes ready before the timeout →
        new endpoint deprovisioned, old endpoint untouched
      - Registry update fails after new endpoint is ready → new endpoint
        is still live but not serving the flip; the call returns 500 and
        the janitor collects the zombie (no silent success)
    """
    _require_identity_for_mutation(project_id, "Upgrade")
    endpoint = registry.get_by_job_id(job_id)
    if endpoint is None:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )
    _forbid_if_project_mismatch(endpoint, project_id)
    new_model_path = _upgrade_model_path(
        request.model_path,
        is_admin=is_admin,
        project_id=project_id,
        user_id=requesting_user,
    )

    # Concurrency gate (Q-13 #3): one in-flight upgrade per endpoint.
    if endpoint.metadata.get("upgrade_in_flight"):
        raise_typed(
            PlatformError.endpoints(
                "conflict", f"an upgrade is already in flight for endpoint {job_id}"
            ),
        )

    from ..orchestrator.service import get_orchestration_service

    service = get_orchestration_service()

    old_metadata = dict(endpoint.metadata or {})
    old_model_path = old_metadata.get("model_path") or endpoint.model

    success, new_job_id, error = await service.upgrade_endpoint(
        job_id=job_id,
        new_model_path=new_model_path,
        timeout_sec=request.timeout_sec,
    )

    final = registry.get_by_job_id(job_id)
    final_status = final.status.value if final else "unknown"

    if not success:
        # Old endpoint MUST still be addressable after a failed upgrade.
        # If it is not, treat it as a failure report (500), never a 2xx.
        if final is None:
            raise_typed(
                PlatformError.endpoints(
                    "failure",
                    (f"upgrade failed and old endpoint is unrecoverable: {error}"),
                ),
            )
        return EndpointUpgradeResponse(
            job_id=job_id,
            old_model_path=str(old_model_path),
            new_model_path=new_model_path,
            status="failed",
            new_job_id=None,
            error=error,
        )

    return EndpointUpgradeResponse(
        job_id=job_id,
        old_model_path=str(old_model_path),
        new_model_path=new_model_path,
        status=final_status if final_status in ("ready", "degraded") else "ready",
        new_job_id=new_job_id,
        error=None,
    )


# ---------------------------------------------------------------------------
# Admin route: programmatic location update (Task 6 Gap B)
# ---------------------------------------------------------------------------


class EndpointLocationUpdate(BaseModel):
    """Payload for PUT /umrs/endpoints/{job_id}/location."""

    host: str | None = None
    port: int | None = None
    metadata: dict[str, Any] | None = None


class EndpointLocationResponse(BaseModel):
    """Response body for a successful location update."""

    job_id: str
    host: str
    port: int
    status: str
    metadata: dict[str, Any] | None = None


@router.put(
    "/{job_id}/location",
    response_model=EndpointLocationResponse,
    status_code=status.HTTP_200_OK,
)
async def update_endpoint_location(
    job_id: str,
    request: EndpointLocationUpdate,
    registry: EndpointRegistry = Depends(get_registry),
    _authorized: None = Depends(_require_internal_endpoint_writer),
) -> EndpointLocationResponse:
    """Update endpoint host/port and merge metadata.

    Admin-only route for operational heals when the boot-time
    reconciler cannot auto-resolve an endpoint (e.g. external
    system modified the registry directly).

    Requires strict authentication mode (UMRS_AUTH_MODE=strict) and an admin
    caller: the docstring said admin-only, but any caller could rewrite any
    endpoint's host and port (H-1, 2026-10-17).
    """
    # Default strict (audit B12): mvp is an explicit dev opt-in.
    if os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() != "strict":
        raise_typed(
            PlatformError.endpoints(
                "forbidden",
                (
                    "Route requires UMRS_AUTH_MODE=strict. "
                    "This admin route is only available in strict auth mode."
                ),
            ),
        )

    endpoint = registry.get_by_job_id(job_id)
    if not endpoint:
        raise_typed(
            PlatformError.endpoints("not_found", f"Endpoint not found: {job_id}"),
        )

    registry.update_location(
        job_id,
        host=request.host,
        port=request.port,
        metadata=request.metadata,
    )

    updated = registry.get_by_job_id(job_id)
    if not updated:
        raise_typed(
            PlatformError.endpoints(
                "failure", "Endpoint disappeared after location update"
            ),
        )

    return EndpointLocationResponse(
        job_id=updated.job_id,
        host=updated.host,
        port=updated.port,
        status=updated.status.value,
        metadata=updated.metadata,
    )
