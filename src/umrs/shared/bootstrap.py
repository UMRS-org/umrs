"""
UMRS Bootstrap Context Builder.

Builds a versioned, capability-aware bootstrap response that clients
use to materialise ephemeral local configuration.  The response is a
*deterministic projection* of server-authoritative state — regeneration
is always safe and idempotent.

Protocol version: 1
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from umrs.shared.auth import AuthContext

logger = logging.getLogger(__name__)

# Current bootstrap protocol version.  Bump when the response schema
# changes in a backward-incompatible way.
BOOTSTRAP_VERSION = 1

# Default freshness window — clients should re-fetch after this period.
DEFAULT_REFRESH_AFTER_HOURS = 12

# Default bootstrap validity — clients MUST re-fetch after this period.
DEFAULT_EXPIRES_AFTER_HOURS = 24


def build_bootstrap_context(
    auth_ctx: AuthContext,
    *,
    site_config: Any | None = None,
    user_projects: list[dict[str, Any]] | None = None,
    capabilities: dict[str, bool] | None = None,
    services: dict[str, str | None] | None = None,
    feature_flags: dict[str, bool] | None = None,
    server_name: str = "umrs",
    refresh_after_hours: int = DEFAULT_REFRESH_AFTER_HOURS,
    expires_after_hours: int = DEFAULT_EXPIRES_AFTER_HOURS,
) -> dict[str, Any]:
    """Build a versioned bootstrap context for the authenticated user.

    The returned dict is serialisable to JSON and delivered to clients
    via ``GET_BOOTSTRAP_CONTEXT`` (ZMQ) or ``GET /umrs/bootstrap``
    (HTTP).

    Args:
        auth_ctx: Authenticated user context (from ``authenticate_request``).
        site_config: Optional ``SiteConfig`` instance.  When provided,
            scheduler and storage profile sections are populated from it.
        user_projects: List of project dicts the user belongs to.  Each
            dict should contain ``id``, ``role``, and optionally
            ``slurm_account``.
        capabilities: Override capability flags.  Merged with defaults.
        services: Override service endpoints.  Merged with defaults.
        feature_flags: Optional feature flags dict.
        server_name: Logical server name used in ``session.issued_by``.
        refresh_after_hours: Hours until client *should* re-fetch.
        expires_after_hours: Hours until client *must* re-fetch.

    Returns:
        Versioned bootstrap dict ready for JSON serialisation.
    """
    now = datetime.now(timezone.utc)

    ctx: dict[str, Any] = {
        "version": BOOTSTRAP_VERSION,
        "issued_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=expires_after_hours)).isoformat(),
        "refresh_after": f"{refresh_after_hours}h",
    }

    # --- session metadata ---
    ctx["session"] = {
        "session_id": auth_ctx.session_id or f"sess-{uuid.uuid4().hex[:12]}",
        "issued_by": server_name,
        "auth_method": "jwt",
    }

    # --- identity ---
    ctx["identity"] = {
        "username": auth_ctx.username,
        "user_id": auth_ctx.user_id,
        "email": auth_ctx.email,
        "role": auth_ctx.role.value,
    }

    # --- projects ---
    ctx["projects"] = user_projects or []

    # --- capabilities ---
    default_caps: dict[str, bool] = {
        "token_refresh": True,
        "interactive_jobs": True,
        "object_storage": True,
        "rag": True,
        "mcp_custom_servers": False,
        "model_download": False,
    }
    if capabilities:
        default_caps.update(capabilities)
    ctx["capabilities"] = default_caps

    # --- profiles (from site_config when available) ---
    ctx["profiles"] = _build_profiles(auth_ctx, site_config)

    # --- services ---
    ctx["services"] = _build_services(site_config, services)

    # --- feature flags ---
    ctx["feature_flags"] = feature_flags or {}

    return ctx


def _build_profiles(
    auth_ctx: AuthContext,
    site_config: Any | None,
) -> dict[str, Any]:
    """Build compositional profiles from site config."""
    profiles: dict[str, Any] = {}

    if site_config is None:
        return profiles

    # Scheduler profile
    sched = getattr(site_config, "scheduler", None)
    if sched is not None:
        profiles["scheduler"] = {
            "type": getattr(sched, "type", "slurm"),
            "default_partition": getattr(sched, "partition", "gpu"),
            "max_walltime": getattr(sched, "time_limit", "02:00:00"),
            "gpus": getattr(sched, "gpus", 1),
        }

    # Storage profile
    storage = getattr(site_config, "storage", None)
    if storage is not None:
        profiles["storage"] = {
            "type": "minio",
            "default_bucket": f"{auth_ctx.username}-data",
            "credentials_endpoint": "/umrs/storage/credentials",
        }

    # Interactive profile
    vllm = getattr(site_config, "vllm", None)
    if vllm is not None:
        profiles["interactive"] = {
            "default_model": "rivault-gpt-oss-static",
            "max_context_length": getattr(vllm, "max_model_len", 4096),
        }

    return profiles


def _build_services(
    site_config: Any | None,
    overrides: dict[str, str | None] | None,
) -> dict[str, str | None]:
    """Build services endpoint map."""
    svc: dict[str, str | None] = {
        "gateway": None,
        "umrs_zmq": None,
        "storage": None,
        "telemetry": None,
    }

    if site_config is not None:
        network = getattr(site_config, "network", None)
        if network is not None:
            zmq_port = getattr(network, "control_port", None)
            if zmq_port:
                svc["umrs_zmq"] = f"tcp://0.0.0.0:{zmq_port}"

        storage = getattr(site_config, "storage", None)
        if storage is not None:
            s3_endpoint = getattr(storage, "s3_endpoint", None)
            if s3_endpoint:
                svc["storage"] = s3_endpoint

    if overrides:
        svc.update(overrides)

    return svc
