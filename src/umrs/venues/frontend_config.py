"""Frontend-venue runtime configuration (wave 2026-09-21, ADR-046 A2).

Every provisioner input that used to be a placeholder default
(``@sha256:PINNED`` images, ``/var/lib/umrs/sessions``) resolves here from
``UMRS_VENUE_*`` env so the session-start route can refuse honestly
(``companionvenue.unavailable`` + the missing names) instead of provisioning
against a placeholder.

Subnet allocation: ``UMRS_VENUE_SUBNET_POOL`` (default ``172.30.0.0/16``) is
sliced into /24 session networks; a slice is free when no provisioning or
running venue-session row holds it.
"""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .frontend_podman import SESSION_DIR_ROOT, ProvisionConfig

_PLACEHOLDER_IMAGE_TAIL = "@sha256:PINNED"
VALID_WEB_POLICIES = ("public", "allowlist", "none")


class SubnetPoolExhausted(RuntimeError):
    """No free /24 left in the configured session pool."""


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class FrontendVenueConfig:
    """Resolved runtime knobs; ``missing()`` names what still blocks a start."""

    enabled: bool = False
    infer_host: str = ""
    infer_port: int = 8443
    infer_uplink: str = "http"  # "http" (bootstrap) | "https" (internal CA)
    infer_resolve_via: str = ""
    internal_ca_path: str = "/etc/umrs/internal-ca.pem"
    # This API as reachable from the egress-net bridge (host:port). The
    # sidecar's rail listener proxies the worker's journal calls here.
    rail_upstream: str = ""
    rail_port: int = 8081
    worker_image: str = ""
    nginx_image: str = ""
    squid_image: str = ""
    subnet_pool: str = "172.30.0.0/16"
    session_root: Path = SESSION_DIR_ROOT
    gateway_host: str = ""
    umrs_host: str = ""
    lease_seconds: int = 3600
    max_lifetime_seconds: int = 7 * 86_400
    worker_memory: str = "16g"
    worker_cpus: str = "8"
    worker_uid: int = 1000
    web_policy_default: str = "public"
    idle_seconds: int = 30 * 60
    rail_ip_check: bool = True
    rail_extend_seconds: int = 900

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FrontendVenueConfig:
        env = os.environ if env is None else env
        return cls(
            enabled=_env_bool(env, "UMRS_VENUE_ENABLED", False),
            infer_host=env.get("UMRS_VENUE_INFER_HOST", "").strip(),
            infer_port=_env_int(env, "UMRS_VENUE_INFER_PORT", 8443),
            infer_uplink=(env.get("UMRS_VENUE_INFER_UPLINK", "http").strip() or "http"),
            infer_resolve_via=env.get("UMRS_VENUE_INFER_RESOLVE_VIA", "").strip(),
            internal_ca_path=(
                env.get("UMRS_VENUE_INTERNAL_CA_PATH", "").strip()
                or "/etc/umrs/internal-ca.pem"
            ),
            rail_upstream=env.get("UMRS_VENUE_API_UPSTREAM", "").strip(),
            rail_port=_env_int(env, "UMRS_VENUE_RAIL_PORT", 8081),
            worker_image=env.get("UMRS_VENUE_WORKER_IMAGE", "").strip(),
            nginx_image=env.get("UMRS_VENUE_NGINX_IMAGE", "").strip(),
            squid_image=env.get("UMRS_VENUE_SQUID_IMAGE", "").strip(),
            subnet_pool=(
                env.get("UMRS_VENUE_SUBNET_POOL", "").strip() or "172.30.0.0/16"
            ),
            session_root=Path(
                env.get("UMRS_VENUE_SESSION_ROOT", "").strip() or str(SESSION_DIR_ROOT)
            ),
            gateway_host=env.get("UMRS_VENUE_GATEWAY_HOST", "").strip(),
            umrs_host=env.get("UMRS_VENUE_UMRS_HOST", "").strip(),
            lease_seconds=_env_int(env, "UMRS_VENUE_LEASE_SECONDS", 3600),
            max_lifetime_seconds=_env_int(
                env, "UMRS_VENUE_MAX_LIFETIME_SECONDS", 7 * 86_400
            ),
            worker_memory=(env.get("UMRS_VENUE_WORKER_MEMORY", "").strip() or "16g"),
            worker_cpus=(env.get("UMRS_VENUE_WORKER_CPUS", "").strip() or "8"),
            worker_uid=_env_int(env, "UMRS_VENUE_WORKER_UID", 1000),
            web_policy_default=(
                env.get("UMRS_VENUE_WEB_POLICY_DEFAULT", "").strip() or "public"
            ),
            idle_seconds=_env_int(env, "UMRS_VENUE_IDLE_SECONDS", 30 * 60),
            rail_ip_check=_env_bool(env, "UMRS_VENUE_RAIL_IP_CHECK", True),
            rail_extend_seconds=_env_int(env, "UMRS_VENUE_RAIL_EXTEND_SECONDS", 900),
        )

    def missing(self) -> list[str]:
        """Names of settings that block provisioning (empty = ready)."""
        out: list[str] = []
        if not self.enabled:
            out.append("UMRS_VENUE_ENABLED")
        if not self.infer_host:
            out.append("UMRS_VENUE_INFER_HOST")
        if not self.rail_upstream:
            out.append("UMRS_VENUE_API_UPSTREAM")
        for name, value in (
            ("UMRS_VENUE_WORKER_IMAGE", self.worker_image),
            ("UMRS_VENUE_NGINX_IMAGE", self.nginx_image),
            ("UMRS_VENUE_SQUID_IMAGE", self.squid_image),
        ):
            if not value or value.endswith(_PLACEHOLDER_IMAGE_TAIL):
                out.append(name)
        if self.web_policy_default not in VALID_WEB_POLICIES:
            out.append("UMRS_VENUE_WEB_POLICY_DEFAULT")
        try:
            ipaddress.ip_network(self.subnet_pool, strict=True)
        except ValueError:
            out.append("UMRS_VENUE_SUBNET_POOL")
        return out

    def provision_config(
        self,
        *,
        session_id: str,
        owner: str,
        source_ids: tuple[str, ...],
        default_source_id: str,
        subnet: str,
        web_policy: str,
        web_allow_domains: tuple[str, ...] = (),
        project: str = "",
    ) -> ProvisionConfig:
        return ProvisionConfig(
            session_id=session_id,
            owner=owner,
            source_ids=source_ids,
            default_source_id=default_source_id,
            subnet=subnet,
            infer_host=self.infer_host,
            infer_port=self.infer_port,
            worker_image=self.worker_image,
            nginx_image=self.nginx_image,
            squid_image=self.squid_image,
            web_policy=web_policy,
            web_allow_domains=web_allow_domains,
            infer_uplink=self.infer_uplink,
            internal_ca_path=self.internal_ca_path,
            infer_resolve_via=self.infer_resolve_via,
            lease_seconds=self.lease_seconds,
            max_lifetime_seconds=self.max_lifetime_seconds,
            worker_memory=self.worker_memory,
            worker_cpus=self.worker_cpus,
            worker_uid=self.worker_uid,
            project=project,
            gateway_host=self.gateway_host,
            umrs_host=self.umrs_host,
            rail_upstream=self.rail_upstream,
            rail_port=self.rail_port,
        )


def allocate_subnet(
    pool_cidr: str, taken: Iterable[str], *, new_prefix: int = 24
) -> str:
    """First free ``/new_prefix`` slice of ``pool_cidr`` not in ``taken``."""
    pool = ipaddress.ip_network(pool_cidr, strict=True)
    used = {str(ipaddress.ip_network(t, strict=False)) for t in taken if t}
    for candidate in pool.subnets(new_prefix=new_prefix):
        text = str(candidate)
        if text not in used:
            return text
    raise SubnetPoolExhausted(f"no free /{new_prefix} left in {pool_cidr}")
