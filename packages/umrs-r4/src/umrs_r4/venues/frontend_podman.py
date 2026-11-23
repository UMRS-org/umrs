"""Frontend-venue provisioner: the r4 worker spec on the core host runtime.

Since SH-H0 (2026-11-24, ADR-098 D-2) the pod and egress provisioning —
internal network, lease, token secret, sidecar configs (nginx inference
listener + squid), egress pod, egress-IP bind, hardened CPU-only worker,
compensating teardown — lives in core ``umrs.hosts.inline``
(``InlineHostRuntime``, the ``inline`` launch mode). This module contributes
only what is r4's: the worker image and command (``r4 venue worker``), the
``R4_*`` environment the worker reads and the companion rail path.

``FrontendPodmanProvisioner``, ``ProvisionConfig``, ``render_nginx_conf`` and
``worker_run_args`` keep their pre-SH-H0 signatures and output (the
characterization test ``tests/test_hosts_runtime_inline.py`` pins them).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umrs.hosts.inline import (
    SELFTEST_SCRIPT,
    InlineHostRuntime,
    WorkerRunArgs,
    render_allowlist_file,
    render_squid_conf,
)
from umrs.hosts.inline import render_nginx_conf as _core_nginx_conf
from umrs.hosts.inline import worker_run_args as _core_worker_run_args
from umrs.hosts.runtime import (
    EGRESS_INFER_URL,
    SECRET_NAME_PREFIX,
    SESSION_DIR_ROOT,
    SESSION_TOKEN_PATH,
    PodmanRunner,
    ProvisionResult,
    ProvisionStepError,
    SandboxConfig,
    WorkerSpec,
    default_podman_runner,
)
from umrs.session_policy.model import normalize_allow_domain

__all__ = [
    "SECRET_NAME_PREFIX",
    "SELFTEST_SCRIPT",
    "SESSION_DIR_ROOT",
    "FrontendPodmanProvisioner",
    "PodmanRunner",
    "ProvisionConfig",
    "ProvisionResult",
    "ProvisionStepError",
    "WorkerRunArgs",
    "default_podman_runner",
    "normalize_allow_domain",
    "r4_worker_spec",
    "rail_path",
    "render_allowlist_file",
    "render_nginx_conf",
    "render_squid_conf",
    "worker_run_args",
    "worker_web_access",
]


@dataclass(frozen=True)
class ProvisionConfig(SandboxConfig):
    """The core sandbox config plus the r4 worker's image and command."""

    worker_image: str = "r4-companion@sha256:PINNED"
    # Command appended after the worker image (image ENTRYPOINT is `r4`).
    worker_command: tuple[str, ...] = ("venue", "worker")


def rail_path(session_id: str) -> str:
    """The companion rail prefix the sidecar forwards (wave 2026-09-21)."""
    return f"/umrs/companion/sessions/{session_id}/rail/"


def worker_web_access(cfg: SandboxConfig) -> str:
    """``R4_WEB_ACCESS`` for the worker: ``MODE[:d1,d2]``, the r4 grammar.

    The same choice squid enforces (``web_policy``, and for an allowlist its
    domains), so the worker's own web guard never claims more or less than
    the proxy allows (ADR-099 D-9, RS-1 fix W1).
    """
    if cfg.web_policy == "allowlist" and cfg.web_allow_domains:
        return f"allowlist:{','.join(cfg.web_allow_domains)}"
    return cfg.web_policy


def r4_worker_spec(cfg: SandboxConfig) -> WorkerSpec:
    """The r4 venue worker: image, command, the env it reads, its rail."""
    env = (
        ("R4_VENUE_SESSION_TOKEN_FILE", SESSION_TOKEN_PATH),
        # Worker loop identity + rail (wave 2026-09-21): references only.
        ("R4_VENUE_SESSION_ID", cfg.session_id),
        ("R4_VENUE_RAIL_URL", f"http://egress:{cfg.rail_port}"),
        ("R4_ENDPOINT_URL", EGRESS_INFER_URL),
        # ADR-099 D-9: the web access the sandbox was started with; the
        # worker's web tools follow it when the restored copy has no choice.
        ("R4_WEB_ACCESS", worker_web_access(cfg)),
        # Read-only rootfs: every writable path the r4 runtime needs lives on /tmp.
        ("HOME", "/tmp/home"),
        ("XDG_CONFIG_HOME", "/tmp/home/.config"),
        ("XDG_DATA_HOME", "/tmp/home/.local/share"),
        ("R4_COMPANION_DB", "/tmp/home/venue-sessions.db"),
    )
    return WorkerSpec(
        image=str(getattr(cfg, "worker_image", "")),
        command=tuple(getattr(cfg, "worker_command", ())),
        env=env,
        rail_path=rail_path(cfg.session_id),
    )


def render_nginx_conf(cfg: SandboxConfig) -> str:
    """The sidecar nginx.conf with the r4 companion rail location."""
    return _core_nginx_conf(cfg, rail_path=rail_path(cfg.session_id))


def worker_run_args(
    cfg: SandboxConfig, *, controllers: frozenset[str] | None = None
) -> WorkerRunArgs:
    """CPU-only sandbox argv for the r4 worker (core hardening + r4 spec)."""
    return _core_worker_run_args(cfg, r4_worker_spec(cfg), controllers=controllers)


class FrontendPodmanProvisioner(InlineHostRuntime):
    """The inline host runtime with the r4 worker spec (ADR-098 D-2)."""

    def __init__(
        self,
        broker: Any,
        podman: PodmanRunner = default_podman_runner,
        session_root: Path = SESSION_DIR_ROOT,
    ) -> None:
        super().__init__(
            broker, r4_worker_spec, podman=podman, session_root=session_root
        )
