"""The host runtime contract (ADR-098 D-1..D-3, decision SH-D3).

A *host* is a machine UMRS places sandboxes on (session workers today, MCP
server instances later). ``HostRuntime`` owns what happens on that host: the
session's internal network, the egress pod (inference listener, rail
listener, web proxy), the lease token secret and the hardened worker
container. An extension contributes only a ``WorkerSpec`` (image, command,
the environment its worker reads); it never shapes the pod or the egress.

Launch modes: ``inline`` (the UMRS process drives its own host's podman; the
default, single node and stand-alone) and, later, ``agent`` (``umrs-hostd``
on another machine, SH-H3). Only ``inline`` exists today.

Core: nothing here imports the r4 extension (ADR-117 D-3).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

# The host id of the UMRS process's own host (inline mode). SH-H1a's
# registry row for the inline front carries this id.
INLINE_HOST_ID = "local"
LAUNCH_MODES = ("inline",)  # "agent" lands with SH-H3

SESSION_DIR_ROOT = Path("/var/lib/umrs/sessions")
SECRET_NAME_PREFIX = "sess-"
# Where the worker container finds its lease token (a podman secret mount).
SESSION_TOKEN_PATH = "/run/secrets/session_token"
# The egress pod's inference listener, as the worker reaches it.
EGRESS_INFER_URL = "http://egress:8080"

PodmanRunner = Callable[..., subprocess.CompletedProcess]


def default_podman_runner(
    *args: str, input_text: str | None = None
) -> subprocess.CompletedProcess:
    """Production runner: podman CLI on PATH, 30 s step timeout."""
    return subprocess.run(  # argv list, no shell
        ["podman", *args],
        capture_output=True,
        text=True,
        input=input_text,
        timeout=30,
        check=False,
    )


@dataclass(frozen=True)
class SandboxConfig:
    """One session's provisioning inputs (references only — no secrets).

    The worker's image and command are not here: they come from the
    extension's ``WorkerSpec``.
    """

    session_id: str
    owner: str
    source_ids: tuple[str, ...]
    default_source_id: str
    subnet: str  # pool-allocated, e.g. "172.18.52.0/24"
    infer_host: str  # inference proxy DNS/IP on the site network
    infer_port: int = 8443
    nginx_image: str = "docker.io/nginx@sha256:PINNED"
    squid_image: str = "docker.io/ubuntu/squid@sha256:PINNED"
    web_policy: str = "public"  # public | allowlist | none
    web_allow_domains: tuple[str, ...] = ()  # EgressGuard format (R2-7); allowlist mode
    # Inference uplink: "https+internal-ca" (production target, cert bundled)
    # or "http" (bootstrap: umrs-infer plain loopback in the trust zone).
    infer_uplink: str = "https"
    internal_ca_path: str = "/etc/umrs/internal-ca.pem"
    # Override nginx's resolver with an explicit bridge-DNS address (live-leg
    # motivation: infer host comes in as a name on a bridge netns; without it
    # stock nginx can't re-resolve per request). "" = no resolver directive.
    infer_resolve_via: str = ""
    lease_seconds: int = 3600
    max_lifetime_seconds: int = 7 * 86_400
    worker_memory: str = "16g"
    worker_cpus: str = "8"
    # The worker image's USER (r4/Dockerfile: r4user, uid 1000). The session
    # token secret must be owned by it: a podman `type=mount` secret without
    # uid/gid is root-owned, and at mode 0400 the worker could not read it
    # (every front-end venue worker exited 4, "session token file unreadable",
    # live 2026-09-25).
    worker_uid: int = 1000
    project: str = ""
    gateway_host: str = ""
    umrs_host: str = ""
    # Worker rail (wave 2026-09-21): the sidecar proxies the worker's journal
    # calls (queue pop, snapshot mirror, heartbeat, approvals) to this API
    # host:port, path-anchored to the spec's ``rail_path``.
    # "" = no rail listener rendered (bootstrap legs without a worker loop).
    rail_upstream: str = ""
    rail_port: int = 8081


@dataclass(frozen=True)
class WorkerSpec:
    """What an extension's worker needs; the runtime adds the hardening.

    ``env`` pairs follow the runtime's egress variables, in order;
    ``rail_path`` is the API path prefix the sidecar's rail listener
    forwards ("" = no rail location, even with a rail upstream).
    """

    image: str
    command: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    rail_path: str = ""


WorkerSpecFn = Callable[[SandboxConfig], WorkerSpec]


@dataclass
class ProvisionResult:
    session_id: str
    state: str  # "running" | "failed"
    reason: str = ""
    lease_id: str = ""
    egress_ip: str = ""
    steps: list[str] = field(default_factory=list)


class ProvisionStepError(RuntimeError):
    """One step failed; the runtime tears down and reports step+detail."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


@runtime_checkable
class HostRuntime(Protocol):
    """Provision and tear down one session's sandbox on a host."""

    def provision(self, cfg: SandboxConfig) -> ProvisionResult: ...

    def teardown(self, session_id: str) -> None: ...
