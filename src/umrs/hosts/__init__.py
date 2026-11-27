"""Session and MCP hosts (ADR-098): the host runtime, core UMRS.

``HostRuntime`` owns pod and egress provisioning on a host; an extension
adds only a ``WorkerSpec``. ``InlineHostRuntime`` is the ``inline`` launch
mode (the UMRS process on its own host), today's only mode. Nothing in this
package imports the r4 extension (ADR-117 D-3).

``HostRegistry`` (ADR-098 D-4, SH-H1a) is the per-site ``hosts`` table the
inline front registers itself in; ``umrs.hosts.janitor`` heartbeats it and
ends the sessions of hosts that miss their heartbeats (``host_lost``).

``umrs.hosts.identity`` and ``umrs.hosts.enroll`` (ADR-098 D-5/D-6, SH-H1b)
hold agent-host identity: one-time enrollment tokens, enrolled ed25519
public keys, short-lived signed host tokens and the host binding of leases.
"""

from umrs.hosts.inline import InlineHostRuntime, WorkerRunArgs, worker_run_args
from umrs.hosts.registry import (
    HOST_STATES,
    PLACEABLE_STATES,
    HostRecord,
    HostRegistry,
    HostRegistryError,
)
from umrs.hosts.runtime import (
    INLINE_HOST_ID,
    LAUNCH_MODES,
    HostRuntime,
    ProvisionResult,
    ProvisionStepError,
    SandboxConfig,
    WorkerSpec,
    WorkerSpecFn,
)

__all__ = [
    "HOST_STATES",
    "INLINE_HOST_ID",
    "LAUNCH_MODES",
    "PLACEABLE_STATES",
    "HostRecord",
    "HostRegistry",
    "HostRegistryError",
    "HostRuntime",
    "InlineHostRuntime",
    "ProvisionResult",
    "ProvisionStepError",
    "SandboxConfig",
    "WorkerRunArgs",
    "WorkerSpec",
    "WorkerSpecFn",
    "worker_run_args",
]
