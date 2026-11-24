"""Session and MCP hosts (ADR-098): the host runtime, core UMRS.

``HostRuntime`` owns pod and egress provisioning on a host; an extension
adds only a ``WorkerSpec``. ``InlineHostRuntime`` is the ``inline`` launch
mode (the UMRS process on its own host), today's only mode. Nothing in this
package imports the r4 extension (ADR-117 D-3).
"""

from umrs.hosts.inline import InlineHostRuntime, WorkerRunArgs, worker_run_args
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
    "INLINE_HOST_ID",
    "LAUNCH_MODES",
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
