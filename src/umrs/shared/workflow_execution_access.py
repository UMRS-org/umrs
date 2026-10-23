"""Foundation-facing access to the workflow-execution registry (ADR-053).

The workflow execution registry's OWNER is `umrs.server`; foundation modules
(jobs/workflow_run) previously lazily imported it — a back-edge this wave
closes via the provider pattern: the server registers the factory at import
time; jobs call `get_execution_registry()` from HERE and never reach into
`umrs.server`.

Error honesty: calling early before registration is a loud RuntimeError,
never a silent None.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_provider: Callable[[], Any] | None = None


def register_execution_registry_provider(provider: Callable[[], Any]) -> None:
    """Install the registry factory. Called by umrs.server on import."""
    global _provider
    _provider = provider


def get_execution_registry() -> Any:
    """Return the server's workflow execution registry (registered once)."""
    if _provider is None:
        raise RuntimeError(
            "workflow execution registry unavailable: umrs.server has not "
            "registered its provider (is the sidecar booted?)"
        )
    return _provider()


__all__ = [
    "VERIFIED_PROJECT_SPEC_KEY",
    "durable_job_verified_project",
    "get_execution_registry",
    "register_execution_registry_provider",
]


# Registry-handler delegate (workflow_run's backend path) — same provider
# pattern: jobs calls _launch_workflow_from_catalog() from HERE, never
# reaching umrs.server.workflow_registry_handler.
_handler: Any | None = None


def register_workflow_handler(handler: Any) -> None:
    """Install the workflow launch handler from umrs.server."""
    global _handler
    _handler = handler


def _launch_workflow_from_catalog(**kwargs: Any) -> Any:
    """Delegate to the server's catalog launcher (registered once)."""
    if _handler is None:
        raise RuntimeError(
            "workflow registry handler unavailable: umrs.server has not "
            "registered its launcher (is the sidecar booted?)"
        )
    return _handler(**kwargs)


# ADR-080 on the durable path (2026-10-23): a ``workflow_run`` job spec is
# built only by the server (both client submit lanes refuse the job type),
# and it records the job's project under this key only when that project was
# gateway-verified at trigger time (an HTTP trigger with a signed identity, a
# research loop started in its starter's verified project). A re-dispatch
# that must re-resolve the workflow (no usable ``source_path``) admits a
# ``project:`` share only through that record — never the bare
# ``jobs.project_id``, which the ZMQ trigger lane takes from a payload claim.
VERIFIED_PROJECT_SPEC_KEY = "verified_project_id"


def durable_job_verified_project(project_id: Any, spec: Any) -> str | None:
    """The job's project when its spec records it as verified, else None."""
    if not isinstance(spec, dict):
        return None
    recorded = spec.get(VERIFIED_PROJECT_SPEC_KEY)
    project = str(project_id or "").strip()
    if not isinstance(recorded, str) or not project or project == "default":
        return None
    return project if recorded.strip() == project else None
