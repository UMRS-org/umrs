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

from typing import Any, Callable

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


__all__ = ["get_execution_registry", "register_execution_registry_provider"]


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
