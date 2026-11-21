"""Loop iteration kinds an extension owns (U3 S3c, ADR-117 §6, u3a map §3 hook 4).

The engine loop is core (D1). An extension may add another iteration executor
(r4: a team plan run per iteration, ADR-092 slice 8b) through
``ExtensionContext.add_loop_iteration_kind``. Core resolves the kind here at
three seams: the start route, the thread host (start and re-host after a
restart) and the job-hosted driver.

Without the owning extension, a start that asks for the kind and a persisted
loop that carries its metadata are refused by name: never run as a plain
engine loop with the extension's part silently dropped.
"""

from __future__ import annotations

from typing import Any

# Persisted loop metadata key -> owning extension.
EXTENSION_ITERATION_KEYS: dict[str, str] = {"iteration_plan": "r4"}
# Kind name -> the StartLoopRequest field that asks for it.
START_FIELDS: dict[str, str] = {"plan": "iteration_plan"}


class IterationKindUnavailable(RuntimeError):
    """No enabled extension provides the iteration kind a loop needs."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _kinds() -> dict[str, Any]:
    from umrs.extensions import get_extension_context

    return get_extension_context().loop_iteration_kinds


def kind_for_loop(loop: Any) -> Any:
    """The kind driving *loop* (None = the core engine)."""
    for kind in _kinds().values():
        if kind.matches(loop):
            return kind
    metadata = getattr(loop, "metadata", None) or {}
    for key, extension in EXTENSION_ITERATION_KEYS.items():
        if metadata.get(key):
            raise IterationKindUnavailable(
                key,
                f"loop {getattr(loop, 'loop_id', '?')} iterates by {key!r}, which "
                f"extension {extension!r} provides; it is not enabled on this server",
            )
    return None


def kind_for_start(request: Any) -> Any:
    """The kind a start request asks for (None = the core engine)."""
    for name, field in START_FIELDS.items():
        if not str(getattr(request, field, None) or "").strip():
            continue
        kind = _kinds().get(name)
        if kind is None:
            raise IterationKindUnavailable(
                name,
                f"{field}: iteration kind {name!r} needs an extension that is "
                "not enabled on this server",
            )
        return kind
    return None
