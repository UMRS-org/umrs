"""The minimal ExtensionContext (ADR-117 §6, U3a map §3).

Internal, versioned and deliberately small: it holds only the six hooks the
umrs-r4 extension needs. It is not a public plugin API until a second
extension exists. Each hook only records a registration; the core seam that
owns it consumes the record (``umrs.extensions.seams``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Bumped on any incompatible hook change; an extension declares the version it
# targets as ``EXTENSION_API_VERSION`` on its module, and a mismatch fails boot.
EXTENSION_API_VERSION = 1


class ExtensionError(RuntimeError):
    """A broken extension registration: fails boot, never runs half-wired."""


@dataclass(frozen=True)
class RouterRegistration:
    extension: str
    router: Any
    family: str
    lane: str | None = None


@dataclass(frozen=True)
class JobTypeRegistration:
    """A durable job type an extension owns (seam: the durable lanes, S3d).

    ``build_lane(db_pool)`` returns the lane's shared ``(backend, checker)``
    pair, or None when the lane cannot be wired on this process.
    """

    value: str
    build_lane: Callable[[Any], tuple[Any, Any] | None]
    backoff: float | None = None


@dataclass(frozen=True)
class LoopIterationKind:
    """A non-engine loop iteration (seam: api/loop.py start + rehost, S3e)."""

    name: str
    matches: Callable[[Any], bool]
    validate_start: Callable[[Any, Any, str], None]
    attach: Callable[[Any, Any], None]
    engine_factory: Callable[[], Any]


@dataclass
class ExtensionContext:
    """Registrations collected from every enabled extension, in load order."""

    routers: list[RouterRegistration] = field(default_factory=list)
    job_types: dict[str, JobTypeRegistration] = field(default_factory=dict)
    startups: list[tuple[str, Callable[[Any], None]]] = field(default_factory=list)
    loop_iteration_kinds: dict[str, LoopIterationKind] = field(default_factory=dict)
    session_archive_participants: list[Callable[[Any, str, str], None]] = field(
        default_factory=list
    )
    approval_listeners: list[Callable[[dict, str, str], None]] = field(
        default_factory=list
    )
    active: list[str] = field(default_factory=list)
    _current: str = "core"

    def bind(self, extension: str) -> ExtensionContext:
        """Attribute the following registrations to *extension*."""
        self._current = extension
        return self

    def add_router(self, router: Any, family: str, lane: str | None = None) -> None:
        self.routers.append(RouterRegistration(self._current, router, family, lane))

    def add_job_type(self, registration: JobTypeRegistration) -> None:
        if registration.value in self.job_types:
            raise ExtensionError(f"job type {registration.value!r} registered twice")
        self.job_types[registration.value] = registration

    def add_startup(self, fn: Callable[[Any], None]) -> None:
        self.startups.append((self._current, fn))

    def add_loop_iteration_kind(self, kind: LoopIterationKind) -> None:
        if kind.name in self.loop_iteration_kinds:
            raise ExtensionError(f"loop iteration kind {kind.name!r} registered twice")
        self.loop_iteration_kinds[kind.name] = kind

    def add_session_archive_participant(
        self, fn: Callable[[Any, str, str], None]
    ) -> None:
        self.session_archive_participants.append(fn)

    def add_approval_listener(self, fn: Callable[[dict, str, str], None]) -> None:
        self.approval_listeners.append(fn)
