"""
B1 — Streaming Progress Tracker

Real-time progress reporting for long-running UMRS operations.
Provides callback-based and polling-based APIs for tracking upload,
pipeline, and inference progress.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ProgressState(str, Enum):
    """Progress lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressEvent:
    """A single progress update event."""

    tracker_id: str
    state: ProgressState
    current: int
    total: int
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(self.current / self.total, 1.0)

    @property
    def percent(self) -> float:
        return self.fraction * 100.0


class ProgressCallback(Protocol):
    """Protocol for progress consumers."""

    def on_progress(self, event: ProgressEvent) -> None: ...


class ProgressTracker:
    """Thread-safe progress state manager.

    Tracks current/total counts, state transitions, and notifies
    registered callbacks on every update.

    Usage::

        tracker = ProgressTracker("upload-123", total=100)
        tracker.add_callback(my_callback)
        tracker.start()
        for i in range(100):
            tracker.advance(1, message=f"Uploading file {i+1}")
        tracker.complete("All files uploaded")
    """

    def __init__(self, tracker_id: str, total: int = 0) -> None:
        self.tracker_id = tracker_id
        self._total = max(total, 0)
        self._current = 0
        self._state = ProgressState.PENDING
        self._message = ""
        self._metadata: dict[str, Any] = {}
        self._callbacks: list[ProgressCallback] = []
        self._lock = threading.Lock()
        self._history: list[ProgressEvent] = []

    @property
    def state(self) -> ProgressState:
        with self._lock:
            return self._state

    @property
    def current(self) -> int:
        with self._lock:
            return self._current

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    @total.setter
    def total(self, value: int) -> None:
        with self._lock:
            self._total = max(value, 0)

    @property
    def fraction(self) -> float:
        with self._lock:
            if self._total <= 0:
                return 0.0
            return min(self._current / self._total, 1.0)

    def add_callback(self, callback: ProgressCallback) -> None:
        with self._lock:
            self._callbacks.append(callback)

    def remove_callback(self, callback: ProgressCallback) -> None:
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c is not callback]

    def _emit(self) -> ProgressEvent:
        """Create and dispatch a progress event (caller must hold lock)."""
        event = ProgressEvent(
            tracker_id=self.tracker_id,
            state=self._state,
            current=self._current,
            total=self._total,
            message=self._message,
            metadata=dict(self._metadata),
        )
        self._history.append(event)
        for cb in self._callbacks:
            try:
                cb.on_progress(event)
            except Exception:
                pass  # Don't let callback errors break progress tracking
        return event

    def start(self, message: str = "Started") -> ProgressEvent:
        with self._lock:
            self._state = ProgressState.RUNNING
            self._message = message
            return self._emit()

    def advance(self, delta: int = 1, message: str = "") -> ProgressEvent:
        with self._lock:
            self._current = (
                min(self._current + delta, self._total)
                if self._total > 0
                else self._current + delta
            )
            if message:
                self._message = message
            return self._emit()

    def set_progress(self, current: int, message: str = "") -> ProgressEvent:
        with self._lock:
            self._current = current
            if message:
                self._message = message
            return self._emit()

    def complete(self, message: str = "Completed") -> ProgressEvent:
        with self._lock:
            self._state = ProgressState.COMPLETED
            self._current = self._total
            self._message = message
            return self._emit()

    def fail(self, message: str = "Failed") -> ProgressEvent:
        with self._lock:
            self._state = ProgressState.FAILED
            self._message = message
            return self._emit()

    def cancel(self, message: str = "Cancelled") -> ProgressEvent:
        with self._lock:
            self._state = ProgressState.CANCELLED
            self._message = message
            return self._emit()

    def get_history(self) -> list[ProgressEvent]:
        with self._lock:
            return list(self._history)

    def set_metadata(self, key: str, value: Any) -> None:
        with self._lock:
            self._metadata[key] = value

    def is_active(self) -> bool:
        with self._lock:
            return self._state in (ProgressState.PENDING, ProgressState.RUNNING)


class ConsoleProgressBar:
    """Terminal progress bar renderer.

    Implements the ProgressCallback protocol to display a text-based
    progress bar on stdout.

    Usage::

        bar = ConsoleProgressBar(width=40)
        tracker.add_callback(bar)
    """

    def __init__(self, width: int = 40, file: Any = None) -> None:
        self.width = width
        self.file = file or sys.stderr

    def on_progress(self, event: ProgressEvent) -> None:
        if event.state == ProgressState.COMPLETED:
            self._render(event)
            print(file=self.file)
            return
        if event.state in (ProgressState.FAILED, ProgressState.CANCELLED):
            self._render(event)
            print(file=self.file)
            return
        self._render(event)

    def _render(self, event: ProgressEvent) -> None:
        pct = event.percent
        filled = int(self.width * event.fraction)
        bar = "█" * filled + "░" * (self.width - filled)
        msg = event.message[:30] if event.message else ""
        line = f"\r[{bar}] {pct:5.1f}% {event.current}/{event.total} {msg}"
        print(line, end="", file=self.file, flush=True)


class CallbackAdapter:
    """Wraps a plain function as a ProgressCallback."""

    def __init__(self, fn: Callable[[ProgressEvent], None]) -> None:
        self._fn = fn

    def on_progress(self, event: ProgressEvent) -> None:
        self._fn(event)


class ProgressGroup:
    """Manage multiple ProgressTrackers as a group.

    Aggregates progress across sub-trackers for overall completion tracking.

    Usage::

        group = ProgressGroup("batch-upload")
        t1 = group.create_tracker("file1", total=100)
        t2 = group.create_tracker("file2", total=200)
        # group.fraction reflects overall progress
    """

    def __init__(self, group_id: str) -> None:
        self.group_id = group_id
        self._trackers: dict[str, ProgressTracker] = {}
        self._lock = threading.Lock()

    def create_tracker(self, tracker_id: str, total: int = 0) -> ProgressTracker:
        with self._lock:
            tracker = ProgressTracker(tracker_id, total)
            self._trackers[tracker_id] = tracker
            return tracker

    def get_tracker(self, tracker_id: str) -> ProgressTracker | None:
        with self._lock:
            return self._trackers.get(tracker_id)

    @property
    def trackers(self) -> dict[str, ProgressTracker]:
        with self._lock:
            return dict(self._trackers)

    @property
    def fraction(self) -> float:
        with self._lock:
            if not self._trackers:
                return 0.0
            total_sum = sum(t.total for t in self._trackers.values())
            if total_sum <= 0:
                return 0.0
            current_sum = sum(t.current for t in self._trackers.values())
            return min(current_sum / total_sum, 1.0)

    @property
    def all_completed(self) -> bool:
        with self._lock:
            return all(
                t.state == ProgressState.COMPLETED for t in self._trackers.values()
            )

    @property
    def any_failed(self) -> bool:
        with self._lock:
            return any(t.state == ProgressState.FAILED for t in self._trackers.values())
