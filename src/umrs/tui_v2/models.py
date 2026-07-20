"""Data models for background operations (upload, download, workflow, experience)."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Progress:
    """Progress state for a single background operation."""

    percent: float = 0.0
    current: int = 0
    total: int = 0
    message: str = ""
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None

    @property
    def bar(self) -> str:
        width = 30
        filled = int(width * self.percent / 100) if self.percent > 0 else 0
        return "█" * filled + "░" * (width - filled)


@dataclass
class Operation:
    """A single background operation (upload, download, workflow run, etc.)."""

    operation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation_type: str = (
        ""  # upload, download, workflow_run, experience_run, workflow_register
    )
    title: str = ""  # Display title (e.g. file path, workflow name)
    status: OperationStatus = OperationStatus.PENDING
    progress: Progress = field(default_factory=Progress)
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    _progress_fn: Any = field(default=None, repr=False)  # Callable to update progress

    @property
    def running(self) -> bool:
        return self.status == OperationStatus.RUNNING

    @property
    def completed(self) -> bool:
        return self.status in (
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
        )

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.monotonic()
        if self.started_at is None:
            return 0.0
        return end - self.started_at

    def start(self, progress_fn: Any | None = None) -> None:
        self.status = OperationStatus.RUNNING
        self.started_at = time.monotonic()
        self.progress.percent = 0.0
        self.progress.message = "Starting…"
        if progress_fn is not None:
            self._progress_fn = progress_fn

    def mark_progress(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self.progress, k):
                setattr(self.progress, k, v)
        if self._progress_fn:
            self._progress_fn(self)

    def mark_done(
        self,
        status: OperationStatus,
        error: str | None = None,
        result: dict | None = None,
    ) -> None:
        self.status = status
        self.finished_at = time.monotonic()
        self.progress.percent = 100.0
        if status == OperationStatus.SUCCEEDED:
            self.progress.message = "Done"
        elif status == OperationStatus.CANCELLED:
            self.progress.message = "Cancelled"
        else:
            self.progress.message = error or "Failed"
            self.error = error
        self.result = result

    def cancel(self) -> None:
        if self.status == OperationStatus.RUNNING:
            self.status = OperationStatus.CANCELLED
            self.finished_at = time.monotonic()
            self.progress.message = "Cancelled"


@dataclass
class OperationResult:
    """Result of a completed background operation."""

    operation_id: str
    operation_type: str
    status: OperationStatus
    message: str
    data: dict[str, Any] | None = None


class OperationQueue:
    """Thread-safe queue for background operations with progress updates."""

    def __init__(self) -> None:
        self._operations: dict[str, Operation] = {}
        self._callbacks: list[Any] = []  # UI callbacks for progress updates
        self._tasks: dict[str, asyncio.Task] = {}  # Running tasks by op_id

    @property
    def operations(self) -> list[Operation]:
        return list(self._operations.values())

    def active(self) -> list[Operation]:
        """Return currently running operations."""
        return [op for op in self._operations.values() if op.running]

    def completed(self) -> list[Operation]:
        """Return completed operations (succeeded/failed/cancelled)."""
        return [op for op in self._operations.values() if op.completed]

    def add(self, operation: Operation) -> None:
        self._operations[operation.operation_id] = operation

    def get(self, op_id: str) -> Operation | None:
        return self._operations.get(op_id)

    def cancel(self, op_id: str) -> bool:
        op = self._operations.get(op_id)
        if op and op.running:
            op.cancel()
            # Cancel the background task if running
            task = self._tasks.pop(op_id, None)
            if task:
                task.cancel()
            return True
        return False

    def cancel_all(self) -> int:
        count = 0
        for op_id in list(self._tasks.keys()):
            self.cancel(op_id)
            count += 1
        return count

    def cleanup(self, max_completed: int = 20) -> None:
        """Remove old completed operations to prevent memory leak."""
        completed = [op for op in self._operations.values() if op.completed]
        if len(completed) > max_completed:
            oldest = completed[:-max_completed]
            for op in oldest:
                del self._operations[op.operation_id]

    def register_callback(self, callback: Any) -> None:
        """Register a callback to be called on progress updates."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Any) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError:
            pass

    def notify_progress(self, operation: Operation) -> None:
        """Notify all registered callbacks about progress update."""
        for cb in self._callbacks:
            try:
                cb(operation)
            except Exception:
                pass  # Silent failure — UI callbacks should not crash the queue

    async def _run_operation(self, operation: Operation, coro_factory) -> None:
        """Internal: run a background operation coroutine."""
        try:
            await coro_factory(operation)
            if operation.status == OperationStatus.RUNNING:
                operation.mark_done(OperationStatus.SUCCEEDED, result=operation.result)
        except asyncio.CancelledError:
            operation.mark_done(OperationStatus.CANCELLED)
        except Exception as e:
            operation.mark_done(OperationStatus.FAILED, error=str(e))
        finally:
            self._tasks.pop(operation.operation_id, None)
            self.cleanup()
            self.notify_progress(operation)

    def start_background(self, operation: Operation, coro_factory) -> None:
        """Start a background coroutine for this operation."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(self._run_operation(operation, coro_factory))
        self._tasks[operation.operation_id] = task
