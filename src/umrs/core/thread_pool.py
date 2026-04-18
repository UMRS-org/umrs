"""Thread pool with backpressure for the UMRS server.

Replaces unbounded ``threading.Thread`` spawning with a bounded
``ThreadPoolExecutor``.  When the pool is saturated (all workers busy
**and** the pending-work queue is full), ``submit()`` raises
``ServerOverloadedError`` so the caller can reject the request
immediately rather than queueing unbounded work.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MAX_WORKERS = 16
_DEFAULT_MAX_PENDING = 64


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default


@dataclass(frozen=True)
class PoolConfig:
    """Immutable configuration for the server thread pool."""

    max_workers: int = field(
        default_factory=lambda: _env_int("UMRS_POOL_MAX_WORKERS", _DEFAULT_MAX_WORKERS)
    )
    max_pending: int = field(
        default_factory=lambda: _env_int("UMRS_POOL_MAX_PENDING", _DEFAULT_MAX_PENDING)
    )

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        if self.max_pending < 0:
            raise ValueError(f"max_pending must be >= 0, got {self.max_pending}")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ServerOverloadedError(Exception):
    """Raised when the thread pool cannot accept more work."""


# ---------------------------------------------------------------------------
# Pool metrics (read-only snapshot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolMetrics:
    """Point-in-time snapshot of pool utilisation."""

    max_workers: int
    max_pending: int
    active_count: int
    pending_count: int

    @property
    def total_queued(self) -> int:
        return self.active_count + self.pending_count

    @property
    def is_saturated(self) -> bool:
        return self.pending_count >= self.max_pending

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "max_pending": self.max_pending,
            "active_count": self.active_count,
            "pending_count": self.pending_count,
            "total_queued": self.total_queued,
            "is_saturated": self.is_saturated,
        }


# ---------------------------------------------------------------------------
# ThreadPoolManager
# ---------------------------------------------------------------------------


class ThreadPoolManager:
    """Bounded thread pool with backpressure.

    Parameters
    ----------
    config : PoolConfig, optional
        Pool sizing.  Defaults are read from ``UMRS_POOL_MAX_WORKERS``
        and ``UMRS_POOL_MAX_PENDING`` environment variables, falling
        back to 16 workers / 64 pending.
    """

    def __init__(self, config: Optional[PoolConfig] = None) -> None:
        self._config = config or PoolConfig()
        self._executor = ThreadPoolExecutor(
            max_workers=self._config.max_workers,
            thread_name_prefix="umrs-pool",
        )
        # Semaphore gates admission: permits = max_workers + max_pending.
        # Each submit() acquires one; each task completion releases one.
        self._gate = threading.Semaphore(
            self._config.max_workers + self._config.max_pending
        )
        self._active_count = 0
        self._pending_count = 0
        self._lock = threading.Lock()
        self._shutdown = False
        logger.info(
            "ThreadPoolManager started: max_workers=%d, max_pending=%d",
            self._config.max_workers,
            self._config.max_pending,
        )

    # -- public API ---------------------------------------------------------

    def submit(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Future:
        """Submit work to the pool.

        Raises ``ServerOverloadedError`` if the pool is saturated
        (all workers busy and pending queue full).
        """
        if self._shutdown:
            raise RuntimeError("ThreadPoolManager has been shut down")

        # Non-blocking acquire: if no permits left, reject immediately.
        if not self._gate.acquire(blocking=False):
            metrics = self.metrics()
            raise ServerOverloadedError(
                f"Server overloaded: {metrics.active_count} active, "
                f"{metrics.pending_count} pending "
                f"(limits: {metrics.max_workers}/{metrics.max_pending})"
            )

        with self._lock:
            self._pending_count += 1

        future = self._executor.submit(self._wrapped, fn, *args, **kwargs)
        return future

    def metrics(self) -> PoolMetrics:
        """Return a point-in-time snapshot of pool utilisation."""
        with self._lock:
            return PoolMetrics(
                max_workers=self._config.max_workers,
                max_pending=self._config.max_pending,
                active_count=self._active_count,
                pending_count=self._pending_count,
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool, optionally waiting for in-flight work."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("ThreadPoolManager shut down (wait=%s)", wait)

    # -- internals ----------------------------------------------------------

    def _wrapped(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Wrapper that maintains counters and releases the gate permit."""
        with self._lock:
            self._pending_count -= 1
            self._active_count += 1

        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self._active_count -= 1
            self._gate.release()
