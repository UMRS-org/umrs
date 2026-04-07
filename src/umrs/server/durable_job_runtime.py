"""Process-owned durable jobs runtime for the synchronous UMRS server."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Optional

from umrs.orchestrator.durable import DurableOrchestrationService
from umrs.server.job_tracking_handler import (
    DurableJobCommandBackend,
    build_durable_job_db_pool_from_env,
)

logger = logging.getLogger(__name__)


class DurableJobRuntimeError(RuntimeError):
    """Raised when the process-owned durable jobs runtime cannot start or stop."""


class DurableJobRuntime:
    """Own a DurableOrchestrationService on a dedicated asyncio loop thread."""

    def __init__(self, db_pool, service: Optional[DurableOrchestrationService] = None):
        self._db_pool = db_pool
        self._service = service or DurableOrchestrationService(db_pool=db_pool)
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._command_backend: Optional[DurableJobCommandBackend] = None
        self._startup_complete = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._started = False

    @property
    def command_backend(self) -> DurableJobCommandBackend:
        if self._command_backend is None:
            raise RuntimeError("DurableJobRuntime not started")
        return self._command_backend

    def start(self, timeout: float = 30.0) -> DurableJobCommandBackend:
        with self._lock:
            if self._started:
                return self.command_backend

            self._startup_complete.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._run_loop,
                name="umrs-durable-jobs",
                daemon=True,
            )
            self._thread.start()

        if not self._startup_complete.wait(timeout):
            self.shutdown()
            raise DurableJobRuntimeError(
                f"Timed out starting durable job runtime after {timeout:.1f}s"
            )

        if self._startup_error is not None:
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            self._close_pool()
            raise DurableJobRuntimeError("Failed to start durable job runtime") from self._startup_error

        self._started = True
        return self.command_backend

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._lock:
            thread = self._thread
            loop = self._loop
            startup_error = self._startup_error
            started = self._started
            self._started = False

        try:
            if started and loop is not None and startup_error is None:
                future = asyncio.run_coroutine_threadsafe(
                    self._service.shutdown(timeout),
                    loop,
                )
                future.result(timeout + 5.0)
        except FutureTimeoutError as exc:
            raise DurableJobRuntimeError("Timed out shutting down durable job runtime") from exc
        finally:
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(loop.stop)
                except RuntimeError:
                    pass
            if thread is not None:
                thread.join(timeout + 5.0)
            self._thread = None
            self._command_backend = None
            self._close_pool()

    async def _startup(self) -> None:
        await self._service.startup()
        self._command_backend = DurableJobCommandBackend(
            self._service.job_service.job_store,
        )

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        try:
            try:
                loop.run_until_complete(self._startup())
            except BaseException as exc:
                self._startup_error = exc
                self._startup_complete.set()
                return

            self._startup_complete.set()
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True),
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._loop = None

    def _close_pool(self) -> None:
        if self._db_pool is None:
            return
        try:
            self._db_pool.closeall()
        except Exception:
            logger.exception("Failed to close durable jobs database pool")
        finally:
            self._db_pool = None


def start_durable_job_runtime_from_env(timeout: float = 30.0) -> Optional[DurableJobRuntime]:
    """Best-effort bootstrap of the process-owned durable jobs runtime."""
    try:
        db_pool = build_durable_job_db_pool_from_env()
        runtime = DurableJobRuntime(db_pool=db_pool)
        runtime.start(timeout=timeout)
        logger.info("Durable job runtime started")
        return runtime
    except Exception:
        logger.exception("Failed to start durable job runtime")
        return None