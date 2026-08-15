"""Process-owned durable jobs runtime for the synchronous UMRS server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

from umrs.jobs.models import JobType
from umrs.metrics import METRICS
from umrs.orchestrator.durable import DurableOrchestrationService
from umrs.server.job_tracking_handler import (
    DurableJobCommandBackend,
    build_durable_job_db_pool_from_env,
)

logger = logging.getLogger(__name__)

_BRIDGE_NAME = "gen1_gen2_runtime"

# Module-level reference for sidecar access (Task 2-delta)
_global_runtime: DurableJobRuntime | None = None


def _set_runtime(runtime: DurableJobRuntime | None) -> None:
    """Store or clear the module-level runtime reference."""
    global _global_runtime
    _global_runtime = runtime


def get_runtime() -> DurableJobRuntime | None:
    """Return the running DurableJobRuntime instance, or None."""
    return _global_runtime


class DurableJobRuntimeError(RuntimeError):
    """Raised when the process-owned durable jobs runtime cannot start or stop."""


class DurableJobRuntime:
    """Own a DurableOrchestrationService on a dedicated asyncio loop thread."""

    def __init__(self, db_pool, service: DurableOrchestrationService | None = None):
        self._db_pool = db_pool
        self._service = service or DurableOrchestrationService(db_pool=db_pool)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._command_backend: DurableJobCommandBackend | None = None
        self._startup_complete = threading.Event()
        self._startup_error: BaseException | None = None
        self._lock = threading.Lock()
        self._started = False
        self._shutdown_called = False
        # Task 2-delta: liveness heartbeat (updated by background task)
        self._heartbeat_ts: float = 0.0
        self._heartbeat_interval: float = 10.0  # seconds
        self._heartbeat_task: asyncio.Task | None = None

    @property
    def command_backend(self) -> DurableJobCommandBackend:
        if self._command_backend is None:
            raise RuntimeError("DurableJobRuntime not started")
        return self._command_backend

    async def submit_job_sync(
        self,
        *,
        job_type: JobType,
        project_id: str,
        spec: dict,
        user_id: str = "",
    ) -> str:
        """Submit a durable job from async HTTP handlers (compatibility shim).

        The knowledge ingest/export routes call this exact signature. The
        orchestration service is synchronous, so the submission runs in the
        default executor to keep the event loop unblocked.
        """
        return await asyncio.to_thread(
            self._service.submit_job,
            job_type=job_type,
            project_id=project_id,
            user_id=user_id,
            spec=spec,
        )

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
            raise DurableJobRuntimeError(
                "Failed to start durable job runtime"
            ) from self._startup_error

        self._started = True
        METRICS.bridge_event(_BRIDGE_NAME, "start")
        return self.command_backend

    def shutdown(self, timeout: float = 10.0) -> None:
        with self._lock:
            if self._shutdown_called:
                logger.debug("DurableJobRuntime.shutdown() already called; skipping")
                return
            self._shutdown_called = True
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
            METRICS.bridge_event(_BRIDGE_NAME, "shutdown_timeout")
            raise DurableJobRuntimeError(
                "Timed out shutting down durable job runtime"
            ) from exc
        finally:
            METRICS.bridge_event(_BRIDGE_NAME, "stop")
            if loop is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(loop.stop)
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
        # Task 2-delta: start liveness heartbeat
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Update heartbeat timestamp at a fixed interval."""
        while True:
            self._heartbeat_ts = time.monotonic()
            await asyncio.sleep(self._heartbeat_interval)

    def thread_status(self) -> dict:
        """Return liveness status for sidecar probes.

        Returns dict with thread_alive, loop_alive, heartbeat_age, error.
        """
        thread_alive = self._thread is not None and self._thread.is_alive()
        loop_alive = self._loop is not None and self._loop.is_running()
        heartbeat_age = (
            round(time.monotonic() - self._heartbeat_ts, 2)
            if self._heartbeat_ts > 0
            else -1.0
        )
        return {
            "thread_alive": thread_alive,
            "loop_alive": loop_alive,
            "heartbeat_age": heartbeat_age,
            "error": str(self._startup_error) if self._startup_error else None,
        }

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


def start_durable_job_runtime_from_env(
    timeout: float = 30.0,
) -> DurableJobRuntime | None:
    """Bootstrap the process-owned durable jobs runtime.

    The runtime hosts the durable-job dispatch scheduler for ALL job types
    (workflow_run, data_ingestion, knowledge_*, endpoint provision, model
    download) — not just LLM endpoint management. The backend factory
    registers process-local backends regardless of the detected execution
    target, so backend detection is ADvisory-only here: an UNKNOWN result
    must NOT disable the runtime (umrs-front containers detect no local
    Slurm/K8s/Docker yet still must dispatch workflow_run jobs).
    Set ``UMRS_DISABLE_DURABLE_RUNTIME=1`` to explicitly opt out.
    """
    import os

    if os.environ.get("UMRS_DISABLE_DURABLE_RUNTIME", "").lower() in (
        "1",
        "true",
        "yes",
    ):
        logger.info("Durable job runtime disabled via UMRS_DISABLE_DURABLE_RUNTIME")
        return None

    from umrs.orchestrator.detection import detect_backend

    detection = detect_backend()
    logger.info("Orchestrator backend detection: %s", detection.backend)

    try:
        db_pool = build_durable_job_db_pool_from_env()
        from umrs.api.catalog_store import init_catalog_meta_store

        init_catalog_meta_store(db_pool)
        runtime = DurableJobRuntime(db_pool=db_pool)
        runtime.start(timeout=timeout)
        _set_runtime(runtime)
        logger.info("Durable job runtime started")
        return runtime
    except Exception:
        logger.exception("Failed to start durable job runtime")
        METRICS.bridge_event(_BRIDGE_NAME, "start_failure")
        _set_runtime(None)
        return None
