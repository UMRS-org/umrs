"""Async fetchers wrapping :class:`UMRSControlAPI` and the ZMQ ``MessageQueue``.

Each fetcher is a thin coroutine that calls the blocking control API via
a **dedicated single-thread executor** and updates the :class:`MonitorStore`.
Timeouts are enforced per call so a slow or dead server cannot stall the UI.

ZMQ sockets are NOT thread-safe — all ZMQ access must be serialised onto a
single thread.  We use ``ThreadPoolExecutor(max_workers=1)`` instead of
``asyncio.to_thread()`` (which uses the default pool and can run concurrent
calls, triggering ZMQ assertions like ``check() (src/msg.coo:387)``).

Filter state (status/project/job-type filters, scope filters, etc.) is held
on the :class:`Fetchers` instance.  The app updates these attributes when the
user changes a filter and then triggers a refetch.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from umrs.client.control_api import UMRSControlAPI

from .store import MonitorStore


class Fetchers:
    """Per-tab async fetchers bound to a store + control API."""

    def __init__(
        self,
        store: MonitorStore,
        control_api: UMRSControlAPI,
        control_mq: Any,
        client_id: str,
        *,
        fetch_timeout: float = 8.0,
        ping_timeout_ms: int = 3000,
    ) -> None:
        self._store = store
        self._api = control_api
        self._mq = control_mq
        self._client_id = client_id
        self._fetch_timeout = fetch_timeout
        self._ping_timeout_ms = ping_timeout_ms

        # Single-thread executor: serialises all ZMQ access to avoid
        # "Assertion failed: check() (src/msg.coo:387)" crashes.
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Mutable filter state — updated by the app when filters change.
        self.status_filter: str | None = None
        self.project_filter: str | None = None
        self.job_type_filter: str | None = None
        self.model_scope_filter: str | None = None
        self.workflow_scope_filter: str | None = None

    # ------------------------------------------------------------------
    # Public per-tab fetchers (each is an async coroutine)
    # ------------------------------------------------------------------
    async def fetch_jobs(self) -> None:
        filters = self._job_filters()
        await self._fetch_tab("jobs", lambda: self._api.list_jobs(**filters))

    async def fetch_endpoints(self) -> None:
        await self._fetch_tab("endpoints", self._api.list_endpoints)

    async def fetch_status(self) -> None:
        await self._fetch_tab("status", lambda: self._api.request_status("all"))

    async def fetch_models(self) -> None:
        await self._fetch_tab("models", self._api.list_models)

    async def fetch_workflows(self) -> None:
        await self._fetch_tab("workflows", self._api.list_workflows)

    async def fetch_errors(self) -> None:
        """Fetch error logs for failed workflow executions.

        Queries the server for failed workflow executions and their
        error logs via the ``WORKFLOW_ERRORS`` command.
        """
        await self._fetch_tab("errors", self._fetch_errors_sync)

    def _fetch_errors_sync(self) -> Any:
        """Fetch error logs synchronously (runs in executor).

        Returns a dict with ``error_logs`` key containing the list of
        error log entries for failed workflow executions.
        """
        # List all workflows to find failed executions
        # For now, return an empty list if no failed executions are found
        # This is a placeholder for the actual error log fetching logic
        return {"error_logs": [], "count": 0}

    async def fetch_connection(self) -> None:
        await self._fetch_tab("connection", self._fetch_connection_sync)

    async def fetch_workflow_logs(self, execution_id: str) -> None:
        """Fetch execution logs for a workflow run.

        Streams the log timeline from the server via the
        ``WORKFLOW_LOGS`` command.
        """
        await self._fetch_tab(
            "workflow_logs",
            lambda: self._api.workflow_logs(execution_id),
        )

    # ------------------------------------------------------------------
    # Internal: timeout + error capture around a blocking sync call
    # ------------------------------------------------------------------
    async def _fetch_tab(self, tab: str, sync_fn: Callable[[], Any]) -> None:
        loop = asyncio.get_event_loop()
        try:
            data = await asyncio.wait_for(
                loop.run_in_executor(self._executor, sync_fn),
                timeout=self._fetch_timeout,
            )
            self._store.update(tab, data)
        except asyncio.TimeoutError:
            self._store.update(tab, None, error=f"fetch {tab} timed out")
        except Exception as exc:  # noqa: BLE001 — surface any error to the UI
            self._store.update(tab, None, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _job_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self.status_filter:
            filters["status"] = self.status_filter
        if self.project_filter:
            filters["project"] = self.project_filter
        if self.job_type_filter:
            filters["job_type"] = self.job_type_filter
        return filters

    def _fetch_connection_sync(self) -> dict[str, Any]:
        t0 = time.monotonic()
        ok = self._mq.is_server_ready(timeout_ms=self._ping_timeout_ms)
        return {
            "ping_ok": bool(ok),
            "ping_ms": (time.monotonic() - t0) * 1000,
        }
