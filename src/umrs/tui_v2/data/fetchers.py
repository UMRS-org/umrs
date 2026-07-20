"""Async fetchers for UMRS monitor v2.

Each fetcher is a thin coroutine that calls the blocking control API
via a single-thread executor (ZMQ sockets are not thread-safe).
Fetchers update the MonitorStore.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .store import MonitorStore


class Fetchers:
    """Per-tab async fetchers with ZMQ-safe single-thread executor."""

    def __init__(
        self,
        store: MonitorStore,
        control_api: Any,
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
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Mutable filter state
        self.status_filter: str | None = None
        self.project_filter: str | None = None
        self.job_type_filter: str | None = None
        self.model_scope_filter: str | None = None
        self.workflow_scope_filter: str | None = None

    # ------------------------------------------------------------------
    # Public fetchers
    # ------------------------------------------------------------------
    async def fetch_jobs(self) -> None:
        filters = self._job_filters()
        try:
            data = await self._run("list_jobs", lambda: self._api.list_jobs(**filters))
            await self._store.set("jobs", data)
        except Exception as exc:
            await self._store.set("jobs", None, error=str(exc))

    async def fetch_endpoints(self) -> None:
        try:
            data = await self._run("list_endpoints", self._api.list_endpoints)
            await self._store.set("endpoints", data)
        except Exception as exc:
            await self._store.set("endpoints", None, error=str(exc))

    async def fetch_status(self) -> None:
        try:
            data = await self._run(
                "request_status", lambda: self._api.request_status("all")
            )
            await self._store.set("status", data)
        except Exception as exc:
            await self._store.set("status", None, error=str(exc))

    async def fetch_models(self) -> None:
        try:
            data = await self._run("list_models", self._api.list_models)
            await self._store.set("models", data)
        except Exception as exc:
            await self._store.set("models", None, error=str(exc))

    async def fetch_workflows(self) -> None:
        try:
            data = await self._run("list_workflows", self._api.list_workflows)
            await self._store.set("workflows", data)
        except Exception as exc:
            await self._store.set("workflows", None, error=str(exc))

    async def fetch_errors(self) -> None:
        try:
            # Placeholder — real implementation needs workflow error fetch
            data = {"error_logs": [], "count": 0}
            await self._store.set("errors", data)
        except Exception as exc:
            await self._store.set("errors", None, error=str(exc))

    async def fetch_workflow_logs(self, execution_id: str) -> None:
        try:
            data = await self._run(
                "workflow_logs", lambda: self._api.workflow_logs(execution_id)
            )
            await self._store.set("workflow_logs", data)
        except Exception as exc:
            await self._store.set("workflow_logs", None, error=str(exc))

    async def fetch_connection(self) -> None:
        try:
            # Try to ping the UMRS API first
            ping = await self._run("ping", lambda: self._api.ping())
            ping_ok = ping.get("status") == "ok" if isinstance(ping, dict) else True

            # Read the current local connection profile from disk
            profile_name: str | None = None
            zmq_info: dict[str, Any] = {}
            tunnel_mode: bool = False
            gateway_url: str | None = None

            try:
                from umrs.shared.connection_profiles import (
                    resolve_connection,
                    resolve_profile_name,
                )
                from pathlib import Path
                import json

                profile_name = resolve_profile_name()
                if profile_name:
                    profile_file = Path(
                        __import__("umrs.shared.config", fromlist=["CONNECTION_PROFILE_DIR"])
                        .CONNECTION_PROFILE_DIR
                    ) / f"{profile_name}.json"
                    if profile_file.exists():
                        with profile_file.open() as f:
                            profile_data = json.load(f)
                        mode = profile_data.get("mode", "direct")
                        tunnel_mode = mode == "tunnel"
                        if mode == "tunnel" and "ssh" in profile_data:
                            zmq_host = profile_data["ssh"]["host"]
                        elif "zmq" in profile_data:
                            zmq_host = profile_data["zmq"].get("host", "localhost")
                        else:
                            zmq_host = "localhost"

                        # Extract ZMQ port info from the profile
                        if "zmq" in profile_data:
                            zmq_info = profile_data["zmq"]
                            if "ports" in zmq_info and isinstance(zmq_info["ports"], list):
                                control_port = zmq_info["ports"][0] if zmq_info["ports"] else None
                                data_ports = zmq_info["ports"][1:] if len(zmq_info["ports"]) > 1 else []
                                zmq_info["control_port"] = control_port
                                zmq_info["data_ports"] = data_ports
                                zmq_info["control_port"] = zmq_info.get("control_port", control_port)
                        gateway_url = profile_data.get("gateway_url") or resolve_connection().gateway_url
            except Exception:
                pass  # Fallback to ? values below

            # Build connection info dict
            connection_info = {
                "ping_ok": ping_ok,
                "profile": profile_name or "?",
                "gateway_url": gateway_url,
                "zmq": {
                    "host": zmq_info.get("host", gateway_url or "localhost"),
                    "control_port": zmq_info.get(
                        "control_port", zmq_info.get("ports", [None])[0] if "ports" in zmq_info else None
                    ),
                    "data_port_range": zmq_info.get(
                        "data_ports",
                        zmq_info.get("data_port_range"),
                    ),
                    "tunnel_mode": tunnel_mode,
                },
                "services": {
                    "minio_url": profile_data.get("minio_url") if "profile_data" in dir() else None,
                    "rag_url": profile_data.get("rag_url") if "profile_data" in dir() else None,
                },
            }

            data = {**ping, **{"ping_ok": ping_ok, "connection_info": connection_info}}
            await self._store.set("connection", data)
        except Exception as exc:
            await self._store.set("connection", None, error=str(exc))

    async def fetch_data_transfer(self) -> dict[str, Any]:
        """Fetch data transfer state (pending uploads/downloads)."""
        return {
            "uploads": [],
            "downloads": [],
            "history": [],
        }

    async def fetch_jobs_submit_options(self) -> dict[str, Any]:
        """Fetch available job types and model options for submission."""
        try:
            models_data = await self._run("list_models", self._api.list_models)
            models = models_data.get("models", []) if models_data else []
            return {
                "job_types": ["embedding", "inference", "training", "text_analysis"],
                "models": models,
                "user_cfg": {},
            }
        except Exception:
            return {
                "job_types": ["embedding", "inference", "training", "text_analysis"],
                "models": [],
                "user_cfg": {},
            }

    async def fetch_config(self) -> dict[str, Any]:
        """Fetch current user config."""
        return {}

    # ------------------------------------------------------------------
    # Internal
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

    async def _run(self, name: str, fn: Any) -> Any:
        """Run a blocking API call in the single-thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, fn)
