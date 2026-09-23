"""MCP_SERVER durable job kind — instance lifecycle + accounting (D-22 first lane).

The design: hatch starts, health-checks, restarts and stops servers;
UMRS accounts for them like sessions. The R4 hatch runtime bridge owns
the local child-process lane; this kind is UMRS's durable accounting
row for a server INSTANCE — one job per running instance, attempt =
the instance's life, status = the registry truth (health-checked).

The start-on-front-end EXECUTOR (launching the mcp-trusted pod) is the
named tail: it needs the per-server manifest format (ADR-061's
``hatch.json`` capabilities) that the SANDBOX-PROFILES-MAP names. What
lands now is the durable lane's full shape: the kind, the spec contract
(registry-resolved, never invented), and the health-check status
checker driving the lifecycle truth.
"""

from __future__ import annotations

from typing import Any

from .models import Job, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker

_REF_PREFIX = "mcp_server|"


def validate_mcp_server_spec(spec: dict) -> dict:
    """spec: {server_id | server_name (one required), project_id}."""
    if not isinstance(spec, dict):
        raise TypeError("mcp_server spec must be an object")
    server_id = str(spec.get("server_id") or "").strip()
    server_name = str(spec.get("server_name") or "").strip()
    if not server_id and not server_name:
        raise ValueError("mcp_server spec requires server_id or server_name")
    if server_id and server_name:
        raise ValueError("mcp_server spec takes server_id or server_name, not both")
    return {"server_id": server_id, "server_name": server_name}


class McpServerBackend:
    """Register the instance row; the registry is the runtime truth.

    The launch executor is the named tail (mcp-trusted manifest); the
    backend's submit resolves the registry row (fail-named when absent)
    and stamps the instance ref — the health checker owns lifecycle
    truth from there.
    """

    def __init__(self, registry=None) -> None:
        # registry: an awaitable/sync factory returning the MCP store
        self._registry_factory = registry

    @property
    def backend_name(self) -> str:
        return "mcp_server"

    async def submit_job(self, job: Job, attempt) -> str:
        spec = validate_mcp_server_spec(dict(job.spec or {}))
        if self._registry_factory is None:
            return f"{_REF_PREFIX}degraded=no_registry"
        registry = self._registry_factory
        if callable(registry):
            registry = registry() if not hasattr(registry, "get_server") else registry
        server = self._resolve(registry, spec)
        if server is None:
            return f"{_REF_PREFIX}error=server_not_in_registry"
        attempt.metrics["mcp_server_id"] = str(getattr(server, "server_id", ""))
        return f"{_REF_PREFIX}id={getattr(server, 'server_id', '')}"

    @staticmethod
    def _resolve(registry: Any, spec: dict):
        if spec["server_id"]:
            try:
                return registry.get_server(spec["server_id"])
            except Exception:  # noqa: BLE001 — named degradation below
                return None
        try:
            return registry.get_server_by_name(spec["server_name"])
        except Exception:  # noqa: BLE001 — named degradation below
            return None


async def _probe_server(url: str) -> tuple[bool, str]:
    """One HTTP probe of the server's health endpoint; (ok, named detail)."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url.rstrip('/')}/health")
        return response.status_code < 500, f"http {response.status_code}"
    except Exception as exc:  # noqa: BLE001 — the probe names its class
        return False, type(exc).__name__


class McpServerStatusChecker(BackendStatusChecker):
    """Health-check the registered server's URL; the registry is truth.

    Healthy → running (a long-lived instance: the attempt IS its life);
    probe failure → failed with the named probe detail; degraded or
    absent-registry refs → failed named (never a silent pass)."""

    def __init__(self, registry_factory=None) -> None:
        self._registry_factory = registry_factory

    @property
    def job_type(self) -> JobType:
        return JobType.MCP_SERVER

    async def check_status(self, attempt: Any, job: Job) -> AttemptCheckResult:
        ref = attempt.backend_ref or ""
        if not ref.startswith(_REF_PREFIX):
            return AttemptCheckResult.failed("Unexpected backend reference")
        if "degraded=" in ref or "error=" in ref:
            return AttemptCheckResult.failed(
                "mcp_server submitted without a registry-resolved instance"
            )
        server_id = ref[len(_REF_PREFIX) + len("id=") :]
        if self._registry_factory is None:
            return AttemptCheckResult.failed("MCP registry unavailable")
        registry = self._registry_factory
        if callable(registry):
            registry = registry() if not hasattr(registry, "get_server") else registry
        try:
            server = registry.get_server(server_id)
        except Exception:  # noqa: BLE001 — named degradation below
            return AttemptCheckResult.failed(
                f"MCP server {server_id!r} left the registry"
            )
        url = str(getattr(server, "url", "") or "")
        if not url:
            return AttemptCheckResult.failed(f"MCP server {server_id!r} has no URL")
        ok, detail = await _probe_server(url)
        if ok:
            result = AttemptCheckResult.running(detail=f"instance healthy ({detail})")
            result.metrics = {"mcp_server_id": server_id}
            return result
        return AttemptCheckResult.failed(f"instance unhealthy: {detail}")


def build_mcp_server_services(registry_factory=None):
    """Adapter-registry entry: backend + checker sharing one registry."""
    return {
        "backend": McpServerBackend(registry=registry_factory),
        "checker": McpServerStatusChecker(registry_factory=registry_factory),
    }
