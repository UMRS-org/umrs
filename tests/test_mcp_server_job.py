"""D-22 first-lane pins: the MCP_SERVER durable kind (instance accounting)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs.mcp_server import (
    McpServerBackend,
    McpServerStatusChecker,
    validate_mcp_server_spec,
)
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType


class _Server:
    def __init__(self, server_id="mcp-1", name="srv", url="http://127.0.0.1:9101"):
        self.server_id = server_id
        self.name = name
        self.url = url


class _Registry:
    def __init__(self, servers=(_Server(),)):
        self._servers = {s.server_id: s for s in servers}
        self._by_name = {s.name: s for s in servers}

    def get_server(self, server_id):
        return self._servers[server_id]

    def get_server_by_name(self, name):
        return self._by_name[name]


def _job(spec):
    return SimpleNamespace(job_id="j-1", attempt_id=None, user_id="u", project_id="p", spec=spec)


def _attempt():
    return SimpleNamespace(attempt_id="a-1", metrics={}, backend_ref="")


def _run(coro):
    return asyncio.run(coro)


def test_spec_validation():
    assert validate_mcp_server_spec({"server_id": "mcp-9"})["server_id"] == "mcp-9"
    assert validate_mcp_server_spec({"server_name": "x"})["server_name"] == "x"
    with pytest.raises(ValueError, match="requires"):
        validate_mcp_server_spec({})
    with pytest.raises(ValueError, match="not both"):
        validate_mcp_server_spec({"server_id": "a", "server_name": "b"})


def test_backend_resolves_registry_row():
    backend = McpServerBackend(registry=_Registry())
    attempt = _attempt()
    ref = _run(backend.submit_job(_job({"server_id": "mcp-1"}), attempt))
    assert ref == "mcp_server|id=mcp-1"
    assert attempt.metrics["mcp_server_id"] == "mcp-1"


def test_backend_names_absent_server():
    backend = McpServerBackend(registry=_Registry())
    ref = _run(backend.submit_job(_job({"server_id": "mcp-ghost"}), _attempt()))
    assert "error=server_not_in_registry" in ref


def test_backend_degrades_without_registry():
    backend = McpServerBackend(registry=None)
    ref = _run(backend.submit_job(_job({"server_id": "mcp-1"}), _attempt()))
    assert "degraded=no_registry" in ref


def test_checker_running_on_healthy_probe(monkeypatch):
    from umrs.jobs import mcp_server as mod

    async def healthy(url):
        return True, "http 200"

    monkeypatch.setattr(mod, "_probe_server", healthy)
    checker = McpServerStatusChecker(registry_factory=_Registry())
    attempt = _attempt()
    attempt.backend_ref = "mcp_server|id=mcp-1"
    result = _run(checker.check_status(attempt, _job({})))
    assert result.status == "running"
    assert result.metrics["mcp_server_id"] == "mcp-1"


def test_checker_failed_on_unhealthy_probe(monkeypatch):
    from umrs.jobs import mcp_server as mod

    async def dead(url):
        return False, "ConnectError"

    monkeypatch.setattr(mod, "_probe_server", dead)
    checker = McpServerStatusChecker(registry_factory=_Registry())
    attempt = _attempt()
    attempt.backend_ref = "mcp_server|id=mcp-1"
    result = _run(checker.check_status(attempt, _job({})))
    assert result.status == "failed"
    assert "ConnectError" in result.exit_reason


def test_checker_refuses_degraded_and_absent_refs():
    checker = McpServerStatusChecker(registry_factory=_Registry())
    attempt = _attempt()
    attempt.backend_ref = "mcp_server|degraded=no_registry"
    assert _run(checker.check_status(attempt, _job({}))).status == "failed"
    attempt.backend_ref = "mcp_server|id=mcp-gone"
    result = _run(checker.check_status(attempt, _job({})))
    assert result.status == "failed"
    assert "left the registry" in result.exit_reason


def test_kind_registered_everywhere():
    assert JobType.MCP_SERVER.value == "mcp_server"
    assert JobType.MCP_SERVER in DEFAULT_BACKOFFS
    assert DEFAULT_BACKOFFS[JobType.MCP_SERVER]["max_attempts"] == 1  # never a restart loop
