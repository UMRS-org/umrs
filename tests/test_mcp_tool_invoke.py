"""Tests for the UMRS MCP tool invocation routes (added in Phase 5)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.mcp import router, set_mcp_store


class MockMCPTool:
    """Minimal tool stub for tests."""

    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockMCPServer:
    """Minimal server stub for tests."""

    def __init__(self, *, name="test-server", url="http://mcp:9000", transport="http"):
        self.server_id = "mcp-001"
        self.name = name
        self.url = url
        self.version = "1.0.0"
        self.transport = transport
        self.status = "active"
        self.description = "Test MCP server"
        self.auth_type = "none"
        self.auth_config = {}
        self.metadata = {}
        self.registered_by = None
        self.project_id = None
        self.tools = [MockMCPTool("search", "Search tool", {"type": "object"})]
        self.created_at = None
        self.updated_at = None


class MockMCPStore:
    """Minimal MCPServerStore stub for tests."""

    def __init__(self, servers=None):
        self._servers = servers or []

    def list_servers(self, **kwargs):
        return self._servers

    def get_server_for_tool(self, tool_name: str):
        for s in self._servers:
            for t in s.tools:
                if t.name == tool_name:
                    return s
        return None

    def list_tools(self, server_name=None):
        tools = []
        for s in self._servers:
            for t in s.tools:
                tools.append({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                    "server": s.name,
                })
        return tools

    def get_stats(self):
        return {"servers": len(self._servers), "tools": sum(len(s.tools) for s in self._servers)}


def _create_client(store: MockMCPStore) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    set_mcp_store(store)
    return TestClient(app)


def test_call_tool_success():
    """POST /umrs/mcp/call should invoke tool and return result."""
    server = MockMCPServer()
    store = MockMCPStore(servers=[server])
    client = _create_client(store)

    mock_result = {
        "content": [{"type": "text", "text": "found 42 results"}],
        "isError": False,
    }

    with patch("umrs.api.mcp._proxy_tool_call", new_callable=AsyncMock, return_value=mock_result):
        response = client.post(
            "/umrs/mcp/call",
            json={"tool": "search", "arguments": {"query": "test"}},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["server"] == "test-server"
    assert data["content"][0]["text"] == "found 42 results"
    assert data["isError"] is False


def test_call_tool_not_found():
    """POST /umrs/mcp/call should return 404 for unknown tool."""
    store = MockMCPStore(servers=[])
    client = _create_client(store)

    response = client.post(
        "/umrs/mcp/call",
        json={"tool": "nonexistent", "arguments": {}},
    )

    assert response.status_code == 404


def test_invoke_alias():
    """POST /umrs/mcp/invoke should also work (alias)."""
    store = MockMCPStore(servers=[])
    client = _create_client(store)

    response = client.post(
        "/umrs/mcp/invoke",
        json={"tool": "nonexistent", "arguments": {}},
    )

    assert response.status_code == 404


def test_list_tools():
    """GET /umrs/mcp/tools should list all tools."""
    server = MockMCPServer()
    store = MockMCPStore(servers=[server])
    client = _create_client(store)

    response = client.get("/umrs/mcp/tools")

    assert response.status_code == 200
    tools = response.json()
    assert len(tools) == 1
    assert tools[0]["name"] == "search"
    assert tools[0]["server"] == "test-server"


def test_mcp_health():
    """GET /umrs/mcp/health should return stats."""
    server = MockMCPServer()
    store = MockMCPStore(servers=[server])
    client = _create_client(store)

    response = client.get("/umrs/mcp/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["servers"] == 1
    assert data["tools"] == 1
