"""MCP server credentials: never returned to callers, always presented upstream.

Two defects, both on every deployment:
- ``ServerResponse`` carried ``auth_config`` verbatim, so listing servers
  disclosed stored credentials (e.g. ``r4 mcp register --auth-token``).
- The tool-call proxy never sent them, so a server requiring a credential
  could not be called through UMRS at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from tests.test_mcp_tool_invoke import MockMCPServer, MockMCPStore, _create_client
from umrs.api import mcp as mcp_api


def _server(auth_type="api_key", auth_config=None):
    s = MockMCPServer()
    s.auth_type = auth_type
    s.auth_config = {"token": "s3cr3t"} if auth_config is None else auth_config
    return s


def test_listing_never_returns_credential_values():
    client = _create_client(MockMCPStore(servers=[_server()]))
    resp = client.get("/umrs/mcp/servers")
    assert resp.status_code == 200
    body = resp.text
    assert "s3cr3t" not in body
    assert resp.json()[0]["auth_config"] == {"token": "***"}


@pytest.mark.parametrize(
    ("auth_type", "auth_config", "expected"),
    [
        ("api_key", {"token": "t1"}, {"Authorization": "Bearer t1"}),
        ("bearer", {"api_key": "t2"}, {"Authorization": "Bearer t2"}),
        ("api_key", {"token": "t3", "header": "X-API-Key"}, {"X-API-Key": "t3"}),
        ("none", {"token": "ignored"}, {}),
        ("api_key", {}, {}),
    ],
)
def test_upstream_auth_headers(auth_type, auth_config, expected):
    assert mcp_api._upstream_auth_headers(_server(auth_type, auth_config)) == expected


def test_call_tool_passes_credentials_to_the_proxy():
    client = _create_client(MockMCPStore(servers=[_server()]))
    result = {"content": [{"type": "text", "text": "ok"}], "isError": False}
    with patch(
        "umrs.api.mcp._proxy_tool_call", new_callable=AsyncMock, return_value=result
    ) as proxy:
        resp = client.post("/umrs/mcp/call", json={"tool": "search", "arguments": {}})
    assert resp.status_code == 200
    assert proxy.await_args.kwargs["auth_headers"] == {"Authorization": "Bearer s3cr3t"}


@pytest.mark.asyncio
async def test_proxy_sends_credentials_on_the_wire():
    with patch(
        "umrs.api.mcp._post_tool_call_once",
        new_callable=AsyncMock,
        return_value=("ok", {"content": [], "isError": False}),
    ) as post:
        await mcp_api._proxy_tool_call(
            "http://mcp:9000", "http", "search", {}, auth_headers={"Authorization": "Bearer x"}
        )
    assert post.await_args.args[4] == {"Authorization": "Bearer x"}
