"""
MCP Server Registry REST API.

Provides:
- GET  /umrs/mcp/servers       - List registered MCP servers
- POST /umrs/mcp/servers       - Register an MCP server
- GET  /umrs/mcp/servers/{id}  - Get server by ID or name
- DELETE /umrs/mcp/servers/{id} - Unregister a server
- GET  /umrs/mcp/tools         - List all tools across servers
- GET  /umrs/mcp/tools/lookup  - Find server for a tool
- POST /umrs/mcp/tools/call    - Invoke a tool on appropriate server
- GET  /umrs/mcp/health        - MCP subsystem health
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import urllib.parse
from typing import Any

import aiohttp
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..mcp.store import (
    MCPServerNotFoundError,
    MCPServerStore,
)


# =============================================================================
# URL Validation / SSRF Protection
# =============================================================================

# Blocked IP ranges: RFC 1918, link-local, loopback, metadata endpoints
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # Loopback
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local / cloud metadata
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Blocked hostnames (cloud metadata endpoints)
_BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.gce.internal",
}

# Env-configurable allowlist overrides (comma-separated CIDRs or hostnames)
_MCP_URL_ALLOWLIST = os.getenv("MCP_URL_ALLOWLIST", "").strip()


def _validate_mcp_url(url: str) -> None:
    """
    Validate MCP server URL to prevent SSRF attacks.

    Blocks:
    - Private/internal IP ranges (RFC 1918, link-local, loopback)
    - Cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    - Non-HTTP(S) schemes
    - URLs without a hostname

    Raises HTTPException(422) on validation failure.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid URL format: {url}",
        )

    # Scheme check
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"MCP server URL must use http or https scheme, got: {parsed.scheme!r}",
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="MCP server URL must include a hostname",
        )

    # Check blocked hostnames
    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"MCP server URL hostname is blocked (metadata endpoint): {hostname}",
        )

    # Check if hostname is an IP address and validate against blocked ranges
    try:
        addr = ipaddress.ip_address(hostname)
        # Check allowlist first
        if _MCP_URL_ALLOWLIST:
            for entry in _MCP_URL_ALLOWLIST.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                try:
                    if addr in ipaddress.ip_network(entry, strict=False):
                        return  # Explicitly allowed
                except ValueError:
                    if entry == hostname:
                        return  # Hostname match in allowlist

        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"MCP server URL resolves to a blocked network ({network}): {hostname}",
                )
    except ValueError:
        # Not an IP address — it's a hostname, check blocklist only
        pass


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/mcp", tags=["mcp"])


# =============================================================================
# Module-level store (set by HTTP sidecar on startup)
# =============================================================================

_store: MCPServerStore | None = None


def set_mcp_store(store: MCPServerStore) -> None:
    """Set the MCP server store (called during HTTP sidecar init)."""
    global _store
    _store = store


def _get_store() -> MCPServerStore:
    """Get the active MCP store or 503."""
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MCP server store not available",
        )
    return _store


# =============================================================================
# Request / Response Models
# =============================================================================


class ServerRegistrationRequest(BaseModel):
    """Request to register an MCP server."""

    name: str = Field(..., description="Server identifier")
    url: str = Field(..., description="Server HTTP endpoint URL")
    version: str = Field(default="0.1.0", description="Server version")
    transport: str = Field(default="http", description="Transport protocol")
    description: str = Field(default="", description="Server description")
    auth_type: str = Field(
        default="none", description="Auth type: none, api_key, oauth"
    )
    auth_config: dict[str, Any] = Field(
        default_factory=dict, description="Auth configuration"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )
    tools: list[dict[str, Any]] = Field(
        default_factory=list, description="Tool definitions"
    )
    project_id: str | None = Field(default=None, description="Owning project ID")


class ServerResponse(BaseModel):
    """MCP server in API responses."""

    server_id: str
    name: str
    url: str
    version: str
    transport: str
    status: str
    description: str
    auth_type: str
    auth_config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    registered_by: str | None = None
    project_id: str | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class ToolResponse(BaseModel):
    """MCP tool in API responses."""

    name: str
    description: str
    inputSchema: dict[str, Any] = Field(default_factory=dict)
    server: str | None = None


class ToolCallRequest(BaseModel):
    """Request to invoke an MCP tool."""

    tool: str = Field(..., description="Tool name to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, alias="_meta")


class ToolCallResponse(BaseModel):
    """Response from tool invocation."""

    content: list[dict[str, Any]]
    isError: bool = False
    server: str | None = None


# Configuration for tool proxying
_MCP_PROXY_TIMEOUT = int(os.getenv("MCP_PROXY_TIMEOUT_SEC", "30"))
_MCP_MAX_RETRIES = int(os.getenv("MCP_MAX_RETRIES", "3"))


# =============================================================================
# Server Routes
# =============================================================================


@router.get("/servers", response_model=list[ServerResponse])
async def list_servers(
    project_id: str | None = Query(None, description="Filter by project"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by status"
    ),
) -> list[ServerResponse]:
    """List all registered MCP servers."""
    store = _get_store()
    servers = store.list_servers(project_id=project_id, status_filter=status_filter)
    return [_server_to_response(s) for s in servers]


@router.post(
    "/servers", status_code=status.HTTP_201_CREATED, response_model=ServerResponse
)
async def register_server(
    req: ServerRegistrationRequest,
) -> ServerResponse:
    """
    Register or update an MCP server.

    If a server with the same name exists, it is updated (upsert).
    """
    # SSRF protection: validate URL before storing
    _validate_mcp_url(req.url)

    # Governance: quota check before registration
    from .governance import check_quota

    check_quota(req.project_id or "", "mcp_servers")

    store = _get_store()
    server = store.register_server(
        name=req.name,
        url=req.url,
        version=req.version,
        transport=req.transport,
        description=req.description,
        auth_type=req.auth_type,
        auth_config=req.auth_config,
        metadata=req.metadata,
        tools=req.tools,
        project_id=req.project_id,
    )
    logger.info("Registered MCP server: %s (id=%s)", server.name, server.server_id)

    from .audit_events import emit_audit

    emit_audit(
        "mcp.registered",
        project_id=req.project_id or "",
        details={"name": req.name, "url": req.url, "server_id": server.server_id},
    )
    from .governance import record_resource_created

    record_resource_created(
        "mcp_servers",
        server.server_id,
        "",
        project_id=req.project_id or "",
        metadata={"name": req.name, "url": req.url},
    )

    return _server_to_response(server)


@router.get("/servers/{server_id_or_name}", response_model=ServerResponse)
async def get_server(server_id_or_name: str) -> ServerResponse:
    """Get an MCP server by ID or name."""
    store = _get_store()
    try:
        if server_id_or_name.startswith("mcp-"):
            server = store.get_server(server_id_or_name)
        else:
            server = store.get_server_by_name(server_id_or_name)
    except MCPServerNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found: {server_id_or_name}",
        ) from None
    return _server_to_response(server)


@router.delete("/servers/{server_id_or_name}")
async def unregister_server(server_id_or_name: str) -> dict[str, str]:
    """Unregister an MCP server by ID or name."""
    store = _get_store()
    if server_id_or_name.startswith("mcp-"):
        deleted = store.unregister_server_by_id(server_id_or_name)
    else:
        deleted = store.unregister_server(server_id_or_name)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server not found: {server_id_or_name}",
        )

    from .audit_events import emit_audit

    emit_audit("mcp.unregistered", details={"server": server_id_or_name})
    from .governance import record_resource_deleted

    record_resource_deleted("mcp_servers", server_id_or_name)

    return {"status": "unregistered", "server": server_id_or_name}


# =============================================================================
# Tool Routes
# =============================================================================


@router.get("/tools", response_model=list[ToolResponse])
async def list_tools(
    server: str | None = Query(None, description="Filter by server name"),
) -> list[ToolResponse]:
    """List all tools across registered MCP servers."""
    store = _get_store()
    tools = store.list_tools(server_name=server)
    return [
        ToolResponse(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
            server=t["server"],
        )
        for t in tools
    ]


@router.get("/tools/lookup", response_model=ServerResponse | None)
async def lookup_tool_server(
    tool: str = Query(..., description="Tool name to look up"),
) -> ServerResponse | None:
    """Find which server provides a given tool."""
    store = _get_store()
    server = store.get_server_for_tool(tool)
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No server found for tool: {tool}",
        )
    return _server_to_response(server)


# =============================================================================
# Health
# =============================================================================


@router.get("/health")
async def mcp_health() -> dict[str, Any]:
    """MCP subsystem health check."""
    store = _get_store()
    stats = store.get_stats()
    return {
        "status": "ok",
        "servers": stats["servers"],
        "tools": stats["tools"],
    }


# =============================================================================
# Helpers
# =============================================================================


def _server_to_response(server) -> ServerResponse:
    """Convert MCPServer dataclass to API response."""
    return ServerResponse(
        server_id=server.server_id,
        name=server.name,
        url=server.url,
        version=server.version,
        transport=server.transport,
        status=server.status,
        description=server.description,
        auth_type=server.auth_type,
        auth_config=server.auth_config,
        metadata=server.metadata,
        registered_by=server.registered_by,
        project_id=server.project_id,
        tools=[
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in server.tools
        ],
        created_at=server.created_at.isoformat() if server.created_at else None,
        updated_at=server.updated_at.isoformat() if server.updated_at else None,
    )


# =============================================================================
# Tool Invocation (moved from Python Gateway)
# =============================================================================


@router.post("/call", response_model=ToolCallResponse)
@router.post("/invoke", response_model=ToolCallResponse)
async def call_tool(
    request: ToolCallRequest,
) -> ToolCallResponse:
    """
    Invoke an MCP tool.

    Looks up which server owns the tool, then proxies the call directly.
    """
    store = _get_store()
    server = store.get_server_for_tool(request.tool)
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool not found: {request.tool}",
        )

    logger.info("Invoking tool %s on server %s", request.tool, server.name)

    try:
        result = await _proxy_tool_call(
            server.url,
            server.transport,
            request.tool,
            request.arguments,
            metadata=request.metadata,
        )
        return ToolCallResponse(
            content=result.get("content", []),
            isError=result.get("isError", False),
            server=server.name,
        )
    except _MCPProxyError as e:
        logger.error("MCP proxy error: %s", e)
        return ToolCallResponse(
            content=[{"type": "text", "text": str(e)}],
            isError=True,
            server=server.name,
        )


# =============================================================================
# Tool Discovery + Proxy Helpers
# =============================================================================


class _MCPProxyError(Exception):
    """Error proxying request to MCP server."""


async def _discover_server_tools(url: str) -> list[dict[str, Any]]:
    """Try to discover tools from an HTTP MCP server."""
    if not url.startswith("http"):
        return []

    timeout = aiohttp.ClientTimeout(total=5)
    tools_url = f"{url.rstrip('/')}/tools"
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(tools_url) as response:
                if response.status != 200:
                    return []
                payload = await response.json()
    except Exception:
        return []

    if isinstance(payload, dict):
        tools = payload.get("tools", [])
    elif isinstance(payload, list):
        tools = payload
    else:
        tools = []

    return [_normalize_tool_def(t) for t in tools if isinstance(t, dict)]


def _normalize_tool_def(tool: dict[str, Any]) -> dict[str, Any]:
    """Normalize MCP tool definitions."""
    normalized = dict(tool)
    if "inputSchema" not in normalized and "parameters" in normalized:
        normalized["inputSchema"] = normalized["parameters"]
    if "parameters" not in normalized and "inputSchema" in normalized:
        normalized["parameters"] = normalized["inputSchema"]
    return normalized


async def _parse_tool_response(
    response: aiohttp.ClientResponse, tool_name: str
) -> dict[str, Any] | None:
    """Parse the JSON-RPC response from a tool call. Returns None for retryable 5xx."""
    if response.status == 200:
        result = await response.json()
        if "result" in result:
            return result["result"]
        if "error" in result:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result["error"].get("message", "Unknown error"),
                    }
                ],
                "isError": True,
            }
        return result
    if response.status == 404:
        raise _MCPProxyError(f"Tool '{tool_name}' not found on server")
    if response.status >= 500:
        return None  # retryable
    text = await response.text()
    raise _MCPProxyError(f"HTTP {response.status}: {text}")


async def _proxy_tool_call(
    server_url: str,
    transport: str,
    tool_name: str,
    arguments: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Proxy a tool call to the MCP server via HTTP."""
    if transport not in ("http", "sse") and not server_url.startswith("http"):
        raise _MCPProxyError(
            f"Transport '{transport}' not yet supported. Please use HTTP transport."
        )

    url = f"{server_url.rstrip('/')}/tools/call"
    params: dict[str, Any] = {"name": tool_name, "arguments": arguments}
    if metadata:
        params["_meta"] = metadata

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": params,
        "id": 1,
    }
    timeout = aiohttp.ClientTimeout(total=_MCP_PROXY_TIMEOUT)

    for attempt in range(_MCP_MAX_RETRIES):
        try:
            # Propagate correlation ID to MCP server
            call_headers: dict[str, str] = {"Content-Type": "application/json"}
            try:
                from umrs.server.correlation import get_request_id

                rid = get_request_id()
                if rid:
                    call_headers["X-Request-ID"] = rid
            except Exception:
                pass

            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    url,
                    json=payload,
                    headers=call_headers,
                ) as response,
            ):
                result = await _parse_tool_response(response, tool_name)
                if result is not None:
                    return result
                if attempt < _MCP_MAX_RETRIES - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise _MCPProxyError(f"HTTP {response.status} after retries")

        except aiohttp.ClientError as e:
            if attempt < _MCP_MAX_RETRIES - 1:
                await asyncio.sleep(2**attempt)
                continue
            raise _MCPProxyError(f"Connection error: {e}") from e

    raise _MCPProxyError("Max retries exceeded")
