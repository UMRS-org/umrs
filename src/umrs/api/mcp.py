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
from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from ..mcp.store import (
    MCPServerNotFoundError,
    MCPServerStore,
)
from .typed_errors import PlatformError, raise_typed

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


def _ip_in_allowlist(addr, hostname: str) -> bool:
    """True when the IP (or hostname) is explicitly allowlisted."""
    if not _MCP_URL_ALLOWLIST:
        return False
    for entry in _MCP_URL_ALLOWLIST.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            if entry == hostname:
                return True
    return False


def _check_blocked_ip(hostname: str) -> None:
    """Raise 422 when the hostname parses as an IP inside a blocked range."""
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        return  # Not an IP address — hostname checked against blocklist only
    if _ip_in_allowlist(addr, hostname):
        return
    for network in _BLOCKED_NETWORKS:
        if addr in network:
            raise_typed(
                PlatformError.mcp(
                    "unprocessable",
                    f"MCP server URL resolves to a blocked network ({network}): {hostname}",
                )
            )


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
    except Exception:  # noqa: BLE001 — proxy/route degradation is named in the response
        raise_typed(PlatformError.mcp("unprocessable", f"Invalid URL format: {url}"))

    if parsed.scheme not in ("http", "https"):
        raise_typed(
            PlatformError.mcp(
                "unprocessable",
                f"MCP server URL must use http or https scheme, got: {parsed.scheme!r}",
            )
        )

    hostname = parsed.hostname
    if not hostname:
        raise_typed(
            PlatformError.mcp("unprocessable", "MCP server URL must include a hostname")
        )

    if hostname.lower() in _BLOCKED_HOSTNAMES:
        raise_typed(
            PlatformError.mcp(
                "unprocessable",
                f"MCP server URL hostname is blocked (metadata endpoint): {hostname}",
            )
        )

    _check_blocked_ip(hostname)


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
        raise_typed(PlatformError.mcp("unavailable", "MCP server store not available"))
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
    placement: str = Field(
        default="local",
        description="D-19/D-49 run placement: local | frontend | cpu:<host> | trusted:<host> (site-declared hosts only)",
    )
    trust_class: str = Field(
        default="local-safe",
        description="ADR-061 trust class: local-safe | front-end | trusted-host-only",
    )


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


def _caller_scope(request):
    """Caller scope (strict default); mvp anonymous stays legacy-permissive."""
    from umrs.api._scope_identity import request_caller_scope

    return request_caller_scope(request)


def _server_visible(srv, project_id: str, unrestricted: bool) -> bool:
    """Read visibility (audit B9): global (unscoped) servers for everyone,
    project-bound servers only for that project; admins/anonymous bypass."""
    if unrestricted:
        return True
    bound = (getattr(srv, "project_id", None) or "").strip()
    return bound == "" or bound == project_id


@router.get("/servers", response_model=list[ServerResponse])
async def list_servers(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by status"
    ),
) -> list[ServerResponse]:
    """List registered MCP servers, scoped to the caller's project."""
    scope = _caller_scope(request)
    unrestricted = scope.is_admin or scope.anonymous
    store = _get_store()
    servers = store.list_servers(project_id=project_id, status_filter=status_filter)
    return [
        _server_to_response(s)
        for s in servers
        if _server_visible(s, scope.project_id, unrestricted)
    ]


def _enforce_registration_scope(
    request: Request, req: ServerRegistrationRequest
) -> ServerRegistrationRequest:
    """Mutation scoping (audit B9 write-path): verified non-admin callers may
    only register servers for their own project; an empty project_id claims
    the caller project rather than the global namespace. Admins and the
    legacy mvp-anonymous path keep full freedom.

    D-24 who-may-publish policy: `trusted-host-only` is the site-
    infrastructure class — admin-only to publish. `local-safe` and
    `front-end` stay project-scoped (any verified caller, own project).
    """
    scope = _caller_scope(request)
    if (req.trust_class or "").strip() == "trusted-host-only" and not scope.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "trust_class_denied",
                "message": (
                    "trusted-host-only is the site-infrastructure class "
                    "(D-24): only admins may publish trusted-host servers."
                ),
            },
        )
    if scope.is_admin or scope.anonymous:
        return req
    requested = (req.project_id or "").strip()
    if requested and requested != scope.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "scope_denied",
                "message": (
                    f"Cannot register MCP servers for project "
                    f"{requested!r}; your project is {scope.project_id!r}."
                ),
            },
        )
    if not requested:
        return req.model_copy(update={"project_id": scope.project_id})
    return req


def _record_registration_audit(server, req: ServerRegistrationRequest) -> None:
    """Audit + resource registry emission for a completed registration."""
    from .audit_events import emit_audit
    from .governance import record_resource_created

    emit_audit(
        "mcp.registered",
        project_id=req.project_id or "",
        details={"name": req.name, "url": req.url, "server_id": server.server_id},
    )
    record_resource_created(
        "mcp_servers",
        server.server_id,
        "",
        project_id=req.project_id or "",
        metadata={"name": req.name, "url": req.url},
    )


@router.post(
    "/servers", status_code=status.HTTP_201_CREATED, response_model=ServerResponse
)
async def register_server(
    req: ServerRegistrationRequest,
    request: Request,
) -> ServerResponse:
    """
    Register or update an MCP server.

    If a server with the same name exists, it is updated (upsert).
    """
    req = _enforce_registration_scope(request, req)

    # SSRF protection: validate URL before storing
    _validate_mcp_url(req.url)

    # Governance: quota check before registration
    from .governance import check_quota

    check_quota(req.project_id or "", "mcp_servers")

    store = _get_store()
    # D-19/D-49: undeclared host / bad class map → typed 400 with reason
    from umrs.shared.venue_placement import PlacementError

    try:
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
            placement=req.placement,
            trust_class=req.trust_class,
        )
    except PlacementError as exc:
        raise_typed(PlatformError.mcp("bad_request", str(exc)), cause=exc)
    logger.info("Registered MCP server: %s (id=%s)", server.name, server.server_id)
    _record_registration_audit(server, req)

    return _server_to_response(server)


@router.get("/servers/{server_id_or_name}", response_model=ServerResponse)
async def get_server(request: Request, server_id_or_name: str) -> ServerResponse:
    """Get an MCP server by ID or name (404-hides servers outside scope)."""
    scope = _caller_scope(request)
    unrestricted = scope.is_admin or scope.anonymous
    store = _get_store()
    try:
        if server_id_or_name.startswith("mcp-"):
            server = store.get_server(server_id_or_name)
        else:
            server = store.get_server_by_name(server_id_or_name)
    except MCPServerNotFoundError:
        raise_typed(
            PlatformError.mcp("not_found", f"Server not found: {server_id_or_name}")
        )
    if not _server_visible(server, scope.project_id, unrestricted):
        raise_typed(
            PlatformError.mcp("not_found", f"Server not found: {server_id_or_name}")
        )
    return _server_to_response(server)


def _enforce_unregister_scope(store, request: Request, server_id_or_name: str) -> None:
    """Non-admin verified callers may only delete servers bound to their own
    project; global servers are admin-only (B9 write-path)."""
    scope = _caller_scope(request)
    if scope.is_admin or scope.anonymous:
        return
    try:
        if server_id_or_name.startswith("mcp-"):
            server = store.get_server(server_id_or_name)
        else:
            server = store.get_server_by_name(server_id_or_name)
    except MCPServerNotFoundError:
        return
    bound = (getattr(server, "project_id", None) or "").strip()
    if bound != scope.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "scope_denied",
                "message": (
                    "Only your own project's MCP servers can be "
                    "unregistered; global servers are admin-only."
                ),
            },
        )


@router.delete("/servers/{server_id_or_name}")
async def unregister_server(request: Request, server_id_or_name: str) -> dict[str, str]:
    """Unregister an MCP server by ID or name."""
    store = _get_store()
    _enforce_unregister_scope(store, request, server_id_or_name)

    if server_id_or_name.startswith("mcp-"):
        deleted = store.unregister_server_by_id(server_id_or_name)
    else:
        deleted = store.unregister_server(server_id_or_name)

    if not deleted:
        raise_typed(
            PlatformError.mcp("not_found", f"Server not found: {server_id_or_name}")
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
    request: Request,
    server: str | None = Query(None, description="Filter by server name"),
) -> list[ToolResponse]:
    """List tools across registered MCP servers visible to the caller."""
    scope = _caller_scope(request)
    unrestricted = scope.is_admin or scope.anonymous
    store = _get_store()
    visible_names = {
        s.name
        for s in store.list_servers()
        if _server_visible(s, scope.project_id, unrestricted)
    }
    tools = store.list_tools(server_name=server)
    return [
        ToolResponse(
            name=t["name"],
            description=t["description"],
            inputSchema=t["inputSchema"],
            server=t["server"],
        )
        for t in tools
        if t["server"] in visible_names
    ]


@router.get("/tools/lookup", response_model=ServerResponse | None)
async def lookup_tool_server(
    tool: str = Query(..., description="Tool name to look up"),
) -> ServerResponse | None:
    """Find which server provides a given tool."""
    store = _get_store()
    server = store.get_server_for_tool(tool)
    if not server:
        raise_typed(PlatformError.mcp("not_found", f"No server found for tool: {tool}"))
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
    D-19 call-time trust enforcement: a trusted-host-only server is
    reachable only from its named trusted:<host> venue rail — no such
    rail exists yet (mcp is not in RAIL_GW_FAMILIES either), so every
    existing lane refuses by name rather than approximating. When the
    trusted-host venue lane lands, the rail's venue marker becomes the
    accepted path.
    """
    store = _get_store()
    server = store.get_server_for_tool(request.tool)
    if not server:
        raise_typed(PlatformError.mcp("not_found", f"Tool not found: {request.tool}"))
    if (getattr(server, "trust_class", "") or "") == "trusted-host-only":
        raise_typed(
            PlatformError.mcp(
                "trust_class_denied",
                f"Tool {request.tool!r} lives on trusted-host-only server "
                f"{server.name!r}: no trusted:<host> venue rail exists yet "
                "(D-19) — refusing by name, never approximating.",
            )
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
    except Exception:  # noqa: BLE001 — proxy/route degradation is named in the response
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


def _tool_call_headers() -> dict[str, str]:
    """JSON content type + correlation ID propagation (best-effort)."""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    try:
        from umrs.server.correlation import get_request_id

        rid = get_request_id()
        if rid:
            headers["X-Request-ID"] = rid
    except Exception:  # noqa: BLE001, S110 — proxy/route degradation is named in the response
        pass
    return headers


async def _post_tool_call_once(
    session_timeout: aiohttp.ClientTimeout,
    url: str,
    payload: dict[str, Any],
    tool_name: str,
) -> tuple[str, Any]:
    """One proxy attempt.

    Returns ("ok", result) when the tool call succeeded, ("retry", None) for
    a retryable HTTP error. Raises _MCPProxyError on hard failures.
    """
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(
            url, json=payload, headers=_tool_call_headers()
        ) as response:
            result = await _parse_tool_response(response, tool_name)
            if result is not None:
                return ("ok", result)
            return ("retry", response.status)


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
    last_status: Any = "unknown"

    for attempt in range(_MCP_MAX_RETRIES):
        last = attempt >= _MCP_MAX_RETRIES - 1
        try:
            outcome, info = await _post_tool_call_once(timeout, url, payload, tool_name)
        except aiohttp.ClientError as e:
            if not last:
                await asyncio.sleep(2**attempt)
                continue
            raise _MCPProxyError(f"Connection error: {e}") from e
        if outcome == "ok":
            return info
        last_status = info
        if not last:
            await asyncio.sleep(2**attempt)
            continue
        raise _MCPProxyError(f"HTTP {last_status} after retries")

    raise _MCPProxyError("Max retries exceeded")
