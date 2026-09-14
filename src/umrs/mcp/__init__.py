"""
MCP Server Registry module for UMRS.

Provides persistent MCP server registration with:
- Server CRUD operations
- Tool indexing per server
- Health tracking

Replaces the in-memory MCPRegistry that lived inside the Gateway.
"""

from .models import MCPServer, MCPTool, generate_server_id, generate_tool_id
from .schema import ensure_schema
from .store import (
    DuplicateMCPServerError,
    MCPServerNotFoundError,
    MCPServerStore,
    MCPServerStoreError,
)

__all__ = [
    "DuplicateMCPServerError",
    "MCPServer",
    "MCPServerNotFoundError",
    "MCPServerStore",
    "MCPServerStoreError",
    "MCPTool",
    "ensure_schema",
    "generate_server_id",
    "generate_tool_id",
]
