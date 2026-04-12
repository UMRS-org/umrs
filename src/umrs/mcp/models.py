"""
MCP Server Registry data models.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_server_id() -> str:
    """Generate a unique MCP server ID."""
    return f"mcp-{secrets.token_hex(6)}"


def generate_tool_id() -> str:
    """Generate a unique MCP tool ID."""
    return f"tool-{secrets.token_hex(6)}"


@dataclass
class MCPTool:
    """A tool provided by an MCP server."""

    tool_id: str
    server_id: str
    name: str
    description: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class MCPServer:
    """A registered MCP server."""

    server_id: str
    name: str
    url: str
    version: str = "0.1.0"
    transport: str = "http"
    status: str = "unknown"
    description: str = ""
    auth_type: str = "none"
    auth_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_by: Optional[str] = None
    project_id: Optional[str] = None
    tools: List[MCPTool] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response format."""
        return {
            "server_id": self.server_id,
            "name": self.name,
            "url": self.url,
            "version": self.version,
            "transport": self.transport,
            "status": self.status,
            "description": self.description,
            "auth_type": self.auth_type,
            "auth_config": self.auth_config,
            "metadata": self.metadata,
            "registered_by": self.registered_by,
            "project_id": self.project_id,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self.tools
            ],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
