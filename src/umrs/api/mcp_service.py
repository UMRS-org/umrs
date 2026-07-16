"""Standalone FastAPI MCP service for umrs-front deployment.

Exposes:
- GET  /umrs/mcp/servers       — List registered MCP servers
- POST /umrs/mcp/servers       — Register an MCP server
- GET  /umrs/mcp/servers/{id}  — Get server by ID or name
- DELETE /umrs/mcp/servers/{id} — Unregister a server
- GET  /umrs/mcp/tools         — List all tools across servers
- GET  /umrs/mcp/tools/lookup  — Find server for a tool
- POST /umrs/mcp/tools/call    — Invoke a tool on appropriate server
- GET  /umrs/mcp/health        — MCP subsystem health

Runs as a CPU-only container on the frontend (port 8005).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .mcp import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load MCP server store from Postgres/MinIO
    try:
        from umrs.mcp.store import MCPServerStore
        from umrs.orchestrator.site_config import get_site_config

        # Ensure site config env vars are injected (DB URL, S3, etc.)
        get_site_config()
        # Store is lazy — initialized on first access
        _store = MCPServerStore()
        app.state.mcp_store = _store  # type: ignore[attr-defined]
        print("MCP service: server store initialized")
    except Exception as exc:
        print(f"MCP service: server store init failed (non-fatal): {exc}")

    yield


app = FastAPI(
    title="UMRS MCP Service",
    description="MCP server registry and tool execution for the AI4S platform",
    version="0.2.0",
    lifespan=lifespan,
)

# Mount the MCP router under /umrs (same prefix as other UMRS APIs)
app.include_router(router, prefix="/umrs")


# Health check at root
@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "mcp-frontend",
        "port": os.getenv("MCP_FRONTEND_PORT", "8005"),
        "uptime": "0",
    }
