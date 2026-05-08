# Demo 26: MCP Server Management

## Purpose

Demonstrates UMRS MCP (Model Context Protocol) server management and the
preferred mounted MCP path for mini-RiVault endpoints: user/project MCP code
is mounted under `/mcp-servers`, validated by `r4_mcp.runners.custom_mcp`, and
run by the `custom-mcp` profile inside the endpoint boundary.

## API Endpoints Covered

| Method | Path                          | Description                |
|--------|-------------------------------|----------------------------|
| GET    | `/umrs/mcp/servers`           | List registered servers    |
| POST   | `/umrs/mcp/servers`           | Register a server          |
| GET    | `/umrs/mcp/servers/{id}`      | Server details             |
| DELETE | `/umrs/mcp/servers/{id}`      | Unregister a server        |
| GET    | `/umrs/mcp/tools`             | List all tools             |
| POST   | `/umrs/mcp/invoke`            | Invoke a tool              |
| GET    | `/umrs/mcp/health`            | Health check               |

External MCP registration remains supported for login-node or existing service
integrations. GPU-adjacent compute-node MCP work should use endpoint mounts,
not a separate compute-node MCP launcher.

## Run

```bash
bash run_demo.sh
```
