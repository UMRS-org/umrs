# Demo 26: MCP Server Management

## Purpose

Demonstrates UMRS MCP (Model Context Protocol) server management —
registration, tool discovery, tool invocation, and health checks.

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

## Run

```bash
bash run_demo.sh
```
