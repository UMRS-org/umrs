# MCP Server Tests

Quick test for the MCP (Model Context Protocol) server integration.

## Build and Run

```bash
docker build --network host -t mcp-test .

docker run --rm -i --network host --cap-drop all --cap-add=sys_nice \
    -v ./model-mcp-test.py:/app/test.py:ro mcp-test
```

## Prerequisites

- The Mini-RiVault stack must be running (vLLM + LiteLLM + MCPO)
- Tests connect via `--network host` to reach the local inference endpoint
