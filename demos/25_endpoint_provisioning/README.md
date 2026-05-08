# Demo 25: Endpoint Provisioning API

## Purpose

Demonstrates the UMRS endpoint provisioning REST API — managing
Mini-RiVault endpoint lifecycle including warm pool reuse,
multi-model co-location, LoRA adapter support, and the project-tier
MCP/RAG/egress mount contract.

## Prerequisites

- UMRS server running (HTTP API on port 8001)
- At least one model registered or pre-staged

## API Endpoints Covered

| Method | Path                       | Description                |
|--------|----------------------------|----------------------------|
| POST   | `/umrs/endpoints/provision`| Provision a new endpoint   |
| POST   | `/umrs/endpoints/up`       | Alias for provision        |
| GET    | `/umrs/endpoints`          | List active endpoints      |
| DELETE | `/umrs/endpoints/{id}`     | Stop an endpoint           |

## Mount Contract Covered

The demo creates sample mount assets and shows a provisioning payload with:

- `mounts.mcp_servers[]` targeting `/mcp-servers/demo-tools`
- `mounts.project_data` targeting `/project`
- `mounts.rag_index` targeting `/rag-index`
- `egress_domains` for the EgressGuard allowlist

Set `DEMO25_PROVISION=1` to submit the request. The default is read-only so
the full demo suite does not allocate a new endpoint unless explicitly asked.

## Run

```bash
bash run_demo.sh
```
