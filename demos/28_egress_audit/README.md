# Demo 28: Egress Audit

## Purpose

Demonstrates the UMRS egress audit API — querying and logging
outbound network events from Mini-RiVault instances. All egress
from MCP servers and tools is recorded with full request/response
metadata for compliance and security review.

## API Endpoints Covered

| Method | Path                          | Description                |
|--------|-------------------------------|----------------------------|
| GET    | `/umrs/egress_audit/audit`    | Query audit log entries    |
| POST   | `/umrs/egress_audit/log`      | Log an egress event        |

## Run

```bash
bash run_demo.sh
```
