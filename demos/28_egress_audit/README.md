# Demo 28: Egress Audit

## Purpose

Demonstrates the UMRS egress audit API — querying and logging
outbound network events from Mini-RiVault instances. All egress
from MCP servers and tools is recorded with full request/response
metadata for compliance and security review.

The demo also ties audit events back to endpoint provisioning: `egress_domains`
become `EGRESS_DOMAIN_ALLOWLIST` for the EgressGuard profile, and mounted MCP
tools inherit proxy settings so outbound traffic remains auditable.

## API Endpoints Covered

| Method | Path                          | Description                |
|--------|-------------------------------|----------------------------|
| GET    | `/umrs/egress/audit`          | Query audit log entries    |
| POST   | `/umrs/egress/audit`          | Log an egress event        |

## Run

```bash
bash run_demo.sh
```
