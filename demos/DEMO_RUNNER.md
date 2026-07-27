# UMRS Demo Runner & Doctor Guide

## Overview

The UMRS platform now includes automated tools for demo execution and system diagnostics.

## Demo Runner

### Quick Start

```bash
# List all available demos
./umrs/demos/run_demo.sh

# Run a specific demo
./umrs/demos/run_demo.sh 01_chat
./umrs/demos/run_demo.sh 05_upload

# Use custom profile
R4_PROFILE=custom-profile ./umrs/demos/run_demo.sh 01_chat
```

### How It Works

1. Checks that `umrs` is installed and reachable
2. Pings the server to verify connectivity
3. Lists visible S3 buckets
4. Lists available models
5. Shows demo header and instructions from README.md

### Prerequisites

- `umrs` installed in PATH or accessible via `UMRS_CLIENT` env var
- Active authentication profile (`r4 auth login` completed)
- `umrs` in PATH: `pip install -e /path/to/umrs/src/`

## Doctor Command

### Usage

```bash
# Run all health checks
umrs doctor

# Check specific subsystem
umrs doctor rag
umrs doctor storage
umrs doctor auth
umrs doctor models
umrs doctor workflows
umrs doctor jobs
umrs doctor demos
```

### Check Results

| Status | Icon | Meaning |
|--------|------|---------|
| OK | ✓ | Everything working |
| Warning | ⚠ | Potential issue, but non-blocking |
| Error | ✗ | Problem needs attention |
| Missing | - | Feature not available |

### Example Output

```
UMRS Doctor Report
========================================
  OK: 5
  Warnings: 2
  Errors: 0

  ✓ Server: Server is reachable
  ✓ Authentication: Profile rccs-cloud configured
  ✓ Storage: Connected, 3 buckets
  ✓ Models: 2 system models (1 embeddings)
  ✓ Workflows: Workflow system operational
  ✓ Jobs: Job system responsive
  ⚠ RAG: No pipelines
    Suggestion: Run: r4 rag ingest create --project default
  ⚠ Demos: 0/30 ready (bucket access needed for many demos)
    Suggestion: Ensure bucket aliases and experience configs are present
```

## Integration Tests

### Run All Demos

```bash
./umrs/tests/run_integration_tests.sh
```

### Run Specific Demos

```bash
./umrs/tests/run_integration_tests.sh 01_chat 02_rag 05_upload
```

### Results

Integration test results are saved to `umrs/tests/results/<demo_name>.txt`.

## Self-Healing

### Bucket Alias Auto-Discovery

The `discovery.py` module provides `smart_bucket_lookup()` which automatically maps bucket aliases to actual bucket paths:

```python
from umrs.client.discovery import smart_bucket_lookup

# Given a ctx with profile info, this finds the bucket for "user1-textfiles"
bucket_path = smart_bucket_lookup(ctx, "user1-textfiles")
```

### Environment Healer

Auto-create missing resources:

```python
from umrs.client.discovery import EnvironmentHealer

healer = EnvironmentHealer(ctx)
healer.heal()  # Auto-creates missing bucket aliases and RAG pipelines
```

## Troubleshooting

### Demo fails with "bucket alias not found"

1. Run `umrs bucket list` to see available buckets
2. Check if bucket aliases are configured in your experience file
3. Run `umrs doctor storage` for diagnostics
4. Create missing bucket aliases with admin access

### Demo fails with "no models"

1. Run `umrs doctor models` to check available models
2. Contact admin to stage models if none visible
3. Verify model download jobs completed successfully

### No RAG pipelines

1. Run `umrs doctor rag`
2. Create pipeline: `r4 rag ingest create --project default`
3. Verify pgvector is installed in RAG database

### Server unreachable

1. Check server URL in your profile: `umrs info`
2. Verify network connectivity to the gateway
3. Check if the server is running and not in maintenance mode
