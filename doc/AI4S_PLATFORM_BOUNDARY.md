# ai4s-platform Integration Boundary

This document describes the boundary between UMRS core development
(`feature/user-defined-workflows`) and the ai4s-platform integration layer
(`ai4s-platform` branch). It helps UMRS developers understand which parts
of the codebase are consumed downstream and what contracts to preserve.

## Branch Relationship

```
feature/user-defined-workflows   ← UMRS core (this branch)
        │
        │  merged into
        ▼
ai4s-platform                    ← UMRS core + integration layer
        │
        │  consumed as git submodule by
        ▼
mnagaso/ai4s-platform (main)     ← Full platform (Gateway, aisci SDK, tests)
```

- `feature/user-defined-workflows` is the **upstream** branch. UMRS
  development happens here independently.
- `ai4s-platform` is a **downstream consumer**. It merges from this branch
  and adds integration-only files (HTTP sidecar, Gateway-facing API routers).
- UMRS developers **do not need to modify** anything on `ai4s-platform` —
  that branch is maintained by the platform team.

## What ai4s-platform Adds (Not Present Here)

The `ai4s-platform` branch adds these files **on top of** this branch:

| Path | Purpose |
|------|---------|
| `src/umrs/server/http_sidecar.py` | FastAPI HTTP server (port 8082) that Gateway calls |
| `src/umrs/api/llm.py` | Proxies LLM requests from Gateway to LiteLLM |
| `src/umrs/api/rag.py` | Proxies RAG requests from Gateway |
| `src/umrs/api/mcp.py` | MCP server registry and tool invocation |
| `src/umrs/api/workflows.py` | Workflow CRUD for Gateway |
| `src/umrs/mcp/` | MCP registry models, schema, store |
| `src/umrs/migrations/0001_mcp_registry.sql` | Database migration for MCP |
| `tests/test_*_api.py`, `tests/test_*_proxy.py` | Integration tests |
| `SYNC_CHECKLIST.md` | Sync procedure documentation |

These files **do not exist** on `feature/user-defined-workflows` and should
not be created here.

## Contracts This Branch Must Preserve

The ai4s-platform integration layer imports from UMRS core modules. Breaking
these contracts will cause merge conflicts or runtime failures on the
`ai4s-platform` branch.

### 1. Endpoint API (`src/umrs/api/endpoints.py`)

The integration layer adds parameters to existing endpoint functions. Keep
the existing function signatures stable:

```python
# These functions are extended downstream — keep signatures compatible
def create_endpoint(...)    # ai4s-platform adds `metadata` param
def list_endpoints(...)     # ai4s-platform adds `partition_override`, `nodelist_hint`
```

**Rule**: Adding new parameters with defaults is safe. Removing or renaming
existing parameters is a breaking change.

### 2. Durable Job Engine (`src/umrs/orchestrator/durable.py`)

```python
# ai4s-platform extends backend_factory with additional kwargs
async def launch_durable_job(backend_factory, ...)
```

**Rule**: The `backend_factory` callable signature and `launch_durable_job`
interface must remain stable.

### 3. Config (`src/umrs/server/config.py`)

```python
# ai4s-platform adds:
#   VISION_STACK = "default"
#   ENABLE_PRETTY_PRINT = False
# to the config module
```

**Rule**: Don't remove or redefine constants that may be added downstream.
Adding new constants here is safe.

### 4. Job Adapters (`src/umrs/jobs/adapters.py`)

```python
# ai4s-platform adds _resolve_registered_model() function
# Keep imports stable (ModelRef, pathlib, etc.)
```

### 5. Rivault Launcher (`src/umrs/server/rivault_launcher.py`)

```python
# ai4s-platform adds BentoML job reuse logic and VISION_STACK selection
# Keep function signatures of:
#   get_slurm_script_from_host()
#   get_slurm_node(script, model_type_key, job_id_to_use=..., sbatch_env=..., job_metadata=...)
```

### 6. Model Registry (`src/umrs/model_registry/`)

```python
# Used by integration layer for model staging/download
# Keep the registry handler interface stable
```

## What You Can Safely Change

- **Internal implementations** of any function (as long as the signature
  is preserved)
- **New modules** — adding files here does not affect the integration layer
- **New parameters with defaults** on existing functions
- **Test files** — UMRS tests and ai4s-platform tests are independent
- **Deployment configs** under `deploy/` — each branch has its own
- **Demo scripts** under `demos/` and `demos_private/`

## What Requires Coordination

If you need to make changes that affect the contracts above:

1. **Rename/remove a function** that's listed above → Coordinate with
   platform team first
2. **Change `__init__.py` exports** in `api/`, `jobs/`, `orchestrator/`,
   `server/` → Check if `ai4s-platform` branch imports them
3. **Restructure modules** (move files between packages) → Will break
   imports on `ai4s-platform`

## Sync Process

When you push commits to `feature/user-defined-workflows`, the platform
team periodically merges them into `ai4s-platform`:

```bash
# Platform team runs:
git checkout ai4s-platform
git merge origin/feature/user-defined-workflows
# Resolve any conflicts in the integration-extended files
git push origin ai4s-platform
```

No action needed from UMRS developers — just be aware of the contracts.

## Questions?

Contact the ai4s-platform team or check the full integration policy at:
`ai4s-platform/docs/INTEGRATION_POLICY.md`
