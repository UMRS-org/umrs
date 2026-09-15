# UMRS Branch Sync Checklist

## Branch Topology

| Branch | Purpose | Upstream Changes |
|--------|---------|------------------|
| `feature/user-defined-workflows` | Core UMRS (independent development) | UMRS team |
| `ai4s-platform` | UMRS + ai4s-platform integration layer | Both teams |

`ai4s-platform` is based on `feature/user-defined-workflows` plus integration-only additions.

## ai4s-platform Integration Files (Do Not Exist in `feature/user-defined-workflows`)

These files are **owned exclusively** by the ai4s-platform integration layer:

- `src/umrs/server/http_sidecar.py` — FastAPI HTTP sidecar (port 8082)
- `src/umrs/api/llm.py` — LLM proxy router (Gateway-facing)
- `src/umrs/api/rag.py` — RAG proxy router (Gateway-facing)
- `src/umrs/api/mcp.py` — MCP registry/invocation router
- `src/umrs/api/workflows.py` — Workflow CRUD router
- `src/umrs/mcp/` — MCP server registry (models, schema, store)
- `src/umrs/migrations/0001_mcp_registry.sql` — MCP migration
- `tests/test_*_api.py`, `tests/test_*_proxy.py` — Integration tests

## Syncing `feature/user-defined-workflows` → `ai4s-platform`

When the UMRS team pushes new commits to `feature/user-defined-workflows`:

```bash
cd umrs/
git checkout ai4s-platform
git fetch origin
git merge origin/feature/user-defined-workflows
# Resolve conflicts (integration files should rarely conflict)
git push origin ai4s-platform
```

## Conflict-Prone Files

These files have both UMRS-core and ai4s-platform changes — resolve carefully:

| File | UMRS Owns | ai4s-platform Adds |
|------|-----------|-------------------|
| `src/umrs/api/endpoints.py` | Core endpoint CRUD | `metadata`, `partition_override` params |
| `src/umrs/jobs/adapters.py` | Backend adapters | `_resolve_registered_model` |
| `src/umrs/orchestrator/durable.py` | Durable job engine | `model_store`, `path_validator` params |
| `src/umrs/server/config.py` | Core config | `VISION_STACK`, `ENABLE_PRETTY_PRINT` |
| `src/umrs/server/rivault_launcher.py` | Launcher core | BentoML reuse, VISION_STACK selection |

## Rules

1. **Never duplicate UMRS functionality** in Gateway or r4 — keep those as proxy/policy layers.
2. **Integration files are additive** — they should import from UMRS core, not modify it.
3. **When in doubt, prefer `feature/user-defined-workflows`** for UMRS-core logic.
4. **Test after sync**: run both UMRS tests and ai4s-platform E2E tests.
