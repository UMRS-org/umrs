# ai4s-platform Integration Boundary

**Updated 2026-09-22.** This document describes how the `umrs` repository
relates to the parent R4 + UMRS platform repository and which contracts the
two sides preserve. Branch policy: parent [ADR-010](../../docs/adr/010-umrs-submodule-branch-policy.md).

## Branch topology (2026-09)

```
UMRS-org/umrs  main                 ← upstream UMRS core (independent development)
        │  merged into (MERGE_CONSTITUTION.md governs every merge)
        ▼
UMRS-org/umrs  ai4s-platform        ← canonical deployment branch: UMRS core + platform integration
        │  mirrored to
        ▼
mnagaso/umrs   ai4s-platform        ← fork, push-mirror (`origin` of the local umrs checkout)

mnagaso/ai4s-platform  main         ← the platform: R4 client, edge, contracts, deploy, docs;
                                       `.gitmodules` → git@github.com:UMRS-org/umrs.git, branch ai4s-platform
```

- `ai4s-platform` is `main` plus everything the platform needs from its
  backbone. It is developed inside the parent's `umrs/` submodule (local
  remotes: `origin` = fork `mnagaso/umrs`, `upstream` = canonical
  `UMRS-org/umrs`); commits land on `ai4s-platform`, are pushed to both, then
  the parent bumps its pointer (docs that cite `umrs/` paths land with or
  after the bump). The parent's `.gitmodules` points at `UMRS-org/umrs`
  since 2026-09-15 (ADR-010 rollout block).
- Merges flow `main → ai4s-platform`. Fixes to shared code go back to `main`
  by PR ([MERGE_CONSTITUTION.md §6](../MERGE_CONSTITUTION.md)).
- Older docs name the upstream remote `ejeannot/main` and a feature branch
  `feature/user-defined-workflows`; both are historical names for `main`.

## What `ai4s-platform` adds on top of `main`

The integration layer is no longer a handful of files; it is most of what
the parent's design calls the backbone. The main additions, by area:

| Area | Paths |
|---|---|
| HTTP sidecar + API families | `src/umrs/server/http_sidecar.py` (router registry), `src/umrs/api/` (llm, rag, mcp, knowledge*, catalog, companion*, jobs, workflows, models, endpoints, files, projects, grants, skills, vision, a2l, audit, ops, zmq_sessions, …) |
| Durable jobs and executor classes | `src/umrs/jobs/` (knowledge, companion worker, frontend session, research loop, vision infer, batch infer, model eval, fine-tune backends) |
| Orchestration | `src/umrs/orchestrator/slurm_podman_backend.py`, hybrid composer, site profiles, `mini-rivault-llm/` compose stack |
| Knowledge spine and catalog parts | `src/umrs/knowledge/`, `knowledge_graph/`, `knowledge_model/`, `scientific_memory/`, `embeddings_api/`, `search_api/`, `evidence/`, `experiment_registry/`, `data_catalog/`, `asset_index/`, `a2l/`, `artifact_registry/` |
| Registries and policy | `src/umrs/mcp/`, `skills/`, `companion_trust/`, `grants/`, `api/projects.py`, `api/agent_quota.py`, `admin/` |
| Venues and proxy | `src/umrs/venues/`, `src/umrs/infer_proxy/`, `security/seal.py`, `sealed_sources.py` |
| Session CURVE wiring | `core/session_curve_provider.py` init in `server_umrs.py`, `core/MessageQueue._apply_socket_security`, `api/zmq_sessions.py` |
| Migrations | `src/umrs/migrations/0001…0100` |
| Deployment roots | `deploy/umrs-front/`, `deploy/ai4s/`, `deploy/bdrt*/`, `deploy/octopus*/`, `deploy/rccs*/`, `deploy/_common/` |
| Tests | `tests/test_*_api.py`, `tests/test_*_proxy.py`, `tests/test_surface_characterization.py`, `tests/test_migration_chain.py`, … |

## Contracts `main` must preserve

The integration layer extends these `main` surfaces; renaming or removing
them breaks `ai4s-platform` at merge or at runtime.

1. **Endpoint API** (`src/umrs/api/endpoints.py`): `create_endpoint(...)`
   and `list_endpoints(...)` keep their signatures (downstream adds
   `metadata`, `partition_override`, `nodelist_hint` with defaults).
2. **Durable job engine** (`src/umrs/orchestrator/durable.py`):
   `launch_durable_job(backend_factory, ...)` and the `backend_factory`
   callable signature stay stable.
3. **Config** (`src/umrs/server/config.py`, `src/umrs/shared/config.py`):
   never remove or redefine constants; never assign a module-level constant
   twice (MERGE_CONSTITUTION §2.1); resolution never yields `None` (§2.2).
4. **Job adapters** (`src/umrs/jobs/adapters.py`): imports (`ModelRef`,
   `pathlib`, …) stay stable; downstream adds `_resolve_registered_model()`.
5. **Rivault launcher** (`src/umrs/server/rivault_launcher.py`):
   `get_slurm_script_from_host()`, `get_slurm_node(script, model_type_key,
   job_id_to_use=..., sbatch_env=..., job_metadata=...)` keep their signatures.
6. **Model registry** (`src/umrs/model_registry/`): registry handler
   interface stays stable.
7. **ZeroMQ core** (`src/umrs/core/MessageQueue.py`): each queue owns its
   `zmq.Context`; ZAP authenticators are tracked and stopped; the two halves
   of session-CURVE (provider init + socket application) stay wired
   (MERGE_CONSTITUTION §2.3–2.7).
8. **HTTP reload context**: `set_reload_context()` is called in
   `server_umrs.py` after initial config and before the sidecar starts (§2.8).

## Contracts the parent relies on (and validates)

- Foundation modules (`core`, `jobs`, `model_registry`, `mcp`, `rag`,
  `pipeline_sdk`, `orchestrator`, `security`, `shared`) never import
  `umrs.api` or `umrs.server`; `umrs/api/__init__.py` imports no routers
  (parent `tools/import_boundary_validator.py`, ADR-053).
- Every `umrs/api` router module is mounted by the sidecar registry or does
  not exist (`tests/test_surface_characterization.py`; `/umrs/context` is
  the one named dormant family, ADR-056).
- Lint, complexity (≤ 9) and fingerprint baselines apply to the submodule
  diff between pointers (`bash scripts/ci-lint-local.sh --umrs-changed`, ADR-052).
- `umrs` never imports `r4` or `gateway`; shared DTOs come from `r4-contracts`.

## Sync process

```bash
cd umrs && git checkout ai4s-platform
git fetch upstream && git merge upstream/main     # resolve per MERGE_CONSTITUTION §3
# run MERGE_CONSTITUTION §5 post-merge validation, then:
git push upstream ai4s-platform && git push origin ai4s-platform   # canonical (UMRS-org) + fork mirror per ADR-010
cd .. && git add umrs && git commit -m "chore(umrs): bump to <sha> — <why>"
```

Remote names differ per checkout; the rule is the branch graph above, not
the remote name.
