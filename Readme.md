# UMRS — the infrastructure backbone

UMRS (Unified Multimodal Retrieval System) is the server half of the
R4 + UMRS platform. One UMRS server runs on the front-end of an institution's
cluster and is the single trusted backbone through which the R4 client
reaches accounts, compute, models and storage without touching any of them
directly. Design: [../docs/R4-UMRS-DESIGN.md](../docs/R4-UMRS-DESIGN.md)
(§"UMRS: the infrastructure backbone"). Where the code still differs from
the design, the row is in
[../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md](../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md).

> This branch (`ai4s-platform`) is consumed by the
> [ai4s-platform](https://github.com/mnagaso/ai4s-platform) repository as a
> git submodule. Boundary and branch policy:
> [doc/AI4S_PLATFORM_BOUNDARY.md](doc/AI4S_PLATFORM_BOUNDARY.md); merge safety:
> [MERGE_CONSTITUTION.md](MERGE_CONSTITUTION.md).

## Responsibilities (design → what ships, 2026-09-22)

| Responsibility (design) | Today | Gap |
|---|---|---|
| **Account management** — identity, authorisation, quotas, mapping to cluster accounts | user records as JSON files (`umrs-server admin enroll / token / slurm-account`), roles user/admin/service, durable projects (ADR-051), per-user agent-process and disk caps, per-user Slurm accounting (`--account=`), optional per-user SSH delegation | D-06 (accounts not in Postgres; no per-project GPU-hour budgets) |
| **Data transfer security** — ZeroMQ + CurveZMQ + ZAP only | ZeroMQ control + per-user data sockets with CURVE/ZAP (`session` mode on umrs-front) **and** a FastAPI HTTP sidecar behind the NGINX + auth-sidecar edge (JWT) | D-01/D-02 (Implemented by design amendment, ADR-059: dual transport accepted, gateway = UMRS's edge process); D-03/D-04/D-05 (closed by the 2026-09-24 wave) *(corrected 2026-10-19, C-4: the row read "decision pending")* |
| **Workflow job scheduling** on Slurm / Kubernetes / Podman | durable Job/Attempt engine; one `execution_authority` per job; Slurm + podman-compose (preferred), Kubernetes, Compose; workflow schema 2.0.0 with seven step kinds | D-08 (no Snakemake/Nextflow/CWL import) |
| **Remote agent session scheduling** | `frontend_session` job kind: rootless-Podman worker sandbox on the front-end with a lease-token rail and credential-injecting inference proxy (ADR-046) | D-09, D-48 (no CPU-only venue, no pause/resume) |
| **Self-deployment of models** in sandboxes on GPU nodes | `ENDPOINT_PROVISION` jobs launch vLLM + LiteLLM (text) and BentoML (vision: SAM3, MAE, HMAE) per Slurm job; model registry with scopes, versions, certification; models pre-staged, offline at boot | D-10 (VLM path undocumented), D-11 (no single inference contract) |
| **mini-RiVault** — firewalled Podman sandbox around every job, session and model | compute stack via podman-compose profiles; remote sessions in rootless Podman; Singularity backend still present; embedding/ingest compute not sandboxed | D-12, D-13 |
| **Raw → vector conversion** into Postgres + pgvector | `KNOWLEDGE_INGEST → UNDERSTAND → EMBED → VERIFY` jobs; `knowledge_node_embeddings`, `memory_vectors` (HNSW); image embeddings for tomography/vision | D-14, D-15 (video absent; embedding policy) |
| **Storage** — S3 by project, Postgres + pgvector, written only through UMRS | `S3_BACKEND` abstraction (SeaweedFS on umrs-front, MinIO elsewhere), attempt-scoped STS credentials, presigned URLs internal only; per-user `s3cfg` still delivered at login | D-16, D-17 |
| **MCP registry** | `/umrs/mcp/*` register / list / tools / call / health; capabilities from Hatch manifests | D-19, D-21, D-22 (placement, trust class, lifecycle accounting) |
| **Process model** — UMRS is the parent of everything it launches | Slurm jobs, K8s jobs, compose stacks and frontend sandboxes are all launched and reconciled by UMRS; status relayed to the owning R4 session | — |

The knowledge spine (graph registry, ontology, proposals, verification,
bundles), the data catalog, skills registry, A2L lineage and companion
trust also live here; they are the parts of the Scientific Catalog, which is
not yet one object (D-39).

## Repository layout

```text
src/
  server_umrs.py                 server entrypoint (ZeroMQ server + HTTP sidecar)
  client_umrs.py                 CLI entrypoint (`umrs`)
  umrs/
    core/                        ZeroMQ MessageQueue, CURVE/ZAP, threads, task manager
    server/                      http_sidecar (router registry), provisioning, handlers, SQL, runtime
    api/                         FastAPI families (26 mounted lanes: llm, rag, mcp, knowledge*, catalog, companion*, jobs, workflows, models, endpoints, files, projects, grants, venue, …)
    jobs/                        durable Job/Attempt schema, scheduler, reconciler, backends per JobType
    orchestrator/                Slurm+Podman (preferred), Slurm+Singularity (legacy), Kubernetes, Compose, hybrid, site profiles
    mini-rivault-llm/            compose stack + Slurm wrapper for GPU inference (vLLM, LiteLLM, NGINX, BentoML)
    venues/                      frontend venue: podman sandbox, egress net, nftables render
    infer_proxy/                 credential-injecting inference proxy for remote sessions
    model_registry/              model catalog, scopes/ACL, versions, conversion, certification, capabilities
    inference/                   embedding batch pipeline, model loaders (sentence-transformers, MAE, HMAE, ResNet, SAM3)
    vision_service/              BentoML vision service + adapters
    knowledge/, knowledge_graph/, knowledge_model/, scientific_memory/, embeddings_api/, search_api/, evidence/, experiment_registry/
                                 knowledge spine: ingest, understand, extract, ontology, graph registry, proposals, verify, loop, bundle export/import
    data_catalog/, asset_index/  data catalog (catalog_objects) + console read-model
    a2l/, artifact_registry/     provenance and lineage (ADR-034)
    skills/, mcp/                skill and MCP registries
    admin/, tenants/, companion_trust/, grants/, security/, shared/
                                 user records, (experimental) tenants, trust allowlist, asset grants, EgressGuard + scanners, config/storage/auth helpers
    workflow/, pipeline_sdk/, plugins/, recipes/
                                 workflow model, persistence, step execution, user pipelines
    migrations/                  0001–0100 SQL migrations (`umrs-server admin migrate`)
    tui/, tui_v2/, cli/          operator TUIs and CLI
deploy/                          site deployment roots (umrs-front, rccs, rccs-front, ai4s, bdrt, bdrt-front, octopus, octopus-front, hpbd-atom, local, templates)
docker/                          local standalone stack (server + Postgres + object store)
demos/, demos_private/, demos_sp8/   demo suites
doc/, docs/                      documentation (see map below)
tests/                           5,1xx tests (2026-09-21)
```

## Quick start

Package manager: [uv](https://docs.astral.sh/uv/). The submodule is a uv
workspace member of the parent repo (light defaults; `server` extra).

### Client only

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e src/
umrs --help
umrs ping                 # against a configured server profile (~/.config/umrs/)
umrs info                 # profile, mode (must be `direct` for umrs-front clusters), ZeroMQ status
```

Site setup scripts under `deploy/<site>/setup_client.sh` create the venv and
the connection profile. Normal users do not use `umrs` directly — they use
`r4`; `umrs` is the operator/diagnostic CLI and `umrs-server admin` the
trusted-host administration CLI.

### Local standalone stack

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker logs -f umrs-server
```

See [docker/README.md](docker/README.md).

### Site deployments

Deployment is by the parent repository's Ansible only
(`deploy/ansible/bootstrap.sh --profile <umrs-front | rccs-cloud | bdrt | octopus | atom-k8s>`);
never hand-edit a deployed checkout. Site roots: [deploy/umrs-front/](deploy/umrs-front/), [deploy/rccs/](deploy/rccs/)
(the Ansible profile is still named `rccs-cloud`; ADR-019 renamed the tree),
[deploy/hpbd-atom/](deploy/hpbd-atom/); operator guide [doc/admin_guide.md](doc/admin_guide.md).

## Inference paths

| Path | Use |
|---|---|
| `SentenceTransformer`, `TransformerEncoder`, `Resnet-50`, `MAE` | local CPU/GPU embedding models |
| `openai://…` | any OpenAI-compatible model (through the proxy for remote sessions) |
| `rivault://openai://…` | Slurm-launched vLLM/LiteLLM in a mini-RiVault |
| `rivault://bentoml://…` | Slurm-launched BentoML vision service |
| `user_script://…`, `user_workflow://…`, `pipeline://…` | user-defined workflow execution |

## Development

```bash
cd umrs && ../.venv/bin/python -m pytest tests -q
bash ../scripts/ci-lint-local.sh --umrs-changed     # the gate on the submodule diff (ADR-052)
```

Rules: [../CONSTITUTION.md](../CONSTITUTION.md) (foundation modules never
import `umrs.api` / `umrs.server`; every router mounted or deleted; complexity
≤ 9; baselines only shrink), [MERGE_CONSTITUTION.md](MERGE_CONSTITUTION.md),
[.github/instructions/](.github/instructions/) (`claude.md`, `umrs-project`,
`demo-execution`, harness rules).

## Documentation map

- [doc/umrs_overview.md](doc/umrs_overview.md) — runtime surfaces and packages
- [doc/AI4S_PLATFORM_BOUNDARY.md](doc/AI4S_PLATFORM_BOUNDARY.md) — branch topology and contracts preserved for the parent
- [DEVELOPMENT_OBJECTIVES.md](DEVELOPMENT_OBJECTIVES.md) — UMRS objectives and the design anchors referenced by docstrings
- [doc/zmq_api_contract.md](doc/zmq_api_contract.md) — ZeroMQ control/data protocol (string commands; versioning is gap D-04)
- [doc/admin_guide.md](doc/admin_guide.md), [doc/user_guide.md](doc/user_guide.md), [doc/onboarding.md](doc/onboarding.md), [doc/access_control.md](doc/access_control.md), [doc/multi_users.md](doc/multi_users.md), [deploy/](deploy/) site roots
- [doc/rag_implementation.md](doc/rag_implementation.md), [doc/set-up_pgvector.md](doc/set-up_pgvector.md), [doc/design_user_defined_models.md](doc/design_user_defined_models.md), [doc/s3_requirements_for_admin.md](doc/s3_requirements_for_admin.md)
- [src/umrs/mini-rivault-llm/README.md](src/umrs/mini-rivault-llm/README.md) — GPU runtime stack
- [demos/README.md](demos/README.md) — canonical demos (`00_onboarding` … `32_backbone_durability`, plus `99_cleanup`) *(corrected 2026-10-19, C-4: it used to read `01_health_check` … `22_monitor`)*
- Parent-side: [../docs/R4-UMRS-PLATFORM-OVERVIEW.md](../docs/R4-UMRS-PLATFORM-OVERVIEW.md) §3.3–3.8, ADRs under [../docs/adr/](../docs/adr/README.md)

Documents in `doc/` dated before 2026-09-22 describe the implementation as of
their date; verify against the code before citing.
