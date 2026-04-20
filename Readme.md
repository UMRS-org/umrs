# UMRS — Unified Multimodal Retrieval System

UMRS is a config-driven platform for ingesting files into S3-compatible
storage, processing them with text, vision, and workflow-based pipelines, and
storing embeddings or derived results in PostgreSQL/pgvector. It supports
local CPU execution, OpenAI-compatible remote models, and Slurm-backed
Mini-RiVault launches for GPU inference.

## What UMRS Includes

- A ZeroMQ-based control plane plus FastAPI service surface.
- Text, image, and video workflows backed by MinIO/S3 and PostgreSQL.
- Local models, OpenAI-compatible models, and Slurm-launched Mini-RiVault
    runtimes for vLLM and BentoML-based vision inference.
- A model registry, workflow upload/registration flow, durable job tracking,
    and a TUI jobs monitor.
- Public demos `01` to `22` plus private/demo-specific variants in
    `demos_private/`.
- Site-driven deployment under `deploy/` with typed `site.yaml` config.

## Current Architecture

At a high level, the current branch organizes UMRS into these layers:

- `umrs-client` / `src/client_umrs.py` for CLI operations.
- `src/server_umrs.py` as the legacy server entrypoint and runtime host.
- `src/umrs/server/` for server-owned handlers, provisioning, SQL, and runtime.
- `src/umrs/orchestrator/` for endpoint lifecycle, readiness, durable launch,
    and Slurm+Podman orchestration.
- `src/umrs/inference/` for model implementations and loading helpers.
- `src/umrs/vision_service/` for the generic BentoML vision service and
    pluggable adapters such as `sam3` and `hmae`.
- `src/umrs/model_registry/`, `src/umrs/jobs/`, `src/umrs/pipeline_sdk/`,
    `src/umrs/plugins/`, and `src/umrs/rag/` for the higher-level control plane.

## Quick Start

> **ai4s-platform integration**: This branch is consumed by the
> [ai4s-platform](https://github.com/mnagaso/ai4s-platform) project as a
> git submodule. See [doc/AI4S_PLATFORM_BOUNDARY.md](doc/AI4S_PLATFORM_BOUNDARY.md)
> for contracts and development boundaries.

### Client only

The recommended package manager for the repo is
[uv](https://docs.astral.sh/uv/).

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e src/

umrs-client --help
```

If you already have a configured UMRS server reachable from your machine:

```bash
umrs-client ping
umrs-client list_buckets
```

Site-specific setup scripts such as `deploy/rccs-cloud/setup_client.sh`
create the venv, install the package, and place the expected config files
under `~/.config/umrs/`.

### Local Docker deployment

For a local standalone stack with UMRS server + Postgres + MinIO:

```bash
docker compose -f docker/docker-compose.yml up -d --build
docker logs -f umrs-server
```

See [docker/README.md](docker/README.md) for the exact container commands and
limitations of the Docker-only setup.

### HPC / site deployments

For Slurm-backed GPU inference with Mini-RiVault, use the site deployment docs:

- [deploy/rccs-cloud/README.md](deploy/rccs-cloud/README.md)
- [doc/admin_guide.md](doc/admin_guide.md)

## Main User-Facing Capabilities

### Data operations

- Upload files and directories to configured buckets.
- Export embeddings.
- Check, update, recompute, and reload server configuration.

### Model and workflow control plane

- Register, upload, list, and download models.
- Upload and list workflows.
- Run durable jobs and cancel them explicitly.
- Use the live TUI monitor for job inspection.

### Inference paths

| Path | Typical use |
|---|---|
| `SentenceTransformer`, `TransformerEncoder`, `Resnet-50`, `MAE` | local CPU inference |
| `openai://...` | any OpenAI-compatible remote model |
| `rivault://openai://...` | Slurm-launched vLLM/LiteLLM inference |
| `rivault://bentoml://...` | Slurm-launched vision inference via BentoML |
| `user_script://...`, `user_workflow://...`, `pipeline://...` | user-defined workflow execution |

## Demos

The canonical demos live in [demos/README.md](demos/README.md). They are
ordered from basic health checks to advanced workflow, jobs, and monitoring
scenarios.

Representative public demos:

- [demos/01_health_check/](demos/01_health_check/)
- [demos/02_text_embedding/](demos/02_text_embedding/)
- [demos/03_sam3_segmentation/](demos/03_sam3_segmentation/)
- [demos/04_vision_inference/](demos/04_vision_inference/)
- [demos/05_rag_search/](demos/05_rag_search/)
- [demos/07_model_registry/](demos/07_model_registry/)
- [demos/08_user_model/](demos/08_user_model/)
- [demos/15_complex_dag/](demos/15_complex_dag/)
- [demos/20_durable_jobs/](demos/20_durable_jobs/)
- [demos/22_monitor/](demos/22_monitor/)

Private or proprietary variants live under [demos_private/README.md](demos_private/README.md).

## Repository Layout

```text
src/
    pyproject.toml                 Python package definition
    client_umrs.py                 CLI entrypoint (`umrs-client` when installed)
    server_umrs.py                 server entrypoint
    umrs/
        api/                         FastAPI services
        core/                        message queues, task manager, command/status types
        shared/                      common config, storage, modality, progress, helpers
        server/                      provisioning, handlers, SQL, durable runtime
        inference/                   model implementations and loading helpers
        vision_service/              generic BentoML vision service + adapters
        model_registry/              model catalog, ACL, policies, storage refs
        orchestrator/                endpoint registry, readiness, Slurm+Podman backend
        jobs/                        durable jobs schema, store, scheduler, monitor
        pipeline_sdk/                pipeline/workflow primitives and connectors
        plugins/                     plugin and user workflow execution helpers
        rag/                         retrieval, reranking, context assembly
        tui/                         Rich-based jobs dashboard
        mini-rivault-llm/            compose/runtime assets for Slurm GPU inference

deploy/                          site-specific deployment assets and wrappers
demos/                           canonical public demos (01-22)
demos_private/                   private/proprietary demo variants
doc/                             architecture, admin, user, and design docs
docker/                          local standalone Docker deployment
tests/                           regression and unit tests
```

## Documentation Map

Core docs:

- [doc/admin_guide.md](doc/admin_guide.md)
- [doc/user_guide.md](doc/user_guide.md)
- [doc/demo_guide.md](doc/demo_guide.md)
- [doc/umrs_overview.md](doc/umrs_overview.md)
- [doc/zmq_api_contract.md](doc/zmq_api_contract.md)
- [doc/design_user_defined_models.md](doc/design_user_defined_models.md)
- [doc/rag_implementation.md](doc/rag_implementation.md)

Branch comparison docs created from the current branch:

- [doc/branch_updates_from_hmae_inference.md](doc/branch_updates_from_hmae_inference.md)
- [doc/branch_updates_from_hmae_inference_executive_summary.md](doc/branch_updates_from_hmae_inference_executive_summary.md)
- [doc/branch_updates_from_hmae_inference_pr_summary.md](doc/branch_updates_from_hmae_inference_pr_summary.md)

Deployment and runtime docs:

- [docker/README.md](docker/README.md)
- [deploy/rccs-cloud/README.md](deploy/rccs-cloud/README.md)
- [src/umrs/mini-rivault-llm/README.md](src/umrs/mini-rivault-llm/README.md)