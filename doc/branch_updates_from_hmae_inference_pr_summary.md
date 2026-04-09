# PR-Style Summary: Current Branch vs `HMAE_inference`

See the full collaborator-facing detail in [branch_updates_from_hmae_inference.md](./branch_updates_from_hmae_inference.md).

## Scope

This branch is a large architectural and operational expansion of UMRS relative
to `HMAE_inference`. It includes repository restructuring, deployment changes,
durable job/runtime additions, a generalized vision serving model, expanded
model and workflow management, a larger CLI/API surface, and a new canonical
demo/documentation/test set.

It should be reviewed as a multi-subsystem platform branch, not as a single
feature branch.

## Primary Review Areas

### 1. Deployment and runtime contract

Representative files:

- [../src/umrs/orchestrator/site_config.py](../src/umrs/orchestrator/site_config.py)
- [../src/umrs/orchestrator/slurm_podman_backend.py](../src/umrs/orchestrator/slurm_podman_backend.py)
- [../deploy/rccs-cloud/site.yaml](../deploy/rccs-cloud/site.yaml)
- [../deploy/hpbd-atom/site.yaml](../deploy/hpbd-atom/site.yaml)
- [../deploy/harness_validate.py](../deploy/harness_validate.py)

What changed:

- deployment now depends on typed site config and wrapper-driven launch.
- Slurm+Podman is the preferred HPC path.
- harness validation is part of the repo and tests real profile combinations.

Reviewer focus:

- confirm that site defaults, wrapper behavior, and environment export remain
  consistent across deploy scripts and backend code.
- confirm that log retention, partition selection, and readiness assumptions are
  correct for both `rccs-cloud` and `hpbd-atom`.

### 2. Durable job lifecycle and endpoint management

Representative files:

- [../src/umrs/jobs/schema.py](../src/umrs/jobs/schema.py)
- [../src/umrs/jobs/service.py](../src/umrs/jobs/service.py)
- [../src/umrs/server/durable_job_runtime.py](../src/umrs/server/durable_job_runtime.py)
- [../src/umrs/orchestrator/durable.py](../src/umrs/orchestrator/durable.py)
- [../src/umrs/orchestrator/endpoint_registry.py](../src/umrs/orchestrator/endpoint_registry.py)

What changed:

- endpoint provisioning and related lifecycle actions are now persisted in a
  Postgres-backed durable job system.
- endpoint discovery/reuse/TTL status is explicit through a registry.

Reviewer focus:

- verify reconciliation, retry, cancellation, and cleanup semantics.
- verify that status transitions and endpoint reuse rules are safe.
- verify that the synchronous server integration with the async durable runtime
  does not create shutdown or state-leak issues.

### 3. Vision stack generalization

Representative files:

- [../src/umrs/vision_service/service.py](../src/umrs/vision_service/service.py)
- [../src/umrs/vision_service/bentoml_adapter.py](../src/umrs/vision_service/bentoml_adapter.py)
- [../src/umrs/vision_service/adapters/base.py](../src/umrs/vision_service/adapters/base.py)
- [../src/umrs/inference/hmae_inference.py](../src/umrs/inference/hmae_inference.py)
- [../src/umrs/inference/sam3_segmentation.py](../src/umrs/inference/sam3_segmentation.py)

What changed:

- the vision service exposes a generic model-type-driven inference path.
- HMAE and SAM3 are implemented as adapters under a common service contract.

Reviewer focus:

- verify adapter loading, model caching, S3/presigned input-output behavior,
  and compatibility with the existing compute embeddings flow.
- verify that HMAE behavior stayed correct while being generalized.

### 4. Workflow, Pipeline SDK, and plugin execution

Representative files:

- [../src/umrs/pipeline_sdk/workflow.py](../src/umrs/pipeline_sdk/workflow.py)
- [../src/umrs/pipeline_sdk/step.py](../src/umrs/pipeline_sdk/step.py)
- [../src/umrs/pipeline_sdk/dag.py](../src/umrs/pipeline_sdk/dag.py)
- [../src/umrs/plugins/user_workflow.py](../src/umrs/plugins/user_workflow.py)
- [../src/umrs/server/workflow_registry_handler.py](../src/umrs/server/workflow_registry_handler.py)

What changed:

- the branch adds a declarative workflow layer, step abstraction, DAG support,
  workflow registration, and user workflow execution.

Reviewer focus:

- verify safety and lifecycle around uploaded workflow code.
- verify step import/loading semantics, workflow validation, and DAG execution.
- verify that this layer composes correctly with the rest of the control plane.

### 5. Client/API/model registry surface expansion

Representative files:

- [../src/client_umrs.py](../src/client_umrs.py)
- [../src/umrs/api/models.py](../src/umrs/api/models.py)
- [../src/umrs/model_registry/store/crud.py](../src/umrs/model_registry/store/crud.py)
- [../src/umrs/tui/dashboard.py](../src/umrs/tui/dashboard.py)

What changed:

- CLI gained model upload/download, job commands, workflow upload/list/run, and
  the monitor dashboard.
- model registry moved toward explicit ACL/path/policy-backed behavior.

Reviewer focus:

- verify user/project/system scope enforcement.
- verify filesystem path validation and remote registration assumptions.
- verify that CLI behavior matches the server and REST API contracts.

## Behavior Changes Collaborators Should Expect

### Operational behavior

- site YAML is now the canonical deployment source of truth.
- durable job state requires a functional Postgres job-tracking backend.
- Slurm launch behavior is now heavily mediated by the site wrapper/backend path.

### Developer behavior

- code moved across multiple canonical packages, so older imports and mental
  models based on top-level modules are stale.
- the preferred review path is package-based rather than file-by-file at repo root.

### Feature behavior

- HMAE is preserved, but it is now one adapter inside a generic vision service.
- workflows/plugins are added on top of the existing UMRS pipeline rather than
  replacing the original data-processing flow.

## Notable Removals And Migrations

Key removals or demotions:

- [../src/embedding_worker.py](../src/embedding_worker.py)
- [../docker/k8s/embedding-worker.yaml](../docker/k8s/embedding-worker.yaml)
- [../src/rivault_cfg/](../src/rivault_cfg)
- [../src/umrs/models_loading.py](../src/umrs/models_loading.py)
- [../demos/slurm-dev/](../demos/slurm-dev)
- [../demos/backup/](../demos/backup)

Migration direction:

- runtime assets moved toward `deploy/*` plus `src/umrs/mini-rivault-llm/`
- server logic moved toward `src/umrs/server/`
- common helpers moved toward `src/umrs/shared/`
- model inference moved toward `src/umrs/inference/`
- search/RAG moved toward `src/umrs/rag/`

## Risk Areas

### 1. Cross-subsystem coupling

- site config, deploy wrappers, orchestration backend, and readiness behavior
  must remain aligned or launches will fail in ways that are hard to debug.

### 2. Durable runtime integration

- async runtime ownership inside a synchronous server process introduces startup,
  shutdown, and recovery paths that deserve extra scrutiny.

### 3. User-provided workflow/plugin execution

- uploaded/custom workflow behavior creates safety and environment-management
  concerns beyond the earlier static model flow.

### 4. Backward-compatibility assumptions

- the branch preserves a lot of old surfaces, but the internal reorganization is
  large enough that hidden import/runtime assumptions are a review hotspot.

## Evidence Of Broader Validation

Representative validation surface:

- [../tests/test_site_config.py](../tests/test_site_config.py)
- [../tests/test_durable_job_runtime.py](../tests/test_durable_job_runtime.py)
- [../tests/test_durable_orchestration.py](../tests/test_durable_orchestration.py)
- [../tests/test_slurm_podman_backend.py](../tests/test_slurm_podman_backend.py)
- [../tests/test_workflow_registry.py](../tests/test_workflow_registry.py)
- [../tests/test_user_workflow.py](../tests/test_user_workflow.py)
- [../tests/test_tui_dashboard.py](../tests/test_tui_dashboard.py)

The branch also adds deploy-side validation and a significantly larger doc/demo
surface, which suggests the author intended this to be an operationally usable
architecture rather than a partial prototype.

## Suggested Review Order

1. [../src/umrs/orchestrator/site_config.py](../src/umrs/orchestrator/site_config.py)
2. [../src/umrs/orchestrator/slurm_podman_backend.py](../src/umrs/orchestrator/slurm_podman_backend.py)
3. [../src/umrs/orchestrator/endpoint_registry.py](../src/umrs/orchestrator/endpoint_registry.py)
4. [../src/umrs/jobs/service.py](../src/umrs/jobs/service.py)
5. [../src/umrs/server/durable_job_runtime.py](../src/umrs/server/durable_job_runtime.py)
6. [../src/umrs/vision_service/service.py](../src/umrs/vision_service/service.py)
7. [../src/umrs/model_registry/store/crud.py](../src/umrs/model_registry/store/crud.py)
8. [../src/umrs/pipeline_sdk/workflow.py](../src/umrs/pipeline_sdk/workflow.py)
9. [../src/client_umrs.py](../src/client_umrs.py)
10. [../deploy/rccs-cloud/README.md](../deploy/rccs-cloud/README.md)
