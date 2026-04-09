# Branch Update Summary: `feature/user-defined-workflows` vs `HMAE_inference`

## Purpose

This document summarizes the updates that exist on the current branch relative to
[`HMAE_inference`](https://github.com/ejeannot/umrs/tree/HMAE_inference).
It is written for collaborator review, so it focuses on the *logical changes*
and the *actual code locations* that implement them, rather than only listing
 commit messages.

This branch is much larger than a narrow "user-defined workflow" feature. It is
effectively a platform-expansion branch that reorganizes the repository,
hardens deployment, adds durable orchestration, generalizes the vision stack,
expands the model/workflow/job control plane, and replaces the old demo layout
with a canonical one.

## Comparison Basis

- Base branch: `origin/HMAE_inference`
- Target branch: current `feature/user-defined-workflows` HEAD
- Comparison method:
  - `git diff origin/HMAE_inference...HEAD`
  - `git log origin/HMAE_inference..HEAD --no-merges`
  - direct inspection of the changed implementation files

## Branch Size

| Metric | Value |
|---|---:|
| Unique commits since `HMAE_inference` | 149 |
| Files changed | 476 |
| Insertions | 53,230 |
| Deletions | 16,576 |

## Executive Summary

Compared with `HMAE_inference`, this branch changes UMRS in five major ways:

1. It converts deployment from a collection of hardcoded/runtime-specific files
   into a site-driven system based on `site.yaml`, typed site config loading,
   and dedicated deploy wrappers for `rccs-cloud` and `hpbd-atom`.
2. It adds a durable orchestration and job-tracking layer backed by Postgres,
   with endpoint registry support, status polling, reconciliation, and cleanup.
3. It generalizes the vision stack from HMAE/SAM3-specific paths into a generic
   BentoML vision service with pluggable adapters and reusable compute bindings.
4. It adds a substantial extensibility layer: model registry improvements,
   workflow upload/registration, the Pipeline SDK, plugin-backed execution, and
   a much larger CLI/API surface.
5. It reorganizes the repo into clearer packages (`core`, `shared`, `server`,
   `inference`, `rag`, `pipeline_sdk`, `vision_service`, etc.), replaces the
   old demo sets with canonical demos `01` to `22`, and adds broader docs/tests.

## Detailed Updates By Area

### 1. Repository Structure And Packaging Were Reorganized

The source tree was reshaped into clearer subpackages, replacing the flatter
and more ad hoc layout present around `HMAE_inference`.

Representative code:

- [`../src/pyproject.toml`](../src/pyproject.toml)
- [`../src/uv.lock`](../src/uv.lock)
- [`../src/umrs/core/`](../src/umrs/core)
- [`../src/umrs/shared/`](../src/umrs/shared)
- [`../src/umrs/server/`](../src/umrs/server)
- [`../src/umrs/inference/`](../src/umrs/inference)
- [`../src/umrs/rag/`](../src/umrs/rag)

Logical changes:

- Message-queue/task machinery moved under `src/umrs/core/`.
- Common helpers were split into `src/umrs/shared/`, including config,
  modality, storage, progress, resource tracking, result diffing, and SQL
  helper utilities.
- Server-only functionality now lives in `src/umrs/server/` rather than being
  mixed into top-level modules.
- Inference code now has a dedicated `src/umrs/inference/` package, separating
  model implementations from orchestration and CLI concerns.
- RAG code moved into `src/umrs/rag/`.
- `uv`/editable-package installation was formalized with `src/pyproject.toml`
  and `src/uv.lock`.

Why it matters:

- This is not just cosmetic. It makes the codebase more modular and easier to
  evolve, and it is the foundation for the later deploy/orchestration/model/
  workflow changes in this branch.

### 2. Deployment Became Site-Driven Instead Of Hardcoded

The branch introduces a canonical `site.yaml` model and deploy directories for
real sites instead of relying on scattered env defaults and older runtime files.

Representative code:

- [`../src/umrs/orchestrator/site_config.py`](../src/umrs/orchestrator/site_config.py)
- [`../src/umrs/orchestrator/site-config.schema.json`](../src/umrs/orchestrator/site-config.schema.json)
- [`../deploy/rccs-cloud/site.yaml`](../deploy/rccs-cloud/site.yaml)
- [`../deploy/hpbd-atom/site.yaml`](../deploy/hpbd-atom/site.yaml)
- [`../deploy/rccs-cloud/start.sh`](../deploy/rccs-cloud/start.sh)
- [`../deploy/rccs-cloud/stop.sh`](../deploy/rccs-cloud/stop.sh)
- [`../deploy/hpbd-atom/start.sh`](../deploy/hpbd-atom/start.sh)
- [`../deploy/hpbd-atom/stop.sh`](../deploy/hpbd-atom/stop.sh)

Logical changes:

- `SiteConfig` now formalizes scheduler, runtime, storage, network, vLLM, and
  database settings as typed dataclasses.
- `SiteConfig.to_env()` and `inject_env()` bridge the new typed config with the
  older env-var-oriented code paths, so newer site YAMLs can drive older code
  without a full rewrite.
- Real deploy targets now live under `deploy/rccs-cloud/` and
  `deploy/hpbd-atom/`, each with client config, tunnel scripts, health checks,
  setup scripts, start/stop scripts, env examples, and site YAML.
- The runtime defaults that used to be spread across wrapper scripts and
  old `rivault_cfg` assets are now centralized.

Why it matters:

- Deployment becomes reproducible and reviewable by site.
- Collaborators no longer need to infer runtime behavior from many unrelated
  shell scripts and hardcoded paths.

### 3. Mini-RiVault Was Turned Into A Canonical Source Tree Component

The branch migrates the Mini-RiVault assets out of the old demo/runtime area
and makes them a first-class part of the main source tree.

Representative code:

- [`../src/umrs/mini-rivault-llm/serve-model.yaml`](../src/umrs/mini-rivault-llm/serve-model.yaml)
- [`../src/umrs/mini-rivault-llm/serve-model.slurm`](../src/umrs/mini-rivault-llm/serve-model.slurm)
- [`../src/umrs/mini-rivault-llm/x_nvidia-override.yaml`](../src/umrs/mini-rivault-llm/x_nvidia-override.yaml)
- [`../src/umrs/mini-rivault-llm/nginx_fs/umrs.conf`](../src/umrs/mini-rivault-llm/nginx_fs/umrs.conf)
- [`../deploy/rccs-cloud/serve-model.slurm`](../deploy/rccs-cloud/serve-model.slurm)
- [`../deploy/hpbd-atom/serve-model.slurm`](../deploy/hpbd-atom/serve-model.slurm)

Logical changes:

- The Slurm wrapper and compose stack are now treated as canonical runtime
  assets instead of being hidden under the old `demos/slurm-dev` layout.
- The runtime is profile-driven (`harness`, `vllm`, `vision-bentoml`, and
  related services) rather than a single fixed stack.
- The wrappers are site-aware and can derive defaults from `site.yaml`.
- GPU/runtime behavior is handled at wrapper level, including output log
  capture, profile selection, and compute-node launch behavior.

Why it matters:

- The branch moves UMRS away from a "demo environment" posture and toward a
  supported deploy/runtime model.

### 4. Orchestration Became Registry-Backed And Backend-Abstracted

Compared with `HMAE_inference`, endpoint provisioning and backend execution are
substantially more structured.

Representative code:

- [`../src/umrs/orchestrator/base.py`](../src/umrs/orchestrator/base.py)
- [`../src/umrs/orchestrator/factory.py`](../src/umrs/orchestrator/factory.py)
- [`../src/umrs/orchestrator/service.py`](../src/umrs/orchestrator/service.py)
- [`../src/umrs/orchestrator/endpoint_registry.py`](../src/umrs/orchestrator/endpoint_registry.py)
- [`../src/umrs/orchestrator/registry_backend.py`](../src/umrs/orchestrator/registry_backend.py)
- [`../src/umrs/orchestrator/slurm_podman_backend.py`](../src/umrs/orchestrator/slurm_podman_backend.py)
- [`../src/umrs/orchestrator/readiness.py`](../src/umrs/orchestrator/readiness.py)
- [`../src/umrs/orchestrator/hybrid/`](../src/umrs/orchestrator/hybrid)

Logical changes:

- Endpoints are now explicit registry entries keyed by project/model and moved
  through statuses such as `PENDING`, `PROVISIONING`, `STARTING`, `READY`,
  `DEGRADED`, and `FAILED`.
- TTL and warm reuse semantics are explicit in `EndpointRegistry`, so a ready
  endpoint can be discovered and reused instead of provisioned again blindly.
- Persistent registry backends were added alongside in-memory behavior.
- The orchestration layer is now backend-oriented instead of being tightly
  coupled to a single runtime path.
- Hybrid orchestration code was expanded with dedicated wrappers and resolvers
  for filesystem, PVC, Hugging Face, and S3-backed models.

Why it matters:

- Collaborators should think of endpoint lifecycle as a durable control-plane
  problem now, not just "launch something and hope it is reachable".

### 5. Slurm+Podman Became The Preferred HPC Launch Path

The new Slurm+Podman backend is a major operational shift relative to older
direct/legacy launch paths.

Representative code:

- [`../src/umrs/orchestrator/slurm_podman_backend.py`](../src/umrs/orchestrator/slurm_podman_backend.py)
- [`../deploy/rccs-cloud/serve-model.slurm`](../deploy/rccs-cloud/serve-model.slurm)
- [`../deploy/hpbd-atom/serve-model.slurm`](../deploy/hpbd-atom/serve-model.slurm)
- [`../deploy/harness_validate.py`](../deploy/harness_validate.py)

Logical changes:

- The backend now submits a site-specific `serve-model.slurm` wrapper and uses
  site config exports rather than generating its own singularity-style script.
- Slurm stdout/stderr is retained under deploy-side log directories so the
  wrapper launch context and readiness clues survive cleanup.
- Partition selection, output log placement, job discovery, and host extraction
  are implemented in one place.
- `deploy/harness_validate.py` and site-side validation scripts exercise text,
  vision, and full-profile launches through the same backend.

Why it matters:

- This is the branch’s real HPC operational path. For collaborator review,
  `slurm_podman_backend.py` is more important than the older ad hoc launch code.

### 6. Durable Job Tracking Was Added As A First-Class Runtime

This branch introduces persistent job tracking instead of relying on transient
in-memory or file-based behavior.

Representative code:

- [`../src/umrs/jobs/schema.py`](../src/umrs/jobs/schema.py)
- [`../src/umrs/jobs/service.py`](../src/umrs/jobs/service.py)
- [`../src/umrs/jobs/store.py`](../src/umrs/jobs/store.py)
- [`../src/umrs/jobs/scheduler.py`](../src/umrs/jobs/scheduler.py)
- [`../src/umrs/jobs/monitor.py`](../src/umrs/jobs/monitor.py)
- [`../src/umrs/jobs/reconciler.py`](../src/umrs/jobs/reconciler.py)
- [`../src/umrs/server/durable_job_runtime.py`](../src/umrs/server/durable_job_runtime.py)
- [`../src/umrs/server/job_tracking_handler.py`](../src/umrs/server/job_tracking_handler.py)
- [`../src/umrs/orchestrator/durable.py`](../src/umrs/orchestrator/durable.py)

Logical changes:

- Postgres-backed durable job schema was added with `jobs`, `job_attempts`,
  `job_locks`, and `umrs_instances` tables.
- `JobService` now wires together store, scheduler, monitor, and reconciliation.
- `DurableJobRuntime` runs a process-owned asyncio loop on a dedicated thread,
  exposing durable job commands to the synchronous UMRS server.
- Durable orchestration wraps endpoint provisioning and makes retries,
  reconciliation, and status inspection persistent across process restarts.

Why it matters:

- Provisioning and cleanup are no longer one-shot control commands; they are
  durable tracked jobs with explicit lifecycle state.

### 7. The Client And Control Plane Surface Expanded Significantly

The CLI on this branch is much broader than the baseline branch.

Representative code:

- [`../src/client_umrs.py`](../src/client_umrs.py)
- [`../src/umrs/core/TaskCmdStatus.py`](../src/umrs/core/TaskCmdStatus.py)
- [`../doc/zmq_api_contract.md`](../doc/zmq_api_contract.md)
- [`../src/umrs/tui/dashboard.py`](../src/umrs/tui/dashboard.py)

Logical changes:

- Model registry commands were expanded: `model register`, `model list`,
  `model download`, and `model upload`.
- Durable job commands were added: `job submit` and `job cancel`.
- Workflow control was added: `upload-workflow`, `run-workflow`, and
  `workflow list`.
- A Rich-based live TUI jobs dashboard (`monitor`) was added.
- The older maintenance commands (`check`, `update`, `recompute`, `reload`) are
  still present, so the branch *adds* a control plane rather than replacing the
  original one.

Why it matters:

- A collaborator reviewing the branch should treat `client_umrs.py` as a major
  interface-expansion file, not just a thin wrapper around the original CLI.

### 8. Model Registry Was Hardened And Given Clearer Storage/ACL Boundaries

Model lifecycle and visibility rules are far more developed on this branch.

Representative code:

- [`../src/umrs/model_registry/store/crud.py`](../src/umrs/model_registry/store/crud.py)
- [`../src/umrs/model_registry/store/acl.py`](../src/umrs/model_registry/store/acl.py)
- [`../src/umrs/model_registry/store/path_validator.py`](../src/umrs/model_registry/store/path_validator.py)
- [`../src/umrs/model_registry/store/errors.py`](../src/umrs/model_registry/store/errors.py)
- [`../src/umrs/api/models.py`](../src/umrs/api/models.py)
- [`../demos/07_model_registry/`](../demos/07_model_registry)
- [`../demos/18_model_lifecycle/`](../demos/18_model_lifecycle)

Logical changes:

- The model store was split into clearer responsibilities: CRUD, ACL, path
  validation, and errors.
- REST APIs were added for model listing, creation, staging, download, and
  policy management.
- Scope is now explicit (`system`, `project`, `user`), and the APIs/CLI make
  that part of the user workflow.
- The CLI supports staged upload/download behavior, including `--register-path`
  for environments where the client filesystem and server filesystem differ.

Why it matters:

- Model onboarding and policy are now actual product features rather than
  configuration tricks or site-specific conventions.

### 9. The Vision Stack Was Generalized Beyond A Single HMAE Path

The original branch baseline centered heavily around HMAE inference support.
This branch preserves HMAE but folds it into a generic and extensible vision
serving architecture.

Representative code:

- [`../src/umrs/vision_service/service.py`](../src/umrs/vision_service/service.py)
- [`../src/umrs/vision_service/adapters/base.py`](../src/umrs/vision_service/adapters/base.py)
- [`../src/umrs/vision_service/default_adapters/hmae_adapter.py`](../src/umrs/vision_service/default_adapters/hmae_adapter.py)
- [`../src/umrs/vision_service/default_adapters/sam3_adapter.py`](../src/umrs/vision_service/default_adapters/sam3_adapter.py)
- [`../src/umrs/vision_service/bentoml_adapter.py`](../src/umrs/vision_service/bentoml_adapter.py)
- [`../src/umrs/vision_service/model_cache.py`](../src/umrs/vision_service/model_cache.py)
- [`../src/umrs/inference/hmae_inference.py`](../src/umrs/inference/hmae_inference.py)
- [`../src/umrs/inference/sam3_segmentation.py`](../src/umrs/inference/sam3_segmentation.py)
- [`../src/umrs/inference/loading.py`](../src/umrs/inference/loading.py)

Logical changes:

- The BentoML service now exposes a generic `/infer` task endpoint keyed by
  `model_type`, rather than hardwiring one serving path per model.
- A formal adapter contract (`VisionModelAdapter`) was introduced, and built-in
  HMAE and SAM3 adapters live beside a dynamic adapter-loading mechanism.
- The service supports either direct S3 or presigned URL data flow.
- `bentoml_adapter.py` bridges the async BentoML task model into UMRS’s
  synchronous `compute_embeddings` contract.
- HMAE inference code was moved into `src/umrs/inference/` and grouped with
  SAM3 and other inference modules.

Why it matters:

- HMAE is still supported, but the branch’s real change is that HMAE is no
  longer a special case. It is one adapter inside a more general vision system.

### 10. Pipeline SDK, Workflow Upload, And Plugin-Style Execution Were Added

This is the area most directly aligned with the branch name, but it is only one
part of the overall delta.

Representative code:

- [`../src/umrs/pipeline_sdk/base.py`](../src/umrs/pipeline_sdk/base.py)
- [`../src/umrs/pipeline_sdk/step.py`](../src/umrs/pipeline_sdk/step.py)
- [`../src/umrs/pipeline_sdk/workflow.py`](../src/umrs/pipeline_sdk/workflow.py)
- [`../src/umrs/pipeline_sdk/dag.py`](../src/umrs/pipeline_sdk/dag.py)
- [`../src/umrs/pipeline_sdk/connectors/`](../src/umrs/pipeline_sdk/connectors)
- [`../src/umrs/plugins/user_workflow.py`](../src/umrs/plugins/user_workflow.py)
- [`../src/umrs/plugins/step_adapter.py`](../src/umrs/plugins/step_adapter.py)
- [`../src/umrs/server/workflow_registry_handler.py`](../src/umrs/server/workflow_registry_handler.py)
- [`../demos/08_user_model/`](../demos/08_user_model)
- [`../demos/09_multistep_vision_pipeline/`](../demos/09_multistep_vision_pipeline)
- [`../demos/10_vision_multi_mode/`](../demos/10_vision_multi_mode)
- [`../demos/12_pipeline_sdk/`](../demos/12_pipeline_sdk)
- [`../demos/13_plugins/`](../demos/13_plugins)
- [`../demos/15_complex_dag/`](../demos/15_complex_dag)
- [`../demos/16_custom_workflow/`](../demos/16_custom_workflow)
- [`../demos/21_remote_plugins/`](../demos/21_remote_plugins)

Logical changes:

- A Pipeline SDK was added with `Pipeline`, `Step`, execution phases, workflow
  definitions, DAG scheduling, connectors, builder/templates, and helper code.
- Workflow definitions can now be uploaded, registered, listed, and invoked.
- `UserWorkflow` merges workflow definitions with plugin-style execution
  metadata, preparing the system for multiple execution modes.
- The demos show custom steps, multi-step pipelines, DAG-style execution,
  uploaded workflow execution, and plugin-like user code.

Why it matters:

- The branch adds a programmable extension layer on top of the original UMRS
  data plane rather than only adding more fixed built-in models.

### 11. RAG Was Expanded Into A Real Service Surface

RAG/search capabilities are broader and more operationally explicit than in the
baseline branch.

Representative code:

- [`../src/umrs/api/rag_service.py`](../src/umrs/api/rag_service.py)
- [`../src/umrs/rag/search.py`](../src/umrs/rag/search.py)
- [`../src/umrs/rag/models.py`](../src/umrs/rag/models.py)
- [`../src/umrs/rag/registry.py`](../src/umrs/rag/registry.py)
- [`../demos/05_rag_search/`](../demos/05_rag_search)
- [`../demos/19_rag_advanced/`](../demos/19_rag_advanced)
- [`../deploy/rccs-cloud/start.sh`](../deploy/rccs-cloud/start.sh)
- [`../deploy/hpbd-atom/start.sh`](../deploy/hpbd-atom/start.sh)

Logical changes:

- RAG now has a dedicated FastAPI service with request/response models.
- Search supports richer retrieval logic (vector/hybrid/reranking workflows and
  context assembly) through the `umrs.rag` package.
- Site scripts and demos explicitly treat the RAG service as a managed process,
  including tunneled port usage (`8081`) when enabled.

Why it matters:

- Search/QA became part of the deployable platform surface rather than a narrow
  implementation detail.

### 12. Demos Were Replaced With A Canonical Public Demo Progression

The public demo structure changed dramatically.

Representative code:

- [`../demos/README.md`](../demos/README.md)
- [`../demos/run_all.sh`](../demos/run_all.sh)
- [`../demos/01_health_check/`](../demos/01_health_check)
- [`../demos/02_text_embedding/`](../demos/02_text_embedding)
- [`../demos/03_sam3_segmentation/`](../demos/03_sam3_segmentation)
- [`../demos/04_vision_inference/`](../demos/04_vision_inference)
- [`../demos/05_rag_search/`](../demos/05_rag_search)
- [`../demos/06_server_maintenance/`](../demos/06_server_maintenance)
- [`../demos/07_model_registry/`](../demos/07_model_registry)
- [`../demos/08_user_model/`](../demos/08_user_model)
- [`../demos/09_multistep_vision_pipeline/`](../demos/09_multistep_vision_pipeline)
- [`../demos/10_vision_multi_mode/`](../demos/10_vision_multi_mode)
- [`../demos/11_job_submission/`](../demos/11_job_submission)
- [`../demos/12_pipeline_sdk/`](../demos/12_pipeline_sdk)
- [`../demos/13_plugins/`](../demos/13_plugins)
- [`../demos/14_video_analysis/`](../demos/14_video_analysis)
- [`../demos/15_complex_dag/`](../demos/15_complex_dag)
- [`../demos/16_custom_workflow/`](../demos/16_custom_workflow)
- [`../demos/17_multi_user/`](../demos/17_multi_user)
- [`../demos/18_model_lifecycle/`](../demos/18_model_lifecycle)
- [`../demos/19_rag_advanced/`](../demos/19_rag_advanced)
- [`../demos/20_durable_jobs/`](../demos/20_durable_jobs)
- [`../demos/21_remote_plugins/`](../demos/21_remote_plugins)
- [`../demos/22_monitor/`](../demos/22_monitor)

Logical changes:

- The branch replaces older backup/slurm-dev demo material with an ordered set
  of public demos covering the platform from basic health checks to durable
  jobs and the TUI monitor.
- `run_all.sh` and `demos/README.md` define the canonical entrypoint.
- Private/demo-only advanced workflows were moved under `demos_private/`.

Why it matters:

- Collaborators should review `demos/README.md` first when trying to understand
  how the feature surface is meant to be exercised now.

### 13. Documentation And Validation Coverage Expanded Substantially

This branch does not just add code; it adds a large supporting doc and test
surface around the new architecture.

Representative code and docs:

- [`../doc/admin_guide.md`](../doc/admin_guide.md)
- [`../doc/demo_guide.md`](../doc/demo_guide.md)
- [`../doc/design_user_defined_models.md`](../doc/design_user_defined_models.md)
- [`../doc/access_control.md`](../doc/access_control.md)
- [`../doc/user_guide.md`](../doc/user_guide.md)
- [`../doc/rag_implementation.md`](../doc/rag_implementation.md)
- [`../doc/s3_requirements_for_admin.md`](../doc/s3_requirements_for_admin.md)
- [`../tests/test_site_config.py`](../tests/test_site_config.py)
- [`../tests/test_durable_job_runtime.py`](../tests/test_durable_job_runtime.py)
- [`../tests/test_workflow_registry.py`](../tests/test_workflow_registry.py)
- [`../tests/test_dag_scheduler.py`](../tests/test_dag_scheduler.py)
- [`../tests/test_slurm_podman_backend.py`](../tests/test_slurm_podman_backend.py)
- [`../tests/test_tui_dashboard.py`](../tests/test_tui_dashboard.py)
- [`../tests/test_user_workflow.py`](../tests/test_user_workflow.py)

Logical changes:

- There is now explicit admin/user/demo/workflow/RAG documentation covering the
  expanded surface area of the system.
- Tests were added across configuration, durable runtime, orchestration,
  workflow execution, uploads, and the dashboard.
- Deploy harness validation became part of the repo itself.

Why it matters:

- The branch is trying to define a maintainable platform, not just introduce
  one feature. The documentation/test growth is part of that move.

## Major Removals, Migrations, And Deprecations

The branch also removes or demotes several older paths.

Representative removals/migrations:

- removed: `../src/embedding_worker.py`
- removed: `../docker/k8s/embedding-worker.yaml`
- removed: `../src/rivault_cfg/`
- removed: `../src/rivault_service.py`
- removed: `../src/umrs/models_loading.py`
- removed: `../src/umrs/embedding_backend.py`
- removed: `../src/umrs/rivault.py`
- removed: `../demos/slurm-dev/`
- removed: `../demos/backup/`
- moved: `../src/umrs/models/` responsibilities into `../src/umrs/inference/`
- moved: `../src/umrs/pipelines/` responsibilities into `../src/umrs/rag/`

What changed logically:

- The old embedding-worker deployment path is no longer the main approach.
- Older runtime assets under `src/rivault_cfg/` were retired in favor of
  `site.yaml` plus deploy wrappers.
- Several top-level/legacy modules were either deleted or folded into new
  canonical packages (`core`, `shared`, `server`, `inference`, `rag`).
- The old slurm-dev/backup demo environments were removed from the main path.

Why it matters:

- If a collaborator is comparing against older design notes or shell workflows,
  this branch intentionally invalidates some of those assumptions.

## What To Tell A Collaborator In One Paragraph

If you need a short explanation: this branch takes the earlier HMAE-focused UMRS
baseline and turns it into a broader platform. HMAE/SAM3 support is still there,
but it now sits inside a generic BentoML vision service, a richer model/workflow
control plane, site-driven deployment, durable endpoint/job orchestration, and a
much larger demo/documentation/test surface. The branch name emphasizes user-
defined workflows, but the actual update is a repo-wide architectural hardening
and capability expansion.

## Recommended Review Order

If your collaborator wants to review the most important implementation files in
order, this is a good sequence:

1. [`../src/umrs/orchestrator/site_config.py`](../src/umrs/orchestrator/site_config.py)
2. [`../src/umrs/orchestrator/slurm_podman_backend.py`](../src/umrs/orchestrator/slurm_podman_backend.py)
3. [`../src/umrs/orchestrator/endpoint_registry.py`](../src/umrs/orchestrator/endpoint_registry.py)
4. [`../src/umrs/jobs/service.py`](../src/umrs/jobs/service.py)
5. [`../src/umrs/server/durable_job_runtime.py`](../src/umrs/server/durable_job_runtime.py)
6. [`../src/umrs/vision_service/service.py`](../src/umrs/vision_service/service.py)
7. [`../src/umrs/model_registry/store/crud.py`](../src/umrs/model_registry/store/crud.py)
8. [`../src/umrs/pipeline_sdk/workflow.py`](../src/umrs/pipeline_sdk/workflow.py)
9. [`../src/umrs/plugins/user_workflow.py`](../src/umrs/plugins/user_workflow.py)
10. [`../src/client_umrs.py`](../src/client_umrs.py)
11. [`../deploy/rccs-cloud/README.md`](../deploy/rccs-cloud/README.md)
12. [`../demos/README.md`](../demos/README.md)
