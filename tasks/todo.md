# TODO

## Robust Orchestration — Converge & Harden (2026-07-01)

**Goal:** Replace fragile rivault_launcher + site-specific shell wrapper with
unified orchestration using the existing (but unused) infrastructure in
`src/umrs/orchestrator/`.

### Discovery: Existing infrastructure already covers Phases 2, 3, 5
- `detection.py` — BackendDetector (Slurm/K8s/Compose auto-detect) ✅
- `ports.py` — PortAllocator (StaticRange, RegistryLock, K8sService) ✅
- `registry_backend.py` — PostgresBackend for endpoint persistence ✅
- `endpoint_registry.py` — EndpointRegistry with TTL + persistence ✅
- `service.py` — OrchestrationService with provision flow ✅
- `readiness.py` — Authoritative readiness polling ✅
- `durable.py` — Crash-safe provisioning via job system ✅

### Phase 1 — SiteConfig (eliminate hardcoded values)
- [x] 1a  Create `src/umrs/orchestrator/site_config.py` (SiteConfig dataclass + YAML loader)
- [x] 1b  Create `deploy/rccs-cloud/site.yaml` (rccs-cloud site definition)
- [x] 1c  Wire SiteConfig into `OrchestratorConfig.from_env()` + `serve-model.slurm`

### Phase 4 — Slurm+Podman-Compose Backend
- [x] 4a  Create `src/umrs/orchestrator/slurm_podman_backend.py`
       (SlurmPodmanBackend: sbatch using site's serve-model.slurm + podman-compose)
- [x] 4b  Register in factory.py as "slurm-podman" backend
- [x] 4c  Tests for site_config + slurm_podman_backend (59/59 pass)

### Phase 6 — Wire rivault_launcher
- [x] 6c  Replace diskcache with EndpointRegistry in rivault_launcher
- [x] 6d  serve-model.slurm reads from `site.yaml` instead of hardcoded paths
- [x] 6e  `get_rivault_remote_port()` falls back to SiteConfig
- [x] 6f  `launch_rivault_model()` injects site.yaml into env before sbatch
- [ ] 6a  Refactor rivault_launcher to use OrchestrationService.provision() (future)
- [ ] 6b  SSH tunnels use allocated ports from PortManager (future)

---

## Open Items

### Architecture Follow-up (2026-04-07)
- [x] A1 Extract client-safe SQL helpers out of `umrs.server.sql`
      so `client_umrs.py` no longer imports server DB internals for naming
      and metadata shaping.
- [x] A2 Rewire `server.job_tracking_handler` from the compatibility
      `~/.config/umrs/jobs.json` store to the durable `umrs.jobs` service once
      server boot owns a shared PostgreSQL-backed job runtime.
- [x] A3 Remove the JSON compatibility backend after ZMQ job commands and the
      REST/API job surface converge on the same `JobStore` state machine.
- [x] A4 Keep `deploy/rccs-cloud/.env` limited to secrets and explicit
      overrides; `site.yaml` remains canonical for site defaults.

#### A2 Review — Process-Owned Durable Runtime (2026-04-07)
- [x] Start a process-owned durable jobs runtime from `server_umrs.py`
      so scheduler and monitor loops live for the server lifetime.
- [x] Share that runtime's `JobStore` with ZMQ job commands instead of opening
      a separate command-only pool when full durable startup succeeds.
- [x] Keep the durable command-backend fallback for cases where the full async
      runtime cannot start.
- [x] Fix latent durable startup bugs that blocked live runtime ownership:
      thread-safe Postgres pool bootstrap, status-checker factory wiring, and
      proper pooled connection return during schema initialization.

#### A3 Review — Remove JSON Compatibility Backend (2026-04-07)
- [x] Delete the `~/.config/umrs/jobs.json` fallback helpers from
      `server.job_tracking_handler` so ZMQ `JOB_*` commands no longer create a
      second source of truth for job state.
- [x] Keep the ZMQ wire contract JSON-shaped while requiring a durable backend
      behind `JOB_SUBMIT`, `JOB_CANCEL`, and `JOB_LIST`.
- [x] Return a structured `{"status": "error", ...}` response when neither the
      process-owned durable runtime nor the command-only durable backend can
      start during server boot.
- [x] Replace fallback-specific tests with durable bridge and dashboard
      contract coverage.

### LiteLLM / Embedding Proxy (2026-03-08)
- [ ] Diagnose LiteLLM Prisma table errors (root cause of proxy failure)
- [ ] Fix nginx→LiteLLM→vLLM chain so embeddings can be computed
- [ ] Re-run e2e demo and verify embeddings in DB via phase3b

### rccs-cloud Demo Execution
- [x] Demo 00: health check (SSH tunnels + UMRS ping)
- [x] Demo 01: text embedding (upload texts, wait for embeddings)
- [ ] Demo 02: RAG search (vector + hybrid + Q&A) — needs UMRS_RAG_ENABLE=1
- [x] Demo 03: SAM3 segmentation (upload images, download results)
- [x] Demo 04: HMAE inference (upload CT images, download results)

#### rccs-cloud Lifecycle Follow-up (2026-04-08)
- [x] L1 Fix `deploy/rccs-cloud/stop.sh --server` so it does not stop
      containers or cancel Slurm jobs during a native server restart.
- [x] L2 Validate a real `stop.sh --server && start.sh --server` cycle on
      rccs-cloud with final health checks.
- [x] L3 Run a live durable `endpoint_provision` smoke test on rccs-cloud using
      the new `ng-dgx-m*` partition-selection path.
- [x] L4 Replace raw `nc -z 127.0.0.1 5555` checks in rccs-cloud lifecycle
      scripts with a real ZMQ handshake so readiness checks stop flapping.

#### rccs-cloud Durable Endpoint Follow-up (2026-04-08)
- [x] E1 Make durable orchestration honor the site-configured backend kind so
      rccs-cloud uses `slurm-podman` instead of the legacy `slurm` backend.
- [x] E2 Allow scheduler-side backend submission failures to move attempts from
      `created` to `failed` cleanly so jobs do not stick in `running`.
- [x] E3 Teach the durable endpoint status checker to refresh pending Slurm
      endpoints from backend job status before polling readiness.
- [x] E4 Re-run focused durable/orchestrator tests, then repeat the live
      `endpoint_provision` smoke test on rccs-cloud.

### User-Defined Model Workflows
- [ ] U-200: Finalize UDM design document and review with stakeholder

### Vision Pipeline
- [ ] V9: Demo 09 (HMAE full pipeline) verification
- [ ] V10: Clean up old tarballs (~27GB of stale images in mini-rivault-llm/images/)

### Harness Engineering (2026-04-07)

**Goal:** Make the UMRS harness a first-class runtime contract between UMRS,
Mini-Rivault, and site-specific wrappers instead of a set of partially-aligned
 startup conventions.

#### Phase 1 — Canonical Contract
- [ ] H1  Define canonical ownership for harness defaults:
      `site.yaml`, source `serve-model.slurm`, deploy wrappers, and docs
- [ ] H2  Decide which profile sets are officially supported by default:
      `harness`, `harness vllm`, `harness vllm vision-bentoml`, optional `rag`
- [ ] H3  Document invariants for `RIVAULT_PROFILES`, host/container paths,
      health readiness, and image reload behavior

#### Phase 2 — Wrapper Convergence
- [x] H4  Diff source Mini-Rivault launcher versus deploy-site wrappers and
      list intentional divergences versus drift
- [x] H5  Remove accidental default mismatches between source launcher,
      deploy wrappers, and `site.yaml` export
- [x] H6  Ensure wrapper behavior is driven by site config or documented
      wrapper-only policy, not hidden hardcoded defaults

#### Phase 3 — Validation Harness
- [ ] H7  Define a minimal harness validation matrix covering text-only,
      vision-enabled, and full-stack profile sets
- [ ] H8  Add or strengthen checks that prove service readiness rather than
      only process/container existence
- [ ] H9  Verify restart, cleanup, reuse, and stale-image paths in addition to
      happy-path startup

#### Phase 4 — Observability & Debugging
- [ ] H10 Standardize what must be visible in logs: active profiles,
      exported env overrides, selected compose files, and readiness failures
- [ ] H11 Create a repeatable debugging order for wrapper issues,
      site-config mismatches, tunnel issues, and stale tarball/image problems
- [ ] H12 Align operator docs and demos with the canonical harness behavior

#### Review — Harness Ownership Audit (2026-04-07)

- Source `src/umrs/mini-rivault-llm/serve-model.slurm` is still the canonical
      compose launcher, but deploy wrappers remain authoritative for several
      site-specific defaults.
- `deploy/rccs-cloud/site.yaml` is the clearest canonical source for
      rccs-cloud values, and `site_config.py` can export those values back into the
      runtime environment.
- `deploy/hpbd-atom/site.yaml` now exists and the deploy-side startup / health
      scripts consume it for model-cache defaults, but hpbd-atom still relies
      on wrapper-local policy for profile defaults and other site-specific
      launch behavior.
- rccs-cloud startup now defaults to the colocated
      `deploy/rccs-cloud/serve-model.slurm`; the legacy
      `~/.config/umrs/rivault_data/serve-model.slurm` path is fallback-only
      unless `RIVAULT_SLURM_SCRIPT` explicitly overrides it.
- Deploy-site wrappers now load their colocated `site.yaml` before applying
      wrapper-local `${VAR:-default}` fallbacks, so profile sets and model-path
      defaults follow the site contract unless the caller explicitly overrides
      them.
- Source `src/umrs/mini-rivault-llm/serve-model.slurm` now does the same when
      `UMRS_SITE_CONFIG` or another discoverable `site.yaml` is available, so
      source-launch defaults and wrapper-launch defaults converge on the same
      site-config export path instead of diverging silently.
- Profile defaults are intentionally divergent today:
      rccs-cloud defaults to `harness vllm vision-bentoml`, while hpbd-atom
      defaults to `harness vision-bentoml`, and source Mini-Rivault still defaults
      to `harness vllm` when no site config is present.
- This means remaining profile divergence is now intentional site policy when a
      site config exists, with the generic source fallback only used outside a
      site-configured deployment.

---

### Source Tree Reorganization (2026-04-06)

Goal: Move flat top-level `src/umrs/*.py` modules into canonical subdirectories
so the source tree clearly communicates what code serves which purpose.
Each moved file gets a **compatibility wrapper** at the old path.

#### Phase 1 — Low-risk infrastructure moves
- [x] 1a  `shared/`: auth, isolation, modality, cache_paths, resource_tracker, scheduler, webhooks, utils
- [x] 1b  `server/`: job_tracking_handler, model_registry_handler, workflow_registry_handler, sql, rivault_launcher
- [x] 1c  `core/`: mq_posix_ipc, mq_sysv_ipc, mq_zeroMQ
- [x] 1d  `shared/storage/`: s3, s3_presign, minio

#### Phase 2 — Domain-specific consolidation
- [x] 2a  `models/`: embeddings, embedding_backend, models_loading (→loading.py), rag_pipeline
- [x] 2b  `shared/`: data_versioning, experiment_io, result_diff
- [x] 2c  `pipeline_sdk/`: pipeline_builder (→builder.py), pipeline_templates (→templates.py)
- [x] 2d  `client/`: metadata, batch_upload; `shared/`: pretty_print, progress

#### Phase 3 — Cleanup
- [x] 3a  Remove all 33 compatibility wrappers (28 new + 5 legacy: config, BucketsCfg, json_utils, bentoml_client)
- [x] 3b  Update all imports in source + test files to canonical paths
- [x] 3c  Update mock.patch targets to canonical module paths
- [x] 3d  Verify: 285/285 affected tests pass, all rccs-cloud demos pass
- [ ] 3e  Move perf_test → tests/ (optional, standalone script)

---

## Completed Archive

<details>
<summary>Slurm-dev demo hardening (2026-03)</summary>

- [x] Audit current UMRS slurm-dev demo flow and identify fragility points
- [x] Redesign `demos/slurm-dev/scripts/run_e2e_demo.sh` for robust preflight and cleanup
- [x] Implement script changes with safer error handling and deterministic service detection
- [x] Verify script syntax and basic execution logic
- [x] Investigate false-positive "All embedding tasks completed" (phase3 bug)
- [x] Fix phase3 to detect Error status; add phase3b DB verification
</details>

<details>
<summary>Full Demo Validation — 15/15 pass (2026-03-27)</summary>

- [x] Demos 00–14 all pass (1 bug found and fixed: Demo 01 status parser)
</details>

<details>
<summary>Library Features B1–B10 (2026-04-05)</summary>

- [x] B1–B10 all implemented (progress, batch_upload, result_diff, webhooks, resource_tracker, data_versioning, pipeline_templates, pipeline_builder, experiment_io, scheduler)
</details>

<details>
<summary>Demos 15–21 (2026-04-05)</summary>

- [x] Demos 15–21 all created (complex_dag, custom_workflow, multi_user, model_lifecycle, rag_advanced, durable_jobs, remote_plugins)
</details>

<details>
<summary>Unit Tests (2026-04-05)</summary>

- [x] 10 test files, 170 tests passing
</details>

<details>
<summary>Documentation Updates (2026-04-05)</summary>

- [x] demos/README.md, Readme.md, DEVELOPMENT_OBJECTIVES.md all updated
</details>

<details>
<summary>rccs-cloud Hybrid Deployment — server + client setup (2026-03-29)</summary>

- [x] Pre-deployment: config sync, rsync
- [x] Server setup: venv, docker-compose fixes, pgvector, health checks (10/10 pass)
- [x] Client setup: venv, SSH tunnels (ports 5555/5556/9000/8081)
</details>

<details>
<summary>User-Defined Model Workflows — implementation (2026-03-16)</summary>

- [x] U-201–U-212 all implemented (StepInterface, WorkflowDefinition, ScriptLoader, WorkflowExecutor, sandbox execution, runners, staging cleanup)
</details>

<details>
<summary>Pipeline Template & Demo Restructuring (2026-03-15)</summary>

- [x] P1–P6: pip-installable client, Mini-Rivault vision, admin/user guides, demo directories, cleanup
</details>

<details>
<summary>Vision Pipeline Fixes V1–V8 (2026-03-24)</summary>

- [x] V1–V8: BentoML status polling, adapter path/name, SystemExit crash, dockerignore, SAM2 optimization, image rebuild, Demo 08 e2e verified (71s)
</details>

<details>
<summary>Phase 1: Step Package Upload (2026-04-04)</summary>

- [x] PH1-1–PH1-7: UPLOAD_WORKFLOW command, server handler, SourceType.UPLOADED, CLI subcommand, 15/15 tests
</details>

<details>
<summary>Phase 2: Step Registry & Discovery (2026-04-04)</summary>

- [x] PH2-1–PH2-8: LIST_WORKFLOWS command, workflow catalog, auto-register, CLI subcommand, extends field
</details>
- [x] PH2-9: Write unit tests in `tests/test_workflow_registry.py` (21/21 pass)

## Phase 3: Unified Plugin + Step Model (2026-04-10)

> **Goal**: Bridge the Plugin system (Slurm/K8s runners) with the Pipeline SDK
> (in-process Steps) via a unified `UserWorkflow` / `UserWorkflowRunner` that
> dispatches step execution to any backend based on `execution_mode`.

### Implementation

- [x] PH3-1: Add `ExecutionMode` enum (IN_PROCESS, SUBPROCESS, SLURM, K8S) +
      `execution_mode` field on `WorkflowDefinition` in `workflow.py`
- [x] PH3-2: Create `src/umrs/plugins/user_workflow.py` —
      `UserWorkflow` dataclass (WorkflowDefinition superset + env_vars,
      output_destinations, user_id, project_id)
- [x] PH3-3: Create `UserWorkflowRunner` orchestrator in same file —
      switches on execution_mode: in_process (direct), slurm (SlurmPluginRunner),
      k8s (K8sPluginRunner), subprocess (placeholder)
- [x] PH3-4: Create `src/umrs/plugins/step_adapter.py` —
      `step_to_plugin_spec()` wraps a Step as PluginSpec for remote backends
- [x] PH3-5: Wire `WorkflowPipeline.run_as_embeddings()` to delegate to
      `UserWorkflowRunner` for non-IN_PROCESS execution modes
- [x] PH3-6: Add `user_workflow://` prefix handler in `models_loading.py`
- [x] PH3-7: Export new classes from `plugins/__init__.py`
      (UserWorkflow, UserWorkflowRunner, step_to_plugin_spec)
- [x] PH3-8: Write unit tests in `tests/test_user_workflow.py` (24/24 pass)

**67 total tests pass (21 PH2 + 22 PH1 + 24 PH3).**

### Phase 4: DAG / Parallel Steps — DONE ✅

- [x] PH4-1: Add `depends_on`, `max_retries`, `retry_delay` fields to `StepSpec`
      with `to_dict()`/`from_dict()` round-trip support
- [x] PH4-2: Create `src/umrs/pipeline_sdk/dag.py` — `DAGScheduler`
      (Kahn's algorithm, execution levels, topological sort) + `execute_dag()`
      (ThreadPoolExecutor for parallel levels, fan-in key merge, retry support)
- [x] PH4-3: Update `WorkflowDefinition.validate()` — depends_on reference
      checks (unknown step, self-dependency) + cycle detection via DAGScheduler
- [x] PH4-4: Wire DAG execution into `WorkflowPipeline.compute()` and
      `run_as_embeddings()` — replaced linear loops with DAGScheduler dispatch.
      Added `_aggregate_dag_results()` helper for extracting final outputs.
- [x] PH4-5: Write unit tests in `tests/test_dag_scheduler.py` (33/33 pass)
      — graph construction, diamond/linear fallback, parallel execution,
      fan-in, retry, progress reporting, _aggregate_dag_results

**100 total tests pass (67 PH1-3 + 33 PH4).**

### Phase 5: Robustness & Polish — DONE ✅

- [x] PH5-1: Per-step retry with `max_retries` / `retry_delay` in StepSpec
      (integrated into `execute_dag()`)
- [x] PH5-2: Cached step imports — `_import_step_class` caches by
      `(entrypoint, script_dir)` with thread-safe `_step_cache_lock`
- [x] PH5-3: Replaced `sys.path` manipulation with
      `importlib.util.spec_from_file_location()` for thread-safe loading
- [x] PH5-4: Step progress reporting — `StepContext.report_progress(pct, msg)`
- [x] PH5-5: Write unit tests in `tests/test_robustness.py` (11/11 pass)
      — importlib loading, sys.path non-mutation, import cache, error handling

**114 total tests pass (67 PH1-3 + 33 PH4 + 14 PH5).**

---

## HMAE Reconstruction Quality + Opossum Demo (2026-04-03)

### Task 1: HMAE Quality Analysis & 3-Mode Demo

**Findings from comparing `tmp/hmae-infer/` (reference) vs UMRS implementation:**

- [x] Analyze core model code (hvit.py, engine.py) — **identical** ✓
- [x] Analyze inference functions (infer_reconstruct/mae/features) — **identical** ✓
- [x] Analyze visualization (visualize_reconstruction) — **identical** ✓
- [x] **BUG: `mode` vs `task` key mismatch** — `demo_hmae_inference.py` sends `inference_config={"mode": mode}` but `hmae_adapter.py` reads `config.get("task", ...)`. Fix adapter to accept both keys.
- [x] **BUG: Demo 03 only runs `reconstruction`** — `vision_models.json` hardcodes `"task": "reconstruction"`. Update `run_demo.sh` to support `HMAE_MODE` env var for all 3 modes.
- [x] Write quality analysis summary in session memory

### Task 2: Opossum Demo

- [x] Create `demos/17_opossum_analysis/` demo directory
- [x] Write run_demo.sh / pipeline scripts for opossum video analysis through UMRS
