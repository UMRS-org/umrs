# UMRS Development Objectives

**Updated 2026-09-22.** UMRS's objectives are the backbone responsibilities of
the platform design ([../docs/R4-UMRS-DESIGN.md](../docs/R4-UMRS-DESIGN.md)
§"UMRS: the infrastructure backbone"). Status per responsibility is tracked
in the parent's gap analysis
([../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md](../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md),
rows D-01…D-24 and D-42, D-47…D-56) and ordered in [../ROADMAP.md](../ROADMAP.md).
The authoritative cross-repository objective ledger is the parent's
`DEVELOPMENT_OBJECTIVES.md`.

## Objectives (design → gap rows)

| Objective | Gap rows |
|---|---|
| Account management with quotas mapped to cluster accounts | D-06, D-55 |
| One authenticated, encrypted client–server transport (ZeroMQ/Curve/ZAP) | D-01, D-03, D-04, D-05 |
| Workflow-job scheduling on Slurm / Kubernetes / Podman; declarative DAG | D-07, D-08 |
| Remote agent-session scheduling (front-end, CPU-only servers) | D-09, D-48, D-56 |
| Self-deployment of LLM / Vision / VLM in sandboxes; one inference contract | D-10, D-11 |
| mini-RiVault around every job, session and model; named sandbox profiles | D-12, D-13 |
| Raw → vector conversion into Postgres + pgvector; embedding policy | D-14, D-15 |
| Storage by project, written only through UMRS | D-16, D-17, D-18 |
| MCP registry with placement, trust class and lifecycle accounting | D-19, D-21, D-22, D-24 |
| Knowledge spine as the store of the Scientific Catalog; remote reach | D-39 … D-42 (shared with R4) |
| One UMRS hub per institution; trusted-host venue | D-47, D-49 |

## Design anchors referenced by source docstrings

Source comments still cite the section numbers of the former roadmap
("Section 1.9", "Section 11.1"). They map to the current implementation as
follows; keep these anchors when editing this file.

### 1. Model Registry and Downloads

UMRS owns model metadata, scope/ACL policy, storage references, staging and
download execution: `src/umrs/api/models.py` (create / list / stage /
download / policy), `src/umrs/model_registry/` (schemas, conversion,
versions, certification, capability detection), `src/umrs/jobs/models.py`
(`MODEL_DOWNLOAD` job type), `src/umrs/jobs/model_download.py`,
`model_download_state.py`, `model_download_sources.py` (Hugging Face, S3,
HTTP, local sources).

#### 1.9 Model Registry Design Anchor

Model downloads are always explicit UMRS jobs; mini-RiVault inference
containers never download models at runtime.

#### 1.9.1 Model Schema

`src/umrs/model_registry/`: `ModelRef` supports filesystem paths, PVC-style
references and external repository/download sources according to policy.

#### 1.9.2 Model API

`src/umrs/api/models.py`: `POST /umrs/models/stage` submits a durable
download job and returns a job id.

### 8. Model Versions and Conversion

`src/umrs/model_registry/versions.py`, `src/umrs/model_registry/conversion.py`.

### 11. Pipelines, Workflows, and RAG

UMRS owns workflow definitions, workflow execution and RAG execution /
indexing: `src/umrs/api/pipelines.py`, `src/umrs/api/workflows.py`,
`src/umrs/server/workflow_registry_handler.py`,
`src/umrs/server/workflow_execution_registry.py`,
`src/umrs/jobs/workflow_run.py`, `src/umrs/pipeline_sdk/`, `src/umrs/rag/`,
`src/umrs/api/rag_service.py`, `src/umrs/knowledge/`.

#### 11.1 Pipeline and RAG Models

Pipeline API, pipeline SDK and RAG model definitions above. R4 and the edge
treat these as UMRS-owned contracts.

## Validation

Focused suites for these areas: `tests/test_model_download*.py`,
`tests/test_shared_storage_download_e2e.py`,
`tests/test_workflow_durable_jobs.py`, `tests/test_workflow_executions_api.py`,
`tests/test_workflows_api.py`, `tests/test_site_profiles.py`,
`tests/test_slurm_orchestration_authority.py`,
`tests/test_surface_characterization.py`, `tests/test_migration_chain.py`,
`tests/test_zmq_sessions_api.py`, `tests/test_zmq_security.py`,
`tests/test_session_curve_wiring.py`. Whole suite: 5,1xx tests (2026-09-21);
gate: `bash ../scripts/ci-lint-local.sh --umrs-changed`.
