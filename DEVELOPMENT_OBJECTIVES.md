# UMRS Current Implementation Notes

This file replaces the older B1-B10 roadmap that referenced standalone progress,
batch-upload, and model-loading modules that no longer match the current tree.
The authoritative cross-repository objective ledger is the parent repository's
`DEVELOPMENT_OBJECTIVES.md`.

The notes below keep the design anchors still referenced by UMRS source
docstrings while pointing at the current implementation.

## 1. Model Registry and Downloads

UMRS owns model metadata, scope/ACL policy, storage references, staging, and
download execution.

Current implementation:

- `src/umrs/api/models.py` exposes model create/list/stage/download/policy
  endpoints.
- `src/umrs/model_registry/` owns metadata, schemas, conversion, scope, access,
  and store behavior.
- `src/umrs/jobs/models.py` defines durable `MODEL_DOWNLOAD` jobs.
- `src/umrs/jobs/model_download.py` runs model downloads for Hugging Face, S3,
  HTTP, and local sources.
- `src/umrs/jobs/model_download_state.py` and
  `src/umrs/jobs/model_download_sources.py` keep download state, progress, and
  source-specific helpers separate from the backend executor.

### 1.9 Model Registry Design Anchor

Source comments that refer to the former Section 1.9 map to the current model
registry package and model-download job flow above. Model downloads are always
explicit UMRS jobs; Mini-RiVault inference containers never download models at
runtime.

### 1.9.1 Model Schema

Current schema and storage-reference types live in `src/umrs/model_registry/`.
Model references support filesystem paths, PVC-style references, and external
repository/download sources according to policy.

### 1.9.2 Model API

Current API behavior is implemented in `src/umrs/api/models.py`. `POST
/umrs/models/stage` submits a durable download job and returns a job ID for
tracking.

## 8. Model Versions and Conversion

Version and conversion helpers live in:

- `src/umrs/model_registry/versions.py`
- `src/umrs/model_registry/conversion.py`

These helpers are part of the model registry implementation rather than the
removed standalone B1-B10 modules from the older roadmap.

## 11. Pipelines, Workflows, and RAG

UMRS owns workflow definitions, workflow execution, and RAG execution/indexing.
Current implementation lives in:

- `src/umrs/api/pipelines.py`
- `src/umrs/api/workflows.py`
- `src/umrs/server/workflow_registry_handler.py`
- `src/umrs/server/workflow_execution_registry.py`
- `src/umrs/jobs/workflow_run.py`
- `src/umrs/pipeline_sdk/`
- `src/umrs/rag/`

### 11.1 Pipeline and RAG Models

Source comments that refer to the former Section 11.1 map to the current
pipeline API, pipeline SDK, and RAG model definitions above. Gateway and `r4`
should treat these as UMRS-owned contracts.

## Validation

Focused coverage for the current implementation includes:

- `tests/test_model_download.py`
- `tests/test_model_download_http_progress.py`
- `tests/test_model_download_huggingface.py`
- `tests/test_model_download_recovery.py`
- `tests/test_shared_storage_download_e2e.py`
- `tests/test_workflow_durable_jobs.py`
- `tests/test_workflow_executions_api.py`
- `tests/test_workflows_api.py`
- `tests/test_site_profiles.py`
- `tests/test_slurm_orchestration_authority.py`
