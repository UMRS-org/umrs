# TODO

- [x] Audit current UMRS slurm-dev demo flow and identify fragility points.
- [x] Redesign `demos/slurm-dev/scripts/run_e2e_demo.sh` for robust preflight, readiness, and cleanup behavior.
- [x] Implement script changes with safer error handling and deterministic service detection.
- [x] Verify script syntax and basic execution logic (`bash -n`, shellcheck-style review).
- [x] Add a short review/results section with verification outcomes.
- [x] Investigate false-positive "All embedding tasks completed" (phase3 bug).
- [x] Fix phase3 to detect Error status; add phase3b DB verification.
- [ ] Diagnose LiteLLM Prisma table errors (root cause of proxy failure).
- [ ] Fix nginx→LiteLLM→vLLM chain so embeddings can be computed.
- [ ] Re-run e2e demo and verify embeddings in DB via phase3b.

## User-Defined Model Workflows (UDM) — Planning (2026-03-16)

> **Goal**: Let users define and run their own models / multi-step workflows
> via user-provided Python scripts — without modifying UMRS internal code.

### Design Status: PLANNING (see doc/design_user_defined_models.md)

- [ ] U-200: Finalize design document and review with stakeholder
- [x] U-201: Implement `UMRSStepInterface` (ABC for user scripts)
- [x] U-202: Implement `UserWorkflowDefinition` model + JSON schema
- [x] U-203: Implement `ScriptLoader` (local path / S3 / git source)
- [x] U-204: Implement `WorkflowExecutor` (step sequencer + S3 I/O)
- [x] U-205: Add `user_script://` model_type in `models_loading.py`
- [ ] U-206: Sandbox execution (isolated venv, resource limits, audit)
- [ ] U-207: End-to-end demo: user-provided script via mounted path
- [ ] U-208: Documentation (user guide + example scripts)

---

## Pipeline Template & Demo Restructuring (2026-03-15)

- [x] P1: Make `umrs-client` pip-installable (pyproject.toml + def main())
- [x] P2: Mini-Rivault vision integration (NGINX /vision/, RIVAULT_PROFILES, env vars)
- [x] P3: Create admin_guide.md
- [x] P4: Create user_guide.md
- [x] P5: Create 4 demo directories (00_start_services, 01_text_embedding, 02_sam3_segmentation, 03_hmae_inference)
- [x] P6: Clean up broken Python-first templates
- [x] Remove stale `import torch` from utils.py (blocked pip install)
- [x] Fix bash brace expansion in demo scripts (`*.{jpg,png}` → `find` command)

## Review

- Replaced the previous linear script with function-based orchestration and strict shell safety (`set -Eeuo pipefail`).
- Added preflight checks for required commands and config files.
- Removed brittle hardcoded vLLM container-name assumptions by resolving service container IDs via compose.
- Added deterministic readiness gates for `ai-login`, `umrs-client`, Slurm node registration, vLLM endpoint, and LiteLLM host endpoint.
- Kept behavior consistent with existing demo flow: control plane up, data upload, embedding run, job cleanup, always-on LLM startup, and RAG QA.
- Verification completed: `bash -n demos/slurm-dev/scripts/run_e2e_demo.sh` passed.
- Verification completed: `python3 -m compileall -q src` passed.
- `shellcheck` could not be run because it is not installed in this environment.

### Investigation: False-Positive Embedding Completion (2026-03-08)
- Root cause: phase3 only counted running tasks (0 running = "complete"), but task status was "Error".
- Error: `Model intfloat/e5-base-v2 not available at http://localhost:42717/v1/models after 300.0s`
- Proxy chain: nginx→LiteLLM→vLLM. vLLM loaded OK, but LiteLLM has Prisma `TableNotFoundError` (missing LiteLLM_Config, LiteLLM_CredentialsTable, LiteLLM_ProxyModelTable tables).
- DB state: `text_rivault_rivault_user.text` table exists with `embedding vector(768)` column, 0 rows, 0 embeddings.
- Fix applied: phase3 now detects Error status; phase3b queries DB directly for embedding verification.
