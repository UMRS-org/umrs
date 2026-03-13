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
