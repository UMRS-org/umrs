# TODO

- [x] Audit current UMRS slurm-dev demo flow and identify fragility points.
- [x] Redesign `demos/slurm-dev/scripts/run_e2e_demo.sh` for robust preflight, readiness, and cleanup behavior.
- [x] Implement script changes with safer error handling and deterministic service detection.
- [x] Verify script syntax and basic execution logic (`bash -n`, shellcheck-style review).
- [x] Add a short review/results section with verification outcomes.
- [x] Investigate false-positive "All embedding tasks completed" (phase3 bug).
- [x] Fix phase3 to detect Error status; add phase3b DB verification.

## Full Demo Validation (2026-03-27)

- [x] Demo 00: Health check — 6/6 checks pass (ai-login, postgres, minio, slurmd, slurm node, pgvector)
- [x] Demo 01: Text embedding — 40 embeddings verified, exported to CSV. **Bug fixed:** status parser crashed on non-dict items in summaries JSON (`check`/`update` categories contain strings, not dicts)
- [x] Demo 02: SAM3 segmentation — 2 images processed in 72s, results downloaded
- [x] Demo 03: HMAE inference — 2 images reconstructed in 23s
- [x] Demo 04: RAG search — vector, hybrid, reranking all return relevant results. RAG Q&A retrieval works (LLM connection error expected — vLLM not active)
- [x] Demo 05: Server maintenance — 7 buckets fully consistent, 0 mismatches/errors
- [x] Demo 06: Model registry — 2 system models listed, duplicate registration handled gracefully
- [x] Demo 07: User model — 31/31 pytest tests pass, integration smoke test passes
- [x] Demo 08: Defect analysis — 4-step pipeline completed in 83s, segmentation results downloaded
- [x] Demo 09: HMAE full pipeline — reconstruct+mae+features (6 files) completed in 100s
- [x] Demo 10: Job submission — endpoint_provision, index_build, cleanup, cancel all work
- [x] Demo 11: Pipeline SDK — TextAnalysisPipeline LOCAL mode: 1 file, 201 words analyzed
- [x] Demo 12: Plugins — Full lifecycle (define→validate→preview→execute→audit) succeeds
- [x] Demo 13: SAM3 Jens — Jens's standalone stack: 2 images in 80s
- [x] Demo 14: HMAE Jens — Jens's standalone stack: 2 images in 60s

**Result: 15/15 demos pass. 1 bug found and fixed (Demo 01 status parser).**

- [ ] Diagnose LiteLLM Prisma table errors (root cause of proxy failure).
- [ ] Fix nginx→LiteLLM→vLLM chain so embeddings can be computed.
- [ ] Re-run e2e demo and verify embeddings in DB via phase3b.

## User-Defined Model Workflows (UDM) — 2026-03-16

> **Goal**: Let users define and run their own models / multi-step workflows
> via user-provided Python scripts — without modifying UMRS internal code.

### Design Status: IMPLEMENTED (see doc/design_user_defined_models.md)

- [ ] U-200: Finalize design document and review with stakeholder
- [x] U-201: Implement `UMRSStepInterface` (ABC for user scripts)
- [x] U-202: Implement `UserWorkflowDefinition` model + JSON schema
- [x] U-203: Implement `ScriptLoader` (local path / S3 / git source)
- [x] U-204: Implement `WorkflowExecutor` (step sequencer + S3 I/O)
- [x] U-205: Add `user_script://` model_type in `models_loading.py`
- [x] U-206: Sandbox execution (subprocess + Slurm + K8s + audit)
- [x] U-207: End-to-end demo: user-provided script via mounted path
- [x] U-208: Documentation (user guide + example scripts)
- [x] U-209: ExecutionMode enum + pluggable runner protocol
- [x] U-210: SlurmStepRunner (sbatch + Singularity bind-mount)
- [x] U-211: K8sStepRunner (Job + ConfigMap + hostPath/PVC)
- [x] U-212: Staging cleanup (_staging/<run_id>/ auto-delete)

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

## Vision Pipeline Fixes (2026-03-24)

- [x] V1: Fix BentoML status polling — `ResultStatus` enum mismatch (completed/failed/canceled vs success/failure/cancelled)
- [x] V2: Fix adapter path — `DEFAULT_ADAPTERS_DIR` pointing to empty `mini-rivault-llm/default_adapters/`
- [x] V3: Fix adapter name — `removesuffix("_adapter")` for filename-to-registry key
- [x] V4: Fix SystemExit crash — adapter loader catching `SystemExit` from rogue scripts
- [x] V5: Fix `.dockerignore` — `*.tar` → `**/*.tar` for recursive matching
- [x] V6: SAM2 optimization — deferred upscaling, batch_size 64→16
- [x] V7: Docker image rebuild with all fixes
- [x] V8: Demo 08 end-to-end verification — both images processed in 71s (was timing out at 300-1200s)
- [ ] V9: Demo 09 (HMAE full) verification
- [ ] V10: Clean up old tarballs (~27GB of stale images in mini-rivault-llm/images/)
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
