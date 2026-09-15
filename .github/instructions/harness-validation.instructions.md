---
description: "Use when reviewing or editing UMRS harness startup and deployment behavior: profile defaults, Slurm wrappers, rootless podman constraints, SSH tunnel assumptions, health checks, and service reuse logic."
applyTo: "deploy/**/*.sh,deploy/**/*.slurm,demos/**/run_demo.sh,demos/**/README.md,doc/admin_guide.md,src/umrs/mini-rivault-llm/**/*.slurm,src/umrs/mini-rivault-llm/**/*.yaml,src/umrs/orchestrator/site_config.py,src/umrs/orchestrator/slurm_podman_backend.py"
---

# Harness Validation

## Validation Standard

Do not accept a harness change unless both the launch path and the reuse path
are clear.

## Required Checks

- Verify which profiles start by default and which are opt-in.
- Verify whether an existing job is safe to reuse or must be replaced.
- Verify that health checks prove the required service is live, not only that a
  Slurm job or container exists.
- Verify that tunnel-facing ports and advertised hosts still match the intended
  deployment mode.
- Verify that rootless podman limitations are still respected.
- Verify that `site.yaml` export and wrapper-side defaults do not drift.

## Regression Questions

- Could this change start fewer services than the caller expects?
- Could this change break vision startup while text startup still looks healthy?
- Could this change work on one site wrapper but fail on another?
- Could this change make docs or demos describe a different profile set than
  the code actually launches?
- Could this change break restart or cleanup behavior while the happy path
  still works?

## Preferred Evidence

- Job output that shows active profiles
- Container or compose output that shows expected services
- Health endpoint checks
- One positive-path proof and one degraded-path explanation
