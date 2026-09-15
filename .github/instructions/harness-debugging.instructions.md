---
description: "Use when debugging UMRS harness failures: missing vision-bentoml, broken RIVAULT_PROFILES propagation, stale vision tarballs, wrapper or site-config mismatches, or harness startup regressions."
applyTo: "deploy/**/*.sh,deploy/**/*.slurm,src/umrs/mini-rivault-llm/**/*.slurm,src/umrs/mini-rivault-llm/**/*.yaml,src/umrs/orchestrator/site_config.py,src/umrs/orchestrator/slurm_podman_backend.py,demos/03_sam3_segmentation/**,demos/04_vision_inference/**,doc/admin_guide.md"
---

# Harness Debugging

## Debugging Order

1. Confirm the intended profile set.
2. Confirm which profile set was actually launched.
3. Confirm whether expected containers or services came up.
4. Confirm whether the advertised endpoint is routable from the caller.
5. Confirm whether the failure is launch-time, health-time, readiness-time, or
   reuse-time.

## Known Failure Patterns

- `RIVAULT_PROFILES` loses values when passed through nested quoting.
- A running Slurm job may exist without the needed vision service being ready.
- A rebuilt image may not be used if tarball reload or image cache logic is
  stale.
- Wrapper defaults may differ from source-template defaults.
- Host paths may be valid on the login node but wrong inside the container.
- Health checks may prove process liveness without proving model-serving
  readiness.

## Debugging Discipline

- Prefer proving facts from logs, health endpoints, active profiles, and
  exported environment values.
- Do not assume that a running job means the required service is ready.
- Do not collapse site-specific wrapper issues into generic runtime issues
  without evidence.
- If a fix depends on a deployment-specific assumption, state it explicitly.
- When a wrapper, `site.yaml`, and source template disagree, identify which one
  is canonical before patching behavior.
