---
description: "Use when working on UMRS harness engineering: Mini-Rivault compose profiles, RIVAULT_PROFILES, serve-model wrappers, harness startup glue, or profile-driven deployment behavior."
applyTo: "src/umrs/mini-rivault-llm/**/*.yaml,src/umrs/mini-rivault-llm/**/*.yml,src/umrs/mini-rivault-llm/**/*.slurm,deploy/**/serve-model.slurm,deploy/**/site.yaml,src/umrs/orchestrator/site_config.py,src/umrs/orchestrator/slurm_podman_backend.py,doc/admin_guide.md"
---

# Harness Engineering

## Intent

UMRS harness engineering means preserving a reliable launch layer between UMRS
and Mini-Rivault. Treat the harness as production infrastructure, not as demo
glue.

## Core Invariants

- The `harness` profile is foundational. Do not break services needed for
  NGINX, LiteLLM, DNS, or proxy wiring.
- Treat `RIVAULT_PROFILES` as a space-separated contract. Do not rely on nested
  shell quoting through `sbatch --export`.
- Preserve the distinction between host paths and container paths for model
  mounts, image tarballs, and caches.
- Keep deployment wrappers and source templates behaviorally aligned unless a
  divergence is explicitly site-specific.
- Prefer one canonical source for profile defaults. If wrappers override
  defaults, document why.

## Change Rules

- When changing profile defaults, reason about at least these cases:
  `harness`, `harness vllm`, and `harness vllm vision-bentoml`.
- Prefer explicit readiness or health checks over sleeps.
- Do not silently remove guardrails that exist for rootless podman, SSH tunnel
  routing, or Slurm wrapper behavior.
- If a change alters startup assumptions, update both implementation and
  operator-facing docs in the same change.
- Preserve compatibility with site-config driven environment export where
  possible; avoid reintroducing hardcoded site values.

## Review Focus

- Profile activation order and defaults
- `RIVAULT_PROFILES` propagation across wrappers and `site.yaml`
- Host versus container path correctness
- Wrapper and template divergence
- Stale image or stale job reuse behavior
- Endpoints that appear live but are not actually ready
