---
description: Agent rules for working inside the umrs submodule (UMRS = infrastructure backbone of the R4 + UMRS platform)
applyTo: '**'
---

# Working in `umrs/` (updated 2026-09-22)

## What this tree is

UMRS is the infrastructure backbone of the R4 + UMRS platform: accounts,
client transport, durable jobs and scheduling, self-deployed models in
mini-RiVault sandboxes, Postgres + pgvector and S3 storage, MCP and skill
registries, the knowledge spine. Design:
[../../../docs/R4-UMRS-DESIGN.md](../../../docs/R4-UMRS-DESIGN.md). Gaps:
[../../../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md](../../../docs/R4-UMRS-GAP-ANALYSIS-2026-09-22.md)
(rows D-01…D-24, D-42, D-47…D-56 are UMRS-side). Parent rules apply:
[../../../CONSTITUTION.md](../../../CONSTITUTION.md), [../../../CLAUDE.md](../../../CLAUDE.md).
Submodule-specific: [../../MERGE_CONSTITUTION.md](../../MERGE_CONSTITUTION.md),
[../../doc/AI4S_PLATFORM_BOUNDARY.md](../../doc/AI4S_PLATFORM_BOUNDARY.md),
[umrs-project.instructions.md](umrs-project.instructions.md) (runtime facts, gotchas).

## ABSOLUTE RULE: the client runs LOCALLY, never on the remote

UMRS is a local-client / remote-server system. The `umrs` and `r4` CLIs and
ALL demo scripts (`demos/`, `demos_private/`) run on the **local machine**,
never via SSH on a login node or the front-end.

- ❌ NEVER `ssh <host> 'bash demos/…/run_demo.sh'` or `ssh <host> 'source .venv/bin/activate && …'`
- ✅ ALWAYS `source .venv/bin/activate && bash demos/…/run_demo.sh` locally, using the repo-root `.venv/`
- Connectivity: umrs-front clusters (rccs, ai4s, bdrt, octopus) use **direct
  TCP** ZeroMQ to `data0.ai.r-ccs.riken.jp` (50110–50118) — never an SSH
  tunnel; `umrs info` must show `mode direct`. The login-direct `rccs-cloud`
  profile is the one place a tunnel script exists
  (`deploy/rccs/ssh_tunnel.sh`; the repo tree is `deploy/rccs/`, the Ansible profile name is `rccs-cloud`) because its
  ports are firewalled.

The only operations that run on a remote host are repo sync and container
builds **through Ansible** (`deploy/ansible/bootstrap.sh --profile …` in the
parent), server lifecycle through the deploy playbooks, and read-only
inspection (`squeue`, logs, `git status`). Hand-editing a deployed checkout
or restarting a service by ad-hoc SSH is forbidden (ADR-019 amendment).

## Submodule discipline

- Commit inside `umrs/` on branch `ai4s-platform`; the parent then bumps the
  pointer (`chore(umrs): bump to <sha> — <why>`). Docs citing `umrs/` paths
  land with or after the bump.
- Gate: `bash ../scripts/ci-lint-local.sh --umrs-changed` (lint, complexity
  ≤ 9, fingerprint baselines that only shrink — ADR-052) and the parent
  validators (`python ../tools/import_boundary_validator.py`).
- Foundation modules never import `umrs.api` / `umrs.server`; every `api/`
  router is mounted or deleted (ADR-053, `tests/test_surface_characterization.py`).
- Tests never touch `sys.path`; E2E lanes opt in with `R4_RUN_E2E=1`.
- Never git-operate on a sibling clone outside the parent repo; only this
  submodule.
- After merging `main`: run MERGE_CONSTITUTION §5 in order.

## Workflow

1. **Plan first** for non-trivial tasks (3+ steps or a design decision);
   write the plan to `tasks/todo.md` with checkable items; name the gap rows.
2. **Subagents** for research and parallel exploration; one task per subagent.
3. **Verify before done**: run the tests and the gate, read logs, show the
   evidence; diff behaviour against `main` when relevant.
4. **Elegance, balanced**: for non-trivial changes ask whether a simpler
   mechanism exists; do not over-engineer simple fixes.
5. **Bugs**: fix from the logs, errors and failing tests without hand-holding.
6. **Lessons**: after a correction, add the pattern to `tasks/lessons.md`;
   read it at session start.
7. **Reports**: what changed, gap rows before → after, commands run, hashes;
   ledger entry in the parent's `docs/development/wave-ledger.md`.

## Core principles

- Simplicity first; minimal impact; root causes, no temporary fixes.
- Fail-closed on safety (transport keys, egress, decision-gated knowledge
  mutation); fail-degraded on user surface.
- Everything durable is Postgres; everything launched is a Job/Attempt;
  everything that talks out goes through EgressGuard.
