---
description: my_instruction
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---

## ABSOLUTE RULE: Client Runs LOCALLY — Never on Remote

UMRS is a **local-client / remote-server** architecture. The `umrs`
CLI and ALL demo scripts (`demos/`, `demos_private/`) MUST run on the
**local Mac**, NEVER via SSH on rccs-cloud.

- ❌ NEVER: `ssh rccs-cloud 'bash demos/…/run_demo.sh'`
- ❌ NEVER: `ssh rccs-cloud 'source .venv/bin/activate && …'`
- ❌ NEVER: Run any demo, client command, or test via SSH on the remote
- ✅ ALWAYS: `source .venv/bin/activate && bash demos/…/run_demo.sh` (locally)
- ✅ ALWAYS: Use the local `.venv/` at repo root (NOT `deploy/rccs-cloud/.venv/`)
- ✅ ALWAYS: Ensure SSH tunnel (`deploy/rccs-cloud/ssh_tunnel.sh`) is running first

The only operations that run on rccs-cloud are:
- `rsync` to sync source code
- `podman build` / `podman save` to rebuild container images
- `server_umrs.py` management (start/stop/restart via deploy scripts)
- Slurm job management (`squeue`, `scancel`)

If you are about to run a demo or client command, STOP and verify it runs locally.

## SSH Connection to rccs-cloud

- SSH alias: `rccs-cloud`
- Hostname: `login.cloud.r-ccs.riken.jp`
- Login node: `login1.cloud.r-ccs.riken.jp`
- User: `masaru.nagaso`
- SSH tunnel script: `deploy/rccs-cloud/ssh_tunnel.sh` (ports 5555, 5556, 9000, 8081)

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.