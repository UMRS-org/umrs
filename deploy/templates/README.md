# UMRS Deployment Templates

Minimal, **site-agnostic** reference templates that show *how* to deploy the
UMRS service. Copy a template directory, fill in the placeholders (`<...>` and
empty secrets), and adapt to your site.

These are **starting points**, not turnkey deployments. The live, site-specific
deployments (e.g. `../rccs-cloud/`, `../umrs-front/`) were distilled into these
templates — refer to them for battle-tested details, but do not copy their
hardcoded hostnames, Lustre paths, or credentials.

## Which template do I want?

| Template | Topology | Use when |
|----------|----------|----------|
| [`login-node/`](login-node/) | UMRS runs **natively on the HPC login node**; `sbatch` is local. | You have direct shell + Slurm access on a login node and want the simplest possible deployment. |
| [`frontend-multi-hpc/`](frontend-multi-hpc/) | One **persistent frontend server** runs one containerized UMRS per cluster; each reaches its HPC backend **over SSH**. A gateway path-routes clients to each cluster. | You want a single stable entry point in front of **multiple** HPC clusters, decoupled from any one login node's uptime. |

```
  ┌─────────────────────────── login-node ───────────────────────────┐
  │  client ──HTTP/ZMQ──> UMRS (native, on login node) ──sbatch──> compute │
  └───────────────────────────────────────────────────────────────────┘

  ┌──────────────────────── frontend-multi-hpc ───────────────────────┐
  │            ┌─> UMRS(cluster-a) ──SSH──> login-a ──sbatch──> compute-a │
  │  client ──> gateway ─┤                                                │
  │            └─> UMRS(cluster-b) ──SSH──> login-b ──sbatch──> compute-b │
  └───────────────────────────────────────────────────────────────────┘
```

## Common invariants (both templates)

- Infra (Postgres + MinIO) runs in **rootless Podman** containers.
- Presigned S3 URLs are internal-only; external clients upload via the gateway.
- Inference containers stay **offline** (`TRANSFORMERS_OFFLINE=1`); only model
  staging jobs get egress.
- All secrets live in `.env` (never commit it). `env.example` documents every var.
- Deploy through Ansible for managed hosts — these templates document the shape
  of what Ansible converges, not an ad-hoc SSH workflow.
