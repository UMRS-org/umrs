# Frontend → Multiple-HPC Deployment Template

A single **persistent frontend server** fronts **multiple HPC clusters**. Each
cluster gets its own containerized UMRS server that reaches its HPC backend
**over SSH** (`scheduler.kind: slurm-remote`). A gateway path-routes clients to
each cluster (`/cluster-a`, `/cluster-b`).

```
                     ┌─> UMRS(cluster-a) ──SSH──> login-a ──sbatch──> compute-a
   client ──> gateway ┤
                     └─> UMRS(cluster-b) ──SSH──> login-b ──sbatch──> compute-b

   frontend server (this host) also runs:
     ├─ Postgres (one DB per cluster)
     └─ MinIO    (shared object store)
```

Why this topology: a stable entry point decoupled from any single login node's
uptime, and one place to add clusters. Distilled from the production
`../umrs-front/` harness. Fill in every `<...>` placeholder.

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yaml` | Postgres + MinIO + one `umrs-server-<cluster>` per HPC backend. |
| `site.cluster-a.yaml` | `SiteConfig` for cluster A (`slurm-remote` + SSH target). |
| `site.cluster-b.yaml` | `SiteConfig` for cluster B — copy per additional cluster. |
| `env.example` | Shared infra secrets + per-cluster ports/DBs. |
| `gateway.nginx.conf` | Path routing `/cluster-a`, `/cluster-b` → each UMRS HTTP port. |
| `start.sh` / `stop.sh` | Bring the whole stack up/down. |

## Add a cluster (the core pattern)

1. Copy `site.cluster-a.yaml` → `site.cluster-c.yaml`; set `name`,
   `scheduler.partition/account`, and `extra.ssh_target/ssh_user` for the new
   login node.
2. Add a `umrs-server-cluster-c` service in `docker-compose.yaml` with a unique
   HTTP port, ZMQ control port, data-port range, and its own Postgres DB.
3. Add an `upstream` + `location /cluster-c/` block in `gateway.nginx.conf`.
4. Add the port/DB vars to `.env`.

## Prerequisites

- Rootless Podman + compose on the frontend host.
- SSH key auth from the frontend host to **each** cluster's login node
  (`ssh <ssh_user>@<ssh_target>` works non-interactively).
- The UMRS server image built once: `podman compose build umrs-server-cluster-a`.

## Deploy

```bash
cp env.example .env
$EDITOR .env                 # secrets + per-cluster ports/DBs
$EDITOR site.cluster-a.yaml  # SSH target + Slurm partition/account per cluster
$EDITOR gateway.nginx.conf   # server_name + upstreams

./start.sh                   # infra + per-cluster UMRS servers
./stop.sh                    # tear down (volumes preserved)
```

## Key `SiteConfig` fields for the remote pattern

- `scheduler.kind: slurm-remote` — Slurm commands are issued over SSH.
- `extra.ssh_target` / `extra.ssh_user` — the cluster's login node.
- `extra.slurm_wrapper_script` — compute-node serve wrapper on the remote FS.
- `extra.ssh_reverse_tunnels` — expose the frontend MinIO on the login node so
  compute nodes reach it through their existing `login:9000` tunnel.
- `storage.model_cache_dir: ""` — the frontend has no local model FS; weights
  live on the cluster's own filesystem.

## Networking notes

- Presigned S3 URLs stay internal; external clients upload through the gateway.
- MinIO binds `0.0.0.0` only on trusted cluster paths — restrict with a firewall.
- Each UMRS server uses host networking and a distinct HTTP/ZMQ port set to avoid
  collisions on the shared frontend host.
