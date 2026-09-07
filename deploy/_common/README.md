# `deploy/_common` — shared cluster harness

Single implementation of the per-cluster deployment harness scripts.

## Why

These scripts previously existed as near-identical copies in every cluster
directory. The copies drifted, and the drift was invisible because nothing
compared them. Consolidation surfaced five real defects:

| Defect | Clusters affected |
|---|---|
| `ssh_tunnel.sh` defaulted to bdrt's login node `nimbus-0` | octopus |
| `ssh_tunnel.sh` used bdrt's gateway port `18080` | octopus |
| `setup_client.sh` wrote a connection profile named `bdrt` | octopus |
| Model root pointed at RCCS's `/lvs0/dne1/rccs-dmp` (no such path there) | octopus |
| `harness_probe.sh` probed bdrt's partitions `spark`/`spark-double` | octopus |
| `health_check_server.sh` asserted `cfg.name == "rccs-cloud"`, so the site-config check could never pass | bdrt, octopus |
| Missing r4-contracts install in `setup_server.sh` | bdrt, octopus |

See [ADR-019](../../../docs/adr/019-deployment-tree-consolidation.md).

## How it works

Nothing here is invoked directly. Each cluster directory holds a shim per
script that exports `UMRS_CLUSTER_DIR` and execs the shared implementation:

```bash
UMRS_CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UMRS_CLUSTER_DIR
exec "${UMRS_CLUSTER_DIR}/../_common/start.sh" "$@"
```

`cluster-context.sh` — sourced first by every shared script — resolves that
directory and loads the cluster's `cluster.env`. All state (`.env`, `.venv`,
`logs/`, `server.pid`, `site.yaml`) stays in the cluster directory, so the
operator- and Ansible-facing contract is unchanged:

```bash
cd deploy/<cluster> && ./start.sh
```

This matters because Ansible invokes these by path — `roles/login_node/tasks/rccs.yaml`
and `roles/front_end/tasks/umrs-front.yaml` call `start.sh`, `stop.sh`,
`setup_server.sh` and `health_check_server.sh` directly.

## Per-cluster parameters

Set in `deploy/<cluster>/cluster.env`. Every key has a default, so omitting one
is safe.

| Variable | Purpose |
|---|---|
| `UMRS_CLUSTER` | Cluster identity; defaults to the directory name |
| `UMRS_SITE_NAME` | Expected `name:` in `site.yaml` (rccs uses the legacy `rccs-cloud`) |
| `UMRS_MODEL_ROOT` | Fallback model root; `site.yaml` `storage.model_cache_dir` wins when loadable |
| `UMRS_SSH_ALIAS` | Login-node SSH alias, used by `ssh_tunnel.sh` and operator hints |
| `UMRS_GATEWAY_PORT` | Local gateway port (bdrt uses 18080; 8080 is taken by system PKI on nimbus-0) |
| `UMRS_CLIENT_PROFILE` | Connection profile name written by `setup_client.sh` |
| `UMRS_PARTITIONS` | Slurm partitions swept by `harness_probe.sh` |

## What stays per-cluster

Data and cluster-specific configuration are **not** shared: `site.yaml`,
`.env` / `env.example`, `docker-compose.yml`, `pool-*.yaml`, `*.slurm`,
`s3cfg`, `conf_umrs.json`, `client/`, `user_files/`.

## Known exception — ai4s

ai4s keeps its own `start.sh` and `stop.sh`. It runs a **fully containerised**
stack (postgres, minio, umrs-server and gateway all in compose), whereas
rccs/bdrt/octopus run the **hybrid** model — infra in podman, `umrs-server`
native on the login node so it can call `sbatch`/`squeue`/`scancel`.

Pointing ai4s at the shared hybrid `start.sh` would start the containerised
server via compose and then a second native server on top of it. ai4s is
currently offline and the change cannot be verified, so it was not made. The
migration steps are recorded in `deploy/ai4s/cluster.env`.

ai4s does use the shared implementation for the other seven scripts, which it
previously lacked entirely.

## Venue contract (ADR-045) — what a compute job may assume

`src/umrs/mini-rivault-llm/venue.sh` is sourced by the wrapper, by
`harness_probe.sh` and by `stage_images.sh`, so the probe validates exactly the
code a serving job runs:

| Concern | Mechanism | site.yaml knob |
|---|---|---|
| podman configuration | job-private `storage.conf`/`containers.conf` via `CONTAINERS_STORAGE_CONF`/`CONTAINERS_CONF`; node-local graphroot + network config dir; runroot under `$XDG_RUNTIME_DIR`; state DB re-pinned when idle. Nothing under `~/.config/containers` or `~/.config/cni` is touched — `$HOME` is shared across nodes. | `runtime.storage_root`, `runtime.network_config_dir`, `runtime.cgroup_manager` |
| private network | dead-job networks reaped (Slurm's view of liveness), then a /24 from the pool walked from `jobid % 254 + 1`, podman's allocator last; every failed attempt printed | `network.private_subnet_pool` (default `172.16.0.0/16`) |
| images | node store → `<archive_dir>/<name>-<tag>-<arch>.tar` (`podman load`) → pull → refuse naming the archive | `runtime.image_archive_dir` |
| GPU | override generated from `/dev/nvidia*` and the host's driver libraries | — |

`stage_images.sh` builds the declared egress-guard image (ADR-040 D4) from the
ai4s-platform checkout root — on a compute node of the target architecture via
`srun` when the login node's differs — and writes the archive. Ansible's
rccs-cloud login play runs it with `--if-missing`.

## Adding a cluster

1. Create `deploy/<cluster>/` with `site.yaml` (start from
   `templates/login-node/site.yaml`; fill `partition_profiles`,
   `runtime.image_archive_dir`, `network.private_subnet_pool`; a front overlay
   may `extends:` the compute file), `env.example`, `docker-compose.yml`.
2. Write `cluster.env`.
3. Copy the shims from an existing cluster — including `serve-model.slurm`,
   which is identical for every podman cluster (it locates everything from
   `RIVAULT_WRAPPER_DIR`; no paths, no cleanup, no per-cluster logic).
4. `./harness_probe.sh` — every partition must report `VENUE_CONFIG=ok`,
   `NETWORK_CREATED=yes`, and `EGRESS_GUARD_IMAGE=ready` (or `not_declared`).
5. `./stage_images.sh` when the site declares `network.egress_proxy_image`.
6. Add the Ansible inventory; deploy through Ansible only.
