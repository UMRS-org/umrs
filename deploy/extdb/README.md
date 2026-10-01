# `deploy/extdb` — external stores for one UMRS project

A UMRS project can keep its objects (S3) and its Postgres **outside** the
front-end — next to an HPC site's large filesystem — and point UMRS at them by
URI/URL ([ADR-077](../../../docs/adr/077-per-project-external-storage-binding.md)).
The front-end deployment is unchanged: projects without a binding keep using
umrs-front's own SeaweedFS and Postgres.

`umrs-extdb.sh` starts such a pair as rootless podman containers on a login
node. It is an operator tool run by hand (maintainer decision 2026-10-01), not
an Ansible-converged service.

| Container | Image | Published | Data |
|---|---|---|---|
| `umrs-extdb-postgres` | `pgvector/pgvector:pg15` (same major as umrs-front) | `${EXTDB_PG_PORT:-25432}` — TLS on, SCRAM, `pg_hba` admits only `EXTDB_PG_ALLOW` (default umrs-front `10.134.0.8/32`) | `$EXTDB_ROOT/postgres` |
| `umrs-extdb-seaweedfs` | `chrislusf/seaweedfs:4.47` (same as umrs-front) | `${EXTDB_S3_PORT:-28900}` — S3 API only; master/volume/filer/IAM stay in the container | `$EXTDB_ROOT/seaweedfs/data` |

`EXTDB_ROOT` defaults to `/lvs0/dne1/rccs-dmp/$USER/umrs-extdb` (rccs Lustre).

## Use

```bash
# on the login node
umrs-extdb.sh up                 # dirs, secrets (once), configs, containers, health
umrs-extdb.sh project opossum    # role + database + pgvector; writes secrets/project-opossum.env
umrs-extdb.sh status             # state + probes (also refreshes /var/tmp atimes)

# on the workstation — the env file carries both locators; r4 sends them once
ssh rccs-cloud cat /lvs0/dne1/rccs-dmp/$USER/umrs-extdb/secrets/project-opossum.env \
  | r4 project storage set opossum --env-file -
r4 project storage check opossum # probe from the UMRS server
```

`umrs-extdb.sh down` removes the containers and keeps the data.

## Why the script looks the way it does (rccs login1, verified 2026-10-01)

- **Isolated podman config.** Every serve-model Slurm job rewrites
  `~/.config/containers/storage.conf` with its own runroot (`$HOME` is shared
  with the compute nodes), which orphans long-running containers. The script
  points `CONTAINERS_STORAGE_CONF` / `CONTAINERS_CONF` at `$EXTDB_ROOT/podman`
  with graphroot and runroot under `/var/tmp/umrs-extdb-$UID` (overlay does not
  work on Lustre).
- **tmpfs socket dirs.** Unix sockets do not work on the rootless
  fuse-overlayfs layer: Postgres failed with "could not bind Unix address"
  after initdb, SeaweedFS' gRPC sockets answered "permission denied". The
  socket directories (`/var/run/postgresql`, SeaweedFS `/tmp`) are tmpfs.
- **keep-id.** `--userns=keep-id --user $UID:$GID` keeps every file on Lustre
  owned by the operator (no subuid-owned files on the parallel filesystem).
- **No secrets in argv.** Container secrets ride `--env-file` (0600): any user
  of a shared login node can read the process table.
- **tmpfiles age-out.** `/var/tmp` is cleaned after 30 days without access;
  `up`/`status` refresh access times. Run `status` at least every two weeks.
- **Bounded size.** SeaweedFS gets 200 × 1 GiB volumes (each new bucket takes
  7 slots), inside the shared project quota.

Plain HTTP on the S3 port is acceptable only on a trusted internal network;
put TLS in front before using a store across sites.
