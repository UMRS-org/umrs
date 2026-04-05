# U.M.R.S. Demo and Usage Guide

> **This document is outdated.** The current demo guide is at **[demos/README.md](demos/README.md)**.
>
> For deployment instructions, see [doc/admin_guide.md](doc/admin_guide.md)
> and [doc/user_guide.md](doc/user_guide.md).

---

<details>
<summary>Legacy bare-metal instructions (archived)</summary>

The original bare-metal demo instructions below are preserved for reference only.
They reference old bucket names (`bucket-1`, `bucket-2`), old config paths, and a
`demo` git branch that is no longer maintained. Use `demos/README.md` instead.

## Prerequisites

- Python 3.10+ (for local/bare-metal)
- PostgreSQL 15+ with pgvector extension
- S3-compatible storage (e.g., MinIO) and s3cfg files for admin and users
- Repo dependencies installed (see `requirements.txt`)


## Quick Start (local)

```bash
cd src
pip install -e .
./server_umrs.py          # Window 1: start server
./client_umrs.py ping     # Window 2: verify connectivity
```

## Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build --abort-on-container-exit --renew-anon-volumes
```

## Remote Deployment

On the server host:
```bash
cd /path/to/umrs/src
./server_umrs.py
```

On the client host:
```bash
./client_umrs.py --server-ip <server-ip-or-dns> ping
```

## Troubleshooting

- "Unknown metadata key(s)…": Add the key(s) to the `sql.meta_data` list for the modality in the config and restart the server.
- "No bucket … for modality … and user …": Verify the bucket alias matches the modality for that user.
- Client can't reach server: set `UMRS_SERVER_HOST` or `--server-ip` to the server hostname/IP; ensure TCP 5555 is reachable.

</details>
