# rccs-cloud Demos

Lightweight demos for the rccs-cloud deployment.
These run **on your local Mac** through SSH tunnels — no `podman exec` needed.

## Prerequisites

1. SSH tunnels are running (`./ssh_tunnel.sh` from `deploy/rccs-cloud/`).
2. Client is set up (`./setup_client.sh` was run successfully).
3. The UMRS server is running on rccs-cloud.

## Demos

| # | Script | What it does |
|---|--------|--------------|
| 00 | [00_health_check.sh](00_health_check.sh) | Ping the server, check MinIO and RAG reachability |
| 01 | [01_text_embedding.sh](01_text_embedding.sh) | Upload text files → compute embeddings → check status |
| 02 | [02_rag_search.sh](02_rag_search.sh) | Semantic search: vector, hybrid, and Q&A queries |

## Quick start

```bash
cd deploy/rccs-cloud

# Terminal 1: start tunnels
./ssh_tunnel.sh

# Terminal 2: run demos
cd demos
./00_health_check.sh
./01_text_embedding.sh
./02_rag_search.sh
```

## Configuration

All demos use the user config installed by `setup_client.sh`:

```
~/.config/umrs/users/user1.json   # user config (-c flag)
~/.config/umrs/s3cfg              # S3 credentials
```

Override the config path with `UMRS_USER_CONF`:

```bash
UMRS_USER_CONF=~/my_custom_user.json ./01_text_embedding.sh
```
