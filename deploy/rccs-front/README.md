# UMRS Front-End Server Deployment (rccs-front)

## Overview

This deployment profile runs all persistent UMRS services on a front-end
server (Linux VM) that is directly accessible to clients via SSH. Slurm job
submission is handled remotely via SSH to the RCCS login node.

## Architecture

```
┌──────────┐          ┌──────────────────────────────────────┐
│  Client  │──SSH────▶ │  Front-End Server (umrs-front)       │
│  (r4 CLI)│          │  ┌──────────┐  ┌──────────────────┐  │
└──────────┘          │  │ Gateway  │  │  UMRS Server     │  │
                      │  │  :8080   │  │  :8000           │  │
                      │  └──────────┘  └────────┬─────────┘  │
                      │  ┌──────────┐           │             │
                      │  │ Postgres │     SSH ControlMaster   │
                      │  │  :5432   │           │             │
                      │  └──────────┘           │             │
                      │  ┌──────────┐           │             │
                      │  │  MinIO   │           │             │
                      │  │  :9000   │◀──────────┼── reverse   │
                      │  └──────────┘    tunnel │             │
                      └──────────────────────────┼─────────────┘
                                                 │
                                   SSH (key-based, multiplexed)
                                                 │
                      ┌──────────────────────────▼─────────────┐
                      │  Login Node (rccs-cloud)                │
                      │  - sbatch/squeue/scancel via SSH        │
                      │  - Reverse tunnel: 9000 → front MinIO  │
                      │  - NO persistent services               │
                      └──────────────────────────┬─────────────┘
                                                 │
                                           sbatch/Slurm
                                                 │
                      ┌──────────────────────────▼─────────────┐
                      │  Compute Nodes (ng-dgx-m*)             │
                      │  - Mini-RiVault (vLLM + LiteLLM)       │
                      │  - Models from Lustre                   │
                      │  - SSH tunnel to login:9000 for MinIO  │
                      └────────────────────────────────────────┘
```

## Ansible Deployment (Recommended)

The `umrs-front` Ansible profile automates setup, bring-up, idempotent
reconciliation, and readiness verification for both clusters.

```bash
cd deploy/ansible

# Deploy RCCS Cloud backend only (on umrs-front)
./bootstrap.sh --profile umrs-front-rccs

# Deploy AI4S backend only (on umrs-front)
./bootstrap.sh --profile umrs-front-ai4s

# Deploy both clusters simultaneously
./bootstrap.sh --profile umrs-front

# Dry run
./bootstrap.sh --profile umrs-front --check

# Stop UMRS servers (keeps Postgres + MinIO running)
./bootstrap.sh --profile umrs-front --teardown
```

The playbook runs `setup.sh` on first deploy (creates `.venv`, seeds `.env`
template), then `start.sh`, and verifies readiness via PID + ZMQ port probes
for each requested cluster. Re-running is idempotent when services are healthy.

See [deploy/ansible/README.md](../../../deploy/ansible/README.md) for full
options and the `inventory/umrs-front/` group_vars reference.

## Manual Setup

1. **SSH Key Configuration**

   Ensure the front-end server can SSH to `rccs-cloud` without a password:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_umrs
   ssh-copy-id -i ~/.ssh/id_umrs rccs-cloud
   ```

   Add to `~/.ssh/config`:
   ```
   Host rccs-cloud
     HostName <login-node-hostname>
     User masaru.nagaso
     IdentityFile ~/.ssh/id_umrs
     ControlMaster auto
     ControlPath /tmp/umrs-ssh-%r@%h:%p
     ControlPersist 10m
   ```

2. **Environment Configuration**

   ```bash
   cp env.example .env
   # Edit .env with real passwords and keys
   ```

3. **Start Services**

   ```bash
   chmod +x start.sh stop.sh
   ./start.sh
   ```

4. **Verify**

   ```bash
   curl http://localhost:8000/health       # UMRS
   curl http://localhost:8080/health       # Gateway
   ```

## Key Differences from rccs-cloud

| Aspect | rccs-cloud | rccs-front |
|--------|------------|------------|
| UMRS runs | Natively on login node | In container on front-end |
| Slurm access | Direct (sbatch on PATH) | Via SSH ControlMaster |
| Lustre access | Direct mount | Via SSH (cat, ls) |
| Client access | SSH tunnel required | Direct SSH (134.160.x.x public range) |
| Persistent services on login | Yes (Postgres, MinIO) | None |
| MinIO accessibility | From compute via SSH tunnel | Via reverse tunnel chain |

## Troubleshooting

- **SSH connection fails**: Check `ssh -v rccs-cloud` from within the UMRS container
- **MinIO unreachable from compute**: Verify reverse tunnel `ss -tlnp | grep 9000` on login node
- **Jobs not submitting**: Check UMRS logs for SSH errors: `podman logs umrs-server`
