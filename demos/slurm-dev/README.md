# Slurm Development Environment

A Docker Compose setup for running a real Slurm cluster on a single workstation.
Use this to test the E2E demo in **live mode** without access to an HPC cluster.

## Prerequisites

- Docker and Docker Compose
- 8GB+ RAM available
- (Optional) NVIDIA GPU + Container Toolkit for GPU support

## Quick Start

```bash
# 1. Configure environment
cp env.template .env
# Edit .env to set MODEL_PATH to your local models directory

# 2. Start the cluster
./setup.sh start

# 3. Verify it's working
./setup.sh status

# 4. Submit a test job
./setup.sh test

# 5. Run the E2E demo
cd ..
python e2e_rivault_slurm_rag.py --live
```

## Components

| Service | Description | Port |
|---------|-------------|------|
| slurmctld | Slurm controller | 6817 |
| compute-01 | Compute node | SSH: 2222 |
| munge | Authentication | - |
| postgres | Database | 5432 |
| minio | S3 storage | 9000, 9001 |

## Shell Commands

```bash
# Start cluster
./setup.sh start

# Stop cluster
./setup.sh stop

# Show status
./setup.sh status

# Submit test job
./setup.sh test

# Clean up everything
./setup.sh clean
```

## Direct Slurm Commands

```bash
# Show nodes
docker exec slurmctld sinfo

# Show job queue
docker exec slurmctld squeue

# Submit a job
docker exec slurmctld sbatch /etc/slurm/mini-rivault-job.sh

# Cancel a job
docker exec slurmctld scancel <job_id>

# View job output
docker exec slurmctld cat /var/run/rivault/rivault-job-<job_id>.json
```

## Mini-RiVault Modes

### Internal Mode (Testing)

The job script includes a mock LiteLLM server for testing without GPU:

```bash
docker exec -e MINI_RIVAULT_MODE=internal slurmctld \
    sbatch /etc/slurm/mini-rivault-job.sh
```

### External Mode (Production)

The job waits for an external mini-rivault to connect:

```bash
# In Docker container (submits job)
docker exec slurmctld sbatch /etc/slurm/mini-rivault-job.sh

# On host (start mini-rivault with GPU)
cd rivault
export LITELLM_PORT=11434
./mini-rivault-start.sh
```

## Customization

### Adding GPU Support

1. Edit `slurm.conf` to uncomment GRES lines:
   ```
   GresTypes=gpu
   NodeName=compute-01 Gres=gpu:1
   ```

2. Edit `docker-compose.yaml` to add GPU resources:
   ```yaml
   compute-01:
     deploy:
       resources:
         reservations:
           devices:
             - driver: nvidia
               count: 1
               capabilities: [gpu]
   ```

### Adding More Compute Nodes

1. Add new service in `docker-compose.yaml`:
   ```yaml
   compute-02:
     build: ...
     hostname: compute-02
   ```

2. Update `slurm.conf`:
   ```
   NodeName=compute-[01-02] CPUs=4 RealMemory=8192 State=UNKNOWN
   PartitionName=dev Nodes=compute-[01-02] Default=YES MaxTime=INFINITE State=UP
   ```

## Troubleshooting

### "Node is down"

```bash
docker exec slurmctld scontrol update nodename=compute-01 state=resume
```

### "Munge authentication failed"

```bash
./setup.sh clean
./setup.sh start
```

### "Connection refused"

Wait for services to start (30-60 seconds), or check logs:
```bash
docker compose logs -f
```

### Check service health

```bash
docker compose ps
docker exec slurmctld sinfo -N
```

## File Structure

```
slurm-dev/
├── docker-compose.yaml      # Main compose file
├── slurm.conf               # Slurm configuration
├── mini-rivault-job.sh      # Job script
├── env.template             # Environment template
├── setup.sh                 # Setup script
├── Dockerfile.munge         # Munge container
├── Dockerfile.slurmctld     # Controller container
├── Dockerfile.slurmd        # Compute container
├── entrypoint-munge.sh      # Munge entrypoint
├── entrypoint-slurmctld.sh  # Controller entrypoint
└── entrypoint-slurmd.sh     # Compute entrypoint
```
