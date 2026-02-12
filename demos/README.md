# UMRS Demos

End-to-end demonstrations of UMRS capabilities for the AI-for-Science Platform.

## Available Demos

### e2e_umrs_full.py (NEW - Docker-based Full Stack)

**Comprehensive UMRS demonstration** using real services in Docker Compose.

This demo showcases the complete UMRS workflow with:
1. **ZeroMQ Control Plane**: Client-server communication via UMRS protocol
2. **S3 Storage (MinIO)**: File upload/download with presigned URLs
3. **PostgreSQL + pgvector**: Embedding storage and similarity search
4. **RAG Pipeline**: Hybrid search with real embeddings
5. **LLM Completion**: Query via Mini-RiVault endpoint (mock/live)

#### Quick Start

```bash
# Start Docker environment (MinIO + PostgreSQL + UMRS Server)
cd umrs/demos/umrs-full
docker compose up -d

# Verify services are running
docker ps

# Run the demo in LIVE mode (uses real services)
cd ../..  # back to umrs/
python demos/e2e_umrs_full.py --live \
    --control-port 15555 \
    --s3-endpoint http://localhost:19000 \
    --db-port 15432

# Clean up
cd demos/umrs-full && docker compose down -v
```

#### UMRS Protocol

The demo uses the real UMRS ZeroMQ protocol:

| Command | Description | Response |
|---------|-------------|----------|
| `HAND_CHECK` | Check server availability | `READY!` |
| `CREATE_DATA_QUEUE:client_id` | Create data transfer queue | `tcp://ip:port` |
| `STATUS:client_id:key` | Get processing status | JSON status |

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--live` | Use real connections (requires Docker environment) | No |
| `--mock` | Use mock implementations (no Docker required) | Yes |
| `--server-host` | UMRS server hostname | `localhost` |
| `--control-port` | ZeroMQ control port | `5555` |
| `--s3-endpoint` | MinIO S3 endpoint URL | `http://localhost:9000` |
| `--db-port` | PostgreSQL port | `5432` |
| `--query` | Query for RAG demo | `"How does UMRS handle file transfer?"` |

#### Docker Services (umrs-full/)

| Service | Port | Description |
|---------|------|-------------|
| `minio` | 19000 | S3-compatible object storage |
| `umrs-postgres` | 15432 | PostgreSQL with pgvector |
| `umrs-server` | 15555, 15556 | UMRS control/data plane |

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `UMRS_SERVER_HOST` | UMRS server hostname |
| `UMRS_CONTROL_PORT` | ZeroMQ control port |
| `S3_ENDPOINT` | MinIO endpoint URL |
| `DB_HOST` | PostgreSQL host |
| `LITELLM_PORT` | LiteLLM proxy port |

#### Sample Output (Live Mode)

```
╔══════════════════════════════════════════════════════════════════════╗
║                      E2E DEMO: UMRS Full Stack                       ║
║                              Mode: LIVE                              ║
╚══════════════════════════════════════════════════════════════════════╝

======================================================================
STEP 1: UMRS Server Connection
======================================================================
✓ UMRS server is ready
  - Host: localhost
  - Control Port: 15555

======================================================================
STEP 2: Bucket Management
======================================================================
✓ Found 1 bucket(s) for user 'demo_user':
  - demo_bucket: text → demo/bucket

======================================================================
STEP 3: File Upload (Client → S3 → Server)
======================================================================
→ Uploading 5 documents...
  ✓ Seismic Wave Propagation → doc1.txt
  ✓ Machine Learning for Geophysics → doc2.txt
  ✓ HPC for Scientific Computing → doc3.txt
  ✓ RiVault Architecture → doc4.txt
  ✓ UMRS File Transfer Protocol → doc5.txt
✓ Uploaded 5 documents successfully

======================================================================
STEP 4: Embedding Computation (Worker)
======================================================================
→ Computing embeddings for 5 documents...
  ✓ doc1: dim=384, norm=1.0000
  ✓ doc2: dim=384, norm=1.0000
  ✓ doc3: dim=384, norm=1.0000
  ✓ doc4: dim=384, norm=1.0000
  ✓ doc5: dim=384, norm=1.0000
✓ Computed 5 embeddings

======================================================================
STEP 5: Similarity Search (pgvector)
======================================================================
Query: "How does UMRS handle file transfer and communication?"
✓ Search results:
  [1] UMRS File Transfer Protocol
      Score: 0.6565
  [2] RiVault Architecture
      Score: 0.2830
  [3] Seismic Wave Propagation
      Score: 0.0698

======================================================================
✓ Demo completed successfully!
======================================================================
```

---

### e2e_rivault_slurm_rag.py

Complete workflow demonstration showing:
1. **Model Registry**: Register and query models
2. **Orchestration**: Submit Mini-RiVault jobs via Slurm
3. **RAG Pipeline**: Hybrid search (BM25 + vector similarity)
4. **LLM Completion**: Query the provisioned endpoint with RAG context

#### Quick Start

```bash
# Mock mode (no Slurm required)
cd umrs
python demos/e2e_rivault_slurm_rag.py --mock

# Custom query
python demos/e2e_rivault_slurm_rag.py --mock \
    --query "How does HPC help scientific computing?"

# Live mode (on HPC cluster with Slurm)
python demos/e2e_rivault_slurm_rag.py --live \
    --model-path /models/qwen-0.5b \
    --partition gpu
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mock` | Use mock implementations (no Slurm) | Yes |
| `--live` | Use actual Slurm orchestrator | No |
| `--model-path` | Path to model weights | `/models/rivault-gpt-oss` |
| `--partition` | Slurm partition name | `gpu` |
| `--query` | Question to ask the RAG system | (seismic wave example) |

#### Environment Variables

| Variable | Description |
|----------|-------------|
| `RIVAULT_ORCHESTRATOR` | Backend: slurm, k8s, compose |
| `SLURM_PARTITION` | Default partition |
| `LITELLM_PORT` | LiteLLM proxy port |
| `LITELLM_MASTER_KEY` | API key for LiteLLM |
| `DEMO_MODEL_PATH` | Default model path |

#### Sample Output

```
╔════════════════════════════════════════════════════════════════════╗
║             E2E DEMO: RiVault + Slurm + RAG                        ║
║                        Mode: MOCK                                   ║
╚════════════════════════════════════════════════════════════════════╝

======================================================================
STEP 1: Model Registry
======================================================================
✓ Model entry created:
  - ID: demo-qwen-0.5b
  - Name: Qwen-0.5B Demo
  - Scope: project
  - Path: /models/rivault-gpt-oss

======================================================================
STEP 2: Orchestration (Slurm → Mini-RiVault)
======================================================================
✓ Using MOCK orchestrator (no Slurm)
✓ Mock endpoint ready:
  - Host: localhost
  - Port: 11434

======================================================================
STEP 3: RAG Pipeline (Hybrid Search)
======================================================================
Query: "How does seismic wave propagation work..."

→ Running hybrid search (BM25 + Vector)...
✓ Search results (top 3):
  [1] Seismic Wave Propagation
      Score: 0.0142 (BM25: 5.2341, Vector: 0.6821)
  [2] Machine Learning for Geophysics
      Score: 0.0138 (BM25: 3.1245, Vector: 0.7123)
  ...

======================================================================
STEP 4: LLM Completion (via RiVault)
======================================================================
✓ [MOCK] Response generated

──────────────────────────────────────────────────
Answer:
Based on the retrieved documents, seismic wave propagation follows
the elastic wave equation. P-waves travel faster than S-waves...
──────────────────────────────────────────────────

======================================================================
✓ Demo completed successfully!
======================================================================
```

## Adding New Demos

When adding demos, follow this pattern:

1. Create a Python script in `demos/`
2. Support both `--mock` and `--live` modes
3. Use `asyncio` for async operations
4. Include comprehensive docstrings
5. Update this README

---

## UMRS Full Stack Environment (umrs-full/)

Docker Compose environment for running the complete UMRS stack locally.

### Overview

The `umrs-full/` directory provides:
- **MinIO**: S3-compatible object storage (port 19000)
- **PostgreSQL + pgvector**: Vector database (port 15432)
- **UMRS Server**: ZeroMQ control plane (port 15555, 15556)

### Quick Start

```bash
cd demos/umrs-full

# Start the environment
docker compose up -d

# Check server logs
docker logs umrs-demo-server --tail 20

# Run demo
cd ../..
python demos/e2e_umrs_full.py --live \
    --control-port 15555 \
    --s3-endpoint http://localhost:19000 \
    --db-port 15432

# Stop environment
cd demos/umrs-full && docker compose down -v
```

### Files

| File | Description |
|------|-------------|
| `docker-compose.yaml` | Main compose file |
| `init-db.sql` | PostgreSQL initialization with pgvector |
| `user_files/demo_user.json` | Demo user configuration |
| `s3cfg` | MinIO access credentials |
| `Dockerfile` | UMRS server container |

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Docker Compose (umrs-full)                                         │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐    │
│  │  umrs-server    │   │  umrs-postgres  │   │     minio       │    │
│  │  Port: 15555    │──▶│  Port: 15432    │   │  Port: 19000    │    │
│  │  (ZeroMQ)       │   │  (pgvector)     │   │  (S3)           │    │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘    │
│          │                      ▲                    ▲              │
│          │                      │ SQL                │ S3 API       │
│          └──────────────────────┴────────────────────┘              │
│                          UMRS Server watches bucket                 │
└─────────────────────────────────────────────────────────────────────┘
         ▲
         │ ZeroMQ (HAND_CHECK, STATUS, etc.)
         │
┌─────────────────────────────────────────────────────────────────────┐
│  HOST MACHINE (e2e_umrs_full.py)                                    │
│  - Connects via ZeroMQ to check server readiness                    │
│  - Uploads files directly to MinIO S3                               │
│  - Stores embeddings in PostgreSQL                                  │
│  - Runs pgvector similarity search                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### S3 Bucket Naming Convention

UMRS uses a specific bucket naming pattern:
```
{username}.{experiment_name}.{bucket_alias}
```

Example: `demo-user.demo-experience.demo-bucket`

**Note**: Names are lowercase with underscores converted to hyphens.

---

## Slurm Development Environment

For running demos in **live mode** on a single workstation, we provide a Docker-based Slurm cluster.

### Overview

The `slurm-dev/` directory contains a Docker Compose setup that creates:
- **slurmctld**: Slurm controller daemon
- **compute-01**: Slurm compute node
- **munge**: Authentication service
- **postgres**: Database for UMRS
- **minio**: S3-compatible storage

Mini-RiVault runs **outside** of UMRS (either on the host or in a separate GPU container).

### Quick Start

```bash
cd demos/slurm-dev

# Configure environment
cp env.template .env
# Edit .env to set MODEL_PATH and other settings

# Start the cluster
chmod +x setup.sh
./setup.sh start

# Check status
./setup.sh status

# Submit a test job
./setup.sh test

# Run the E2E demo in live mode
cd ..
python e2e_rivault_slurm_rag.py --live \
    --model-path /models/qwen-0.5b
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  HOST MACHINE                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Docker Compose (slurm-dev)                                 │    │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │    │
│  │  │ slurmctld   │   │ compute-01  │   │   munge     │        │    │
│  │  │ Port: 6817  │◀─▶│ SSH: 2222   │◀─▶│             │        │    │
│  │  └─────────────┘   └─────────────┘   └─────────────┘        │    │
│  │         │                 │                                 │    │
│  │         └─────────┬───────┘                                 │    │
│  │                   ▼                                         │    │
│  │         ┌─────────────────┐                                 │    │
│  │         │ /var/run/rivault│ (job info files)                │    │
│  │         └─────────────────┘                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Mini-RiVault (external - runs on HOST or GPU container)    │    │
│  │  - vLLM: Port 8000                                          │    │
│  │  - LiteLLM: Port 11434                                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Files

| File | Description |
|------|-------------|
| `docker-compose.yaml` | Main compose file |
| `slurm.conf` | Slurm configuration |
| `mini-rivault-job.sh` | Job script that runs on compute node |
| `env.template` | Environment variable template |
| `setup.sh` | Convenience setup script |
| `Dockerfile.*` | Container definitions |

### Running with GPU

To use GPU acceleration:

1. Install NVIDIA Container Toolkit on the host
2. Uncomment GRES lines in `slurm.conf`
3. Modify `docker-compose.yaml` to add GPU runtime:

```yaml
compute-01:
  # ... existing config ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Connecting External Mini-RiVault

When a job starts, it can operate in two modes:

**Internal mode** (default for testing):
- Mini-RiVault runs inside the compute container
- Uses a mock LiteLLM server

**External mode** (for GPU inference):
- Job script waits for external mini-rivault
- Start mini-rivault on the host with GPU access:

```bash
# On host machine
cd rivault
./mini-rivault-start.sh
```

### Troubleshooting

**Munge authentication failure:**
```bash
./setup.sh clean
./setup.sh start
```

**Node not responding:**
```bash
docker exec slurmctld scontrol update nodename=compute-01 state=resume
```

**View logs:**
```bash
docker compose logs -f slurmctld
docker compose logs -f compute-01
```

---

## Related Documentation

- [UMRS Documentation](../doc/)
- [RAG Pipeline Tests](../tests/test_rag_pipeline.py)
- [Orchestrator Module](../src/umrs/orchestrator/)
- [Model Registry](../src/umrs/model_registry/)
- [Mini-RiVault Guide](../../rivault/MINI-RIVAULT.md)

