# UMRS Full Stack Demo

Comprehensive end-to-end demonstration of all UMRS components and functionality.

## Components

| Service | Port(s) | Description |
|---------|---------|-------------|
| `umrs-server` | 8001 (HTTP), 5555/5556 (ZMQ) | Main UMRS server with ZeroMQ control plane |
| `umrs-embed-worker` | 6000 | Embedding computation worker |
| `postgres` | 5432 | PostgreSQL with pgvector extension |
| `minio` | 9000 (API), 9001 (Console) | S3-compatible object storage |
| `umrs-client` | - | Interactive CLI client container |

## Quick Start

### 1. Start the Environment

```bash
cd demos/umrs-full
docker compose up -d
```

### 2. Verify Services

```bash
# Check all services are running
docker compose ps

# Check UMRS server logs
docker compose logs -f umrs-server

# Check embedding worker logs
docker compose logs -f umrs-embed-worker
```

### 3. Run the Demo

```bash
# Mock mode (no Docker required - for testing)
python ../e2e_umrs_full.py --mock

# Live mode (requires running Docker environment)
python ../e2e_umrs_full.py --live
```

### 4. Interactive CLI Testing

```bash
# Enter the CLI client container
docker compose exec umrs-client bash

# Run UMRS client commands
python -m client_umrs ping
python -m client_umrs list_buckets
python -m client_umrs status
```

### 5. Access Web UIs

- **MinIO Console**: http://localhost:9001 (login: minioadmin/minioadmin)
- **PostgreSQL**: Connect to `localhost:5432` with psql or any database client

## Demo Workflow

The E2E demo demonstrates the following steps:

```
┌─────────────────────────────────────────────────────────────────────┐
│  E2E Demo: UMRS Full Stack                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Server Connection                                               │
│     └── Ping UMRS server via ZeroMQ                                │
│                                                                     │
│  2. Bucket Management                                               │
│     └── List available buckets for user                            │
│                                                                     │
│  3. File Upload (Client → S3 → Server)                             │
│     └── Upload documents to MinIO via UMRS                         │
│                                                                     │
│  4. Embedding Computation (Worker)                                  │
│     └── Compute embeddings using sentence-transformers             │
│                                                                     │
│  5. Similarity Search (pgvector)                                    │
│     └── Query for similar documents                                │
│                                                                     │
│  6. LLM Completion (via RiVault)                                    │
│     └── Generate response using retrieved context                  │
│                                                                     │
│  7. File Download (S3 → Client)                                     │
│     └── Download results/files from storage                        │
│                                                                     │
│  8. Server Status                                                   │
│     └── Get server statistics and health                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           UMRS Client                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Commands: ping, upload, download, status, recompute, ...   │    │
│  └────────────────────────┬────────────────────────────────────┘    │
│                           │                                         │
└───────────────────────────┼─────────────────────────────────────────┘
                            │ ZeroMQ (tcp://server:5555)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        UMRS Server                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ ZMQ Control     │  │ S3 Manager      │  │ DB Manager      │      │
│  │ (ROUTER/DEALER) │  │ (boto3)         │  │ (psycopg2)      │      │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │
│           │                    │                    │               │
│           │                    ▼                    ▼               │
│           │           ┌────────────────┐   ┌────────────────┐       │
│           │           │    MinIO       │   │   PostgreSQL   │       │
│           │           │  (S3 Storage)  │   │  (+ pgvector)  │       │
│           │           └────────────────┘   └────────────────┘       │
│           │                                                         │
│           │  ┌──────────────────────────────────────────────┐       │
│           └─►│ Embedding Worker Proxy                       │       │
│              └──────────────────┬───────────────────────────┘       │
│                                 │                                   │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │ ZeroMQ (tcp://worker:6000)
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Embedding Worker                                │
│  ┌─────────────────┐  ┌─────────────────┐                           │
│  │ Model Loader    │  │ Embedding       │                           │
│  │ (transformers)  │  │ Computation     │                           │
│  └────────┬────────┘  └────────┬────────┘                           │
│           │                    │                                    │
│           ▼                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  sentence-transformers: all-MiniLM-L6-v2 (384 dims)         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## UMRS Client Commands

The UMRS client supports the following operations:

| Command | Description |
|---------|-------------|
| `ping` | Check if server is ready |
| `list_buckets` | List available buckets for current user |
| `upload_file` | Upload a single file to S3 |
| `upload_dir` | Upload a directory recursively |
| `download_results` | Download files from S3 |
| `status` | Get server status and statistics |
| `recompute` | Trigger embedding recomputation |
| `update` | Update bucket configuration |
| `check` | Validate bucket configuration |
| `reload` | Reload server configuration |

### Example Client Usage

```bash
# From the client container
docker compose exec umrs-client bash

# Ping the server
python -m client_umrs ping

# List buckets
python -m client_umrs list_buckets

# Upload a file
python -m client_umrs upload_file /demo_data/sample.txt demo_bucket

# Get status
python -m client_umrs status all

# Trigger recomputation
python -m client_umrs recompute
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UMRS_SERVER_HOST` | localhost | UMRS server hostname |
| `UMRS_CONTROL_PORT` | 5555 | ZeroMQ control channel port |
| `UMRS_DATA_PORT` | 5556 | ZeroMQ data channel port |
| `S3_ENDPOINT` | http://localhost:9000 | MinIO endpoint |
| `S3_ACCESS_KEY` | minioadmin | MinIO access key |
| `S3_SECRET_KEY` | minioadmin | MinIO secret key |
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_USER` | postgres | PostgreSQL user |
| `DB_PASSWORD` | postgres | PostgreSQL password |
| `DB_NAME` | umrs_demo | Database name |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |

## Development

### Adding Demo Data

Create files in the `demo_data/` directory:

```bash
mkdir -p demo_data
echo "Sample document content" > demo_data/sample.txt
```

These will be mounted in the client container at `/demo_data/`.

### Rebuilding Images

```bash
docker compose build --no-cache
docker compose up -d
```

### Viewing Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f umrs-server

# Last 100 lines
docker compose logs --tail=100 umrs-embed-worker
```

### Database Access

```bash
# Connect to PostgreSQL
docker compose exec postgres psql -U postgres -d umrs_demo

# Query embeddings
SELECT doc_id, model_name FROM embeddings;

# Query documents
SELECT id, title, s3_key FROM documents;
```

## Cleanup

```bash
# Stop services
docker compose down

# Stop and remove volumes (data will be lost)
docker compose down -v

# Full cleanup including images
docker compose down -v --rmi local
```

## Troubleshooting

### Server Connection Failed

```bash
# Check if server is running
docker compose ps umrs-server

# Check server logs
docker compose logs umrs-server

# Test ZeroMQ connection
docker compose exec umrs-client python -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://umrs-server:5555')
sock.send_string('ping')
print(sock.recv_string())
"
```

### MinIO Connection Failed

```bash
# Check MinIO health
curl http://localhost:9000/minio/health/live

# Check MinIO logs
docker compose logs minio
```

### Database Connection Failed

```bash
# Check PostgreSQL
docker compose exec postgres pg_isready -U postgres

# Check if pgvector is installed
docker compose exec postgres psql -U postgres -d umrs_demo -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
```

## Related Documentation

- [UMRS Main Documentation](../../Readme.md)
- [E2E RiVault/Slurm Demo](../e2e_rivault_slurm_rag.py)
- [Slurm Development Environment](../slurm-dev/)
- [Mini-RiVault](../../rivault/MINI-RIVAULT.md)
