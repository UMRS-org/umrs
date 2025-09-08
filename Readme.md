# UMRS - Unified Multi-modal Retrieval System

UMRS is a comprehensive multi-modal retrieval system that supports secure HTTPS communication and multi-user environments. The system processes text and images through various AI models and stores embeddings in PostgreSQL with PGVector extension.

## Key Features

- **HTTPS Support**: Secure SSL/TLS encrypted communication between all components
- **Multi-User Architecture**: Isolated user environments with separate buckets and data
- **Multi-Modal Processing**: Support for text, images, and text files
- **Flexible AI Models**: CLIP, ResNet, E5, and MPNET model support
- **FastAPI Integration**: RESTful API with comprehensive endpoints
- **Docker Containerization**: Complete containerized deployment with Docker Compose
- **S3 Storage**: MinIO-based object storage with bucket management
- **Vector Database**: PostgreSQL with PGVector for efficient similarity search

## Directory Structure

- `src/` — Main UMRS source code and implementation
- `docker/` — Docker containerization and multi-user setup
- `data/` — Test data used in experiments (images, texts, models)
- `requirements.txt` — Python dependencies for server
- `requirements-client.txt` — Python dependencies for client

## Quick Start

### Multi-User HTTPS Setup & Automated Testing

1. **Navigate to the project root (recommended):**
  ```bash
  cd /path/to/umrs
  ```

2. **Run the full build, deployment, and test suite:**
  ```bash
  ./docker/rebuild_and_start.sh
  ```
  This script will:
  - Stop and clean up any previous containers, networks, and volumes
  - Build all Docker images from scratch
  - Start all services (UMRS server, MinIO, PostgreSQL, test containers)
  - Run the automated multi-user test suite (see `docker/test_multiuser.sh` and `docker/test_umrs_api.sh`)
  - Output test results and shut down containers when complete

3. **Access the system (if running interactively):**
  - UMRS Server: `https://localhost:8443`
  - Health check: `curl -k https://localhost:8443/health`

4. **Manual API/Client Testing (optional):**
  After running the stack, you can interact with the ZeroMQ-based API using the `src/client_umrs_api.py` script.

  The client is ZeroMQ-only (pyzmq required) and uses the `--host` and `--port` flags to reach the UMRS ZeroMQ server (default port 8443).

  Example usage (same conventions as `docker/test_umrs_api.sh`):
  ```bash
  # Local variables
  PY_CLIENT="python3 src/client_umrs_api.py"
  HOST=localhost         # or the container name like 'umrs-server'
  PORT=8443
  CONFIG_PATH=src/conf_umrs_user1.json
  DATA_DIR=./data

  # Health check
  $PY_CLIENT --host "$HOST" --port $PORT health

  # List buckets for a user
  USERNAME=$(jq -r '.username' "$CONFIG_PATH")
  $PY_CLIENT --host "$HOST" --port $PORT list_buckets "$USERNAME"

  # Upload text to a bucket (positional args: text bucket_alias username metadata)
  BUCKET_ALIAS=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text") | .s3.bucket_alias' "$CONFIG_PATH")
  $PY_CLIENT --host "$HOST" --port $PORT upload_text "Hello from test script!" "$BUCKET_ALIAS" "$USERNAME" '{}'

  # Upload an image (positional args: file_path bucket_alias username metadata)
  IMG_PATH="$DATA_DIR/img/Cat_1.jpeg"
  BUCKET_ALIAS_IMG=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="image") | .s3.bucket_alias' "$CONFIG_PATH")
  $PY_CLIENT --host "$HOST" --port $PORT upload_image "$IMG_PATH" "$BUCKET_ALIAS_IMG" "$USERNAME" '{}'

  # Upload a text file
  TXTFILE_PATH="$DATA_DIR/texts/beast_in_the_jungle.txt"
  BUCKET_ALIAS_TXT=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text file") | .s3.bucket_alias' "$CONFIG_PATH")
  $PY_CLIENT --host "$HOST" --port $PORT upload_textfile "$TXTFILE_PATH" "$BUCKET_ALIAS_TXT" "$USERNAME" '{}'

  # Recompute all embeddings
  $PY_CLIENT --host "$HOST" --port $PORT recompute_embeddings

  # Update missing/mismatched embeddings
  $PY_CLIENT --host "$HOST" --port $PORT update_embeddings

  # Check consistency between S3 and database
  $PY_CLIENT --host "$HOST" --port $PORT check_consistency

  # Admin: Create and initialize a user
  $PY_CLIENT --host "$HOST" --port $PORT create_user "$CONFIG_PATH"

  # Admin: Create all buckets for a user
  $PY_CLIENT --host "$HOST" --port $PORT create_buckets "$CONFIG_PATH"

  # Admin: Add a user with hot reload
  USER2_CONFIG=src/conf_umrs_user2.json
  $PY_CLIENT --host "$HOST" --port $PORT add_user_hot_reload "$USER2_CONFIG"
  ```
  See `python3 src/client_umrs_api.py --help` for all available commands and options.

## Security


### Overview
UMRS is designed with security in mind for both development and production. Below are best practices and recommendations for securing your deployment.

**Note:** The `--insecure` flag is not required for API or client usage. It is only used internally for MinIO admin CLI operations when working with self-signed certificates. All API and client communication should use proper SSL verification.

#### 1. HTTPS & Certificates
- **Development**: Uses self-signed certificates (see `docker/ssl/`). Use the `--insecure` flag for local testing.
- **Production**: Always use CA-signed certificates. Remove the `--insecure` flag from all clients. Store certificates securely and restrict access to private keys.
- **Port**: HTTPS server runs on port 8443 (default). HTTP (8000) is for legacy/testing only—disable in production.

#### 2. Authentication & Authorization
- **API Access**: Protect all sensitive endpoints with authentication (e.g., tokens, API keys, or OAuth2).
- **User Isolation**: Each user has isolated buckets and credentials. Never share tokens or credentials between users.
- **Admin Operations**: Restrict admin endpoints to trusted users only. Use strong, unique admin tokens and rotate them regularly.

#### 3. Secrets & Credentials Management
- **Environment Variables**: Store secrets (DB passwords, S3 keys, tokens) in environment variables or Docker secrets, not in code or public repos.
- **.env Files**: If using `.env` files, ensure they are excluded from version control (`.gitignore`).
- **Rotation**: Rotate all secrets and credentials periodically, especially after personnel changes or suspected leaks.

#### 4. Network & Firewall
- **Internal-Only Services**: Expose only the HTTPS API port to the public. Keep DB, MinIO, and other services on internal/private networks.
- **Firewalls**: Use host and cloud firewalls to restrict access to only necessary ports and IPs.
- **Reverse Proxy**: For production, consider running UMRS behind a reverse proxy (e.g., NGINX) for additional security and rate limiting.

#### 5. Docker & Container Security
- **Least Privilege**: Run containers as non-root users whenever possible.
- **Image Hardening**: Use minimal, trusted base images. Regularly update images to patch vulnerabilities.
- **Volumes**: Restrict access to mounted volumes and secrets. Use Docker secrets for sensitive data.
- **Isolation**: Use separate Docker networks for internal communication.

#### 6. Logging, Monitoring & Auditing
- **Logs**: Enable and monitor logs for all services. Store logs securely and restrict access.
- **Audit**: Regularly audit access logs for suspicious activity.
- **Monitoring**: Use monitoring tools to detect anomalies and performance issues.

#### 7. Production Hardening Checklist
- Replace all self-signed certificates with CA-signed certificates.
- Remove all `--insecure` flags from API clients and scripts. (The flag is only used for MinIO admin CLI commands with self-signed certs.)
- Set strong, unique passwords and tokens for all users and services.
- Restrict admin endpoints and rotate admin credentials regularly.
- Disable or firewall all non-essential ports.
- Regularly update all dependencies and Docker images.
- Back up all critical data (PostgreSQL, MinIO) securely and test restores.

For more details, see the [OWASP Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html) and [FastAPI Security Best Practices](https://fastapi.tiangolo.com/advanced/security/).

## API Documentation

### API Endpoints

| Method | Path                        | Description                                      |
|--------|-----------------------------|--------------------------------------------------|
| GET    | /health                     | Health check endpoint                            |
| GET    | /status                     | Server status and configuration                  |
| GET    | /buckets                    | List buckets for a user or all                   |
| POST   | /upload/text                | Upload text data                                 |
| POST   | /upload/image               | Upload image file                                |
| POST   | /upload/textfile            | Upload text file                                 |
| POST   | /recompute                  | Recompute all embeddings                         |
| POST   | /update                     | Update missing/mismatched embeddings             |
| GET    | /check                      | Check S3/database consistency                    |
| POST   | /admin/add_user_hot_reload  | Add user and hot-reload config                   |
| POST   | /admin/create_user          | Create and initialize user                       |
| POST   | /admin/create_buckets       | Create all buckets for a user                    |

#### Example Usage with Python Client (ZeroMQ)

The recommended way to interact with the API is via the ZeroMQ Python CLI client at `src/client_umrs_api.py`.
This client uses ZeroMQ (REQ/REP JSON-RPC) and requires `pyzmq` to be installed. The client global flags are `--host`, `--port` and `--timeout` (ms). Below are typical commands — the global flags go before the command.

```bash
# Example setup (adjust host/port as needed)
PY_CLIENT="python3 src/client_umrs_api.py"
HOST=localhost
PORT=8443
CONFIG_PATH=src/conf_umrs_user1.json
DATA_DIR=./data

# Health check
$PY_CLIENT --host "$HOST" --port $PORT health

# List buckets for a user (username optional)
USERNAME=$(jq -r '.username' "$CONFIG_PATH")
$PY_CLIENT --host "$HOST" --port $PORT list_buckets "$USERNAME"

# Upload text to a bucket
BUCKET_ALIAS=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text") | .s3.bucket_alias' "$CONFIG_PATH")
$PY_CLIENT --host "$HOST" --port $PORT upload_text "Hello from test script!" "$BUCKET_ALIAS" "$USERNAME" '{}'

# Upload an image (the client reads and base64-encodes the file)
IMG_PATH="$DATA_DIR/img/Cat_1.jpeg"
BUCKET_ALIAS_IMG=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="image") | .s3.bucket_alias' "$CONFIG_PATH")
$PY_CLIENT --host "$HOST" --port $PORT upload_image "$IMG_PATH" "$BUCKET_ALIAS_IMG" "$USERNAME" '{}'

# Upload a text file (base64-encoded by the client)
TXTFILE_PATH="$DATA_DIR/texts/beast_in_the_jungle.txt"
BUCKET_ALIAS_TXT=$(jq -r '.experiences[0].modalities[] | select(.modality_name=="text file") | .s3.bucket_alias' "$CONFIG_PATH")
$PY_CLIENT --host "$HOST" --port $PORT upload_textfile "$TXTFILE_PATH" "$BUCKET_ALIAS_TXT" "$USERNAME" '{}'

# Recompute all embeddings
$PY_CLIENT --host "$HOST" --port $PORT recompute_embeddings

# Update missing/mismatched embeddings
$PY_CLIENT --host "$HOST" --port $PORT update_embeddings

# Check consistency between S3 and database
$PY_CLIENT --host "$HOST" --port $PORT check_consistency

# Admin: Create and initialize a user (pass a JSON config file path)
$PY_CLIENT --host "$HOST" --port $PORT create_user "$CONFIG_PATH"

# Admin: Create all buckets for a user
$PY_CLIENT --host "$HOST" --port $PORT create_buckets "$CONFIG_PATH"

# Admin: Add a user with hot reload
USER2_CONFIG=src/conf_umrs_user2.json
$PY_CLIENT --host "$HOST" --port $PORT add_user_hot_reload "$USER2_CONFIG"
```

Notes:
- The client encodes files in base64 before sending, so provide local file paths for `upload_image` and `upload_textfile`.
- `metadata` parameters (where shown) should be a JSON string (e.g. '{}') or omitted.

## Architecture

The project is run with Docker Compose (see `docker/docker-compose.yml`). The main services and how they are wired in the compose file are shown below.

```
Layer 1 (infrastructure):
┌──────────────┐  ┌──────────────┐
│  postgres    │  │    minio     │
│ (container)  │  │  (container) │
│   port 5432  │  │ ports 9000/9001│
└──────┬───────┘  └──────┬───────┘
       │                 │
       │                 │
       └──────┬──────────┘
              │
Layer 2 (application):
       ┌──────▼────────┐
       │  umrs-server  │
       │  (container)  │
       │ ports: 8000,  │
       │        8443   │
       └──────┬────────┘
              │
              │
Layer 3 (tests):
       ┌──────▼────────┐
       │ umrs-api-test │
       │  (test image) │
       │ depends_on:   │
       │   umrs-server │
       └───────────────┘

All services run on the same Docker network (`umrs-net`). In `docker/docker-compose.yml`:
- `umrs-server` declares `depends_on` for `postgres` and `minio` (compose uses healthchecks to control start order).
- `umrs-api-test` declares `depends_on` for `umrs-server` (the test container runs after the server is healthy).

Key ports exposed by compose (host:container):
- umrs-server: 8000 (HTTP, legacy), 8443 (HTTPS / ZeroMQ TCP port used by the test runner and clients)
- minio: 9000 (MinIO API - configured with SSL certs), 9001 (MinIO Console)
- postgres: 5432

Notes:
- The canonical client in the repo (`src/client_umrs_api.py`) connects to the UMRS server over TCP (default port 8443) using ZeroMQ JSON-RPC semantics. In compose the test runner sets UMRS_API_URL=https://umrs-server:8443 for compatibility with test scripts — the server listens on 8443 for its control plane.
- `umrs-server` communicates with MinIO and PostgreSQL over the internal Docker network; both MinIO and Postgres run in separate containers and are required before `umrs-server` reports healthy.
- The `umrs-api-test` container runs test suites (mounts configs and data) and is started only after `umrs-server` is healthy.

See `docker/docker-compose.yml` for the exact service names, env vars, volumes and healthcheck commands.
```


## Testing & Automation

- **Automated Multi-User Tests**: The main test workflow is triggered by `./docker/rebuild_and_start.sh`, which runs all integration and multi-user tests in a clean environment. `docker/test_umrs_api.sh` for details.
- **Test Output**: Results are printed to the console. The test containers will exit with code 0 on success.
- **Manual Testing**: You can use the client scripts in `src/` to interact with the running server for further manual or exploratory testing.

## Documentation

### Project Documentation

- **Multi-User Setup**: See `docker/README_multiuser.md` (if present) for details on configuring and running multi-user environments.
- **API Reference**: See the "API Documentation" section above for a full list of endpoints and usage examples.
- **Client Usage**: The recommended way to interact with the API is via the Python CLI client (`src/client_umrs_api.py`). See the Quick Start and API Documentation sections for command examples.
- **HTTPS & Certificates**: Self-signed certificates for development are in `docker/ssl/`. For production, use CA-signed certificates as described in the Security section.
- **Testing**: Automated multi-user and API tests are in `docker/test_multiuser.sh` and `docker/test_umrs_api.sh`. These scripts demonstrate end-to-end usage and can be used as templates for your own tests.
- **Development Notes**: Additional legacy or experimental documentation may be found in `Infinia_demo/doc/`.

For further help, see comments in the scripts and source files, or contact the maintainers.

## Supported Models

- **CLIP**: clip-ViT-B-32 for image embeddings
- **ResNet**: Custom ResNet50 for image processing  
- **E5**: intfloat/e5-base-v2 for text embeddings
- **MPNET**: all-mpnet-base-v2 for text processing

## Production Deployment

1. **Replace SSL Certificates**: Use CA-signed certificates instead of self-signed
2. **Remove --insecure Flag**: Ensure all API client calls use proper SSL verification. The `--insecure` flag is only relevant for MinIO admin CLI operations with self-signed certificates, not for API or client usage.
3. **Environment Variables**: Use proper secret management for credentials
4. **Network Security**: Configure proper firewall and network policies
5. **Monitoring**: Add monitoring and logging solutions
6. **Backup**: Implement backup strategies for PostgreSQL and MinIO data