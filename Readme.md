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

### Multi-User HTTPS Setup (Recommended)

1. **Navigate to the Docker directory:**
   ```bash
   cd docker
   ```

2. **Start all services with HTTPS:**
   ```bash
   docker compose up --build
   ```

3. **Access the system:**
   - UMRS Server: `https://localhost:8443`
   - Health check: `curl -k https://localhost:8443/health`
   - Multi-user test runs automatically on startup

4. **Test with users:**
   ```bash
   # Test as user1
   python3 src/client_umrs.py list-buckets \
     --server-url https://localhost:8443 \
     --token user1 --username user1 --insecure

   # Test as user2  
   python3 src/client_umrs.py list-buckets \
     --server-url https://localhost:8443 \
     --token user2 --username user2 --insecure
   ```

## Security

### HTTPS Implementation

- **Development**: Uses self-signed certificates with `--insecure` flag
- **Production**: Replace with CA-signed certificates and remove `--insecure`
- **Encryption**: All client-server communication encrypted with TLS
- **Port**: HTTPS server runs on port 8443 (HTTP legacy on 8000)

### Certificates

- Location: `docker/ssl/` directory
- Files: `server.crt`, `server.key`, `server.pem`
- Self-signed for development, replace for production

## API Documentation

### Health and Status
- `GET /health` - Returns `{"status": "ok"}`
- `GET /status` - Returns server status and last event timestamp

### Admin Operations (Requires admin token)
- `POST /api/admin/create_user` - Create new users with S3 and PostgreSQL setup

### User Operations (Requires user token)
- `POST /api/user/create_bucket` - Create new bucket for user
- `POST /api/user/upload_file` - Upload files to user buckets  
- `GET /api/user/list_buckets` - List user's buckets
- `DELETE /api/user/delete_bucket` - Delete bucket and all contents

### Example API Usage

```bash
# Health check
curl -k https://localhost:8443/health

# Create user (admin operation)
curl -k -X POST https://localhost:8443/api/admin/create_user \
  -H "Authorization: Bearer test_admin_token" \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "password", "s3_secret": "secret"}'

# List user buckets
curl -k -H "Authorization: Bearer user1" \
  "https://localhost:8443/api/user/list_buckets?username=user1"
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User 1        │    │   User 2        │    │   Admin         │
│   Container     │    │   Container     │    │   Container     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ - user1 env     │    │ - user2 env     │    │ - UMRS Server   │
│ - config_user1  │    │ - config_user2  │    │   (HTTPS:8443)  │
│                 │    │                 │    │ - PostgreSQL    │
│                 │    │                 │    │ - MinIO         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                            HTTPS/TLS Encrypted
```

## Documentation

- **Multi-User Setup**: See `docker/README_multiuser.md`
- **HTTPS Configuration**: Self-signed certificates in `docker/ssl/`
- **Testing**: Automated multi-user tests in `docker/test_multiuser.sh`
- **Development**: Legacy documentation in `Infinia_demo/doc/`

## Supported Models

- **CLIP**: clip-ViT-B-32 for image embeddings
- **ResNet**: Custom ResNet50 for image processing  
- **E5**: intfloat/e5-base-v2 for text embeddings
- **MPNET**: all-mpnet-base-v2 for text processing

## Production Deployment

1. **Replace SSL Certificates**: Use CA-signed certificates instead of self-signed
2. **Remove --insecure Flag**: Update all client calls to use proper SSL verification
3. **Environment Variables**: Use proper secret management for credentials
4. **Network Security**: Configure proper firewall and network policies
5. **Monitoring**: Add monitoring and logging solutions
6. **Backup**: Implement backup strategies for PostgreSQL and MinIO data