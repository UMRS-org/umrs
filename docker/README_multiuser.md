# UMRS Multi-User Setup

This document describes the multi-user setup for the UMRS (Unified Multi-modal Retrieval System) project.

## Overview

The multi-user setup allows multiple users to:
- Have separate S3 buckets and credentials
- Process texts and images independently
- Store embeddings in separate database tables
- Use different AI models for different experiences

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User 1        │    │   User 2        │    │   Admin         │
│   Container     │    │   Container     │    │   Container     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ - user1 env     │    │ - user2 env     │    │ - UMRS Server   │
│ - s3cfg_user1   │    │ - s3cfg_user2   │    │ - PostgreSQL    │
│ - config_user1  │    │ - config_user2  │    │ - MinIO         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## File Structure

```
docker/
├── conf_umrs_user1.json          # User 1 configuration
├── conf_umrs_user2.json          # User 2 configuration
├── s3cfg_user1                   # User 1 S3 credentials
├── s3cfg_user2                   # User 2 S3 credentials
├── test_multiuser.sh             # Multi-user test script
├── test_user_client.py           # Individual user test script
├── multiuser_helper.sh           # Helper script for management
├── Dockerfile.umrs-client-tester # Client tester container
└── docker-compose.yml            # Updated with client tester
```

## Configuration Files

### User Configuration (conf_umrs_user*.json)

Each user has their own configuration file with:
- Unique username and credentials
- Separate S3 bucket names
- Different AI models for different experiences
- Separate database tables

### S3 Configuration (s3cfg_user*)

Each user has separate S3 credentials:
- `s3cfg_user1`: Access keys for user1
- `s3cfg_user2`: Access keys for user2

## Quick Start

1. **Setup Environment**
   ```bash
   cd docker
   ./multiuser_helper.sh setup
   ```

2. **Run Multi-User Test**
   ```bash
   ./multiuser_helper.sh test
   ```

3. **Interactive Testing**
   ```bash
   # Open shell in client tester
   ./multiuser_helper.sh shell
   
   # Switch to specific user
   ./multiuser_helper.sh user1
   ./multiuser_helper.sh user2
   ```

## Manual Testing

### Test Individual User

```bash
# Test user1
docker-compose exec umrs-client-tester python3 /client/test_user_client.py --user user1

# Test user2
docker-compose exec umrs-client-tester python3 /client/test_user_client.py --user user2
```

### Test S3 Connection

```bash
# As user1
docker-compose exec umrs-client-tester su - user1 -c "s3cmd -c ~/.s3cfg ls"

# As user2
docker-compose exec umrs-client-tester su - user2 -c "s3cmd -c ~/.s3cfg ls"
```

## FastAPI Web Server for UMRS

UMRS now supports web access via FastAPI, allowing client containers and external services to interact with the server over HTTP.

### How to Run the UMRS Server with FastAPI

To start the UMRS server with web access, use the `--web` flag:

```bash
docker-compose run --rm umrs-server python src/server_umrs.py --conf conf_umrs_init.json --web
```

This will launch a FastAPI server on port 8000.

### Available Endpoints

- **Health Check:**  
  `GET /health`  
  Returns `{ "status": "ok" }` if the server is running.

- **Status:**  
  `GET /status`  
  Returns server status and last event timestamp.

### Example: Accessing the API from umrs-client-tester

```bash
docker exec umrs-client-tester curl -s umrs-server:8000/health
docker exec umrs-client-tester curl -s umrs-server:8000/status
```

### Notes

- All containers must be on the same Docker network (e.g., `umrs-net`) for service discovery.
- You can extend the FastAPI server to expose more endpoints for embedding, retrieval, or management.

### Create Buckets

```bash
# For user1
docker-compose exec umrs-client-tester su - user1 -c "s3cmd -c ~/.s3cfg mb s3://user1-bucket-1"

# For user2
docker-compose exec umrs-client-tester su - user2 -c "s3cmd -c ~/.s3cfg mb s3://user2-bucket-1"
```

## User Buckets

Each user has 5 buckets for different purposes:

**User 1:**
- `user1-bucket-1`: CLIP images (Text and Images experience)
- `user1-bucket-2`: MPNET text (Text and Images experience)
- `user1-bucket-3`: ResNet images (Images and Texts experience)
- `user1-bucket-4`: E5 text (Images and Texts experience)
- `user1-bucket-5`: E5 text files (Images and Texts experience)

**User 2:**
- `user2-bucket-1`: CLIP images (Text and Images experience)
- `user2-bucket-2`: MPNET text (Text and Images experience)
- `user2-bucket-3`: ResNet images (Images and Texts experience)
- `user2-bucket-4`: E5 text (Images and Texts experience)
- `user2-bucket-5`: E5 text files (Images and Texts experience)

## Troubleshooting

### Common Issues

1. **S3 Connection Failed**
   - Check if MinIO is running: `docker-compose ps`
   - Verify S3 credentials in `s3cfg_user*` files
   - Check network connectivity between containers

2. **Bucket Creation Failed**
   - Ensure MinIO is accessible
   - Check bucket naming conventions
   - Verify user permissions

3. **UMRS Server Connection Failed**
   - Check if UMRS server is running
   - Verify server health: `curl http://umrs-server:8000/health`
   - Check container logs: `docker-compose logs umrs-server`

### Debug Commands

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f umrs-server
docker-compose logs -f minio
docker-compose logs -f postgres

# Test network connectivity
docker-compose exec umrs-client-tester ping umrs-server
docker-compose exec umrs-client-tester ping minio
docker-compose exec umrs-client-tester ping postgres
```

## Security Notes

- In production, use proper secret management
- Implement proper access controls for MinIO
- Use environment variables for sensitive data
- Consider using proper authentication mechanisms

## Next Steps

1. Implement proper MinIO user management
2. Add authentication to UMRS server
3. Implement proper error handling
4. Add monitoring and logging
5. Create production deployment scripts
