# UMRS Multi-User Setup

This document describes the multi-user setup for the UMRS (Unified Multi-modal Retrieval System) project.

## Overview

The multi-user setup allows multiple users to:
- Have separate S3 buckets with application-level access control
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
│ - config_user1  │    │ - config_user2  │    │ - PostgreSQL    │
│                 │    │                 │    │ - MinIO         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## File Structure

```
docker/
├── conf_umrs_user1.json          # User 1 configuration
├── conf_umrs_user2.json          # User 2 configuration
├── s3cfg                         # Admin S3 credentials (minioadmin/minioadmin)
├── test_multiuser.sh             # Multi-user test script (automated)
├── test_multiuser_fixed.sh       # Fixed version of test script
├── Dockerfile.umrs-client-tester # Client tester container
├── docker-compose.yml            # Complete multi-user setup
└── data/                         # Test data for images and texts
```

## Configuration Files

### User Configuration (conf_umrs_user*.json)

Each user has their own configuration file with:
- Unique username and credentials
- Separate S3 bucket names
- Different AI models for different experiences
- Separate database tables

### S3 Configuration (s3cfg)

**Admin Configuration (s3cfg):**
- Contains admin credentials (minioadmin/minioadmin)
- Used by UMRS server for all bucket management operations
- Provides reliable access for all S3 operations including list_buckets API

**Implementation Note:** The system now uses admin credentials exclusively for all S3 operations to ensure reliability and avoid signature mismatch errors. User-specific S3 configurations are no longer needed as the UMRS server handles all S3 access internally.

## Quick Start

1. **Build and Start All Services**
   ```bash
   cd docker
   docker compose up --build -d
   ```

2. **Run Multi-User Test (Automated)**
   ```bash
   # The test runs automatically when containers start
   # Check test results:
   docker logs umrs-client-tester
   
   # Or run manually:
   docker compose up --build
   ```

3. **Interactive Testing**
   ```bash
   # Open shell in client tester
   docker exec -it umrs-client-tester bash
   
   # Switch to specific user
   su - user1
   su - user2
   ```

## Manual Testing

### Test Complete Multi-User System

```bash
# Run the automated test suite
docker compose up --build

# View test results
docker logs umrs-client-tester

# Check specific test sections
docker logs umrs-client-tester | grep -E "(===|✅|✗|Contents of)"
```

### Test Individual User

```bash
# Enter client tester container
docker exec -it umrs-client-tester bash

# Test as user1
su - user1
python3 /home/user1/src/client_umrs.py list-buckets --server-url http://umrs-server:8000 --token user1 --username user1

# Test as user2
su - user2
python3 /home/user2/src/client_umrs.py list-buckets --server-url http://umrs-server:8000 --token user2 --username user2
```

### Test Bucket Contents and Deletion

```bash
# The test script automatically:
# 1. Creates users via UMRS API
# 2. Creates buckets for each user
# 3. Uploads sample files
# 4. Lists bucket contents using MinIO client
# 5. Tests bucket deletion functionality
# 6. Verifies deletion success
```

### Test S3 Connection

```bash
# Check MinIO server status
docker exec -it umrs-client-tester mc --help

# Test admin access (recommended method)
docker exec -it umrs-client-tester mc alias set myminio http://minio:9000 minioadmin minioadmin
docker exec -it umrs-client-tester mc ls myminio

# Test via UMRS API (preferred method for users)
docker exec -it umrs-client-tester curl -H "Authorization: Bearer user1" "http://umrs-server:8000/api/user/list_buckets?username=user1"
docker exec -it umrs-client-tester curl -H "Authorization: Bearer user2" "http://umrs-server:8000/api/user/list_buckets?username=user2"
```

### Create Additional Buckets

```bash
# Using UMRS client (recommended)
docker exec -it umrs-client-tester su - user1 -c "
python3 /home/user1/src/client_umrs.py create-bucket \
  --server-url http://umrs-server:8000 \
  --token user1 \
  --username user1 \
  --bucket-alias user1-custom-bucket \
  --xp-name 'Custom Experience' \
  --modality text \
  --model 'all-mpnet-base-v2'
"

# Using MinIO client directly
docker exec -it umrs-client-tester mc mb myminio/user1-direct-bucket
```

## FastAPI Web Server for UMRS

UMRS provides a comprehensive FastAPI web interface that supports multi-user operations, including user management, bucket operations, and file uploads.

### How to Run the UMRS Server with FastAPI

The UMRS server runs automatically with the multi-user setup:

```bash
docker compose up --build -d
```

The FastAPI server will be available on port 8000 with health monitoring.

### Available Endpoints

**Health and Status:**
- `GET /health` - Returns `{ "status": "ok" }`
- `GET /status` - Returns server status and last event timestamp

**Admin Operations:**
- `POST /api/admin/create_user` - Create new users with S3 and PostgreSQL setup
- Requires admin token authentication

**User Operations:**
- `POST /api/user/create_bucket` - Create new bucket for user
- `POST /api/user/upload_file` - Upload files to user buckets
- `GET /api/user/list_buckets` - List user's buckets
- `DELETE /api/user/delete_bucket` - Delete bucket and all contents

**Authentication:**
- Admin operations require `UMRS_ADMIN_TOKEN`
- User operations require user-specific tokens

### Example API Usage

```bash
# Health check
curl http://localhost:8000/health

# Create user (admin operation)
curl -X POST http://localhost:8000/api/admin/create_user \
  -H "Authorization: Bearer test_admin_token" \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "password", "s3_secret": "secret"}'

# List user buckets
curl -H "Authorization: Bearer user1" \
  "http://localhost:8000/api/user/list_buckets?username=user1"

# Upload file
curl -X POST http://localhost:8000/api/user/upload_file \
  -H "Authorization: Bearer user1" \
  -F "file=@/path/to/file.txt" \
  -F "username=user1" \
  -F "bucket_alias=user1-bucket-1" \
  -F "xp_name=Test Experience" \
  -F "modality=text"
```

### Create Additional Buckets

```bash
# Using UMRS client (recommended)
docker exec -it umrs-client-tester su - user1 -c "
python3 /home/user1/src/client_umrs.py create-bucket \
  --server-url http://umrs-server:8000 \
  --token user1 \
  --username user1 \
  --bucket-alias user1-custom-bucket \
  --xp-name 'Custom Experience' \
  --modality text \
  --model 'all-mpnet-base-v2'
"

# Using MinIO client directly
docker exec -it umrs-client-tester mc mb myminio/user1-direct-bucket
```

## Automated Testing Features

The enhanced test script (`test_multiuser.sh`) provides comprehensive testing:

### Test Capabilities

1. **User Creation**: Uses UMRS API to create users with proper S3 and PostgreSQL setup
2. **Bucket Management**: Creates buckets using both manual and config-based approaches
3. **File Upload**: Tests file upload for different modalities (text, images)
4. **Bucket Contents Inspection**: Uses MinIO client to show actual bucket contents
5. **Bucket Deletion**: Tests bucket deletion with confirmation
6. **API Validation**: Verifies all operations through UMRS FastAPI endpoints

### Test Flow

```bash
# Automatic execution when containers start
docker compose up --build

# Manual execution
docker exec umrs-client-tester /client/test_multiuser.sh
```

### Test Output Example

```
=== UMRS Multi-User Test Script ===
✅ MinIO is ready!
✅ PostgreSQL is ready!
✅ UMRS Server is ready!

Creating users via UMRS client command...
✅ User user1 created successfully
✅ User user2 created successfully

=== Testing User 1 ===
✅ Bucket 'user1-bucket-5' created successfully
✅ Bucket 'user1-bucket-2' created successfully
✅ Bucket 'user1-bucket-1' created successfully

=== Checking Bucket Contents and Testing Deletion ===
--- Contents of user1.images-and-texts.user1-bucket-5 ---
(empty or inaccessible)

✅ Bucket 'user1.images-and-texts.user1-bucket-5' deleted successfully
=== Multi-user test completed! ===
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

1. **S3 Connection Failed / SignatureDoesNotMatch**
   - **Solution**: The system uses admin credentials exclusively for all S3 operations
   - Check if MinIO is running: `docker ps | grep minio`
   - Verify admin credentials are configured (minioadmin/minioadmin)
   - Test connection: `docker exec umrs-client-tester mc ls myminio`

2. **Bucket Creation Failed**
   - Ensure UMRS server is accessible: `curl http://localhost:8000/health`
   - Check user authentication tokens
   - Verify bucket naming follows conventions (username.experience.bucket-alias)

3. **UMRS Server Connection Failed**
   - Check if UMRS server is healthy: `docker logs umrs-server`
   - Verify FastAPI server is running on port 8000
   - Test API: `curl http://localhost:8000/health`

4. **File Upload Issues**
   - Verify file exists and is accessible
   - Check file size limits and permissions
   - Ensure correct modality specification (text, image, text file)

5. **Test Script Issues**
   - Container uses internal Docker network (minio:9000, umrs-server:8000)
   - MinIO client (mc) is pre-installed in client tester container
   - Test script runs automatically on container start

### Debug Commands

```bash
# Check all services
docker ps

# View logs
docker logs umrs-server
docker logs minio
docker logs postgres
docker logs umrs-client-tester

# Test network connectivity
docker exec umrs-client-tester ping umrs-server
docker exec umrs-client-tester ping minio
docker exec umrs-client-tester ping postgres

# Test MinIO client
docker exec umrs-client-tester mc alias set myminio http://minio:9000 minioadmin minioadmin
docker exec umrs-client-tester mc ls myminio

# Test UMRS API
docker exec umrs-client-tester curl -s http://umrs-server:8000/health
docker exec umrs-client-tester curl -s http://umrs-server:8000/status

# Check test script execution
docker exec umrs-client-tester cat /client/test_multiuser.sh
docker exec umrs-client-tester /client/test_multiuser.sh

# View bucket contents directly
docker exec umrs-client-tester mc ls myminio --recursive
```

## Security Notes

- **Production Deployment**: Use proper secret management (Docker secrets, environment variables)
- **MinIO Security**: Implement proper access controls and policies
- **Authentication**: Current implementation uses simple token-based auth for testing
- **Network Security**: Containers communicate via internal Docker network
- **Credential Management**: Single admin credential set for all S3 operations with user authentication handled at application level
- **API Security**: FastAPI endpoints include basic authentication validation

## Implementation Notes

### Recent Improvements

1. **Fixed S3 SignatureDoesNotMatch**: Modified `list_buckets` API to use admin credentials
2. **Enhanced Testing**: Added comprehensive bucket content inspection and deletion testing
3. **Docker Network Integration**: Test script works within container network using service names
4. **Automated User Creation**: Users are created via UMRS API instead of manual scripts
5. **MinIO Client Integration**: Direct MinIO client usage for bucket management and inspection

### Architecture Decisions

- **Admin Credentials Only**: Admin credentials exclusively for all S3 operations ensuring reliability and consistency
- **FastAPI Integration**: Comprehensive REST API for all UMRS operations
- **Container Orchestration**: Single `docker compose up` command for complete setup
- **Automated Testing**: Self-contained test execution with detailed reporting

## Next Steps

### Immediate Improvements
1. ✅ **Fixed S3 credential issues** - Admin credentials for bucket operations
2. ✅ **Enhanced test automation** - Comprehensive bucket testing with content inspection
3. ✅ **Improved Docker integration** - Internal network communication

### Future Enhancements
1. **Advanced Authentication** - JWT tokens, OAuth integration
2. **Production MinIO Setup** - Proper user policies and access controls  
3. **Enhanced Error Handling** - Better error messages and recovery
4. **Monitoring and Logging** - Prometheus metrics, structured logging
5. **Kubernetes Deployment** - Production-ready orchestration
6. **Multi-tenancy** - Proper resource isolation and quotas
7. **API Rate Limiting** - Prevent abuse and ensure fair usage
8. **Backup and Recovery** - Automated backup strategies for data protection
