# UMRS Multi-User Setup with HTTPS

This document describes the secure multi-user setup for the UMRS (Unified Multi-modal Retrieval System) project with HTTPS encryption.

## Overview

The multi-user setup allows multiple users to:
- Have separate S3 buckets with application-level access control
- Process texts and images independently through HTTPS APIs
- Store embeddings in separate database tables
- Use different AI models for different experiences
- Communicate securely through SSL/TLS encryption

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
                         (Self-signed certificates)
```

## File Structure

```
docker/
├── conf_umrs_user1.json          # User 1 configuration
├── conf_umrs_user2.json          # User 2 configuration
├── s3cfg                         # Admin S3 credentials (minioadmin/minioadmin)
├── test_multiuser.sh             # Multi-user HTTPS test script (automated)
├── Dockerfile.umrs-client-tester # Client tester container
├── docker-compose.yml            # Complete multi-user HTTPS setup
├── ssl/                          # SSL certificates for HTTPS
│   ├── server.crt               # SSL certificate
│   ├── server.key               # SSL private key
│   └── server.pem               # SSL certificate in PEM format
└── data/                         # Test data for images and texts
```

## HTTPS Configuration

### SSL Certificates

The system uses self-signed SSL certificates for development:
- **Location**: `docker/ssl/` directory
- **Files**: `server.crt`, `server.key`, `server.pem`
- **Usage**: UMRS server runs with HTTPS on port 8443
- **Client Flag**: Use `--insecure` flag to accept self-signed certificates

### Production Considerations

For production deployment:
1. Replace self-signed certificates with CA-signed certificates
2. Remove the `--insecure` flag from all client commands
3. Update firewall rules for port 8443 (HTTPS)
4. Configure proper SSL certificate management

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

1. **Build and Start All Services with HTTPS**
   ```bash
   cd docker
   docker compose up --build -d
   ```

2. **Run Multi-User HTTPS Test (Automated)**
   ```bash
   # The test runs automatically when containers start
   # Check test results:
   docker logs umrs-client-tester
   
   # Or run manually:
   docker compose up --build
   ```

3. **Access HTTPS Services**
   ```bash
   # Health check
   curl -k https://localhost:8443/health
   
   # Server status
   curl -k https://localhost:8443/status
   ```

4. **Interactive Testing**
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

# Test as user1 with HTTPS
su - user1
python3 /home/user1/src/client_umrs.py list-buckets \
  --server-url https://umrs-server:8443 \
  --token user1 --username user1 --insecure

# Test as user2 with HTTPS
su - user2
python3 /home/user2/src/client_umrs.py list-buckets \
  --server-url https://umrs-server:8443 \
  --token user2 --username user2 --insecure
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

# Test via UMRS HTTPS API (preferred method for users)
docker exec -it umrs-client-tester curl -k -H "Authorization: Bearer user1" \
  "https://umrs-server:8443/api/user/list_buckets?username=user1"
docker exec -it umrs-client-tester curl -k -H "Authorization: Bearer user2" \
  "https://umrs-server:8443/api/user/list_buckets?username=user2"
```

### Create Additional Buckets

```bash
# Using UMRS client with HTTPS (recommended)
docker exec -it umrs-client-tester su - user1 -c "
python3 /home/user1/src/client_umrs.py create-bucket \
  --server-url https://umrs-server:8443 \
  --token user1 \
  --username user1 \
  --bucket-alias user1-custom-bucket \
  --xp-name 'Custom Experience' \
  --modality text \
  --model 'all-mpnet-base-v2' \
  --insecure
"

# Using MinIO client directly
docker exec -it umrs-client-tester mc mb myminio/user1-direct-bucket
```

## FastAPI Web Server for UMRS with HTTPS

UMRS provides a comprehensive FastAPI web interface that supports multi-user operations with HTTPS encryption, including user management, bucket operations, and file uploads.

### How to Run the UMRS Server with FastAPI and HTTPS

The UMRS server runs automatically with HTTPS on the multi-user setup:

```bash
docker compose up --build -d
```

The FastAPI server will be available on port 8443 (HTTPS) with health monitoring.

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
- All communication encrypted with HTTPS/TLS

### Example HTTPS API Usage

```bash
# Health check with HTTPS
curl -k https://localhost:8443/health

# Create user (admin operation) with HTTPS
curl -k -X POST https://localhost:8443/api/admin/create_user \
  -H "Authorization: Bearer test_admin_token" \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "password": "password", "s3_secret": "secret"}'

# List user buckets with HTTPS
curl -k -H "Authorization: Bearer user1" \
  "https://localhost:8443/api/user/list_buckets?username=user1"

# Upload file with HTTPS
curl -k -X POST https://localhost:8443/api/user/upload_file \
  -H "Authorization: Bearer user1" \
  -F "file=@/path/to/file.txt" \
  -F "username=user1" \
  -F "bucket_alias=user1-bucket-1" \
  -F "xp_name=Test Experience" \
  -F "modality=text"
```

### Create Additional Buckets

```bash
# Using UMRS client with HTTPS (recommended)
docker exec -it umrs-client-tester su - user1 -c "
python3 /home/user1/src/client_umrs.py create-bucket \
  --server-url https://umrs-server:8443 \
  --token user1 \
  --username user1 \
  --bucket-alias user1-custom-bucket \
  --xp-name 'Custom Experience' \
  --modality text \
  --model 'all-mpnet-base-v2' \
  --insecure
"

# Using MinIO client directly
docker exec -it umrs-client-tester mc mb myminio/user1-direct-bucket
```

## Automated Testing Features

The enhanced test script (`test_multiuser.sh`) provides comprehensive HTTPS testing:

### Test Capabilities

1. **User Creation**: Uses UMRS HTTPS API to create users with proper S3 and PostgreSQL setup
2. **Bucket Management**: Creates buckets using both manual and config-based approaches via HTTPS
3. **File Upload**: Tests file upload for different modalities (text, images) through HTTPS
4. **Bucket Contents Inspection**: Uses MinIO client to show actual bucket contents
5. **Bucket Deletion**: Tests bucket deletion with confirmation via HTTPS API
6. **API Validation**: Verifies all operations through UMRS FastAPI HTTPS endpoints
7. **SSL Certificate Validation**: Tests HTTPS connectivity with self-signed certificates

### Test Flow

```bash
# Automatic execution when containers start
docker compose up --build

# Manual execution
docker exec umrs-client-tester /client/test_multiuser.sh
```

### Test Output Example

```
=== UMRS Multi-User Test Script (HTTPS) ===
✅ MinIO is ready!
✅ PostgreSQL is ready!  
✅ UMRS Server is ready!

Creating users via UMRS client command...
✅ User user1 created successfully
✅ User user2 created successfully

=== Testing User 1 ===
✅ Bucket 'user1-bucket-5' created successfully
✅ File 'beast_in_the_jungle.txt' uploaded successfully
✅ Bucket 'user1-bucket-1' created successfully
✅ File 'cat.jpeg' uploaded successfully

=== Checking Bucket Contents and Testing Deletion ===
--- Contents of user1.images-and-texts.user1-bucket-5 ---
[2025-08-07 15:30:24 UTC] 101KiB STANDARD tmp3dn7rzuo.txt

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

# Test UMRS HTTPS API
docker exec umrs-client-tester curl -k -s https://umrs-server:8443/health
docker exec umrs-client-tester curl -k -s https://umrs-server:8443/status

# Test SSL certificate
docker exec umrs-client-tester openssl s_client -connect umrs-server:8443 -servername umrs-server

# Check test script execution
docker exec umrs-client-tester cat /client/test_multiuser.sh
docker exec umrs-client-tester /client/test_multiuser.sh

# View bucket contents directly
docker exec umrs-client-tester mc ls myminio --recursive
```

## Security Notes

- **HTTPS Encryption**: All client-server communication encrypted with TLS
- **Self-signed Certificates**: Development uses self-signed certificates with `--insecure` flag
- **Production Deployment**: Use proper SSL certificates and remove `--insecure` flag for production
- **MinIO Security**: Implement proper access controls and policies
- **Authentication**: Current implementation uses token-based auth with HTTPS encryption
- **Network Security**: Containers communicate via internal Docker network with HTTPS overlay
- **Credential Management**: Single admin credential set for all S3 operations with user authentication handled at application level
- **API Security**: FastAPI endpoints include basic authentication validation over HTTPS
- **Certificate Management**: Store production certificates securely and implement proper rotation policies

## Implementation Notes

### Recent Improvements

1. **HTTPS Implementation**: Added complete SSL/TLS encryption with self-signed certificates
2. **Fixed S3 SignatureDoesNotMatch**: Modified `list_buckets` API to use admin credentials
3. **Enhanced HTTPS Testing**: Added comprehensive bucket content inspection and deletion testing via HTTPS
4. **Docker Network Integration**: Test script works within container network using HTTPS service names
5. **Automated User Creation**: Users are created via UMRS HTTPS API instead of manual scripts
6. **MinIO Client Integration**: Direct MinIO client usage for bucket management and inspection
7. **SSL Certificate Management**: Self-signed certificates for development, production-ready certificate support

### Architecture Decisions

- **HTTPS-First Design**: All API communication encrypted by default
- **Admin Credentials Only**: Admin credentials exclusively for all S3 operations ensuring reliability and consistency
- **FastAPI HTTPS Integration**: Comprehensive REST API with SSL/TLS encryption for all UMRS operations
- **Container Orchestration**: Single `docker compose up` command for complete HTTPS setup
- **Automated HTTPS Testing**: Self-contained test execution with detailed HTTPS connectivity reporting
- **Development SSL**: Self-signed certificates with `--insecure` flag for development simplicity

## Next Steps

### Immediate Improvements
1. ✅ **Fixed S3 credential issues** - Admin credentials for bucket operations
2. ✅ **Enhanced test automation** - Comprehensive bucket testing with content inspection
3. ✅ **Improved Docker integration** - Internal network communication

### Future Enhancements
1. **Production SSL Certificates** - Integration with Let's Encrypt or CA-signed certificates
2. **Advanced Authentication** - JWT tokens, OAuth integration with HTTPS
3. **Production MinIO Setup** - Proper user policies and access controls with HTTPS endpoints
4. **Enhanced Error Handling** - Better HTTPS error messages and SSL troubleshooting
5. **Monitoring and Logging** - Prometheus metrics with HTTPS endpoints, structured SSL logging
6. **Kubernetes Deployment** - Production-ready orchestration with SSL certificate management
7. **Multi-tenancy** - Proper resource isolation and quotas with HTTPS APIs
8. **API Rate Limiting** - Prevent abuse and ensure fair usage on HTTPS endpoints
9. **Backup and Recovery** - Automated backup strategies for data protection with encrypted transfers
10. **SSL Certificate Automation** - Automatic certificate renewal and management
