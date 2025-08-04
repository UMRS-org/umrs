# UMRS Docker Script Consolidation

## Overview
The UMRS system has been updated to eliminate the need for separate shell scripts for user and infrastructure initialization. All operations are now handled through the UMRS API.

## Replaced Scripts

### 1. `init_minio_users.sh` → **REMOVED** ✅
**Previous functionality:**
- Created MinIO users using `mc admin user add`
- Created MinIO policies using `mc admin policy create` 
- Attached policies using `mc admin policy attach`

**Status:** **DELETED** - No longer needed as functionality moved to UMRS API

**New UMRS API approach:**
```bash
python3 /client/src/client_umrs.py init-user \
  --server-url http://umrs-server:8000 \
  --admin-token "$UMRS_ADMIN_TOKEN" \
  --username user1 \
  --password user1pass \
  --s3-secret user1secret
```

**Benefits:**
- Single command creates both MinIO and PostgreSQL users
- Consistent error handling and logging
- No need for separate MinIO client (mc) configuration
- Automatic policy creation and attachment

### 2. `init_postgres_users.sh` → **REMOVED** ✅
**Previous functionality:**
- Created PostgreSQL users using direct `psql` commands
- Required PostgreSQL admin credentials

**Status:** **DELETED** - No longer needed as functionality moved to UMRS API

**New UMRS API approach:**
Same command as above - the UMRS `init-user` API endpoint handles both MinIO and PostgreSQL user creation.

**Benefits:**
- Unified user creation process
- No need for direct PostgreSQL access
- Consistent with UMRS architecture

### 3. `test_multiuser.sh` → **ENHANCED**
**Changes made:**
- Added `create_users_via_umrs()` function to replace shell script calls
- Centralized user creation before running tests
- Removed redundant user creation from individual test functions
- Added comprehensive documentation

## API Endpoints Used

### `/api/admin/create_user` (POST)
**Purpose:** Creates users in both MinIO and PostgreSQL
**Parameters:**
- `username`: Username for both services
- `password`: PostgreSQL password  
- `s3_secret`: MinIO secret key

**Implementation:**
- Calls `umrs.minio.minio_s3_create_user()`
- Calls `umrs.sql.create_db_user()`
- Returns unified success/error responses

### Other Endpoints Already in Use:
- `/api/user/create_bucket` - Bucket creation
- `/api/user/list_buckets` - List user buckets
- `/api/user/info` - User information
- `/health` - Server health check

## Files Modified

### `docker/test_multiuser.sh`
- **Added:** `create_users_via_umrs()` function
- **Modified:** `main()` function to call centralized user creation
- **Modified:** `run_umrs_test()` function to remove duplicate user creation
- **Enhanced:** Documentation and comments

### `requirements-client.txt`
- **Added:** `requests` module for API client functionality

### `src/client_umrs.py`
- **Enhanced:** Combined local message queue and remote API functionality
- **Added:** API client commands (`init-user`, `create-bucket`, `list-buckets`, etc.)

## Benefits of Consolidation

1. **Simplified Deployment:** No need to manage separate initialization scripts
2. **Unified Error Handling:** All operations go through UMRS logging and error handling
3. **Consistent API:** All operations use the same authentication and communication patterns
4. **Reduced Dependencies:** No need for direct MinIO client or PostgreSQL access
5. **Better Testing:** Centralized user creation makes test scenarios more predictable
6. **Maintainability:** Single codebase for all user management operations

## Still Required

### `init_pgvector.sql`
**Purpose:** Creates PostgreSQL extensions and initial database setup
**Reason to keep:** UMRS doesn't currently handle database extension management
**Future enhancement:** Could be integrated into UMRS `create_db()` function

## Migration Guide

### For Existing Deployments
1. Ensure UMRS server is running with web API enabled (`--web` flag)
2. Set `UMRS_ADMIN_TOKEN` environment variable
3. Replace calls to `init_minio_users.sh` and `init_postgres_users.sh` with:
   ```bash
   python3 client_umrs.py init-user --server-url $UMRS_URL --admin-token $ADMIN_TOKEN --username $USER --password $PASS --s3-secret $SECRET
   ```
4. Update any automation scripts to use the new consolidated approach

### Testing the Changes
Run the updated test script:
```bash
cd docker
./test_multiuser.sh
```

The script will now:
1. Create all users via UMRS API in one step
2. Run individual user tests with bucket creation and data upload
3. Verify API functionality with list-buckets calls

## Future Enhancements

1. **Database Extension Management:** Integrate pgvector setup into UMRS
2. **Bulk Operations:** Add endpoints for creating multiple users at once
3. **User Management:** Add endpoints for user deletion, password changes, etc.
4. **Monitoring:** Add more detailed health checks and user status endpoints
