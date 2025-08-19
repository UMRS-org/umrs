---
applyTo: '**'
---

# User Memory

## User Preferences
- Programming languages: Python
- Code style preferences: Follows project conventions, modular, class-based for API
- Development environment: Linux, VS Code, Docker
- Communication style: Concise, thorough explanations

## Project Context
- Current project type: Multi-modal retrieval system (UMRS) with automated Docker-based API testing
- Tech stack: Python, FastAPI, Uvicorn, S3-compatible storage, PostgreSQL (likely with pgvector), PyTorch, Docker
- Architecture patterns: Modular, class-based API, multi-user/multi-bucket, RESTful, config-driven tests
- Key requirements: Multi-user, multi-bucket, multi-modality support; embedding computation; robust API; containerized deployment; automated API endpoint testing for multiple users, config-driven, data directory usage
- All admin and user API endpoint tests are performed via the Python client (client_umrs_api.py) for consistency and reliability. No curl or CLI wrappers are used in automation.
- Both Postgres and S3 databases are controlled by admin; no user-specific credentials are necessary. All database and S3 access should use admin credentials only.
- API mode now uses MessageQueue for embedding computation, making uploads asynchronous and decoupling heavy processing from request/response cycle.

## Coding Patterns
- Preferred patterns and practices: Class-based API, dynamic imports to avoid circular dependencies, modular design, config-driven, modular scripts, Docker automation, all API/admin endpoint automation via Python client
- Code organization preferences: src/umrs/ for core modules, docker/ for deployment, data/ for assets, scripts and Dockerfiles in docker/, client in src/
- Testing approaches: Automated, command-line, config-based, Docker-based
- Documentation style: Docstrings and comments in code


## Context7 Research History
- Libraries researched on Context7: Python API client best practices (no actionable docs found); Python requests/Docker SSL CA bundle best practices (no direct answer, used official docs/StackOverflow)
- Best practices discovered: Use requests, modular class, error handling, endpoint mapping; CA bundle for requests must be a valid CA cert (CA:TRUE), not just a server cert
- Implementation patterns used: Class-based API client, endpoint-to-method mapping, requests for HTTP; Docker Compose mounts CA cert as REQUESTS_CA_BUNDLE
- Version-specific findings: N/A


## Conversation History
- Important decisions made:
	- Initial project understanding and memory setup
	- Created src/client_umrs_api.py as a modular Python API client for UMRS endpoints
	- Class UMRSApiClient provides methods for health, status, bucket listing, text/image/textfile upload, and embedding management
	- Confirmed both user config files use the same format
	- Implemented Dockerfile, test script, and runner for automated API endpoint testing
	- Test script runs all endpoints (admin and user) for both users using their config and data directory, exclusively through the Python client
- SSL troubleshooting: Determined that mounting a server certificate as REQUESTS_CA_BUNDLE does not work; must use a CA cert (CA:TRUE) as the bundle. Created docker/ssl/SSL_FIX_INSTRUCTIONS.md with step-by-step fix for Dockerized requests SSL verification.
	- July 2024: Resolved MinIO S3 SSL trust for UMRS server in Docker Compose by:
		- Adding MinIO public.crt to system CA store in Dockerfile
		- Setting ENV SSL_CERT_FILE to /etc/ssl/certs/ca-certificates.crt
		- Refactoring s3.py to use system CA bundle for boto3 verify param
		- Fixing indentation and scope errors in check_bucket_path (now only uses passed-in s3 client)
		- Validated: UMRS server starts, connects to MinIO S3 over HTTPS, all buckets created, no SSL or NameError issues
		- All changes tested via docker compose up --build and confirmed in logs
- Recurring questions or topics: N/A
- Solutions that worked well: Modular class-based API client, Docker-based automated testing
- Things to avoid or that didn't work: N/A


## Notes
- Project supports uploading and storing text, images, and text files with optional metadata and embedding computation
- API endpoints for health, status, upload, bucket listing, and embedding management
- Uses S3 for storage and PyTorch for embeddings
- Designed for extensibility and modularity
- src/client_umrs_api.py: Modular Python API client for UMRS endpoints, class-based, uses requests, robust error handling, covers all admin and user endpoints. Now supports CLI entry point for all major methods (health, status, list_buckets, upload_text, upload_image, upload_textfile, recompute_embeddings, update_embeddings, check_consistency, add_user_hot_reload, create_user, create_buckets) using argparse. Example: `python3 client_umrs_api.py --base_url https://localhost:8000 health`.
- Test script: docker/test_umrs_api.sh (Bash wrapper, all logic calls client_umrs_api.py for every endpoint)
- Dockerfile: docker/Dockerfile.umrs-api-test
- Runner: docker/run_umrs_api_test.sh
- User configs: docker/conf_umrs_user1.json, docker/conf_umrs_user2.json
- Data: docker/data/
- Input: command + config.json + data
- On container start, all tests run automatically for both users

- MinIO SSL requirements: MinIO expects public.crt (certificate) and private.key (private key) in its SSL directory. Optionally, a CAs/ directory can be used for custom CA chains. The generate_ssl_certs.sh script was updated to robustly create or symlink public.crt and private.key from server.crt and server.key, and to create a CAs/ directory if not present. This ensures MinIO can use the same self-signed certs as the UMRS API for development/testing.
