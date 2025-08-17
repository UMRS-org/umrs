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
- Current project type: Multi-modal retrieval system (UMRS)
- Tech stack: Python, FastAPI, Uvicorn, S3-compatible storage, PostgreSQL (likely with pgvector), PyTorch, Docker
- Architecture patterns: Modular, class-based API, multi-user/multi-bucket, RESTful
- Key requirements: Multi-user, multi-bucket, multi-modality support; embedding computation; robust API; containerized deployment
- API mode now uses MessageQueue for embedding computation, making uploads asynchronous and decoupling heavy processing from request/response cycle.

## Coding Patterns
- Preferred patterns and practices: Class-based API, dynamic imports to avoid circular dependencies, modular design
- Code organization preferences: src/umrs/ for core modules, docker/ for deployment, data/ for assets
- Testing approaches: Not specified yet
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
- SSL troubleshooting: Determined that mounting a server certificate as REQUESTS_CA_BUNDLE does not work; must use a CA cert (CA:TRUE) as the bundle. Created docker/ssl/SSL_FIX_INSTRUCTIONS.md with step-by-step fix for Dockerized requests SSL verification.
- Recurring questions or topics: N/A
- Solutions that worked well: Modular class-based API client
- Things to avoid or that didn't work: N/A


## Notes
- Project supports uploading and storing text, images, and text files with optional metadata and embedding computation
- API endpoints for health, status, upload, bucket listing, and embedding management
- Uses S3 for storage and PyTorch for embeddings
- Designed for extensibility and modularity
- src/client_umrs_api.py: Modular Python API client for UMRS endpoints, class-based, uses requests, robust error handling to be added
