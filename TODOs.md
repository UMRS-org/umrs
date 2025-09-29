# TODO list

## admin users
- We need to have admin users/roles
	- Purpose: allow privileged operations such as CREATE_USER, RELOAD, UPDATE, CHECK, RECOMPUTE, and viewing multi-user summaries.
	- Scope: authorization model and credential store are out of scope here; at minimum, server-side checks must enforce role on incoming commands.
	- Acceptance:
		- Admin-only commands are rejected for non-admins with a clear message; accepted for admins.
		- Admins may list all users and their resources; regular users can only list their own.

## to make the user environments more separated
- Server-configured user configuration directory
	- Add a `user_files_dir` field in the server configuration file (e.g., `conf_umrs.json`) to define where per-user configuration JSON files are stored on the server side.
	- Default to `/root/user_files` if not specified; allow override via environment variable `UMRS_USER_FILES_DIR`.
	- On server startup, validate the directory: create it if missing, ensure it is writable by the server process, and set restrictive permissions (e.g., 700).
	- Canonical per-user config path: `<user_files_dir>/<username>.json`; all reads/writes must go through this location (single source of truth).
	- Update Docker and docs to reference this path and remove any hardcoded alternatives to avoid drift.
	- Acceptance:
		- Server logs the resolved absolute `user_files_dir` on startup; changing the value redirects subsequent reads/writes to the new location.
		- If the directory is missing or not writable, startup fails fast with a clear error; when valid, creating/updating a user writes the file under the configured directory.
		- Only one directory is in use (no conflicting paths remain in configs or code).

- Define and persist a per-user stable identifier (user_id)
	- Generate a server-side user_id (UUID v4 or short hash) when creating a user.
	- Persist user_id in the canonical config at `/root/user_files/<username>.json` alongside `username`.
	- Use `username` + `user_id` when deriving S3 paths and DB namespaces (where applicable) to avoid collisions.
	- Acceptance: After user creation or first config load, `/root/user_files/<username>.json` contains both fields, and all resources can be traced back to this pair.

- Validate uploaded user configuration identity
	- On FILE uploads, check that the `username` in the JSON matches the client identity (if available) and that `user_id` (if present) matches the stored canonical config for that username.
	- Reject mismatches; accept valid updates and atomically replace the canonical file; ensure only one path per user is referenced in `conf_umrs.json`.
	- Acceptance: Mismatched uploads are rejected with a clear server log; valid uploads replace the canonical file and are reflected in the server conf.

- Clarify server logs vs client feedback
	- Server stdout/stderr is not streamed to clients; only acknowledgements and concise summaries are sent back.
	- Update client UI/CLI to state that operations are asynchronous; show high-level statuses (READY, CHECK, UPDATE, RECOMPUTE, RELOAD) and final summaries.
	- Acceptance: Client shows succinct status without raw server logs; operators still see full logs on the server.

- Per-user scoping for list operations
	- Any listing (buckets, experiences, tables) must be filtered server-side by the authenticated requester (username/user_id).
	- Admins may pass a target user to list cross-user; non-admins cannot.
	- Acceptance: Regular users see only their own resources; cross-user data is never returned to them.

## to make the processes more flexible
- we want to add more processes but not only the embedding calculation.
- we need to find the most general case of this usage, to plan implementation, which may be
	- Stage model (per modality):
		- Provisioning (bucket creation)
		- Pre-process (e.g., OCR, normalization, resizing, text cleaning)
		- Embedding calculation (current behavior)
		- Post-process (e.g., clustering, index building, QA metrics)
		- Optional custom stages
	- Stage contract (for each stage):
		- Inputs: S3 keys and/or DB rows, metadata
		- Outputs: derived artifacts and/or DB updates; must be idempotent
		- Storage: write derived artifacts under deterministic prefixes, e.g., `pre/<stage_name>/...` or `post/<stage_name>/...`
		- Errors: fail the item with a recorded reason; do not block other items
	- Orchestration:
		- Sequence stages per bucket/modality; support CHECK/UPDATE/RECOMPUTE per stage.
		- Extend pretty_print summaries to include per-stage metrics (count, duration, data size).
	- Acceptance:
		- Existing single-stage embedding flow remains supported unchanged when other stages are disabled.
		- Enabling a pre/post stage produces expected outputs in S3/DB without breaking ingestion.

## Notes

- Backward compatibility: if a stored config lacks `user_id`, generate and persist it on next reload/update and log the upgrade.
- Security/authn details will be defined separately; this file captures behavioral requirements and acceptance criteria.

## codebase hygiene: separate server and client code
- Goal: decouple server and client into clearly separated modules and packages to simplify maintenance, testing, and deployment.
	- Create `src/umrs_server/` for all server-only code (entrypoint, MessageQueue server loop, provisioning, compute pipeline).
	- Create `src/umrs_client/` for all client-only code (CLI, MessageQueue client wrapper, S3 upload helpers).
	- Move shared utilities to `src/umrs_core/` (types, json/schema utils, pretty print, s3/sql helpers that are truly shared).
	- Keep current top-level CLI files as thin wrappers that import from the new packages to maintain backward compatibility.
	- Update Dockerfiles/tests to consume the refactored paths; ensure both API and legacy flows still work.
- Security-driven separation (hide server code from client):
	- Build and ship separate artifacts/images:
		- Client wheel/image must not contain any modules from `umrs_server` (verify via CI grep and import checks).
		- Server wheel/image must not include client test tooling beyond the CLI entrypoint.
	- Do not include server-only configs or admin tools in client image:
		- No `conf_umrs.json`, no `/root/user_files` baked in client image.
		- Exclude admin S3 tools (e.g., `mc admin`) and DB superuser utilities from client image.
	- Define and enforce a minimal, stable client-server interface:
		- Public client API = ZeroMQ control plane + S3 data plane; document allowed commands and expected replies.
		- Server validates and authorizes control commands; reject unknown/privileged commands from non-admin clients.
	- Secrets and credentials hygiene:
		- No server secrets embedded in client artifacts; client uses user-scoped s3cfg mounted/provided at runtime.
		- Server-side env/secret management remains in server image or orchestrator, never shipped to clients.
- Acceptance (security):
	- Client artifact (wheel + Docker) contains zero files under `umrs_server/` (CI check fails if present).
	- No admin binaries/configs in client image; attempting to run admin flows from client fails by design.
	- A “public API” doc exists describing the client-facing commands; attempting an undeclared command is rejected server-side.
	- SBOM or manifest for each image lists only expected packages; CI compares against an allowlist.
- Acceptance:
	- Running `python src/server_umrs.py` and `python src/client_umrs.py` still works unchanged for end users.
	- Unit tests for server and client can be run independently (e.g., `pytest tests/server` vs `tests/client`).
	- A minimal README explains the three-layer layout (server/client/core) and public entrypoints.


