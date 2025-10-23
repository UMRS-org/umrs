# TODO list

## Completed (recent)
- Per-user scoping for STATUS summaries
    - Server stores and returns maintenance summaries (check/update/recompute/reload) scoped to the requesting user only.
    - Clients querying `status` only see their own summaries; cross-user data is never included.
- Server pretty-print tables muted
    - Pretty-printed tables (config, check/update/recompute) are no longer printed on the server console; they are generated as strings for client retrieval.
    - Server still logs concise, non-sensitive progress messages for operators.
- Client-visible STATUS end-to-end
	- Implemented client `status` subcommands (all, recompute, update, check, reload, config) wired to server-side summaries.
	- pretty_print functions now return strings and are reused by the server to build responses.
	- Integrated thread status table printing on the server for operator visibility (not exposed to clients).
- Integration test hardening (docker/test_umrs_api.sh)
	- Added explicit status calls with comments after each maintenance operation.
	- Added a second full pass using `user2` to validate per-user isolation; verified isolation end-to-end.
- Upstream sync
	- Merged latest `main` into `update_docker`, resolved conflicts while preserving per-user STATUS behavior and thread status integration.

## Short-term next steps (actionable)
- Fix minor path bug in server user-config persistence
	- In `_persist_uploaded_user_config`, use `os.path.dirname(...)` instead of `os.path.dir(...)`.
- Guard against MinIO storage exhaustion during demos/tests
	- Add a small preflight space check or warn when free space is low.
	- Prefer object-level cleanup: document how to delete unnecessary files/objects from the affected buckets (e.g., via `s3cmd` or `mc`) to quickly recover space without resetting the service.
	- Provide a documented cleanup target to reset MinIO state between runs (e.g., prune `docker_minio_data` volume) as a last resort.
	- Add an optional flag in `docker/test_umrs_api.sh` to skip bulk uploads when running in constrained environments.
- Authorization/roles (foundation)
	- Enforce admin-only for global commands: RELOAD, RECOMPUTE (all users), UPDATE (all), CHECK (all), and cross-user STATUS.
	- Non-admins can run per-user scope for CHECK/UPDATE/RECOMPUTE and fetch only their own STATUS.
	- Return clear "unauthorized" errors for disallowed commands.

## admin users
- We need to have admin users/roles
	- Purpose: allow privileged operations such as CREATE_USER, RELOAD, UPDATE, CHECK, RECOMPUTE, and viewing multi-user summaries.
	- Scope: authorization model and credential store are out of scope here; at minimum, server-side checks must enforce role on incoming commands.
	- Acceptance:
		- Admin-only commands are rejected for non-admins with a clear message; accepted for admins.
		- Admins may list all users and their resources and fetch any user's status; regular users can only list/fetch their own.

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

- Adopt XDG Base Directory standard for configuration
	- Default config directory resolution order (first match wins):
		1. If environment variable `UMRS_CONFIG` is set: use its value (must be an absolute path; if relative, resolve to cwd and warn).
		2. Else if `XDG_CONFIG_HOME` is set: use `$XDG_CONFIG_HOME/umrs`.
		3. Else fallback to `~/.config/umrs`.
	- On startup, create the resolved directory if missing with permissions 700 (do not relax existing stricter perms) and log the final absolute path exactly once.
	- All global (non per-user JSON) configuration files (e.g., main `conf_umrs.json`, staging/pipeline defaults, logging config) must live under this directory; remove / deprecate ad‑hoc scattered locations.
	- Do not silently mix legacy locations: if a legacy `conf_umrs.json` is detected elsewhere while an XDG path is active, log a deprecation warning and ignore the legacy file (never merge implicitly).
	- Provide a helper function `resolve_config_dir()` used consistently across server and client code to avoid drift.
	- Document override behavior and precedence in README / HOWTO.
	- Acceptance:
		- Running without `UMRS_CONFIG` creates/uses `~/.config/umrs` (or `$XDG_CONFIG_HOME/umrs`) and logs: `UMRS config dir: <abs_path>`.
		- Setting `UMRS_CONFIG=/tmp/custom_umrs_cfg` causes all config reads/writes to occur solely under that directory; no accesses to the default path appear in logs (can verify via debug logging / grep).
		- Supplying both a legacy path and new XDG path results in a single clear deprecation warning and only the XDG-resolved directory being used.
		- A unit test (to be added later) can monkeypatch env vars and assert the resolution order.

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
	- Pretty-print tables are not emitted to server logs; summaries are returned to clients on demand via the `status` command.
	- Update client UI/CLI to state that operations are asynchronous; show high-level statuses (READY, CHECK, UPDATE, RECOMPUTE, RELOAD) and final summaries.
	- Acceptance: Client shows succinct status without raw server logs/tables; operators still see concise progress logs on the server.

- Per-user scoping for list and status operations
	- Any listing (buckets, experiences, tables) and status retrieval must be filtered server-side by the authenticated requester (username/user_id).
	- Admins may pass a target user to list cross-user and to fetch cross-user status; non-admins cannot.
	- Acceptance: Regular users see only their own resources and their own status summaries; cross-user data is never returned to them.

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

## API command simplification & legacy mode removal
- Rationale: Current API flow/command set is redundant and not generic; modality is already implied by the bucket so explicit modality arguments and legacy branching are unnecessary.
- Remove: All legacy mode code paths and flags ("legacy mode" concept entirely eliminated).
- Unified command set (authoritative list; replace prior/legacy commands):
	- `ping`                : Check server readiness.
	- `upload_file`         : Upload a single file (text/image/etc.) to a given bucket; modality inferred from bucket.
	- `upload_dir`          : Recursively upload all files in a directory to a given bucket; optional include/exclude patterns.
	- `list_buckets`        : List bucket aliases for the authenticated username (admins may specify a target user).
	- `recompute`           : Recompute all embeddings for all buckets (server-side, admin or per-user scope depending on auth model).
	- `update`              : Compute embeddings only for missing/mismatched entries (idempotent, resumable).
	- `check`               : Check consistency between S3 objects and DB entries (count, missing, orphan, hash mismatch stats).
	- `reload`              : Reload server configuration / models (admin only).
- Behavior changes:
	- Modality resolution: Server derives modality strictly from bucket metadata / naming convention; no client-provided modality allowed (reject if supplied to prevent drift).
	- Upload validation: On `upload_file` and `upload_dir`, server performs the same content-type / extension checks previously done in legacy path; errors reported per-file (multi-file continues on failures, summarizing at end).
	- Idempotency: `upload_dir` must skip already ingested identical files (hash or size+mtime heuristic) unless a `--force` flag is provided.
	- Progress feedback: For bulk uploads, client shows a concise progress bar + final summary (counts: uploaded, skipped, failed).
- Removal tasks:
	- Delete / refactor legacy branching code and flags in both client and server entrypoints.
	- Remove obsolete command names from help/README and docs (`TODO: enumerate after code scan`).
	- Add migration note: "Legacy mode removed; use unified commands; modality now inferred from bucket." in HOWTO / README.
- Implementation tasks:
	- Refactor command dispatch layer to a single registry dict: {command_name: handler} shared across client & server interface definition (avoid drift).
	- Introduce a structured request object schema (command, args, username, user_id) and validate before dispatch.
	- Implement bucket->modality lookup helper (cached) with clear errors if bucket unknown or misconfigured.
	- Add filtering/pattern options for `upload_dir` (e.g. `--include "*.txt" --exclude "*.bak"`).
	- Ensure `recompute`, `update`, `check`, `reload` enforce admin/non-admin authorization rules defined in earlier admin section.
	- Update pretty_print summaries to include: command, duration, counts, error summary.
	- Extend tests: command parser, bucket modality inference, upload idempotency, error handling for mixed success in directory upload.
- Acceptance criteria:
	- Help output (client `--help`) lists only the unified commands above and no legacy options.
	- Attempting a removed/legacy command returns a clear error: `Unknown command '<name>' (legacy commands removed)`.
	- Uploading a file to a bucket with known modality proceeds without requiring modality argument; providing one yields a validation error.
	- Directory upload with mixed file types processes only those valid for the inferred modality and reports skipped/invalid separately.
	- `recompute`, `update`, `check`, `reload` rejected for non-admin users with clear authorization message; succeed for admins.
	- All prior embedding-only flows remain functional via `upload_file` / `upload_dir` + `update` / `recompute` without regressions.
	- Tests cover: command dispatch (positive + unknown), bucket modality inference failure, idempotent skip logic, partial failure reporting.
	- Legacy code paths fully removed (grep for markers / flags returns no matches) and CI check passes.
- Follow-up (optional enhancements):
	- Add `dry-run` flag for `upload_dir` to report would-be actions without side effects.
	- Add concurrency controls / rate limiting for large directory uploads.
	- Support checksum pre-manifest to accelerate idempotency check for massive directories.


