# UMRS Roadmap

> Forward-looking architectural work items. Completed items are tracked in
> [tasks/todo.md](tasks/todo.md) and [DEVELOPMENT_OBJECTIVES.md](DEVELOPMENT_OBJECTIVES.md).

---

## Short-term

- Fix minor path bug in server user-config persistence
	- In `_persist_uploaded_user_config`, use `os.path.dirname(...)` instead of `os.path.dir(...)`.
- Guard against MinIO storage exhaustion during demos/tests
	- Add a small preflight space check or warn when free space is low.
	- Document how to delete unnecessary files/objects from affected buckets (e.g., via `s3cmd` or `mc`) to recover space.
	- Document a cleanup target to reset MinIO state between runs (e.g., prune `docker_minio_data` volume) as a last resort.
	- Add an optional flag in `docker/test_umrs_api.sh` to skip bulk uploads in constrained environments.

## Authorization & Roles

- Enforce admin-only for global commands: RELOAD, RECOMPUTE (all users), UPDATE (all), CHECK (all), and cross-user STATUS.
- Non-admins can run per-user scope for CHECK/UPDATE/RECOMPUTE and fetch only their own STATUS.
- Return clear "unauthorized" errors for disallowed commands.
- Admins may list all users and their resources and fetch any user's status; regular users can only list/fetch their own.

## User Environment Separation

### Server-configured user configuration directory
- Add a `user_files_dir` field in the server configuration file (e.g., `conf_umrs.json`).
- Default to `/root/user_files` if not specified; allow override via `UMRS_USER_FILES_DIR`.
- On startup, validate the directory: create if missing, ensure writable, set restrictive permissions (700).
- Canonical per-user config path: `<user_files_dir>/<username>.json`.
- Update Docker and docs to reference this path; remove hardcoded alternatives.

### XDG Base Directory standard for configuration
- Resolution order: `UMRS_CONFIG` → `$XDG_CONFIG_HOME/umrs` → `~/.config/umrs`.
- Create the resolved directory if missing (permissions 700); log the final absolute path once.
- Deprecation: if legacy conf_umrs.json detected elsewhere, log warning and ignore.
- Provide `resolve_config_dir()` helper used consistently across server and client.

### Per-user stable identifier (user_id)
- Generate server-side UUID v4 when creating a user.
- Persist in `<user_files_dir>/<username>.json` alongside `username`.
- Backward compatibility: if stored config lacks `user_id`, generate on next reload and log the upgrade.

### Validate uploaded user configuration identity
- On FILE uploads, verify `username` and `user_id` match stored canonical config.
- Reject mismatches; accept valid updates and atomically replace the canonical file.

### Clarify server logs vs client feedback
- Server stdout/stderr not streamed to clients; only ack + summaries sent back.
- Client UI/CLI states that operations are asynchronous; shows high-level statuses.

### Per-user scoping for list and status operations
- Filter listings server-side by authenticated requester.
- Admins may pass a target user for cross-user visibility.

## Flexible Processing Stages

- Generalize beyond embedding calculation to support arbitrary stage pipelines:
	- Provisioning → Pre-process → Embedding → Post-process → Optional custom stages
- Stage contract: inputs (S3 keys / DB rows), outputs (derived artifacts / DB updates), idempotent, error isolation.
- Orchestration: sequence stages per bucket/modality, CHECK/UPDATE/RECOMPUTE per stage, per-stage metrics.
- Existing single-stage embedding flow must remain unchanged when other stages are disabled.

## Codebase Hygiene: Server / Client Separation

- Split into `src/umrs_server/`, `src/umrs_client/`, `src/umrs_core/` (shared).
- Keep current top-level scripts as thin wrappers for backward compatibility.
- Client artifact must contain zero files from `umrs_server/` (CI check).
- No server secrets or admin tools in client image.
- Define public client-server interface (ZMQ control + S3 data planes).
- Update Dockerfiles/tests to consume refactored paths.

## API Command Simplification & Legacy Removal

- Remove all legacy mode code paths and flags.
- Unified commands: `ping`, `upload_file`, `upload_dir`, `list_buckets`, `recompute`, `update`, `check`, `reload`.
- Server derives modality strictly from bucket metadata — no client-provided modality.
- `upload_dir`: skip already-ingested identical files (hash/mtime) unless `--force`.
- Progress feedback: client shows progress bar + final summary.
- Single registry dict `{command: handler}` for both client and server.
- Add `--include`/`--exclude` patterns for `upload_dir`.
- Legacy command attempts return: `Unknown command '<name>' (legacy commands removed)`.

---

## Notes

- Security/authn details will be defined separately.
- Follow-up enhancements: `dry-run` for upload_dir, concurrency controls, checksum pre-manifest.
