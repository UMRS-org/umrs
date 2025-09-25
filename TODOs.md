# TODO list

## admin users
- We need to have admin users/roles
	- Purpose: allow privileged operations such as CREATE_USER, RELOAD, UPDATE, CHECK, RECOMPUTE, and viewing multi-user summaries.
	- Scope: authorization model and credential store are out of scope here; at minimum, server-side checks must enforce role on incoming commands.
	- Acceptance:
		- Admin-only commands are rejected for non-admins with a clear message; accepted for admins.
		- Admins may list all users and their resources; regular users can only list their own.

## to make the user environments more separated
- Define and persist a per-user stable identifier (user_id)
	- Generate a server-side user_id (UUID v4 or short hash) when creating a user.
	- Persist user_id in the canonical config at `/app/user_files/<username>.json` alongside `username`.
	- Use `username` + `user_id` when deriving S3 paths and DB namespaces (where applicable) to avoid collisions.
	- Acceptance: After user creation or first config load, `/app/user_files/<username>.json` contains both fields, and all resources can be traced back to this pair.

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


