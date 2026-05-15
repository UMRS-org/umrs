"""``umrs-server admin`` argparse dispatcher."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from typing import Any, cast

from umrs.admin.daemon_signal import (
    resolve_pid_file_path,
    signal_reload,
)
from umrs.admin.user_store import (
    VALID_ROLES,
    UserRecord,
    UserStore,
    UserStoreError,
)
from umrs.shared.config import default_s3cfg_path

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _get_hostname() -> str:
    """Return a short hostname for scp instructions."""
    import socket
    return socket.gethostname()


# Actions that mutate user records and should trigger an auto-reload.
_MUTATING_ACTIONS: frozenset = frozenset(
    {
        "add",
        "delete",
        "disable",
        "enable",
        "set-role",
        "set-slurm-account",
        "add-to-project",
        "remove-from-project",
    }
)


def _maybe_signal_reload(args: argparse.Namespace) -> None:
    """Best-effort SIGHUP to the daemon after mutating admin ops.

    Suppressed when ``--no-reload`` is set, when the operation is read-only
    (``user list/show``, ``token list``, ``slurm accounts``), or when no
    daemon is running. Prints a one-line note so operators know the result.
    """
    if getattr(args, "no_reload", False):
        return
    group = getattr(args, "group", None)
    action = getattr(args, "action", None)
    if group != "user" or action not in _MUTATING_ACTIONS:
        return
    pid_file = getattr(args, "pid_file", None) or resolve_pid_file_path()
    result = signal_reload(pid_file)
    if result.sent:
        print(f"[reload] {result.reason}")
    else:
        print(f"[reload] skipped: {result.reason}")


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=False, default=str))


def _record_row(rec: UserRecord) -> dict[str, Any]:
    return {
        "username": rec.username,
        "role": rec.role,
        "slurm_account": rec.slurm_account or "",
        "projects": ",".join(rec.projects),
        "disabled": rec.disabled,
        "email": rec.email,
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(no users)")
        print(
            "hint: register one with `umrs-server admin user add <username> "
            "--fullname \"Your Name\" --role user`"
        )
        return
    columns = ["username", "role", "slurm_account", "projects", "disabled", "email"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))


def _confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# helpers — client config template
# ---------------------------------------------------------------------------


def _build_client_config_template(username: str, fullname: str = "") -> dict:
    """Return a minimal client config dict (experiences/modalities/buckets)."""
    return {
        "username": username,
        "fullname": fullname,
        "s3_configuration_file": default_s3cfg_path(username),
        "experiences": [
            {
                "xp_name": "default",
                "sql_username": username,
                "modalities": [
                    {
                        "modality_name": "text",
                        "s3": {
                            "bucket_alias": f"{username}-text",
                            "data": {
                                "type": "inline",
                                "mime_type": "text/plain",
                            },
                        },
                        "model": {
                            "model_name": "intfloat/e5-base-v2",
                            "vector_size": "768",
                            "model_type": "TransformerEncoder",
                            "return_type": None,
                        },
                        "sql": {
                            "store_embeddings": "yes",
                            "table_name": "text",
                            "meta_data": [
                                {"name": "keywords", "type": "TEXT"},
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _write_client_template(username: str, fullname: str, out_path) -> None:
    """Write a client config template JSON to *out_path*."""
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    template = _build_client_config_template(username, fullname)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# user subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_user_add(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.create(
            args.username,
            fullname=args.fullname or "",
            email=args.email or "",
            role=args.role,
            slurm_account=args.slurm_account,
            projects=args.project or [],
        )
    except UserStoreError as exc:
        # If the user already exists, still try to generate a client template
        # for migration (pre-existing users created before this feature).
        if "already exists" in str(exc):
            try:
                rec = store.get(args.username)
            except UserStoreError:
                rec = None
            if rec is not None:
                print(f"error: {exc}", file=sys.stderr)
                _maybe_emit_template(args, store, rec)
            else:
                print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Auto-generate a client config template alongside the admin record.
    _maybe_emit_template(args, store, rec)

    if args.json:
        out = rec.to_json()
        tpl_path = _client_template_path(args, store, rec)
        if tpl_path and tpl_path.exists():
            out["client_template"] = str(tpl_path)
        _print_json(out)
    else:
        print(f"created user {rec.username} at {store.dir / (rec.username + '.json')}")
    return 0


def _client_template_path(args, store, rec):
    from pathlib import Path
    if getattr(args, "no_client_template", False):
        return None
    if getattr(args, "client_template", None):
        return Path(args.client_template).expanduser()
    return store.dir / f"{rec.username}.client.json"


def _maybe_emit_template(args, store, rec) -> None:
    tpl_path = _client_template_path(args, store, rec)
    if tpl_path is None:
        return
    if tpl_path.exists():
        print(
            f"warning: client template {tpl_path} already exists (skipped; "
            f"use 'user generate-template --force' to regenerate)",
            file=sys.stderr,
        )
        return
    _write_client_template(rec.username, rec.fullname, tpl_path)
    if not getattr(args, "json", False):
        print(f"client template written to {tpl_path}")

        # Issue an onboarding JWT if requested.
        token_hint = ""
        if getattr(args, "issue_token", False):
            try:
                secret = _resolve_jwt_secret(args)
                payload = _build_jwt_payload(
                    user=rec.username,
                    email=rec.email or "",
                    project=rec.projects[0] if rec.projects else "",
                    roles=[rec.role],
                    days=getattr(args, "token_days", 30),
                    projects=rec.projects or None,
                )
                from jose import jwt as jose_jwt

                token = jose_jwt.encode(payload, secret, algorithm="HS256")
                token_hint = token
                print()
                print(f"  Onboarding token (valid {getattr(args, 'token_days', 30)}d):")
                print(f"    {token}")
            except Exception as exc:
                print(f"  warning: could not issue onboarding JWT: {exc}", file=sys.stderr)

        print()
        print("  Onboarding steps for the new user:")
        if token_hint:
            print(f"  r4 (HTTP path):")
            print(f"       r4 auth login --gateway-url <GATEWAY_URL> --api-key {token_hint}")
            print(f"  umrs-client (ZMQ path):")
            print(f"       umrs-client login --host <HOST> --api-key {token_hint}")
        else:
            print(f"  1. Generate a token for the user:")
            print(f"       umrs-server admin token generate-jwt --user {rec.username}")
            print(f"  2. Give the token to the user, who runs:")
            print(f"       r4 auth login --gateway-url <GATEWAY_URL> --api-key <TOKEN>")
            print(f"     or:")
            print(f"       umrs-client login --host <HOST> --api-key <TOKEN>")


def _cmd_user_list(args: argparse.Namespace, store: UserStore) -> int:
    records = store.list()
    if args.role:
        records = [r for r in records if r.role == args.role]
    if args.disabled_only:
        records = [r for r in records if r.disabled]
    rows = [_record_row(r) for r in records]
    if args.json:
        _print_json([r.to_json() for r in records])
    else:
        _print_table(rows)
    return 0


def _cmd_user_show(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.get(args.username)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_json(rec.to_json())
    return 0


def _cmd_user_delete(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.get(args.username)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.deprovision:
        if not args.purge:
            print("error: --deprovision requires --purge", file=sys.stderr)
            return 1
        if not _confirm(
            f"Deprovision and hard-delete user {rec.username!r}? "
            f"This removes all S3 buckets, Postgres databases, and the user JSON file.",
            args.yes,
        ):
            print("aborted", file=sys.stderr)
            return 1

        # Load the user config to get experiences/modalities for teardown.
        user_json_path = store._path_for(args.username)
        try:
            with open(user_json_path, encoding="utf-8") as f:
                user_cfg = json.load(f)
        except Exception as exc:
            print(f"error: cannot load user config for deprovisioning: {exc}", file=sys.stderr)
            return 2

        # Resolve admin S3 config from the server config file.
        from umrs.shared.config import (
            get_default_json_server_file,
            get_resolved_umrs_cfg_dir,
        )
        import os
        server_json = os.path.join(get_resolved_umrs_cfg_dir(), get_default_json_server_file())
        admin_s3_cfg = None
        if os.path.isfile(server_json):
            with open(server_json, encoding="utf-8") as f:
                srv = json.load(f)
            admin_s3_cfg = srv.get("admin_s3_configuration_file")

        if not admin_s3_cfg:
            print("warning: admin_s3_configuration_file not found in server config; "
                  "S3/MinIO cleanup will be skipped.", file=sys.stderr)

        from umrs.server.provisioning import deprovision_user
        deprovision_user(user_cfg, admin_s3_cfg or "")

        path = store.delete(args.username)
        print(f"deprovisioned and deleted {path}")
        return 0

    if args.purge:
        if not _confirm(
            f"Hard-delete user {rec.username!r} (JSON file will be removed)?",
            args.yes,
        ):
            print("aborted", file=sys.stderr)
            return 1
        path = store.delete(args.username)
        print(f"deleted {path}")
        return 0
    # Soft delete: mark disabled. Preserve job/audit history (Decision 1).
    store.set_disabled(args.username, True)
    print(f"soft-deleted (disabled) user {rec.username}")
    print("Use `--purge` to remove the JSON file and revoke filesystem presence.")
    print("Use `--purge --deprovision` to also remove S3 buckets, Postgres databases, and MinIO IAM user.")
    return 0


def _cmd_user_disable(args: argparse.Namespace, store: UserStore) -> int:
    try:
        store.set_disabled(args.username, True)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"disabled {args.username}")
    return 0


def _cmd_user_enable(args: argparse.Namespace, store: UserStore) -> int:
    try:
        store.set_disabled(args.username, False)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"enabled {args.username}")
    return 0


def _cmd_user_set_role(args: argparse.Namespace, store: UserStore) -> int:
    try:
        store.set_role(args.username, args.role)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"set role={args.role} on {args.username}")
    return 0


def _cmd_user_set_slurm_account(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.set_slurm_account(args.username, args.account)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"set slurm_account={rec.slurm_account or '(unset)'} on {args.username}")
    return 0


def _cmd_user_add_project(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.add_project(args.username, args.project_id)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"projects for {args.username}: {rec.projects}")
    return 0


def _cmd_user_remove_project(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.remove_project(args.username, args.project_id)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"projects for {args.username}: {rec.projects}")
    return 0


def _cmd_user_generate_template(args: argparse.Namespace, _store: UserStore) -> int:
    """Generate a template user.json for a new client."""
    from pathlib import Path

    username = args.username
    fullname = args.fullname or ""

    if args.json:
        _print_json(_build_client_config_template(username, fullname))
        return 0

    output = args.output_file
    if output is None:
        output = f"user_{username}.json"

    out_path = Path(output).expanduser()
    if out_path.exists() and not args.force:
        print(f"error: {out_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 2

    _write_client_template(username, fullname, out_path)

    print(f"Template written to {out_path}")
    print()
    print(f"  Onboarding steps for the user:")
    print(f"  1. Send '{out_path.name}' to the user (or copy via scp)")
    print(f"  2. On their local machine:")
    print(f"       r4 config init --gateway-url <GATEWAY_URL> --umrs-username {username}")
    print(f"  3. Config is stored locally at ~/.config/r4/profiles.json")
    return 0


# ---------------------------------------------------------------------------
# config subcommand handlers
# ---------------------------------------------------------------------------


def _cmd_config_show(args: argparse.Namespace, _store: UserStore) -> int:
    """Show which config files the server would read."""
    import os

    from umrs.shared.config import (
        UMRS_CFG_ENV_VAR_NAME,
        UMRS_USER_FILES_ENV_VAR_NAME,
        get_default_json_server_file,
        get_resolved_umrs_cfg_dir,
        resolve_server_user_files_dir,
    )

    cfg_dir = get_resolved_umrs_cfg_dir()
    cfg_dir_source = (
        f"${UMRS_CFG_ENV_VAR_NAME}" if os.environ.get(UMRS_CFG_ENV_VAR_NAME)
        else "default"
    )
    server_json = os.path.join(cfg_dir, get_default_json_server_file())
    server_json_exists = os.path.isfile(server_json)

    user_files_dir = resolve_server_user_files_dir()
    user_files_source = (
        f"${UMRS_USER_FILES_ENV_VAR_NAME}" if os.environ.get(UMRS_USER_FILES_ENV_VAR_NAME)
        else "default"
    )
    user_files_exists = os.path.isdir(user_files_dir)

    if args.json:
        _print_json({
            "config_dir": cfg_dir,
            "config_dir_source": cfg_dir_source,
            "server_config": server_json,
            "server_config_exists": server_json_exists,
            "user_files_dir": user_files_dir,
            "user_files_dir_source": user_files_source,
            "user_files_dir_exists": user_files_exists,
        })
    else:
        ok = "\u2713"
        no = "\u2717"
        print(f"Config directory:  {cfg_dir}")
        print(f"  source:          {cfg_dir_source}")
        print(f"  exists:          {ok if os.path.isdir(cfg_dir) else no}")
        print()
        print(f"Server config:     {server_json}")
        print(f"  exists:          {ok if server_json_exists else no}")
        print()
        print(f"User files dir:    {user_files_dir}")
        print(f"  source:          {user_files_source}")
        print(f"  exists:          {ok if user_files_exists else no}")
        if user_files_exists:
            count = len([f for f in os.listdir(user_files_dir) if f.endswith(".json")])
            print(f"  user records:    {count}")
    return 0


# ---------------------------------------------------------------------------
# token subcommand handlers
# ---------------------------------------------------------------------------


def _token_manager():
    from umrs.shared.auth import AdminTokenManager
    return AdminTokenManager()


def _cmd_token_generate(args: argparse.Namespace, _store: UserStore) -> int:
    mgr = _token_manager()
    existing = {t["name"] for t in mgr.list_tokens()}
    overwrite = args.name in existing
    token = mgr.generate_token(args.name, description=args.description or "")
    if args.json:
        _print_json({"name": args.name, "token": token, "overwritten": overwrite})
    else:
        action = "Regenerated" if overwrite else "Generated"
        print(f"{action} admin token '{args.name}' (for umrs-client --admin-token / $UMRS_ADMIN_TOKEN)")
        print(f"token: {token}")
        print("  (this is the only time the raw token will be shown)")
        print()
        print("  Use with: umrs-client --admin-token <token> <command>")
        print("         or: export UMRS_ADMIN_TOKEN=<token>")
        print()
        print("  NOTE: this token is for the umrs-client ZMQ control plane only.")
        print("        To authenticate r4/Gateway, use 'umrs-server admin token generate-jwt' instead.")
    return 0


def _resolve_jwt_secret(args: argparse.Namespace) -> str:
    """Resolve the JWT signing secret from CLI flag, env, Gateway secret, or default."""
    # Check --secret CLI flag first (admin-only override).
    secret = getattr(args, "secret", None)
    if secret:
        return secret
    # Delegate to shared resolution logic.
    from umrs.shared.auth import resolve_jwt_secret

    return resolve_jwt_secret()


def _build_jwt_payload(
    user: str,
    email: str,
    project: str,
    roles: list[str],
    days: int,
    *,
    projects: list[str] | None = None,
) -> dict[str, Any]:
    """Build the JWT claims payload.

    When *projects* is supplied the token carries a
    ``project_memberships`` list so a single JWT can be used across
    multiple projects.  ``project_id`` is always emitted for backward
    compatibility (set to the first entry of *projects* when given).
    """
    import time

    now = int(time.time())
    effective_project = project
    if projects:
        effective_project = projects[0]
    payload: dict[str, Any] = {
        "sub": user,
        "email": email,
        "project_id": effective_project,
        "roles": roles,
        "iat": now,
        "exp": now + (days * 86400),
    }
    if projects and len(projects) > 1:
        payload["project_memberships"] = list(projects)
    return payload


def _cmd_token_generate_jwt(args: argparse.Namespace, _store: UserStore) -> int:
    """Generate a project-scoped dev/testing JWT for Gateway authentication."""
    import time

    try:
        from jose import jwt as jose_jwt
    except ImportError:
        print(
            "error: python-jose is required. "
            "Install with: pip install python-jose[cryptography]",
            file=sys.stderr,
        )
        return 2

    secret = _resolve_jwt_secret(args)
    email = args.email or f"{args.user}@dev.local"
    roles = args.roles or ["admin"]
    projects = getattr(args, "projects", None) or None
    payload = _build_jwt_payload(
        args.user, email, args.project, roles, args.days,
        projects=projects,
    )
    token = jose_jwt.encode(payload, secret, algorithm="HS256")

    using_default_user = args.user == "admin"
    using_default_project = args.project == "default" and not projects

    if args.json:
        _print_json({
            "token": token,
            "claims": payload,
            "expires_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(payload["exp"])
            ),
        })
    else:
        memberships = payload.get("project_memberships")
        scope_label = (
            f"projects {', '.join(memberships)}"
            if memberships
            else f"project '{payload['project_id']}'"
        )
        print(f"Generated JWT for user '{payload['sub']}' in {scope_label}")
        print(f"token: {token}")
        print(f"  sub:        {payload['sub']}")
        print(f"  email:      {payload['email']}")
        print(f"  project_id: {payload['project_id']}")
        if memberships:
            print(f"  projects:   {', '.join(memberships)}")
        print(f"  roles:      {', '.join(roles)}")
        print(f"  expires:    {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(payload['exp']))}")
        if using_default_user or using_default_project:
            warnings = []
            if using_default_user:
                warnings.append("--user was not set, defaulted to 'admin'")
            if using_default_project:
                warnings.append("--project was not set, defaulted to 'default'")
            print()
            for w in warnings:
                print(f"  warning: {w}", file=sys.stderr)
            print(
                "  hint: pass --user <username> --project <project-id> to scope this token correctly.",
                file=sys.stderr,
            )
        print()
        print("  Use with: r4 auth login --api-key <token>")
    return 0


def _cmd_token_list(args: argparse.Namespace, _store: UserStore) -> int:
    mgr = _token_manager()
    tokens = mgr.list_tokens()
    if args.json:
        _print_json(tokens)
    else:
        if not tokens:
            print("(no admin tokens)")
            return 0
        rows = [
            {
                "name": t.get("name", ""),
                "created_at": t.get("created_at", ""),
                "expires_at": t.get("expires_at", "never"),
                "description": t.get("description", ""),
            }
            for t in tokens
        ]
        columns = ["name", "created_at", "expires_at", "description"]
        widths = {
            column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
            for column in columns
        }
        print("  ".join(column.ljust(widths[column]) for column in columns))
        print("  ".join("-" * widths[column] for column in columns))
        for row in rows:
            print("  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns))
    return 0


def _cmd_token_revoke(args: argparse.Namespace, _store: UserStore) -> int:
    mgr = _token_manager()
    if not _confirm(f"Revoke admin token {args.name!r}?", args.yes):
        print("aborted", file=sys.stderr)
        return 1
    ok = mgr.revoke_token(args.name)
    print("revoked" if ok else f"token not found: {args.name}")
    return 0 if ok else 2


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _cmd_bootstrap(args: argparse.Namespace, store: UserStore) -> int:
    """Idempotent first-run: create admin user + initial admin token."""
    created_user = False
    try:
        rec = store.get(args.admin_user)
    except UserStoreError:
        rec = store.create(
            args.admin_user,
            fullname=args.fullname or "Platform Administrator",
            email=args.email or "",
            role="admin",
        )
        created_user = True

    mgr = _token_manager()
    existing = mgr.list_tokens()
    token: str | None = None
    if not any(t.get("name") == args.token_name for t in existing):
        token = mgr.generate_token(
            args.token_name, description="Bootstrap admin token"
        )

    result: dict[str, Any] = {
        "admin_user": rec.username,
        "created_user": created_user,
        "token_name": args.token_name,
        "token": token,
        "token_created": token is not None,
        "user_files_dir": str(store.dir),
    }
    if args.json:
        _print_json(result)
    else:
        if created_user:
            print(f"✓ Created admin user: {rec.username}")
        else:
            print(f"✓ Admin user already exists: {rec.username} (no changes)")
        if token:
            print(f"✓ Generated bootstrap token ({args.token_name}): {token}")
            print("  (this is the only time the raw token will be shown)")
        else:
            print(f"✓ Token {args.token_name!r} already exists (not regenerated)")
        if not created_user and not token:
            print("  Bootstrap already complete — nothing to do.")
    return 0


# ---------------------------------------------------------------------------
# slurm
# ---------------------------------------------------------------------------


def _cmd_slurm_accounts(args: argparse.Namespace, _store: UserStore) -> int:
    """List Slurm accounts via ``sacctmgr``.

    Read-only convenience; requires ``sacctmgr`` to be on PATH and the caller
    to have permission to query the Slurm accounting DB.
    """
    import shutil
    import subprocess

    if shutil.which("sacctmgr") is None:
        print("sacctmgr not found on PATH", file=sys.stderr)
        return 2
    cmd = ["sacctmgr", "show", "account", "-P", "format=Account,Description,Organization"]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"sacctmgr failed: {exc}", file=sys.stderr)
        return 2
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    sys.stdout.write(result.stdout)
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

_GENERATE_JWT_DESCRIPTION = (
    "Create a signed HS256 JWT for r4 CLI / Gateway HTTP authentication. "
    "The token carries sub, project_id, roles, and exp claims. "
    "Use --projects to grant access to multiple projects in a single token "
    "(the first project becomes the default project_id). "
    "Use it with 'r4 auth login --api-key <token>'. "
    "For shared user sessions prefer OIDC/Okta. "
    "The signing secret is read from --secret, $JWT_SECRET, the persisted Gateway secret, "
    "or the built-in dev default. "
    "NOTE: this token is NOT for umrs-client --admin-token — "
    "for that, use 'generate' instead."
)


def _add_user_parsers(user_sub: argparse._SubParsersAction) -> None:
    add_p = user_sub.add_parser(
        "add",
        help="Register a new user",
        description="Register a general UMRS user in the server-side user_files store.",
        epilog=(
            "Example:\n"
            "  umrs-server admin user add alice --fullname \"Alice Smith\" "
            "--email alice@example.org --role user --project proj-alpha"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_p.add_argument("username")
    add_p.add_argument("--fullname", default="")
    add_p.add_argument("--email", default="")
    add_p.add_argument("--role", choices=VALID_ROLES, default="user")
    add_p.add_argument(
        "--slurm-account",
        dest="slurm_account",
        default=None,
        help="Slurm accounting name used for #SBATCH --account when submitting",
    )
    add_p.add_argument(
        "--project",
        action="append",
        default=None,
        help="Project ID to add (may be repeated)",
    )
    add_p.add_argument("--json", action="store_true")
    add_p.add_argument(
        "--client-template",
        dest="client_template",
        default=None,
        metavar="PATH",
        help="Path to write the client config template (default: <user_files>/<username>.client.json)",
    )
    add_p.add_argument(
        "--no-client-template",
        dest="no_client_template",
        action="store_true",
        help="Do not auto-generate a client config template file",
    )
    add_p.add_argument(
        "--issue-token",
        dest="issue_token",
        action="store_true",
        help="Issue an onboarding JWT and print dual-path login one-liner",
    )
    add_p.add_argument(
        "--token-days",
        dest="token_days",
        type=int,
        default=30,
        help="Onboarding token validity in days (default: 30)",
    )
    add_p.set_defaults(func=_cmd_user_add)

    list_p = user_sub.add_parser(
        "list",
        help="List registered users",
        description="List registered UMRS users from the server-side user_files store.",
        epilog=(
            "Examples:\n"
            "  umrs-server admin user list\n"
            "  umrs-server admin user list --role user\n"
            "  umrs-server admin user list --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_p.add_argument("--role", choices=VALID_ROLES)
    list_p.add_argument("--disabled-only", action="store_true")
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_user_list)

    show_p = user_sub.add_parser("show", help="Show a user record as JSON")
    show_p.add_argument("username")
    show_p.set_defaults(func=_cmd_user_show)

    del_p = user_sub.add_parser("delete", help="Soft-delete (disable) a user; --purge to remove file")
    del_p.add_argument("username")
    del_p.add_argument("--purge", action="store_true", help="Remove the JSON file")
    del_p.add_argument("--deprovision", action="store_true",
                       help="Also remove S3 buckets, Postgres DBs, and MinIO IAM user (requires --purge)")
    del_p.add_argument("-y", "--yes", action="store_true", help="Assume yes on prompts")
    del_p.set_defaults(func=_cmd_user_delete)

    disable_p = user_sub.add_parser("disable", help="Disable a user")
    disable_p.add_argument("username")
    disable_p.set_defaults(func=_cmd_user_disable)

    enable_p = user_sub.add_parser("enable", help="Enable a user")
    enable_p.add_argument("username")
    enable_p.set_defaults(func=_cmd_user_enable)

    role_p = user_sub.add_parser("set-role", help="Change a user's role")
    role_p.add_argument("username")
    role_p.add_argument("role", choices=VALID_ROLES)
    role_p.set_defaults(func=_cmd_user_set_role)

    slurm_p = user_sub.add_parser(
        "set-slurm-account",
        help="Set the Slurm accounting name used for this user's job submissions",
    )
    slurm_p.add_argument("username")
    slurm_p.add_argument(
        "account",
        nargs="?",
        default=None,
        help="Slurm account (omit to clear)",
    )
    slurm_p.set_defaults(func=_cmd_user_set_slurm_account)

    add_proj_p = user_sub.add_parser("add-to-project", help="Add a user to a project")
    add_proj_p.add_argument("username")
    add_proj_p.add_argument("project_id")
    add_proj_p.set_defaults(func=_cmd_user_add_project)

    rm_proj_p = user_sub.add_parser(
        "remove-from-project", help="Remove a user from a project"
    )
    rm_proj_p.add_argument("username")
    rm_proj_p.add_argument("project_id")
    rm_proj_p.set_defaults(func=_cmd_user_remove_project)

    tpl_p = user_sub.add_parser(
        "generate-template",
        help="Generate a template user.json to give to a new client",
        description=(
            "Generate a ready-to-use user config JSON template. "
            "The file can be sent to the new user who then imports it with:\n"
            "  umrs-client config init --template <file> --username <username>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tpl_p.add_argument("username", help="Username for the template")
    tpl_p.add_argument("--fullname", default="", help="Full name of the user")
    tpl_p.add_argument(
        "-o", "--output-file", default=None,
        help="Output file path (default: user_<username>.json in current directory)",
    )
    tpl_p.add_argument("--force", action="store_true", help="Overwrite existing file")
    tpl_p.add_argument("--json", action="store_true", help="Print template as JSON to stdout instead of writing a file")
    tpl_p.set_defaults(func=_cmd_user_generate_template)


def _add_token_parsers(token_sub: argparse._SubParsersAction) -> None:
    gen_p = token_sub.add_parser(
        "generate",
        help="Generate an opaque admin token for umrs-client --admin-token (ZMQ control plane)",
        description=(
            "Generate (or regenerate) a named opaque admin token stored in admin_tokens.json. "
            "This token is used with 'umrs-client --admin-token' or $UMRS_ADMIN_TOKEN for "
            "privileged ZMQ control-plane operations (reload, recompute, monitor, etc.). "
            "It is NOT used for r4 CLI / Gateway HTTP authentication — for that, use 'generate-jwt'."
        ),
    )
    gen_p.add_argument("name")
    gen_p.add_argument("--description", default="")
    gen_p.add_argument("--json", action="store_true")
    gen_p.set_defaults(func=_cmd_token_generate)

    jwt_p = token_sub.add_parser(
        "generate-jwt",
        help="Generate a project-scoped JWT for r4 CLI / Gateway HTTP authentication",
        description=_GENERATE_JWT_DESCRIPTION,
    )
    jwt_p.add_argument(
        "--user", "-u", default="admin",
        help="User identity (sub claim). Default: admin",
    )
    jwt_p.add_argument(
        "--email", "-e", default=None,
        help="Email claim. Default: <user>@dev.local",
    )
    jwt_p.add_argument(
        "--project", "-p", default="default",
        help="Primary project ID claim. Default: default",
    )
    jwt_p.add_argument(
        "--projects", nargs="+", default=None,
        metavar="ID",
        help=(
            "Grant access to multiple projects. The first ID becomes "
            "the default project_id; the full list is stored in "
            "project_memberships. Overrides --project when given."
        ),
    )
    jwt_p.add_argument(
        "--roles", "-r", nargs="*", default=None,
        help="Roles list. Default: admin",
    )
    jwt_p.add_argument(
        "--days", "-d", type=int, default=30,
        help="Token validity in days. Default: 30",
    )
    jwt_p.add_argument(
        "--secret", "-s", default=None,
        help="JWT signing secret. Default: $JWT_SECRET, Gateway secret, or dev default",
    )
    jwt_p.add_argument("--json", action="store_true")
    jwt_p.set_defaults(func=_cmd_token_generate_jwt)

    list_p = token_sub.add_parser("list", help="List admin tokens (metadata only)")
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_token_list)

    rev_p = token_sub.add_parser("revoke", help="Revoke an admin token by name")
    rev_p.add_argument("name")
    rev_p.add_argument("-y", "--yes", action="store_true")
    rev_p.set_defaults(func=_cmd_token_revoke)


def _add_slurm_parsers(slurm_sub: argparse._SubParsersAction) -> None:
    acc_p = slurm_sub.add_parser(
        "accounts", help="List Slurm accounts via sacctmgr (read-only)"
    )
    acc_p.set_defaults(func=_cmd_slurm_accounts)


def _cmd_reload(args: argparse.Namespace, _store: UserStore) -> int:
    """Explicitly send SIGHUP to the daemon to hot-reload user records."""
    pid_file = getattr(args, "pid_file", None) or resolve_pid_file_path()
    result = signal_reload(pid_file)
    if getattr(args, "json", False):
        _print_json(result.to_dict())
    else:
        status = "ok" if result.sent else "skipped"
        print(f"{status}: {result.reason}")
    return 0 if result.sent else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umrs-server admin",
        description=(
            "Offline admin CLI for UMRS user records, tokens, and Slurm "
            "account assignment. Manipulates user_files/*.json and "
            "admin_tokens.json directly — no running daemon required."
        ),
        epilog=(
            "Common onboarding flow:\n"
            "  1. Register a user:\n"
            "       umrs-server admin user add <username> --fullname \"Your Name\" --role user\n"
            "  2. List registered users:\n"
            "       umrs-server admin user list\n"
            "  3. Show one user record:\n"
            "       umrs-server admin user show <username>\n"
            "  4. Ask the user to initialize the client side:\n"
            "       umrs-client config init --username <username>\n\n"
            "Shell completion:\n"
            "  eval \"$(umrs-server completion bash)\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user-files-dir",
        dest="user_files_dir",
        default=None,
        help="Override the user_files directory (defaults to resolve_server_user_files_dir())",
    )
    parser.add_argument(
        "--pid-file",
        dest="pid_file",
        default=None,
        help="Override the daemon PID file path (defaults to UMRS_SERVER_PID_FILE "
        "or <cfg_dir>/umrs-server.pid)",
    )
    parser.add_argument(
        "--no-reload",
        dest="no_reload",
        action="store_true",
        help="Skip the SIGHUP auto-reload after mutating user operations.",
    )

    sub = parser.add_subparsers(dest="group", required=True)

    user_p = sub.add_parser(
        "user",
        help="Manage registered users",
        description=(
            "Create, list, inspect, disable, and update server-side UMRS user "
            "records. Use this command on the server/operator side; "
            "umrs-client does not perform admin actions."
        ),
        epilog=(
            "Most common commands:\n"
            "  umrs-server admin user add <username> --fullname \"Your Name\" --role user\n"
            "  umrs-server admin user list\n"
            "  umrs-server admin user show <username>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_user_parsers(user_p.add_subparsers(dest="action", required=True))

    token_p = sub.add_parser("token", help="Manage admin API tokens")
    _add_token_parsers(token_p.add_subparsers(dest="action", required=True))

    slurm_p = sub.add_parser("slurm", help="Slurm helpers (read-only)")
    _add_slurm_parsers(slurm_p.add_subparsers(dest="action", required=True))

    config_p = sub.add_parser(
        "config",
        help="Show which config files the server reads",
        description=(
            "Display the resolved server configuration directory, server JSON file, "
            "and user_files directory. Useful for verifying the environment before "
            "starting the daemon."
        ),
    )
    config_sub = config_p.add_subparsers(dest="action", required=True)
    cfg_show_p = config_sub.add_parser("show", help="Show active server config paths")
    cfg_show_p.add_argument("--json", action="store_true")
    cfg_show_p.set_defaults(func=_cmd_config_show)

    boot_p = sub.add_parser(
        "bootstrap",
        help="First-run idempotent setup (create admin user + initial token)",
    )
    boot_p.add_argument("--admin-user", default="admin", dest="admin_user")
    boot_p.add_argument("--fullname", default="")
    boot_p.add_argument("--email", default="")
    boot_p.add_argument("--token-name", default="bootstrap", dest="token_name")
    boot_p.add_argument("--json", action="store_true")
    boot_p.set_defaults(func=_cmd_bootstrap)

    reload_p = sub.add_parser(
        "reload",
        help="Send SIGHUP to the running daemon so it hot-reloads user records",
    )
    reload_p.add_argument("--json", action="store_true")
    reload_p.set_defaults(func=_cmd_reload)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = UserStore(user_files_dir=args.user_files_dir)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    handler = cast(Callable[[argparse.Namespace, UserStore], int], func)
    try:
        rc = int(handler(args, store) or 0)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if rc == 0:
        _maybe_signal_reload(args)
    return rc


__all__ = ["build_parser", "main"]
