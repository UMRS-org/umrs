"""``umrs-server admin`` argparse dispatcher."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

from umrs.admin.user_store import (
    UserRecord,
    UserStore,
    UserStoreError,
    VALID_ROLES,
)
from umrs.admin.daemon_signal import (
    resolve_pid_file_path,
    signal_reload,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


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


def _record_row(rec: UserRecord) -> Dict[str, Any]:
    return {
        "username": rec.username,
        "role": rec.role,
        "slurm_account": rec.slurm_account or "",
        "projects": ",".join(rec.projects),
        "disabled": rec.disabled,
        "email": rec.email,
    }


def _print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("(no users)")
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
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(rec.to_json())
    else:
        print(f"created user {rec.username} at {store.dir / (rec.username + '.json')}")
    return 0


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
    """Resolve the JWT signing secret from CLI flag, env, or default."""
    import os

    secret = getattr(args, "secret", None)
    if secret:
        return secret
    env_secret = os.environ.get("JWT_SECRET")
    if env_secret:
        return env_secret
    print(
        "WARNING: JWT_SECRET is not set — using insecure default. "
        "Set JWT_SECRET or pass --secret for production use.",
        file=sys.stderr,
    )
    return "dev-secret-change-in-production"


def _build_jwt_payload(
    user: str,
    email: str,
    project: str,
    roles: List[str],
    days: int,
) -> Dict[str, Any]:
    """Build the JWT claims payload."""
    import time

    now = int(time.time())
    return {
        "sub": user,
        "email": email,
        "project_id": project,
        "roles": roles,
        "iat": now,
        "exp": now + (days * 86400),
    }


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
    payload = _build_jwt_payload(args.user, email, args.project, roles, args.days)
    token = jose_jwt.encode(payload, secret, algorithm="HS256")

    using_default_user = args.user == "admin"
    using_default_project = args.project == "default"

    if args.json:
        _print_json({
            "token": token,
            "claims": payload,
            "expires_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(payload["exp"])
            ),
        })
    else:
        print(f"Generated JWT for user '{payload['sub']}' in project '{payload['project_id']}'")
        print(f"token: {token}")
        print(f"  sub:        {payload['sub']}")
        print(f"  email:      {payload['email']}")
        print(f"  project_id: {payload['project_id']}")
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
        print(f"  Use with: r4 auth login --api-key <token>")
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
        print(f"{'name':<30}  {'created_at':<25}  description")
        print("-" * 80)
        for t in tokens:
            print(
                f"{t.get('name',''):<30}  "
                f"{t.get('created_at',''):<25}  "
                f"{t.get('description','')}"
            )
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
    token: Optional[str] = None
    if not any(t.get("name") == args.token_name for t in existing):
        token = mgr.generate_token(
            args.token_name, description="Bootstrap admin token"
        )

    result: Dict[str, Any] = {
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
    "The token carries sub, project_id, roles, and exp claims and is "
    "intentionally scoped to one user and project. "
    "Use it with 'r4 auth login --api-key <token>'. "
    "For shared user sessions prefer OIDC/Okta. "
    "The signing secret is read from --secret, $JWT_SECRET, or the built-in dev default. "
    "NOTE: this token is NOT for umrs-client --admin-token — "
    "for that, use 'generate' instead."
)


def _add_user_parsers(user_sub: argparse._SubParsersAction) -> None:
    add_p = user_sub.add_parser("add", help="Create a user record")
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
    add_p.set_defaults(func=_cmd_user_add)

    list_p = user_sub.add_parser("list", help="List user records")
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
        help="Project ID claim / active scope for this token. Default: default",
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
        help="JWT signing secret. Default: $JWT_SECRET or dev default",
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

    user_p = sub.add_parser("user", help="Manage user records")
    _add_user_parsers(user_p.add_subparsers(dest="action", required=True))

    token_p = sub.add_parser("token", help="Manage admin API tokens")
    _add_token_parsers(token_p.add_subparsers(dest="action", required=True))

    slurm_p = sub.add_parser("slurm", help="Slurm helpers (read-only)")
    _add_slurm_parsers(slurm_p.add_subparsers(dest="action", required=True))

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


def main(argv: Optional[Sequence[str]] = None) -> int:
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
