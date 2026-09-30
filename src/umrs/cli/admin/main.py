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


def _resolve_slurm_account_requirement() -> tuple[bool, str | None]:
    """Return ``(require_user_slurm_account, site_master_account)``.

    Best-effort: when the site config cannot be loaded (e.g. running offline
    without ``UMRS_SITE_CONFIG``), assume per-user accounts are not required
    and no master account is known.
    """
    try:
        from umrs.orchestrator.site_config import load_site_config

        site = load_site_config()
        require = bool(getattr(site.scheduler, "require_user_slurm_account", False))
        master = getattr(site.scheduler, "account", "") or None
        return require, master
    except Exception:  # noqa: BLE001 — admin CLI degrades best-effort, named in output
        return False, None


def _record_row(
    rec: UserRecord, *, require_acct: bool, master_account: str | None
) -> dict[str, Any]:
    account = rec.slurm_account or ""
    if not account:
        if require_acct:
            account = "(REQUIRED — not set)"
        elif master_account:
            account = f"(not required → {master_account})"
        else:
            account = "(not required)"
    return {
        "username": rec.username,
        "role": rec.role,
        "slurm_account": account,
        "projects": ",".join(rec.projects),
        "disabled": rec.disabled,
        "email": rec.email,
    }


def _print_table(
    rows: list[dict[str, Any]],
    *,
    require_acct: bool = False,
    master_account: str | None = None,
) -> None:
    if not rows:
        print("(no users)")
        print(
            "hint: register one with `umrs-server admin user add <username> "
            '--fullname "Your Name" --role user`'
        )
        return
    columns = ["username", "role", "slurm_account", "projects", "disabled", "email"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for row in rows:
        print("  ".join(str(row.get(c, "")).ljust(widths[c]) for c in columns))

    if require_acct:
        print(
            "\nnote: per-user Slurm accounts are REQUIRED "
            "(scheduler.require_user_slurm_account=true); "
            "users without one cannot provision endpoints."
        )
    elif master_account:
        print(
            "\nnote: per-user Slurm accounts are optional "
            "(scheduler.require_user_slurm_account=false); "
            f"blank entries fall back to the site account '{master_account}'."
        )
    else:
        print(
            "\nnote: per-user Slurm accounts are optional "
            "(scheduler.require_user_slurm_account=false); "
            "blank entries fall back to the site master Slurm account."
        )


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


def _maybe_register_in_server_config(
    args: argparse.Namespace, store: UserStore, username: str
) -> None:
    """Register the user JSON in conf_umrs.json's ``users`` list.

    Only acts when ``--config`` was given. Adds the path if not already
    present, using a relative path when the user file is under the same
    directory tree as the config file.
    """
    import os

    server_config = getattr(args, "server_config", None)
    if not server_config:
        return
    config_path = os.path.abspath(os.path.expanduser(server_config))
    if not os.path.isfile(config_path):
        return

    user_file = str(store.dir / f"{username}.json")
    config_dir = os.path.dirname(config_path)

    # Prefer a relative path when the user file is under the config directory.
    try:
        rel = os.path.relpath(user_file, config_dir)
        if not rel.startswith(".."):
            # Normalise to ./ prefix for consistency with existing entries.
            entry = f"./{rel}" if not rel.startswith("./") else rel
        else:
            entry = user_file
    except ValueError:
        entry = user_file

    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"warning: could not read {config_path} to register user: {exc}",
            file=sys.stderr,
        )
        return

    users_list = cfg.setdefault("users", [])

    # Check both the entry we'd add and the absolute path to avoid duplicates.
    existing_abs = set()
    for u in users_list:
        try:
            existing_abs.add(
                os.path.abspath(os.path.join(config_dir, os.path.expanduser(u)))
            )
        except Exception:  # noqa: BLE001, S110 — admin CLI degrades best-effort, named in output
            pass
    if os.path.abspath(user_file) in existing_abs:
        return

    users_list.append(entry)

    try:
        tmp_path = config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        try:
            os.replace(tmp_path, config_path)
        except OSError:
            # Bind-mounted files block rename; fall back to in-place write.
            with open(tmp_path, encoding="utf-8") as src:
                content = src.read()
            os.unlink(tmp_path)
            with open(config_path, "w", encoding="utf-8") as dst:
                dst.write(content)
        print(f"registered {entry} in {os.path.basename(config_path)}")
    except OSError as exc:
        print(
            f"warning: could not update {config_path}: {exc}",
            file=sys.stderr,
        )


def _load_server_conf_users(config_path: str) -> "tuple[dict, list] | None":
    """(cfg, users) from conf_umrs.json; None when unreadable (warned)."""
    try:
        with open(config_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"warning: could not read {config_path} to unregister user: {exc}",
            file=sys.stderr,
        )
        return None
    return cfg, list(cfg.get("users") or [])


def _drop_user_from_conf_list(users_list: list, config_dir: str, user_file_abs: str) -> "tuple[list, int]":
    """(kept, removed_count) — absolute-path match covers relative forms too."""
    import os

    kept: list = []
    removed = 0
    for entry in users_list:
        try:
            if (
                os.path.abspath(os.path.join(config_dir, os.path.expanduser(entry)))
                == user_file_abs
            ):
                removed += 1
                continue
        except Exception:  # noqa: BLE001, S110 — entry form unknown to us, keep it
            pass
        kept.append(entry)
    return kept, removed


def _write_server_conf_atomic(config_path: str, cfg: dict, username: str) -> None:
    """tmp+rename write; bind-mounted files fall back to in-place."""
    import os

    tmp_path = config_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        try:
            os.replace(tmp_path, config_path)
        except OSError:
            # Bind-mounted files block rename; fall back to in-place write.
            with open(tmp_path, encoding="utf-8") as src:
                content = src.read()
            os.unlink(tmp_path)
            with open(config_path, "w", encoding="utf-8") as dst:
                dst.write(content)
        print(f"unregistered {username} from {os.path.basename(config_path)}")
    except OSError as exc:
        print(
            f"warning: could not update {config_path}: {exc}",
            file=sys.stderr,
        )


def _maybe_unregister_from_server_config(
    args: argparse.Namespace, store: UserStore, username: str
) -> None:
    """Drop the user JSON from conf_umrs.json's ``users`` list.

    Mirror of :func:`_maybe_register_in_server_config` for the delete path
    (2026-09-30, batch-benchmark cleanup discovery): without it, deleted
    users re-provision on the daemon's next reload/restart because their
    stale conf entries still point at (now missing) user files.

    Matches by absolute path, so the register-side relative forms
    (``./user_files/<u>.json`` etc.) are covered as well.
    """
    import os

    server_config = getattr(args, "server_config", None)
    if not server_config:
        return
    config_path = os.path.abspath(os.path.expanduser(server_config))
    if not os.path.isfile(config_path):
        return

    loaded = _load_server_conf_users(config_path)
    if loaded is None:
        return
    cfg, users_list = loaded
    config_dir = os.path.dirname(config_path)
    user_file_abs = os.path.abspath(str(store.dir / f"{username}.json"))

    kept, removed = _drop_user_from_conf_list(users_list, config_dir, user_file_abs)
    if not removed:
        return
    cfg["users"] = kept
    _write_server_conf_atomic(config_path, cfg, username)


def _cmd_user_add(args: argparse.Namespace, store: UserStore) -> int:
    # When --site is given, scope the slurm account to that site.
    site_name = getattr(args, "site", None)
    create_slurm_account = args.slurm_account if not site_name else None
    oidc_subject = getattr(args, "oidc_subject", "") or ""

    # Include s3_configuration_file and default experience so the daemon can
    # provision MinIO buckets from the same JSON it reads for admin metadata.
    client_extras = _build_client_config_template(args.username, args.fullname or "")
    # Remove keys that are already first-class admin fields.
    for k in ("username", "fullname"):
        client_extras.pop(k, None)

    try:
        rec = store.create(
            args.username,
            fullname=args.fullname or "",
            email=args.email or "",
            role=args.role,
            slurm_account=create_slurm_account,
            projects=args.project or [],
            oidc_subject=oidc_subject,
            extras=client_extras,
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

    # Apply per-site slurm account after creation if --site was given.
    if site_name and args.slurm_account:
        try:
            rec = store.set_slurm_account_for_site(
                args.username,
                site_name,
                args.slurm_account,
            )
        except UserStoreError as exc:
            print(
                f"warning: user created but failed to set site account: {exc}",
                file=sys.stderr,
            )

    # Auto-generate a client config template alongside the admin record.
    _maybe_emit_template(args, store, rec)

    # Warn if Slurm account is not set (required for endpoint provisioning
    # only when the site enforces per-user accounts).
    if not rec.slurm_account and not rec.slurm_accounts:
        require_acct, master_account = _resolve_slurm_account_requirement()
        if require_acct:
            print(
                f"warning: no Slurm account set for {rec.username}. "
                f"This site requires per-user Slurm accounts "
                f"(scheduler.require_user_slurm_account=true); endpoint "
                f"provisioning (r4 endpoint up) will fail until one is set.\n"
                f"  hint: umrs-server admin user set-slurm-account {rec.username} <ACCOUNT>",
                file=sys.stderr,
            )
        else:
            fallback = f" '{master_account}'" if master_account else ""
            print(
                f"note: no Slurm account set for {rec.username}. "
                f"This site does not require per-user accounts "
                f"(scheduler.require_user_slurm_account=false); jobs fall back "
                f"to the site master Slurm account{fallback}.\n"
                f"  to assign one anyway: umrs-server admin user set-slurm-account {rec.username} <ACCOUNT>",
                file=sys.stderr,
            )

    if args.json:
        out = rec.to_json()
        tpl_path = _client_template_path(args, store, rec)
        if tpl_path and tpl_path.exists():
            out["client_template"] = str(tpl_path)
        _print_json(out)
    else:
        print(f"created user {rec.username} at {store.dir / (rec.username + '.json')}")

    # Auto-register in conf_umrs.json when --config is given.
    _maybe_register_in_server_config(args, store, rec.username)

    return 0


def _client_template_path(args, store, rec):
    from pathlib import Path

    if getattr(args, "no_client_template", False):
        return None
    if getattr(args, "client_template", None):
        return Path(args.client_template).expanduser()
    return store.dir / f"{rec.username}.client.json"


def _remove_client_template(store, username: str) -> None:
    """Remove the auto-generated .client.json if it exists."""
    client_path = store.dir / f"{username}.client.json"
    if client_path.is_file():
        client_path.unlink()
        print(f"  removed {client_path}")


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
                    gateway_url=getattr(args, "gateway_url", None) or None,
                    profile=getattr(args, "profile", None) or None,
                )
                from jose import jwt as jose_jwt

                token = jose_jwt.encode(payload, secret, algorithm="HS256")
                token_hint = token
                print()
                print(f"  Onboarding token (valid {getattr(args, 'token_days', 30)}d):")
                print(f"    {token}")
            except Exception as exc:  # noqa: BLE001 — admin CLI degrades best-effort, named in output
                print(
                    f"  warning: could not issue onboarding JWT: {exc}", file=sys.stderr
                )

        print()
        print("  Onboarding steps for the new user:")
        if token_hint:
            gw_hint = payload.get("gateway_url", "")
            if gw_hint:
                print(
                    "  1. (recommended) r4 via Gateway HTTP (gateway_url embedded — no flags needed):"
                )
                print(f"       r4 auth login --api-key {token_hint}")
            else:
                print("  1. (recommended) r4 via Gateway HTTP:")
                print(
                    f"       r4 auth login --gateway-url <GATEWAY_URL> --api-key {token_hint}"
                )
            print(
                "  2. (alternative) umrs direct ZMQ — requires CURVE keys if CURVE auth is enabled:"
            )
            print(f"       umrs login --host <HOST> --api-key {token_hint}")
        else:
            print(
                "  1. Generate a token for the user (embed --gateway-url for single-flag login):"
            )
            print(
                f"       umrs-server admin token generate-jwt --user {rec.username} --gateway-url <GATEWAY_URL>"
            )
            print("  2. Give the token to the user, who runs:")
            print("       r4 auth login --api-key <TOKEN>")
            print(
                "     (alternative) umrs direct ZMQ — requires CURVE keys if CURVE auth is enabled:"
            )
            print("       umrs login --host <HOST> --api-key <TOKEN>")
        if not rec.slurm_account and not rec.slurm_accounts:
            require_acct, _ = _resolve_slurm_account_requirement()
            if require_acct:
                print(
                    "  NOTE: set a Slurm account before the user can provision endpoints:"
                )
                print(
                    f"       umrs-server admin user set-slurm-account {rec.username} <ACCOUNT>"
                )
            else:
                print(
                    "  NOTE: per-user Slurm account is optional on this site; jobs use the site master account."
                )
                print(
                    f"       to assign one anyway: umrs-server admin user set-slurm-account {rec.username} <ACCOUNT>"
                )


def _pg_user_rows() -> list | None:
    """PG-lane read for the admin CLI (D-06 tail): this site's rows from
    the 0110 table (0113 venue scope) when the DSN is configured; None when not (the JSON lane answers,
    never an invented row)."""
    try:
        from umrs.admin.user_store import site_venue
        from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

        dsn, _ = _resolve_knowledge_dsn()
        if not dsn:
            return None
        import psycopg2

        from umrs.shared.pg_settings import pg_connect_kwargs

        conn = psycopg2.connect(dsn, **pg_connect_kwargs("admin-cli"))
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT username, fullname, email, role, disabled, slurm_account,"
                " slurm_accounts, projects, quota, oidc_subject, curve_public_key,"
                " curve_key_revoked_at, extras, created_at, last_updated"
                " FROM users WHERE venue = %s ORDER BY username",
                (site_venue(),),
            )
            return cur.fetchall()
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 — the JSON lane answers; PG is optional
        return None


def _filter_user_rows(rows, role: str | None, disabled_only: bool) -> list:
    """Apply the role/disabled filters shared by both listing lanes."""
    if role:
        rows = [r for r in rows if r[3] == role]
    if disabled_only:
        rows = [r for r in rows if r[4]]
    return rows


def _list_users_from_pg(args: argparse.Namespace) -> int | None:
    """D-06 tail: the PG lane's rendering. ``None`` = lane-less (the JSON
    lane answers; never an invented row)."""
    rows = _pg_user_rows()
    if rows is None:
        return None
    rows = _filter_user_rows(
        rows, getattr(args, "role", None), getattr(args, "disabled_only", False)
    )
    if args.json:
        import json as _json

        _print_json(
            [
                {
                    "username": r[0],
                    "fullname": r[1],
                    "email": r[2],
                    "role": r[3],
                    "disabled": r[4],
                    "slurm_account": r[5],
                    "projects": r[7]
                    if isinstance(r[7], list)
                    else _json.loads(r[7] or "[]"),
                    "oidc_subject": r[9],
                    "created_at": r[13],
                    "last_updated": r[14],
                }
                for r in rows
            ]
        )
    else:
        for r in rows:
            dis = " (disabled)" if r[4] else ""
            print(f"  {r[0]:20s}  {r[3]:8s}{dis}  {r[1]}")
        from umrs.admin.user_store import site_venue

        print(f"\n{len(rows)} row(s) from Postgres (0110, venue {site_venue()!r})")
    return 0


def _filter_user_records(records, role: str | None, disabled_only: bool) -> list:
    """Role/disabled filters for the JSON lane's UserRecord rows."""
    if role:
        records = [r for r in records if r.role == role]
    if disabled_only:
        records = [r for r in records if r.disabled]
    return records


def _list_users_from_json(args: argparse.Namespace, store: UserStore) -> int:
    """The offline JSON lane — byte-identical pre-0110 behavior."""
    records = _filter_user_records(store.list(), args.role, args.disabled_only)
    require_acct, master_account = _resolve_slurm_account_requirement()
    rows = [
        _record_row(r, require_acct=require_acct, master_account=master_account)
        for r in records
    ]
    if args.json:
        _print_json([r.to_json() for r in records])
    else:
        _print_table(rows, require_acct=require_acct, master_account=master_account)
    return 0


def _cmd_user_list(args: argparse.Namespace, store: UserStore) -> int:
    if getattr(args, "from_pg", False):
        result = _list_users_from_pg(args)
        if result is None:
            print(
                "error: --from-pg requires the Postgres lane (knowledge DSN "
                "unconfigured or unreachable); the JSON lane answers without it.",
                file=sys.stderr,
            )
            return 2
        return result
    return _list_users_from_json(args, store)


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

        # UserRecord preserves unknown top-level config in extras, so the
        # deprovisioner can consume the same payload shape without reloading
        # the JSON file through a store-private path helper.
        user_cfg = rec.to_json()

        # Resolve admin S3 config from the server config file.
        import os

        from umrs.shared.config import (
            get_default_json_server_file,
            get_resolved_umrs_cfg_dir,
        )

        server_json = os.path.join(
            get_resolved_umrs_cfg_dir(), get_default_json_server_file()
        )
        admin_s3_cfg = None
        if os.path.isfile(server_json):
            with open(server_json, encoding="utf-8") as f:
                srv = json.load(f)
            admin_s3_cfg = srv.get("admin_s3_configuration_file")

        if not admin_s3_cfg:
            print(
                "warning: admin_s3_configuration_file not found in server config; "
                "S3/MinIO cleanup will be skipped.",
                file=sys.stderr,
            )

        from umrs.server.provisioning import deprovision_user

        deprovision_user(user_cfg, admin_s3_cfg or "")

        path = store.delete(args.username)
        _remove_client_template(store, args.username)
        _maybe_unregister_from_server_config(args, store, args.username)
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
        _remove_client_template(store, args.username)
        _maybe_unregister_from_server_config(args, store, args.username)
        print(f"deleted {path}")
        return 0
    # Soft delete: mark disabled. Preserve job/audit history (Decision 1).
    store.set_disabled(args.username, True)
    print(f"soft-deleted (disabled) user {rec.username}")
    print("Use `--purge` to remove the JSON file and revoke filesystem presence.")
    print(
        "Use `--purge --deprovision` to also remove S3 buckets, Postgres databases, and MinIO IAM user."
    )
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
        if args.site:
            rec = store.set_slurm_account_for_site(
                args.username,
                args.site,
                args.account,
            )
            label = f"slurm_accounts[{args.site}]"
            value = rec.slurm_accounts.get(args.site, "(unset)")
        else:
            rec = store.set_slurm_account(args.username, args.account)
            label = "slurm_account"
            value = rec.slurm_account or "(unset)"
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"set {label}={value} on {args.username}")
    return 0


def _cmd_user_set_oidc_subject(args: argparse.Namespace, store: UserStore) -> int:
    try:
        rec = store.set_oidc_subject(args.username, args.subject or "")
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    value = rec.oidc_subject or "(unset)"
    print(f"set oidc_subject={value} on {args.username}")
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
        print(
            f"error: {out_path} already exists (use --force to overwrite)",
            file=sys.stderr,
        )
        return 2

    _write_client_template(username, fullname, out_path)

    print(f"Template written to {out_path}")
    print()
    print("  Onboarding steps for the user:")
    print(f"  1. Send '{out_path.name}' to the user (or copy via scp)")
    print("  2. On their local machine:")
    print(
        f"       r4 config init --gateway-url <GATEWAY_URL> --umrs-username {username}"
    )
    print("  3. Config is stored locally at ~/.config/r4/profiles.json")
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
        f"${UMRS_CFG_ENV_VAR_NAME}"
        if os.environ.get(UMRS_CFG_ENV_VAR_NAME)
        else "default"
    )

    # When --config is given, use that as the server JSON; otherwise fallback
    server_config_path = getattr(args, "server_config", None)
    if server_config_path:
        server_json = os.path.abspath(os.path.expanduser(server_config_path))
    else:
        server_json = os.path.join(cfg_dir, get_default_json_server_file())
    server_json_exists = os.path.isfile(server_json)

    # Resolve user_files_dir from --config or --user-files-dir or defaults
    explicit_dir = getattr(args, "user_files_dir", None)
    if explicit_dir:
        user_files_dir = os.path.abspath(os.path.expanduser(explicit_dir))
        user_files_source = "--user-files-dir"
    elif server_config_path and server_json_exists:
        try:
            with open(server_json, encoding="utf-8") as fh:
                server_cfg = json.load(fh)
            user_files_dir = resolve_server_user_files_dir(
                config_file=server_json, server_cfg=server_cfg
            )
            user_files_source = f"--config ({os.path.basename(server_json)})"
        except Exception:  # noqa: BLE001 — admin CLI degrades best-effort, named in output
            user_files_dir = resolve_server_user_files_dir()
            user_files_source = "default"
    else:
        user_files_dir = resolve_server_user_files_dir()
        user_files_source = (
            f"${UMRS_USER_FILES_ENV_VAR_NAME}"
            if os.environ.get(UMRS_USER_FILES_ENV_VAR_NAME)
            else "default"
        )
    user_files_exists = os.path.isdir(user_files_dir)

    if args.json:
        _print_json(
            {
                "config_dir": cfg_dir,
                "config_dir_source": cfg_dir_source,
                "server_config": server_json,
                "server_config_exists": server_json_exists,
                "user_files_dir": user_files_dir,
                "user_files_dir_source": user_files_source,
                "user_files_dir_exists": user_files_exists,
            }
        )
    else:
        ok = "\u2713"
        no = "\u2717"
        deploy_dir = getattr(args, "deploy_dir", None)
        if deploy_dir:
            print(f"Deploy directory:  {os.path.abspath(deploy_dir)}")
            site_yaml = os.path.join(os.path.abspath(deploy_dir), "site.yaml")
            print(f"  site.yaml:       {ok if os.path.isfile(site_yaml) else no}")
            env_file = os.path.join(os.path.abspath(deploy_dir), ".env")
            print(f"  .env:            {ok if os.path.isfile(env_file) else no}")
            print()
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
        print(
            f"{action} admin token '{args.name}' (for umrs --admin-token / $UMRS_ADMIN_TOKEN)"
        )
        print(f"token: {token}")
        print("  (this is the only time the raw token will be shown)")
        print()
        print("  Use with: umrs --admin-token <token> <command>")
        print("         or: export UMRS_ADMIN_TOKEN=<token>")
        print()
        print("  NOTE: this token is for the umrs ZMQ control plane only.")
        print(
            "        To authenticate r4/Gateway, use 'umrs-server admin token generate-jwt' instead."
        )
    return 0


def _resolve_jwt_secret(args: argparse.Namespace) -> str:
    """Resolve the JWT signing secret from CLI flag, env, Gateway secret, or default."""
    # Check --secret CLI flag first (admin-only override).
    secret = getattr(args, "secret", None)
    if secret:
        return secret
    # Delegate to shared resolution logic.
    from umrs.shared.auth import resolve_jwt_secret

    resolved = resolve_jwt_secret()
    if resolved == "dev-secret-change-in-production":
        print(
            "warning: using insecure dev-default JWT secret. "
            "Set $JWT_SECRET or deploy ~/.config/umrs/gateway/jwt.secret "
            "for production use.",
            file=sys.stderr,
        )
    return resolved


def _build_jwt_payload(
    user: str,
    email: str,
    project: str,
    roles: list[str],
    days: int,
    *,
    projects: list[str] | None = None,
    gateway_url: str | None = None,
    profile: str | None = None,
) -> dict[str, Any]:
    """Build the JWT claims payload.

    When *projects* is supplied the token carries a
    ``project_memberships`` list so a single JWT can be used across
    multiple projects.  ``project_id`` is always emitted for backward
    compatibility (set to the first entry of *projects* when given).

    Optional *gateway_url* and *profile* embed client-connection hints
    so ``r4 auth login --api-key <token>`` can auto-configure everything.
    OIDC tokens naturally lack these claims — the client falls through
    to the ``--gateway-url`` flag as before.
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
    if gateway_url:
        payload["gateway_url"] = gateway_url
    if profile:
        payload["profile"] = profile
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
    # Look up the user record for role/email defaults.
    _user_rec: UserRecord | None = None
    try:
        _user_rec = _store.get(args.user)
    except UserStoreError:
        pass
    if _user_rec and _user_rec.disabled:
        print(
            f"error: user {args.user!r} is disabled; refusing to generate token",
            file=sys.stderr,
        )
        return 2
    email = args.email or (
        _user_rec.email if _user_rec and _user_rec.email else f"{args.user}@dev.local"
    )
    if args.roles:
        roles = args.roles
    else:
        roles = [_user_rec.role] if _user_rec else ["user"]
    projects = getattr(args, "projects", None) or None
    payload = _build_jwt_payload(
        args.user,
        email,
        args.project,
        roles,
        args.days,
        projects=projects,
        gateway_url=getattr(args, "gateway_url", None),
        profile=getattr(args, "profile", None),
    )
    token = jose_jwt.encode(payload, secret, algorithm="HS256")

    using_default_user = args.user == "admin"
    using_default_project = args.project == "default" and not projects

    if args.json:
        _print_json(
            {
                "token": token,
                "claims": payload,
                "expires_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(payload["exp"])
                ),
            }
        )
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
        if payload.get("gateway_url"):
            print(f"  gateway:    {payload['gateway_url']}")
        if payload.get("profile"):
            print(f"  profile:    {payload['profile']}")
        print(
            f"  expires:    {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(payload['exp']))}"
        )
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
            print(
                "  ".join(
                    str(row.get(column, "")).ljust(widths[column]) for column in columns
                )
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
    token: str | None = None
    if not any(t.get("name") == args.token_name for t in existing):
        token = mgr.generate_token(args.token_name, description="Bootstrap admin token")

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
    cmd = [
        "sacctmgr",
        "show",
        "account",
        "-P",
        "format=Account,Description,Organization",
    ]
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
    "Use --gateway-url and --profile to embed client-connection hints "
    "so 'r4 auth login --api-key <token>' auto-configures everything "
    "(OIDC tokens lack these claims \u2014 the client falls through to flags). "
    "For shared user sessions prefer OIDC/Okta. "
    "The signing secret is read from --secret, $JWT_SECRET, the persisted Gateway secret, "
    "or the built-in dev default. "
    "NOTE: this token is NOT for umrs --admin-token — "
    "for that, use 'generate' instead."
)


def _add_user_parsers(user_sub: argparse._SubParsersAction) -> None:
    add_p = user_sub.add_parser(
        "add",
        help="Register a new user",
        description="Register a general UMRS user in the server-side user_files store.",
        epilog=(
            "Example:\n"
            '  umrs-server admin user add alice --fullname "Alice Smith" '
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
        "--site",
        default=None,
        help="Scope the --slurm-account to a specific site (e.g. rccs-front, ai4s)",
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
    add_p.add_argument(
        "--gateway-url",
        dest="gateway_url",
        default=None,
        help="Embed Gateway URL in the onboarding JWT so users can login with just --api-key",
    )
    add_p.add_argument(
        "--profile",
        dest="profile",
        default=None,
        help="Embed suggested profile name in the onboarding JWT",
    )
    add_p.add_argument(
        "--oidc-subject",
        dest="oidc_subject",
        default="",
        help=(
            "OIDC external identity subject (e.g. RIKEN ID '241723'). "
            "Enables OIDC login to resolve the IdP subject to this UMRS username."
        ),
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
    list_p.add_argument(
        "--from-pg",
        action="store_true",
        help="Read the Postgres lane (0110) instead of the JSON files",
    )
    list_p.set_defaults(func=_cmd_user_list)

    show_p = user_sub.add_parser("show", help="Show a user record as JSON")
    show_p.add_argument("username")
    show_p.set_defaults(func=_cmd_user_show)

    del_p = user_sub.add_parser(
        "delete", help="Soft-delete (disable) a user; --purge to remove file"
    )
    del_p.add_argument("username")
    del_p.add_argument("--purge", action="store_true", help="Remove the JSON file")
    del_p.add_argument(
        "--deprovision",
        action="store_true",
        help="Also remove S3 buckets, Postgres DBs, and MinIO IAM user (requires --purge)",
    )
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
    slurm_p.add_argument(
        "--site",
        default=None,
        help="Scope the account to a specific site (e.g. rccs-front, ai4s)",
    )
    slurm_p.set_defaults(func=_cmd_user_set_slurm_account)

    oidc_p = user_sub.add_parser(
        "set-oidc-subject",
        help="Set the OIDC external identity subject for a user",
        description=(
            "Map an OIDC identity provider subject (e.g. RIKEN ID '241723') "
            "to a UMRS username.  When a JWT arrives with this subject as "
            "the 'sub' claim, UMRS resolves it to the local user record."
        ),
    )
    oidc_p.add_argument("username")
    oidc_p.add_argument(
        "subject",
        nargs="?",
        default="",
        help="OIDC subject value (omit to clear)",
    )
    oidc_p.set_defaults(func=_cmd_user_set_oidc_subject)

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
            "  umrs config init --template <file> --username <username>"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tpl_p.add_argument("username", help="Username for the template")
    tpl_p.add_argument("--fullname", default="", help="Full name of the user")
    tpl_p.add_argument(
        "-o",
        "--output-file",
        default=None,
        help="Output file path (default: user_<username>.json in current directory)",
    )
    tpl_p.add_argument("--force", action="store_true", help="Overwrite existing file")
    tpl_p.add_argument(
        "--json",
        action="store_true",
        help="Print template as JSON to stdout instead of writing a file",
    )
    tpl_p.set_defaults(func=_cmd_user_generate_template)


def _add_token_parsers(token_sub: argparse._SubParsersAction) -> None:
    gen_p = token_sub.add_parser(
        "generate",
        help="Generate an opaque admin token for umrs --admin-token (ZMQ control plane)",
        description=(
            "Generate (or regenerate) a named opaque admin token stored in admin_tokens.json. "
            "This token is used with 'umrs --admin-token' or $UMRS_ADMIN_TOKEN for "
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
        "--user",
        "-u",
        default="admin",
        help="User identity (sub claim). Default: admin",
    )
    jwt_p.add_argument(
        "--email",
        "-e",
        default=None,
        help="Email claim. Default: <user>@dev.local",
    )
    jwt_p.add_argument(
        "--project",
        "-p",
        default="default",
        help="Primary project ID claim. Default: default",
    )
    jwt_p.add_argument(
        "--projects",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Grant access to multiple projects. The first ID becomes "
            "the default project_id; the full list is stored in "
            "project_memberships. Overrides --project when given."
        ),
    )
    jwt_p.add_argument(
        "--roles",
        "-r",
        nargs="*",
        default=None,
        help="Roles list. Default: read from user record (falls back to 'user')",
    )
    import os as _os

    _default_days = int(_os.environ.get("UMRS_JWT_DEFAULT_DAYS", "30"))
    jwt_p.add_argument(
        "--days",
        "-d",
        type=int,
        default=_default_days,
        help="Token validity in days. Default: 30 (or $UMRS_JWT_DEFAULT_DAYS)",
    )
    jwt_p.add_argument(
        "--secret",
        "-s",
        default=None,
        help="JWT signing secret. Default: $JWT_SECRET, Gateway secret, or dev default",
    )
    jwt_p.add_argument(
        "--gateway-url",
        default=None,
        help="Embed Gateway URL so 'r4 auth login --api-key <token>' auto-configures (ignored for OIDC)",
    )
    jwt_p.add_argument(
        "--profile",
        default=None,
        help="Embed suggested profile name in token (default: not set)",
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


# ---------------------------------------------------------------------------
# zmq subcommand handlers
# ---------------------------------------------------------------------------

_DEFAULT_CLIENT_KEYS_DIR = "~/.config/umrs/curve-keys/clients"
_DEFAULT_SERVER_KEYS_DIR = "~/.config/umrs/curve-keys/server"


def _cmd_zmq_revoke_curve_key(args: argparse.Namespace, store: UserStore) -> int:
    """D-03: revoke the account's long-lived CURVE key (ZAP refuses it at
    the next live keystore sync — one sweep tick, no reboot)."""
    from datetime import datetime, timezone

    record = store.update(
        args.name, curve_key_revoked_at=datetime.now(timezone.utc).isoformat()
    )
    print(
        f"CURVE key for '{args.name}' revoked at {record.curve_key_revoked_at} — "
        "the live ZAP keystore sync unregisters it within one sweep tick "
        "(no server reboot needed)."
    )
    return 0


def _stamp_account_key_on_record(store: UserStore, name: str, pub_path: str) -> bool:
    """D-03: a client key for a real account stamps the account record —
    the UMRS account becomes the key authority (revocation lives there).
    Opportunistic: a keygen for a non-account name is legal, just unstamped."""
    try:
        store.get(name)
    except Exception:  # noqa: BLE001 — a keygen for a non-account name is legal
        return False
    from umrs.core.zmq_security import load_public_key_z85

    pub_z85 = load_public_key_z85(pub_path)
    store.update(name, curve_public_key=pub_z85, curve_key_revoked_at="")
    print(
        f"Account '{name}' is now the key authority "
        f"(curve_public_key stamped; revoke with "
        f"'umrs-server admin user revoke-curve-key {name}')."
    )
    return True


def _cmd_zmq_keygen(args: argparse.Namespace, _store: UserStore) -> int:
    """Generate a CURVE keypair for a user (client) or the server."""
    from pathlib import Path

    from umrs.core.zmq_security import generate_curve_keypair

    name = args.name
    is_server = args.server
    out_dir = args.output_dir

    if out_dir:
        keys_dir = out_dir
    elif is_server:
        keys_dir = str(Path(_DEFAULT_SERVER_KEYS_DIR).expanduser())
    else:
        keys_dir = str(Path(_DEFAULT_CLIENT_KEYS_DIR).expanduser())

    pub_path, sec_path = generate_curve_keypair(keys_dir, name)

    if not is_server:
        _stamp_account_key_on_record(_store, name, pub_path)

    if args.json:
        _print_json(
            {
                "name": name,
                "type": "server" if is_server else "client",
                "public_key_file": pub_path,
                "secret_key_file": sec_path,
            }
        )
    else:
        kind = "server" if is_server else "client"
        print(f"Generated CURVE {kind} keypair for '{name}':")
        print(f"  Public key:  {pub_path}")
        print(f"  Secret key:  {sec_path}")
        if not is_server:
            print()
            print("Next steps:")
            print(
                f"  1. Copy {name}.key to the server's UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR"
            )
            print(f"  2. Give {sec_path} to the user (keep secret!)")
            print("  3. The user sets UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY and")
            print(f"     UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY from {sec_path}")
        else:
            print()
            print("Next steps:")
            print(f"  1. Set UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY from {pub_path}")
            print(f"  2. Set UMRS_ZMQ_CURVE_SERVER_SECRET_KEY from {sec_path}")
            print("  3. Distribute the public key to all clients")
    return 0


def _cmd_zmq_list_keys(args: argparse.Namespace, _store: UserStore) -> int:
    """List CURVE client keys in the keys directory."""
    from pathlib import Path

    keys_dir = args.keys_dir or str(Path(_DEFAULT_CLIENT_KEYS_DIR).expanduser())
    keys_path = Path(keys_dir)

    if not keys_path.is_dir():
        print(f"Keys directory does not exist: {keys_dir}", file=sys.stderr)
        return 1

    rows = []
    for key_file in sorted(keys_path.glob("*.key")):
        if key_file.name.endswith(".key_secret"):
            continue
        username = key_file.stem
        try:
            from umrs.core.zmq_security import load_public_key_z85

            z85 = load_public_key_z85(str(key_file))
        except Exception as exc:  # noqa: BLE001 — admin CLI degrades best-effort, named in output
            z85 = f"<error: {exc}>"
        has_secret = key_file.with_name(f"{username}.key_secret").exists()
        rows.append(
            {
                "username": username,
                "public_key": z85,
                "has_secret": has_secret,
                "file": str(key_file),
            }
        )

    if args.json:
        _print_json(rows)
    elif not rows:
        print(f"No client keys found in {keys_dir}")
    else:
        for row in rows:
            secret_marker = " (+ secret)" if row["has_secret"] else ""
            print(f"  {row['username']:20s}  {row['public_key'][:16]}…{secret_marker}")
        print(f"\n{len(rows)} key(s) in {keys_dir}")
    return 0


def _add_zmq_parsers(zmq_sub: argparse._SubParsersAction) -> None:
    keygen_p = zmq_sub.add_parser(
        "keygen",
        help="Generate a CURVE keypair for a user or the server",
        description=(
            "Generate a ZMTP CURVE keypair. For client keys, the public key "
            "file (<name>.key) should be placed in UMRS_ZMQ_CURVE_CLIENT_KEYS_DIR "
            "on the server. The secret key file (<name>.key_secret) is given to "
            "the user.\n\n"
            "For server keys, set UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY and "
            "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY from the generated files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    keygen_p.add_argument(
        "name", help="Key name (typically the UMRS username, or 'server')"
    )
    keygen_p.add_argument(
        "--server",
        action="store_true",
        help="Generate a server keypair instead of a client keypair",
    )
    keygen_p.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory (default: ~/.config/umrs/curve-keys/{clients,server})",
    )
    keygen_p.add_argument("--json", action="store_true")
    keygen_p.set_defaults(func=_cmd_zmq_keygen)

    revoke_p = zmq_sub.add_parser(
        "revoke-curve-key",
        help="Revoke a user's long-lived CURVE key (D-03: account = authority)",
    )
    revoke_p.add_argument("name", help="UMRS username")
    revoke_p.set_defaults(func=_cmd_zmq_revoke_curve_key)

    list_p = zmq_sub.add_parser(
        "list-keys",
        help="List registered CURVE client public keys",
    )
    list_p.add_argument(
        "--keys-dir",
        dest="keys_dir",
        default=None,
        help="Client keys directory (default: ~/.config/umrs/curve-keys/clients)",
    )
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_zmq_list_keys)


# ---------------------------------------------------------------------------
# endpoint subcommand handlers (admin-scoped, reads from running UMRS)
# ---------------------------------------------------------------------------


def _umrs_base_url(args: argparse.Namespace) -> str:
    """Resolve the UMRS HTTP base URL."""
    import os

    return getattr(args, "umrs_url", None) or os.environ.get(
        "UMRS_BASE_URL", "http://localhost:8000"
    )


def _umrs_api_get(url: str) -> dict:
    """GET from UMRS API. Returns parsed JSON."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        print(f"error: UMRS API returned {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(2) from exc
    except urllib.error.URLError as exc:
        print(
            f"error: cannot reach UMRS at {url}: {exc.reason}\n"
            "hint: is the UMRS server running? Set --umrs-url or $UMRS_BASE_URL.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _umrs_api_delete(url: str) -> dict | None:
    """DELETE from UMRS API. Returns parsed JSON or None."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        print(f"error: UMRS API returned {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(2) from exc
    except urllib.error.URLError as exc:
        print(
            f"error: cannot reach UMRS at {url}: {exc.reason}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _cmd_endpoint_list(args: argparse.Namespace, _store: UserStore) -> int:
    """List all endpoints across all projects (admin view)."""
    base = _umrs_base_url(args)
    data = _umrs_api_get(f"{base}/umrs/endpoints")

    endpoints = data if isinstance(data, list) else data.get("endpoints", [])

    # Apply filters
    status_filter = getattr(args, "status", None)
    if status_filter:
        endpoints = [e for e in endpoints if e.get("status") == status_filter]

    tier_filter = getattr(args, "tier", None)
    if tier_filter:
        endpoints = [
            e
            for e in endpoints
            if e.get("tier") == tier_filter or e.get("owner_type") == tier_filter
        ]

    if getattr(args, "json", False):
        _print_json(endpoints)
        return 0

    if not endpoints:
        print("(no endpoints)")
        return 0

    # Table output
    cols = [
        "JOB_ID",
        "MODEL",
        "PROJECT",
        "STATUS",
        "TIER",
        "OWNER",
        "BACKEND",
        "HOST:PORT",
    ]
    widths = {c: len(c) for c in cols}
    rows = []
    for ep in endpoints:
        job_id = str(ep.get("job_id", ""))[-12:]
        model = str(ep.get("model", ""))[:30]
        project = str(ep.get("project_id", ""))[:16]
        ep_status = str(ep.get("status", ""))
        tier = str(ep.get("tier", ep.get("owner_type", "")))
        owner = str(ep.get("owner_id", ""))[:16]
        backend = str(ep.get("backend", ""))
        host_port = f"{ep.get('host', '?')}:{ep.get('port', '?')}"
        row = [job_id, model, project, ep_status, tier, owner, backend, host_port]
        rows.append(row)
        for i, c in enumerate(cols):
            widths[c] = max(widths[c], len(row[i]))

    fmt = "  ".join(f"{{:<{widths[c]}}}" for c in cols)
    print(fmt.format(*cols))
    print(fmt.format(*(("-" * widths[c]) for c in cols)))
    for row in rows:
        print(fmt.format(*row))

    print(f"\nTotal: {len(endpoints)} endpoint(s)")
    return 0


def _cmd_endpoint_stop(args: argparse.Namespace, _store: UserStore) -> int:
    """Force-stop an endpoint by job_id (admin)."""
    base = _umrs_base_url(args)
    job_id = args.job_id

    if not getattr(args, "yes", False):
        if not _confirm(f"Force-stop endpoint {job_id}?", False):
            print("aborted", file=sys.stderr)
            return 1

    result = _umrs_api_delete(f"{base}/umrs/endpoints/{job_id}?force=true")
    if getattr(args, "json", False):
        _print_json(result or {"status": "stopped", "job_id": job_id})
    else:
        print(f"stopped endpoint {job_id}")
    return 0


def _cmd_endpoint_show(args: argparse.Namespace, _store: UserStore) -> int:
    """Show detailed info for a single endpoint."""
    base = _umrs_base_url(args)
    data = _umrs_api_get(f"{base}/umrs/endpoints/{args.job_id}")
    _print_json(data)
    return 0


def _cmd_endpoint_stats(args: argparse.Namespace, _store: UserStore) -> int:
    """Show endpoint registry statistics."""
    base = _umrs_base_url(args)
    data = _umrs_api_get(f"{base}/umrs/endpoints/stats")
    _print_json(data)
    return 0


def _umrs_api_post(url: str, data: dict | None = None) -> dict:
    """POST to UMRS API. Returns parsed JSON."""
    import json
    import urllib.error
    import urllib.request

    payload = json.dumps(data or {}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode() if exc.fp else ""
        print(f"error: UMRS API returned {exc.code}: {body}", file=sys.stderr)
        raise SystemExit(2) from exc
    except urllib.error.URLError as exc:
        print(
            f"error: cannot reach UMRS at {url}: {exc.reason}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _cmd_endpoint_refresh(args: argparse.Namespace, _store: UserStore) -> int:
    """Refresh a stuck endpoint (re-resolve host + re-probe readiness)."""
    base = _umrs_base_url(args)
    job_id = args.job_id

    result = _umrs_api_post(f"{base}/umrs/endpoints/{job_id}/refresh")
    if getattr(args, "json", False):
        _print_json(result)
    else:
        status = result.get("status", "unknown")
        host = result.get("host", "?")
        port = result.get("port", "?")
        print(f"✓ Refreshed endpoint {job_id}")
        print(f"  Status: {status}")
        print(f"  Host: {host}")
        print(f"  Port: {port}")
    return 0


def _add_endpoint_parsers(endpoint_sub: argparse._SubParsersAction) -> None:
    list_p = endpoint_sub.add_parser(
        "list",
        help="List all endpoints across all users/projects (admin view)",
        description=(
            "Query the running UMRS endpoint registry and display all "
            "active endpoints. Unlike 'r4 endpoint list' which is scoped "
            "to the current user's project, this shows everything."
        ),
        epilog=(
            "Examples:\n"
            "  umrs-server admin endpoint list\n"
            "  umrs-server admin endpoint list --status ready\n"
            "  umrs-server admin endpoint list --tier system --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_p.add_argument(
        "--status",
        choices=[
            "pending",
            "provisioning",
            "starting",
            "ready",
            "degraded",
            "expired",
            "failed",
            "terminated",
        ],
        help="Filter by endpoint status",
    )
    list_p.add_argument(
        "--tier",
        choices=["system", "project", "user"],
        help="Filter by deployment tier",
    )
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_endpoint_list)

    show_p = endpoint_sub.add_parser(
        "show",
        help="Show detailed info for an endpoint",
    )
    show_p.add_argument("job_id", help="Endpoint job ID")
    show_p.set_defaults(func=_cmd_endpoint_show)

    stop_p = endpoint_sub.add_parser(
        "stop",
        help="Force-stop an endpoint (admin kill)",
        description=(
            "Force-stop an endpoint by job_id. This tears down the underlying "
            "Slurm/K8s/Compose job and removes the endpoint from the registry."
        ),
    )
    stop_p.add_argument("job_id", help="Endpoint job ID to stop")
    stop_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    stop_p.add_argument("--json", action="store_true")
    stop_p.set_defaults(func=_cmd_endpoint_stop)

    stats_p = endpoint_sub.add_parser(
        "stats",
        help="Show endpoint registry statistics",
    )
    stats_p.add_argument("--json", action="store_true")
    stats_p.set_defaults(func=_cmd_endpoint_stats)

    refresh_p = endpoint_sub.add_parser(
        "refresh",
        help="Refresh a stuck endpoint (re-resolve host + re-probe readiness)",
        description=(
            "Re-resolve host/port from the backend and re-probe readiness. "
            "Use this to recover endpoints stuck in PROVISIONING/STARTING "
            'with host="pending" after a _finalize_provision task died '
            "(e.g. UMRS restart)."
        ),
        epilog=(
            "Examples:\n"
            "  umrs-server admin endpoint refresh 42315\n"
            "  umrs-server admin endpoint refresh 42315 --json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    refresh_p.add_argument("job_id", help="Endpoint job ID to refresh")
    refresh_p.add_argument("--json", action="store_true")
    refresh_p.set_defaults(func=_cmd_endpoint_refresh)


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
            "  1. Register a user (with Slurm account for HPC):\n"
            '       umrs-server admin user add <username> --fullname "Your Name" '
            "--role user --slurm-account <ACCOUNT>\n"
            "  2. Generate an authentication token:\n"
            "       umrs-server admin token generate-jwt --user <username> --project <project>\n"
            "  3. Give the token to the user, who runs:\n"
            "       r4 auth login --gateway-url <GATEWAY_URL> --api-key <TOKEN>\n"
            "  Shortcut: combine steps 1-3 with --issue-token:\n"
            "       umrs-server admin user add <username> --role user "
            "--slurm-account <ACCOUNT> --issue-token\n\n"
            "Diagnostics:\n"
            "  umrs-server admin user list\n"
            "  umrs-server admin user show <username>\n\n"
            "Shell completion:\n"
            '  eval "$(umrs-server completion bash)"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--deploy-dir",
        dest="deploy_dir",
        default=None,
        metavar="DIR",
        help=(
            "Path to a deployment directory (e.g. umrs/deploy/rccs or "
            "umrs/deploy/ai4s). Loads .env and site.yaml from that directory "
            "automatically, so you don't need to cd + source .env + export "
            "UMRS_SITE_CONFIG manually. Resolves --config from the same dir."
        ),
    )
    parser.add_argument(
        "--config",
        dest="server_config",
        default=None,
        metavar="CONF_UMRS_JSON",
        help=(
            "Path to the daemon's conf_umrs.json. Auto-discovered from "
            "$UMRS_CONFIG, CWD, or ~/.config/umrs when omitted."
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

    user_p = sub.add_parser(
        "user",
        help="Manage registered users",
        description=(
            "Create, list, inspect, disable, and update server-side UMRS user "
            "records. Use this command on the server/operator side; "
            "umrs does not perform admin actions."
        ),
        epilog=(
            "Most common commands:\n"
            '  umrs-server admin user add <username> --fullname "Your Name" --role user\n'
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

    zmq_p = sub.add_parser(
        "zmq",
        help="ZMQ CURVE key management",
        description=(
            "Generate and manage ZMTP CURVE keypairs for authenticated ZMQ "
            "transport. Use 'keygen' to create server or client keypairs, "
            "and 'list-keys' to inspect the client keys directory."
        ),
    )
    _add_zmq_parsers(zmq_p.add_subparsers(dest="action", required=True))

    endpoint_p = sub.add_parser(
        "endpoint",
        help="Inspect and manage live endpoints (admin view, requires running UMRS)",
        description=(
            "Admin-scoped endpoint management. Queries the running UMRS "
            "server to list, inspect, and force-stop endpoints across all "
            "users and projects. Unlike 'r4 endpoint' which is scoped to "
            "the current user's project, these commands see everything.\n\n"
            "Requires a running UMRS server (set --umrs-url or $UMRS_BASE_URL)."
        ),
        epilog=(
            "Examples:\n"
            "  umrs-server admin endpoint list\n"
            "  umrs-server admin endpoint list --status ready --json\n"
            "  umrs-server admin endpoint show <job_id>\n"
            "  umrs-server admin endpoint stop <job_id> -y\n"
            "  umrs-server admin endpoint stats"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    endpoint_p.add_argument(
        "--umrs-url",
        dest="umrs_url",
        default=None,
        help="UMRS server URL (default: $UMRS_BASE_URL or http://localhost:8000)",
    )
    _add_endpoint_parsers(endpoint_p.add_subparsers(dest="action", required=True))

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


def _apply_deploy_dir(deploy_dir: str) -> None:
    """Load .env and set UMRS_SITE_CONFIG from a deployment directory.

    This replaces the manual workflow::

        cd /path/to/deploy/rccs-front
        set -o allexport && source .env && set +o allexport
        export UMRS_SITE_CONFIG="$PWD/site.yaml"
        export PYTHONPATH="$(cd ../.. && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

    with a single ``--deploy-dir /path/to/deploy/rccs-front`` flag.
    """
    import os
    from pathlib import Path

    d = Path(deploy_dir).resolve()
    if not d.is_dir():
        print(f"error: --deploy-dir does not exist: {d}", file=sys.stderr)
        sys.exit(2)

    # Load .env (key=value lines, skip comments and blanks)
    env_file = d / ".env"
    if env_file.is_file():
        with open(env_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Don't overwrite existing env vars (same as set -o allexport
                # followed by source — existing vars from outer shell win).
                if key not in os.environ:
                    os.environ[key] = value

    # Set UMRS_SITE_CONFIG
    site_yaml = d / "site.yaml"
    if site_yaml.is_file():
        os.environ["UMRS_SITE_CONFIG"] = str(site_yaml)
    else:
        print(
            f"warning: no site.yaml in --deploy-dir {d}",
            file=sys.stderr,
        )

    # Add UMRS source tree to PYTHONPATH if present
    src_dir = d.parent.parent / "src"
    if src_dir.is_dir():
        existing = os.environ.get("PYTHONPATH", "")
        if str(src_dir) not in existing.split(os.pathsep):
            os.environ["PYTHONPATH"] = (
                str(src_dir) + os.pathsep + existing if existing else str(src_dir)
            )
            # Also update sys.path for the current process
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))


def _discover_server_config() -> str | None:
    """Find conf_umrs.json using the same discovery chain as the daemon."""
    import os

    from umrs.shared.config import (
        get_default_json_server_file,
        get_resolved_umrs_cfg_dir,
    )

    default_name = get_default_json_server_file()  # "conf_umrs.json"

    # 1. CWD — the admin often runs from the deploy directory
    cwd_candidate = os.path.join(os.getcwd(), default_name)
    if os.path.isfile(cwd_candidate):
        return os.path.abspath(cwd_candidate)

    # 2. $UMRS_CONFIG / XDG / ~/.config/umrs  (same as daemon)
    cfg_dir = get_resolved_umrs_cfg_dir()
    env_candidate = os.path.join(cfg_dir, default_name)
    if os.path.isfile(env_candidate):
        return os.path.abspath(env_candidate)

    return None


def _resolve_user_files_dir_from_config(
    server_config: str | None, explicit_dir: str | None
) -> str | None:
    """Derive user_files_dir from conf_umrs.json (explicit or auto-discovered)."""
    if explicit_dir:
        return explicit_dir
    if not server_config:
        server_config = _discover_server_config()
    if not server_config:
        return None
    import os

    from umrs.shared.config import resolve_server_user_files_dir

    config_path = os.path.abspath(os.path.expanduser(server_config))
    if not os.path.isfile(config_path):
        print(
            f"warning: --config file not found: {config_path}; "
            f"falling back to default user_files resolution",
            file=sys.stderr,
        )
        return None
    try:
        with open(config_path, encoding="utf-8") as fh:
            server_cfg = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"warning: cannot parse --config {config_path}: {exc}; "
            f"falling back to default user_files resolution",
            file=sys.stderr,
        )
        return None
    return resolve_server_user_files_dir(config_file=config_path, server_cfg=server_cfg)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # --deploy-dir: load .env + site.yaml before anything else.
    deploy_dir = getattr(args, "deploy_dir", None)
    if deploy_dir:
        _apply_deploy_dir(deploy_dir)
        # Auto-discover conf_umrs.json from the deploy directory itself.
        import os

        if not args.server_config:
            candidate = os.path.join(os.path.abspath(deploy_dir), "conf_umrs.json")
            if os.path.isfile(candidate):
                args.server_config = candidate

    # Auto-discover conf_umrs.json so --config is rarely needed.
    if not args.server_config:
        args.server_config = _discover_server_config()
    user_files_dir = _resolve_user_files_dir_from_config(
        args.server_config, args.user_files_dir
    )
    store = UserStore(user_files_dir=user_files_dir)
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
