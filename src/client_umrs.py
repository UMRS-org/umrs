#!/usr/bin/env python3

import argparse
import contextlib
import json as _json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from glob import glob
from pathlib import Path

from umrs.client.control_api import UMRSControlAPI
from umrs.client.control_commands import (
    MODEL_SCOPE_CHOICES,
    MODEL_TYPE_CHOICES,
    configure_job_cancel_parser,
    configure_job_list_parser,
    configure_job_status_parser,
    configure_job_submit_parser,
    configure_model_download_parser,
    configure_model_list_parser,
    configure_model_register_parser,
    configure_ping_parser,
    configure_status_parser,
    configure_workflow_list_parser,
    configure_workflow_run_parser,
    parse_json_object_arg,
)
from umrs.cli.completion import (
    emit_completion_if_requested,
    install_completion,
    print_completion_script,
    uninstall_completion,
)
from umrs.client.metadata import check_and_build_metadata
from umrs.core.MessageQueue import MessageQueue
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.shared.auth import create_auth_context_from_request
from umrs.shared.buckets import BucketsCfg, ClientBucketEntry
from umrs.shared.config import (
    get_default_docker_s3cfg,
)
from umrs.shared.connection_profiles import (
    discover_connection_profiles,
    get_profile_path,
    import_connection_profile,
    resolve_connection,
    set_current_profile_name,
)
from umrs.shared.json_config import load_client_cfg_file
from umrs.shared.loaded_config_profiles import (
    apply_s3_config_override,
    discover_loaded_config_profiles,
    get_user_config_search_locations,
    resolve_loaded_config,
    set_current_s3_config,
    set_current_server_config,
    set_current_user_config,
)
from umrs.shared.modality import modality_category, resolve_modality_task
from umrs.shared.sql_helpers import create_db_name, metadata_dict
from umrs.shared.storage.s3 import (
    check_bucket_name,
    connect_to_s3,
    decompose_bucket_alias_arg,
    download_objects,
    list_s3_bucket,
    make_path_to_bucket_string,
    s3_store_image,
    s3_store_text_file,
    s3_store_txt,
)


# ---------------------------------------------------------------------------
# ANSI helpers — auto-disabled when stdout is not a terminal
# ---------------------------------------------------------------------------
def _use_color():
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()


def _c(code, text):
    """Wrap *text* in ANSI escape if color is enabled."""
    if not _use_color():
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def _bold(t):     return _c("1", t)
def _dim(t):      return _c("2", t)
def _green(t):    return _c("32", t)
def _red(t):      return _c("31", t)
def _yellow(t):   return _c("33", t)
def _cyan(t):     return _c("36", t)
def _blue(t):     return _c("34", t)
def _magenta(t):  return _c("35", t)


def _fmt_cell(text, width=None, *, align="<", style=None):
    text = "" if text is None else str(text)
    if width is not None:
        if align == ">":
            text = f"{text:>{width}s}"
        elif align == "^":
            text = f"{text:^{width}s}"
        else:
            text = f"{text:<{width}s}"
    return style(text) if style else text


def _section(title):
    """Return a colored section header line."""
    return _bold(_cyan(f"[{title}]"))


def _kv(key, value, indent=2):
    """Format a key=value line with colored key."""
    pad = " " * indent
    return f"{pad}{_fmt_cell(key, 30, align='>', style=_bold)}  {value}"


def _status_dot(ok):
    return _green("●") if ok else _red("●")


def _is_dev_mode():
    """Server config is a server-side concern.  Expose it on the client only
    when ``UMRS_DEV_MODE=1`` or ``--server-conf`` was passed explicitly."""
    return os.environ.get('UMRS_DEV_MODE', '') == '1'


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------
def _format_profile_text(connection, selected_profile, errors):
    lines = []
    lines.append(_section("Connection Profile"))
    if selected_profile:
        lines.append(_kv("profile", _bold(selected_profile)))
    lines.append(_kv("mode", connection.mode))
    lines.append(_kv("server_host", _bold(connection.server_host)))
    lines.append(_kv("control_port", connection.control_port))
    lines.append(_kv("data_base_port", connection.data_base_port))
    lines.append(_kv("data_port_range", connection.data_port_range_label))
    lines.append(_kv("tunnel_mode", str(connection.tunnel_mode).lower()))
    lines.append("")
    lines.append(_section("Services"))
    lines.append(_kv("minio_url", connection.minio_url))
    lines.append(_kv("rag_url", connection.rag_url))
    if connection.profile_path:
        lines.append("")
        lines.append(_section("File"))
        lines.append(_kv("profile_path", connection.profile_path))
    if connection.ssh_host or connection.local_forwards or connection.command_hint:
        lines.append("")
        lines.append(_section("SSH Tunnel"))
        if connection.ssh_host:
            lines.append(_kv("ssh_host", connection.ssh_host))
        if connection.local_forwards:
            for fwd in connection.local_forwards:
                lines.append(_kv("local_forward", f"{fwd.get('local_port')}:{fwd.get('remote_host', 'localhost')}:{fwd.get('remote_port')}"))
        if connection.command_hint:
            lines.append(_kv("command_hint", _dim(connection.command_hint)))
    if errors:
        lines.append("")
        for err in errors:
            lines.append(f"  {_red('!')} {err}")
    if hasattr(connection, 'sources') and connection.sources:
        lines.append("")
        lines.append(_section("Sources"))
        for src_key, src_val in sorted(connection.sources.items()):
            lines.append(_kv(src_key, _dim(src_val)))
    return "\n".join(lines)


def _extract_bucket_aliases(user_cfg):
    if not isinstance(user_cfg, dict):
        return []

    aliases = set()
    for xp in user_cfg.get("experiences", []):
        for modality in xp.get("modalities", []):
            bucket_alias = (modality.get("s3") or {}).get("bucket_alias")
            if bucket_alias:
                aliases.add(bucket_alias)
    return sorted(aliases)


def _resolve_control_client_id(user_cfg=None):
    if isinstance(user_cfg, dict):
        username = user_cfg.get("username")
        if username:
            return username
    return os.environ.get('UMRS_CLIENT_ID', os.environ.get('USER', '_'))


def _build_monitor_capabilities(client_id, admin_token=None, user_cfg=None):
    auth_ctx = create_auth_context_from_request(client_id, token=admin_token)
    target_user = auth_ctx.username or client_id
    return {
        "username": auth_ctx.username,
        "is_admin": auth_ctx.is_admin,
        "has_user_config": isinstance(user_cfg, dict),
        "bucket_aliases": _extract_bucket_aliases(user_cfg),
        "allowed_actions": {
            "recompute": auth_ctx.can_execute_command(TaskCmd.RECOMPUTE, target_user=target_user, log_denial=False),
            "update": auth_ctx.can_execute_command(TaskCmd.UPDATE, target_user=target_user, log_denial=False),
            "check": auth_ctx.can_execute_command(TaskCmd.CHECK, target_user=target_user, log_denial=False),
            "reload": auth_ctx.can_execute_command(TaskCmd.RELOAD, target_user=target_user, log_denial=False),
        },
    }


def _format_config_text(resolved, *, show_server=False):
    kind_labels = {"server": "Server Config", "user": "User Config", "s3": "S3 Config"}
    keys = ("server", "user", "s3") if show_server else ("user", "s3")
    lines = []
    for key in keys:
        layer = getattr(resolved, key)
        dot = _status_dot(layer.valid)
        lines.append(_section(f"{kind_labels[key]}  {dot}"))
        lines.append(_kv("path", layer.path or _dim("(none)")))
        lines.append(_kv("source", _dim(layer.source or "(none)")))
        if layer.error:
            lines.append(_kv("error", _red(layer.error)))
        for detail_key, detail_value in sorted(layer.details.items()):
            if isinstance(detail_value, list):
                rendered = ", ".join(str(item) for item in detail_value)
            else:
                rendered = detail_value
            lines.append(_kv(detail_key, rendered))
        lines.append("")
    if resolved.warnings:
        lines.append(_section(f"Warnings  {_yellow('⚠')}  {len(resolved.warnings)}"))
        for warning in resolved.warnings:
            lines.append(f"  {_yellow('⚠')} {warning}")
    return "\n".join(lines)


def _print_config_candidates_text(discovered, *, show_server=False):
    headings = {
        "server": "Server Configs",
        "user": "User Configs",
        "s3": "S3 Configs",
    }
    keys = ("server", "user", "s3") if show_server else ("user", "s3")
    for key in keys:
        print(_section(headings[key]))
        candidates = discovered.get(key) or []
        if not candidates:
            print(f"  {_dim('(none)')}")
            print("")
            continue
        for candidate in candidates:
            marker = _green("▸") if candidate.current else " "
            dot = _status_dot(candidate.valid)
            path_str = _bold(candidate.path) if candidate.current else candidate.path
            print(f"  {marker} {dot} {path_str}")
            if candidate.error:
                print(f"      {_red(candidate.error)}")
        print("")


def _build_user_config_template(username: str, fullname: str) -> dict:
    """Return a minimal user config dict that passes client_schema validation."""
    return {
        "username": username,
        "fullname": fullname,
        "s3_configuration_file": f"~/.config/umrs/s3cfg_{username}",
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


def _handle_config_init(args):
    username = args.username
    fullname = args.fullname

    # Prompt interactively when required fields are missing.
    if not username:
        try:
            username = input("Username: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1
    if not username:
        raise ValueError("Username must not be empty.")

    if not fullname:
        try:
            fullname = input("Full name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1
    if not fullname:
        raise ValueError("Full name must not be empty.")

    # Determine output path.
    if args.output_path:
        out_path = Path(args.output_path).expanduser()
    else:
        from umrs.shared.config import get_resolved_umrs_cfg_dir, get_users_default_cfg_subdir
        users_dir = Path(get_resolved_umrs_cfg_dir()).expanduser() / get_users_default_cfg_subdir()
        out_path = users_dir / f"{username}.json"

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Config already exists: {out_path}\n"
            f"Use --force to overwrite, or 'umrs-client config use --user {out_path}' to select it."
        )

    # Build config from template or from scratch.
    if args.template:
        template_path = Path(args.template).expanduser()
        if not template_path.is_file():
            raise FileNotFoundError(f"Template not found: {template_path}")
        import json as _json_mod
        with open(template_path, encoding="utf-8") as f:
            cfg = _json_mod.load(f)
        cfg["username"] = username
        cfg["fullname"] = fullname
        cfg["s3_configuration_file"] = f"~/.config/umrs/s3cfg_{username}"
    else:
        # Check for a site-level template via UMRS_CONFIG_TEMPLATE env var.
        site_template = os.environ.get("UMRS_CONFIG_TEMPLATE")
        if site_template:
            site_template_path = Path(site_template).expanduser()
            if site_template_path.is_file():
                import json as _json_mod
                with open(site_template_path, encoding="utf-8") as f:
                    cfg = _json_mod.load(f)
                cfg["username"] = username
                cfg["fullname"] = fullname
                cfg["s3_configuration_file"] = f"~/.config/umrs/s3cfg_{username}"
            else:
                cfg = _build_user_config_template(username, fullname)
        else:
            cfg = _build_user_config_template(username, fullname)

    # Write the file.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json_mod
    with open(out_path, "w", encoding="utf-8") as f:
        _json_mod.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Register as current user config.
    saved = set_current_user_config(str(out_path))

    if args.output == 'json':
        print(api_json(
            message="User config created and set as current",
            path=str(out_path),
            current=str(saved),
            output_format=args.output,
        ))
    else:
        print(f"{_green('✓')} Created {_bold(str(out_path))}")
        print(f"  Set as current user config: {_dim(str(saved))}")
        print("  Open the JSON file in an editor to customise experiences and modalities.")
        print(f"  {_dim('Do not run the .json path directly in the shell.')}")
        print("  Then run:")
        print(f"  {_cyan('umrs-client reload')}")

    # Auto-install shell tab completion
    comp_msg = install_completion('umrs-client')
    if args.output != 'json':
        print(f"  {comp_msg}")

    return 0


def _handle_config_command(args):
    if args.config_action == 'init':
        return _handle_config_init(args)

    if args.config_action == 'list':
        discovered = discover_loaded_config_profiles()
        show_srv = _is_dev_mode() or bool(args.server_conf)
        payload = {
            key: [candidate.to_dict() for candidate in value]
            for key, value in discovered.items()
            if show_srv or key != 'server'
        }
        if args.output == 'json':
            print(api_json(configs=payload, output_format=args.output))
        else:
            _print_config_candidates_text(discovered, show_server=show_srv)
        return 0

    if args.config_action == 'use':
        if not any((args.server, args.user, args.s3)):
            raise ValueError("Provide at least one of --server, --user, or --s3.")

        updates = {}
        if args.server:
            if not (_is_dev_mode() or args.server_conf):
                raise ValueError(
                    "--server is only available in dev mode "
                    "(set UMRS_DEV_MODE=1). "
                    "Server config is managed on the server side."
                )
            updates['server'] = str(set_current_server_config(args.server))
        if args.user:
            updates['user'] = str(set_current_user_config(args.user))
        if args.s3:
            updates['s3'] = str(set_current_s3_config(args.s3))

        if args.output == 'json':
            print(api_json(message="Current config selections updated", updates=updates, output_format=args.output))
        else:
            print(f"{_green('✓')} Current config selections updated")
            for key, value in updates.items():
                print(_kv(key, value))
        return 0

    resolved = resolve_loaded_config(
        explicit_server_config_path=args.server_conf,
        explicit_user_config_path=args.conf,
        explicit_s3_config_path=args.s3cfg,
    )
    show_srv = _is_dev_mode() or bool(args.server_conf)
    if not show_srv:
        resolved.warnings = [
            w for w in resolved.warnings
            if "server config" not in w.lower()
        ]
    if args.output == 'json':
        payload = resolved.to_dict()
        if not show_srv:
            payload.pop('server', None)
        print(api_json(config=payload, output_format=args.output))
    else:
        print(_format_config_text(resolved, show_server=show_srv))
    return 0


def _handle_admin_user_command(args):
    """Handle ``admin user {add,list,remove}``."""
    import json as _json_mod

    if args.user_action == 'add':
        config_path = args.config_path
        if not os.path.isfile(config_path):
            print(f"{_red('✗')} File not found: {config_path}")
            return 1

        # Peek at the username for display
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg_data = _json_mod.load(fh)
            username = cfg_data.get("username", "<unknown>")
        except Exception:
            username = "<unknown>"

        admin_token = getattr(args, 'admin_token', None) or os.environ.get('UMRS_ADMIN_TOKEN')
        if not admin_token:
            print(f"{_red('✗')} Admin token required. Set UMRS_ADMIN_TOKEN or pass --admin-token.")
            return 1

        connection = resolve_connection(
            explicit_profile_name=args.profile,
            explicit_server_ip=args.server_ip,
            explicit_control_port=args.control_port,
        )
        control_mq = MessageQueue(
            role="client",
            queue_type="control",
            server_ip=connection.server_host,
            port=connection.control_port,
            tunnel_mode=connection.tunnel_mode,
            prefer_env_server_host=False,
        )
        try:
            client_id = _resolve_control_client_id()
            api = UMRSControlAPI(control_mq, client_id, admin_token=admin_token)
            result = api.enroll_user(config_path)
        except Exception as e:
            if args.output == 'json':
                print(api_json(action='user_add', status='error', message=str(e), output_format=args.output))
            else:
                print(f"{_red('✗')} Failed to enroll user: {e}")
            return 1
        finally:
            with contextlib.suppress(Exception):
                control_mq.close()

        if args.output == 'json':
            print(api_json(action='user_add', **result, output_format=args.output))
        else:
            if result.get("status") == "ok":
                print(f"{_green('✓')} User {_bold(username)} enrolled on the server.")
                print(f"  {_dim('The server has auto-reloaded to provision MinIO/Postgres resources.')}")
            else:
                print(f"{_red('✗')} Enrollment failed: {result.get('message', 'unknown error')}")
        return 0 if result.get("status") == "ok" else 1

    if args.user_action == 'list':
        admin_token = getattr(args, 'admin_token', None) or os.environ.get('UMRS_ADMIN_TOKEN')
        connection = resolve_connection(
            explicit_profile_name=args.profile,
            explicit_server_ip=args.server_ip,
            explicit_control_port=args.control_port,
        )
        control_mq = MessageQueue(
            role="client",
            queue_type="control",
            server_ip=connection.server_host,
            port=connection.control_port,
            tunnel_mode=connection.tunnel_mode,
            prefer_env_server_host=False,
        )
        try:
            client_id = _resolve_control_client_id()
            api = UMRSControlAPI(control_mq, client_id, admin_token=admin_token)
            status = api.request_status("all")
        except Exception as e:
            if args.output == 'json':
                print(api_json(action='user_list', status='error', message=str(e), output_format=args.output))
            else:
                print(f"{_red('✗')} Failed to query server: {e}")
            return 1
        finally:
            with contextlib.suppress(Exception):
                control_mq.close()

        # Extract enrolled users from status data
        users = set()
        if isinstance(status, dict):
            for _key, entry in status.items():
                if isinstance(entry, dict):
                    for work in entry.get("work", []):
                        u = work.get("user")
                        if u:
                            users.add(u)

        if args.output == 'json':
            print(api_json(action='user_list', users=sorted(users), output_format=args.output))
        else:
            print(_section("Enrolled Users"))
            if users:
                for u in sorted(users):
                    print(f"  {_status_dot('green')} {u}")
            else:
                print(f"  {_dim('No users found in server status.')}")
                print(f"  {_dim('(Users appear here after at least one processing job has run.)')}")
        return 0

    if args.user_action == 'remove':
        if args.output == 'json':
            print(api_json(action='user_remove', status='error', message='Not yet implemented', output_format=args.output))
        else:
            print(f"{_yellow('⚠')} User removal is not yet implemented.")
            print(f"  {_dim('De-provisioning (MinIO IAM cleanup, Postgres role removal) will be added in a future release.')}")
        return 1

    raise ValueError(f"Unknown user action: {args.user_action}")


def _handle_admin_command(args):
    """Handle ``admin token {generate,list,revoke}`` and ``admin user {add,list,remove}``."""
    from umrs.shared.auth import AdminTokenManager

    if args.admin_action == 'user':
        return _handle_admin_user_command(args)

    if args.admin_action != 'token':
        raise ValueError(f"Unknown admin action: {args.admin_action}")

    mgr = AdminTokenManager()

    if args.token_action == 'generate':
        existing = {t['name'] for t in mgr.list_tokens()}
        overwrite = args.name in existing
        token = mgr.generate_token(args.name, description=args.description)
        if args.output == 'json':
            print(api_json(action='token_generate', name=args.name, token=token, overwritten=overwrite, output_format=args.output))
        else:
            action = "Regenerated" if overwrite else "Generated"
            print(f"{_green('✓')} {action} admin token for {_bold(args.name)} (for umrs-client --admin-token / $UMRS_ADMIN_TOKEN):")
            print(f"  {_yellow(token)}")
            print("")
            print(f"  {_dim('Save this token — it cannot be retrieved later.')}")
            print(f"  {_dim('Use with: umrs-client --admin-token <token> <command>')}")
            print(f"  {_dim('NOTE: this is NOT a Gateway/r4 JWT — use umrs-server admin token generate-jwt for that.')}")
        return 0

    if args.token_action == 'list':
        tokens = mgr.list_tokens()
        if args.output == 'json':
            print(api_json(action='token_list', tokens=tokens, output_format=args.output))
        else:
            print(_section("Admin Tokens"))
            if tokens:
                print(
                    "  "
                    + " ".join(
                        [
                            _fmt_cell("Name", 20, style=_bold),
                            _fmt_cell("Description", 30, style=_bold),
                            _fmt_cell("Created", 25, style=_bold),
                        ]
                    )
                )
                print(f"  {_dim('-' * 75)}")
                for t in tokens:
                    print(
                        "  "
                        + " ".join(
                            [
                                _fmt_cell(t['name'], 20),
                                _fmt_cell(t.get('description', ''), 30),
                                _fmt_cell(t.get('created_at', ''), 25, style=_dim),
                            ]
                        )
                    )
            else:
                print(f"  {_dim('No admin tokens configured.')}")
        return 0

    if args.token_action == 'revoke':
        ok = mgr.revoke_token(args.name)
        if args.output == 'json':
            print(api_json(action='token_revoke', name=args.name, revoked=ok, output_format=args.output))
        else:
            if ok:
                print(f"{_green('✓')} Token {_bold(args.name)} revoked.")
            else:
                print(f"{_red('✗')} Token {_bold(args.name)} not found.")
        return 0 if ok else 1

    raise ValueError(f"Unknown token action: {args.token_action}")


def _handle_profile_init(args):
    import json as _json_mod

    name = args.name
    if not name:
        try:
            name = input("Profile name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1
    if not name:
        raise ValueError("Profile name must not be empty.")

    # Prompt for host and mode when not provided via CLI flags.
    host = args.host
    if host == '127.0.0.1' and '--host' not in sys.argv:
        try:
            entered = input(f"Server host [{host}]: ").strip()
            if entered:
                host = entered
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1

    mode = args.mode
    if mode == 'direct' and '--mode' not in sys.argv:
        try:
            entered = input(f"Connection mode (direct/ssh_tunnel) [{mode}]: ").strip()
            if entered in ('direct', 'ssh_tunnel'):
                mode = entered
            elif entered:
                print(f"Invalid mode '{entered}', using '{mode}'")
        except (EOFError, KeyboardInterrupt):
            print("")
            return 1
    minio_url = args.minio_url or f"http://{host}:9000"
    rag_url = args.rag_url or f"http://{host}:8081"

    profile_data = {
        "name": name,
        "mode": args.mode,
        "zmq": {
            "host": host,
            "control_port": args.control_port,
            "data_port_range": {
                "start": args.data_port_start,
                "end": args.data_port_end,
            },
            "tunnel_mode": args.mode == "ssh_tunnel",
            "health_check_timeout_ms": 5000,
        },
        "services": {
            "minio_url": minio_url,
            "rag_url": rag_url,
        },
    }

    if args.mode == "ssh_tunnel":
        ssh_host = args.ssh_host or host
        profile_data["ssh"] = {
            "host": ssh_host,
            "command_hint": f"ssh -L {args.control_port}:localhost:{args.control_port} {ssh_host}",
            "local_forwards": [
                {"local_port": args.control_port, "remote_host": "localhost", "remote_port": args.control_port},
                {"local_port": args.data_port_start, "remote_host": "localhost", "remote_port": args.data_port_start},
            ],
        }

    out_path = get_profile_path(name)

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Profile already exists: {out_path}\n"
            f"Use --force to overwrite, or 'umrs-client profile use {name}' to select it."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json_mod.dump(profile_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    set_current_profile_name(name)

    if args.output == 'json':
        print(api_json(
            message="Connection profile created and set as current",
            name=name,
            path=str(out_path),
            output_format=args.output,
        ))
    else:
        print(f"{_green('✓')} Created {_bold(str(out_path))}")
        print(f"  Set as current profile: {_bold(name)}")
        if args.mode == "ssh_tunnel":
            print("  Open the JSON file in an editor to customise SSH tunnels.")
            print(f"  {_dim('Do not run the .json path directly in the shell.')}")
            print("  Then run:")
            print(f"  {_cyan(profile_data['ssh']['command_hint'])}")
    return 0


def _handle_login(args):
    """Unified login: save JWT, create profile + user config, verify connection."""
    import getpass

    host = args.host
    port = args.port
    mode = args.mode
    api_key = args.api_key
    username = args.username

    # --- 1. Get the token ---
    if not api_key:
        api_key = getpass.getpass("Enter JWT or onboarding token: ")
    if not api_key:
        print(_red("✗ No token provided."), file=sys.stderr)
        return 1

    # --- 2. Extract username from JWT if not provided ---
    from umrs.shared.auth import decode_jwt_claims_unsafe

    claims = decode_jwt_claims_unsafe(api_key)
    if claims:
        jwt_user = claims.get("sub")
        if jwt_user and not username:
            username = jwt_user
        print(f"  JWT identity (display only, not authoritative):")
        if jwt_user:
            print(f"    Username: {jwt_user}")
        if "email" in claims:
            print(f"    Email: {claims['email']}")
        if "project_id" in claims:
            print(f"    Project: {claims['project_id']}")
        if "exp" in claims:
            import time as _time

            remaining = claims["exp"] - _time.time()
            if remaining > 0:
                hours = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                print(f"    Expires in: {hours}h {mins}m")
            else:
                print(f"    Token EXPIRED")

    if not username:
        username = input("UMRS username: ").strip()
    if not username:
        print(_red("✗ Username is required."), file=sys.stderr)
        return 1

    # --- 3. Create connection profile ---
    profile_name = args.profile_name or f"{host}-{mode}"
    profile_path = get_profile_path()
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or create profiles JSON.
    if profile_path.exists():
        try:
            all_profiles = _json.loads(profile_path.read_text())
        except Exception:
            all_profiles = {}
    else:
        all_profiles = {}

    all_profiles[profile_name] = {
        "server_host": host,
        "control_port": port,
        "tunnel_mode": mode == "ssh_tunnel",
        "jwt": api_key,
    }
    all_profiles["_current"] = profile_name
    profile_path.write_text(_json.dumps(all_profiles, indent=2))
    profile_path.chmod(0o600)
    print(f"✓ Connection profile '{profile_name}' saved")

    # --- 4. Verify connection ---
    print(f"  Verifying connection to {host}:{port} ...")
    try:
        mq = MessageQueue(
            role="client",
            queue_type="control",
            server_ip=host,
            port=port,
            tunnel_mode=(mode == "ssh_tunnel"),
            prefer_env_server_host=False,
        )
        mq.set_timeouts(5000, 5000)
        if mq.ping():
            print(f"  ✓ Server is reachable")
        else:
            print(f"  ✗ Server did not respond to ping (may still be starting)")
    except Exception as e:
        print(f"  ✗ Connection check failed: {e}")
        print(f"  Profile saved — you can retry with `umrs-client ping`")

    print(f"\n✓ Login complete for user '{username}' on {host}:{port}")
    print(f"  Profile: {profile_name}")
    print(f"  Run `umrs-client ping` to verify full connectivity.")
    return 0


def _handle_profile_command(args):
    if args.profile_action == 'init':
        return _handle_profile_init(args)

    if args.profile_action == 'list':
        profiles, errors = discover_connection_profiles()
        current = None
        try:
            connection = resolve_connection(explicit_profile_name=args.profile)
            current = connection.profile_name
        except Exception:
            current = None

        payload = {
            "profiles": [
                {
                    "name": profile.name,
                    "mode": profile.mode,
                    "path": profile.path,
                    "current": profile.name == current,
                }
                for profile in profiles
            ],
            "errors": errors,
        }
        if args.output == 'json':
            print(api_json(**payload, output_format=args.output))
        else:
            print(_section("Connection Profiles"))
            if profiles:
                for profile in profiles:
                    if profile.name == current:
                        marker = _green("\u25b8")
                        name_str = _fmt_cell(profile.name, 20, style=_bold)
                    else:
                        marker = " "
                        name_str = _fmt_cell(profile.name, 20)
                    print(f"  {marker} {name_str} {_fmt_cell(profile.mode, 10, style=_dim)} {_dim(profile.path)}")
            else:
                print(f"  {_dim('No connection profiles found.')}")
            for entry in errors:
                print(f"  {_red('!')} {entry['name']}: {entry['error']}")
        return 0

    if args.profile_action == 'use':
        path = set_current_profile_name(args.name)
        if args.output == 'json':
            print(api_json(message=f"Current connection profile set to '{args.name}'", path=str(path), output_format=args.output))
        else:
            print(f"{_green('✓')} Current connection profile set to {_bold(args.name)}")
            print(_kv("path", _dim(str(path))))
        return 0

    if args.profile_action == 'import':
        profile = import_connection_profile(args.file, overwrite=args.force)
        payload = {
            "message": f"Imported connection profile '{profile.name}'",
            "name": profile.name,
            "path": profile.path,
        }
        if args.output == 'json':
            print(api_json(**payload, output_format=args.output))
        else:
            print(f"{_green('✓')} Imported connection profile {_bold(profile.name)}")
            print(_kv("path", _dim(payload['path'])))
        return 0

    connection = resolve_connection(explicit_profile_name=args.profile, explicit_server_ip=args.server_ip, explicit_control_port=args.control_port)
    conn_dict = connection.to_dict()
    # Hide internal sources dict from JSON output (available in text via [sources] section)
    conn_dict.pop("sources", None)
    payload = {
        "connection": conn_dict,
        "selected_profile": connection.profile_name,
    }
    if args.output == 'json':
        print(api_json(**payload, output_format=args.output))
    else:
        print(_format_profile_text(connection, connection.profile_name, []))
    return 0


# ======================= Helper / API abstractions (new) =======================
def configure_buckets(user_cfg, buckets_cfg):
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        sql_username  = xp["sql_username"]
        xp_name = xp["xp_name"]
        db_name = create_db_name(sql_username, xp_name)
        for modality in xp["modalities"]:
            # Support both old (modality_name) and new (modality + task) config
            raw_modality_name = modality.get("modality_name") or modality.get("modality")
            raw_task = modality.get("task")
            modality_name, task = resolve_modality_task(raw_modality_name, raw_task)
            bucket_alias =  modality["s3"]["bucket_alias"]
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            if modality["s3"].get("need_output_bucket", False):
                output_path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username, output=True)
            else:
                output_path_to_bucket = None
            # print(bucket_alias, username, xp_name, modality_name)
            metadata = metadata_dict(modality["sql"])
            entry = ClientBucketEntry(
                path_to_bucket,
                username,
                xp_name,
                modality_name,
                task,
                s3_configuration_file,
                metadata,
                sql_username,
                db_name,
                output_path_to_bucket
            )
            buckets_cfg.store(bucket_alias, entry)

def build_buckets_cfg(user_cfg):
    buckets_cfg = BucketsCfg()
    configure_buckets(user_cfg, buckets_cfg)
    return buckets_cfg

def api_json(status="ok", output_format='json', **payload):
    out = {"status": status}
    out.update(payload)
    if output_format == 'json':
        return _json.dumps(out, ensure_ascii=False)
    lines = []
    for k, v in out.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}=[{', '.join(map(str,v))}]")
        else:
            lines.append(f"{k}={v}")
    return "\n".join(lines)


def _resolve_status_target(args):
    target = getattr(args, "target", "all") or "all"
    if getattr(args, "status_all", False):
        if target != "all":
            raise ValueError("Use either '--all' or a specific status key, not both.")
        return "all"
    return target

def parse_metadata_kv(pairs_list):
    meta = {}
    if not pairs_list:
        return meta
    for p in pairs_list:
        if ':' not in p:
            raise ValueError(f"Metadata must be key:value got '{p}'")
        k, v = p.split(':', 1)
        meta[k.strip()] = v.strip()
    return meta

def merge_metadata(base_metadata, override_meta_dict):
    # base_metadata is the structure produced by metadata_dict(sql_conf).
    # For now we only inject overrides into a flat list under Metadata (matching existing code paths)
    # Existing code path expects metadata_args = { 'Metadata': {...}}
    merged = dict(base_metadata.items()) if base_metadata else {}
    md = merged.get('Metadata', {})
    md.update(override_meta_dict)
    merged['Metadata'] = md
    return merged



def format_status_text(response_payload, default_key):
    if response_payload.get('status') != 'ok':
        message = response_payload.get('message', 'status unavailable')
        if message == "No status for 'reload'":
            message = (
                "No status for 'reload'. Reload is synchronous; use "
                "'umrs-client status all' for a summary or "
                "'umrs-client list_buckets' to verify onboarding."
            )
        return f"{_red('✗')} {message}"

    client_label = response_payload.get('client')
    lines = []

    def append_entry(name, entry):
        header_parts = []
        if client_label:
            header_parts.append(client_label)
        if name:
            header_parts.append(name)
        generated_at = entry.get('generated_at')
        last_updated_at = entry.get('last_updated_at')
        if header_parts or generated_at or last_updated_at:
            header = ' - '.join(header_parts)
            time_parts = []
            if generated_at:
                time_parts.append(f"generated {generated_at}")
            if last_updated_at:
                time_parts.append(f"last update {last_updated_at}")
            if time_parts:
                time_text = '; '.join(time_parts)
                header = f"{header} {_dim('(' + time_text + ')')}" if header else _dim(time_text)
            if header:
                lines.append(_section(header))
        summary_body = entry.get('summary')
        if summary_body:
            lines.append(f"  {summary_body}")
        else:
            lines.append(f"  {_dim('(no summary)')}")

    if 'summary' in response_payload:
        append_entry(default_key, response_payload['summary'])
    elif 'summaries' in response_payload:
        summaries = response_payload['summaries'] or {}
        for key in sorted(summaries.keys()):
            append_entry(key, summaries[key])
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    else:
        return f"{_red('✗')} Unexpected status payload"

    return "\n".join(lines)

def action_ping(control_mq, output_format, timeout_ms=2000, connection=None):
    ready = control_mq.is_server_ready(timeout_ms=timeout_ms)
    endpoint_info = {}
    if connection is not None:
        endpoint_info["profile"] = connection.profile_name or "(none)"
        endpoint_info["server_host"] = connection.server_host
        endpoint_info["control_port"] = connection.control_port
    return api_json(server_ready=ready, output_format=output_format, **endpoint_info)

def action_list_buckets(bucket_cfg, username, output_format):
    buckets = []
    for alias, entry in bucket_cfg.items():
        if entry.username == username :
            buckets.append({
                'bucket_alias': alias,
                'modality': entry.modality_name,
                'task': entry.task,
                'path_to_bucket': entry.path_to_bucket
            })
    if output_format == 'json':
        return api_json(buckets=buckets, output_format=output_format)
    # Text mode: formatted table
    lines = [_section(f"Buckets for {_bold(username)}")]
    if not buckets:
        lines.append(f"  {_dim('No buckets found.')}")
    else:
        lines.append(
            "  "
            + "  ".join(
                [
                    _fmt_cell("Alias", 20, style=_bold),
                    _fmt_cell("Modality", 15, style=_bold),
                    _fmt_cell("Task", 15, style=_bold),
                    _bold("Path"),
                ]
            )
        )
        lines.append(f"  {_dim('-' * 80)}")
        for b in buckets:
            lines.append(
                "  "
                + "  ".join(
                    [
                        _fmt_cell(b['bucket_alias'], 20),
                        _fmt_cell(b['modality'], 15),
                        _fmt_cell(b.get('task', ''), 15),
                        _dim(b['path_to_bucket']),
                    ]
                )
            )
    return "\n".join(lines)

def perform_upload_text(s3, full_bucket_name, text, data_mq, metadata_args):
    s3_key = s3_store_txt(s3, full_bucket_name, "", text, data_mq, metadata_args)
    return s3_key

def perform_upload_image(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_image(s3, full_bucket_name, "", file_path, data_mq, metadata_args)

def perform_upload_textfile(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_text_file(s3, full_bucket_name, "", file_path, data_mq, metadata_args)

def resolve_bucket(bucket_store, bucket_alias):
    entry = bucket_store.get(
        bucket_alias,
        fields=[
            "path_to_bucket",
            "username",
            "xp_name",
            "modality_name",
            "s3_configuration_file",
            "metadata",
        ],
    )
    if entry is None:
        raise KeyError(f"Bucket alias not found: {bucket_alias}")
    return entry


def send_files_to_server(username, file_path_list, control_mq, data_mq):
    for file_path in file_path_list:
        data_mq.send_file(file_path)
    success = control_mq.notify_server(username, TaskCmd.FILE, len(file_path_list))
    if success:
        print(f"Files {file_path_list} received by server")
    else:
        print(f"Error sending file(s) {file_path_list} to server")


class _ClientHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


_CLIENT_EPILOG = (
    "Onboarding:\n"
    "  1. Ask the operator to register you on the server:\n"
    "       umrs-server admin user add <username> --fullname \"Your Name\" --role user\n"
    "  2. Create or select your local client config:\n"
    "       umrs-client config init --username <username>\n"
    "  3. Check the resolved config and server connectivity:\n"
    "       umrs-client config show\n"
    "       umrs-client ping\n\n"
    "Shell completion:\n"
    "  eval \"$(umrs-client completion bash)\"\n"
)


def _disabled_admin_message():
    return (
        "umrs-client admin is disabled; admin operations are server-side only. "
        "Use 'umrs-server admin --help' on the server. Examples: "
        "'umrs-server admin user add <username> --fullname \"Your Name\" --role user' "
        "and 'umrs-server admin user list'."
    )


# ---------------------------------------------------------------------------
# Client context — shared state passed to every action handler
# ---------------------------------------------------------------------------
@dataclass
class _ClientContext:
    """Shared state passed to every action handler."""

    args: argparse.Namespace
    control_mq: MessageQueue
    connection: object  # ResolvedConnection
    admin_token: str | None
    config_state: object
    configure_file: str | None
    user: dict | None
    buckets_cfg: object | None  # BucketsCfg or None


# ---------------------------------------------------------------------------
# Per-action handlers — each returns an int exit code
# ---------------------------------------------------------------------------

def _action_ping(ctx):
    args = ctx.args
    api = UMRSControlAPI(
        ctx.control_mq,
        _resolve_control_client_id(ctx.user),
        admin_token=ctx.admin_token,
    )
    try:
        ping_result = api.ping(timeout_ms=ctx.connection.health_check_timeout_ms)
        endpoint_info = {
            "profile": ctx.connection.profile_name or "(none)",
            "server_host": ctx.connection.server_host,
            "control_port": ctx.connection.control_port,
        }
        if args.output == 'json':
            print(api_json(output_format='json', **ping_result, **endpoint_info))
        else:
            ready = ping_result.get("server_ready", False)
            dot = _status_dot(ready)
            status_word = _green("ready") if ready else _red("not ready")
            profile_label = _bold(ctx.connection.profile_name) if ctx.connection.profile_name else _dim("(none)")
            print(f"  {dot} Server {status_word}  {_dim('|')}  profile={profile_label}  {_dim('|')}  {ctx.connection.server_host}:{ctx.connection.control_port}")
        return 0
    finally:
        ctx.control_mq.close()


def _action_model(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id, admin_token=ctx.admin_token)
    model_action = args.model_action
    exit_code = 0
    try:
        if model_action == 'register':
            parsed = api.register_model(
                name=args.name,
                path=args.path,
                model_type=args.model_type,
                scope=args.scope,
                project_id=args.project_id,
                description=args.description,
                huggingface_id=args.huggingface_id,
            )
        elif model_action == 'list':
            parsed = api.list_models(scope=args.scope)
        elif model_action == 'download':
            target_name = args.name or args.repo_id.split('/')[-1]
            if args.server_side:
                parsed = api.download_model_server_side(
                    repo_id=args.repo_id,
                    name=target_name,
                    scope=args.scope,
                    project_id=args.project_id,
                    staging_dir=args.staging_dir,
                )
            else:
                staging_root = args.staging_dir or os.environ.get('UMRS_MODEL_STAGING_ROOT')
                if not staging_root:
                    from umrs.shared.config import get_default_umrs_cfg_dir
                    staging_root = os.path.join(get_default_umrs_cfg_dir(), 'models')
                target_path = os.path.join(staging_root, target_name)

                if os.path.exists(target_path) and os.listdir(target_path):
                    print(f'Model already exists at {target_path}, skipping download.')
                else:
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError:
                        print(api_json(status='error', message='huggingface_hub not installed. Run: pip install huggingface_hub', output_format=args.output))
                        return 1
                    print(f'Downloading {args.repo_id} to {target_path} ...')
                    os.makedirs(target_path, exist_ok=True)
                    snapshot_download(
                        repo_id=args.repo_id,
                        local_dir=target_path,
                        token=os.environ.get('HF_TOKEN'),
                    )
                    print(f'Download complete: {target_path}')

                parsed = api.register_model(
                    name=target_name,
                    path=target_path,
                    model_type='embedding',
                    scope=args.scope,
                    project_id=args.project_id,
                    huggingface_id=args.repo_id,
                    description=f'Downloaded from {args.repo_id}',
                )
        elif model_action == 'upload':
            import shutil
            src_path = os.path.abspath(args.path)
            if not os.path.isdir(src_path):
                print(api_json(status='error', message=f'Source path is not a directory: {src_path}', output_format=args.output))
                return 1
            staging_root = args.staging_dir or os.environ.get('UMRS_MODEL_STAGING_ROOT')
            if not staging_root:
                from umrs.shared.config import get_default_umrs_cfg_dir
                staging_root = os.path.join(get_default_umrs_cfg_dir(), 'models')
            target_path = os.path.join(staging_root, args.name)
            if os.path.exists(target_path):
                if args.force:
                    shutil.rmtree(target_path)
                else:
                    print(f'Model already exists at {target_path}. Use --force to overwrite.')
                    return 1
            os.makedirs(staging_root, exist_ok=True)
            print(f'Copying {src_path} → {target_path} ...')
            shutil.copytree(src_path, target_path)
            print('Copy complete. Registering model ...')
            register_path = args.register_path or os.path.abspath(target_path)
            parsed = api.register_model(
                name=args.name,
                path=register_path,
                model_type=args.model_type,
                scope=args.scope,
                project_id=args.project_id,
                description=args.description,
            )
        else:
            print(api_json(status='error', message=f'Unknown model action: {model_action}', output_format=args.output))
            return 1

        if args.output == 'json':
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get('status') != 'ok':
                exit_code = 1
        else:
            if parsed.get('status') == 'ok':
                if model_action == 'list':
                    models = parsed.get('models', [])
                    if not models:
                        print(f"  {_dim('No models registered.')}")
                    else:
                        print(_section("Models"))
                        print(
                            "  "
                            + "  ".join(
                                [
                                    _fmt_cell("Name", 30, style=_bold),
                                    _fmt_cell("Type", 10, style=_bold),
                                    _fmt_cell("Scope", 8, style=_bold),
                                    _bold("Storage"),
                                ]
                            )
                        )
                        print(f"  {_dim('-' * 80)}")
                        for m in models:
                            scope_str = m.get('scope', '')
                            print(
                                "  "
                                + "  ".join(
                                    [
                                        _fmt_cell(m['name'], 30),
                                        _fmt_cell(m.get('model_type', ''), 10),
                                        _fmt_cell(scope_str, 8),
                                        _dim(m.get('storage_ref', '')),
                                    ]
                                )
                            )
                else:
                    print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(api_json(status='error', action='model', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_job(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    job_action = args.job_action
    exit_code = 0
    try:
        if job_action == 'submit':
            spec = parse_json_object_arg(args.spec, field_name='spec') if args.spec else {}
            parsed = api.submit_job(
                job_type=args.job_type,
                spec=spec,
                project=args.project,
                max_attempts=args.max_attempts,
                keep_alive=args.keep_alive,
                partition=getattr(args, 'partition', None),
                node=getattr(args, 'node', None),
            )
        elif job_action == 'cancel':
            parsed = api.cancel_job(args.job_id)
        elif job_action == 'status':
            parsed = api.get_job(args.job_id)
        elif job_action == 'list':
            parsed = api.list_jobs(
                project=getattr(args, 'project', None),
                status=getattr(args, 'status', None),
                job_type=getattr(args, 'job_type', None),
                limit=getattr(args, 'limit', 200),
                offset=getattr(args, 'offset', 0),
            )
        else:
            print(api_json(status='error', message=f'Unknown job action: {job_action}', output_format=args.output))
            return 1

        if args.output == 'json':
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get('status') != 'ok':
                exit_code = 1
        else:
            if parsed.get('status') == 'ok':
                if job_action == 'list':
                    jobs = parsed.get('jobs', [])
                    if not jobs:
                        print(f"  {_dim('No jobs found.')}")
                    else:
                        print(_section("Jobs"))
                        print(
                            "  "
                            + "  ".join(
                                [
                                    _fmt_cell("Job ID", 18, style=_bold),
                                    _fmt_cell("Status", 10, style=_bold),
                                    _fmt_cell("Type", 18, style=_bold),
                                    _bold("Project"),
                                ]
                            )
                        )
                        print(f"  {_dim('-' * 70)}")
                        for job in jobs:
                            st = job.get('status', '')
                            if st in ('completed', 'done'):
                                st_str = _green(st)
                            elif st in ('failed', 'cancelled'):
                                st_str = _red(st)
                            elif st in ('running',):
                                st_str = _cyan(st)
                            else:
                                st_str = _yellow(st)
                            print(
                                f"  {job.get('job_id',''):18s}  "
                                f"{st_str}  "
                                f"{job.get('job_type',''):18s}  "
                                f"{_dim(job.get('project_id',''))}"
                            )
                elif job_action == 'status':
                    job = parsed.get('job', {})
                    st = job.get('status', '')
                    if st in ('completed', 'done'):
                        st_str = _green(st)
                    elif st in ('failed', 'cancelled'):
                        st_str = _red(st)
                    elif st in ('running',):
                        st_str = _cyan(st)
                    else:
                        st_str = _yellow(st)
                    print(_section("Job Details"))
                    print(_kv("job_id", _bold(job.get('job_id', ''))))
                    print(_kv("status", st_str))
                    print(_kv("job_type", job.get('job_type', '')))
                    print(_kv("project_id", job.get('project_id', '')))
                else:
                    print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(api_json(status='error', action='job', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_monitor(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    try:
        from umrs.tui.dashboard import Dashboard
    except ImportError:
        ctx.control_mq.close()
        print("The 'rich' library is required for the monitor dashboard.\n"
              "Install it with: pip install 'rich>=13.0'")
        return 1
    try:
        conn_info = ctx.connection.to_dict() if ctx.connection else {}
    except Exception:
        conn_info = {}
    capabilities = _build_monitor_capabilities(
        client_id,
        admin_token=ctx.admin_token,
        user_cfg=ctx.user if isinstance(ctx.user, dict) else None,
    )
    try:
        dash = Dashboard(
            control_mq=ctx.control_mq,
            client_id=client_id,
            interval=args.interval,
            status_filter=args.status,
            project_filter=args.project,
            initial_tab=args.tab,
            config_path=ctx.configure_file if ctx.configure_file and ctx.config_state.user.valid else None,
            user_cfg=ctx.user if isinstance(ctx.user, dict) else None,
            connection_info=conn_info,
            capabilities=capabilities,
        )
        if args.once:
            dash.render_once()
        else:
            dash.run()
    except KeyboardInterrupt:
        pass
    finally:
        ctx.control_mq.close()
    return 0


def _action_workflow(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    wf_action = args.workflow_action
    exit_code = 0
    try:
        if wf_action == 'list':
            parsed = api.list_workflows(scope=args.scope, user=args.user, name=args.name)
        elif wf_action == 'run':
            params = parse_json_object_arg(args.params, field_name='params') if args.params else {}
            parsed = api.run_workflow(workflow_name=args.workflow_name, params=params, bucket=args.bucket)
        else:
            print(api_json(status='error', message=f'Unknown workflow action: {wf_action}', output_format=args.output))
            return 1

        if args.output == 'json':
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get('status') != 'ok':
                exit_code = 1
        else:
            if parsed.get('status') == 'ok':
                if wf_action == 'run':
                    print(f"{_green('✓')} {parsed.get('message', 'OK')}")
                else:
                    workflows = parsed.get('workflows', [])
                    if not workflows:
                        print(f"  {_dim('No workflows registered.')}")
                    else:
                        print(_section("Workflows"))
                        print(
                            "  "
                            + "  ".join(
                                [
                                    _fmt_cell("Name", 30, style=_bold),
                                    _fmt_cell("Version", 8, style=_bold),
                                    _fmt_cell("Scope", 6, style=_bold),
                                    _fmt_cell("User", 12, style=_bold),
                                    _fmt_cell("Steps", 8, style=_bold),
                                    _bold("Description"),
                                ]
                            )
                        )
                        print(f"  {_dim('-' * 100)}")
                        for w in workflows:
                            n_steps = len(w.get('steps', []))
                            print(f"  {w['name']:30s}  v{w.get('version','?'):8s}  {w.get('scope',''):6s}  {w.get('user_id',''):12s}  {n_steps:5d}     {_dim(w.get('description',''))}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(api_json(status='error', action='workflow', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_upload_workflow(ctx):
    import tempfile
    import zipfile
    args = ctx.args
    exit_code = 0
    client_id = _resolve_control_client_id(ctx.user)
    src_path = os.path.abspath(args.path)
    if not os.path.isdir(src_path):
        print(api_json(status='error', message=f'Source path is not a directory: {src_path}', output_format=args.output))
        ctx.control_mq.close()
        return 1
    wf_json = os.path.join(src_path, 'workflow.json')
    if not os.path.isfile(wf_json):
        print(api_json(status='error', message=f'No workflow.json found in {src_path}', output_format=args.output))
        ctx.control_mq.close()
        return 1
    workflow_name = args.name or os.path.basename(src_path)
    tmp_zip_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name
        with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(src_path):
                for f in files:
                    abs_path = os.path.join(root, f)
                    arc_name = os.path.relpath(abs_path, src_path)
                    zf.write(abs_path, arc_name)
        data_mq = ctx.control_mq.get_data_queue(client_id, admin_token=ctx.admin_token)
        try:
            ctx.control_mq.set_timeouts(120_000, 120_000)
            ctx.control_mq.send_string(f"{TaskCmd.UPLOAD_WORKFLOW}:{client_id}:{workflow_name}")
            data_mq.send_file(tmp_zip_path)
            _, reply = ctx.control_mq.recv_string()
            if reply != "ACKNOWLEDGE":
                raise RuntimeError(f"Unexpected reply from server during workflow upload: {reply}")
            if args.output == 'json':
                print(api_json(action='upload-workflow', workflow_name=workflow_name, accepted=True, output_format='json'))
            else:
                print(f"{_green('✓')} Workflow {_bold(workflow_name)} uploaded successfully")
        finally:
            ctx.control_mq.set_timeouts(ctx.control_mq.timeout, ctx.control_mq.timeout)
            with contextlib.suppress(Exception):
                data_mq.close()
    except Exception as e:
        exit_code = 1
        print(api_json(status='error', action='upload-workflow', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        if tmp_zip_path:
            with contextlib.suppress(Exception):
                os.remove(tmp_zip_path)
        ctx.control_mq.close()
    return exit_code


def _action_run_workflow(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    exit_code = 0
    try:
        params = parse_json_object_arg(args.params, field_name='params') if args.params else {}
        parsed = api.run_workflow(workflow_name=args.workflow_name, params=params, bucket=args.bucket)
        if args.output == 'json':
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get('status') != 'ok':
                exit_code = 1
        else:
            if parsed.get('status') == 'ok':
                print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(api_json(status='error', action='run-workflow', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_list_buckets(ctx):
    args = ctx.args
    effective_username = args.username or (ctx.user.get('username') if isinstance(ctx.user, dict) else None)
    if not effective_username:
        print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
        ctx.control_mq.close()
        return 1
    print(action_list_buckets(ctx.buckets_cfg, effective_username, args.output))
    ctx.control_mq.close()
    return 0


def _action_status(ctx):
    args = ctx.args
    effective_username = args.username
    if not effective_username and isinstance(ctx.user, dict):
        effective_username = ctx.user.get('username')
    if not effective_username:
        print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
        ctx.control_mq.close()
        return 1
    api = UMRSControlAPI(ctx.control_mq, effective_username)
    try:
        status_target = _resolve_status_target(args)
        parsed = api.request_status(status_target)
        if args.output == 'json':
            print(_json.dumps(parsed, ensure_ascii=False))
        else:
            print(format_status_text(parsed, status_target))
    except Exception as e:
        print(api_json(status='error', action=args.action, message=str(e), output_format=args.output))
    finally:
        ctx.control_mq.close()
    return 0


def _action_maintenance(ctx):
    args = ctx.args
    client_id = (ctx.user.get('username') if isinstance(ctx.user, dict) else None) or os.environ.get('UMRS_CLIENT_ID', '_')
    api = UMRSControlAPI(
        ctx.control_mq,
        client_id,
        ctx.configure_file if ctx.configure_file and ctx.config_state.user.valid else None,
        admin_token=ctx.admin_token,
    )
    try:
        parsed = api.maintenance(args.action)
        if parsed.get('status') == 'ok':
            if args.output == 'json':
                print(api_json(action=args.action, accepted=True, output_format='json'))
            else:
                print(f"{_green('✓')} {_bold(args.action)} accepted")
        else:
            if args.output == 'json':
                print(api_json(status='error', action=args.action, accepted=False, message=parsed.get('message', 'rejected'), output_format='json'))
            else:
                print(f"{_red('✗')} {_bold(args.action)} rejected: {parsed.get('message', '')}")
    except Exception as e:
        if args.output == 'json':
            print(api_json(status='error', action=args.action, message=str(e), output_format='json'))
        else:
            print(f"{_red('✗')} {_bold(args.action)} error: {e}")
    finally:
        ctx.control_mq.close()
    return 0


def _action_export_embeddings(ctx):
    args = ctx.args
    try:
        bucket_alias_arg = getattr(args, 'bucket_alias')
        bucket_alias, _ = decompose_bucket_alias_arg(bucket_alias_arg)
        entry = ctx.buckets_cfg.get(bucket_alias)
        if entry is None:
            print(api_json(status='error', message=f'Bucket alias not found: {bucket_alias}', output_format=args.output))
            ctx.control_mq.close()
            return 1
        path_to_bucket = entry.path_to_bucket
        effective_username = getattr(args, 'username', None) or (
            ctx.user.get('username') if isinstance(ctx.user, dict) else None
        )
        if not effective_username:
            print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
            ctx.control_mq.close()
            return 1
        raw_response = ctx.control_mq.request_export_embeddings(
            effective_username, path_to_bucket, args.export_format, args.export_profile,
        )
        response = _json.loads(raw_response)
        if response.get('status') != 'ok':
            print(api_json(status='error', action='export_embeddings',
                           message=response.get('message', 'Server returned an error'),
                           output_format=args.output))
            ctx.control_mq.close()
            return 1
        columns = response['columns']
        rows = response['rows']
        local_path = args.local_path
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        if args.export_format == 'csv':
            import csv
            with open(local_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([
                        str(v) if v is not None else '' for v in row
                    ])
        else:  # jsonl
            with open(local_path, 'w', encoding='utf-8') as f:
                for row in rows:
                    record = {}
                    for col, val in zip(columns, row, strict=False):
                        record[col] = val
                    f.write(_json.dumps(record, ensure_ascii=False) + '\n')
        print(api_json(
            action='export_embeddings',
            database=response.get('database', ''),
            table=response.get('table', ''),
            rows_exported=response.get('rows_exported', len(rows)),
            columns=columns,
            local_path=local_path,
            format=args.export_format,
            profile=response.get('profile', args.export_profile),
            output_format=args.output
        ))
    except Exception as e:
        print(api_json(status='error', action='export_embeddings', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return 0


def _action_s3_transfer(ctx):
    """Handle upload_file, upload_dir, and download_results actions."""
    args = ctx.args
    bucket_alias_arg = getattr(args, 'bucket_alias')
    bucket_alias, subdir = decompose_bucket_alias_arg(bucket_alias_arg)

    selected_fields = ["username", "xp_name", "modality_name", "s3_configuration_file", "metadata"]
    if args.action == "download_results":
        selected_fields.append("output_path_to_bucket")
    else:
        selected_fields.append("path_to_bucket")
    _bucket_result = ctx.buckets_cfg.get(bucket_alias, filter=selected_fields)
    if _bucket_result is None:
        raise ValueError(
            f"Bucket alias '{bucket_alias}' not found in configuration. "
            f"Check your config file or use one of the configured aliases."
        )
    username, xp_name, modality, s3_configuration_file, metadata, path_to_bucket = _bucket_result

    username_arg = getattr(args, 'username', None)
    username = username_arg or username

    try:
        data_mq = ctx.control_mq.get_data_queue(username, admin_token=ctx.admin_token)
    except Exception as e:
        print(api_json(status="error", message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
        ctx.control_mq.close()
        return 1

    expanded_cfg = os.path.expanduser(s3_configuration_file)
    default_docker_s3cfg = get_default_docker_s3cfg()
    if not os.path.exists(expanded_cfg) and os.path.exists(default_docker_s3cfg):
        expanded_cfg = default_docker_s3cfg

    s3, bucket_location = connect_to_s3(expanded_cfg)

    if args.action == "download_results":
        full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, output=True, indent=False)
        if full_bucket_name != path_to_bucket:
            ctx.control_mq.close()
            data_mq.close()
            print(
                f"Error:\tNo output bucket for {bucket_alias} for  modality {modality} and for user {username}\n"
                f"\tCheck that this modality has 'need_output_bucket' set to 'true'"
            )
            return 1
        if not success:
            if args.verbose:
                print(f"Note: Output bucket '{full_bucket_name}' does not exist yet on S3 — will wait for server.")
    else:
        full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, indent=False)
        if not success or full_bucket_name != path_to_bucket:
            ctx.control_mq.close()
            data_mq.close()
            print(f"Error:\tNo bucket {bucket_alias} for modality {modality} and for user {username}")
            return 1

    if args.verbose:
        print(f"Bucket '{bucket_alias}' for modality '{modality}' and for user '{username}': OK!")

    try:
        metadata_args = check_and_build_metadata(metadata, args.k)
    except Exception as e:
        print(api_json(status="error", message=f"Metadata error: {e}", output_format=args.output))
        ctx.control_mq.close()
        data_mq.close()
        return 1

    if args.verbose:
        action_label = "Retreiving" if args.action == "download_results" else "Storing"
        print(f"{action_label} data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

    def process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=True):
        if not os.path.isfile(file_path):
            print(f"Error: file {file_path} is not a valid.")
            return None, 0

        category = modality_category(modality)
        print(f"Processing file '{file_path}' for modality '{modality}' (category: {category})")

        if category == "image":
            s3_key = s3_store_image(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args, check=(modality not in ("volume", "movie", "volume_movie")))
            if s3_key and notify_server:
                ctx.control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
        elif category == "text_file":
            s3_key = s3_store_text_file(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args)
            if s3_key and notify_server:
                ctx.control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
        elif category == "text":
            try:
                with open(file_path) as file:
                    nb_objects = 0
                    s3_key_list = []
                    for line in file.readlines():
                        line = line.strip()
                        if line != "":
                            s3_key = s3_store_txt(s3, full_bucket_name, subdir, line, data_mq, metadata_args)
                            s3_key_list.append(s3_key)
                            nb_objects += 1
                    if notify_server and nb_objects > 0:
                        ctx.control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
                    return s3_key_list, nb_objects
            except UnicodeDecodeError as e:
                print(f"Unicode decoding error: {e}")
                print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file.")
        else:
            print(f"Error: Unknown modality category '{category}' (modality: '{modality}'). Nothing has been sent to the server")

        return None, 0

    def process_dir(path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username):
        if not os.path.isdir(path):
            print(f"Error: {path} is not a valid directory.")
            return [], []

        s3_key_list = []
        file_list = []
        nb_objects = 0
        for file_path in glob(os.path.join(path, "*")):
            if os.path.isfile(file_path):
                s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=False)
                if s3_key:
                    if isinstance(s3_key, list):
                        s3_key_list.extend(s3_key)
                        assert(len(s3_key) == nb_sent)
                        file_list.extend([file_path] * nb_sent)
                    else:
                        s3_key_list.append(s3_key)
                        file_list.append(file_path)
                    nb_objects += nb_sent

        if nb_objects > 0:
            ctx.control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
        return s3_key_list, file_list

    def download_results_fn(full_bucket_name, modality, subdir):
        effective_username = args.username
        verbose = args.verbose
        if not effective_username and isinstance(ctx.user, dict):
            effective_username = ctx.user.get('username')
        if not effective_username:
            response = {
                "status": "error",
                "message": f"Cannot find effective username for {args.username}",
                "bucket": full_bucket_name,
                "subdir": subdir,
                "modality": modality,
                "output_format": args.output,
            }
            return False, response

        if not os.path.exists(args.local_path):
            response = {
                "status": "error",
                "message": f"Local path {args.local_path} does not exists",
                "bucket": full_bucket_name,
                "subdir": subdir,
                "modality": modality,
                "output_format": args.output,
            }
            return False, response

        done = False
        print_only_new = False
        objects_received = set()
        timeout = 1
        _wait_start = time.time()
        while not done:
            raw_response = ctx.control_mq.request_status(effective_username, TaskCmd.PROCESS_OBJECTS)
            if raw_response is None:
                break
            parsed = _json.loads(raw_response)

            message = parsed.get("message")
            if message == "No status entries":
                done = True
            else:
                data = parsed["summary"]["data"]
                done = not any(
                    entry["status"] != "Done"
                    and any(
                        full_bucket_name.startswith(w["bucket"])
                        for w in entry["work"]
                    )
                    for entry in data
                )
            if not done:
                if verbose:
                    elapsed = int(time.time() - _wait_start)
                    print(f"Waiting for update... (elapsed: {elapsed // 60}m {elapsed % 60}s)")
                res = ctx.control_mq.wait_for_new_data(effective_username, full_bucket_name, timeout=timeout)
                if res == "TIMEOUT":
                    timeout = min(timeout * 2, 64)
                elif res == "UPDATE":
                    timeout = 1
                else:
                    if verbose:
                        print(f"Error: receiving '{res}' from server while waiting to be notified for new data")
        objects_list, _, checksums = list_s3_bucket(s3, full_bucket_name, prefix=subdir)
        downloaded_files, _ = download_objects(s3, objects_list, checksums, args.local_path,
                            files_allready_received=objects_received,
                            force_overwrite=args.force_overwrite, print_only_new=print_only_new)

        objects_received.update(downloaded_files)

        response = {
                "status": "ok",
                "message": f"Files downloaded with overwrite={args.force_overwrite}",
                "bucket": full_bucket_name,
                "local_path": args.local_path,
                "subdir": subdir,
                "modality": modality,
                "output_format": args.output,
                "objects_received": f"{sorted(objects_received)}",
            }
        return True, response

    try:
        if args.action == "upload_file":
            s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, args.file_path, modality, data_mq, metadata_args, username, notify_server=True)
            if s3_key:
                print(api_json(s3_key=s3_key, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message=f"Cannot process file: {args.file_path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))

        elif args.action == "upload_dir":
            s3_key_list, file_list = process_dir(args.path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username)
            if len(s3_key_list) > 0:
                print(api_json(s3_key_list=s3_key_list, file_list=file_list, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message=f"Cannot process directory: {args.path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
        elif args.action == "download_results":
            ok, response = download_results_fn(full_bucket_name, modality, subdir)
            print(api_json(**response))
    except Exception as e:
        print(api_json(status='error', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()

    ctx.control_mq.close()
    data_mq.close()
    return 0


# Action dispatch table — maps CLI action name to handler function
_ACTION_HANDLERS = {
    'ping': _action_ping,
    'model': _action_model,
    'job': _action_job,
    'monitor': _action_monitor,
    'workflow': _action_workflow,
    'upload-workflow': _action_upload_workflow,
    'run-workflow': _action_run_workflow,
    'list_buckets': _action_list_buckets,
    'status': _action_status,
    'recompute': _action_maintenance,
    'update': _action_maintenance,
    'check': _action_maintenance,
    'reload': _action_maintenance,
    'export_embeddings': _action_export_embeddings,
    'upload_file': _action_s3_transfer,
    'upload_dir': _action_s3_transfer,
    'download_results': _action_s3_transfer,
}


def main():
    parser = argparse.ArgumentParser(
        description="UMRS client",
        epilog=_CLIENT_EPILOG,
        formatter_class=_ClientHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", help="API action", required=True)

    # Global / legacy flags
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument('-c', '--conf', default=None, help='User config JSON path for this invocation (overrides the selected user*.json file)')
    parser.add_argument('--server-conf', default=None, help='Server config JSON path for this invocation (overrides the selected conf_umrs.json file)')
    parser.add_argument('--s3cfg', default=None, help='S3 config file path for this invocation (overrides the selected s3cfg file)')
    parser.add_argument('--profile', default=None, help='Named connection profile to use from the UMRS config directory')
    parser.add_argument("-k", action="append", help="Metadatata key:value", required=False)
    parser.add_argument('--no-check-server', action='store_true', help='Do not check if the server is ready. Just copy data in the S3 server if no ready')
    parser.add_argument('--server-ip', default=None, help='Server IP for ZeroMQ control channel')
    parser.add_argument('--control-port', type=int, default=None, help='Control port for the ZeroMQ control channel')
    parser.add_argument('--output', choices=['json','text'], default='text', help='Output format for API actions')
    parser.add_argument('--admin-token', default=None,
                        help='Admin token for privileged commands (or set UMRS_ADMIN_TOKEN)')
    # Removed deprecated auto-add-columns behavior; metadata must be declared in server config

    completion_p = subparsers.add_parser('completion', help='Shell completion setup')
    comp_sub = completion_p.add_subparsers(dest='completion_action')
    comp_sub.add_parser('bash', help='Print bash completion script')
    comp_sub.add_parser('zsh', help='Print zsh completion script')
    inst_p = comp_sub.add_parser('install', help='Install completion into your shell RC file')
    inst_p.add_argument('--shell', choices=('bash', 'zsh'), default=None)
    uninst_p = comp_sub.add_parser('uninstall', help='Remove completion from your shell RC file')
    uninst_p.add_argument('--shell', choices=('bash', 'zsh'), default=None)

    # API subcommands
    ping_p = subparsers.add_parser('ping', help='Check server readiness')
    configure_ping_parser(ping_p)

    ui_p = subparsers.add_parser('upload_file', help="Upload file to a bucket (ommit empty lines in case of 'text' modality)")
    ui_p.add_argument('file_path')
    ui_p.add_argument(
        'bucket_alias',
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        )
    )


    ui_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

    utf_p = subparsers.add_parser('upload_dir', help='Upload all files of a directory to a bucket')
    utf_p.add_argument('path')
    utf_p.add_argument(
        'bucket_alias',
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        )
    )
    utf_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

    dr_p = subparsers.add_parser('download_results', help="Download file results computed by the server")
    dr_p.add_argument(
        'bucket_alias',
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        )
    )
    dr_p.add_argument(
        'local_path',
        help=(
            "local Unix path where the results will be stored.\n"
            "New files will erase local files."
        )
    )

    dr_p.add_argument(
        '--force-overwrite',
        action='store_true',
        default=False,
        help="Force overwriting existing local files even if checksums mismatch."
    )
    dr_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

    # New actions
    lb_p = subparsers.add_parser('list_buckets', help='List bucket aliases for a username')
    lb_p.add_argument('username', nargs='?', help='Username to filter buckets (defaults to config if omitted)')

    status_p = subparsers.add_parser('status', help='Fetch the latest server summary for maintenance commands')
    configure_status_parser(status_p, include_username=True)

    # Embedding maintenance actions
    subparsers.add_parser('recompute', help='Recompute embeddings for your own buckets (admin: all users)')
    subparsers.add_parser('update', help='Update embeddings for missing/mismatched entries in your own buckets (admin: all users)')
    subparsers.add_parser('check', help='Check consistency between S3 objects and DB (admin-only: checks all users)')
    subparsers.add_parser('reload', help='Reload server configuration (users, buckets, models) without restart (admin-only); if -c points to a user JSON, it will be uploaded and persisted on the server before reloading')

    # Export embeddings from pgvector (server-mediated)
    ee_p = subparsers.add_parser('export_embeddings', help='Export computed embeddings to a local file (via server)')
    ee_p.add_argument(
        'bucket_alias',
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user."
        )
    )
    ee_p.add_argument(
        'local_path',
        help='Local file path to write the exported embeddings'
    )
    ee_p.add_argument(
        '--format', choices=['csv', 'jsonl'], default='csv', dest='export_format',
        help='Output format (default: csv)'
    )
    ee_p.add_argument(
        '--profile', choices=['full', 'minimal', 'embedding-only'], default='minimal', dest='export_profile',
        help='Export profile: minimal keeps embedding plus essential metadata; embedding-only keeps only the embedding vector'
    )
    ee_p.add_argument('username', nargs='?', help='Configured username owning the bucket (defaults to config if omitted)')

    # --- Model registry commands ---
    model_p = subparsers.add_parser('model', help='Model registry commands (register, list, download, upload)')
    model_sub = model_p.add_subparsers(dest='model_action', required=True)

    # model register
    mr_p = model_sub.add_parser('register', help='Register a local model with UMRS')
    configure_model_register_parser(mr_p)

    # model list
    ml_p = model_sub.add_parser('list', help='List registered models')
    configure_model_list_parser(ml_p)

    # model download (default: client-side; --server-side sends to server as background task)
    md_p = model_sub.add_parser('download', help='Download model from HuggingFace and register with UMRS')
    configure_model_download_parser(md_p)
    md_p.add_argument('--staging-dir', default=None,
                       help='Directory to store downloaded model (default: ~/.config/umrs/models/ or UMRS_MODEL_STAGING_ROOT)')
    md_p.add_argument('--server-side', action='store_true', default=False,
                       help='Download on the server machine instead of locally (requires server internet access)')

    # model upload (copy local directory to staging root, then register)
    mu_p = model_sub.add_parser('upload', help='Copy a local model directory to the staging root and register with UMRS')
    mu_p.add_argument('path', help='Local directory containing model files')
    mu_p.add_argument('--name', required=True, help='Model name for registration')
    mu_p.add_argument('--type', choices=MODEL_TYPE_CHOICES, default='embedding', dest='model_type', help='Model type')
    mu_p.add_argument('--scope', choices=MODEL_SCOPE_CHOICES, default='user', help='Model scope')
    mu_p.add_argument('--project-id', default=None, help='Project identifier (required for project scope)')
    mu_p.add_argument('--description', default=None, help='Model description')
    mu_p.add_argument('--staging-dir', default=None,
                       help='Staging root directory (default: UMRS_MODEL_STAGING_ROOT or ~/.config/umrs/models/)')
    mu_p.add_argument('--register-path', default=None,
                       help='Server-side path to register (use when client and server have different filesystems, e.g. Mac → HPC)')
    mu_p.add_argument('--force', action='store_true', default=False,
                       help='Overwrite existing model directory in staging root')

    # --- Job tracking commands ---
    job_p = subparsers.add_parser('job', help='Job commands (submit, status, list, cancel)')
    job_sub = job_p.add_subparsers(dest='job_action', required=True)

    # job submit
    js_p = job_sub.add_parser('submit', help='Submit a new job')
    configure_job_submit_parser(js_p)

    # job cancel
    jc_p = job_sub.add_parser('cancel', help='Cancel a running job')
    configure_job_cancel_parser(jc_p)

    # job status
    jst_p = job_sub.add_parser('status', help='Show a specific durable job')
    configure_job_status_parser(jst_p)

    # job list
    jl_p = job_sub.add_parser('list', help='List durable jobs with optional filters')
    configure_job_list_parser(jl_p)

    # --- Workflow upload command ---
    uw_p = subparsers.add_parser('upload-workflow', help='Upload a workflow directory (with workflow.json + step scripts) to the server')
    uw_p.add_argument('path', help='Local directory containing workflow.json and step scripts')
    uw_p.add_argument('--name', default=None, help='Workflow name (defaults to directory basename)')

    # --- Run workflow command ---
    rw_p = subparsers.add_parser('run-workflow', help='Trigger execution of a registered workflow on the server')
    configure_workflow_run_parser(rw_p)

    # --- Workflow registry commands ---
    wf_p = subparsers.add_parser('workflow', help='Workflow registry commands (list, run)')
    wf_sub = wf_p.add_subparsers(dest='workflow_action', required=True)

    # workflow list
    wl_p = wf_sub.add_parser('list', help='List registered workflows')
    configure_workflow_list_parser(wl_p)

    # workflow run
    wr_p = wf_sub.add_parser('run', help='Trigger execution of a registered workflow on the server')
    configure_workflow_run_parser(wr_p)

    # --- Monitor (TUI dashboard) ---
    mon_p = subparsers.add_parser('monitor', help='Live-refreshing multi-tab system monitor (TUI)')
    mon_p.add_argument('--interval', type=float, default=3.0, help='Refresh interval in seconds')
    mon_p.add_argument('--status', default=None,
                       choices=['running', 'queued', 'planned', 'completed', 'failed', 'cancelled'],
                       help='Initial status filter (Jobs tab)')
    mon_p.add_argument('--project', default=None, help='Filter by project ID (Jobs tab)')
    mon_p.add_argument('--once', action='store_true', default=False,
                       help='Print dashboard once and exit (no live refresh)')
    mon_p.add_argument('--tab', default='jobs',
                       choices=['jobs', 'status', 'models', 'workflows', 'connection'],
                       help='Tab to display on startup (default: jobs)')

    # --- Disabled admin command ---
    admin_p = subparsers.add_parser(
        'admin',
        help='Disabled; use umrs-server admin on the server',
        description=_disabled_admin_message(),
    )
    admin_p.add_argument('admin_args', nargs='*', help=argparse.SUPPRESS)

    # --- login: unified auth + profile + config setup ---
    login_p = subparsers.add_parser(
        'login',
        help='Authenticate and configure connection in one step',
    )
    login_p.add_argument('--host', default='127.0.0.1', help='UMRS server host')
    login_p.add_argument('--port', type=int, default=5555, help='UMRS ZMQ control port')
    login_p.add_argument('--api-key', '-k', help='JWT or onboarding token')
    login_p.add_argument('--mode', default='direct', choices=['direct', 'ssh_tunnel'],
                         help='Connection mode')
    login_p.add_argument('--username', default=None, help='UMRS username (auto-extracted from JWT)')
    login_p.add_argument('--profile-name', default=None, help='Connection profile name (default: auto)')

    profile_p = subparsers.add_parser('profile', help='Connection profile commands (init, list, show, use, import)')
    profile_sub = profile_p.add_subparsers(dest='profile_action', required=True)
    pn_p = profile_sub.add_parser('init', help='Create a new connection profile and set it as current')
    pn_p.add_argument('--name', default=None, help='Profile name (prompted if omitted)')
    pn_p.add_argument('--mode', default='direct', choices=['direct', 'ssh_tunnel'], help='Connection mode (default: direct)')
    pn_p.add_argument('--host', default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    pn_p.add_argument('--control-port', type=int, default=5555, help='ZMQ control port (default: 5555)')
    pn_p.add_argument('--data-port-start', type=int, default=5556, help='Data port range start (default: 5556)')
    pn_p.add_argument('--data-port-end', type=int, default=5563, help='Data port range end (default: 5563)')
    pn_p.add_argument('--minio-url', default=None, help='MinIO URL (default: http://<host>:9000)')
    pn_p.add_argument('--rag-url', default=None, help='RAG URL (default: http://<host>:8081)')
    pn_p.add_argument('--ssh-host', default=None, help='SSH host for tunnel mode')
    pn_p.add_argument('--force', action='store_true', default=False, help='Overwrite if the profile already exists')
    profile_sub.add_parser('list', help='List available connection profiles')
    profile_sub.add_parser('show', help='Show the effective connection resolved from profile/env/flags')
    pu_p = profile_sub.add_parser('use', help='Set the current default connection profile')
    pu_p.add_argument('name', help='Profile name to persist as the current default')
    pi_p = profile_sub.add_parser('import', help='Import a connection profile JSON file into the UMRS config directory')
    pi_p.add_argument('file', help='Path to a connection profile JSON file')
    pi_p.add_argument('--force', action='store_true', default=False, help='Overwrite an existing profile with the same name')

    config_p = subparsers.add_parser('config', help='Loaded config file commands (init, list, show, use)')
    config_sub = config_p.add_subparsers(dest='config_action', required=True)
    ci_p = config_sub.add_parser('init', help='Create a new user config JSON and set it as current')
    ci_p.add_argument('--username', default=None, help='Username for the new config (prompted if omitted)')
    ci_p.add_argument('--fullname', default=None, help='Full name of the user (prompted if omitted)')
    ci_p.add_argument('--output-path', default=None, help='Where to write the JSON (default: ~/.config/umrs/users/<username>.json)')
    ci_p.add_argument('--template', default=None, help='Path to an existing user JSON to use as template')
    ci_p.add_argument('--force', action='store_true', default=False, help='Overwrite if the output file already exists')
    config_sub.add_parser('list', help='List discovered conf_umrs.json, user*.json, and s3cfg candidates')
    config_sub.add_parser('show', help='Show the resolved config stack for server, user, and s3 files')
    cu_p = config_sub.add_parser('use', help='Persist the current server/user/s3 config selections')
    cu_p.add_argument('--server', default=None, help='Server config file path or unique basename to select')
    cu_p.add_argument('--user', default=None, help='User config file path or unique basename to select')
    cu_p.add_argument('--s3', default=None, help='S3 config file path or unique basename to select')

    if emit_completion_if_requested(parser, 'umrs-client', '_UMRS_CLIENT_COMPLETE'):
        return 0

    args = parser.parse_args()

    if args.action == 'completion':
        action = getattr(args, 'completion_action', None)
        if action == 'install':
            print(install_completion('umrs-client', shell=args.shell))
            return 0
        if action == 'uninstall':
            print(uninstall_completion('umrs-client', shell=args.shell))
            return 0
        if action in ('bash', 'zsh'):
            print_completion_script('umrs-client', '_UMRS_CLIENT_COMPLETE', action)
            return 0
        print_completion_script('umrs-client', '_UMRS_CLIENT_COMPLETE', 'bash')
        return 0

    if args.action == 'profile':
        try:
            sys.exit(_handle_profile_command(args))
        except Exception as e:
            print(api_json(status='error', message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == 'login':
        try:
            sys.exit(_handle_login(args))
        except Exception as e:
            print(api_json(status='error', message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == 'config':
        try:
            sys.exit(_handle_config_command(args))
        except Exception as e:
            print(api_json(status='error', message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == 'admin':
        print(api_json(status='error', message=_disabled_admin_message(), output_format=args.output))
        sys.exit(2)

    connection = resolve_connection(
        explicit_profile_name=args.profile,
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )


    # Connect to message queue
    control_mq = MessageQueue(
        role = "client",
        queue_type = 'control',
        server_ip=connection.server_host,
        port=connection.control_port,
        tunnel_mode=connection.tunnel_mode,
        prefer_env_server_host=False,
    )

    if (
        not args.no_check_server
        and args.action != 'ping'
        and not control_mq.is_server_ready(timeout_ms=connection.health_check_timeout_ms)
    ):
        sys.exit(f"{_red('✗')} Server connection timeout — cannot continue!")

    if args.action != 'ping':
        profile_name = getattr(connection, "profile_name", None)
        profile_label = _bold(profile_name) if profile_name else _dim("(none)")
        print(f"{_green('✓')} Server ready  {_dim('|')}  profile={profile_label}  {_dim('|')}  {connection.server_host}:{connection.control_port}")

    # Resolve admin token from CLI flag or environment
    admin_token = args.admin_token or os.environ.get("UMRS_ADMIN_TOKEN")

    # Deprecated 'auto_add_columns' flag removed

    config_state = resolve_loaded_config(
        explicit_server_config_path=args.server_conf,
        explicit_user_config_path=args.conf,
        explicit_s3_config_path=args.s3cfg,
    )
    configure_file = config_state.user.path if config_state.user.path else None

    # Only enforce config file for legacy mode or actions that need bucket/user config
    status_needs_config = args.action == 'status' and not getattr(args, 'username', None)
    requires_config =  args.action in {
        'list_buckets', 'upload_file', 'upload_dir', "download_results",
        "recompute", "update", "reload", "export_embeddings"
    } or status_needs_config

    explicit_s3_selected = bool(
        config_state.s3.source and not str(config_state.s3.source).startswith('user:')
    )

    if requires_config and not config_state.user.valid:
        if configure_file:
            sys.exit(
                f"Configuration file not usable: {configure_file}\n"
                f"{config_state.user.error}"
            )
        search_locations = "\n  ".join(get_user_config_search_locations())
        sys.exit(
            f"Configuration file not found.\n"
            f"Provide one with -c/--conf or place it in one of:\n  {search_locations}"
        )

    if requires_config and explicit_s3_selected and config_state.s3.path and not config_state.s3.valid:
        sys.exit(
            f"S3 config file not usable: {config_state.s3.path}\n"
            f"{config_state.s3.error}"
        )

    user = None
    buckets_cfg = None
    if requires_config:
        if args.verbose or args.conf is None:
            print(f"Using config: {configure_file}")
        user = load_client_cfg_file(configure_file)
        if config_state.s3.valid and config_state.s3.path:
            user = apply_s3_config_override(user, config_state.s3.path)
        buckets_cfg = build_buckets_cfg(user)
        if args.verbose:
            print("Configuration loaded successfully.")
    elif configure_file and config_state.user.valid:
        # Opportunistically load config for commands that benefit from
        # knowing the username (e.g. upload-workflow) but don't require it.
        try:
            user = load_client_cfg_file(configure_file)
            if config_state.s3.valid and config_state.s3.path:
                user = apply_s3_config_override(user, config_state.s3.path)
        except (Exception, SystemExit):
            pass

    # Build shared context and dispatch to the appropriate action handler
    ctx = _ClientContext(
        args=args,
        control_mq=control_mq,
        connection=connection,
        admin_token=admin_token,
        config_state=config_state,
        configure_file=configure_file,
        user=user,
        buckets_cfg=buckets_cfg,
    )
    handler = _ACTION_HANDLERS.get(args.action)
    if handler is None:
        print(api_json(status='error', message=f'Unknown action: {args.action}', output_format=args.output))
        control_mq.close()
        sys.exit(1)
    sys.exit(handler(ctx))

if __name__ == "__main__":
    main()
