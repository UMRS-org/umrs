#!/usr/bin/env python3

import argparse
import contextlib
import json as _json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from umrs.cli.completion import (
    emit_completion_if_requested,
    install_completion,
    print_completion_script,
    uninstall_completion,
)
from umrs.client.control_api import UMRSControlAPI
from umrs.client.control_commands import (
    MODEL_SCOPE_CHOICES,
    MODEL_TYPE_CHOICES,
    configure_endpoint_list_parser,
    configure_endpoint_status_parser,
    configure_endpoint_stop_parser,
    configure_endpoint_up_parser,
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
    parse_duration_seconds,
    parse_json_object_arg,
)
from umrs.client.download_results import (
    classify_wait_reply,
    format_wait_progress,
    incomplete_error_payload,
    parse_process_objects_readiness,
)
from umrs.client.metadata import check_and_build_metadata
from umrs.client.warnings_cli import emit_cli_warning
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.shared.auth import create_auth_context_from_request
from umrs.shared.buckets import BucketsCfg, ClientBucketEntry
from umrs.shared.config import (
    default_s3cfg_path,
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
from umrs.shared.utils import utc_to_local
from umrs.shared.warnings import operation_ignored_warning, print_warning_structured

# ---------------------------------------------------------------------------
# Optional heavy transport dependencies (lazy)
#
# The ZMQ control/data transport (pyzmq) and the S3 storage helpers (boto3,
# pillow) are only needed for direct-transport operations: the ZMQ control
# channel and bucket upload/download. They are imported lazily so the CLI can be
# installed in a lightweight client configuration (``pip install umrs[client]``)
# that only needs httpx/pyyaml/rich for gateway-mediated operations. Commands
# that genuinely require the heavy transport raise a friendly error pointing at
# the ``umrs[client-full]`` extra instead of an opaque ImportError.
#
# The wrappers below preserve the original call signatures so the rest of this
# module can keep calling ``MessageQueue(...)``, ``connect_to_s3(...)`` etc.
# unchanged.
# ---------------------------------------------------------------------------

_CLIENT_FULL_HINT = (
    "Install the full client transport stack with:\n    pip install 'umrs[client-full]'"
)


def _require_client_full(feature, exc):
    """Abort with a friendly message when a heavy optional dependency is missing."""
    raise SystemExit(
        f"Error: '{feature}' requires the full UMRS client transport stack "
        f"(pyzmq/boto3), which is not installed in this lightweight client.\n"
        f"{_CLIENT_FULL_HINT}\n"
        f"(missing dependency: {exc})"
    )


def _message_queue_cls():
    """Lazily import the ZMQ MessageQueue class (requires ``umrs[client-full]``)."""
    try:
        from umrs.core.MessageQueue import MessageQueue as _MessageQueue

        return _MessageQueue
    except ImportError as exc:  # pragma: no cover - depends on install extra
        _require_client_full("ZMQ transport", exc)


def MessageQueue(*args, **kwargs):
    """Construct a ZMQ MessageQueue via the lazily-imported class."""
    return _message_queue_cls()(*args, **kwargs)


def _s3():
    """Lazily import the S3 storage helpers (requires ``umrs[client-full]``)."""
    try:
        import umrs.shared.storage.s3 as _s3_mod

        return _s3_mod
    except ImportError as exc:  # pragma: no cover - depends on install extra
        _require_client_full("S3 storage", exc)


def check_bucket_name(*args, **kwargs):
    return _s3().check_bucket_name(*args, **kwargs)


def connect_to_s3(*args, **kwargs):
    return _s3().connect_to_s3(*args, **kwargs)


def decompose_bucket_alias_arg(*args, **kwargs):
    return _s3().decompose_bucket_alias_arg(*args, **kwargs)


def download_objects(*args, **kwargs):
    return _s3().download_objects(*args, **kwargs)


def list_s3_bucket(*args, **kwargs):
    return _s3().list_s3_bucket(*args, **kwargs)


def make_path_to_bucket_string(*args, **kwargs):
    return _s3().make_path_to_bucket_string(*args, **kwargs)


def s3_store_image(*args, **kwargs):
    return _s3().s3_store_image(*args, **kwargs)


def s3_store_text_file(*args, **kwargs):
    return _s3().s3_store_text_file(*args, **kwargs)


def s3_store_txt(*args, **kwargs):
    return _s3().s3_store_txt(*args, **kwargs)


# ---------------------------------------------------------------------------
# ANSI helpers — auto-disabled when stdout is not a terminal
# ---------------------------------------------------------------------------
def _use_color():
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code, text):
    """Wrap *text* in ANSI escape if color is enabled."""
    if not _use_color():
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def _bold(t):
    return _c("1", t)


def _dim(t):
    return _c("2", t)


def _green(t):
    return _c("32", t)


def _red(t):
    return _c("31", t)


def _yellow(t):
    return _c("33", t)


def _cyan(t):
    return _c("36", t)


def _blue(t):
    return _c("34", t)


def _magenta(t):
    return _c("35", t)


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


def _fmt_cell_raw(styled_text, width):
    """Pad a string that may contain ANSI codes to a visual width."""
    import re as _re_mod

    visible = _re_mod.sub(r"\x1b\[[0-9;]*m", "", styled_text)
    pad = max(0, width - len(visible))
    return styled_text + " " * pad


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
    return os.environ.get("UMRS_DEV_MODE", "") == "1"


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
                lines.append(
                    _kv(
                        "local_forward",
                        f"{fwd.get('local_port')}:{fwd.get('remote_host', 'localhost')}:{fwd.get('remote_port')}",
                    )
                )
        if connection.command_hint:
            lines.append(_kv("command_hint", _dim(connection.command_hint)))
    if errors:
        lines.append("")
        for err in errors:
            lines.append(f"  {_red('!')} {err}")
    if hasattr(connection, "sources") and connection.sources:
        # Only show non-profile sources (env overrides, defaults) to reduce noise
        profile_name = getattr(connection, "profile_name", None)
        profile_tag = f"profile:{profile_name}" if profile_name else None
        overrides = {k: v for k, v in connection.sources.items() if v != profile_tag}
        if overrides:
            lines.append("")
            lines.append(_section("Overrides"))
            for src_key, src_val in sorted(overrides.items()):
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


# ---------------------------------------------------------------------------
# umrs info — one-shot connection diagnostic
# ---------------------------------------------------------------------------
def _handle_info(args):
    """Show transport, auth, and S3 status at a glance."""
    import configparser as _cp

    lines: list[str] = []

    # --- Connection profile ---
    connection = resolve_connection(
        explicit_profile_name=args.profile,
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )
    prof = connection.profile_name or "(none)"
    lines.append(_section("Connection Profile"))
    lines.append(_kv("profile", _bold(prof)))
    lines.append(_kv("mode", connection.mode))
    lines.append(_kv("profile_path", _dim(connection.profile_path or "n/a")))

    # --- Transport ---
    lines.append("")
    lines.append(_section("Transport"))

    gateway_url = getattr(connection, "gateway_url", None)
    zmq_host = connection.server_host
    zmq_port = connection.control_port
    zmq_auth = getattr(connection, "zmq_auth_type", None) or "none"
    tunnel = connection.tunnel_mode

    lines.append(_kv("zmq_host", f"{zmq_host}:{zmq_port}"))
    lines.append(_kv("zmq_auth_type", zmq_auth))
    lines.append(_kv("tunnel_mode", str(tunnel).lower()))
    lines.append(_kv("data_port_range", connection.data_port_range_label))

    # Which actions use which transport?
    gateway_actions = {"reload", "list_buckets", "model list"} if gateway_url else set()
    zmq_actions = {
        "ping",
        "status",
        "upload_file",
        "upload_dir",
        "download_results",
        "recompute",
        "update",
        "check",
        "export_embeddings",
        "job *",
        "endpoint *",
        "workflow *",
        "monitor",
    }
    lines.append("")
    lines.append(
        _kv(
            "gateway_actions",
            ", ".join(sorted(gateway_actions)) if gateway_actions else _dim("none"),
        )
    )
    lines.append(
        _kv("zmq_actions", _dim("all others (ping, upload, jobs, endpoints, ...)"))
    )

    # --- ZMQ reachability ---
    lines.append("")
    lines.append(_section("ZMQ Status"))
    try:
        # Load CURVE session so ZMQ check uses auth if available
        _maybe_load_saved_curve_session()
        mq = MessageQueue(
            role="client",
            queue_type="control",
            server_ip=zmq_host,
            port=zmq_port,
            tunnel_mode=tunnel,
            prefer_env_server_host=False,
        )
        ready = mq.is_server_ready(
            timeout_ms=min(connection.health_check_timeout_ms, 3000)
        )
        mq.close()
        # Session-mode failures are often a stale session (server restarted and
        # dropped its in-memory registry). Try a one-shot fresh registration so
        # `info` self-heals and reports the true reachability.
        if (
            not ready
            and _is_session_mode(connection)
            and _force_refresh_curve_session(connection)
        ):
            mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip=zmq_host,
                port=zmq_port,
                tunnel_mode=tunnel,
                prefer_env_server_host=False,
            )
            ready = mq.is_server_ready(
                timeout_ms=min(connection.health_check_timeout_ms, 3000)
            )
            mq.close()
        if ready:
            lines.append(
                _kv(
                    "server",
                    f"{_green('● reachable')}  {_dim(f'{zmq_host}:{zmq_port}')}",
                )
            )
        else:
            lines.append(
                _kv(
                    "server",
                    f"{_red('● not responding')}  {_dim(f'{zmq_host}:{zmq_port}')}",
                )
            )
    except Exception as exc:
        lines.append(_kv("server", f"{_red('● error')}  {_dim(str(exc)[:80])}"))

    # --- CURVE session ---
    lines.append("")
    lines.append(_section("CURVE Session"))
    try:
        import time as _time
        from pathlib import Path as _Path

        xdg = os.environ.get("XDG_CONFIG_HOME")
        curve_path = (
            _Path(xdg, "r4") if xdg else _Path.home() / ".config" / "r4"
        ) / "curve-session.json"
        if curve_path.is_file():
            data = _json.loads(curve_path.read_text(encoding="utf-8"))
            expires_at = data.get("expires_at", 0)
            ttl = expires_at - _time.time()
            if ttl > 60:
                lines.append(_kv("status", _green("active")))
                lines.append(
                    _kv(
                        "ttl",
                        f"{int(ttl)}s ({int(ttl / 3600)}h {int((ttl % 3600) / 60)}m)",
                    )
                )
            elif ttl > 0:
                lines.append(_kv("status", _yellow("expiring soon")))
                lines.append(_kv("ttl", f"{int(ttl)}s"))
            else:
                lines.append(_kv("status", _red("expired")))
            lines.append(_kv("file", _dim(str(curve_path))))
        else:
            lines.append(_kv("status", _dim("no session file")))
            lines.append(_kv("file", _dim(str(curve_path))))
    except Exception as exc:
        lines.append(_kv("status", f"{_red('error')}  {_dim(str(exc)[:60])}"))

    # --- HTTP Gateway ---
    lines.append("")
    lines.append(_section("HTTP Gateway"))
    if gateway_url:
        api_key = _get_api_key_from_profile_static(connection)
        ping_result = _http_ping(gateway_url, api_key)
        gw_ready = ping_result.get("server_ready", False)
        if gw_ready:
            lines.append(_kv("status", f"{_green('● reachable')}  {_dim(gateway_url)}"))
        else:
            msg = ping_result.get("message", "")
            lines.append(
                _kv("status", f"{_red('● not reachable')}  {_dim(gateway_url)}")
            )
            if msg:
                lines.append(_kv("error", _dim(msg[:80])))
        lines.append(
            _kv("auth_token", _green("present") if api_key else _red("missing"))
        )
    else:
        lines.append(_kv("status", _dim("(no gateway_url configured)")))

    # --- Auth / JWT ---
    lines.append("")
    lines.append(_section("Auth"))
    profile_name = getattr(connection, "profile_name", None)

    # Collect all token sources
    jwt = None
    jwt_source = None

    # 1) r4 tokens.json (legacy per-profile store)
    r4_token = _try_read_r4_token(profile_name)
    if r4_token:
        jwt = r4_token
        jwt_source = f"tokens.json (profile={profile_name})"

    # 2) r4 sources.json (active source api_key)
    if not jwt:
        try:
            from pathlib import Path as _Path

            xdg = os.environ.get("XDG_CONFIG_HOME")
            sources_path = (
                _Path(xdg, "r4") if xdg else _Path.home() / ".config" / "r4"
            ) / "sources.json"
            if sources_path.is_file():
                src_data = _json.loads(sources_path.read_text(encoding="utf-8"))
                active = src_data.get("active")
                for src in src_data.get("sources", []):
                    if src.get("name") == active:
                        ak = src.get("api_key")
                        if ak:
                            jwt = ak
                            jwt_source = f"sources.json (source={active})"
                        break
        except Exception:
            pass

    # 3) Profile .token file
    profile_token = None
    profile_token_path = None
    profile_path = getattr(connection, "profile_path", None)
    if profile_path:
        from pathlib import Path as _Path

        token_path = _Path(profile_path).with_suffix(".token")
        if token_path.is_file():
            profile_token = token_path.read_text(encoding="utf-8").strip()
            profile_token_path = str(token_path)
            if not jwt:
                jwt = profile_token
                jwt_source = f"profile .token ({profile_name})"

    # Display token sources
    if jwt_source:
        lines.append(_kv("token", _green("present") + f"  {_dim(jwt_source)}"))
    else:
        lines.append(_kv("token", _red("not found")))

    # Show JWT claims from whichever token we found
    if jwt:
        try:
            import base64

            payload_b64 = jwt.split(".")[1]
            padding = 4 - len(payload_b64) % 4
            payload_b64 += "=" * padding
            claims = _json.loads(base64.urlsafe_b64decode(payload_b64))
            sub = claims.get("sub", "?")
            exp = claims.get("exp")
            if exp:
                import time as _time

                ttl = exp - _time.time()
                exp_str = f"{int(ttl)}s remaining" if ttl > 0 else _red("expired")
            else:
                exp_str = _dim("no expiry")
            lines.append(_kv("subject", sub))
            lines.append(_kv("expires", exp_str))
            project = claims.get("project_id")
            if project:
                lines.append(_kv("project", project))
        except Exception:
            lines.append(_kv("claims", _dim("(decode failed)")))

    # List all token files found
    if profile_token_path:
        lines.append(
            _kv("profile_token", _green("found") + f"  {_dim(profile_token_path)}")
        )
    elif profile_path:
        lines.append(_kv("profile_token", _dim("(not found)")))

    # --- S3 / MinIO ---
    lines.append("")
    lines.append(_section("S3 / MinIO"))
    lines.append(_kv("minio_url", connection.minio_url))

    # Find s3cfg
    config_state = resolve_loaded_config(
        explicit_server_config_path=args.server_conf,
        explicit_user_config_path=args.conf,
        explicit_s3_config_path=args.s3cfg,
    )
    s3cfg_path = None
    user = None
    if config_state.user.valid and config_state.user.path:
        try:
            user = load_client_cfg_file(config_state.user.path)
            if config_state.s3.valid and config_state.s3.path:
                user = apply_s3_config_override(user, config_state.s3.path)
            s3cfg_path = user.get("s3_configuration_file")
        except Exception:
            pass
    if not s3cfg_path and config_state.s3.valid and config_state.s3.path:
        s3cfg_path = config_state.s3.path

    if s3cfg_path:
        expanded = os.path.expanduser(s3cfg_path)
        if os.path.exists(expanded):
            lines.append(_kv("s3cfg", _green("found") + f"  {_dim(expanded)}"))
            # Try to read and test connection
            try:
                cfg = _cp.ConfigParser(inline_comment_prefixes=("#", ";"))
                cfg.read(expanded)
                host_base = cfg.get("default", "host_base", fallback="?")
                access_key = cfg.get("default", "access_key", fallback="?")
                use_https = cfg.getboolean("default", "use_https", fallback=False)
                lines.append(_kv("host_base", host_base))
                lines.append(_kv("access_key", access_key))
                lines.append(_kv("use_https", str(use_https).lower()))

                # Quick S3 connectivity test
                protocol = "https" if use_https else "http"
                secret_key = cfg.get("default", "secret_key", fallback="")
                import boto3
                from botocore.config import Config as BotoConfig

                s3_client = boto3.client(
                    "s3",
                    endpoint_url=f"{protocol}://{host_base}",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    config=BotoConfig(
                        signature_version="s3v4",
                        connect_timeout=5,
                        read_timeout=5,
                        retries={"max_attempts": 1},
                    ),
                    region_name=cfg.get(
                        "default", "bucket_location", fallback="us-east-1"
                    ),
                    verify=cfg.getboolean(
                        "default", "check_ssl_certificate", fallback=True
                    ),
                )
                buckets = s3_client.list_buckets()
                count = len(buckets.get("Buckets", []))
                lines.append(
                    _kv(
                        "connectivity",
                        f"{_green('● ok')}  {_dim(f'{count} bucket(s)')}",
                    )
                )
            except Exception as exc:
                err_msg = str(exc)
                if "SignatureDoesNotMatch" in err_msg:
                    lines.append(
                        _kv(
                            "connectivity",
                            f"{_red('● auth failed')}  {_dim('SignatureDoesNotMatch — secret_key may be stale')}",
                        )
                    )
                elif "ConnectTimeoutError" in err_msg or "timed out" in err_msg.lower():
                    lines.append(
                        _kv(
                            "connectivity",
                            f"{_red('● timeout')}  {_dim(f'{protocol}://{host_base}')}",
                        )
                    )
                else:
                    lines.append(
                        _kv("connectivity", f"{_red('● error')}  {_dim(err_msg[:80])}")
                    )
        else:
            lines.append(_kv("s3cfg", f"{_red('missing')}  {_dim(expanded)}"))
    else:
        lines.append(_kv("s3cfg", _dim("(no s3cfg discovered)")))

    # --- User Config ---
    lines.append("")
    lines.append(_section("User Config"))
    if config_state.user.valid and config_state.user.path:
        lines.append(_kv("user_config", _dim(config_state.user.path)))
        if user:
            lines.append(_kv("username", user.get("username", "?")))
            xps = user.get("experiences", [])
            lines.append(_kv("experiences", str(len(xps))))
    else:
        lines.append(_kv("user_config", _dim("(not found)")))

    if args.output == "json":
        # Structured JSON output
        info = {
            "profile": prof,
            "mode": connection.mode,
            "zmq": {
                "host": zmq_host,
                "control_port": zmq_port,
                "auth_type": zmq_auth,
                "tunnel_mode": tunnel,
                "data_port_range": connection.data_port_range_label,
            },
            "gateway_url": gateway_url,
            "minio_url": connection.minio_url,
            "s3cfg": s3cfg_path,
        }
        print(api_json(**info, output_format="json"))
    else:
        print("\n".join(lines))
    return 0


def _get_api_key_from_profile_static(connection) -> str | None:
    """Read JWT from profile token file or r4 tokens (no ctx needed)."""
    profile_path = getattr(connection, "profile_path", None)
    if profile_path:
        from pathlib import Path as _Path

        token_path = _Path(profile_path).with_suffix(".token")
        if token_path.is_file():
            return token_path.read_text(encoding="utf-8").strip()
    profile_name = getattr(connection, "profile_name", None)
    return _try_read_r4_token(profile_name)


def _resolve_control_client_id(user_cfg=None):
    if isinstance(user_cfg, dict):
        username = user_cfg.get("username")
        if username:
            return username
    # Try connection profile username (set by r4 auth login)
    try:
        from umrs.shared.connection_profiles import (
            get_current_profile_name,
            load_connection_profile,
        )

        profile_name = get_current_profile_name()
        if profile_name:
            profile = load_connection_profile(profile_name)
            if getattr(profile, "username", None):
                return profile.username
    except Exception:
        pass
    return os.environ.get("UMRS_CLIENT_ID", os.environ.get("USER", "_"))


def _build_monitor_capabilities(client_id, admin_token=None, user_cfg=None):
    auth_ctx = create_auth_context_from_request(client_id, token=admin_token)
    target_user = auth_ctx.username or client_id
    return {
        "username": auth_ctx.username,
        "is_admin": auth_ctx.is_admin,
        "has_user_config": isinstance(user_cfg, dict),
        "bucket_aliases": _extract_bucket_aliases(user_cfg),
        "allowed_actions": {
            "recompute": auth_ctx.can_execute_command(
                TaskCmd.RECOMPUTE, target_user=target_user, log_denial=False
            ),
            "update": auth_ctx.can_execute_command(
                TaskCmd.UPDATE, target_user=target_user, log_denial=False
            ),
            "check": auth_ctx.can_execute_command(
                TaskCmd.CHECK, target_user=target_user, log_denial=False
            ),
            "reload": auth_ctx.can_execute_command(
                TaskCmd.RELOAD, target_user=target_user, log_denial=False
            ),
        },
    }


def _select_experience(user_cfg, experience_name=None):
    if not isinstance(user_cfg, dict):
        if experience_name:
            raise ValueError(
                "--experience requires a loaded user config (-c/--conf or selected config)"
            )
        return None
    experiences = user_cfg.get("experiences") or []
    if experience_name:
        for xp in experiences:
            if xp.get("xp_name") == experience_name:
                return xp
        raise ValueError(f"Experience not found in user config: {experience_name}")
    for xp in experiences:
        if xp.get("xp_name") == "default":
            return xp
    return experiences[0] if experiences else None


def _experience_endpoint_value(experience, *keys):
    if not isinstance(experience, dict):
        return None
    endpoint_cfg = experience.get("endpoint")
    if isinstance(endpoint_cfg, dict):
        for key in keys:
            if key in endpoint_cfg:
                return endpoint_cfg[key]
    for key in keys:
        if key in experience:
            return experience[key]
    return None


def _resolve_endpoint_idle_ttl_sec(args, user_cfg) -> int:
    if getattr(args, "idle_ttl", None):
        return parse_duration_seconds(args.idle_ttl, field_name="idle_ttl")

    experience = _select_experience(user_cfg, getattr(args, "experience", None))
    raw = _experience_endpoint_value(
        experience,
        "idle_ttl",
        "idle_ttl_sec",
        "endpoint_idle_ttl",
        "endpoint_idle_ttl_sec",
    )
    if raw is None:
        return parse_duration_seconds("30m", field_name="idle_ttl")
    return parse_duration_seconds(raw, field_name="idle_ttl")


def _latest_attempt(job):
    attempts = job.get("attempts") or []
    return attempts[-1] if attempts else {}


def _endpoint_backend_ref(job) -> str:
    for attempt in reversed(job.get("attempts") or []):
        backend_ref = attempt.get("backend_ref")
        if backend_ref:
            return str(backend_ref)
    return ""


def _format_duration_label(seconds: int | None) -> str:
    if not seconds:
        return ""
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


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
    # Resolve which configs are actually in use (handles auto-discovery)
    resolved_paths: dict[str, str | None] = {}
    try:
        resolved = resolve_loaded_config()
        for kind in ("server", "user", "s3"):
            layer = getattr(resolved, kind, None)
            resolved_paths[kind] = layer.path if layer and layer.valid else None
    except Exception:
        pass

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
            print()
            continue
        active_path = resolved_paths.get(key)
        for candidate in candidates:
            is_active = candidate.current or (
                active_path
                and os.path.realpath(candidate.path) == os.path.realpath(active_path)
            )
            marker = _green("▸") if is_active else " "
            dot = _status_dot(candidate.valid)
            path_str = _bold(candidate.path) if is_active else candidate.path
            print(f"  {marker} {dot} {path_str}")
            if candidate.error:
                print(f"      {_red(candidate.error)}")
        print()


def _build_user_config_template(username: str, fullname: str) -> dict:
    """Return a minimal user config dict that passes client_schema validation."""
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


def _handle_config_init(args):
    username = args.username
    fullname = args.fullname

    # Prompt interactively when required fields are missing.
    if not username:
        try:
            username = input("Username: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
    if not username:
        raise ValueError("Username must not be empty.")

    if not fullname:
        try:
            fullname = input("Full name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
    if not fullname:
        raise ValueError("Full name must not be empty.")

    # Determine output path.
    if args.output_path:
        out_path = Path(args.output_path).expanduser()
    else:
        from umrs.shared.config import (
            get_resolved_umrs_cfg_dir,
            get_users_default_cfg_subdir,
        )

        users_dir = (
            Path(get_resolved_umrs_cfg_dir()).expanduser()
            / get_users_default_cfg_subdir()
        )
        out_path = users_dir / f"{username}.json"

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Config already exists: {out_path}\n"
            f"Use --force to overwrite, or 'umrs config use --user {out_path}' to select it."
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
        cfg["s3_configuration_file"] = default_s3cfg_path(username)
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
                cfg["s3_configuration_file"] = default_s3cfg_path(username)
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

    if args.output == "json":
        print(
            api_json(
                message="User config created and set as current",
                path=str(out_path),
                current=str(saved),
                output_format=args.output,
            )
        )
    else:
        print(f"{_green('✓')} Created {_bold(str(out_path))}")
        print(f"  Set as current user config: {_dim(str(saved))}")
        print(
            "  Open the JSON file in an editor to customise experiences and modalities."
        )
        print(f"  {_dim('Do not run the .json path directly in the shell.')}")
        print("  Then run:")
        print(f"  {_cyan('umrs reload')}")

    # Auto-install shell tab completion
    comp_msg = install_completion("umrs")
    if args.output != "json":
        print(f"  {comp_msg}")

    return 0


def _handle_config_add_experience(ctx, args):
    """Merge experience fragment(s) into the current user config JSON."""
    import json as _json_mod

    fragment_path = Path(args.fragment).expanduser()
    if not fragment_path.is_file():
        raise FileNotFoundError(f"Fragment not found: {fragment_path}")

    target = _resolve_target_user_config_path(args)

    with open(fragment_path, encoding="utf-8") as f:
        fragment = _json_mod.load(f)

    with open(target, encoding="utf-8") as f:
        user_cfg = _json_mod.load(f)

    added, skipped = _merge_experience_fragments(
        user_cfg,
        _coerce_experience_fragments(fragment),
        update=args.update,
    )
    _write_json_file(target, user_cfg)

    if args.output == "json":
        print(
            api_json(
                message="Experience fragment merged",
                added=added,
                skipped=skipped,
                target=str(target),
                output_format=args.output,
            )
        )
    else:
        if added:
            print(f"{_green('✓')} Added experience(s): {', '.join(added)}")
        if skipped:
            print(f"{_dim('Skipped (already exists):')} {', '.join(skipped)}")
            print(f"  {_dim('Use --update to replace existing experiences.')}")
        print(f"  Config: {target}")

    return 0


def _handle_config_remove_experience(ctx, args):
    """Remove experience(s) by xp_name from the current user config JSON."""
    import json as _json_mod

    target = _resolve_target_user_config_path(args)

    with open(target, encoding="utf-8") as f:
        user_cfg = _json_mod.load(f)

    experiences = user_cfg.get("experiences", [])
    names_to_remove = set(args.xp_names)
    original_count = len(experiences)

    removed = [
        xp["xp_name"] for xp in experiences if xp.get("xp_name") in names_to_remove
    ]
    user_cfg["experiences"] = [
        xp for xp in experiences if xp.get("xp_name") not in names_to_remove
    ]

    not_found = names_to_remove - set(removed)

    _write_json_file(target, user_cfg)

    if args.output == "json":
        print(
            api_json(
                message="Experiences removed",
                removed=removed,
                not_found=sorted(not_found),
                remaining=len(user_cfg["experiences"]),
                resources_retained=bool(removed),
                target=str(target),
                output_format=args.output,
            )
        )
    else:
        if removed:
            print(f"{_green('✓')} Removed experience(s): {', '.join(removed)}")
        if not_found:
            print(f"{_dim('Not found (skipped):')} {', '.join(sorted(not_found))}")
        print(f"  Remaining experiences: {len(user_cfg['experiences'])}")
        print(f"  Config: {target}")
        if removed:
            print(
                f"  {_dim('MinIO buckets and Postgres/pgvector data are retained until explicit purge.')}"
            )

    return 0


def _resolve_target_user_config_path(args) -> Path:
    conf_path = getattr(args, "conf", None)
    if conf_path:
        target = Path(conf_path).expanduser()
    else:
        target = _resolve_current_user_config_path()
    if target is None or not target.is_file():
        raise FileNotFoundError(
            "No user config found. Run 'umrs config init' first, or pass -c <path>."
        )
    return target


def _write_json_file(path: Path, payload: dict | list) -> None:
    import json as _json_mod

    with open(path, "w", encoding="utf-8") as f:
        _json_mod.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _coerce_experience_fragments(fragment) -> list[dict]:
    if isinstance(fragment, dict):
        fragment_list = [fragment]
    elif isinstance(fragment, list):
        fragment_list = fragment
    else:
        raise ValueError(
            "Fragment must be a JSON object (single experience) or array of experiences."
        )
    for xp in fragment_list:
        if not isinstance(xp, dict):
            raise ValueError("Each experience fragment must be a JSON object.")
        if not xp.get("xp_name"):
            raise ValueError("Each experience fragment must have an 'xp_name' field.")
    return fragment_list


def _merge_experience_fragments(
    user_cfg: dict, fragment_list: list[dict], *, update: bool
) -> tuple[list[str], list[str]]:
    experiences = user_cfg.setdefault("experiences", [])
    existing_names = {xp["xp_name"] for xp in experiences if "xp_name" in xp}

    added = []
    skipped = []
    for xp in fragment_list:
        name = xp["xp_name"]
        if name in existing_names:
            if update:
                experiences[:] = [x for x in experiences if x.get("xp_name") != name]
                experiences.append(xp)
                added.append(name)
            else:
                skipped.append(name)
        else:
            experiences.append(xp)
            added.append(name)
            existing_names.add(name)
    return added, skipped


@dataclass
class _RecoveredBucket:
    bucket: str
    username: str
    xp_name: str
    bucket_alias: str
    output: bool = False
    object_count: int | None = None
    total_size_bytes: int | None = None
    error: str | None = None


@dataclass
class _RecoveredTable:
    db_name: str
    xp_name: str
    sql_username: str
    table_name: str
    columns: list[dict]
    row_count: int | None = None
    distinct_s3_objects: int | None = None
    vector_size: int | None = None
    task: str | None = None


# Suffix → task mapping used to infer the task from a bucket alias when
# Postgres is unavailable.
_ALIAS_TASK_SUFFIXES: list[tuple[str, str]] = [
    ("seg", "segmentation"),
    ("segment", "segmentation"),
    ("segmentation", "segmentation"),
    ("mae", "mae_reconstruction"),
    ("reconstruct", "reconstruction"),
    ("reconstruction", "reconstruction"),
    ("features", "feature_extraction"),
    ("feature-extraction", "feature_extraction"),
    ("detection", "detection"),
    ("embedding", "embedding"),
    ("text", "embedding"),
]


def _infer_task_from_alias(alias: str) -> str | None:
    """Best-effort task inference from a bucket alias suffix."""
    normalised = _normalise_recovery_part(alias)
    for suffix, task in _ALIAS_TASK_SUFFIXES:
        if normalised.endswith("-" + suffix) or normalised == suffix:
            return task
    return None


def _normalise_recovery_part(value: str) -> str:
    return str(value).strip().lower().replace(" ", "-").replace("_", "-")


def _parse_recoverable_bucket_name(
    bucket_name: str, username: str
) -> _RecoveredBucket | None:
    username_part = _normalise_recovery_part(username)
    name = str(bucket_name).strip().lower()
    prefix = f"{username_part}."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    if "." not in rest:
        return None
    xp_name, bucket_alias = rest.split(".", 1)
    if not xp_name or not bucket_alias:
        return None
    output = bucket_alias.endswith("-output")
    if output:
        bucket_alias = bucket_alias[: -len("-output")]
    if not bucket_alias:
        return None
    return _RecoveredBucket(
        bucket=name,
        username=username_part,
        xp_name=xp_name,
        bucket_alias=bucket_alias,
        output=output,
    )


def _read_s3cfg(path: Path) -> dict[str, str]:
    import configparser

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(path):
        raise FileNotFoundError(f"S3 config not found: {path}")
    if not config.has_section("default"):
        raise ValueError(f"S3 config has no [default] section: {path}")
    return dict(config.items("default"))


def _s3_endpoint_from_cfg(cfg: dict[str, str]) -> tuple[str, bool]:
    host = cfg.get("host_base") or cfg.get("host_bucket")
    if not host:
        raise ValueError("S3 config must define host_base or host_bucket")
    use_https = str(cfg.get("use_https", "false")).lower() in ("1", "true", "yes", "on")
    verify_ssl = str(cfg.get("check_ssl_certificate", "true")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    protocol = "https" if use_https else "http"
    return f"{protocol}://{host}", verify_ssl


def _bucket_object_stats(s3, bucket_name: str) -> tuple[int, int]:
    object_count = 0
    total_size = 0
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket_name}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        contents = response.get("Contents", [])
        object_count += len(contents)
        total_size += sum(int(obj.get("Size", 0)) for obj in contents)
        if not response.get("IsTruncated"):
            return object_count, total_size
        continuation_token = response.get("NextContinuationToken")


def _list_recovery_buckets_from_s3(
    *,
    s3cfg_path: Path,
    username: str,
    xp_names: set[str] | None = None,
    include_object_counts: bool = True,
) -> tuple[list[_RecoveredBucket], list[str]]:
    warnings = []
    try:
        import boto3
    except ImportError as exc:
        return [], [f"boto3 is required for MinIO/S3 recovery: {exc}"]

    cfg = _read_s3cfg(s3cfg_path)
    endpoint_url, verify_ssl = _s3_endpoint_from_cfg(cfg)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=cfg.get("access_key"),
        aws_secret_access_key=cfg.get("secret_key"),
        verify=verify_ssl,
    )
    response = s3.list_buckets()
    recovered = []
    normalised_xp_names = {_normalise_recovery_part(xp) for xp in xp_names or set()}
    for item in response.get("Buckets", []):
        parsed = _parse_recoverable_bucket_name(item.get("Name", ""), username)
        if parsed is None:
            continue
        if (
            normalised_xp_names
            and _normalise_recovery_part(parsed.xp_name) not in normalised_xp_names
        ):
            continue
        if include_object_counts:
            try:
                parsed.object_count, parsed.total_size_bytes = _bucket_object_stats(
                    s3, parsed.bucket
                )
            except Exception as exc:
                parsed.error = str(exc)
                warnings.append(
                    f"could not count objects in bucket {parsed.bucket}: {exc}"
                )
        recovered.append(parsed)
    return recovered, warnings


def _postgres_recovery_params() -> dict[str, object]:
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": int(os.environ.get("DB_PORT", "5432")),
        "user": os.environ.get(
            "DB_SUPERUSER", os.environ.get("POSTGRES_USER", "postgres")
        ),
        "password": os.environ.get(
            "DB_SUPERUSER_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "postgres")
        ),
        "connect_timeout": 3,
    }


def _pg_database_match(db_name: str, xp_name: str) -> tuple[str, str] | None:
    candidates = {xp_name, xp_name.replace("-", "_"), xp_name.replace("_", "-")}
    for candidate in sorted(candidates, key=len, reverse=True):
        prefix = f"{candidate}_"
        if db_name == candidate:
            return candidate, ""
        if db_name.startswith(prefix):
            return candidate, db_name[len(prefix) :]
    return None


def _parse_vector_size(type_name: str | None) -> int | None:
    if not type_name:
        return None
    match = __import__("re").fullmatch(
        r"vector\((\d+)\)", str(type_name).strip().lower()
    )
    return int(match.group(1)) if match else None


def _inspect_recovery_pg_tables(
    db_name: str, xp_name: str, sql_username: str, params: dict[str, object]
) -> list[_RecoveredTable]:
    import psycopg2
    from psycopg2 import sql as psql

    conn = psycopg2.connect(dbname=db_name, **params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE' "
                "ORDER BY table_name"
            )
            table_names = [row[0] for row in cur.fetchall()]
            tables = []
            for table_name in table_names:
                cur.execute(
                    "SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod) "
                    "FROM pg_catalog.pg_attribute a "
                    "JOIN pg_catalog.pg_class c ON a.attrelid = c.oid "
                    "JOIN pg_catalog.pg_namespace n ON c.relnamespace = n.oid "
                    "WHERE n.nspname = 'public' AND c.relname = %s "
                    "AND a.attnum > 0 AND NOT a.attisdropped "
                    "ORDER BY a.attnum",
                    (table_name,),
                )
                columns = [{"name": row[0], "type": row[1]} for row in cur.fetchall()]
                column_names = {column["name"] for column in columns}
                row_count = None
                distinct_s3_objects = None
                with contextlib.suppress(Exception):
                    cur.execute(
                        psql.SQL("SELECT COUNT(*) FROM {}").format(
                            psql.Identifier(table_name)
                        )
                    )
                    row_count = int(cur.fetchone()[0])
                if "s3_object" in column_names:
                    with contextlib.suppress(Exception):
                        cur.execute(
                            psql.SQL(
                                "SELECT COUNT(DISTINCT s3_object) FROM {} WHERE s3_object IS NOT NULL"
                            ).format(psql.Identifier(table_name))
                        )
                        distinct_s3_objects = int(cur.fetchone()[0])
                vector_size = None
                for column in columns:
                    if column["name"] == "embedding":
                        vector_size = _parse_vector_size(column.get("type"))
                        break
                tables.append(
                    _RecoveredTable(
                        db_name=db_name,
                        xp_name=xp_name,
                        sql_username=sql_username,
                        table_name=table_name,
                        columns=columns,
                        row_count=row_count,
                        distinct_s3_objects=distinct_s3_objects,
                        vector_size=vector_size,
                    )
                )
            return tables
    finally:
        conn.close()


def _list_recovery_postgres_status(
    xp_names: set[str],
) -> tuple[list[_RecoveredTable], list[str]]:
    warnings = []
    if not xp_names:
        return [], warnings
    try:
        import psycopg2
    except ImportError as exc:
        return [], [f"psycopg2 is required for Postgres recovery: {exc}"]

    params = _postgres_recovery_params()
    try:
        conn = psycopg2.connect(dbname="postgres", **params)
    except Exception as exc:
        return [], [f"could not inspect Postgres status: {exc}"]

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT datname FROM pg_database "
                "WHERE datistemplate = false AND datallowconn = true "
                "ORDER BY datname"
            )
            db_names = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    tables = []
    for xp_name in xp_names:
        for db_name in db_names:
            match = _pg_database_match(db_name, xp_name)
            if match is None:
                continue
            recovered_xp_name, sql_username = match
            try:
                tables.extend(
                    _inspect_recovery_pg_tables(
                        db_name, recovered_xp_name, sql_username, params
                    )
                )
            except Exception as exc:
                warnings.append(f"could not inspect Postgres database {db_name}: {exc}")
    return tables, warnings


def _metadata_columns_from_recovered_table(table: _RecoveredTable | None) -> list[dict]:
    if table is None:
        return []
    system_columns = {
        "id",
        "s3_object",
        "checksum",
        "date",
        "time",
        "embedding",
        "model",
        "saved_s3_object",
        "_content",
        "chunk_start",
        "chunk_end",
        "chunk_md5",
    }
    return [
        {"name": column["name"], "type": column["type"].upper()}
        for column in table.columns
        if column.get("name") not in system_columns
    ]


def _match_recovered_table(
    tables: list[_RecoveredTable],
    bucket_alias: str,
    used_tables: set[tuple[str, str]],
) -> _RecoveredTable | None:
    alias_key = _normalise_recovery_part(bucket_alias)
    for table in tables:
        key = (table.db_name, table.table_name)
        if key in used_tables:
            continue
        if _normalise_recovery_part(table.table_name) == alias_key:
            used_tables.add(key)
            return table
    for table in tables:
        key = (table.db_name, table.table_name)
        if key not in used_tables:
            used_tables.add(key)
            return table
    return None


def _build_recovered_experience_fragments(
    *,
    username: str,
    buckets: list[_RecoveredBucket],
    tables: list[_RecoveredTable],
    model_name: str,
    model_type: str,
    fallback_vector_size: str,
    return_type: str,
) -> list[dict]:
    groups: dict[str, dict[str, object]] = {}
    for bucket in buckets:
        group = groups.setdefault(bucket.xp_name, {"buckets": {}, "tables": []})
        bucket_map = group["buckets"]
        item = bucket_map.setdefault(
            bucket.bucket_alias, {"input": None, "output": None}
        )
        item["output" if bucket.output else "input"] = bucket

    for table in tables:
        xp_key = _normalise_recovery_part(table.xp_name)
        group = groups.setdefault(xp_key, {"buckets": {}, "tables": []})
        group.setdefault("xp_name", table.xp_name)
        group.setdefault("sql_username", table.sql_username)
        group["tables"].append(table)

    experiences = []
    for xp_key in sorted(groups):
        group = groups[xp_key]
        xp_name = str(group.get("xp_name") or xp_key)
        sql_username = str(group.get("sql_username") or username)
        group_tables = list(group.get("tables") or [])
        used_tables: set[tuple[str, str]] = set()
        modalities = []
        bucket_map = group.get("buckets") or {}
        aliases = sorted(bucket_map.keys()) or [
            table.table_name for table in group_tables
        ]
        for alias in aliases:
            pair = bucket_map.get(alias, {}) if isinstance(bucket_map, dict) else {}
            output_bucket = pair.get("output") if isinstance(pair, dict) else None
            table = _match_recovered_table(group_tables, alias, used_tables)
            vector_size = (
                str(table.vector_size)
                if table and table.vector_size
                else str(fallback_vector_size)
            )
            recovered_return_type = None
            if return_type == "auto":
                recovered_return_type = "file" if output_bucket else None
            elif return_type != "none":
                recovered_return_type = return_type
            modality_name = table.table_name if table else alias
            # Infer task from the modality_name / bucket alias when possible.
            recovered_task = None
            if table and table.task:
                recovered_task = table.task
            else:
                recovered_task = _infer_task_from_alias(alias)
            # store_embeddings: default to "no" when Postgres is unavailable —
            # safer for vision workflows; user can edit to "yes" before reload.
            store_embeddings = "no"
            if table is not None:
                store_embeddings = "yes" if table.vector_size else "no"

            modality_entry: dict = {
                "modality_name": modality_name,
                "s3": {
                    "bucket_alias": alias,
                    "data": {
                        "type": "file",
                        "mime_type": "application/octet-stream",
                    },
                    **({"need_output_bucket": True} if output_bucket else {}),
                },
                "model": {
                    "model_name": model_name,
                    "vector_size": vector_size,
                    "model_type": model_type,
                    "return_type": recovered_return_type,
                },
                "sql": {
                    "store_embeddings": store_embeddings,
                    "table_name": table.table_name if table else alias,
                    "meta_data": _metadata_columns_from_recovered_table(table),
                },
            }
            if recovered_task:
                modality_entry["task"] = recovered_task
            modalities.append(modality_entry)
        experiences.append(
            {
                "xp_name": xp_name,
                "sql_username": sql_username,
                "modalities": modalities,
            }
        )
    return experiences


def _recovered_bucket_to_dict(bucket: _RecoveredBucket) -> dict:
    return {
        "bucket": bucket.bucket,
        "username": bucket.username,
        "xp_name": bucket.xp_name,
        "bucket_alias": bucket.bucket_alias,
        "output": bucket.output,
        "object_count": bucket.object_count,
        "total_size_bytes": bucket.total_size_bytes,
        "error": bucket.error,
    }


def _recovered_table_to_dict(table: _RecoveredTable) -> dict:
    return {
        "db_name": table.db_name,
        "xp_name": table.xp_name,
        "sql_username": table.sql_username,
        "table_name": table.table_name,
        "columns": table.columns,
        "row_count": table.row_count,
        "distinct_s3_objects": table.distinct_s3_objects,
        "vector_size": table.vector_size,
    }


def _resolve_recovery_username(args, user_cfg: dict | None) -> str:
    username = getattr(args, "username", None) or (user_cfg or {}).get("username")
    if not username:
        raise ValueError(
            "Recovering experiences requires --username or a loaded user config."
        )
    return username


def _handle_experience_recover(ctx, args):
    import json as _json_mod

    target = None
    user_cfg = None
    with contextlib.suppress(Exception):
        target = _resolve_target_user_config_path(args)
        with open(target, encoding="utf-8") as f:
            user_cfg = _json_mod.load(f)

    username = _resolve_recovery_username(args, user_cfg)
    xp_names = set(getattr(args, "xp_names", None) or [])
    s3cfg_path = Path(
        getattr(args, "s3cfg", None)
        or (user_cfg or {}).get("s3_configuration_file")
        or default_s3cfg_path(username)
    ).expanduser()
    buckets, warnings = _list_recovery_buckets_from_s3(
        s3cfg_path=s3cfg_path,
        username=username,
        xp_names=xp_names or None,
        include_object_counts=not getattr(args, "no_object_count", False),
    )
    inferred_xp_names = {_normalise_recovery_part(bucket.xp_name) for bucket in buckets}
    inferred_xp_names.update(_normalise_recovery_part(xp) for xp in xp_names)
    tables = []
    if not getattr(args, "no_postgres", False):
        tables, pg_warnings = _list_recovery_postgres_status(inferred_xp_names)
        warnings.extend(pg_warnings)
    else:
        warnings.append(
            "Postgres inspection skipped; recovered SQL/model fields may use defaults."
        )
    if not tables:
        warnings.append(
            "No Postgres/pgvector table status recovered; review model, vector_size, and table_name before reload."
        )

    experiences = _build_recovered_experience_fragments(
        username=username,
        buckets=buckets,
        tables=tables,
        model_name=args.model_name,
        model_type=args.model_type,
        fallback_vector_size=str(args.vector_size),
        return_type=args.return_type,
    )
    output_path = (
        Path(args.output_path).expanduser()
        if getattr(args, "output_path", None)
        else None
    )
    if output_path:
        output_payload = experiences[0] if len(experiences) == 1 else experiences
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_file(output_path, output_payload)

    merge_result = None
    if getattr(args, "merge", False):
        if target is None:
            target = _resolve_target_user_config_path(args)
        with open(target, encoding="utf-8") as f:
            target_cfg = _json_mod.load(f)
        added, skipped = _merge_experience_fragments(
            target_cfg,
            experiences,
            update=getattr(args, "update", False),
        )
        _write_json_file(target, target_cfg)
        merge_result = {"target": str(target), "added": added, "skipped": skipped}

    payload = {
        "username": username,
        "s3cfg": str(s3cfg_path),
        "experiences": experiences,
        "s3_buckets": [_recovered_bucket_to_dict(bucket) for bucket in buckets],
        "postgres_tables": [_recovered_table_to_dict(table) for table in tables],
        "warnings": warnings,
    }
    if output_path:
        payload["output_path"] = str(output_path)
    if merge_result:
        payload["merge"] = merge_result

    if args.output == "json":
        print(api_json(**payload, output_format=args.output))
    else:
        if output_path:
            print(
                f"{_green('✓')} Wrote recovered experience JSON: {_bold(str(output_path))}"
            )
        if merge_result:
            print(
                f"{_green('✓')} Merged recovered experience(s) into: {_bold(merge_result['target'])}"
            )
        print(_section("Recovered Experiences"))
        print(_json_mod.dumps(experiences, indent=2, ensure_ascii=False))
        if warnings:
            print(_section("Warnings"))
            for warning in warnings:
                print(f"  {_yellow(warning)}")
        print(
            f"  {_dim('Review model fields before reload if recovery used S3/Postgres status only.')}"
        )
    return 0


def _handle_experience_command(ctx):
    args = getattr(ctx, "args", ctx)
    if args.experience_action == "add":
        return _handle_config_add_experience(ctx, args)
    if args.experience_action == "remove":
        return _handle_config_remove_experience(ctx, args)
    if args.experience_action == "recover":
        return _handle_experience_recover(ctx, args)
    if args.experience_action == "list":
        return _handle_experience_list(ctx, args)
    if args.experience_action == "validate":
        return _handle_experience_validate(ctx, args)
    raise ValueError(f"Unknown experience action: {args.experience_action}")


def _configure_experience_list_parser(parser):
    parser.add_argument(
        "-c", "--conf", default=None, help="User config JSON path to read"
    )


def _handle_experience_list(ctx, args):
    """List experiences in the current user config JSON."""
    import json as _json_mod

    target = _resolve_target_user_config_path(args)

    with open(target, encoding="utf-8") as f:
        user_cfg = _json_mod.load(f)

    experiences = user_cfg.get("experiences", [])

    if args.output == "json":
        print(
            api_json(
                message="Experiences listed",
                experiences=experiences,
                count=len(experiences),
                target=str(target),
                output_format=args.output,
            )
        )
        return 0

    if not experiences:
        print(f"  {_dim('No experiences found in user config.')}")
        print(f"  Config: {target}")
        return 0

    print(f"  {_green('✓')} {len(experiences)} experience(s) in {target}")
    for xp in experiences:
        name = xp.get("xp_name", "<unnamed>")
        sql_user = xp.get("sql_username", "-")
        modalities = xp.get("modalities", []) or []
        mod_names = []
        for m in modalities:
            mname = m.get("modality_name") or m.get("modality") or "<unnamed>"
            mod_names.append(mname)
        mods_str = ", ".join(mod_names) if mod_names else _dim("(none)")
        print(f"    {_bold(name)}")
        print(f"      sql_username: {sql_user}")
        print(f"      modalities ({len(modalities)}): {mods_str}")
    return 0


def _handle_experience_validate(ctx, args):
    """Validate experience fragment(s) against the server model catalog.

    When no fragment is provided, validates the currently loaded user config
    (i.e. all experiences the user has configured).
    """
    import json as _json_mod
    import sys as _sys_mod

    # Use user from context if available
    user_cfg = ctx.user

    # Resolve user config and experiences
    if args.fragment is not None:
        fragment_path = Path(args.fragment).expanduser()
        if not fragment_path.is_file():
            print(
                f"  {_red('✗')} Fragment not found: {fragment_path}",
                file=_sys_mod.stderr,
            )
            return 1

        with open(fragment_path, encoding="utf-8") as f:
            fragment = _json_mod.load(f)

        fragments = _coerce_experience_fragments(fragment)
    else:
        # Validate the full loaded user config
        if user_cfg is None:
            # Fallback: load from file
            target = _resolve_target_user_config_path(args)
            if not target.is_file():
                print(
                    f"  {_red('✗')} No user config found at: {target}",
                    file=_sys_mod.stderr,
                )
                return 1

            with open(target, encoding="utf-8") as f:
                user_cfg = _json_mod.load(f)
        else:
            # user_cfg already loaded in context
            pass

        # Extract experiences from user config
        experiences = user_cfg.get("experiences", [])
        if not experiences:
            print(f"  {_dim('ℹ')} No experiences configured in user config")
            if user_cfg and isinstance(user_cfg, dict):
                pass  # ok
            return 0

        # Wrap in the same structure expected by validation logic
        fragments = [
            {"xp_name": f"user_experience_{i}", **xp}
            for i, xp in enumerate(experiences)
        ]

    errors = []
    warnings = []

    for idx, xp in enumerate(fragments):
        xp_name = xp.get("xp_name", f"<unnamed at index {idx}>")
        print(f"\n  Checking experience: {_bold(xp_name)}")

        # Validate required fields
        if "modalities" not in xp:
            errors.append(f"  {xp_name}: missing 'modalities' key")
            continue

        for mod in xp.get("modalities", []):
            mod_name = mod.get("modality_name", mod.get("modality", "?"))
            model = mod.get("model", {})
            model_name = model.get("model_name", "?")

            print(f"    modality: {mod_name}")
            print(f"    model:    {model_name}")

            # Check if model exists in server catalog (non-destructive)
            if ctx.control_mq is None:
                warnings.append("    ⚠  ZMQ not available — skipping model check")
                print(f"    {warnings[-1]}")
                continue

            try:
                from umrs.client.control_api import UMRSControlAPI

                # Get client_id from user config or args
                client_id = (
                    ctx.args._client_id if hasattr(ctx.args, "_client_id") else None
                )
                if not client_id and user_cfg and isinstance(user_cfg, dict):
                    client_id = user_cfg.get("username")
                if not client_id:
                    client_id = (
                        ctx.connection.server_host.split(".")[-1]
                        if hasattr(ctx, "connection") and ctx.connection
                        else "_"
                    )

                api = UMRSControlAPI(
                    control_mq=ctx.control_mq,
                    client_id=client_id,
                    admin_token=ctx.admin_token,
                )
                # Check both system and user scopes
                catalog_names = set()
                for scope in ("system", "user"):
                    models_resp = api.list_models(scope=scope)
                    for m in models_resp.get("models", []):
                        catalog_names.add(m.get("name") or m.get("model_name"))

                # Normalize names for matching: experience fragment may use
                # HuggingFace-style name (intfloat/e5-base-v2) while the
                # catalog may store a canonical short name (e5-base-v2).
                # Match by checking if one normalizes to the other.
                def _norm(name: str) -> str:
                    """Normalize model name for comparison."""
                    n = name.lower()
                    # Strip HuggingFace org prefix: intfloat/e5-base-v2 → e5-base-v2
                    if "/" in n:
                        n = n.split("/", 1)[1]
                    return n

                matched = False
                for catalog_name in catalog_names:
                    if model_name == catalog_name:
                        matched = True
                        break
                    if _norm(model_name) == _norm(catalog_name):
                        matched = True
                        break

                if not matched:
                    warnings.append(
                        f"    ⚠  Model '{model_name}' not found in model catalog (catalog names: {', '.join(sorted(catalog_names))})"
                    )
                    print(f"    {warnings[-1]}")
                else:
                    print(f"    ✓  Model '{model_name}' found in model catalog")
            except Exception:
                warnings.append(
                    "    ⚠  Could not verify model (server unreachable or auth issue)"
                )
                print(f"    {warnings[-1]}")

    if errors:
        print(f"\n  {_red('✗')} {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return 1

    if warnings:
        print(f"\n  {_dim('ℹ')} {len(warnings)} warning(s):")
        for w in warnings:
            print(f"    {w}")

    print(f"\n  {_green('✓')} Experience fragment is valid")
    return 0


def _configure_experience_validate_parser(parser):
    parser.add_argument(
        "fragment",
        nargs="?",
        default=None,
        help="Path to a JSON file containing one experience object or an array of experiences. "
        "Omit to validate the currently loaded user config.",
    )


def _configure_experience_add_parser(parser):
    parser.add_argument(
        "fragment",
        help="Path to a JSON file containing one experience object or an array of experiences",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Replace existing experience(s) with matching xp_name instead of skipping",
    )
    parser.add_argument(
        "-c", "--conf", default=None, help="User config JSON path to edit"
    )


def _configure_experience_recover_parser(parser):
    parser.add_argument(
        "xp_names",
        nargs="*",
        help="Optional xp_name values to recover; defaults to all accessible user buckets",
    )
    parser.add_argument(
        "--username",
        default=None,
        help="UMRS username whose resources should be recovered",
    )
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        help="User config JSON path used for username/s3cfg defaults",
    )
    parser.add_argument(
        "--s3cfg", default=None, help="S3 config used to inspect MinIO buckets"
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Write recovered experience JSON to this path",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Merge recovered experiences into the user config",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Replace existing xp_name entries when merging",
    )
    parser.add_argument(
        "--no-postgres",
        action="store_true",
        default=False,
        help="Skip optional Postgres/pgvector inspection",
    )
    parser.add_argument(
        "--no-object-count",
        action="store_true",
        default=False,
        help="Skip MinIO object counts for faster recovery",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/e5-base-v2",
        help="Model name to place in recovered JSON",
    )
    parser.add_argument(
        "--model-type",
        default="TransformerEncoder",
        help="Model type to place in recovered JSON",
    )
    parser.add_argument(
        "--vector-size",
        default="768",
        help="Fallback vector size when Postgres cannot infer it",
    )
    parser.add_argument(
        "--return-type",
        default="auto",
        choices=("auto", "none", "file", "embedding", "raw"),
        help="Recovered model return_type; auto uses file when an output bucket exists",
    )


def _configure_experience_remove_parser(parser):
    parser.add_argument(
        "xp_names",
        nargs="+",
        help="One or more xp_name values to remove from the user JSON; MinIO/Postgres data is retained",
    )
    parser.add_argument(
        "-c", "--conf", default=None, help="User config JSON path to edit"
    )


def _resolve_current_user_config_path():
    """Return the Path to the currently selected user config, or None."""
    from umrs.shared.loaded_config_profiles import resolve_loaded_config

    resolved = resolve_loaded_config()
    if resolved.user and resolved.user.valid and resolved.user.path:
        return Path(resolved.user.path)
    return None


def _handle_config_command(args):
    if args.config_action == "init":
        return _handle_config_init(args)

    if args.config_action == "add-experience":
        return _handle_config_add_experience(args)

    if args.config_action == "import":
        return _handle_config_add_experience(args)

    if args.config_action == "remove-experience":
        return _handle_config_remove_experience(args)

    if args.config_action == "recover-experience":
        return _handle_experience_recover(args)

    if args.config_action == "list":
        discovered = discover_loaded_config_profiles()
        show_srv = _is_dev_mode() or bool(args.server_conf)
        payload = {
            key: [candidate.to_dict() for candidate in value]
            for key, value in discovered.items()
            if show_srv or key != "server"
        }
        if args.output == "json":
            print(api_json(configs=payload, output_format=args.output))
        else:
            _print_config_candidates_text(discovered, show_server=show_srv)
        return 0

    if args.config_action == "use":
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
            updates["server"] = str(set_current_server_config(args.server))
        if args.user:
            updates["user"] = str(set_current_user_config(args.user))
        if args.s3:
            updates["s3"] = str(set_current_s3_config(args.s3))

        if args.output == "json":
            print(
                api_json(
                    message="Current config selections updated",
                    updates=updates,
                    output_format=args.output,
                )
            )
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
            w for w in resolved.warnings if "server config" not in w.lower()
        ]
    if args.output == "json":
        payload = resolved.to_dict()
        if not show_srv:
            payload.pop("server", None)
        print(api_json(config=payload, output_format=args.output))
    else:
        print(_format_config_text(resolved, show_server=show_srv))
    return 0


def _handle_admin_user_command(args):
    """Handle ``admin user {add,list,remove}``."""
    import json as _json_mod

    if args.user_action == "add":
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

        admin_token = getattr(args, "admin_token", None) or os.environ.get(
            "UMRS_ADMIN_TOKEN"
        )
        if not admin_token:
            print(
                f"{_red('✗')} Admin token required. Set UMRS_ADMIN_TOKEN or pass --admin-token."
            )
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
            if args.output == "json":
                print(
                    api_json(
                        action="user_add",
                        status="error",
                        message=str(e),
                        output_format=args.output,
                    )
                )
            else:
                print(f"{_red('✗')} Failed to enroll user: {e}")
            return 1
        finally:
            with contextlib.suppress(Exception):
                control_mq.close()

        if args.output == "json":
            print(api_json(action="user_add", **result, output_format=args.output))
        else:
            if result.get("status") == "ok":
                print(f"{_green('✓')} User {_bold(username)} enrolled on the server.")
                print(
                    f"  {_dim('The server has auto-reloaded to provision MinIO/Postgres resources.')}"
                )
            else:
                print(
                    f"{_red('✗')} Enrollment failed: {result.get('message', 'unknown error')}"
                )
        return 0 if result.get("status") == "ok" else 1

    if args.user_action == "list":
        admin_token = getattr(args, "admin_token", None) or os.environ.get(
            "UMRS_ADMIN_TOKEN"
        )
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
            if args.output == "json":
                print(
                    api_json(
                        action="user_list",
                        status="error",
                        message=str(e),
                        output_format=args.output,
                    )
                )
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

        if args.output == "json":
            print(
                api_json(
                    action="user_list", users=sorted(users), output_format=args.output
                )
            )
        else:
            print(_section("Enrolled Users"))
            if users:
                for u in sorted(users):
                    print(f"  {_status_dot('green')} {u}")
            else:
                print(f"  {_dim('No users found in server status.')}")
                print(
                    f"  {_dim('(Users appear here after at least one processing job has run.)')}"
                )
        return 0

    if args.user_action == "remove":
        if args.output == "json":
            print(
                api_json(
                    action="user_remove",
                    status="error",
                    message="Not yet implemented",
                    output_format=args.output,
                )
            )
        else:
            print(f"{_yellow('⚠')} User removal is not yet implemented.")
            print(
                f"  {_dim('De-provisioning (MinIO IAM cleanup, Postgres role removal) will be added in a future release.')}"
            )
        return 1

    raise ValueError(f"Unknown user action: {args.user_action}")


def _handle_admin_command(args):
    """Handle ``admin token {generate,list,revoke}`` and ``admin user {add,list,remove}``."""
    from umrs.shared.auth import AdminTokenManager

    if args.admin_action == "user":
        return _handle_admin_user_command(args)

    if args.admin_action != "token":
        raise ValueError(f"Unknown admin action: {args.admin_action}")

    mgr = AdminTokenManager()

    if args.token_action == "generate":
        existing = {t["name"] for t in mgr.list_tokens()}
        overwrite = args.name in existing
        token = mgr.generate_token(args.name, description=args.description)
        if args.output == "json":
            print(
                api_json(
                    action="token_generate",
                    name=args.name,
                    token=token,
                    overwritten=overwrite,
                    output_format=args.output,
                )
            )
        else:
            action = "Regenerated" if overwrite else "Generated"
            print(
                f"{_green('✓')} {action} admin token for {_bold(args.name)} (for umrs --admin-token / $UMRS_ADMIN_TOKEN):"
            )
            print(f"  {_yellow(token)}")
            print()
            print(f"  {_dim('Save this token — it cannot be retrieved later.')}")
            print(f"  {_dim('Use with: umrs --admin-token <token> <command>')}")
            print(
                f"  {_dim('NOTE: this is NOT a Gateway/r4 JWT — use umrs-server admin token generate-jwt for that.')}"
            )
        return 0

    if args.token_action == "list":
        tokens = mgr.list_tokens()
        if args.output == "json":
            print(
                api_json(action="token_list", tokens=tokens, output_format=args.output)
            )
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
                                _fmt_cell(t["name"], 20),
                                _fmt_cell(t.get("description", ""), 30),
                                _fmt_cell(t.get("created_at", ""), 25, style=_dim),
                            ]
                        )
                    )
            else:
                print(f"  {_dim('No admin tokens configured.')}")
        return 0

    if args.token_action == "revoke":
        ok = mgr.revoke_token(args.name)
        if args.output == "json":
            print(
                api_json(
                    action="token_revoke",
                    name=args.name,
                    revoked=ok,
                    output_format=args.output,
                )
            )
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
            print()
            return 1
    if not name:
        raise ValueError("Profile name must not be empty.")

    # Prompt for host and mode when not provided via CLI flags.
    host = args.host
    if host == "127.0.0.1" and "--host" not in sys.argv:
        try:
            entered = input(f"Server host [{host}]: ").strip()
            if entered:
                host = entered
        except (EOFError, KeyboardInterrupt):
            print()
            return 1

    mode = args.mode
    if mode == "direct" and "--mode" not in sys.argv:
        try:
            entered = input(f"Connection mode (direct/ssh_tunnel) [{mode}]: ").strip()
            if entered in ("direct", "ssh_tunnel"):
                mode = entered
            elif entered:
                print(f"Invalid mode '{entered}', using '{mode}'")
        except (EOFError, KeyboardInterrupt):
            print()
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
                {
                    "local_port": args.control_port,
                    "remote_host": "localhost",
                    "remote_port": args.control_port,
                },
                {
                    "local_port": args.data_port_start,
                    "remote_host": "localhost",
                    "remote_port": args.data_port_start,
                },
            ],
        }

    out_path = get_profile_path(name)

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Profile already exists: {out_path}\n"
            f"Use --force to overwrite, or 'umrs profile use {name}' to select it."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        _json_mod.dump(profile_data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    set_current_profile_name(name)

    if args.output == "json":
        print(
            api_json(
                message="Connection profile created and set as current",
                name=name,
                path=str(out_path),
                output_format=args.output,
            )
        )
    else:
        print(f"{_green('✓')} Created {_bold(str(out_path))}")
        print(f"  Set as current profile: {_bold(name)}")
        if args.mode == "ssh_tunnel":
            print("  Open the JSON file in an editor to customise SSH tunnels.")
            print(f"  {_dim('Do not run the .json path directly in the shell.')}")
            print("  Then run:")
            print(f"  {_cyan(profile_data['ssh']['command_hint'])}")
    return 0


def _ensure_user_config_after_login(username: str, args) -> int:
    """Create a default user config after login if none exists, or ask to overwrite."""
    from umrs.shared.config import (
        get_resolved_umrs_cfg_dir,
        get_users_default_cfg_subdir,
    )

    users_dir = (
        Path(get_resolved_umrs_cfg_dir()).expanduser() / get_users_default_cfg_subdir()
    )
    cfg_path = users_dir / f"{username}.json"

    if cfg_path.exists():
        # Ask whether to overwrite.
        print(f"\n  User config already exists: {_dim(str(cfg_path))}")
        try:
            answer = (
                input("  Overwrite with a fresh default config? [y/N] ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if answer not in ("y", "yes"):
            # Keep existing, just make sure it's the current config.
            set_current_user_config(str(cfg_path))
            print(f"  {_green('✓')} Kept existing config, set as current.")
            return 0

    cfg = _build_user_config_template(username, fullname="")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json_mod

    with open(cfg_path, "w", encoding="utf-8") as f:
        _json_mod.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")

    set_current_user_config(str(cfg_path))
    print(f"\n  {_green('✓')} User config created → {_bold(str(cfg_path))}")
    print(f"  Add experiences via: {_cyan('umrs experience add <experience.json>')}")
    print(f"  Then reload:        {_cyan('umrs reload')}")
    return 0


def _handle_login(args):
    """Unified login: save JWT, create profile + user config, verify connection."""
    import getpass

    host = args.host
    port = args.port
    gateway_url = getattr(args, "gateway_url", None)

    # If stdin is a TTY and no gateway URL was given, prompt for it so a bare
    # ``umrs login`` is sufficient for the shared umrs-front deployment.
    if not gateway_url and sys.stdin.isatty():
        entered = input(
            f"Gateway URL (leave blank for direct ZMQ at {host}:{port}): "
        ).strip()
        if entered:
            gateway_url = entered

    # --gateway-url implies --mode gateway
    if gateway_url and args.mode == "direct":
        args.mode = "gateway"
    mode = args.mode
    api_key = args.api_key
    username = args.username

    # --- 1. Get the token (positional JWT or --api-key) ---
    jwt_arg = getattr(args, "jwt", None)
    api_key = jwt_arg or api_key
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
        print("  JWT identity (display only, not authoritative):")
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
                print("    Token EXPIRED")

    if not username:
        username = input("UMRS username: ").strip()
    if not username:
        print(_red("✗ Username is required."), file=sys.stderr)
        return 1

    # --- 3. Create connection profile ---
    profile_name = args.profile_name or f"{host}-{mode}"
    profile_path = get_profile_path(profile_name)
    profile_path.parent.mkdir(parents=True, exist_ok=True)

    minio_url = f"http://{host}:9000"
    rag_url = f"http://{host}:8081"
    profile_data = {
        "name": profile_name,
        "mode": mode,
    }
    if gateway_url:
        profile_data["gateway_url"] = gateway_url.rstrip("/")
    if mode != "gateway":
        # ZMQ config only for direct/ssh_tunnel modes
        profile_data["zmq"] = {
            "host": host,
            "control_port": port,
            "data_port_range": {"start": port + 1, "end": port + 8},
            "tunnel_mode": mode == "ssh_tunnel",
            "health_check_timeout_ms": 5000,
        }
    profile_data["services"] = {
        "minio_url": minio_url,
        "rag_url": rag_url,
    }
    if mode == "ssh_tunnel":
        profile_data["ssh"] = {
            "host": host,
            "command_hint": f"ssh -L {port}:localhost:{port} {host}",
            "local_forwards": [
                {"local_port": port, "remote_host": "localhost", "remote_port": port},
            ],
        }

    import json as _json_mod

    with open(profile_path, "w", encoding="utf-8") as f:
        _json_mod.dump(profile_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    profile_path.chmod(0o600)
    set_current_profile_name(profile_name)

    # Store the JWT in a separate token file (profile schema does not allow extra fields).
    token_path = profile_path.with_suffix(".token")
    token_path.write_text(api_key + "\n", encoding="utf-8")
    token_path.chmod(0o600)

    print(f"✓ Connection profile '{profile_name}' saved → {profile_path}")
    print(f"  JWT stored → {token_path}")

    # --- 4. Verify connection ---
    if mode == "gateway" and gateway_url:
        print(f"  Verifying connection to {gateway_url} ...")
        try:
            result = _http_ping(gateway_url, api_key)
            if result.get("server_ready"):
                print("  ✓ Server is reachable via gateway")
            else:
                print("  ✗ Server not ready via gateway (may still be starting)")
        except Exception as e:
            print(f"  ✗ Connection check failed: {e}")
            print("  Profile saved — you can retry with `umrs ping`")
        print(f"\n✓ Login complete for user '{username}' via {gateway_url}")
    else:
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
            if mq.is_server_ready(timeout_ms=5000):
                print("  ✓ Server is reachable")
            else:
                print("  ✗ Server did not respond to ping (may still be starting)")
        except Exception as e:
            print(f"  ✗ Connection check failed: {e}")
            print("  Profile saved — you can retry with `umrs ping`")
        print(f"\n✓ Login complete for user '{username}' on {host}:{port}")
    print(f"  Profile: {profile_name}")

    # --- 5. Auto-create user config if missing ---
    rc = _ensure_user_config_after_login(username, args)
    if rc:
        return rc

    print("  Run `umrs ping` to verify full connectivity.")
    return 0


def _handle_profile_command(args):
    if args.profile_action == "init":
        return _handle_profile_init(args)

    if args.profile_action == "list":
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
        if args.output == "json":
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
                    print(
                        f"  {marker} {name_str} {_fmt_cell(profile.mode, 10, style=_dim)} {_dim(profile.path)}"
                    )
            else:
                print(f"  {_dim('No connection profiles found.')}")
            for entry in errors:
                print(f"  {_red('!')} {entry['name']}: {entry['error']}")
        return 0

    if args.profile_action == "use":
        path = set_current_profile_name(args.name)
        if args.output == "json":
            print(
                api_json(
                    message=f"Current connection profile set to '{args.name}'",
                    path=str(path),
                    output_format=args.output,
                )
            )
        else:
            print(f"{_green('✓')} Current connection profile set to {_bold(args.name)}")
            print(_kv("path", _dim(str(path))))
        return 0

    if args.profile_action == "import":
        profile = import_connection_profile(args.file, overwrite=args.force)
        payload = {
            "message": f"Imported connection profile '{profile.name}'",
            "name": profile.name,
            "path": profile.path,
        }
        if args.output == "json":
            print(api_json(**payload, output_format=args.output))
        else:
            print(f"{_green('✓')} Imported connection profile {_bold(profile.name)}")
            print(_kv("path", _dim(payload["path"])))
        return 0

    connection = resolve_connection(
        explicit_profile_name=args.profile,
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )
    conn_dict = connection.to_dict()
    # Hide internal sources dict from JSON output (available in text via [sources] section)
    conn_dict.pop("sources", None)
    payload = {
        "connection": conn_dict,
        "selected_profile": connection.profile_name,
    }
    if args.output == "json":
        print(api_json(**payload, output_format=args.output))
    else:
        print(_format_profile_text(connection, connection.profile_name, []))
    return 0


# ======================= Helper / API abstractions (new) =======================
def configure_buckets(user_cfg, buckets_cfg):
    username = user_cfg["username"]
    s3_configuration_file = user_cfg["s3_configuration_file"]
    for xp in user_cfg["experiences"]:
        sql_username = xp["sql_username"]
        xp_name = xp["xp_name"]
        db_name = create_db_name(sql_username, xp_name, username)
        for modality in xp["modalities"]:
            # Support both old (modality_name) and new (modality + task) config
            raw_modality_name = modality.get("modality_name") or modality.get(
                "modality"
            )
            raw_task = modality.get("task")
            modality_name, task = resolve_modality_task(raw_modality_name, raw_task)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            if modality["s3"].get("need_output_bucket", False):
                output_path_to_bucket = make_path_to_bucket_string(
                    xp_name, bucket_alias, username, output=True
                )
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
                output_path_to_bucket,
            )
            buckets_cfg.store(bucket_alias, entry)


def build_buckets_cfg(user_cfg):
    buckets_cfg = BucketsCfg()
    configure_buckets(user_cfg, buckets_cfg)
    return buckets_cfg


def api_json(status="ok", output_format="json", **payload):
    out = {"status": status}
    out.update(payload)
    if output_format == "json":
        return _json.dumps(out, ensure_ascii=False)
    lines = []
    for k, v in out.items():
        if isinstance(v, (list, tuple)):
            lines.append(f"{k}=[{', '.join(map(str, v))}]")
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
        if ":" not in p:
            raise ValueError(f"Metadata must be key:value got '{p}'")
        k, v = p.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta


def merge_metadata(base_metadata, override_meta_dict):
    # base_metadata is the structure produced by metadata_dict(sql_conf).
    # For now we only inject overrides into a flat list under Metadata (matching existing code paths)
    # Existing code path expects metadata_args = { 'Metadata': {...}}
    merged = dict(base_metadata.items()) if base_metadata else {}
    md = merged.get("Metadata", {})
    md.update(override_meta_dict)
    merged["Metadata"] = md
    return merged


def format_status_text(response_payload, default_key):
    if response_payload.get("status") != "ok":
        message = response_payload.get("message", "status unavailable")
        if message == "No status for 'reload'":
            message = (
                "No status for 'reload'. Reload is synchronous; use "
                "'umrs status all' for a summary or "
                "'umrs bucket list' to verify onboarding."
            )
        return f"{_red('✗')} {message}"

    client_label = response_payload.get("client")
    lines = []

    def append_entry(name, entry):
        header_parts = []
        if client_label:
            header_parts.append(client_label)
        if name:
            header_parts.append(name)
        generated_at = entry.get("generated_at")
        last_updated_at = entry.get("last_updated_at")
        if header_parts or generated_at or last_updated_at:
            header = " - ".join(header_parts)
            time_parts = []
            if generated_at:
                time_parts.append(f"generated {utc_to_local(generated_at)}")
            if last_updated_at:
                time_parts.append(f"last update {utc_to_local(last_updated_at)}")
            if time_parts:
                time_text = "; ".join(time_parts)
                header = (
                    f"{header} {_dim('(' + time_text + ')')}"
                    if header
                    else _dim(time_text)
                )
            if header:
                lines.append(_section(header))
        summary_body = entry.get("summary")
        if summary_body:
            lines.append(f"  {summary_body}")
        else:
            lines.append(f"  {_dim('(no summary)')}")

    if "summary" in response_payload:
        append_entry(default_key, response_payload["summary"])
    elif "summaries" in response_payload:
        summaries = response_payload["summaries"] or {}
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


def _probe_remote_buckets(username: str, user_cfg: dict | None) -> dict:
    """Connect to MinIO/S3 and return remote bucket names with object counts.

    Returns dict with:
      - remote_names: {bucket_name: object_count}
      - warning: optional error string if S3 probe failed
    """
    result: dict = {"remote_names": {}}
    s3cfg_path_str = (user_cfg or {}).get(
        "s3_configuration_file"
    ) or default_s3cfg_path(username)
    s3cfg_path = Path(s3cfg_path_str).expanduser()
    if not s3cfg_path.is_file():
        result["warning"] = f"S3 config not found: {s3cfg_path} — cannot check remote"
        return result
    try:
        import boto3
    except ImportError:
        result["warning"] = "boto3 not installed — cannot check remote"
        return result
    try:
        cfg = _read_s3cfg(s3cfg_path)
        endpoint_url, verify_ssl = _s3_endpoint_from_cfg(cfg)
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=cfg.get("access_key"),
            aws_secret_access_key=cfg.get("secret_key"),
            verify=verify_ssl,
        )
        response = s3.list_buckets()
    except Exception as exc:
        result["warning"] = f"S3 connection failed: {exc}"
        return result

    remote_names: dict[str, int] = {}
    for item in response.get("Buckets", []):
        bname = item.get("Name", "")
        try:
            count, _ = _bucket_object_stats(s3, bname)
        except Exception:
            count = 0
        remote_names[bname] = count
    result["remote_names"] = remote_names
    return result


def _build_remote_info(
    remote_names: dict[str, int], configured_paths: set[str], username: str
) -> dict:
    """Build the remote_info dict from raw remote bucket names and configured paths."""
    result: dict = {"buckets": {}, "extra_buckets": []}
    username_prefix = _normalise_recovery_part(username) + "."
    for path in configured_paths:
        if path in remote_names:
            result["buckets"][path] = {
                "status": "synced",
                "object_count": remote_names[path],
            }
        else:
            result["buckets"][path] = {"status": "missing"}
    for bname, count in sorted(remote_names.items()):
        if bname in configured_paths:
            continue
        # Only show extra buckets belonging to this user
        if bname.startswith(username_prefix):
            result["extra_buckets"].append({"name": bname, "object_count": count})
    return result


def _xp_name_from_bucket_path(bucket_path: str, username: str) -> str:
    """Extract the experience name from a bucket path.

    Bucket paths follow ``<user>.<xp_name>.<alias>[-output]``.
    Returns the ``xp_name`` segment or ``"?"`` if the path cannot be parsed.
    """
    prefix = _normalise_recovery_part(username) + "."
    if not bucket_path.startswith(prefix):
        return "?"
    rest = bucket_path[len(prefix) :]
    parts = rest.split(".", 1)
    return parts[0] if len(parts) >= 2 else "?"


def action_list_buckets(bucket_cfg, username, output_format, *, remote_info=None):
    buckets = []
    for alias, entry in bucket_cfg.items():
        if entry.username == username:
            buckets.append(
                {
                    "bucket_alias": alias,
                    "experience": entry.xp_name,
                    "modality": entry.modality_name,
                    "task": entry.task,
                    "path_to_bucket": entry.path_to_bucket,
                }
            )
    if output_format == "json":
        if remote_info is not None:
            for b in buckets:
                path = b["path_to_bucket"]
                ri = remote_info.get("buckets", {}).get(path)
                b["remote_status"] = ri["status"] if ri else "missing"
                b["object_count"] = ri.get("object_count") if ri else None
            extra = remote_info.get("extra_buckets", [])
            for eb in extra:
                eb["experience"] = _xp_name_from_bucket_path(eb["name"], username)
            return api_json(
                buckets=buckets, extra_remote_buckets=extra, output_format=output_format
            )
        return api_json(buckets=buckets, output_format=output_format)

    # Text mode: formatted table with dynamic column widths
    lines = [_section(f"Buckets for {_bold(username)}")]
    if not buckets:
        lines.append(f"  {_dim('No buckets found.')}")
    else:
        show_remote = remote_info is not None
        w_xp = max(len("Experience"), *(len(b["experience"]) for b in buckets))
        w_alias = max(len("Alias"), *(len(b["bucket_alias"]) for b in buckets))
        w_mod = max(len("Modality"), *(len(b["modality"]) for b in buckets))
        w_task = max(len("Task"), *(len(b.get("task", "")) for b in buckets))
        header_cells = [
            _fmt_cell("Experience", w_xp, style=_bold),
            _fmt_cell("Alias", w_alias, style=_bold),
            _fmt_cell("Modality", w_mod, style=_bold),
            _fmt_cell("Task", w_task, style=_bold),
        ]
        if show_remote:
            w_status = max(len("Remote"), 7)  # "synced" is 6, "missing" is 7
            w_objects = len("Objects")
            header_cells.append(_fmt_cell("Remote", w_status, style=_bold))
            header_cells.append(_fmt_cell("Objects", w_objects, style=_bold))
        header_cells.append(_bold("Path"))
        lines.append("  " + "  ".join(header_cells))
        sep_width = w_xp + w_alias + w_mod + w_task + 8 + 6
        if show_remote:
            sep_width += w_status + w_objects + 4
        lines.append(f"  {_dim('-' * sep_width)}")
        for b in buckets:
            row_cells = [
                _fmt_cell(b["experience"], w_xp),
                _fmt_cell(b["bucket_alias"], w_alias),
                _fmt_cell(b["modality"], w_mod),
                _fmt_cell(b.get("task", ""), w_task),
            ]
            if show_remote:
                path = b["path_to_bucket"]
                ri = (remote_info or {}).get("buckets", {}).get(path)
                if ri and ri["status"] == "synced":
                    status_str = _green("synced")
                    obj_str = str(ri.get("object_count", ""))
                else:
                    status_str = _red("missing")
                    obj_str = _dim("-")
                row_cells.append(_fmt_cell_raw(status_str, w_status))
                row_cells.append(_fmt_cell_raw(obj_str, w_objects))
            row_cells.append(_dim(b["path_to_bucket"]))
            lines.append("  " + "  ".join(row_cells))

        if show_remote:
            extra = (remote_info or {}).get("extra_buckets", [])
            if extra:
                lines.append("")
                lines.append(
                    _section(f"Extra remote buckets {_dim('(not in config)')}")
                )
                for eb in extra:
                    xp = _xp_name_from_bucket_path(eb["name"], username)
                    xp_label = f"  {_dim('experience=')}{xp}"
                    lines.append(
                        f"  {eb['name']}  {_dim(str(eb.get('object_count', '')) + ' objects')}{xp_label}"
                    )
                lines.append("")
                lines.append(f"  {_dim('Hints:')}")
                lines.append(f"  {_dim('  Remove extra buckets:')}  umrs bucket purge")
                lines.append(
                    f"  {_dim('  Add experience back:')}   umrs experience add <experience.json> --update"
                )
                lines.append(
                    f"  {_dim('  Edit user config:')}      umrs config edit --user"
                )
            warn = (remote_info or {}).get("warning")
            if warn:
                lines.append(f"  {_yellow(warn)}")

    return "\n".join(lines)


def perform_upload_text(s3, full_bucket_name, text, data_mq, metadata_args):
    s3_key = s3_store_txt(s3, full_bucket_name, "", text, data_mq, metadata_args)
    return s3_key


def perform_upload_image(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_image(s3, full_bucket_name, "", file_path, data_mq, metadata_args)


def perform_upload_textfile(s3, full_bucket_name, file_path, data_mq, metadata_args):
    return s3_store_text_file(
        s3, full_bucket_name, "", file_path, data_mq, metadata_args
    )


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
    '       umrs-server admin user add <username> --fullname "Your Name" --role user\n'
    "  2. Create or select your local client config:\n"
    "       umrs config init --username <username>\n"
    "  3. Check the resolved config and server connectivity:\n"
    "       umrs config show\n"
    "       umrs ping\n\n"
    "Shell completion:\n"
    '  eval "$(umrs completion bash)"\n\n'
    "Related CLIs:\n"
    "  r4            Gateway-mediated research workflows (chat, RAG, endpoints).\n"
    "  umrs   UMRS diagnostics, endpoint control, experience recovery, migration.\n"
    "  umrs-server   Offline operator administration (user management, server runtime).\n\n"
    "Install variants:\n"
    "  pip install 'umrs[client]'        Lightweight gateway client (no pyzmq/boto3).\n"
    "  pip install 'umrs[client-full]'   Full client with direct ZMQ + S3 transport.\n"
    "  pip install 'umrs[server]'        Server runtime and ML stack.\n"
)


def _disabled_admin_message():
    return (
        "umrs admin is disabled; admin operations are server-side only. "
        "Use 'umrs-server admin --help' on the server. Examples: "
        "'umrs-server admin user add <username> --fullname \"Your Name\" --role user' "
        "and 'umrs-server admin user list'."
    )


# ---------------------------------------------------------------------------
# HTTP transport helpers — used when gateway_url is configured
# ---------------------------------------------------------------------------


def _http_ping(gateway_url: str, api_key: str | None = None) -> dict:
    """Ping the UMRS server via the HTTP gateway."""
    import urllib.error
    import urllib.request

    url = f"{gateway_url.rstrip('/')}/gw/ops/ping"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
            return body
    except urllib.error.HTTPError as e:
        return {
            "status": "error",
            "server_ready": False,
            "message": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {"status": "error", "server_ready": False, "message": str(e)}


def _http_reload(
    gateway_url: str,
    api_key: str | None = None,
    *,
    user_config: dict | None = None,
    user_id: str | None = None,
) -> dict:
    """Trigger config reload via the HTTP gateway."""
    import urllib.error
    import urllib.request

    url = f"{gateway_url.rstrip('/')}/gw/ops/reload"
    payload: dict = {}
    if user_config is not None:
        payload["user_config"] = user_config
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    if user_id:
        req.add_header("X-User-ID", user_id)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
            return body
    except urllib.error.HTTPError as e:
        return {"status": "error", "message": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _http_model_list(
    gateway_url: str, api_key: str | None = None, scope: str | None = None
) -> dict:
    """List models via the HTTP gateway endpoint."""
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"{gateway_url.rstrip('/')}/gw/models"
    if scope:
        url = f"{url}?{urllib.parse.urlencode({'scope': scope})}"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = _json.loads(resp.read().decode("utf-8"))
            # Normalize HTTP response ({models:[...], total:N}) to ZMQ-compatible format
            if isinstance(body, dict) and "models" in body and "status" not in body:
                return {"status": "ok", "models": body["models"]}
            return body
    except urllib.error.HTTPError as e:
        return {"status": "error", "message": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _http_model_delete(
    gateway_url: str,
    *,
    name: str,
    scope: str = "user",
    delete_files: bool = False,
    api_key: str | None = None,
) -> dict:
    """Delete a model via the HTTP gateway endpoint.

    First lists models to resolve name+scope to model_id, then sends
    DELETE /gw/models/{model_id}.
    """
    import urllib.error
    import urllib.parse
    import urllib.request

    # Resolve name+scope to model_id via list
    list_result = _http_model_list(gateway_url, api_key, scope=scope)
    if list_result.get("status") != "ok":
        return list_result
    matches = [
        m
        for m in list_result.get("models", [])
        if m.get("name") == name and m.get("scope") == scope
    ]
    if not matches:
        return {
            "status": "error",
            "message": f"Model '{name}' not found in {scope} scope",
        }
    model_id = matches[0].get("model_id") or matches[0].get("id")
    if not model_id:
        return {
            "status": "error",
            "message": f"Model '{name}' has no model_id — cannot delete via HTTP",
        }

    url = f"{gateway_url.rstrip('/')}/gw/models/{model_id}"
    if delete_files:
        url = f"{url}?{urllib.parse.urlencode({'delete_files': 'true'})}"
    req = urllib.request.Request(url, method="DELETE")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 204:
                return {"status": "ok", "message": f"Model '{name}' deleted"}
            body = resp.read().decode("utf-8")
            try:
                return _json.loads(body)
            except Exception:
                return {"status": "ok", "message": f"Model '{name}' deleted"}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            detail = _json.loads(err_body).get("detail", err_body)
            if isinstance(detail, dict) and "message" in detail:
                detail = detail["message"]
        except Exception:
            detail = e.reason
        return {"status": "error", "message": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _http_model_register(
    gateway_url: str,
    *,
    name: str,
    path: str,
    model_type: str = "llm",
    scope: str = "user",
    project_id: str | None = None,
    description: str | None = None,
    huggingface_id: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Register a model via the HTTP gateway endpoint (POST /gw/models)."""
    import urllib.error
    import urllib.request

    url = f"{gateway_url.rstrip('/')}/gw/models"
    body: dict = {
        "name": name,
        "storage_ref": os.path.abspath(path),
        "model_type": model_type,
        "scope": scope,
    }
    if project_id:
        body["project_id"] = project_id
    if description:
        body["description"] = description
    if huggingface_id:
        body["huggingface_id"] = huggingface_id
    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read().decode("utf-8"))
            if isinstance(result, dict):
                # Preserve the model's own 'status' field (e.g. 'available')
                # under a different key, and set the envelope status to 'ok'.
                model_status = result.pop("status", None)
                envelope = {"status": "ok", **result}
                if model_status is not None:
                    envelope["model_status"] = model_status
                model_name = result.get("name", name)
                envelope["message"] = f"Model '{model_name}' registered ({scope} scope)"
                return envelope
            return result
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            detail = _json.loads(err_body).get("detail", err_body)
            # UMRS structured errors nest message inside detail dict
            if isinstance(detail, dict) and "message" in detail:
                detail = detail["message"]
        except Exception:
            detail = e.reason
        return {"status": "error", "message": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _jwt_user_id(api_key: str | None) -> str | None:
    """Best-effort extract the user identity from a JWT (no signature check)."""
    if not api_key or api_key.count(".") < 2:
        return None
    import base64

    try:
        payload_b64 = api_key.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = _json.loads(base64.urlsafe_b64decode(payload_b64))
    except Exception:
        return None
    return (
        claims.get("user_id") or claims.get("sub") or claims.get("preferred_username")
    )


def _http_upload_workflow(
    gateway_url: str,
    *,
    zip_path: str,
    workflow_name: str,
    user_id: str,
    scope: str = "user",
    api_key: str | None = None,
) -> dict:
    """Upload a workflow zip via the HTTP gateway (POST /gw/workflows/upload).

    Used for external clients where the ZMQ data-plane ports are not
    reachable. Mirrors the multipart contract of the UMRS workflows upload
    route, which requires a ``user_id`` form field (the Gateway-verified
    ``X-User-ID`` header takes precedence server-side).
    """
    import urllib.error
    import urllib.request
    import uuid

    with open(zip_path, "rb") as fh:
        file_bytes = fh.read()

    boundary = f"----umrs{uuid.uuid4().hex}"
    boundary_bytes = boundary.encode()
    parts: list[bytes] = []
    for name, value in (
        ("workflow_name", workflow_name),
        ("user_id", user_id),
        ("scope", scope),
    ):
        parts.append(b"--" + boundary_bytes)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode())
    parts.append(b"--" + boundary_bytes)
    parts.append(
        f'Content-Disposition: form-data; name="file"; '
        f'filename="{workflow_name}.zip"'.encode()
    )
    parts.append(b"Content-Type: application/zip")
    parts.append(b"")
    parts.append(file_bytes)
    parts.append(b"--" + boundary_bytes + b"--")
    parts.append(b"")
    body = b"\r\n".join(parts)

    url = f"{gateway_url.rstrip('/')}/gw/workflows/upload"
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
        try:
            return _json.loads(raw)
        except Exception:
            return {"status": "ok", "message": raw}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
            detail = _json.loads(err_body).get("detail", err_body)
            if isinstance(detail, dict) and "message" in detail:
                detail = detail["message"]
        except Exception:
            detail = e.reason
        return {"status": "error", "message": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _try_read_r4_token(profile_name: str | None) -> str | None:
    """Read a JWT from r4's token store (stdlib only, no r4 import).

    Falls back to r4's ``~/.config/r4/tokens.json`` when the umrs
    ``.token`` file doesn't exist — sharing credentials across tools.
    """
    if not profile_name:
        profile_name = "default"
    try:
        from pathlib import Path as _Path

        xdg = os.environ.get("XDG_CONFIG_HOME")
        r4_dir = _Path(xdg, "r4") if xdg else _Path.home() / ".config" / "r4"
        tokens_path = r4_dir / "tokens.json"
        if not tokens_path.is_file():
            return None
        import json as _json

        data = _json.loads(tokens_path.read_text(encoding="utf-8"))
        entry = data.get(profile_name, {})
        token = entry.get("access_token")
        if not token:
            return None
        # Check expiry if present (with 60s buffer)
        import time as _time

        expires_at = entry.get("expires_at")
        if expires_at and _time.time() > (expires_at - 60):
            return None
        return token
    except Exception:
        return None


def _get_gateway_url_from_ctx(ctx) -> str | None:
    """Extract gateway_url from the connection profile if available."""
    conn = ctx.connection
    return getattr(conn, "gateway_url", None) if conn else None


def _get_api_key_from_profile(ctx) -> str | None:
    """Read the JWT from the profile's token file, falling back to r4 tokens."""
    conn = ctx.connection
    profile_path = getattr(conn, "profile_path", None)
    if profile_path:
        from pathlib import Path as _Path

        token_path = _Path(profile_path).with_suffix(".token")
        if token_path.is_file():
            return token_path.read_text(encoding="utf-8").strip()
    # Fall back to r4 token store
    profile_name = getattr(conn, "profile_name", None)
    return _try_read_r4_token(profile_name)


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

    # --- ZMQ path: preferred when a live MessageQueue is available ---
    # Falls through to HTTP if ZMQ reports not-ready (e.g. port unreachable,
    # CURVE bootstrap failed, or server genuinely not ready).
    if ctx.control_mq is not None:
        api = UMRSControlAPI(
            ctx.control_mq,
            _resolve_control_client_id(ctx.user),
            admin_token=ctx.admin_token,
        )
        zmq_ready = False
        ping_result = {}
        try:
            timeout = (
                getattr(args, "timeout_ms", None)
                or ctx.connection.health_check_timeout_ms
            )
            ping_result = api.ping(timeout_ms=timeout)
            zmq_ready = ping_result.get("server_ready", False)
        except Exception:
            pass
        finally:
            ctx.control_mq.close()

        # Session-mode CURVE failures are often a stale session (server restarted
        # and dropped its in-memory registry). Force a fresh registration and
        # retry once over a rebuilt ZMQ control socket before HTTP fallback.
        if not zmq_ready and _force_refresh_curve_session(ctx.connection):
            try:
                refreshed_mq = MessageQueue(
                    role="client",
                    queue_type="control",
                    server_ip=ctx.connection.server_host,
                    port=ctx.connection.control_port,
                    tunnel_mode=ctx.connection.tunnel_mode,
                    prefer_env_server_host=False,
                )
                try:
                    timeout = (
                        getattr(args, "timeout_ms", None)
                        or ctx.connection.health_check_timeout_ms
                    )
                    ping_result = UMRSControlAPI(
                        refreshed_mq,
                        _resolve_control_client_id(ctx.user),
                        admin_token=ctx.admin_token,
                    ).ping(timeout_ms=timeout)
                    zmq_ready = ping_result.get("server_ready", False)
                finally:
                    refreshed_mq.close()
            except Exception:
                pass

        if zmq_ready:
            endpoint_info = {
                "profile": ctx.connection.profile_name or "(none)",
                "server_host": ctx.connection.server_host,
                "control_port": ctx.connection.control_port,
            }
            if args.output == "json":
                print(api_json(output_format="json", **ping_result, **endpoint_info))
            else:
                dot = _status_dot(True)
                profile_label = (
                    _bold(ctx.connection.profile_name)
                    if ctx.connection.profile_name
                    else _dim("(none)")
                )
                print(
                    f"  {dot} Server {_green('ready')}  {_dim('|')}  profile={profile_label}  {_dim('|')}  {ctx.connection.server_host}:{ctx.connection.control_port} (zmq)"
                )
            return 0
        # ZMQ not ready — fall through to HTTP gateway fallback

    # --- HTTP gateway fallback ---
    gateway_url = _get_gateway_url_from_ctx(ctx)
    if gateway_url:
        api_key = _get_api_key_from_profile(ctx)
        ping_result = _http_ping(gateway_url, api_key)
        endpoint_info = {
            "profile": ctx.connection.profile_name or "(none)",
            "gateway_url": gateway_url,
        }
        if args.output == "json":
            print(api_json(output_format="json", **ping_result, **endpoint_info))
        else:
            ready = ping_result.get("server_ready", False)
            dot = _status_dot(ready)
            status_word = _green("ready") if ready else _red("not ready")
            profile_label = (
                _bold(ctx.connection.profile_name)
                if ctx.connection.profile_name
                else _dim("(none)")
            )
            print(
                f"  {dot} Server {status_word}  {_dim('|')}  profile={profile_label}  {_dim('|')}  {gateway_url}"
            )
            if not ready and ping_result.get("message"):
                print(f"    {_dim(ping_result['message'])}")
        return 0

    # Both ZMQ and HTTP failed — print diagnostic
    profile_label = (
        _bold(ctx.connection.profile_name)
        if ctx.connection and ctx.connection.profile_name
        else _dim("(none)")
    )
    host = getattr(ctx.connection, "server_host", "?") if ctx.connection else "?"
    port = getattr(ctx.connection, "control_port", "?") if ctx.connection else "?"
    print(
        f"  {_status_dot(False)} Server {_red('unreachable')}  {_dim('|')}  profile={profile_label}"
    )
    print(f"    ZMQ: {host}:{port} — not responding")
    print("    HTTP: no gateway_url in connection profile")
    print("    hint: switch profile with `umrs config use --profile <name>`")
    print(
        f"          or add gateway_url to ~/.config/umrs/connections/{getattr(ctx.connection, 'profile_name', '?')}.json"
    )
    return 1


def _action_model(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id, admin_token=ctx.admin_token)
    model_action = args.model_action
    exit_code = 0
    try:
        if model_action == "register":
            if ctx.control_mq is None:
                # ZMQ not available (gateway-only mode) — use HTTP fallback
                gateway_url = _get_gateway_url_from_ctx(ctx)
                if not gateway_url:
                    print(
                        api_json(
                            status="error",
                            action="model",
                            message="model register requires either a ZMQ connection or a gateway_url in the connection profile",
                            output_format=args.output,
                        )
                    )
                    return 1
                api_key = _get_api_key_from_profile(ctx)
                parsed = _http_model_register(
                    gateway_url,
                    name=args.name,
                    path=args.path,
                    model_type=args.model_type,
                    scope=args.scope,
                    project_id=args.project_id,
                    description=args.description,
                    huggingface_id=args.huggingface_id,
                    api_key=api_key,
                )
            else:
                parsed = api.register_model(
                    name=args.name,
                    path=args.path,
                    model_type=args.model_type,
                    scope=args.scope,
                    project_id=args.project_id,
                    description=args.description,
                    huggingface_id=args.huggingface_id,
                )
        elif model_action == "list":
            if ctx.control_mq is None:
                # ZMQ not available (gateway-only mode) — use HTTP fallback
                gateway_url = _get_gateway_url_from_ctx(ctx)
                if not gateway_url:
                    print(
                        api_json(
                            status="error",
                            action="model",
                            message="model list requires either a ZMQ connection or a gateway_url in the connection profile",
                            output_format=args.output,
                        )
                    )
                    return 1
                api_key = _get_api_key_from_profile(ctx)
                parsed = _http_model_list(gateway_url, api_key, scope=args.scope)
            else:
                parsed = api.list_models(scope=args.scope)
        elif model_action == "download":
            target_name = args.name or args.repo_id.split("/")[-1]
            if args.server_side:
                parsed = api.download_model_server_side(
                    repo_id=args.repo_id,
                    name=target_name,
                    scope=args.scope,
                    project_id=args.project_id,
                    staging_dir=args.staging_dir,
                )
            else:
                staging_root = args.staging_dir or os.environ.get(
                    "UMRS_MODEL_STAGING_ROOT"
                )
                if not staging_root:
                    from umrs.shared.config import get_default_umrs_cfg_dir

                    staging_root = os.path.join(get_default_umrs_cfg_dir(), "models")
                target_path = os.path.join(staging_root, target_name)

                if os.path.exists(target_path) and os.listdir(target_path):
                    print(f"Model already exists at {target_path}, skipping download.")
                else:
                    try:
                        from huggingface_hub import snapshot_download
                    except ImportError:
                        print(
                            api_json(
                                status="error",
                                message="huggingface_hub not installed. Run: pip install huggingface_hub",
                                output_format=args.output,
                            )
                        )
                        return 1
                    print(f"Downloading {args.repo_id} to {target_path} ...")
                    os.makedirs(target_path, exist_ok=True)
                    snapshot_download(
                        repo_id=args.repo_id,
                        local_dir=target_path,
                        token=os.environ.get("HF_TOKEN"),
                    )
                    print(f"Download complete: {target_path}")

                parsed = api.register_model(
                    name=target_name,
                    path=target_path,
                    model_type="embedding",
                    scope=args.scope,
                    project_id=args.project_id,
                    huggingface_id=args.repo_id,
                    description=f"Downloaded from {args.repo_id}",
                )
        elif model_action == "delete":
            gateway_url = _get_gateway_url_from_ctx(ctx)
            if not gateway_url:
                print(
                    api_json(
                        status="error",
                        action="model",
                        message="model delete requires a gateway_url in the connection profile",
                        output_format=args.output,
                    )
                )
                return 1
            api_key = _get_api_key_from_profile(ctx)
            is_model_id = args.name.startswith("mdl-")
            if not args.force:
                if is_model_id:
                    prompt_msg = f"Delete model '{args.name}'? (y/N): "
                else:
                    prompt_msg = (
                        f"Delete model '{args.name}' from {args.scope} scope? (y/N): "
                    )
                confirm = input(prompt_msg).strip().lower()
                if confirm != "y":
                    print("Cancelled.")
                    return 0
            parsed = _http_model_delete(
                gateway_url,
                name=args.name,
                scope=args.scope,
                delete_files=args.delete_files,
                api_key=api_key,
            )
        elif model_action == "upload":
            import shutil

            src_path = os.path.abspath(args.path)
            if not os.path.isdir(src_path):
                print(
                    api_json(
                        status="error",
                        message=f"Source path is not a directory: {src_path}",
                        output_format=args.output,
                    )
                )
                return 1
            staging_root = args.staging_dir or os.environ.get("UMRS_MODEL_STAGING_ROOT")
            if not staging_root:
                from umrs.shared.config import get_default_umrs_cfg_dir

                staging_root = os.path.join(get_default_umrs_cfg_dir(), "models")
            target_path = os.path.join(staging_root, args.name)
            if os.path.exists(target_path):
                if args.force:
                    shutil.rmtree(target_path)
                else:
                    print(
                        f"Model already exists at {target_path}. Use --force to overwrite."
                    )
                    return 1
            os.makedirs(staging_root, exist_ok=True)
            print(f"Copying {src_path} → {target_path} ...")
            shutil.copytree(src_path, target_path)
            print("Copy complete. Registering model ...")
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
            print(
                api_json(
                    status="error",
                    message=f"Unknown model action: {model_action}",
                    output_format=args.output,
                )
            )
            return 1

        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get("status") != "ok":
                exit_code = 1
        else:
            if parsed.get("status") == "ok":
                if model_action == "list":
                    models = parsed.get("models", [])
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
                            scope_str = m.get("scope", "")
                            print(
                                "  "
                                + "  ".join(
                                    [
                                        _fmt_cell(m["name"], 30),
                                        _fmt_cell(m.get("model_type", ""), 10),
                                        _fmt_cell(scope_str, 8),
                                        _dim(m.get("storage_ref", "")),
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
        print(
            api_json(
                status="error",
                action="model",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        if ctx.control_mq is not None:
            ctx.control_mq.close()
    return exit_code


def _action_job(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    job_action = args.job_action
    exit_code = 0
    try:
        if job_action == "submit":
            spec = (
                parse_json_object_arg(args.spec, field_name="spec") if args.spec else {}
            )
            parsed = api.submit_job(
                job_type=args.job_type,
                spec=spec,
                project=args.project,
                max_attempts=args.max_attempts,
                keep_alive=args.keep_alive,
                partition=getattr(args, "partition", None),
                node=getattr(args, "node", None),
            )
        elif job_action == "cancel":
            parsed = api.cancel_job(args.job_id)
        elif job_action == "status":
            parsed = api.get_job(args.job_id)
        elif job_action == "list":
            parsed = api.list_jobs(
                project=getattr(args, "project", None),
                status=getattr(args, "status", None),
                job_type=getattr(args, "job_type", None),
                limit=getattr(args, "limit", 200),
                offset=getattr(args, "offset", 0),
            )
        else:
            print(
                api_json(
                    status="error",
                    message=f"Unknown job action: {job_action}",
                    output_format=args.output,
                )
            )
            return 1

        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get("status") != "ok":
                exit_code = 1
        else:
            if parsed.get("status") == "ok":
                if job_action == "list":
                    jobs = parsed.get("jobs", [])
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
                            st = job.get("status", "")
                            if st in ("completed", "done"):
                                st_str = _green(st)
                            elif st in ("failed", "cancelled"):
                                st_str = _red(st)
                            elif st in ("running",):
                                st_str = _cyan(st)
                            else:
                                st_str = _yellow(st)
                            print(
                                f"  {job.get('job_id', ''):18s}  "
                                f"{st_str}  "
                                f"{job.get('job_type', ''):18s}  "
                                f"{_dim(job.get('project_id', ''))}"
                            )
                elif job_action == "status":
                    job = parsed.get("job", {})
                    st = job.get("status", "")
                    if st in ("completed", "done"):
                        st_str = _green(st)
                    elif st in ("failed", "cancelled"):
                        st_str = _red(st)
                    elif st in ("running",):
                        st_str = _cyan(st)
                    else:
                        st_str = _yellow(st)
                    print(_section("Job Details"))
                    print(_kv("job_id", _bold(job.get("job_id", ""))))
                    print(_kv("status", st_str))
                    print(_kv("job_type", job.get("job_type", "")))
                    print(_kv("project_id", job.get("project_id", "")))
                else:
                    print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error", action="job", message=str(e), output_format=args.output
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_endpoint(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    endpoint_action = args.endpoint_action
    exit_code = 0
    try:
        if endpoint_action == "up":
            spec = (
                parse_json_object_arg(args.spec, field_name="spec") if args.spec else {}
            )
            idle_ttl_sec = _resolve_endpoint_idle_ttl_sec(args, ctx.user)
            parsed = api.endpoint_up(
                model=args.model,
                model_path=args.model_path,
                project=args.project,
                idle_ttl_sec=idle_ttl_sec,
                tier=args.tier,
                gpu_count=args.gpu_count,
                max_model_len=args.max_model_len,
                max_attempts=args.max_attempts,
                spec=spec,
                partition=getattr(args, "partition", None),
                node=getattr(args, "node", None),
            )
        elif endpoint_action == "list":
            parsed = api.list_endpoints(
                project=getattr(args, "project", None),
                status=getattr(args, "status", None),
                limit=getattr(args, "limit", 200),
                offset=getattr(args, "offset", 0),
            )
        elif endpoint_action == "status":
            parsed = api.get_endpoint(args.job_id)
        elif endpoint_action == "stop":
            parsed = api.stop_endpoint(args.job_id)
        else:
            print(
                api_json(
                    status="error",
                    message=f"Unknown endpoint action: {endpoint_action}",
                    output_format=args.output,
                )
            )
            return 1

        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get("status") != "ok":
                exit_code = 1
        else:
            if parsed.get("status") == "ok":
                if endpoint_action == "list":
                    endpoints = parsed.get("endpoints") or parsed.get("jobs", [])
                    if not endpoints:
                        print(f"  {_dim('No endpoint jobs found.')}")
                    else:
                        print(_section("Endpoints"))
                        print(
                            "  "
                            + "  ".join(
                                [
                                    _fmt_cell("Job ID", 18, style=_bold),
                                    _fmt_cell("Status", 10, style=_bold),
                                    _fmt_cell("Model", 24, style=_bold),
                                    _fmt_cell("Idle TTL", 8, style=_bold),
                                    _bold("Backend"),
                                ]
                            )
                        )
                        print(f"  {_dim('-' * 92)}")
                        for job in endpoints:
                            spec = job.get("spec") or {}
                            status = job.get("status", "")
                            if status in ("failed", "cancelled"):
                                status_str = _red(status)
                            elif status == "running":
                                status_str = _cyan(status)
                            elif status == "completed":
                                status_str = _green(status)
                            else:
                                status_str = _yellow(status)
                            print(
                                "  "
                                + "  ".join(
                                    [
                                        _fmt_cell(job.get("job_id", ""), 18),
                                        _fmt_cell(status_str, 10),
                                        _fmt_cell(spec.get("model", ""), 24),
                                        _fmt_cell(
                                            _format_duration_label(
                                                spec.get("idle_ttl_sec")
                                            ),
                                            8,
                                        ),
                                        _dim(_endpoint_backend_ref(job) or "-"),
                                    ]
                                )
                            )
                elif endpoint_action == "status":
                    job = parsed.get("endpoint") or parsed.get("job", {})
                    spec = job.get("spec") or {}
                    latest = _latest_attempt(job)
                    print(_section("Endpoint Details"))
                    print(_kv("job_id", _bold(job.get("job_id", ""))))
                    print(_kv("status", job.get("status", "")))
                    print(_kv("project_id", job.get("project_id", "")))
                    print(_kv("model", spec.get("model", "")))
                    print(
                        _kv(
                            "idle_ttl",
                            _format_duration_label(spec.get("idle_ttl_sec")) or "-",
                        )
                    )
                    print(_kv("backend_ref", latest.get("backend_ref") or "-"))
                elif endpoint_action == "up":
                    endpoint = parsed.get("endpoint") or {}
                    print(f"{_green('✓')} Endpoint provisioning submitted")
                    print(
                        _kv(
                            "job_id",
                            _bold(
                                endpoint.get("job_id")
                                or (parsed.get("job") or {}).get("job_id", "")
                            ),
                        )
                    )
                    print(_kv("model", endpoint.get("model") or args.model))
                    print(
                        _kv(
                            "idle_ttl",
                            _format_duration_label(endpoint.get("idle_ttl_sec")),
                        )
                    )
                else:
                    print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error",
                action="endpoint",
                message=str(e),
                output_format=args.output,
            )
        )
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
        print(
            "The 'rich' library is required for the monitor dashboard.\n"
            "Install it with: pip install 'rich>=13.0'"
        )
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
            job_type_filter=None,
            config_path=ctx.configure_file
            if ctx.configure_file and ctx.config_state.user.valid
            else None,
            user_cfg=ctx.user if isinstance(ctx.user, dict) else None,
            connection_info=conn_info,
            capabilities=capabilities,
            workflow_id=getattr(args, "workflow_id", None),
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
    """Unified dispatch for ``umrs workflow <action>``.

    Supported actions:
        upload          — Upload a workflow directory to the server
        list            — List registered workflows
        run             — Trigger execution of a registered workflow
        status          — Show execution status for a workflow run
        results         — Fetch results for a completed workflow run
        cancel          — Cancel a running workflow execution
        errors          — Show error logs for a failed workflow execution
        visualize       — Render the workflow DAG as ASCII art or Graphviz dot
        validate-sandbox — Validate a workflow sandbox (Dockerfile + workflow.json)
    """
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    wf_action = args.workflow_action
    exit_code = 0

    # ---- upload ----------------------------------------------------------
    if wf_action == "upload":
        return _action_upload_workflow(ctx)

    # ---- validate-sandbox ------------------------------------------------
    if wf_action == "validate-sandbox":
        return _action_validate_sandbox(ctx)

    # ---- visualize -------------------------------------------------------
    if wf_action == "visualize":
        return _action_visualize_workflow(ctx)

    # ---- list / run / status / results / cancel / errors -----------------
    try:
        if wf_action == "list":
            parsed = api.list_workflows(
                scope=args.scope, user=args.user, name=args.name
            )
        elif wf_action == "run":
            params = (
                parse_json_object_arg(args.params, field_name="params")
                if args.params
                else {}
            )
            resolved_bucket = _resolve_workflow_bucket(ctx, args.bucket)
            parsed = api.run_workflow(
                workflow_name=args.workflow_name,
                params=params,
                bucket=resolved_bucket,
            )
        elif wf_action == "status":
            parsed = api.workflow_status(args.execution_id)
        elif wf_action == "executions":
            parsed = api.workflow_executions(
                user=args.user,
                status=args.status,
                project_id=getattr(args, "project_id", None),
                limit=args.limit,
            )
        elif wf_action == "results":
            parsed = api.workflow_results(args.execution_id)
        elif wf_action == "cancel":
            parsed = api.workflow_cancel(args.execution_id)
        elif wf_action == "errors":
            parsed = api.workflow_errors(args.execution_id, tail=args.tail)
        elif wf_action == "logs":
            return _action_workflow_logs(ctx)
        else:
            print(
                api_json(
                    status="error",
                    message=f"Unknown workflow action: {wf_action}",
                    output_format=args.output,
                )
            )
            return 1

        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get("status") != "ok":
                exit_code = 1
        else:
            if parsed.get("status") == "ok":
                if wf_action == "run":
                    _print_workflow_run(parsed)
                elif wf_action == "list":
                    _print_workflow_list(parsed)
                elif wf_action == "status":
                    _print_workflow_status(parsed)
                elif wf_action == "executions":
                    _print_workflow_executions(parsed)
                elif wf_action == "results":
                    _print_workflow_results(parsed)
                elif wf_action == "cancel":
                    print(f"{_green('✓')} {parsed.get('message', 'Cancelled')}")
                elif wf_action == "errors":
                    _print_workflow_errors(parsed)
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error",
                action="workflow",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _print_workflow_list(parsed: dict) -> None:
    """Pretty-print the ``workflow list`` response."""
    workflows = parsed.get("workflows", [])
    if not workflows:
        print(f"  {_dim('No workflows registered.')}")
        return
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
        n_steps = len(w.get("steps", []))
        print(
            f"  {w['name']:30s}  v{w.get('version', '?'):8s}  {w.get('scope', ''):6s}  "
            f"{w.get('user_id', ''):12s}  {n_steps:5d}     "
            f"{_dim(w.get('description', ''))}"
        )


def _print_workflow_status(parsed: dict) -> None:
    """Pretty-print the ``workflow status`` response."""
    execution = parsed.get("execution", parsed)
    print(f"{_bold('Execution ID')}: {execution.get('execution_id', '?')}")
    print(f"{_bold('Workflow')}: {execution.get('workflow_name', '?')}")
    print(f"{_bold('Status')}: {execution.get('status', '?')}")
    print(f"{_bold('Progress')}: {execution.get('progress', 0.0):.0%}")
    if execution.get("submitted_at"):
        print(f"{_bold('Submitted')}: {execution['submitted_at']}")
    if execution.get("started_at"):
        print(f"{_bold('Started')}: {execution['started_at']}")
    if execution.get("finished_at"):
        print(f"{_bold('Finished')}: {execution['finished_at']}")
    if execution.get("error"):
        print(f"{_red('Error')}: {execution['error']}")
    steps = execution.get("steps", [])
    if steps:
        print(f"\n{_bold('Steps')}:")
        for s in steps:
            status = s.get("status", "?")
            marker = (
                _green("✓")
                if status == "succeeded"
                else _red("✗")
                if status == "failed"
                else _dim("•")
                if status == "skipped"
                else _dim("→")
            )
            print(
                f"  {marker} {s.get('name', '?'):30s}  {status:10s}  "
                f"{s.get('progress', 0.0):.0%}"
            )
            if s.get("error"):
                print(f"      {_red(s['error'])}")


def _print_workflow_executions(parsed: dict) -> None:
    """Pretty-print the ``workflow executions`` response."""
    rows = parsed.get("executions", [])
    print(_section("Workflow Executions"))
    if not rows:
        print(f"  {_dim('No executions in scope.')}")
        return
    for row in rows:
        status = str(row.get("status", "?"))
        marker = (
            _green("✓")
            if status == "succeeded"
            else _red("✗")
            if status == "failed"
            else _dim("■")
            if status == "cancelled"
            else _dim("▶")
        )
        wid = str(row.get("execution_id", "?"))
        name = str(row.get("workflow_name", "?"))
        user = str(row.get("user_id", "?"))
        steps = f"{row.get('steps_done', 0)}/{row.get('steps_total', 0)}"
        progress = f"{float(row.get('progress', 0.0)):.0%}"
        print(
            f"  {marker} {wid}  {name}  {status:10s}  steps {steps}  {progress:>4s}  {_dim(user)}"
        )
        if row.get("error"):
            print(f"      {_red(row['error'])}")
    count = int(parsed.get("count", len(rows)))
    print(f"  {_dim(f'({count} shown, newest first)')}")


def _print_workflow_results(parsed: dict) -> None:
    """Pretty-print the ``workflow results`` response."""
    execution = parsed.get("execution", parsed)
    results = execution.get("results", {})
    print(f"{_bold('Execution ID')}: {execution.get('execution_id', '?')}")
    print(f"{_bold('Status')}: {execution.get('status', '?')}")
    if results:
        print(f"\n{_bold('Results')}:")
        for key, value in results.items():
            if key == "blobs":
                continue
            print(f"  {_bold(key)}: {value}")
    artifacts = execution.get("artifacts", [])
    if artifacts:
        print(f"\n{_bold('Artifacts')}:")
        for a in artifacts:
            print(f"  • {a}")


def _print_workflow_errors(parsed: dict) -> None:
    """Pretty-print the ``workflow errors`` response (failed Attempts)."""
    execution = parsed.get("execution", parsed)
    error_logs = parsed.get("error_logs", [])
    print(f"{_bold('Execution ID')}: {execution.get('execution_id', '?')}")
    print(f"{_bold('Status')}: {execution.get('status', '?')}")
    if execution.get("error"):
        print(f"\n{_bold('Workflow Error')}:")
        print(f"  {_red(execution['error'])}")
    steps = execution.get("steps", [])
    failed_steps = [s for s in steps if s.get("status") == "failed"]
    if failed_steps:
        print(f"\n{_bold('Failed Steps')}:")
        for s in failed_steps:
            print(f"  {_red('✗')} {s.get('name', '?')}")
            if s.get("error"):
                print(f"      {_red(s['error'])}")
    if error_logs:
        print(f"\n{_bold('Error Logs')}:")
        for log_entry in error_logs:
            timestamp = log_entry.get("timestamp", "")
            level = log_entry.get("level", "ERROR")
            message = log_entry.get("message", "")
            print(f"  [{timestamp}] {level}: {message}")
    else:
        print(f"\n  {_dim('No error logs available.')}")


def _print_workflow_run(parsed: dict) -> None:
    """Pretty-print the ``workflow run`` response with execution ID."""
    execution_id = parsed.get("execution_id", "?")
    message = parsed.get("message", "OK")
    workflow_name = parsed.get("workflow_name", "?")
    steps = parsed.get("steps", 0)
    durable_job_id = parsed.get("durable_job_id")

    print(f"{_green('✓')} {message}")
    print(f"  {_bold('Workflow')}: {workflow_name}")
    print(f"  {_bold('Steps')}: {steps}")
    print(f"  {_bold('Execution ID')}: {_cyan(execution_id)}")
    if durable_job_id:
        print(f"  {_bold('Job ID')}: {durable_job_id}")
    print(f"  {_dim('Track with:')} umrs workflow status {execution_id}")
    print(f"  {_dim('Stream logs:')} umrs workflow logs {execution_id} --follow")


def _print_workflow_logs(parsed: dict) -> None:
    """Pretty-print the ``workflow logs`` response."""
    execution = parsed.get("execution", parsed)
    logs = parsed.get("logs", [])
    status = execution.get("status", "?")
    exec_id = execution.get("execution_id", "?")

    print(f"{_bold('Execution ID')}: {exec_id}")
    print(f"{_bold('Status')}: {status}")
    if logs:
        print(f"\n{_bold('Logs')}:")
        for log_entry in logs:
            timestamp = log_entry.get("timestamp", "")
            level = log_entry.get("level", "INFO")
            source = log_entry.get("source", "")
            message = log_entry.get("message", "")
            if level == "ERROR":
                level_str = _red(level)
            elif level == "WARN":
                level_str = _yellow(level)
            else:
                level_str = _dim(level)
            source_str = _dim(f"[{source}]") if source else ""
            print(f"  [{timestamp}] {level_str} {source_str} {message}")
    else:
        print(f"\n  {_dim('No logs available.')}")


def _action_workflow_logs(ctx):
    """Handle ``workflow logs`` — stream or fetch execution logs."""
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)

    execution_id = args.execution_id
    follow = getattr(args, "follow", False)
    interval = getattr(args, "interval", 2.0)
    tail = getattr(args, "tail", 200)

    if not follow:
        try:
            parsed = api.workflow_logs(execution_id, tail=tail)
            if args.output == "json":
                print(_json.dumps(parsed, ensure_ascii=False))
                if parsed.get("status") != "ok":
                    return 1
            else:
                if parsed.get("status") == "ok":
                    _print_workflow_logs(parsed)
                else:
                    print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                    return 1
            return 0
        except Exception as e:
            print(
                api_json(
                    status="error",
                    action="workflow_logs",
                    message=str(e),
                    output_format=args.output,
                )
            )
            if args.verbose:
                traceback.print_exc()
            return 1

    # --follow mode: poll until terminal state
    import time as _time

    seen_logs = 0
    try:
        while True:
            try:
                parsed = api.workflow_logs(execution_id, tail=tail)
            except Exception as e:
                print(f"{_red('✗')} Polling error: {e}")
                _time.sleep(interval)
                continue

            if parsed.get("status") != "ok":
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                return 1

            logs = parsed.get("logs", [])
            new_logs = logs[seen_logs:]
            seen_logs = len(logs)

            for log_entry in new_logs:
                timestamp = log_entry.get("timestamp", "")
                level = log_entry.get("level", "INFO")
                source = log_entry.get("source", "")
                message = log_entry.get("message", "")
                if level == "ERROR":
                    level_str = _red(level)
                elif level == "WARN":
                    level_str = _yellow(level)
                else:
                    level_str = _dim(level)
                source_str = _dim(f"[{source}]") if source else ""
                print(f"  [{timestamp}] {level_str} {source_str} {message}")

            execution = parsed.get("execution", {})
            status = execution.get("status", "")
            if status in ("succeeded", "failed", "cancelled"):
                if execution.get("error"):
                    print(f"\n{_red('Error')}: {execution['error']}")
                print(f"\n{_dim('Workflow reached terminal state:')} {status}")
                break

            _time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{_dim('Stopped following logs.')}")
    return 0


def _action_visualize_workflow(ctx):
    """Render the workflow DAG as ASCII art or Graphviz dot.

    Reads the workflow definition from the server (via ``workflow list``
    with a name filter) and uses :class:`DAGScheduler` to compute
    execution levels.  The resulting graph is printed to stdout.
    """
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    wf_name = args.name
    fmt = args.format
    exit_code = 0
    try:
        parsed = api.list_workflows(name=wf_name)
        if parsed.get("status") != "ok":
            print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
            return 1
        workflows = parsed.get("workflows", [])
        if not workflows:
            print(f"{_red('✗')} Workflow '{wf_name}' not found")
            return 1
        wf = workflows[0]
        steps = wf.get("steps", [])
        if not steps:
            print(f"{_red('✗')} Workflow '{wf_name}' has no steps")
            return 1

        from umrs.pipeline_sdk.dag import DAGScheduler
        from umrs.pipeline_sdk.workflow import StepSpec

        step_specs: list[StepSpec] = []
        for s in steps:
            spec = StepSpec(
                name=s.get("name", "?"),
                entrypoint=s.get("entrypoint", ""),
                config=s.get("config", {}),
                depends_on=s.get("depends_on"),
                requires=s.get("requires", []),
                model=s.get("model"),
                model_type=s.get("model_type", "llm"),
            )
            step_specs.append(spec)

        scheduler = DAGScheduler(step_specs)
        levels = scheduler.execution_levels()

        if fmt == "dot":
            _print_dag_dot(step_specs, scheduler, wf_name)
        else:
            _print_dag_ascii(step_specs, levels, wf_name)

    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error",
                action="workflow visualize",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _print_dag_ascii(step_specs: list, levels: list[list[int]], wf_name: str) -> None:
    """Print the DAG as ASCII art with execution levels."""
    print(f"\n{_bold('Workflow DAG')}: {wf_name}")
    print(f"{_dim('─' * 60)}")
    for level_idx, level in enumerate(levels):
        level_steps = [step_specs[i] for i in level]
        level_label = f"Level {level_idx}"
        if len(level_steps) == 1:
            step = level_steps[0]
            deps = step.depends_on or []
            dep_str = f" ← {', '.join(deps)}" if deps else ""
            print(f"  {level_label:12s} │ {step.name}{dep_str}")
        else:
            print(f"  {level_label:12s} │ (parallel)")
            for step in level_steps:
                deps = step.depends_on or []
                dep_str = f" ← {', '.join(deps)}" if deps else ""
                print(f"  {' ':12s} │ ├─ {step.name}{dep_str}")
        if level_idx < len(levels) - 1:
            print(f"  {' ':12s} │")
    print(f"{_dim('─' * 60)}")
    print(f"  {_dim(f'Total steps: {len(step_specs)}, Levels: {len(levels)}')}")


def _print_dag_dot(
    step_specs: list,
    scheduler: Any,
    wf_name: str,
) -> None:
    """Print the DAG in Graphviz dot format."""
    print(f"digraph {wf_name.replace('-', '_')} {{")
    print("  rankdir=TB;")
    print("  node [shape=box, style=filled, fillcolor=lightblue];")
    for spec in step_specs:
        print(f'  "{spec.name}";')
    for spec in step_specs:
        deps = spec.depends_on or []
        for dep in deps:
            print(f'  "{dep}" -> "{spec.name}";')
    print("}")


def _action_validate_sandbox(ctx):
    """Validate a workflow sandbox (Dockerfile + workflow.json) before running.

    Checks:
    1. workflow.json exists and is valid
    2. Dockerfile exists (if sandbox mode is desired)
    3. Dockerfile syntax is valid (basic checks)
    4. If --build is specified, attempt a real container build
    """
    args = ctx.args
    src_path = os.path.abspath(args.path)
    dry_run = args.dry_run
    build = args.build
    exit_code = 0
    errors: list[str] = []
    warnings: list[str] = []

    print(f"\n{_bold('Validating sandbox')}: {src_path}")
    print(f"{_dim('─' * 60)}")

    # 1. Check workflow.json
    wf_path = os.path.join(src_path, "workflow.json")
    if not os.path.isfile(wf_path):
        errors.append(f"workflow.json not found in {src_path}")
    else:
        try:
            with open(wf_path, encoding="utf-8") as fh:
                wf_data = _json.load(fh)
            from umrs.pipeline_sdk.workflow import WorkflowDefinition

            WorkflowDefinition.from_dict(wf_data)
            print(f"  {_green('✓')} workflow.json is valid")
        except Exception as e:
            errors.append(f"Invalid workflow.json: {e}")

    # 2. Check Dockerfile
    dockerfile_path = os.path.join(src_path, "Dockerfile")
    has_dockerfile = os.path.isfile(dockerfile_path)
    if has_dockerfile:
        print(f"  {_green('✓')} Dockerfile found")
        # 3. Validate Dockerfile syntax
        dockerfile_errors = _validate_dockerfile_syntax(dockerfile_path)
        if dockerfile_errors:
            for err in dockerfile_errors:
                errors.append(f"Dockerfile: {err}")
        else:
            print(f"  {_green('✓')} Dockerfile syntax is valid")
    else:
        warnings.append("No Dockerfile found — workflow will run in-process")

    # 4. Build test (optional)
    if build and has_dockerfile:
        print(f"  {_dim('Building container image...')}")
        build_ok, build_err = _build_sandbox_image(src_path, "sandbox-validate")
        if build_ok:
            print(f"  {_green('✓')} Container image built successfully")
        else:
            errors.append(f"Container build failed: {build_err}")

    # 5. Check step scripts
    step_script_errors = _validate_step_scripts(src_path, wf_path)
    if step_script_errors:
        for err in step_script_errors:
            warnings.append(f"Step script: {err}")

    print(f"{_dim('─' * 60)}")
    if errors:
        print(f"  {_red('✗')} Validation failed with {len(errors)} error(s):")
        for err in errors:
            print(f"    {_red('•')} {err}")
        exit_code = 1
    else:
        print(f"  {_green('✓')} Validation passed")
    if warnings:
        print(f"\n  {_dim('Warnings:')}")
        for w in warnings:
            print(f"    {_dim('•')} {w}")

    _close_control_mq(ctx)
    return exit_code


def _validate_dockerfile_syntax(dockerfile_path: str) -> list[str]:
    """Perform basic Dockerfile syntax validation.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []
    try:
        with open(dockerfile_path, encoding="utf-8") as fh:
            lines = fh.readlines()
    except Exception as e:
        return [f"Cannot read Dockerfile: {e}"]

    if not lines:
        return ["Dockerfile is empty"]

    # Check for FROM instruction
    has_from = False
    valid_instructions = {
        "FROM",
        "RUN",
        "CMD",
        "LABEL",
        "MAINTAINER",
        "EXPOSE",
        "ENV",
        "ADD",
        "COPY",
        "ENTRYPOINT",
        "VOLUME",
        "USER",
        "WORKDIR",
        "ARG",
        "ONBUILD",
        "STOPSIGNAL",
        "HEALTHCHECK",
        "SHELL",
    }
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        instruction = parts[0].upper() if parts else ""
        if instruction not in valid_instructions:
            errors.append(
                f"Line {line_num}: Unknown instruction '{parts[0] if parts else ''}'"
            )
        if instruction == "FROM":
            has_from = True
            if len(parts) < 2:
                errors.append(f"Line {line_num}: FROM requires an image reference")

    if not has_from:
        errors.append("Dockerfile must contain at least one FROM instruction")

    return errors


def _build_sandbox_image(src_path: str, workflow_name: str) -> tuple[bool, str | None]:
    """Build a container image from a Dockerfile.

    Returns (success, error_message).
    """
    import subprocess

    dockerfile_path = os.path.join(src_path, "Dockerfile")
    image_tag = f"{workflow_name}:latest"
    cmd = [
        "podman",
        "build",
        "--userns=host",
        "-t",
        image_tag,
        "-f",
        dockerfile_path,
        src_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            return False, result.stderr[-500:] if result.stderr else "Unknown error"
        return True, None
    except FileNotFoundError:
        return False, "podman not found — install podman or use --dry-run"
    except subprocess.TimeoutExpired:
        return False, "Container build timed out (600s)"
    except Exception as e:
        return False, str(e)


def _validate_step_scripts(src_path: str, wf_path: str) -> list[str]:
    """Validate that step scripts referenced in workflow.json exist.

    Returns a list of warning messages (empty if all scripts found).
    """
    warnings: list[str] = []
    if not os.path.isfile(wf_path):
        return warnings
    try:
        with open(wf_path, encoding="utf-8") as fh:
            wf_data = _json.load(fh)
    except Exception:
        return warnings

    steps = wf_data.get("steps", [])
    for step in steps:
        entrypoint = step.get("entrypoint", "")
        if not entrypoint or ":" not in entrypoint:
            continue
        module_name = entrypoint.split(":")[0]
        # Try common Python file extensions
        found = False
        for ext in (".py",):
            script_path = os.path.join(src_path, f"{module_name}{ext}")
            if os.path.isfile(script_path):
                found = True
                break
        if not found:
            warnings.append(
                f"Step '{step.get('name', '?')}': module '{module_name}' "
                f"not found in {src_path}"
            )
    return warnings


def _upload_workflow_via_zmq(ctx, tmp_zip_path, client_id, workflow_name):
    """Upload a workflow zip over the ZMQ control/data planes (local/tunnel)."""
    args = ctx.args
    data_mq = ctx.control_mq.get_data_queue(client_id, admin_token=ctx.admin_token)
    try:
        ctx.control_mq.set_timeouts(120_000, 120_000)
        ctx.control_mq.send_string(
            f"{TaskCmd.UPLOAD_WORKFLOW}:{client_id}:{workflow_name}"
        )
        data_mq.send_file(tmp_zip_path)
        _, reply = ctx.control_mq.recv_string()
        if reply != "ACKNOWLEDGE":
            raise RuntimeError(
                f"Unexpected reply from server during workflow upload: {reply}"
            )
        if args.output == "json":
            print(
                api_json(
                    action="upload-workflow",
                    workflow_name=workflow_name,
                    accepted=True,
                    output_format="json",
                )
            )
        else:
            print(
                f"{_green('✓')} Workflow {_bold(workflow_name)} uploaded successfully"
            )
    finally:
        ctx.control_mq.set_timeouts(ctx.control_mq.timeout, ctx.control_mq.timeout)
        with contextlib.suppress(Exception):
            data_mq.close()


def _upload_workflow_via_gateway(
    ctx, gateway_url, api_key, tmp_zip_path, client_id, workflow_name
):
    """Upload a workflow zip via the HTTP gateway. Returns an exit code."""
    args = ctx.args
    parsed = _http_upload_workflow(
        gateway_url,
        zip_path=tmp_zip_path,
        workflow_name=workflow_name,
        user_id=_jwt_user_id(api_key) or client_id,
        scope=getattr(args, "scope", "user") or "user",
        api_key=api_key,
    )
    if parsed.get("status") == "error":
        print(
            api_json(
                status="error",
                action="upload-workflow",
                message=parsed.get("message", "workflow upload failed"),
                output_format=args.output,
            )
        )
        return 1
    if args.output == "json":
        print(
            api_json(
                action="upload-workflow",
                workflow_name=workflow_name,
                accepted=True,
                output_format="json",
            )
        )
    else:
        print(f"{_green('✓')} Workflow {_bold(workflow_name)} uploaded successfully")
    return 0


def _close_control_mq(ctx):
    """Close the control ZMQ socket if one is attached to the context."""
    if ctx.control_mq is not None:
        ctx.control_mq.close()


def _validate_workflow_source(src_path, output_format):
    """Return an error response string if src_path is not a valid workflow dir."""
    if not os.path.isdir(src_path):
        return api_json(
            status="error",
            message=f"Source path is not a directory: {src_path}",
            output_format=output_format,
        )
    if not os.path.isfile(os.path.join(src_path, "workflow.json")):
        return api_json(
            status="error",
            message=f"No workflow.json found in {src_path}",
            output_format=output_format,
        )
    return None


def _zip_workflow_dir(src_path):
    """Zip the contents of src_path into a temp file and return its path."""
    import tempfile
    import zipfile

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip_path = tmp_zip.name
    with zipfile.ZipFile(tmp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_path):
            for f in files:
                abs_path = os.path.join(root, f)
                zf.write(abs_path, os.path.relpath(abs_path, src_path))
    return tmp_zip_path


def _dispatch_workflow_upload(ctx, tmp_zip_path, client_id, workflow_name):
    """Pick the upload transport and return an exit code.

    Prefer the HTTP gateway for external clients: the ZMQ data-plane ports are
    not reachable behind the auth gateway, so a data-queue upload would hang and
    time out. The gateway route streams the zip to UMRS over HTTP and matches
    the platform invariant that external clients upload via Gateway HTTP.
    """
    gateway_url = _get_gateway_url_from_ctx(ctx)
    api_key = _get_api_key_from_profile(ctx)
    if gateway_url and api_key:
        return _upload_workflow_via_gateway(
            ctx, gateway_url, api_key, tmp_zip_path, client_id, workflow_name
        )
    if ctx.control_mq is not None:
        _upload_workflow_via_zmq(ctx, tmp_zip_path, client_id, workflow_name)
        return 0
    print(
        api_json(
            status="error",
            action="upload-workflow",
            message=(
                "upload-workflow requires a ZMQ connection or a "
                "gateway_url + token in the connection profile"
            ),
            output_format=ctx.args.output,
        )
    )
    return 1


def _action_upload_workflow(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    src_path = os.path.abspath(args.path)
    err = _validate_workflow_source(src_path, args.output)
    if err is not None:
        print(err)
        _close_control_mq(ctx)
        return 1
    workflow_name = args.name or os.path.basename(src_path)
    tmp_zip_path = None
    exit_code = 0
    try:
        tmp_zip_path = _zip_workflow_dir(src_path)
        exit_code = _dispatch_workflow_upload(
            ctx, tmp_zip_path, client_id, workflow_name
        )
    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error",
                action="upload-workflow",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        if tmp_zip_path:
            with contextlib.suppress(Exception):
                os.remove(tmp_zip_path)
        _close_control_mq(ctx)
    return exit_code


def _resolve_workflow_bucket(ctx, bucket_arg):
    """Resolve a bucket alias to the full MinIO bucket name.

    The workflow runner expects the actual S3 bucket name
    (e.g. ``mnagaso2.opossum-analysis.demo-opossum``), not the
    short alias (e.g. ``demo-opossum``).  This helper performs the
    alias → full-name lookup against the client's bucket config.

    If *bucket_arg* is already a full bucket name (contains dots
    matching ``<user>.<xp>.<alias>``) the value is returned unchanged.
    If *bucket_arg* is ``None`` the workflow runs without a bucket.
    """
    if bucket_arg is None:
        return None

    buckets_cfg = getattr(ctx, "buckets_cfg", None)
    if buckets_cfg is None:
        return bucket_arg

    entry = buckets_cfg.get(bucket_arg)
    if entry is not None:
        path_to_bucket = getattr(entry, "path_to_bucket", None)
        if path_to_bucket:
            return path_to_bucket

    return bucket_arg


def _action_run_workflow(ctx):
    args = ctx.args
    client_id = _resolve_control_client_id(ctx.user)
    api = UMRSControlAPI(ctx.control_mq, client_id)
    exit_code = 0
    try:
        params = (
            parse_json_object_arg(args.params, field_name="params")
            if args.params
            else {}
        )
        resolved_bucket = _resolve_workflow_bucket(ctx, args.bucket)
        parsed = api.run_workflow(
            workflow_name=args.workflow_name,
            params=params,
            bucket=resolved_bucket,
        )
        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
            if parsed.get("status") != "ok":
                exit_code = 1
        else:
            if parsed.get("status") == "ok":
                print(f"{_green('✓')} {parsed.get('message', 'OK')}")
            else:
                print(f"{_red('✗')} {parsed.get('message', 'unknown error')}")
                exit_code = 1
    except Exception as e:
        exit_code = 1
        print(
            api_json(
                status="error",
                action="run-workflow",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return exit_code


def _action_list_buckets(ctx):
    args = ctx.args
    effective_username = args.username or (
        ctx.user.get("username") if isinstance(ctx.user, dict) else None
    )
    if not effective_username:
        print(
            api_json(
                status="error",
                message="No username provided and none found in config",
                output_format=args.output,
            )
        )
        if ctx.control_mq is not None:
            ctx.control_mq.close()
        return 1
    remote_info = None
    if not getattr(args, "no_check_remote", False):
        probe = _probe_remote_buckets(effective_username, ctx.user)
        # Collect configured bucket paths so we can compare
        configured_paths: set[str] = set()
        if ctx.buckets_cfg:
            for alias, entry in ctx.buckets_cfg.items():
                if entry.username == effective_username:
                    configured_paths.add(entry.path_to_bucket)
                    if entry.output_path_to_bucket:
                        configured_paths.add(entry.output_path_to_bucket)
        if "warning" in probe:
            remote_info = {
                "buckets": {},
                "extra_buckets": [],
                "warning": probe["warning"],
            }
        else:
            remote_info = _build_remote_info(
                probe["remote_names"], configured_paths, effective_username
            )
    print(
        action_list_buckets(
            ctx.buckets_cfg, effective_username, args.output, remote_info=remote_info
        )
    )
    if ctx.control_mq is not None:
        ctx.control_mq.close()
    return 0


def _action_purge_remote_buckets(ctx):
    """Delete extra remote S3 buckets that are not in the user config."""
    args = ctx.args
    effective_username = args.username or (
        ctx.user.get("username") if isinstance(ctx.user, dict) else None
    )
    if not effective_username:
        print(
            api_json(
                status="error",
                message="No username provided and none found in config",
                output_format=args.output,
            )
        )
        ctx.control_mq.close()
        return 1

    # Probe remote buckets
    probe = _probe_remote_buckets(effective_username, ctx.user)
    if "warning" in probe:
        print(
            api_json(
                status="error", message=probe["warning"], output_format=args.output
            )
            if args.output == "json"
            else f"  {_red('✗')} {probe['warning']}"
        )
        ctx.control_mq.close()
        return 1

    # Build configured paths set
    configured_paths: set[str] = set()
    if ctx.buckets_cfg:
        for alias, entry in ctx.buckets_cfg.items():
            if entry.username == effective_username:
                configured_paths.add(entry.path_to_bucket)
                if entry.output_path_to_bucket:
                    configured_paths.add(entry.output_path_to_bucket)

    remote_info = _build_remote_info(
        probe["remote_names"], configured_paths, effective_username
    )
    extra = remote_info.get("extra_buckets", [])

    # Filter by --bucket if specified
    bucket_filter = getattr(args, "buckets", None)
    if bucket_filter:
        filter_set = set(bucket_filter)
        extra = [eb for eb in extra if eb["name"] in filter_set]
        # Warn about filter names that don't match any extra bucket
        found_names = {eb["name"] for eb in extra}
        for f in bucket_filter:
            if f not in found_names:
                if f in configured_paths:
                    if args.output != "json":
                        print(f"  {_yellow('⚠')} {f} is a configured bucket — skipping")
                elif f not in probe["remote_names"]:
                    if args.output != "json":
                        print(f"  {_yellow('⚠')} {f} not found on remote — skipping")

    if not extra:
        msg = "No extra remote buckets to purge."
        print(
            api_json(status="ok", message=msg, output_format=args.output)
            if args.output == "json"
            else f"  {_green('✓')} {msg}"
        )
        ctx.control_mq.close()
        return 0

    # Show what will be deleted
    if args.output != "json":
        print(_section("Extra remote buckets to purge"))
        for eb in extra:
            print(
                f"  {eb['name']}  {_dim(str(eb.get('object_count', 0)) + ' objects')}"
            )
        print()

    # Confirmation
    if not getattr(args, "yes", False):
        try:
            answer = (
                input(f"Delete {len(extra)} bucket(s) and all their objects? [y/N] ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("  Aborted.")
            ctx.control_mq.close()
            return 0

    # Connect to S3 and delete
    try:
        import boto3
    except ImportError:
        print(
            api_json(
                status="error", message="boto3 not installed", output_format=args.output
            )
            if args.output == "json"
            else f"  {_red('✗')} boto3 not installed"
        )
        ctx.control_mq.close()
        return 1

    s3cfg_path_str = (ctx.user or {}).get(
        "s3_configuration_file"
    ) or default_s3cfg_path(effective_username)
    s3cfg_path = Path(s3cfg_path_str).expanduser()
    cfg = _read_s3cfg(s3cfg_path)
    endpoint_url, verify_ssl = _s3_endpoint_from_cfg(cfg)
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=cfg.get("access_key"),
        aws_secret_access_key=cfg.get("secret_key"),
        verify=verify_ssl,
    )

    deleted = []
    emptied = []
    errors = []
    for eb in extra:
        bname = eb["name"]
        try:
            # Delete all objects first (S3 requires empty bucket before deletion)
            obj_deleted = 0
            continuation_token = None
            while True:
                kwargs = {"Bucket": bname}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                resp = s3.list_objects_v2(**kwargs)
                objects = resp.get("Contents", [])
                if objects:
                    s3.delete_objects(
                        Bucket=bname,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objects]},
                    )
                    obj_deleted += len(objects)
                if not resp.get("IsTruncated"):
                    break
                continuation_token = resp.get("NextContinuationToken")
            # Try to remove the bucket itself
            try:
                s3.delete_bucket(Bucket=bname)
                deleted.append(bname)
                if args.output != "json":
                    print(
                        f"  {_green('✓')} Deleted {bname} ({obj_deleted} objects removed)"
                    )
            except Exception:
                # Local deletion denied — queue for server-side purge
                emptied.append(bname)
        except Exception as exc:
            errors.append({"bucket": bname, "error": str(exc)})
            if args.output != "json":
                print(f"  {_red('✗')} Failed on {bname}: {exc}")

    # Server-side purge for buckets that couldn't be deleted locally
    if emptied:
        if args.output != "json":
            print(
                f"  {_dim('Requesting server-side deletion for')} {len(emptied)} {_dim('bucket(s)...')}"
            )
        try:
            api = UMRSControlAPI(ctx.control_mq, effective_username)
            result = api.purge_buckets(emptied)
            srv_deleted = result.get("deleted", [])
            srv_errors = result.get("errors", [])
            deleted.extend(srv_deleted)
            for sd in srv_deleted:
                if args.output != "json":
                    print(f"  {_green('✓')} Deleted {sd} (via server)")
            for se in srv_errors:
                errors.append(se)
                if args.output != "json":
                    print(
                        f"  {_red('✗')} Server failed on {se.get('bucket', '?')}: {se.get('error', '?')}"
                    )
            # Remove successfully deleted from emptied list
            emptied = [b for b in emptied if b not in set(srv_deleted)]
        except Exception as exc:
            if args.output != "json":
                print(f"  {_yellow('⚠')} Server-side purge failed: {exc}")
            for b in emptied:
                errors.append({"bucket": b, "error": f"server purge failed: {exc}"})
            emptied = []

    ctx.control_mq.close()

    if args.output == "json":
        return api_json(
            status="ok" if not errors else "partial",
            deleted=deleted,
            emptied=emptied,
            errors=errors,
            output_format=args.output,
        )

    if deleted and not errors and not emptied:
        print(f"\n  {_green('✓')} Purged {len(deleted)} bucket(s).")
    elif errors:
        print(f"\n  {_yellow('⚠')} Deleted {len(deleted)}, failed {len(errors)}.")

    # Remove orphaned experiences from local user config so they stay gone
    # after server reload.
    if deleted and isinstance(ctx.user, dict) and ctx.config_state.user.valid:
        purged_xp_names = set()
        for bname in deleted:
            xp = _xp_name_from_bucket_path(bname, effective_username)
            if xp != "?":
                purged_xp_names.add(xp)
        if purged_xp_names:
            try:
                import json as _json_mod

                user_cfg_path = Path(ctx.config_state.user.path).expanduser()
                with open(user_cfg_path, encoding="utf-8") as f:
                    user_cfg = _json_mod.load(f)
                experiences = user_cfg.get("experiences", [])
                # Normalise xp_name to match bucket-path form (lower, hyphens)
                kept = []
                removed_names = []
                for xp in experiences:
                    xp_norm = _normalise_recovery_part(xp.get("xp_name", ""))
                    if xp_norm in purged_xp_names:
                        removed_names.append(xp.get("xp_name", xp_norm))
                    else:
                        kept.append(xp)
                if removed_names:
                    user_cfg["experiences"] = kept
                    _write_json_file(user_cfg_path, user_cfg)
                    for rn in removed_names:
                        print(
                            f"  {_green('✓')} Removed experience '{rn}' from local config"
                        )
                    print(f"  {_dim('Run')} umrs reload {_dim('to sync with server.')}")
            except Exception as exc:
                print(f"  {_yellow('⚠')} Could not update local config: {exc}")

    return 1 if errors else 0


def _maybe_write_s3cfg_from_status(
    parsed, user_cfg, configure_file=None, connection=None
):
    """Auto-write local s3cfg if the server delivered credentials in a reload status.

    When *connection* carries a ``minio_url`` that differs from the
    ``host_base`` inside the delivered content, the file is rewritten so the
    client can reach MinIO directly (the server-generated s3cfg often contains
    a localhost address that is unreachable from a remote client).
    """
    if not isinstance(parsed, dict) or not isinstance(user_cfg, dict):
        return
    # Prefer the original s3_configuration_file from the JSON config (before
    # any profile auto-discovery override), so we write to the path the server
    # knows about rather than an unrelated discovered s3cfg.
    s3cfg_path = None
    if configure_file:
        try:
            with open(os.path.expanduser(configure_file), encoding="utf-8") as _f:
                orig = _json.load(_f)
            s3cfg_path = orig.get("s3_configuration_file")
        except Exception:
            pass
    if not s3cfg_path:
        s3cfg_path = user_cfg.get("s3_configuration_file")
    if not s3cfg_path:
        return
    expanded = os.path.expanduser(s3cfg_path)
    if os.path.exists(expanded):
        return  # already have a local s3cfg

    # Look for s3cfg_content in any status entry (summary or summaries)
    s3cfg_content = None
    if "summary" in parsed and isinstance(parsed["summary"], dict):
        s3cfg_content = parsed["summary"].get("s3cfg_content")
    elif "summaries" in parsed and isinstance(parsed["summaries"], dict):
        for _k, entry in parsed["summaries"].items():
            if isinstance(entry, dict) and "s3cfg_content" in entry:
                s3cfg_content = entry["s3cfg_content"]
                break
    if not s3cfg_content:
        return

    # Rewrite host_base/host_bucket if the profile's minio_url points
    # elsewhere (e.g. remote host vs server-local 127.0.0.1).
    profile_minio = getattr(connection, "minio_url", None) if connection else None
    if profile_minio:
        try:
            from urllib.parse import urlparse

            parsed_url = urlparse(profile_minio)
            profile_host = parsed_url.hostname
            profile_port = parsed_url.port
            if profile_host:
                target_host = (
                    f"{profile_host}:{profile_port}" if profile_port else profile_host
                )
                import configparser as _cp

                cfg = _cp.ConfigParser(inline_comment_prefixes=("#", ";"))
                cfg.read_string(s3cfg_content)
                if cfg.has_option("default", "host_base"):
                    existing_host = cfg.get("default", "host_base").strip()
                    if existing_host != target_host:
                        s3cfg_content = s3cfg_content.replace(
                            existing_host, target_host
                        )
        except Exception:
            pass  # best-effort; write the original content

    try:
        os.makedirs(os.path.dirname(expanded), exist_ok=True)
        with open(expanded, "w", encoding="utf-8") as f:
            f.write(s3cfg_content)
        print(f"\n{_green('✓')} S3 credentials written to: {expanded}")
    except Exception as e:
        print(f"\n{_red('⚠')} Failed to write S3 credentials to '{expanded}': {e}")


def _action_status(ctx):
    args = ctx.args
    effective_username = args.username
    if not effective_username and isinstance(ctx.user, dict):
        effective_username = ctx.user.get("username")
    if not effective_username:
        print(
            api_json(
                status="error",
                message="No username provided and none found in config",
                output_format=args.output,
            )
        )
        ctx.control_mq.close()
        return 1
    api = UMRSControlAPI(ctx.control_mq, effective_username)
    try:
        status_target = _resolve_status_target(args)
        parsed = api.request_status(status_target)
        if args.output == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
        else:
            print(format_status_text(parsed, status_target))

        # Auto-write local s3cfg if the server delivered credentials
        _maybe_write_s3cfg_from_status(
            parsed,
            ctx.user,
            configure_file=ctx.configure_file,
            connection=ctx.connection,
        )
    except Exception as e:
        print(
            api_json(
                status="error",
                action=args.action,
                message=str(e),
                output_format=args.output,
            )
        )
        return 1
    finally:
        ctx.control_mq.close()
    return 0


def format_status_all_text(parsed):
    """Render a STATUS_ALL response as human-readable text."""
    if parsed.get("status") != "ok":
        return f"{_red('✗')} {parsed.get('message', 'status unavailable')}"

    clients = parsed.get("clients", {})
    if not clients:
        return _dim("(no clients known to server)")

    generated_at = parsed.get("generated_at", "")
    header_line = _bold("Status — all clients")
    if generated_at:
        header_line += f"  {_dim('(' + utc_to_local(generated_at) + ')')}"

    blocks = [header_line, ""]
    for client_id in sorted(clients.keys()):
        if client_id == "system":
            continue
        entry = clients[client_id]
        per_client = {"status": "ok", "client": client_id, "summaries": entry.get("summaries", {})}
        blocks.append(format_status_text(per_client, "process_objects"))
        blocks.append("")

    if blocks and blocks[-1] == "":
        blocks.pop()
    return "\n".join(blocks)


def _action_status_all(ctx):
    """Fetch the status of every client known to the server in one request."""
    args = ctx.args
    api = UMRSControlAPI(ctx.control_mq, "")
    try:
        parsed = api.request_status_all()
        if getattr(args, "output", "text") == "json":
            print(_json.dumps(parsed, ensure_ascii=False))
        else:
            print(format_status_all_text(parsed))
    except Exception as e:
        print(
            api_json(
                status="error",
                action="status-all",
                message=str(e),
                output_format=getattr(args, "output", "text"),
            )
        )
    finally:
        ctx.control_mq.close()
    return 0


def _action_maintenance(ctx):
    args = ctx.args

    # --- HTTP gateway path for reload ---
    gateway_url = _get_gateway_url_from_ctx(ctx)
    if gateway_url and args.action == "reload":
        api_key = _get_api_key_from_profile(ctx)
        # Include local user config in reload request so the server persists
        # any locally-added experiences before re-reading its configuration.
        user_config_json = None
        if isinstance(ctx.user, dict) and ctx.config_state.user.valid:
            try:
                cfg_path = Path(ctx.config_state.user.path).expanduser()
                if cfg_path.is_file():
                    with open(cfg_path, encoding="utf-8") as f:
                        user_config_json = _json.load(f)
            except Exception:
                pass  # best-effort; reload still proceeds without config upload
        parsed = _http_reload(
            gateway_url,
            api_key,
            user_config=user_config_json,
            user_id=ctx.user.get("username") if isinstance(ctx.user, dict) else None,
        )
        if parsed.get("status") == "ok":
            if args.output == "json":
                print(api_json(action=args.action, accepted=True, output_format="json"))
            else:
                print(f"{_green('✓')} {_bold(args.action)} accepted")
                summary = parsed.get("summary", "")
                if summary:
                    print(summary)
            # After HTTP reload, try to fetch s3cfg via ZMQ status
            if isinstance(ctx.user, dict) and ctx.control_mq is not None:
                try:
                    effective_username = ctx.user.get("username", "")
                    if effective_username:
                        status_api = UMRSControlAPI(ctx.control_mq, effective_username)
                        status_parsed = status_api.request_status("all")
                        _maybe_write_s3cfg_from_status(
                            status_parsed,
                            ctx.user,
                            configure_file=ctx.configure_file,
                            connection=ctx.connection,
                        )
                except Exception:
                    pass  # s3cfg delivery is best-effort
        else:
            if args.output == "json":
                print(
                    api_json(
                        status="error",
                        action=args.action,
                        accepted=False,
                        message=parsed.get("message", "rejected"),
                        output_format="json",
                    )
                )
            else:
                msg = parsed.get("message") or parsed.get("error") or "rejected"
                print(f"{_red('✗')} {_bold(args.action)} rejected: {msg}")
        return 0

    # --- ZMQ path (original) ---
    client_id = (
        ctx.user.get("username") if isinstance(ctx.user, dict) else None
    ) or os.environ.get("UMRS_CLIENT_ID", "_")
    api = UMRSControlAPI(
        ctx.control_mq,
        client_id,
        ctx.configure_file
        if ctx.configure_file and ctx.config_state.user.valid
        else None,
        admin_token=ctx.admin_token,
    )
    try:
        parsed = api.maintenance(args.action)
        if parsed.get("status") == "ok":
            if args.output == "json":
                print(api_json(action=args.action, accepted=True, output_format="json"))
            else:
                print(f"{_green('✓')} {_bold(args.action)} accepted")
                summary = parsed.get("summary", "")
                if summary:
                    print(summary)
            # After successful reload, auto-write s3cfg if delivered
            if args.action == "reload" and isinstance(ctx.user, dict):
                _maybe_write_s3cfg_from_status(
                    parsed,
                    ctx.user,
                    configure_file=ctx.configure_file,
                    connection=ctx.connection,
                )
        else:
            if args.output == "json":
                print(
                    api_json(
                        status="error",
                        action=args.action,
                        accepted=False,
                        message=parsed.get("message", "rejected"),
                        output_format="json",
                    )
                )
            else:
                print(
                    f"{_red('✗')} {_bold(args.action)} rejected: {parsed.get('message', '')}"
                )
    except Exception as e:
        if args.output == "json":
            print(
                api_json(
                    status="error",
                    action=args.action,
                    message=str(e),
                    output_format="json",
                )
            )
        else:
            print(f"{_red('✗')} {_bold(args.action)} error: {e}")
    finally:
        ctx.control_mq.close()
    return 0


def _action_export_embeddings(ctx):
    args = ctx.args
    try:
        bucket_alias_arg = args.bucket_alias
        bucket_alias, _ = decompose_bucket_alias_arg(bucket_alias_arg)
        entry = ctx.buckets_cfg.get(bucket_alias)
        if entry is None:
            print(
                api_json(
                    status="error",
                    message=f"Bucket alias not found: {bucket_alias}",
                    output_format=args.output,
                )
            )
            ctx.control_mq.close()
            return 1
        path_to_bucket = entry.path_to_bucket
        effective_username = getattr(args, "username", None) or (
            ctx.user.get("username") if isinstance(ctx.user, dict) else None
        )
        if not effective_username:
            print(
                api_json(
                    status="error",
                    message="No username provided and none found in config",
                    output_format=args.output,
                )
            )
            ctx.control_mq.close()
            return 1
        raw_response = ctx.control_mq.request_export_embeddings(
            effective_username,
            path_to_bucket,
            args.export_format,
            args.export_profile,
        )
        response = _json.loads(raw_response)
        if response.get("status") != "ok":
            print(
                api_json(
                    status="error",
                    action="export_embeddings",
                    message=response.get("message", "Server returned an error"),
                    output_format=args.output,
                )
            )
            ctx.control_mq.close()
            return 1
        columns = response["columns"]
        rows = response["rows"]
        local_path = args.local_path
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        if args.export_format == "csv":
            import csv

            with open(local_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([str(v) if v is not None else "" for v in row])
        else:  # jsonl
            with open(local_path, "w", encoding="utf-8") as f:
                for row in rows:
                    record = {}
                    for col, val in zip(columns, row, strict=False):
                        record[col] = val
                    f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        print(
            api_json(
                action="export_embeddings",
                database=response.get("database", ""),
                table=response.get("table", ""),
                rows_exported=response.get("rows_exported", len(rows)),
                columns=columns,
                local_path=local_path,
                format=args.export_format,
                profile=response.get("profile", args.export_profile),
                output_format=args.output,
            )
        )
    except Exception as e:
        print(
            api_json(
                status="error",
                action="export_embeddings",
                message=str(e),
                output_format=args.output,
            )
        )
        if args.verbose:
            traceback.print_exc()
    finally:
        ctx.control_mq.close()
    return 0


def _action_s3_transfer(ctx):
    """Handle upload_file, upload_dir, and download_results actions."""
    args = ctx.args
    bucket_alias_arg = args.bucket_alias
    bucket_alias, subdir = decompose_bucket_alias_arg(bucket_alias_arg)

    selected_fields = [
        "username",
        "xp_name",
        "modality_name",
        "s3_configuration_file",
        "metadata",
    ]
    if args.action == "download_results":
        selected_fields.append("output_path_to_bucket")
    else:
        selected_fields.append("path_to_bucket")
    _bucket_result = ctx.buckets_cfg.get(bucket_alias, filter=selected_fields)
    if _bucket_result is None:
        available = list(ctx.buckets_cfg.keys())
        if available:
            message = (
                f"Bucket alias '{bucket_alias}' not found in configuration. "
                f"Available aliases: {', '.join(sorted(available))}. "
                f"To register a new bucket, add its experience and reload:\n"
                f"  umrs experience add <experience.json>\n"
                f"  umrs reload"
            )
        else:
            message = (
                f"Bucket alias '{bucket_alias}' not found in configuration. "
                f"No bucket aliases are configured. "
                f"Add an experience and reload to register its bucket:\n"
                f"  umrs experience add <experience.json>\n"
                f"  umrs reload"
            )
        print(api_json(status="error", message=message, output_format=args.output))
        ctx.control_mq.close()
        return 1
    username, xp_name, modality, s3_configuration_file, metadata, path_to_bucket = (
        _bucket_result
    )

    username_arg = getattr(args, "username", None)
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

    # Use the profile's minio_url to override the s3cfg host_base when the
    # client is connecting remotely (the server-generated s3cfg may contain a
    # localhost address that is unreachable from the client machine).
    # Only apply when a profile file was explicitly given (-c flag), so we
    # don't override use_https=True in .s3cfg with http:// from the profile.
    endpoint_override = None
    if hasattr(ctx, "connection") and ctx.connection is not None:
        if getattr(ctx.connection, "profile_name", None) is not None:
            profile_minio = getattr(ctx.connection, "minio_url", None)
            if profile_minio:
                endpoint_override = profile_minio

    s3, bucket_location = connect_to_s3(
        expanded_cfg, endpoint_url_override=endpoint_override
    )

    if args.action == "download_results":
        full_bucket_name, success = check_bucket_name(
            s3, xp_name, bucket_alias, username, output=True, indent=False
        )
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
                print(
                    f"Note: Output bucket '{full_bucket_name}' does not exist yet on S3 — will wait for server."
                )
    else:
        full_bucket_name, success = check_bucket_name(
            s3, xp_name, bucket_alias, username, indent=False
        )
        if not success or full_bucket_name != path_to_bucket:
            ctx.control_mq.close()
            data_mq.close()
            print(
                f"Error:\tNo bucket {bucket_alias} for modality {modality} and for user {username}"
            )
            return 1

    if args.verbose:
        print(
            f"Bucket '{bucket_alias}' for modality '{modality}' and for user '{username}': OK!"
        )

    try:
        metadata_args = check_and_build_metadata(metadata, args.k)
    except Exception as e:
        print(
            api_json(
                status="error",
                message=f"Metadata error: {e}",
                output_format=args.output,
            )
        )
        ctx.control_mq.close()
        data_mq.close()
        return 1

    if args.verbose:
        action_label = "Retrieving" if args.action == "download_results" else "Storing"
        print(
            f"{action_label} data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'"
        )

    def _notify_after_upload(username, count, blocking=False):
        ok = ctx.control_mq.notify_server(
            username, TaskCmd.PROCESS_OBJECTS, count, blocking=blocking
        )
        if ok:
            return True
        warning = operation_ignored_warning(
            "notify_server",
            f"server rejected PROCESS_OBJECTS notification for {count} object(s)",
        )
        body = emit_cli_warning(warning, output_format=args.output)
        if args.output == "json":
            print(api_json(**body))
        return False

    def process_file(
        s3,
        full_bucket_name,
        subdir,
        file_path,
        modality,
        data_mq,
        metadata_args,
        username,
        notify_server=True,
        blocking=False
    ):
        if not os.path.isfile(file_path):
            print(f"Error: file {file_path} is not a valid.")
            return None, 0, True

        category = modality_category(modality)
        print(
            f"Processing file '{file_path}' for modality '{modality}' (category: {category})"
        )

        if category == "image":
            s3_key = s3_store_image(
                s3,
                full_bucket_name,
                subdir,
                file_path,
                data_mq,
                metadata_args,
                check=(modality not in ("volume", "movie", "volume_movie")),
            )
            notify_ok = True
            if s3_key and notify_server:
                notify_ok = _notify_after_upload(username, 1, blocking=blocking)
            return s3_key, 1, notify_ok
        elif category == "text_file":
            s3_key = s3_store_text_file(
                s3, full_bucket_name, subdir, file_path, data_mq, metadata_args
            )
            notify_ok = True
            if s3_key and notify_server:
                notify_ok = _notify_after_upload(username, 1, blocking=blocking)
            return s3_key, 1, notify_ok
        elif category == "text":
            try:
                with open(file_path) as file:
                    nb_objects = 0
                    s3_key_list = []
                    for line in file:
                        line = line.strip()
                        if line != "":
                            s3_key = s3_store_txt(
                                s3,
                                full_bucket_name,
                                subdir,
                                line,
                                data_mq,
                                metadata_args,
                            )
                            s3_key_list.append(s3_key)
                            nb_objects += 1
                    notify_ok = True
                    if notify_server and nb_objects > 0:
                        notify_ok = _notify_after_upload(
                            username, nb_objects, blocking=blocking
                        )
                    return s3_key_list, nb_objects, notify_ok
            except UnicodeDecodeError as e:
                print(f"Unicode decoding error: {e}")
                print(
                    f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file."
                )
        else:
            print(
                f"Error: Unknown modality category '{category}' (modality: '{modality}'). Nothing has been sent to the server"
            )

        return None, 0, True

    def process_dir(
        path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username, blocking = False
    ):
        if not os.path.isdir(path):
            print(f"Error: {path} is not a valid directory.")
            return [], [], True

        s3_key_list = []
        file_list = []
        nb_objects = 0
        # Recursively walk *path* so deeply nested data (e.g. the
        # opossum ``opossum_NNN_X/opossumX/Schedule/YYYYMMDD/HH/*.mp4``
        # hierarchy) is uploaded. Each file's relative directory becomes
        # part of the S3 key prefix, preserving the camera/Schedule date
        # hierarchy expected by the workflow step.
        norm_path = os.path.abspath(path)
        nb_files_seen = 0
        broken_symlinks = []
        for root, _dirs, files in os.walk(norm_path):
            for fname in files:
                nb_files_seen += 1
                file_path = os.path.join(root, fname)
                if not os.path.isfile(file_path):
                    if os.path.islink(file_path):
                        broken_symlinks.append(
                            f"{file_path} -> {os.readlink(file_path)}"
                        )
                    continue

                rel_dir = os.path.relpath(root, norm_path)
                if rel_dir in (".", ""):
                    file_subdir = subdir
                elif subdir:
                    file_subdir = f"{subdir}/{rel_dir}"
                else:
                    file_subdir = rel_dir

                s3_key, nb_sent, _notify_ok = process_file(
                    s3,
                    full_bucket_name,
                    file_subdir,
                    file_path,
                    modality,
                    data_mq,
                    metadata_args,
                    username,
                    notify_server=False,
                )
                if s3_key:
                    if isinstance(s3_key, list):
                        s3_key_list.extend(s3_key)
                        assert len(s3_key) == nb_sent
                        file_list.extend([file_path] * nb_sent)
                    else:
                        s3_key_list.append(s3_key)
                        file_list.append(file_path)
                    nb_objects += nb_sent

        notify_ok = True
        if nb_objects > 0:
            notify_ok = _notify_after_upload(username, nb_objects, blocking=blocking)
        elif nb_files_seen == 0:
            print(f"Error: directory '{path}' contains no files to upload.")
        elif broken_symlinks:
            print(
                f"Error: none of the {nb_files_seen} entries in '{path}' "
                "could be uploaded because they are broken symlinks "
                "(their targets do not exist):"
            )
            for link in broken_symlinks:
                print(f"  - {link}")
            print(
                "Re-stage the input files so the symlinks point at existing "
                "targets, then retry."
            )
        else:
            print(
                f"Error: found {nb_files_seen} file(s) in '{path}' but none "
                f"were accepted for modality '{modality}'. Check that the "
                "files match the expected type for this modality."
            )
        return s3_key_list, file_list, notify_ok

    def download_results_fn(full_bucket_name, modality, subdir):
        effective_username = args.username
        verbose = args.verbose
        wait_for_ready = getattr(args, "wait", True)
        timeout_val = getattr(args, "timeout", None)
        max_wait_sec = int(timeout_val) if timeout_val is not None else 3600
        if not effective_username and isinstance(ctx.user, dict):
            effective_username = ctx.user.get("username")
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

        def _fetch_readiness():
            raw_response = ctx.control_mq.request_status(
                effective_username, TaskCmd.PROCESS_OBJECTS
            )
            if raw_response is None:
                return parse_process_objects_readiness(None, full_bucket_name)
            try:
                parsed = _json.loads(raw_response)
            except (_json.JSONDecodeError, TypeError):
                return parse_process_objects_readiness(None, full_bucket_name)
            if not isinstance(parsed, dict):
                return parse_process_objects_readiness(None, full_bucket_name)
            return parse_process_objects_readiness(parsed, full_bucket_name)

        readiness = _fetch_readiness()
        if not readiness.ready:
            if not wait_for_ready:
                payload, _warning = incomplete_error_payload(
                    readiness,
                    full_bucket_name=full_bucket_name,
                    subdir=subdir,
                    modality=modality,
                    output_format=args.output,
                )
                return False, payload

            poll_timeout = 1
            wait_start = time.time()
            print(
                format_wait_progress(readiness, elapsed_sec=0),
                flush=True,
                file=sys.stderr,
            )
            while not readiness.ready:
                elapsed = int(time.time() - wait_start)
                if elapsed >= max_wait_sec:
                    payload, _warning = incomplete_error_payload(
                        readiness,
                        full_bucket_name=full_bucket_name,
                        subdir=subdir,
                        modality=modality,
                        output_format=args.output,
                        elapsed_sec=elapsed,
                        timed_out=True,
                        max_wait_sec=max_wait_sec,
                    )
                    return False, payload

                res = ctx.control_mq.wait_for_new_data(
                    effective_username, full_bucket_name, timeout=poll_timeout
                )
                kind = classify_wait_reply(res)
                if kind == "timeout":
                    poll_timeout = min(poll_timeout * 2, 64)
                elif kind == "update":
                    poll_timeout = 1
                elif kind == "hard_error":
                    response = {
                        "status": "error",
                        "message": (
                            f"Cannot wait for results for bucket "
                            f"'{full_bucket_name}': server returned '{res}'."
                        ),
                        "bucket": full_bucket_name,
                        "subdir": subdir,
                        "modality": modality,
                        "server_reply": res,
                        "output_format": args.output,
                    }
                    return False, response
                else:
                    response = {
                        "status": "error",
                        "message": (
                            f"Unexpected reply while waiting for results: '{res}'."
                        ),
                        "bucket": full_bucket_name,
                        "subdir": subdir,
                        "modality": modality,
                        "server_reply": res,
                        "output_format": args.output,
                    }
                    return False, response

                readiness = _fetch_readiness()
                if not readiness.ready:
                    print(
                        format_wait_progress(
                            readiness,
                            elapsed_sec=int(time.time() - wait_start),
                        ),
                        flush=True,
                        file=sys.stderr,
                    )
                elif verbose:
                    print(readiness.message, flush=True)

        print_only_new = False
        objects_received = set()
        objects_list, _, checksums = list_s3_bucket(s3, full_bucket_name, prefix=subdir)
        download_result = download_objects(
            s3,
            objects_list,
            checksums,
            args.local_path,
            files_allready_received=objects_received,
            force_overwrite=args.force_overwrite,
            print_only_new=print_only_new,
        )
        if download_result is None:
            response = {
                "status": "error",
                "message": (
                    f"Local path {args.local_path} is not a directory. Nothing downloaded."
                ),
                "bucket": full_bucket_name,
                "subdir": subdir,
                "modality": modality,
                "output_format": args.output,
            }
            return False, response

        downloaded_files, _ = download_result
        objects_received.update(downloaded_files or [])

        if not objects_list:
            response = {
                "status": "warning",
                "message": (
                    f"Task is complete but no objects found in output bucket "
                    f"'{full_bucket_name}'"
                    + (f" (prefix '{subdir}')" if subdir else "")
                    + "."
                ),
                "bucket": full_bucket_name,
                "local_path": args.local_path,
                "subdir": subdir,
                "modality": modality,
                "output_format": args.output,
                "objects_received": "[]",
            }
            return True, response

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

    upload_exit_code = 0
    try:
        if args.action == "upload_file":
            s3_key, nb_sent, notify_ok = process_file(
                s3,
                full_bucket_name,
                subdir,
                args.file_path,
                modality,
                data_mq,
                metadata_args,
                username,
                notify_server=True,
                blocking=args.blocking
            )
            if s3_key and not notify_ok:
                upload_exit_code = 1
            elif s3_key:
                print(
                    api_json(
                        s3_key=s3_key,
                        bucket=full_bucket_name,
                        subdir=subdir,
                        modality=modality,
                        output_format=args.output,
                    )
                )
            else:
                print(
                    api_json(
                        status="error",
                        message=f"Cannot process file: {args.file_path}",
                        bucket=full_bucket_name,
                        subdir=subdir,
                        modality=modality,
                        output_format=args.output,
                    )
                )
                upload_exit_code = 1

        elif args.action == "upload_dir":
            s3_key_list, file_list, notify_ok = process_dir(
                args.path,
                s3,
                full_bucket_name,
                subdir,
                modality,
                data_mq,
                metadata_args,
                username,
                blocking=args.blocking
            )
            if len(s3_key_list) > 0 and not notify_ok:
                upload_exit_code = 1
            elif len(s3_key_list) > 0:
                print(
                    api_json(
                        s3_key_list=s3_key_list,
                        file_list=file_list,
                        bucket=full_bucket_name,
                        subdir=subdir,
                        modality=modality,
                        output_format=args.output,
                    )
                )
            else:
                print(
                    api_json(
                        status="error",
                        message=f"Cannot process directory: {args.path}",
                        bucket=full_bucket_name,
                        subdir=subdir,
                        modality=modality,
                        output_format=args.output,
                    )
                )
                upload_exit_code = 1
        elif args.action == "download_results":
            ok, response = download_results_fn(full_bucket_name, modality, subdir)
            if not ok and args.output != "json" and response.get("warnings"):
                from umrs.shared.warnings import UMRSWarning, WarningCategory

                for warning_dict in response["warnings"]:
                    category_value = warning_dict.get("category", "recoverable_error")
                    warning = UMRSWarning(
                        category=WarningCategory(category_value),
                        message=warning_dict.get("message", ""),
                        details=warning_dict.get("details", {}),
                        suggestion=warning_dict.get("suggestion"),
                        retry_after=warning_dict.get("retry_after_seconds"),
                    )
                    print_warning_structured(warning)
            print(api_json(**response))
            ctx.control_mq.close()
            data_mq.close()
            return 0 if ok and response.get("status") != "error" else 1
    except Exception as e:
        print(api_json(status="error", message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
        ctx.control_mq.close()
        data_mq.close()
        return 1

    ctx.control_mq.close()
    data_mq.close()
    if args.action in ("upload_file", "upload_dir"):
        return upload_exit_code
    return 0


# === Umbrella command handlers ===


def _action_bucket(ctx):
    """Dispatch bucket subcommands."""
    action = getattr(ctx.args, "bucket_action", None)
    if action == "list":
        return _action_list_buckets(ctx)
    elif action == "purge":
        return _action_purge_remote_buckets(ctx)
    elif action == "create":
        try:
            s3_mod = _s3()
            if not hasattr(s3_mod, "create_bucket"):
                raise SystemExit(
                    "Bucket creation requires client-full transport stack.\n"
                    + _CLIENT_FULL_HINT
                )
            bucket_name = ctx.args.bucket_create_name  # set by argparse as 'name'
            s3_mod.create_bucket(bucket_name)
            print(f"✓ Bucket '{bucket_name}' created")
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        print(
            "Unknown bucket action. Use: bucket list|purge|create",
            file=sys.stderr,
        )
        return 1


def _action_data(ctx):
    """Dispatch data subcommands."""
    action = getattr(ctx.args, "data_action", None)
    if action == "upload":
        ctx.args.action = "upload_file"
        return _action_s3_transfer(ctx)
    elif action == "dir-upload":
        ctx.args.action = "upload_dir"
        return _action_s3_transfer(ctx)
    elif action == "download":
        ctx.args.action = "download_results"
        return _action_s3_transfer(ctx)
    else:
        print(
            "Unknown data action. Use: data upload|dir-upload|download",
            file=sys.stderr,
        )
        return 1


def _action_project(ctx):
    """Dispatch project subcommands (admin-only)."""
    action = getattr(ctx.args, "project_action", None)
    admin_token = ctx.args.admin_token or os.environ.get("UMRS_ADMIN_TOKEN")
    if not admin_token:
        print("Error: project commands require --admin-token", file=sys.stderr)
        return 2

    control_mq = ctx.control_mq
    if not control_mq:
        print(
            "Error: project commands require ZMQ control connection",
            file=sys.stderr,
        )
        return 1

    try:
        from umrs.client.control_api import UMRSControlAPI

        api = UMRSControlAPI(control_mq=control_mq, admin_token=admin_token)

        if action == "list":
            result = api._request_json(TaskCmd.ADMIN_LIST_PROJECTS, {})
            if ctx.args.output == "json":
                print(_json.dumps(result, indent=2))
            else:
                projects = result.get("projects", [])
                if not projects:
                    print("(no projects)")
                else:
                    print(
                        f"{'Project ID':<30} {'Name':<30} {'Members':<10} {'Status':<10}"
                    )
                    print("-" * 80)
                    for p in projects:
                        print(
                            f"{p.get('project_id', ''):<30}"
                            f"{p.get('name', ''):<30}"
                            f"{len(p.get('members', [])):<10}"
                            f"{p.get('status', ''):<10}"
                        )
            return 0

        elif action == "add":
            result = api._request_json(
                TaskCmd.ADMIN_ADD_PROJECT_MEMBER,
                {
                    "username": ctx.args.username,
                    "project_id": ctx.args.project_id,
                },
            )
            print(f"✓ Added {ctx.args.username} to project {ctx.args.project_id}")
            return 0

        elif action == "remove":
            result = api._request_json(
                TaskCmd.ADMIN_REMOVE_PROJECT_MEMBER,
                {
                    "username": ctx.args.username,
                    "project_id": ctx.args.project_id,
                },
            )
            print(f"✓ Removed {ctx.args.username} from project {ctx.args.project_id}")
            return 0

        elif action == "info":
            result = api._request_json(
                TaskCmd.ADMIN_GET_PROJECT_INFO,
                {"project_id": ctx.args.project_id},
            )
            if ctx.args.output == "json":
                print(_json.dumps(result, indent=2))
            else:
                print(_json.dumps(result, indent=2, default=str))
            return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# Action dispatch table — maps CLI action name to handler function
_ACTION_HANDLERS = {
    "ping": _action_ping,
    "model": _action_model,
    "job": _action_job,
    "endpoint": _action_endpoint,
    "monitor": _action_monitor,
    "workflow": _action_workflow,
    # New umbrella handlers
    "bucket": _action_bucket,
    "data": _action_data,
    "project": _action_project,
    # Legacy flat commands (backward compat)
    "list_buckets": _action_list_buckets,
    "purge_remote_buckets": _action_purge_remote_buckets,
    "upload_file": _action_s3_transfer,
    "upload_dir": _action_s3_transfer,
    "download_results": _action_s3_transfer,
    "status": _action_status,
    "status-all": _action_status_all,
    "recompute": _action_maintenance,
    "update": _action_maintenance,
    "check": _action_maintenance,
    "reload": _action_maintenance,
    "export_embeddings": _action_export_embeddings,
}


def _maybe_load_saved_curve_session() -> bool:
    """Try to load a saved CURVE session from ``~/.config/r4/curve-session.json``.

    If a valid (non-expired, >60s remaining) session exists, inject its keys
    into the environment so MessageQueue picks them up.  Returns True on success.

    This avoids creating a new session on every client invocation when ``r4 login``
    has already established one.  No r4 package import — reads the JSON directly.
    """
    try:
        import json as _json
        import time as _time
        from pathlib import Path as _Path

        xdg = os.environ.get("XDG_CONFIG_HOME")
        session_path = (
            _Path(xdg, "r4") if xdg else _Path.home() / ".config" / "r4"
        ) / "curve-session.json"
        if not session_path.is_file():
            return False
        data = _json.loads(session_path.read_text(encoding="utf-8"))
        expires_at = data.get("expires_at", 0)
        if _time.time() >= (expires_at - 60):
            return False
        server_pk = data.get("server_public_key", "")
        client_pk = data.get("client_public_key", "")
        client_sk = data.get("client_secret_key", "")
        if not (server_pk and client_pk and client_sk):
            return False
        os.environ["UMRS_ZMQ_AUTH_TYPE"] = "session"
        os.environ["UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY"] = server_pk
        os.environ["UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY"] = client_pk
        os.environ["UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY"] = client_sk
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Loaded saved CURVE session from %s (TTL %.0fs)",
            session_path,
            expires_at - _time.time(),
        )
        return True
    except Exception:
        return False


def _is_session_mode(connection) -> bool:
    """True when the connection uses session-mode CURVE ZMQ auth via a gateway."""
    if not getattr(connection, "gateway_url", None):
        return False
    profile_auth_type = getattr(connection, "zmq_auth_type", None) or ""
    auth_type = (profile_auth_type or os.environ.get("UMRS_ZMQ_AUTH_TYPE", "")).lower()
    return auth_type == "session"


def _resolve_session_jwt(connection) -> str | None:
    """Resolve a JWT for CURVE session registration.

    Order: env vars → profile ``.token`` file → r4 tokens store.
    """
    jwt_token = os.environ.get("UMRS_JWT_TOKEN") or os.environ.get("UMRS_API_KEY")
    if not jwt_token:
        profile_path = getattr(connection, "profile_path", None)
        if profile_path:
            from pathlib import Path as _Path

            token_path = _Path(profile_path).with_suffix(".token")
            if token_path.is_file():
                jwt_token = token_path.read_text(encoding="utf-8").strip()
    if not jwt_token:
        profile_name = getattr(connection, "profile_name", None)
        jwt_token = _try_read_r4_token(profile_name)
    return jwt_token or None


def _persist_curve_session(info, gateway_url: str) -> None:
    """Persist a freshly-bootstrapped CURVE session to ``~/.config/r4/curve-session.json``.

    Matches the on-disk format written by ``r4 login`` so later client/r4
    invocations reuse it.  Best-effort: failures are swallowed (the in-process
    env vars set by ``bootstrap_curve_session`` already cover this invocation).
    """
    try:
        import json as _json
        from pathlib import Path as _Path

        xdg = os.environ.get("XDG_CONFIG_HOME")
        config_dir = _Path(xdg, "r4") if xdg else _Path.home() / ".config" / "r4"
        config_dir.mkdir(parents=True, exist_ok=True)
        session_path = config_dir / "curve-session.json"
        payload = {
            "session_id": info.session_id,
            "server_public_key": info.server_public_key,
            "client_public_key": info.client_public_key,
            "client_secret_key": info.client_secret_key,
            "expires_at": info.expires_at,
            "umrs_base_url": gateway_url.rstrip("/"),
        }
        session_path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
        try:
            os.chmod(session_path, 0o600)
        except OSError:
            pass
    except Exception:
        pass


def _force_refresh_curve_session(connection) -> bool:
    """Force a fresh CURVE session registration, bypassing any cached session.

    Used when a session-mode ZMQ readiness check fails — typically because the
    server restarted and dropped its in-memory session registry, leaving the
    locally-cached session valid by TTL but unknown to the server (the ZAP
    authenticator rejects the handshake, which surfaces as a silent timeout).

    Performs a fresh ``POST /umrs/zmq/session`` (always, ignoring the cached
    file), sets the ZMQ env vars, and persists the new session to disk. Returns
    True on success.
    """
    if not _is_session_mode(connection):
        return False
    gateway_url = getattr(connection, "gateway_url", None)
    jwt_token = _resolve_session_jwt(connection)
    if not gateway_url or not jwt_token:
        return False
    try:
        from umrs.client.session_bootstrap import bootstrap_curve_session

        info = bootstrap_curve_session(
            gateway_url=gateway_url,
            jwt_token=jwt_token,
        )
        _persist_curve_session(info, gateway_url)
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Re-bootstrapped CURVE session %s after stale-session detection",
            info.session_id,
        )
        return True
    except Exception as exc:
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "Forced CURVE session refresh failed (non-fatal): %s",
            exc,
        )
        return False


def _maybe_bootstrap_curve_session(connection) -> None:
    """Bootstrap a CURVE session if the connection uses session-mode ZMQ auth.

    First tries to reuse a saved session from ``~/.config/r4/curve-session.json``
    (written by ``r4 login``).  Falls back to bootstrapping a new session via HTTP.

    This is a no-op when:
    - No gateway_url is configured (direct ZMQ or static CURVE)
    - zmq.auth_type in profile (or UMRS_ZMQ_AUTH_TYPE env) is not 'session'
    - No JWT token is available (env vars or profile .token file)
    """
    if not _is_session_mode(connection):
        return

    # Try to reuse a saved session from r4 login (avoids creating new sessions)
    if _maybe_load_saved_curve_session():
        return

    gateway_url = getattr(connection, "gateway_url", None)
    jwt_token = _resolve_session_jwt(connection)
    if not jwt_token:
        return

    try:
        from umrs.client.session_bootstrap import bootstrap_curve_session

        info = bootstrap_curve_session(
            gateway_url=gateway_url,
            jwt_token=jwt_token,
        )
        # bootstrap_curve_session(set_env=True by default) injects correct
        # UMRS_ZMQ_AUTH_TYPE / UMRS_ZMQ_CURVE_* env vars so MessageQueue picks
        # them up automatically on construction. Persist so later invocations
        # reuse this session instead of creating a new one each time.
        _persist_curve_session(info, gateway_url)
    except Exception as exc:
        # Non-fatal: fall back to whatever auth the MessageQueue already supports
        import logging as _logging

        _logging.getLogger(__name__).debug(
            "CURVE session bootstrap failed (non-fatal): %s",
            exc,
        )
        # logged above at DEBUG level


def main():
    parser = argparse.ArgumentParser(
        description="UMRS client",
        epilog=_CLIENT_EPILOG,
        formatter_class=_ClientHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", help="API action", required=False)

    # Global / legacy flags
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "-c",
        "--conf",
        default=None,
        help="User config JSON path for this invocation (overrides the selected user*.json file)",
    )
    parser.add_argument(
        "--server-conf",
        default=None,
        help="Server config JSON path for this invocation (overrides the selected conf_umrs.json file)",
    )
    parser.add_argument(
        "--s3cfg",
        default=None,
        help="S3 config file path for this invocation (overrides the selected s3cfg file)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Named connection profile to use from the UMRS config directory",
    )
    parser.add_argument(
        "-k", action="append", help="Metadatata key:value", required=False
    )
    parser.add_argument(
        "--no-check-server",
        action="store_true",
        help="Do not check if the server is ready. Just copy data in the S3 server if no ready",
    )
    parser.add_argument(
        "--server-ip", default=None, help="Server IP for ZeroMQ control channel"
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help="Control port for the ZeroMQ control channel",
    )
    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="text",
        help="Output format for API actions",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="Admin token for privileged commands (or set UMRS_ADMIN_TOKEN)",
    )
    # Removed deprecated auto-add-columns behavior; metadata must be declared in server config

    completion_p = subparsers.add_parser("completion", help="Shell completion setup")
    comp_sub = completion_p.add_subparsers(dest="completion_action")
    comp_sub.add_parser("bash", help="Print bash completion script")
    comp_sub.add_parser("zsh", help="Print zsh completion script")
    inst_p = comp_sub.add_parser(
        "install", help="Install completion into your shell RC file"
    )
    inst_p.add_argument("--shell", choices=("bash", "zsh"), default=None)
    uninst_p = comp_sub.add_parser(
        "uninstall", help="Remove completion from your shell RC file"
    )
    uninst_p.add_argument("--shell", choices=("bash", "zsh"), default=None)

    # API subcommands
    ping_p = subparsers.add_parser("ping", help="Check server readiness")
    configure_ping_parser(ping_p)

    subparsers.add_parser(
        "info",
        help="Show connection setup, transport modes, auth, and S3 status at a glance",
    )

    # === Bucket commands ===
    bucket_p = subparsers.add_parser(
        "bucket", help="Bucket management (list, purge, create)"
    )
    bucket_sub = bucket_p.add_subparsers(dest="bucket_action", required=True)

    bl_p = bucket_sub.add_parser("list", help="List bucket aliases for a username")
    bl_p.add_argument(
        "username",
        nargs="?",
        help="Username to filter buckets (defaults to config if omitted)",
    )
    bl_p.add_argument(
        "--no-check-remote",
        action="store_true",
        default=False,
        help="Skip probing MinIO/S3 for remote bucket status (faster)",
    )

    bp_p = bucket_sub.add_parser(
        "purge", help="Delete extra remote S3 buckets not in user config"
    )
    bp_p.add_argument(
        "username", nargs="?", help="Username (defaults to config if omitted)"
    )
    bp_p.add_argument(
        "--bucket",
        action="append",
        dest="buckets",
        metavar="NAME",
        help="Specific bucket name(s) to purge (repeatable; default: all extra)",
    )
    bp_p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    bc_p = bucket_sub.add_parser(
        "create", help="Create a new S3 bucket (if it doesn't exist)"
    )
    bc_p.add_argument(
        "name",
        help="Bucket name to create",
    )

    # === Data transfer commands ===
    data_p = subparsers.add_parser("data", help="Data transfer (upload, download)")
    data_sub = data_p.add_subparsers(dest="data_action", required=True)

    # data upload (replaces upload_file)
    du_p = data_sub.add_parser("upload", help="Upload file to a bucket")
    du_p.add_argument("file_path")
    du_p.add_argument(
        "--blocking",
        action="store_true",
        default=False,
        help="Wait indefinitely for the server acknowledgement after the upload",
    )
    du_p.add_argument(
        "bucket_alias",
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        ),
    )
    du_p.add_argument(
        "username",
        nargs="?",
        help="Configured username owning the bucket (defaults to config if omitted)",
    )

    # data dir-upload (replaces upload_dir)
    ddu_p = data_sub.add_parser(
        "dir-upload", help="Upload all files of a directory to a bucket"
    )
    ddu_p.add_argument("path")
    ddu_p.add_argument(
        "--blocking",
        action="store_true",
        default=False,
        help="Wait indefinitely for the server acknowledgement after the upload",
    )
    ddu_p.add_argument(
        "bucket_alias",
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        ),
    )
    ddu_p.add_argument(
        "username",
        nargs="?",
        help="Configured username owning the bucket (defaults to config if omitted)",
    )

    # data download (replaces download_results)
    dd_p = data_sub.add_parser(
        "download", help="Download file results computed by the server"
    )
    dd_p.add_argument(
        "bucket_alias",
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user.\n"
            "You may append a subdirectory using '/' as in a typical Unix path."
        ),
    )
    dd_p.add_argument(
        "local_path",
        help=(
            "local Unix path where the results will be stored.\n"
            "New files will erase local files."
        ),
    )
    dd_p.add_argument(
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Force overwriting existing local files even if checksums mismatch.",
    )
    wait_group = dd_p.add_mutually_exclusive_group()
    wait_group.add_argument(
        "--wait",
        dest="wait",
        action="store_true",
        default=True,
        help=(
            "Block until PROCESS_OBJECTS work for this bucket is complete "
            "before downloading (default)."
        ),
    )
    wait_group.add_argument(
        "--no-wait",
        dest="wait",
        action="store_false",
        help=(
            "Fail immediately with a warning if the task is still incomplete "
            "or status is unknown."
        ),
    )
    dd_p.add_argument(
        "--timeout",
        type=int,
        default=3600,
        metavar="SECONDS",
        help="Max seconds to wait when --wait is set (default: 3600).",
    )
    dd_p.add_argument(
        "username",
        nargs="?",
        help="Configured username owning the bucket (defaults to config if omitted)",
    )

    # === Project commands (admin-only) ===
    project_p = subparsers.add_parser(
        "project", help="Project management (admin-only: list, add, remove, info)"
    )
    project_sub = project_p.add_subparsers(dest="project_action", required=True)

    pj_p = project_sub.add_parser("list", help="List all projects (admin-only)")

    pa_p = project_sub.add_parser("add", help="Add a user to a project (admin-only)")
    pa_p.add_argument("username", help="Username to add")
    pa_p.add_argument("project_id", help="Project ID")
    pa_p.add_argument("-y", "--yes", action="store_true", help="Assume yes on prompts")

    pr_p = project_sub.add_parser(
        "remove", help="Remove a user from a project (admin-only)"
    )
    pr_p.add_argument("username", help="Username to remove")
    pr_p.add_argument("project_id", help="Project ID")
    pr_p.add_argument("-y", "--yes", action="store_true", help="Assume yes on prompts")

    pi_p = project_sub.add_parser("info", help="Show project details (admin-only)")
    pi_p.add_argument("project_id", help="Project ID")
    pi_p.add_argument(
        "--full",
        action="store_true",
        default=False,
        help="Include full details (members, resources)",
    )

    # === STATUS_ALL (ejeannot batch lane) ===
    subparsers.add_parser(
        "status-all",
        help="Fetch every client's status in one request (admin sessions; others get their own entry)",
    )

    # === Embedding maintenance commands ===
    subparsers.add_parser(
        "recompute", help="Recompute embeddings for your own buckets (admin: all users)"
    )
    subparsers.add_parser(
        "update",
        help="Update embeddings for missing/mismatched entries in your own buckets (admin: all users)",
    )
    subparsers.add_parser(
        "check",
        help="Check consistency between S3 objects and DB (admin-only: checks all users)",
    )
    subparsers.add_parser(
        "reload",
        help="Reload server configuration (users, buckets, models) without restart; if -c points to a user JSON, it will be uploaded and persisted on the server before reloading",
    )

    # === Status command ===
    status_p = subparsers.add_parser(
        "status", help="Fetch the latest server summary for maintenance commands"
    )
    configure_status_parser(status_p, include_username=True)

    # === Doctor command ===
    from umrs.client.doctor import add_doctor_parser as _add_doctor_parser

    _add_doctor_parser(subparsers)

    # === Export embeddings from pgvector (server-mediated) ===
    ee_p = subparsers.add_parser(
        "export_embeddings",
        help="Export computed embeddings to a local file (via server)",
    )
    ee_p.add_argument(
        "bucket_alias",
        help=(
            "Alias of a bucket as defined in the client configuration file.\n"
            "Use the list_bucket command to show available bucket aliases for a given user."
        ),
    )
    ee_p.add_argument(
        "local_path", help="Local file path to write the exported embeddings"
    )
    ee_p.add_argument(
        "--format",
        choices=["csv", "jsonl"],
        default="csv",
        dest="export_format",
        help="Output format (default: csv)",
    )
    ee_p.add_argument(
        "--profile",
        choices=["full", "minimal", "embedding-only"],
        default="minimal",
        dest="export_profile",
        help="Export profile: minimal keeps embedding plus essential metadata; embedding-only keeps only the embedding vector",
    )
    ee_p.add_argument(
        "username",
        nargs="?",
        help="Configured username owning the bucket (defaults to config if omitted)",
    )

    # --- Model registry commands ---
    model_p = subparsers.add_parser(
        "model", help="Model registry commands (register, list, download, upload)"
    )
    model_sub = model_p.add_subparsers(dest="model_action", required=True)

    # model register
    mr_p = model_sub.add_parser("register", help="Register a local model with UMRS")
    configure_model_register_parser(mr_p)

    # model list
    ml_p = model_sub.add_parser("list", help="List registered models")
    configure_model_list_parser(ml_p)

    # model download (default: client-side; --server-side sends to server as background task)
    md_p = model_sub.add_parser(
        "download", help="Download model from HuggingFace and register with UMRS"
    )
    configure_model_download_parser(md_p)
    md_p.add_argument(
        "--staging-dir",
        default=None,
        help="Directory to store downloaded model (default: ~/.config/umrs/models/ or UMRS_MODEL_STAGING_ROOT)",
    )
    md_p.add_argument(
        "--server-side",
        action="store_true",
        default=False,
        help="Download on the server machine instead of locally (requires server internet access)",
    )

    # model delete
    mdel_p = model_sub.add_parser(
        "delete", help="Remove a registered model from the catalog"
    )
    mdel_p.add_argument("name", help="Model name or model ID (mdl-...) to delete")
    mdel_p.add_argument(
        "--scope",
        choices=MODEL_SCOPE_CHOICES,
        default="user",
        help="Scope of the model to delete",
    )
    mdel_p.add_argument(
        "--delete-files",
        action="store_true",
        default=False,
        help="Also remove model files from disk",
    )
    mdel_p.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    # model upload (copy local directory to staging root, then register)
    mu_p = model_sub.add_parser(
        "upload",
        help="Copy a local model directory to the staging root and register with UMRS",
    )
    mu_p.add_argument("path", help="Local directory containing model files")
    mu_p.add_argument("--name", required=True, help="Model name for registration")
    mu_p.add_argument(
        "--type",
        choices=MODEL_TYPE_CHOICES,
        default="embedding",
        dest="model_type",
        help="Model type",
    )
    mu_p.add_argument(
        "--scope", choices=MODEL_SCOPE_CHOICES, default="user", help="Model scope"
    )
    mu_p.add_argument(
        "--project-id",
        default=None,
        help="Project identifier (required for project scope)",
    )
    mu_p.add_argument("--description", default=None, help="Model description")
    mu_p.add_argument(
        "--staging-dir",
        default=None,
        help="Staging root directory (default: UMRS_MODEL_STAGING_ROOT or ~/.config/umrs/models/)",
    )
    mu_p.add_argument(
        "--register-path",
        default=None,
        help="Server-side path to register (use when client and server have different filesystems, e.g. Mac → HPC)",
    )
    mu_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing model directory in staging root",
    )

    # --- Job tracking commands ---
    job_p = subparsers.add_parser(
        "job", help="Job commands (submit, status, list, cancel)"
    )
    job_sub = job_p.add_subparsers(dest="job_action", required=True)

    # job submit
    js_p = job_sub.add_parser("submit", help="Submit a new job")
    configure_job_submit_parser(js_p)

    # job cancel
    jc_p = job_sub.add_parser("cancel", help="Cancel a running job")
    configure_job_cancel_parser(jc_p)

    # job status
    jst_p = job_sub.add_parser("status", help="Show a specific durable job")
    configure_job_status_parser(jst_p)

    # job list
    jl_p = job_sub.add_parser("list", help="List durable jobs with optional filters")
    configure_job_list_parser(jl_p)

    # --- Endpoint control commands ---
    endpoint_p = subparsers.add_parser(
        "endpoint", help="Endpoint lifecycle commands (up, list, status, stop)"
    )
    endpoint_sub = endpoint_p.add_subparsers(dest="endpoint_action", required=True)
    eup_p = endpoint_sub.add_parser("up", help="Provision a reusable endpoint job")
    configure_endpoint_up_parser(eup_p)
    el_p = endpoint_sub.add_parser("list", help="List endpoint provision jobs")
    configure_endpoint_list_parser(el_p)
    est_p = endpoint_sub.add_parser(
        "status", help="Show endpoint provision job details"
    )
    configure_endpoint_status_parser(est_p)
    esp_p = endpoint_sub.add_parser("stop", help="Stop an endpoint provision job")
    configure_endpoint_stop_parser(esp_p)

    # --- Workflow commands (unified) ---
    # All workflow operations are consolidated under the `workflow` subcommand:
    #   workflow upload <path> [--name N] [--scope user|project|system]
    #   workflow list [--scope S] [--user U] [--name N]
    #   workflow run <name> [--bucket B] [--params '{}']
    #   workflow status <execution_id>
    #   workflow results <execution_id>
    #   workflow cancel <execution_id>
    #   workflow errors <execution_id>   -- error logs for failed Attempts
    #   workflow visualize <name>       -- render the workflow DAG
    #   workflow validate-sandbox <path> -- validate Dockerfile sandbox
    wf_p = subparsers.add_parser(
        "workflow",
        help="Workflow commands (upload, list, run, status, results, cancel, errors, visualize, validate-sandbox)",
    )
    wf_sub = wf_p.add_subparsers(dest="workflow_action", required=True)

    # workflow upload
    wu_p = wf_sub.add_parser(
        "upload",
        help="Upload a workflow directory (with workflow.json + step scripts) to the server",
    )
    wu_p.add_argument(
        "path", help="Local directory containing workflow.json and step scripts"
    )
    wu_p.add_argument(
        "--name", default=None, help="Workflow name (defaults to directory basename)"
    )
    wu_p.add_argument(
        "--scope",
        default="user",
        choices=["user", "project", "system"],
        help="Workflow visibility scope (default: user). Used for HTTP gateway uploads.",
    )

    # workflow list
    wl_p = wf_sub.add_parser("list", help="List registered workflows")
    configure_workflow_list_parser(wl_p)

    # workflow run
    wr_p = wf_sub.add_parser(
        "run", help="Trigger execution of a registered workflow on the server"
    )
    configure_workflow_run_parser(wr_p)

    # workflow status
    wst_p = wf_sub.add_parser("status", help="Show execution status for a workflow run")
    wst_p.add_argument("execution_id", help="Execution ID to inspect")

    # workflow results
    wres_p = wf_sub.add_parser(
        "results", help="Fetch results for a completed workflow run"
    )
    wres_p.add_argument("execution_id", help="Execution ID to fetch results for")

    # workflow cancel
    wcan_p = wf_sub.add_parser("cancel", help="Cancel a running workflow execution")
    wcan_p.add_argument("execution_id", help="Execution ID to cancel")

    # workflow errors -- error logs for failed Attempts
    werr_p = wf_sub.add_parser(
        "errors",
        help="Show error logs for a failed workflow execution (Attempts)",
    )
    werr_p.add_argument("execution_id", help="Execution ID to fetch error logs for")
    werr_p.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Maximum number of error log lines to show (default: 50)",
    )

    # workflow visualize -- render the workflow DAG
    wvis_p = wf_sub.add_parser(
        "visualize",
        help="Render the workflow DAG (directed acyclic graph) as ASCII art",
    )
    wvis_p.add_argument("name", help="Registered workflow name to visualize")
    wvis_p.add_argument(
        "--format",
        default="ascii",
        choices=["ascii", "dot"],
        help="Output format: ascii (default) or dot (Graphviz)",
    )

    # workflow validate-sandbox -- validate Dockerfile sandbox
    wvs_p = wf_sub.add_parser(
        "validate-sandbox",
        help="Validate a workflow sandbox (Dockerfile + workflow.json) before running",
    )
    wvs_p.add_argument(
        "path", help="Local directory containing Dockerfile + workflow.json"
    )
    wvs_p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate syntax without building the image",
    )
    wvs_p.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Attempt a real container build (podman/docker) to verify the Dockerfile",
    )

    # --- Monitor (TUI dashboard) ---
    mon_p = subparsers.add_parser(
        "monitor", help="Live-refreshing multi-tab system monitor (TUI)"
    )
    mon_p.add_argument(
        "--interval", type=float, default=3.0, help="Refresh interval in seconds"
    )
    mon_p.add_argument(
        "--status",
        default=None,
        choices=["running", "queued", "planned", "completed", "failed", "cancelled"],
        help="Initial status filter (Jobs tab)",
    )
    mon_p.add_argument(
        "--project", default=None, help="Filter by project ID (Jobs tab)"
    )
    mon_p.add_argument(
        "--once",
        action="store_true",
        default=False,
        help="Print dashboard once and exit (no live refresh)",
    )
    mon_p.add_argument(
        "--tab",
        default="jobs",
        choices=[
            "jobs",
            "endpoints",
            "status",
            "models",
            "workflows",
            "connection",
            "logs",
            "errors",
            "workflow_logs",
        ],
        help="Tab to display on startup (default: jobs)",
    )
    mon_p.add_argument(
        "--workflow-id",
        default=None,
        help="Workflow execution ID to stream logs for (Workflow Logs tab)",
    )

    # --- Disabled admin command ---
    admin_p = subparsers.add_parser(
        "admin",
        help="Disabled; use umrs-server admin on the server",
        description=_disabled_admin_message(),
    )
    admin_p.add_argument("admin_args", nargs="*", help=argparse.SUPPRESS)

    # --- login: unified auth + profile + config setup ---
    login_p = subparsers.add_parser(
        "login",
        help="Authenticate and configure connection in one step",
        description="Login: `umrs login <JWT>` or `umrs login --api-key <JWT>`",
    )
    login_p.add_argument("jwt", nargs="?", help="JWT token (positional or --api-key)")
    login_p.add_argument("--host", default="127.0.0.1", help="UMRS server host")
    login_p.add_argument("--port", type=int, default=5555, help="UMRS ZMQ control port")
    login_p.add_argument(
        "--api-key",
        "-k",
        help="JWT or onboarding token (positional JWT takes precedence)",
        default=None,
    )
    login_p.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "ssh_tunnel", "gateway"],
        help="Connection mode (gateway uses HTTP through the gateway)",
    )
    login_p.add_argument(
        "--gateway-url",
        default=None,
        help="Gateway URL (e.g. https://data0.ai.r-ccs.riken.jp/rccs). Implies --mode gateway",
    )
    login_p.add_argument(
        "--username", default=None, help="UMRS username (auto-extracted from JWT)"
    )
    login_p.add_argument(
        "--profile-name", default=None, help="Connection profile name (default: auto)"
    )

    profile_p = subparsers.add_parser(
        "profile", help="Connection profile commands (init, list, show, use, import)"
    )
    profile_sub = profile_p.add_subparsers(dest="profile_action", required=True)
    pn_p = profile_sub.add_parser(
        "init", help="Create a new connection profile and set it as current"
    )
    pn_p.add_argument("--name", default=None, help="Profile name (prompted if omitted)")
    pn_p.add_argument(
        "--mode",
        default="direct",
        choices=["direct", "ssh_tunnel"],
        help="Connection mode (default: direct)",
    )
    pn_p.add_argument(
        "--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    pn_p.add_argument(
        "--control-port",
        type=int,
        default=5555,
        help="ZMQ control port (default: 5555)",
    )
    pn_p.add_argument(
        "--data-port-start",
        type=int,
        default=5556,
        help="Data port range start (default: 5556)",
    )
    pn_p.add_argument(
        "--data-port-end",
        type=int,
        default=5563,
        help="Data port range end (default: 5563)",
    )
    pn_p.add_argument(
        "--minio-url", default=None, help="MinIO URL (default: http://<host>:9000)"
    )
    pn_p.add_argument(
        "--rag-url", default=None, help="RAG URL (default: http://<host>:8081)"
    )
    pn_p.add_argument("--ssh-host", default=None, help="SSH host for tunnel mode")
    pn_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite if the profile already exists",
    )
    profile_sub.add_parser("list", help="List available connection profiles")
    profile_sub.add_parser(
        "show", help="Show the effective connection resolved from profile/env/flags"
    )
    pu_p = profile_sub.add_parser(
        "use", help="Set the current default connection profile"
    )
    pu_p.add_argument("name", help="Profile name to persist as the current default")
    pi_p = profile_sub.add_parser(
        "import",
        help="Import a connection profile JSON file into the UMRS config directory",
    )
    pi_p.add_argument("file", help="Path to a connection profile JSON file")
    pi_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing profile with the same name",
    )

    config_p = subparsers.add_parser(
        "config", help="Loaded config file commands (init, list, show, use)"
    )
    config_sub = config_p.add_subparsers(dest="config_action", required=True)
    ci_p = config_sub.add_parser(
        "init", help="Create a new user config JSON and set it as current"
    )
    ci_p.add_argument(
        "--username",
        default=None,
        help="Username for the new config (prompted if omitted)",
    )
    ci_p.add_argument(
        "--fullname", default=None, help="Full name of the user (prompted if omitted)"
    )
    ci_p.add_argument(
        "--output-path",
        default=None,
        help="Where to write the JSON (default: ~/.config/umrs/users/<username>.json)",
    )
    ci_p.add_argument(
        "--template",
        default=None,
        help="Path to an existing user JSON to use as template",
    )
    ci_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite if the output file already exists",
    )
    config_sub.add_parser(
        "list", help="List discovered conf_umrs.json, user*.json, and s3cfg candidates"
    )
    config_sub.add_parser(
        "show", help="Show the resolved config stack for server, user, and s3 files"
    )
    cu_p = config_sub.add_parser(
        "use", help="Persist the current server/user/s3 config selections"
    )
    cu_p.add_argument(
        "--server",
        default=None,
        help="Server config file path or unique basename to select",
    )
    cu_p.add_argument(
        "--user",
        default=None,
        help="User config file path or unique basename to select",
    )
    cu_p.add_argument(
        "--s3", default=None, help="S3 config file path or unique basename to select"
    )

    experience_p = subparsers.add_parser(
        "experience", help="Experience config commands (add, remove, recover)"
    )
    experience_sub = experience_p.add_subparsers(
        dest="experience_action", required=True
    )
    ea_p = experience_sub.add_parser(
        "add",
        help="Merge experience fragment(s) from a JSON file into the current user config",
    )
    _configure_experience_add_parser(ea_p)
    er_p = experience_sub.add_parser(
        "recover",
        help="Recover experience JSON from MinIO/S3 and optional Postgres status",
    )
    _configure_experience_recover_parser(er_p)
    erm_p = experience_sub.add_parser(
        "remove", help="Remove experience(s) by xp_name from the current user config"
    )
    _configure_experience_remove_parser(erm_p)
    el_p = experience_sub.add_parser(
        "list", help="List experiences in the current user config"
    )
    _configure_experience_list_parser(el_p)
    va_p = experience_sub.add_parser(
        "validate", help="Validate an experience fragment and verify model availability"
    )
    _configure_experience_validate_parser(va_p)

    if emit_completion_if_requested(parser, "umrs", "_UMRS_CLIENT_COMPLETE"):
        return 0

    args = parser.parse_args()

    if not args.action:
        # Bare `umrs` with no subcommand: show full help including the
        # onboarding/shell-completion hints instead of a terse argparse error.
        parser.print_help()
        return 0

    if args.action == "completion":
        action = getattr(args, "completion_action", None)
        if action == "install":
            print(install_completion("umrs", shell=args.shell))
            return 0
        if action == "uninstall":
            print(uninstall_completion("umrs", shell=args.shell))
            return 0
        if action in ("bash", "zsh"):
            print_completion_script("umrs", "_UMRS_CLIENT_COMPLETE", action)
            return 0
        print_completion_script("umrs", "_UMRS_CLIENT_COMPLETE", "bash")
        return 0

    if args.action == "profile":
        try:
            sys.exit(_handle_profile_command(args))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "login":
        try:
            sys.exit(_handle_login(args))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "config":
        try:
            sys.exit(_handle_config_command(args))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "experience":
        # Resolve full context (connection, ZMQ, user config) before dispatching
        # so that experience subcommands (especially validate) can reach
        # the server for model-catalog checks via ZMQ.
        try:
            connection = resolve_connection(
                explicit_profile_name=args.profile,
                explicit_server_ip=args.server_ip,
                explicit_control_port=args.control_port,
            )
            _maybe_bootstrap_curve_session(connection)

            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip=connection.server_host,
                port=connection.control_port,
                tunnel_mode=connection.tunnel_mode,
                prefer_env_server_host=False,
            )

            # Check server readiness
            if not args.no_check_server and not control_mq.is_server_ready(
                timeout_ms=connection.health_check_timeout_ms
            ):
                # Try CURVE session refresh once
                if _force_refresh_curve_session(connection):
                    control_mq.close()
                    control_mq = MessageQueue(
                        role="client",
                        queue_type="control",
                        server_ip=connection.server_host,
                        port=connection.control_port,
                        tunnel_mode=connection.tunnel_mode,
                        prefer_env_server_host=False,
                    )
                    if not control_mq.is_server_ready(
                        timeout_ms=connection.health_check_timeout_ms
                    ):
                        sys.exit(
                            f"{_red('✗')} Server connection timeout — cannot continue!"
                        )

            # Load user config
            config_state = resolve_loaded_config(
                explicit_server_config_path=args.server_conf,
                explicit_user_config_path=args.conf,
                explicit_s3_config_path=args.s3cfg,
            )
            configure_file = config_state.user.path if config_state.user.path else None
            user = None
            if configure_file and config_state.user.valid:
                user = load_client_cfg_file(configure_file)
                if config_state.s3.valid and config_state.s3.path:
                    user = apply_s3_config_override(user, config_state.s3.path)

            admin_token = args.admin_token or os.environ.get("UMRS_ADMIN_TOKEN")

            # Set client_id on args so subcommands that access it directly can find it
            args._client_id = user.get("username") if user else None

            ctx = _ClientContext(
                args=args,
                control_mq=control_mq,
                connection=connection,
                admin_token=admin_token,
                config_state=config_state,
                configure_file=configure_file,
                user=user,
                buckets_cfg=build_buckets_cfg(user) if user else None,
            )
            sys.exit(_handle_experience_command(ctx))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "doctor":
        from umrs.client.doctor import run_doctor

        # Resolve the actual command path so doctor checks use the same binary
        _cmd = sys.argv[0] if sys.argv[0] else "umrs"

        # Minimal context for doctor checks (no ZMQ required)
        class _DocCtx:
            executable = _cmd
            profile_name = args.profile

        try:
            sys.exit(run_doctor(_DocCtx(), args))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "info":
        try:
            sys.exit(_handle_info(args))
        except Exception as e:
            print(api_json(status="error", message=str(e), output_format=args.output))
            sys.exit(1)

    if args.action == "admin":
        print(
            api_json(
                status="error",
                message=_disabled_admin_message(),
                output_format=args.output,
            )
        )
        sys.exit(2)

    connection = resolve_connection(
        explicit_profile_name=args.profile,
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )

    # --- Gateway-only mode: skip ZMQ for actions that work over HTTP or local config ---
    _gateway_http_actions = {"reload", "list_buckets"}
    gateway_url = getattr(connection, "gateway_url", None)
    _is_model_list = (
        args.action == "model" and getattr(args, "model_action", None) == "list"
    )
    _is_model_register = (
        args.action == "model" and getattr(args, "model_action", None) == "register"
    )
    _is_model_delete = (
        args.action == "model" and getattr(args, "model_action", None) == "delete"
    )
    if gateway_url and (
        args.action in _gateway_http_actions
        or _is_model_list
        or _is_model_register
        or _is_model_delete
    ):
        # Resolve admin token and config state without ZMQ
        admin_token = args.admin_token or os.environ.get("UMRS_ADMIN_TOKEN")
        config_state = resolve_loaded_config(
            explicit_server_config_path=args.server_conf,
            explicit_user_config_path=args.conf,
            explicit_s3_config_path=args.s3cfg,
        )
        configure_file = config_state.user.path if config_state.user.path else None
        user = None
        buckets_cfg = None
        if configure_file and config_state.user.valid:
            try:
                user = load_client_cfg_file(configure_file)
                if config_state.s3.valid and config_state.s3.path:
                    user = apply_s3_config_override(user, config_state.s3.path)
                buckets_cfg = build_buckets_cfg(user)
            except (Exception, SystemExit):
                pass
        ctx = _ClientContext(
            args=args,
            control_mq=None,  # type: ignore[arg-type]
            connection=connection,
            admin_token=admin_token,
            config_state=config_state,
            configure_file=configure_file,
            user=user,
            buckets_cfg=buckets_cfg,
        )
        handler = _ACTION_HANDLERS.get(args.action)
        if handler:
            sys.exit(handler(ctx))
        sys.exit(1)

    # Connect to message queue
    # If the server uses session-mode CURVE auth and we have a gateway URL + JWT,
    # bootstrap the CURVE session first so the ZMQ connection is authenticated.
    _maybe_bootstrap_curve_session(connection)

    control_mq = MessageQueue(
        role="client",
        queue_type="control",
        server_ip=connection.server_host,
        port=connection.control_port,
        tunnel_mode=connection.tunnel_mode,
        prefer_env_server_host=False,
    )

    if (
        not args.no_check_server
        and args.action != "ping"
        and not control_mq.is_server_ready(
            timeout_ms=connection.health_check_timeout_ms
        )
    ):
        # In session-mode CURVE auth a readiness failure is often a stale session:
        # the server restarted and dropped its in-memory session registry, so the
        # locally-cached (TTL-valid) session is rejected by ZAP and the handshake
        # times out silently. Force a fresh session registration and retry once.
        recovered = False
        if _force_refresh_curve_session(connection):
            control_mq.close()
            control_mq = MessageQueue(
                role="client",
                queue_type="control",
                server_ip=connection.server_host,
                port=connection.control_port,
                tunnel_mode=connection.tunnel_mode,
                prefer_env_server_host=False,
            )
            recovered = control_mq.is_server_ready(
                timeout_ms=connection.health_check_timeout_ms
            )
        if not recovered:
            sys.exit(f"{_red('✗')} Server connection timeout — cannot continue!")

    if args.action != "ping":
        profile_name = getattr(connection, "profile_name", None)
        profile_label = _bold(profile_name) if profile_name else _dim("(none)")
        print(
            f"{_green('✓')} Server ready  {_dim('|')}  profile={profile_label}  {_dim('|')}  {connection.server_host}:{connection.control_port}"
        )

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
    status_needs_config = args.action == "status" and not getattr(
        args, "username", None
    )
    workflow_bucket_needs_config = args.action == "run-workflow" and bool(
        getattr(args, "bucket", None)
    )
    requires_config = (
        args.action
        in {
            "list_buckets",
            "purge_remote_buckets",
            "upload_file",
            "upload_dir",
            "download_results",
            "recompute",
            "update",
            "reload",
            "export_embeddings",
        }
        or workflow_bucket_needs_config
        or status_needs_config
    )

    explicit_s3_selected = bool(
        config_state.s3.source and not str(config_state.s3.source).startswith("user:")
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

    if (
        requires_config
        and explicit_s3_selected
        and config_state.s3.path
        and not config_state.s3.valid
    ):
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
            buckets_cfg = build_buckets_cfg(user)
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
        print(
            api_json(
                status="error",
                message=f"Unknown action: {args.action}",
                output_format=args.output,
            )
        )
        control_mq.close()
        sys.exit(1)
    sys.exit(handler(ctx))


if __name__ == "__main__":
    main()
