#!/usr/bin/env python3

import os
import sys
from umrs.shared.sql_helpers import create_db_name, metadata_dict
from umrs.shared.storage.s3 import *
from umrs.core.MessageQueue import MessageQueue
from umrs.client.metadata import check_and_build_metadata
from umrs.client.control_api import UMRSControlAPI
from umrs.client.control_commands import (
    MODEL_SCOPE_CHOICES,
    MODEL_TYPE_CHOICES,
    STATUS_CHOICES,
    configure_admin_token_parser,
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
from umrs.shared.modality import *
from umrs.shared.modality import resolve_modality_task
from umrs.shared.connection_profiles import (
    discover_connection_profiles,
    import_connection_profile,
    resolve_connection,
    set_current_profile_name,
)
from umrs.shared.loaded_config_profiles import (
    apply_s3_config_override,
    discover_loaded_config_profiles,
    get_user_config_search_locations,
    resolve_loaded_config,
    set_current_s3_config,
    set_current_server_config,
    set_current_user_config,
)
from umrs.shared.json_config import load_client_cfg_file
from umrs.shared.pretty_print import pretty_print_user_cfg
from umrs.shared.auth import create_auth_context_from_request
from umrs.shared.config import (
    get_default_docker_s3cfg,
)
from glob import glob
import argparse
from collections import defaultdict
import traceback
import json as _json
import time
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.shared.buckets import BucketsCfg, ClientBucketEntry


def _format_profile_text(connection, selected_profile, errors):
    lines = [
        f"mode={connection.mode}",
        f"server_host={connection.server_host}",
        f"control_port={connection.control_port}",
        f"data_base_port={connection.data_base_port}",
        f"data_port_range={connection.data_port_range_label}",
        f"tunnel_mode={str(connection.tunnel_mode).lower()}",
        f"minio_url={connection.minio_url}",
        f"rag_url={connection.rag_url}",
    ]
    if selected_profile:
        lines.insert(0, f"profile={selected_profile}")
    if connection.profile_path:
        lines.append(f"profile_path={connection.profile_path}")
    if connection.command_hint:
        lines.append(f"command_hint={connection.command_hint}")
    if errors:
        lines.append(f"profile_errors={len(errors)}")
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


def _format_config_text(resolved):
    lines = []
    for key in ("server", "user", "s3"):
        layer = getattr(resolved, key)
        lines.append(f"{key}_config={layer.path or ''}")
        lines.append(f"{key}_source={layer.source or ''}")
        lines.append(f"{key}_exists={str(layer.exists).lower()}")
        lines.append(f"{key}_valid={str(layer.valid).lower()}")
        if layer.error:
            lines.append(f"{key}_error={layer.error}")
        for detail_key, detail_value in sorted(layer.details.items()):
            if isinstance(detail_value, list):
                rendered = ", ".join(str(item) for item in detail_value)
            else:
                rendered = detail_value
            lines.append(f"{key}_{detail_key}={rendered}")
    lines.append(f"warnings={len(resolved.warnings)}")
    for index, warning in enumerate(resolved.warnings, start=1):
        lines.append(f"warning_{index}={warning}")
    return "\n".join(lines)


def _print_config_candidates_text(discovered):
    headings = {
        "server": "Server Configs",
        "user": "User Configs",
        "s3": "S3 Configs",
    }
    for key in ("server", "user", "s3"):
        print(f"[{headings[key]}]")
        candidates = discovered.get(key) or []
        if not candidates:
            print("(none)")
            print("")
            continue
        for candidate in candidates:
            marker = "*" if candidate.current else " "
            state = "ok" if candidate.valid else "invalid"
            print(f"{marker} {state:7s} {candidate.path}")
            if candidate.error:
                print(f"  error={candidate.error}")
        print("")


def _handle_config_command(args):
    if args.config_action == 'list':
        discovered = discover_loaded_config_profiles()
        payload = {
            key: [candidate.to_dict() for candidate in value]
            for key, value in discovered.items()
        }
        if args.output == 'json':
            print(api_json(configs=payload, output_format=args.output))
        else:
            _print_config_candidates_text(discovered)
        return 0

    if args.config_action == 'use':
        if not any((args.server, args.user, args.s3)):
            raise ValueError("Provide at least one of --server, --user, or --s3.")

        updates = {}
        if args.server:
            updates['server'] = str(set_current_server_config(args.server))
        if args.user:
            updates['user'] = str(set_current_user_config(args.user))
        if args.s3:
            updates['s3'] = str(set_current_s3_config(args.s3))

        if args.output == 'json':
            print(api_json(message="Current config selections updated", updates=updates, output_format=args.output))
        else:
            print("Current config selections updated")
            for key, value in updates.items():
                print(f"{key}={value}")
        return 0

    resolved = resolve_loaded_config(
        explicit_server_config_path=args.server_conf,
        explicit_user_config_path=args.conf,
        explicit_s3_config_path=args.s3cfg,
    )
    if args.output == 'json':
        print(api_json(config=resolved.to_dict(), output_format=args.output))
    else:
        print(_format_config_text(resolved))
    return 0


def _handle_admin_command(args):
    """Handle ``admin token {generate,list,revoke}`` — purely local."""
    from umrs.shared.auth import AdminTokenManager

    if args.admin_action != 'token':
        raise ValueError(f"Unknown admin action: {args.admin_action}")

    mgr = AdminTokenManager()

    if args.token_action == 'generate':
        token = mgr.generate_token(args.name, description=args.description)
        if args.output == 'json':
            print(api_json(action='token_generate', name=args.name, token=token, output_format=args.output))
        else:
            print(f"Token generated for '{args.name}':")
            print(f"  {token}")
            print("")
            print("Save this token — it cannot be retrieved later.")
        return 0

    if args.token_action == 'list':
        tokens = mgr.list_tokens()
        if args.output == 'json':
            print(api_json(action='token_list', tokens=tokens, output_format=args.output))
        else:
            if tokens:
                print(f"{'Name':20s} {'Description':30s} {'Created':25s}")
                print("-" * 75)
                for t in tokens:
                    print(f"{t['name']:20s} {t.get('description', ''):30s} {t.get('created_at', ''):25s}")
            else:
                print("No admin tokens configured.")
        return 0

    if args.token_action == 'revoke':
        ok = mgr.revoke_token(args.name)
        if args.output == 'json':
            print(api_json(action='token_revoke', name=args.name, revoked=ok, output_format=args.output))
        else:
            if ok:
                print(f"Token '{args.name}' revoked.")
            else:
                print(f"Token '{args.name}' not found.")
        return 0 if ok else 1

    raise ValueError(f"Unknown token action: {args.token_action}")


def _handle_profile_command(args):
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
            if profiles:
                for profile in profiles:
                    marker = "*" if profile.name == current else " "
                    print(f"{marker} {profile.name:20s} {profile.mode:10s} {profile.path}")
            else:
                print("No connection profiles found.")
            for entry in errors:
                print(f"! {entry['name']}: {entry['error']}")
        return 0

    if args.profile_action == 'use':
        path = set_current_profile_name(args.name)
        if args.output == 'json':
            print(api_json(message=f"Current connection profile set to '{args.name}'", path=str(path), output_format=args.output))
        else:
            print(f"Current connection profile set to '{args.name}'")
            print(f"path={path}")
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
            print(payload["message"])
            print(f"path={payload['path']}")
        return 0

    connection = resolve_connection(explicit_profile_name=args.profile, explicit_server_ip=args.server_ip, explicit_control_port=args.control_port)
    payload = {
        "connection": connection.to_dict(),
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
    merged = {k: v for k, v in base_metadata.items()} if base_metadata else {}
    md = merged.get('Metadata', {})
    md.update(override_meta_dict)
    merged['Metadata'] = md
    return merged



def format_status_text(response_payload, default_key):
    if response_payload.get('status') != 'ok':
        message = response_payload.get('message', 'status unavailable')
        return api_json(status='error', message=message, output_format='text')

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
                header = f"{header} ({time_text})" if header else time_text
            if header:
                lines.append(header)
        summary_body = entry.get('summary')
        if summary_body:
            lines.append(summary_body)
        else:
            lines.append("(no summary)")

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
        return api_json(status='error', message='Unexpected status payload', output_format='text')

    return "\n".join(lines)

def action_ping(control_mq, output_format, timeout_ms=2000):
    ready = control_mq.is_server_ready(timeout_ms=timeout_ms)
    return api_json(server_ready=ready, output_format=output_format)

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
    return api_json(buckets=buckets, output_format=output_format)

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


def main():
    parser = argparse.ArgumentParser(description="UMRS client", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('--output', choices=['json','text'], default='json', help='Output format for API actions')
    parser.add_argument('--admin-token', default=None,
                        help='Admin token for privileged commands (or set UMRS_ADMIN_TOKEN)')
    # Removed deprecated auto-add-columns behavior; metadata must be declared in server config

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
    subparsers.add_parser('recompute', help='Recompute all embeddings for all buckets (server-side)')
    subparsers.add_parser('update', help='Update embeddings only for missing/mismatched entries (server-side)')
    subparsers.add_parser('check', help='Check consistency between S3 objects and DB (server-side)')
    rl_p = subparsers.add_parser('reload', help='Reload server configuration (users, buckets, models) without restart; if -c points to a user JSON, it will be uploaded and persisted on the server before reloading')

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

    # --- Admin token management (local-only, no server connection needed) ---
    admin_p = subparsers.add_parser('admin', help='Admin token management (generate, list, revoke)')
    admin_sub = admin_p.add_subparsers(dest='admin_action', required=True)
    token_p = admin_sub.add_parser('token', help='Manage admin API tokens')
    configure_admin_token_parser(token_p)

    profile_p = subparsers.add_parser('profile', help='Connection profile commands (list, show, use, import)')
    profile_sub = profile_p.add_subparsers(dest='profile_action', required=True)
    profile_sub.add_parser('list', help='List available connection profiles')
    profile_sub.add_parser('show', help='Show the effective connection resolved from profile/env/flags')
    pu_p = profile_sub.add_parser('use', help='Set the current default connection profile')
    pu_p.add_argument('name', help='Profile name to persist as the current default')
    pi_p = profile_sub.add_parser('import', help='Import a connection profile JSON file into the UMRS config directory')
    pi_p.add_argument('file', help='Path to a connection profile JSON file')
    pi_p.add_argument('--force', action='store_true', default=False, help='Overwrite an existing profile with the same name')

    config_p = subparsers.add_parser('config', help='Loaded config file commands (list, show, use)')
    config_sub = config_p.add_subparsers(dest='config_action', required=True)
    config_sub.add_parser('list', help='List discovered conf_umrs.json, user*.json, and s3cfg candidates')
    config_sub.add_parser('show', help='Show the resolved config stack for server, user, and s3 files')
    cu_p = config_sub.add_parser('use', help='Persist the current server/user/s3 config selections')
    cu_p.add_argument('--server', default=None, help='Server config file path or unique basename to select')
    cu_p.add_argument('--user', default=None, help='User config file path or unique basename to select')
    cu_p.add_argument('--s3', default=None, help='S3 config file path or unique basename to select')


    args = parser.parse_args()

    if args.action == 'profile':
        try:
            sys.exit(_handle_profile_command(args))
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
        try:
            sys.exit(_handle_admin_command(args))
        except Exception as e:
            print(api_json(status='error', message=str(e), output_format=args.output))
            sys.exit(1)

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

    if not args.no_check_server:
        if not control_mq.is_server_ready(timeout_ms=connection.health_check_timeout_ms):
            sys.exit("Server connection timeout...\n"
                     "Cannot continue!"
                     )

    print (f"Server ready.\nConnected to zeroMQ message queue'")

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



    try:
        # API mode: pre-initialize a data queue for the provided username in action args (if any)
        # Some actions require username (upload_*), ping does not.
        if args.action == 'ping':
            api = UMRSControlAPI(
                control_mq,
                _resolve_control_client_id(user),
                admin_token=admin_token,
            )
            print(api_json(output_format=args.output, **api.ping(timeout_ms=connection.health_check_timeout_ms)))
            control_mq.close()
            sys.exit(0)
        if args.action == 'model':
            # Model registry commands can use the configured username when available.
            client_id = _resolve_control_client_id(user)
            api = UMRSControlAPI(control_mq, client_id, admin_token=admin_token)
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
                        # Client-side download: runs on the login node, then registers with server
                        staging_root = args.staging_dir or os.environ.get('UMRS_MODEL_STAGING_ROOT')
                        if not staging_root:
                            from umrs.shared.config import get_default_umrs_cfg_dir
                            staging_root = os.path.join(get_default_umrs_cfg_dir(), 'models')
                        target_path = os.path.join(staging_root, target_name)

                        # Step 1: Download locally
                        if os.path.exists(target_path) and os.listdir(target_path):
                            print(f'Model already exists at {target_path}, skipping download.')
                        else:
                            try:
                                from huggingface_hub import snapshot_download
                            except ImportError:
                                print(api_json(status='error', message='huggingface_hub not installed. Run: pip install huggingface_hub', output_format=args.output))
                                control_mq.close()
                                sys.exit(1)
                            print(f'Downloading {args.repo_id} to {target_path} ...')
                            os.makedirs(target_path, exist_ok=True)
                            snapshot_download(
                                repo_id=args.repo_id,
                                local_dir=target_path,
                                token=os.environ.get('HF_TOKEN'),
                            )
                            print(f'Download complete: {target_path}')

                        # Step 2: Register with server via ZMQ
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
                        control_mq.close()
                        sys.exit(1)
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
                            control_mq.close()
                            sys.exit(1)
                    os.makedirs(staging_root, exist_ok=True)
                    print(f'Copying {src_path} → {target_path} ...')
                    shutil.copytree(src_path, target_path)
                    print(f'Copy complete. Registering model ...')
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
                    control_mq.close()
                    sys.exit(1)

                if args.output == 'json':
                    print(_json.dumps(parsed, ensure_ascii=False))
                    if parsed.get('status') != 'ok':
                        exit_code = 1
                else:
                    if parsed.get('status') == 'ok':
                        if model_action == 'list':
                            models = parsed.get('models', [])
                            if not models:
                                print("No models registered.")
                            else:
                                for m in models:
                                    scope_str = m.get('scope', '')
                                    print(f"  {m['name']:30s}  {m.get('model_type',''):10s}  {scope_str:8s}  {m.get('storage_ref','')}")
                        else:
                            print(f"OK: {parsed.get('message', '')}")
                    else:
                        print(f"Error: {parsed.get('message', 'unknown error')}")
                        exit_code = 1
            except Exception as e:
                exit_code = 1
                print(api_json(status='error', action='model', message=str(e), output_format=args.output))
                if args.verbose:
                    traceback.print_exc()
            finally:
                control_mq.close()
            sys.exit(exit_code)
        if args.action == 'job':
            client_id = _resolve_control_client_id(user)
            api = UMRSControlAPI(control_mq, client_id)
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
                    control_mq.close()
                    sys.exit(1)

                if args.output == 'json':
                    print(_json.dumps(parsed, ensure_ascii=False))
                    if parsed.get('status') != 'ok':
                        exit_code = 1
                else:
                    if parsed.get('status') == 'ok':
                        if job_action == 'list':
                            jobs = parsed.get('jobs', [])
                            if not jobs:
                                print('No jobs found.')
                            else:
                                for job in jobs:
                                    print(
                                        f"  {job.get('job_id',''):18s}  "
                                        f"{job.get('status',''):10s}  "
                                        f"{job.get('job_type',''):18s}  "
                                        f"{job.get('project_id','')}"
                                    )
                        elif job_action == 'status':
                            job = parsed.get('job', {})
                            print(f"job_id={job.get('job_id', '')}")
                            print(f"status={job.get('status', '')}")
                            print(f"job_type={job.get('job_type', '')}")
                            print(f"project_id={job.get('project_id', '')}")
                        else:
                            print(f"OK: {parsed.get('message', '')}")
                    else:
                        print(f"Error: {parsed.get('message', 'unknown error')}")
                        exit_code = 1
            except Exception as e:
                exit_code = 1
                print(api_json(status='error', action='job', message=str(e), output_format=args.output))
                if args.verbose:
                    traceback.print_exc()
            finally:
                control_mq.close()
            sys.exit(exit_code)
        if args.action == 'monitor':
            # Multi-tab TUI system monitor — config file optional (enables reload)
            client_id = _resolve_control_client_id(user)
            try:
                from umrs.tui.dashboard import Dashboard
            except ImportError:
                sys.exit("The 'rich' library is required for the monitor dashboard.\n"
                         "Install it with: pip install 'rich>=13.0'")
            try:
                conn_info = connection.to_dict() if connection else {}
            except Exception:
                conn_info = {}
            capabilities = _build_monitor_capabilities(
                client_id,
                admin_token=admin_token,
                user_cfg=user if isinstance(user, dict) else None,
            )
            try:
                dash = Dashboard(
                    control_mq=control_mq,
                    client_id=client_id,
                    interval=args.interval,
                    status_filter=args.status,
                    project_filter=args.project,
                    initial_tab=args.tab,
                    config_path=configure_file if configure_file and config_state.user.valid else None,
                    user_cfg=user if isinstance(user, dict) else None,
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
                control_mq.close()
            sys.exit(0)
        if args.action == 'workflow':
            client_id = _resolve_control_client_id(user)
            api = UMRSControlAPI(control_mq, client_id)
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
                    control_mq.close()
                    sys.exit(1)

                if args.output == 'json':
                    print(_json.dumps(parsed, ensure_ascii=False))
                    if parsed.get('status') != 'ok':
                        exit_code = 1
                else:
                    if parsed.get('status') == 'ok':
                        if wf_action == 'run':
                            print(f"OK: {parsed.get('message', '')}")
                        else:
                            workflows = parsed.get('workflows', [])
                            if not workflows:
                                print("No workflows registered.")
                            else:
                                for w in workflows:
                                    n_steps = len(w.get('steps', []))
                                    print(f"  {w['name']:30s}  v{w.get('version','?'):8s}  {w.get('scope',''):6s}  {w.get('user_id',''):12s}  {n_steps} step(s)  {w.get('description','')}")
                    else:
                        print(f"Error: {parsed.get('message', 'unknown error')}")
                        exit_code = 1
            except Exception as e:
                exit_code = 1
                print(api_json(status='error', action='workflow', message=str(e), output_format=args.output))
                if args.verbose:
                    traceback.print_exc()
            finally:
                control_mq.close()
            sys.exit(exit_code)
        if args.action == 'upload-workflow':
            import tempfile
            import zipfile
            exit_code = 0
            client_id = _resolve_control_client_id(user)
            src_path = os.path.abspath(args.path)
            if not os.path.isdir(src_path):
                print(api_json(status='error', message=f'Source path is not a directory: {src_path}', output_format=args.output))
                control_mq.close()
                sys.exit(1)
            wf_json = os.path.join(src_path, 'workflow.json')
            if not os.path.isfile(wf_json):
                print(api_json(status='error', message=f'No workflow.json found in {src_path}', output_format=args.output))
                control_mq.close()
                sys.exit(1)
            workflow_name = args.name or os.path.basename(src_path)
            # Create a zip archive of the directory
            try:
                tmp_zip = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
                tmp_zip_path = tmp_zip.name
                tmp_zip.close()
                with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, dirs, files in os.walk(src_path):
                        for f in files:
                            abs_path = os.path.join(root, f)
                            arc_name = os.path.relpath(abs_path, src_path)
                            zf.write(abs_path, arc_name)
                # Establish data channel, notify the server to begin receiving,
                # then stream the zip over the data socket.
                data_mq = control_mq.get_data_queue(client_id, admin_token=admin_token)
                try:
                    control_mq.set_timeouts(120_000, 120_000)
                    control_mq.send_string(f"{TaskCmd.UPLOAD_WORKFLOW}:{client_id}:{workflow_name}")
                    data_mq.send_file(tmp_zip_path)
                    _, reply = control_mq.recv_string()
                    if reply != "ACKNOWLEDGE":
                        raise RuntimeError(f"Unexpected reply from server during workflow upload: {reply}")
                    print(api_json(action='upload-workflow', workflow_name=workflow_name, accepted=True, output_format=args.output))
                finally:
                    control_mq.set_timeouts(control_mq.timeout, control_mq.timeout)
                    try:
                        data_mq.close()
                    except Exception:
                        pass
            except Exception as e:
                exit_code = 1
                print(api_json(status='error', action='upload-workflow', message=str(e), output_format=args.output))
                if args.verbose:
                    traceback.print_exc()
            finally:
                try:
                    os.remove(tmp_zip_path)
                except Exception:
                    pass
                control_mq.close()
            sys.exit(exit_code)
        if args.action == 'run-workflow':
            client_id = _resolve_control_client_id(user)
            api = UMRSControlAPI(control_mq, client_id)
            workflow_name = args.workflow_name
            exit_code = 0
            try:
                params = parse_json_object_arg(args.params, field_name='params') if args.params else {}
                parsed = api.run_workflow(workflow_name=workflow_name, params=params, bucket=args.bucket)
                if args.output == 'json':
                    print(_json.dumps(parsed, ensure_ascii=False))
                    if parsed.get('status') != 'ok':
                        exit_code = 1
                else:
                    if parsed.get('status') == 'ok':
                        print(f"OK: {parsed.get('message', '')}")
                    else:
                        print(f"Error: {parsed.get('message', 'unknown error')}")
                        exit_code = 1
            except Exception as e:
                exit_code = 1
                print(api_json(status='error', action='run-workflow', message=str(e), output_format=args.output))
                if args.verbose:
                    traceback.print_exc()
            finally:
                control_mq.close()
            sys.exit(exit_code)
        if args.action == 'list_buckets':
            effective_username = args.username or (user.get('username') if isinstance(user, dict) else None)
            if not effective_username:
                print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
                control_mq.close()
                sys.exit(1)
            print(action_list_buckets(buckets_cfg, effective_username, args.output))
            control_mq.close()
            sys.exit(0)
        if args.action == 'status':
            effective_username = args.username
            if not effective_username and isinstance(user, dict):
                effective_username = user.get('username')
            if not effective_username:
                print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
                control_mq.close()
                sys.exit(1)
            api = UMRSControlAPI(control_mq, effective_username)
            try:
                parsed = api.request_status(args.target)
                if args.output == 'json':
                    print(_json.dumps(parsed, ensure_ascii=False))
                else:
                    print(format_status_text(parsed, args.target))
            except Exception as e:
                print(api_json(status='error', action=args.action, message=str(e), output_format=args.output))
            finally:
                control_mq.close()
            sys.exit(0)
        # Embedding maintenance actions don't require bucket resolution or S3 access
        if args.action in ('recompute', 'update', 'check', 'reload'):
            # Use configured username as client id for control plane
            # Determine a stable client id for control-plane requests
            client_id = (user.get('username') if isinstance(user, dict) else None) or os.environ.get('UMRS_CLIENT_ID', '_')
            api = UMRSControlAPI(
                control_mq,
                client_id,
                configure_file if configure_file and config_state.user.valid else None,
                admin_token=admin_token,
            )
            try:
                parsed = api.maintenance(args.action)
                if parsed.get('status') == 'ok':
                    print(api_json(action=args.action, accepted=True, output_format=args.output))
                else:
                    print(api_json(status='error', action=args.action, accepted=False, message=parsed.get('message', 'rejected'), output_format=args.output))
            except Exception as e:
                print(api_json(status='error', action=args.action, message=str(e), output_format=args.output))
            finally:
                control_mq.close()
            sys.exit(0)
        # Export embeddings via server (Client → Server → Database)
        if args.action == 'export_embeddings':
            try:
                bucket_alias_arg = getattr(args, 'bucket_alias')
                bucket_alias, _ = decompose_bucket_alias_arg(bucket_alias_arg)
                entry = buckets_cfg.get(bucket_alias)
                if entry is None:
                    print(api_json(status='error', message=f'Bucket alias not found: {bucket_alias}', output_format=args.output))
                    control_mq.close()
                    sys.exit(1)
                # The server keys buckets by path_to_bucket
                path_to_bucket = entry.path_to_bucket
                effective_username = getattr(args, 'username', None) or (
                    user.get('username') if isinstance(user, dict) else None
                )
                if not effective_username:
                    print(api_json(status='error', message='No username provided and none found in config', output_format=args.output))
                    control_mq.close()
                    sys.exit(1)
                # Ask the server to fetch embeddings from the DB
                raw_response = control_mq.request_export_embeddings(
                    effective_username, path_to_bucket, args.export_format, args.export_profile,
                )
                response = _json.loads(raw_response)
                if response.get('status') != 'ok':
                    print(api_json(status='error', action='export_embeddings',
                                   message=response.get('message', 'Server returned an error'),
                                   output_format=args.output))
                    control_mq.close()
                    sys.exit(1)
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
                            for col, val in zip(columns, row):
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
                control_mq.close()
            sys.exit(0)
        # For upload_* actions we need configuration + bucket alias
        bucket_alias_arg = getattr(args, 'bucket_alias')
        bucket_alias, subdir = decompose_bucket_alias_arg(bucket_alias_arg)

        selected_fields = ["username", "xp_name", "modality_name", "s3_configuration_file", "metadata"]
        if args.action == "download_results":
            selected_fields.append("output_path_to_bucket")
        else :
            selected_fields.append("path_to_bucket")
        _bucket_result = buckets_cfg.get(bucket_alias, filter=selected_fields)
        if _bucket_result is None:
            raise ValueError(
                f"Bucket alias '{bucket_alias}' not found in configuration. "
                f"Check your config file or use one of the configured aliases."
            )
        username, xp_name, modality, s3_configuration_file, metadata, path_to_bucket = _bucket_result

        # If username is provided as arg change it otherwise stay default
        username_arg = getattr(args, 'username', None)
        username = username_arg or username

        data_mq = control_mq.get_data_queue(username, admin_token=admin_token)
    except Exception as e:
        print(api_json(status="error", message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()
        control_mq.close()
        sys.exit(1)

    # Allow fallback to default docker s3cfg if user-specific config is missing inside container
    expanded_cfg = os.path.expanduser(s3_configuration_file)
    default_docker_s3cfg = get_default_docker_s3cfg()
    if not os.path.exists(expanded_cfg) and os.path.exists(default_docker_s3cfg):
        expanded_cfg = default_docker_s3cfg

    s3, bucket_location = connect_to_s3(expanded_cfg)

    if args.action == "download_results":
        full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, output=True, indent=False)
        if full_bucket_name != path_to_bucket:
                sys.exit(f"Error:\tNo output bucket for {bucket_alias} for  modality {modality} and for user {username}\n"
                         f"\tCheck that this modality has 'need_output_bucket' set to 'true'")
        if not success:
            # Output bucket may not exist yet if the workflow/processing hasn't started.
            # Proceed anyway — download_results will poll and wait for server completion.
            if args.verbose:
                print(f"Note: Output bucket '{full_bucket_name}' does not exist yet on S3 — will wait for server.")
    else :
        full_bucket_name, success = check_bucket_name(s3, xp_name, bucket_alias, username, indent=False)
        if not success or full_bucket_name != path_to_bucket:
            sys.exit(f"Error:\tNo bucket {bucket_alias} for modality {modality} and for user {username}")

    if args.verbose :
        print(f"Bucket '{bucket_alias}' for modality '{modality}' and for user '{username}': OK!")

    try:
        metadata_args = check_and_build_metadata(metadata, args.k)
    except Exception as e:
        print(api_json(status="error", message=f"Metadata error: {e}", output_format=args.output))
        control_mq.close(); data_mq.close(); sys.exit(1)

    # In API mode, metadata is provided via repeated -k flags like legacy mode

    if args.verbose :
       action = "Retreiving" if args.action == "download_results" else "Storing"
       print(f"{action} data for:\n\tUser: '{username}'\n\tXP: '{xp_name}'\n\tModality: '{modality}'")

    server_notify_result = None


    def process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=True):
        if not os.path.isfile(file_path):
            print(f"Error: file {file_path} is not a valid.")
            return None, 0

        category = modality_category(modality)
        print(f"Processing file '{file_path}' for modality '{modality}' (category: {category})")

        if category == "image":
            s3_key = s3_store_image(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args, check=(modality not in ("volume", "movie", "volume_movie")))
            if s3_key and notify_server:
                control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
        elif category == "text_file":
            s3_key = s3_store_text_file(s3, full_bucket_name, subdir, file_path, data_mq, metadata_args)
            if s3_key and notify_server:
                control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, 1)
            return s3_key, 1
        elif category == "text":  # consider each line of the file as a string to be processed
            try:
                with open(file_path,"r") as file:
                    nb_objects = 0
                    s3_key_list = []
                    for line in file.readlines():
                        line = line.strip()
                        if line != "" : # skip empty lines as it brings nothing and as it slows down the process
                            s3_key = s3_store_txt(s3, full_bucket_name, subdir, line, data_mq, metadata_args)
                            s3_key_list.append(s3_key)
                            nb_objects += 1
                    if notify_server and nb_objects > 0:
                        control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
                    return s3_key_list, nb_objects
            except UnicodeDecodeError as e:
                print(f"Unicode decoding error: {e}")
                print(f"{args.data} is not a text file. Cannot use modality: '{modality}' for this file.")
        else:
            print(f"Error: Unknown modality category '{category}' (modality: '{modality}'). Nothing has been sent to the server")

        return None, 0

    def process_dir(path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username, control_mq):
        if not os.path.isdir(path):
            print(f"Error: {path} is not a valid directory.")
            return [], []

        s3_key_list = []
        file_list = []
        nb_objects= 0
        for file_path in glob(os.path.join(path, "*")):
            if os.path.isfile(file_path):
                s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, file_path, modality, data_mq, metadata_args, username, notify_server=False)
                if s3_key:
                    if isinstance(s3_key, list): # files of text are processed line by line (each leading to a file stored on the s3 server)
                        s3_key_list.extend(s3_key)  # add each element individually if s3_key is a list
                        assert(len(s3_key)==nb_sent)
                        file_list.extend([file_path] * nb_sent) #ensure that file_list is of same size as s3_key_list
                    else:
                        s3_key_list.append(s3_key)
                        file_list.append(file_path)
                    nb_objects += nb_sent

        if nb_objects>0:
            control_mq.notify_server(username, TaskCmd.PROCESS_OBJECTS, nb_objects)
        return s3_key_list, file_list

    def download_results(args, full_bucket_name, modality, subdir):
        effective_username = args.username
        verbose = args.verbose
        if not effective_username and isinstance(user, dict):
            effective_username = user.get('username')
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


        # check if the target path exists
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
        while not done: # loop until we are sure the server has done with this bucket
            # Check whether the server is still computing for this bucket before
            # downloading results. Some workflows create temporary artifacts that
            # are deleted before completion; downloading too early can capture
            # those transient files locally.
            raw_response = control_mq.request_status(effective_username, TaskCmd.PROCESS_OBJECTS)
            if raw_response is None:
                break;
            # raw response is a string of a serialized json. transform it into json dict
            parsed = _json.loads(raw_response)

            # if there is no entry, we are done
            message = parsed.get("message")
            if  message == "No status entries":
                done=True
            else : #if there are entries, check the one that correspond to this bucket
                # first extract data
                # from pprint import pprint

                data = parsed["summary"]["data"]
                # pprint(data, width=100)
                # Check whether all tasks related to this bucket are finished.
                # If there exists at least one task whose status is not "Done"
                # and whose work bucket matches the requested bucket prefix
                # and are of processing S3 objectstype ,
                # then the computation is still running.
                done = not any(
                    entry["status"] != "Done"
                    and any(
                        full_bucket_name.startswith(w["bucket"]) # and w["work_type"] == "process_objects"
                        for w in entry["work"]
                    )
                    for entry in data
                )
            if not done :
                if verbose:
                    elapsed = int(time.time() - _wait_start)
                    print(f"Waiting for update... (elapsed: {elapsed // 60}m {elapsed % 60}s)")
                res = control_mq.wait_for_new_data(effective_username, full_bucket_name, timeout=timeout)
                if res == "TIMEOUT":
                    timeout = min(timeout*2, 64)
                elif res == "UPDATE":
                    timeout= 1
                else :
                    if verbose:
                        print(f"Error: receiving '{res}' from server while waiting to be notified for new data")
        # print("Done. Final check...")
        # Do it one more time in case the server finished after the last check
        # get the list of objects on s3 server
        objects_list, _, checksums = list_s3_bucket(s3, full_bucket_name, prefix=subdir)
        #down load these object talink into account if we force overwrite or not
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

    # Determine action specifics
    action = args.action
    try:
        if action == "upload_file":
            s3_key, nb_sent = process_file(s3, full_bucket_name, subdir, args.file_path, modality, data_mq, metadata_args, username, notify_server=True)
            if s3_key:
                print(api_json(s3_key=s3_key, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message=f"Cannot process file: {args.file_path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))

        elif action == "upload_dir":
            s3_key_list, file_list =  process_dir(args.path, s3, full_bucket_name, subdir, modality, data_mq, metadata_args, username, control_mq)
            if len(s3_key_list)>0:
                print(api_json(s3_key_list=s3_key_list, file_list=file_list, bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
            else:
                print(api_json(status='error', message=f"Cannot process directory: {args.path}", bucket=full_bucket_name, subdir=subdir, modality=modality, output_format=args.output))
        elif action == "download_results":
            ok, response = download_results(args, full_bucket_name, modality, subdir)
            print(api_json(**response))


    except Exception as e:
        print(api_json(status='error', message=str(e), output_format=args.output))
        if args.verbose:
            traceback.print_exc()


    # close queues before exiting
    control_mq.close()
    data_mq.close()


if __name__ == "__main__":
    main()
