"""Server maintenance operations: recompute, update, check, reload dispatch."""

import logging
import time

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.provisioning import ServerCfg, configure_user
from umrs.server.sql import check_S3_object_status_in_db
from umrs.shared.pretty_print import (
    pretty_print_cfg,
    pretty_print_check,
    pretty_print_recompute,
)
from umrs.shared.sql_helpers import create_db_name
from umrs.shared.storage.s3 import list_s3_bucket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maintenance commands
# ---------------------------------------------------------------------------

def recompute_all_objects(
    active_threads,
    username,
    buckets_cfg,
    device,
    embedding_backend,
    key,
    log_output=True,
    update_status_callback=None,
):
    if log_output:
        print(
            "\n######################\n"
            "Recomputing all objects in all buckets\n"
            "######################\n"
        )

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    recompute_dict = {}

    for path_to_bucket, entry in buckets_cfg.items():
        with entry.reentrant_lock:
            # skip none matching username
            if username is not None and username != entry.s3_username:
                continue
            s3_objects, total_size, _ = list_s3_bucket(
                entry.s3_connection, path_to_bucket
            )
            if len(s3_objects) > 0:
                from umrs.server.processing import process_s3_objects

                start_time = time.perf_counter()
                process_s3_objects(
                    active_threads,
                    s3_objects,
                    buckets_cfg,
                    device,
                    embedding_backend,
                    key,
                    update_when_done=False,
                    force_recompute=True,
                )
                duration = round((time.perf_counter() - start_time), 3)
            else:
                duration = 0

            recompute_dict[path_to_bucket] = {
                "bucket_alias": entry.bucket_alias,
                "duration": duration,
                "nb_objects": len(s3_objects),
                "modality_name": entry.modality_name,
                "s3_username": entry.s3_username,
                "total_size_MB": round(total_size / (1024**2), 3),
                "model": entry.model,
            }

    summary = pretty_print_recompute(recompute_dict, print_output=False)
    if log_output:
        print(summary)

    if active_threads.is_worker_thread():
        active_threads.update_status_to_done()
        users = {entry.s3_username for entry in buckets_cfg.values()}
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome = {"summary": summary, "data": recompute_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key)
    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


def update_objects(
    active_threads,
    username,
    buckets_cfg,
    device,
    embedding_backend,
    key,
    verbose=False,
    log_output=True,
    update_status_callback=None,
):
    if log_output:
        print(
            "\n######################\n"
            "Updating missing/mismatched objects in all buckets\n"
            "######################\n"
        )

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    update_dict = {}

    for path_to_bucket, entry in buckets_cfg.items():
        with entry.reentrant_lock:
            # skip none matching username
            if username is not None and username != entry.s3_username:
                continue
            s3_objects, total_size, checksums = list_s3_bucket(
                entry.s3_connection, path_to_bucket
            )
            status = check_S3_object_status_in_db(
                path_to_bucket, s3_objects, checksums, buckets_cfg
            )

            to_update_objects = []
            for i, s3_object in enumerate(s3_objects):
                s = status[i]["status"]
                if verbose:
                    print(f"{s3_object}: {s}")
                if s != "match":
                    to_update_objects.append(s3_object)

            if len(to_update_objects) > 0:
                from umrs.server.processing import process_s3_objects

                start_time = time.perf_counter()
                process_s3_objects(
                    active_threads,
                    to_update_objects,
                    buckets_cfg,
                    device,
                    embedding_backend,
                    key,
                    update_when_done=False,
                )
                duration = round(time.perf_counter() - start_time, 3)
            else:
                duration = 0.0

            update_dict[path_to_bucket] = {
                "bucket_alias": entry.bucket_alias,
                "duration": duration,
                "nb_objects": f"{len(s3_objects) - len(to_update_objects)}/{len(s3_objects)}",
                "modality_name": entry.modality_name,
                "s3_username": entry.s3_username,
                "total_size_MB": round(total_size / (1024**2), 3),
                "model": entry.model,
            }

    summary = pretty_print_recompute(update_dict, update=True, print_output=False)
    if log_output:
        print(summary)

    if active_threads.is_worker_thread():
        active_threads.update_status_to_done()
        users = {entry.s3_username for entry in buckets_cfg.values()}
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome = {"summary": summary, "data": update_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key)

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


def check_objects(
    active_threads,
    buckets_cfg,
    key,
    verbose=False,
    log_output=True,
    update_status_callback=None,
):
    if log_output:
        print(
            "\n######################\n"
            "Checking all objects in all buckets\n"
            "######################\n"
        )

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    perform_update = False
    check_dict = {}
    for path_to_bucket, entry in buckets_cfg.items():
        nb_not_found, nb_match, nb_mismatch, nb_error = 0, 0, 0, 0
        s3_objects, total_size, checksums = list_s3_bucket(
            entry.s3_connection, path_to_bucket
        )
        status = check_S3_object_status_in_db(
            path_to_bucket, s3_objects, checksums, buckets_cfg
        )
        for i, s3_object in enumerate(s3_objects):
            s = status[i]["status"]
            if verbose:
                print(f"{s3_object}: {s}")
            if s == "match":
                nb_match += 1
            elif s == "mismatch":
                nb_mismatch += 1
            elif s == "not_found":
                nb_not_found += 1
            else:
                print(f"Error: {status[i]['message']}")
                nb_error += 1

        if len(s3_objects) != nb_match:
            perform_update = True

        check_dict[path_to_bucket] = {
            "bucket_alias": entry.bucket_alias,
            "nb_objects": len(s3_objects),
            "modality_name": entry.modality_name,
            "s3_username": entry.s3_username,
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry.model,
            "matches": nb_match,
            "Mismatches": nb_mismatch,
            "Not_found": nb_not_found,
            "Errors": nb_error,
        }

    if active_threads.is_worker_thread():
        active_threads.update_status_to_done()
        active_threads.remove_from_running_list()

    summary = pretty_print_check(check_dict, print_output=False)
    if log_output:
        print(summary)
    if perform_update:
        note = (
            "\n⚠️ Warning: The database is not fully consistent with the "
            "objects stored in S3.\n→ To recompute and update the missing or "
            "mismatched entries, restart the server with the `--force-update` option."
        )
    else:
        note = (
            "\n ✅ No problem: The database is fully consistent with the "
            "objects stored in S3."
        )
    if log_output:
        print(note)
    summary_with_note = summary + note
    outcome = {
        "summary": summary_with_note,
        "data": check_dict,
        "requires_update": perform_update,
    }
    filter_outcome(active_threads, outcome, buckets_cfg, key)

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Outcome filtering (per-user scoping for STATUS replies)
# ---------------------------------------------------------------------------

def filter_outcome(active_threads, outcome, buckets_cfg, key):
    """For each user in the configuration, store its relevant data
    to be later visible via STATUS."""
    users = {entry.s3_username for entry in buckets_cfg.values()}
    for user in users:
        filtered_outcome = _filter_outcome_for_client(key, outcome, user, buckets_cfg)
        active_threads.record_status(user, key, filtered_outcome)


def _filter_dict_by_user(data_dict, client_id):
    """Return a shallow copy of *data_dict* entries belonging to *client_id*."""
    if not isinstance(data_dict, dict):
        return {}
    return {
        k: v
        for k, v in data_dict.items()
        if isinstance(v, dict) and v.get("s3_username") == client_id
    }


def _filter_outcome_for_client(cmd, outcome, client_id, buckets_cfg):
    """Filter an operation outcome so it only contains information for *client_id*.

    For recompute/update/check: filter the 'data' dict by s3_username then
    rebuild the summary.
    For reload: build a per-user configuration summary from the current
    buckets_cfg.
    """
    # Normalize to uppercase for comparison with TaskCmd constants
    # (callers may pass the lowercase status-store key).
    cmd_upper = cmd.upper() if isinstance(cmd, str) else cmd
    try:
        if cmd_upper in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE):
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            summary = pretty_print_recompute(
                filtered, update=(cmd_upper == TaskCmd.UPDATE), print_output=False
            )
            return {"summary": summary, "data": filtered}

        if cmd_upper == TaskCmd.CHECK:
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            summary = pretty_print_check(filtered, print_output=False)
            # Recompute requires_update flag scoped to this client's subset
            requires_update = False
            for entry in filtered.values():
                mism = int(entry.get("Mismatches", 0))
                not_found = int(entry.get("Not_found", 0))
                errs = int(entry.get("Errors", 0))
                matches = int(entry.get("matches", 0))
                nb_objects = int(entry.get("nb_objects", 0))
                if mism > 0 or not_found > 0 or errs > 0 or matches != nb_objects:
                    requires_update = True
                    break
            return {
                "summary": summary,
                "data": filtered,
                "requires_update": requires_update,
            }

        if cmd_upper == TaskCmd.RELOAD:
            # Use the summary already produced by reload_configuration().
            # (pretty_print_cfg expects a ServerCfg, not a plain dict, so we
            # cannot easily rebuild a per-user view here.)
            summary = outcome.get("summary", "") if isinstance(outcome, dict) else str(outcome)
            result = {"summary": summary}
            # Attach s3cfg credentials for this user (if available)
            s3cfg_map = outcome.get("s3cfg_by_user") if isinstance(outcome, dict) else None
            if s3cfg_map and client_id in s3cfg_map:
                result["s3cfg_content"] = s3cfg_map[client_id]
            return result
    except Exception as e:
        # If filtering fails for any reason, fall back to original outcome
        print(
            f"Warning: failed to filter outcome for client '{client_id}': {e}"
        )
    return outcome


# ---------------------------------------------------------------------------
# Configuration reload and validation
# ---------------------------------------------------------------------------

def check_duplicated_database(users):
    """Abort if two users share the same database name."""
    import sys

    databases_name_dict = {}
    for user in users:
        username = user["username"]
        for xp in user["experiences"]:
            sql_username = xp["sql_username"]
            xp_name = xp["xp_name"]
            db_name = create_db_name(sql_username, xp_name)
            if db_name in databases_name_dict:
                sys.exit(
                    f"\t❌ Error: duplicated database '{db_name}' for users "
                    f"'{username}' and '{databases_name_dict[db_name]}'\n"
                    f"\tChange xp_name or sql_username in json file.\n"
                    f"\tExiting..."
                )
            databases_name_dict[db_name] = username


def reload_configuration(
    configure_file, buckets_cfg, user_config_map, log_output=True
):
    """Hot-reload server configuration from disk."""
    from umrs.server.json_config import load_server_cfg_file
    from umrs.server.provisioning import create_s3_sql_users

    if log_output:
        print(
            "\n######################\n"
            "Reloading configuration...\n"
            "######################\n"
        )
    try:
        new_admin_s3_cfg, new_users, _ = load_server_cfg_file(configure_file)
        check_duplicated_database(new_users)
        for entry in buckets_cfg.values():
            try:
                conn = entry.s3_connection
                if conn:
                    conn.close()
            except Exception as e:
                print(
                    f"Warning: failed to close S3 connection during reload: {e}"
                )

        s3cfg_by_user = {}
        for u in new_users:
            s3cfg_content = create_s3_sql_users(u, new_admin_s3_cfg)
            if s3cfg_content:
                s3cfg_by_user[u["username"]] = s3cfg_content

        new_server_cfg = ServerCfg()
        for u in new_users:
            configure_user(u, new_server_cfg)

        buckets_cfg.replace_cfg(new_server_cfg.buckets_cfg)
        user_config_map.clear()
        user_config_map.update(new_server_cfg.user_config_map)
        summary = pretty_print_cfg(new_server_cfg, print_output=False)
        if log_output:
            print(summary)
            print("\n✅ Reload complete.")
        result = {"summary": summary}
        if s3cfg_by_user:
            result["s3cfg_by_user"] = s3cfg_by_user
        return result
    except Exception as e:
        error_msg = f"\n❌ Reload failed: {e}"
        if log_output:
            print(error_msg)
        return {"summary": error_msg, "error": str(e)}


# ---------------------------------------------------------------------------
# Maintenance command dispatcher
# ---------------------------------------------------------------------------

def handle_maintenance_command(
    active_threads,
    cmd,
    client_id,
    buckets_cfg,
    device,
    embedding_backend,
    configure_file,
    user_config_map,
    key,
    update_status_callback=None,
):
    """Dispatch a maintenance command (RECOMPUTE / UPDATE / CHECK / RELOAD).

    Returns the thread ident for async commands, ``None`` for sync, or
    ``False`` when RELOAD encounters an error.
    """
    from umrs.server.config import ENABLE_PRETTY_PRINT

    if cmd == TaskCmd.RECOMPUTE:
        return active_threads.launch_recompute_all_objects(
            client_id,
            buckets_cfg,
            device,
            embedding_backend,
            key,
            log_output=ENABLE_PRETTY_PRINT,
            update_status_callback=update_status_callback,
        )

    elif cmd == TaskCmd.UPDATE:
        return active_threads.launch_update_objects(
            client_id,
            buckets_cfg,
            device,
            embedding_backend,
            key,
            verbose=False,
            log_output=ENABLE_PRETTY_PRINT,
            update_status_callback=update_status_callback,
        )
    elif cmd == TaskCmd.CHECK:
        return active_threads.launch_check_objects(
            buckets_cfg,
            key,
            verbose=False,
            log_output=ENABLE_PRETTY_PRINT,
            update_status_callback=update_status_callback,
        )
    elif cmd == TaskCmd.RELOAD:
        outcome = reload_configuration(
            configure_file,
            buckets_cfg,
            user_config_map,
            log_output=ENABLE_PRETTY_PRINT,
        )
        filter_outcome(active_threads, outcome, buckets_cfg, key)
        if "error" in outcome:
            return False  # Signal error to caller
        return None
    else:
        print(f"====> Error: unknown command '{cmd}' returned by maintenance handler")
        return None


# ---------------------------------------------------------------------------
# Bucket purge (server-side, uses admin S3 credentials)
# ---------------------------------------------------------------------------

def handle_purge_buckets(
    cmd: str,
    client_id: str,
    payload_str: str,
    configure_file: str = "",
) -> str:
    """Delete user-owned extra S3 buckets via admin credentials.

    Security invariants:
      - Only buckets whose name starts with the normalised username prefix
        (``<username>.``) can be deleted.
      - Configured buckets (those still referenced in the user's experience
        config) are excluded from deletion.
      - Admin credentials are used server-side; the user never sees them.

    Payload JSON:
        {"buckets": ["bucket-name-1", "bucket-name-2"]}

    Returns JSON: {"status": "ok"|"partial"|"error", "deleted": [...], "errors": [...]}
    """
    import json
    import os

    try:
        payload = json.loads(payload_str) if payload_str else {}
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Invalid payload: {exc}"})

    requested_buckets: list[str] = payload.get("buckets", [])
    if not requested_buckets:
        return json.dumps({"status": "error", "message": "No buckets specified"})

    # Normalise username for prefix matching (same logic as client-side)
    username_prefix = client_id.lower().replace("_", "-") + "."

    # Validate every requested bucket belongs to this user
    for bname in requested_buckets:
        if not bname.startswith(username_prefix):
            logger.warning(
                "PURGE_BUCKETS denied: user=%s tried to delete bucket=%s (prefix mismatch)",
                client_id, bname,
            )
            return json.dumps({
                "status": "error",
                "message": f"Bucket '{bname}' does not belong to user '{client_id}'",
            })

    # Load admin S3 config to get the mc alias
    from umrs.shared.json_config import load_server_cfg_file
    try:
        admin_s3_cfg_file, _users, _cfg = load_server_cfg_file(configure_file)
    except Exception as exc:
        return json.dumps({"status": "error", "message": f"Server config load failed: {exc}"})

    admin_cfg_path = os.path.expanduser(admin_s3_cfg_file)
    if not os.path.isfile(admin_cfg_path):
        return json.dumps({"status": "error", "message": "Admin S3 config not found on server"})

    import configparser
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read(admin_cfg_path)
    alias = cfg.get("default", "alias", fallback="myminio")

    from umrs.shared.storage.minio import minio_remove_bucket

    deleted = []
    errors = []
    for bname in requested_buckets:
        try:
            minio_remove_bucket(bname, alias)
            deleted.append(bname)
        except Exception as exc:
            errors.append({"bucket": bname, "error": str(exc)})

    status = "ok" if not errors else ("partial" if deleted else "error")
    return json.dumps({"status": status, "deleted": deleted, "errors": errors})
