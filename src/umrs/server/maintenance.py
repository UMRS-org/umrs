"""Server maintenance operations: recompute, update, check, reload dispatch."""

import logging
import time
from typing import Any

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
        logger.info(
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
        logger.info(summary)

    if active_threads.is_worker_thread():
        active_threads.update_status_to_done()
        users = (
            {username}
            if username is not None
            else {entry.s3_username for entry in buckets_cfg.values()}
        )
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome = {"summary": summary, "data": recompute_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key, username=username)
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
        logger.info(
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
                    logger.info(f"{s3_object}: {s}")
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
        logger.info(summary)

    if active_threads.is_worker_thread():
        active_threads.update_status_to_done()
        users = (
            {username}
            if username is not None
            else {entry.s3_username for entry in buckets_cfg.values()}
        )
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome = {"summary": summary, "data": update_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key, username=username)

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


def check_objects(
    active_threads,
    username,
    buckets_cfg,
    key,
    verbose=False,
    log_output=True,
    update_status_callback=None,
):
    if log_output:
        logger.info(
            "\n######################\n"
            "Checking all objects in all buckets\n"
            "######################\n"
        )

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    perform_update = False
    check_dict = {}
    for path_to_bucket, entry in buckets_cfg.items():
        # skip none matching username
        if username is not None and username != entry.s3_username:
            continue
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
                logger.info(f"{s3_object}: {s}")
            if s == "match":
                nb_match += 1
            elif s == "mismatch":
                nb_mismatch += 1
            elif s == "not_found":
                nb_not_found += 1
            else:
                logger.info(f"Error: {status[i]['message']}")
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
        logger.info(summary)
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
        logger.info(note)
    summary_with_note = summary + note
    outcome = {
        "summary": summary_with_note,
        "data": check_dict,
        "requires_update": perform_update,
    }
    filter_outcome(active_threads, outcome, buckets_cfg, key, username=username)

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


# ---------------------------------------------------------------------------
# Outcome filtering (per-user scoping for STATUS replies)
# ---------------------------------------------------------------------------


def filter_outcome(active_threads, outcome, buckets_cfg, key, username=None):
    """Store outcome in the status store, scoped to the requesting user.

    username=None (admin/bootstrap): store for every user in buckets_cfg.
    username set (regular user): store only for that user.
    """
    users = (
        {username}
        if username is not None
        else {entry.s3_username for entry in buckets_cfg.values()}
    )
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
    try:
        if cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE):
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            summary = pretty_print_recompute(
                filtered, update=(cmd == TaskCmd.UPDATE), print_output=False
            )
            return {"summary": summary, "data": filtered}

        if cmd == TaskCmd.CHECK:
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

        if cmd == TaskCmd.RELOAD:
            # Derive a per-user configuration view from the live buckets_cfg
            filtered_store = buckets_cfg.build_dict_for_user(client_id)
            summary = pretty_print_cfg(
                {"buckets_cfg": filtered_store}, print_output=False
            )
            return {"summary": summary}
    except Exception as e:
        # If filtering fails for any reason, fall back to original outcome
        logger.info(f"Warning: failed to filter outcome for client '{client_id}': {e}")
    return outcome


# ---------------------------------------------------------------------------
# Configuration reload and validation
# ---------------------------------------------------------------------------


def check_duplicated_database(users):
    """Quarantine experiences whose database name collides with another's.

    Each experience resolves to a Postgres database named
    ``create_db_name(sql_username, xp_name, username)``. Since the platform
    ``username`` is part of the name, two *different* users can safely reuse the
    same ``(xp_name, sql_username)`` pair — they get isolated databases. The only
    way two experiences now collide is when a *single* user lists the same
    ``(xp_name, sql_username)`` twice (a genuine config error).

    Historically any duplicate aborted startup via ``sys.exit`` — a single bad
    entry crash-looped the whole server and took the front offline for every
    user. Instead, keep the first claimant of each database name and skip
    (quarantine) any later colliding experience, logging a loud warning so the
    conflict is visible. The server stays up for everyone else.

    The ``users`` list is mutated in place: colliding experiences are dropped
    from the affected users' ``experiences`` lists so downstream provisioning
    never double-provisions a database.
    """
    database_owner: dict[str, str] = {}
    for user in users:
        username = user["username"]
        kept_experiences = []
        for xp in user["experiences"]:
            sql_username = xp["sql_username"]
            xp_name = xp["xp_name"]
            db_name = create_db_name(sql_username, xp_name, username)
            prior_owner = database_owner.get(db_name)
            if prior_owner is not None:
                logger.warning(
                    "Skipping experience '%s' for user '%s': database name '%s' "
                    "is already claimed by user '%s'. Experiences must map to "
                    "distinct databases — change xp_name or sql_username. "
                    "Booting without this experience.",
                    xp_name,
                    username,
                    db_name,
                    prior_owner,
                )
                continue
            database_owner[db_name] = username
            kept_experiences.append(xp)
        user["experiences"] = kept_experiences


def _close_buckets_connections(buckets_cfg):
    """Close all active S3 connections in the current bucket config."""
    for entry in buckets_cfg.values():
        try:
            conn = entry.s3_connection
            if conn:
                conn.close()
        except Exception as e:
            logger.info(f"Warning: failed to close S3 connection during reload: {e}")


def _provision_all_users(
    new_users, new_admin_s3_cfg
) -> tuple[ServerCfg, list[dict[str, Any]]]:
    """Provision every user and return the new server config plus errors."""
    from umrs.server.provisioning import create_s3_sql_users

    new_server_cfg = ServerCfg()
    modality_errors: list[dict[str, Any]] = []
    for u in new_users:
        create_s3_sql_users(u, new_admin_s3_cfg)
        configure_user(u, new_server_cfg, modality_errors=modality_errors)
    return new_server_cfg, modality_errors


def _format_modality_errors(modality_errors: list[dict[str, Any]]) -> str:
    """Format per-modality provisioning errors for display."""
    return "\n".join(
        f"- {e['username']}/{e['xp_name']}/{e['modality']}: {e['error']}"
        for e in modality_errors
    )


def reload_configuration(configure_file, buckets_cfg, user_config_map, log_output=True):
    """Hot-reload server configuration from disk."""
    from umrs.server.json_config import load_server_cfg_file

    if log_output:
        logger.info(
            "\n######################\n"
            "Reloading configuration...\n"
            "######################\n"
        )
    try:
        new_admin_s3_cfg, new_users, _ = load_server_cfg_file(configure_file)
        check_duplicated_database(new_users)
        _close_buckets_connections(buckets_cfg)

        new_server_cfg, modality_errors = _provision_all_users(
            new_users, new_admin_s3_cfg
        )

        buckets_cfg.replace_cfg(new_server_cfg.buckets_cfg)
        user_config_map.clear()
        user_config_map.update(new_server_cfg.user_config_map)
        summary = pretty_print_cfg(new_server_cfg, print_output=False)
        if modality_errors:
            error_lines = _format_modality_errors(modality_errors)
            summary += (
                f"\n\n❌ {len(modality_errors)} modality(s) failed to provision:\n"
                f"{error_lines}"
            )
            if log_output:
                logger.info(summary)
            return {
                "summary": summary,
                "error": error_lines,
                "modality_errors": modality_errors,
            }
        if log_output:
            logger.info(summary)
            logger.info("\n✅ Reload complete.")
        return {"summary": summary}
    except Exception as e:
        error_msg = f"\n❌ Reload failed: {e}"
        if log_output:
            logger.info(error_msg)
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
            client_id,
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
        logger.info(
            f"====> Error: unknown command '{cmd}' returned by maintenance handler"
        )
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

       Returns JSON: {"status": "ok"|"partial"|"error", "deleted": [...], "errors":
    [...]}
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
                client_id,
                bname,
            )
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Bucket '{bname}' does not belong to user '{client_id}'",
                }
            )

    # Load admin S3 config to get the mc alias
    from umrs.shared.json_config import load_server_cfg_file

    try:
        admin_s3_cfg_file, _users, _cfg = load_server_cfg_file(configure_file)
    except Exception as exc:
        return json.dumps(
            {"status": "error", "message": f"Server config load failed: {exc}"}
        )

    admin_cfg_path = os.path.expanduser(admin_s3_cfg_file)
    if not os.path.isfile(admin_cfg_path):
        return json.dumps(
            {"status": "error", "message": "Admin S3 config not found on server"}
        )

    from umrs.shared.storage.backends import get_backend

    backend = get_backend()

    deleted = []
    errors = []
    for bname in requested_buckets:
        try:
            backend.remove_bucket(bname, admin_cfg_path)
            deleted.append(bname)
        except Exception as exc:
            errors.append({"bucket": bname, "error": str(exc)})

    # Remove orphaned experiences from the server-side user config so
    # reload does not recreate the purged buckets.
    removed_experiences = []
    if deleted:
        try:
            removed_experiences = _remove_purged_experiences_from_user_config(
                deleted,
                client_id,
                username_prefix,
                configure_file,
            )
        except Exception as exc:
            logger.warning("Could not update user config after purge: %s", exc)

    status = "ok" if not errors else ("partial" if deleted else "error")
    result = {"status": status, "deleted": deleted, "errors": errors}
    if removed_experiences:
        result["removed_experiences"] = removed_experiences
    return json.dumps(result)


def _remove_purged_experiences_from_user_config(
    deleted_buckets: list[str],
    client_id: str,
    username_prefix: str,
    configure_file: str,
) -> list[str]:
    """Remove experiences whose buckets were all purged from the user config.

    Returns the list of experience names that were removed.
    """
    import json as _json
    import os

    # Extract normalised xp_names from deleted bucket names.
    # Bucket format: <username_norm>.<xp_name_norm>.<alias>[-output]
    purged_xp_names: set[str] = set()
    for bname in deleted_buckets:
        rest = bname[len(username_prefix) :]  # already validated prefix
        parts = rest.split(".", 1)
        if len(parts) >= 2:
            purged_xp_names.add(parts[0])

    if not purged_xp_names:
        return []

    # Find the user's config file via server config
    from umrs.shared.json_config import load_server_cfg_file

    _admin_s3, users, _server_cfg = load_server_cfg_file(configure_file)
    user_cfg = None
    user_cfg_path = None
    for u in users:
        if u.get("username") == client_id:
            user_cfg = u
            user_cfg_path = u.get("config_file_path")
            break

    if not user_cfg or not user_cfg_path:
        return []

    experiences = user_cfg.get("experiences", [])
    kept = []
    removed_names = []
    for xp in experiences:
        xp_name = xp.get("xp_name", "")
        # Normalise the same way as bucket names
        xp_norm = xp_name.strip().lower().replace(" ", "-").replace("_", "-")
        if xp_norm in purged_xp_names:
            removed_names.append(xp_name)
        else:
            kept.append(xp)

    if not removed_names:
        return []

    # Re-read the raw file to preserve structure, then update experiences
    cfg_path = os.path.expanduser(user_cfg_path)
    with open(cfg_path, encoding="utf-8") as f:
        raw_cfg = _json.load(f)

    raw_cfg["experiences"] = [
        xp
        for xp in raw_cfg.get("experiences", [])
        if xp.get("xp_name", "").strip().lower().replace(" ", "-").replace("_", "-")
        not in purged_xp_names
    ]

    with open(cfg_path, "w", encoding="utf-8") as f:
        _json.dump(raw_cfg, f, indent=4, ensure_ascii=False)
        f.write("\n")

    logger.info(
        "Removed %d experience(s) from %s after purge: %s",
        len(removed_names),
        cfg_path,
        ", ".join(removed_names),
    )
    return removed_names
