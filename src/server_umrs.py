#!/usr/bin/env python3

import argparse
import os
import sys
from collections import defaultdict
import time
import json
import diskcache

# Add the parent directory of this script (src) to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


import threading
from umrs.shared.storage.s3 import *
from umrs.server.sql import check_S3_object_status_in_db, create_db_name
from umrs.core.MessageQueue import MessageQueue
from umrs.shared.pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check, pretty_print_thread_status

from umrs.server.json_config import load_server_cfg_file, load_client_cfg_file
from umrs.core.TaskManager import TaskManager
from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.models.embedding_backend import LocalEmbeddingBackend

from umrs.server.model_registry_handler import handle_model_command
from umrs.server.workflow_registry_handler import handle_workflow_command, register_workflow
from umrs.server.job_tracking_handler import handle_job_command
from umrs.shared.config import *
from umrs.server.config import *
from umrs.server.provisioning import (
    ServerCfg,
    configure_user,
    create_s3_sql_users,
    is_client_configured,
    reconcile_status_on_startup,
)

_status_store_global = None
ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag

rivault_lock = threading.Lock()

# --------------------------- Maintenance commands ---------------------------

def recompute_all_objects(active_threads, username, buckets_cfg, device, embedding_backend, key, log_output=True, update_status_callback=None):

    if log_output:
        print(f"\n######################\nRecomputing all objects in all buckets\n######################\n")

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    recompute_dict = {}

    for path_to_bucket, entry in buckets_cfg.items():
        with entry.reentrant_lock :
            # skip none matching username
            if username is not None and username != entry.s3_username:
                continue
            s3_objects, total_size, _ = list_s3_bucket(entry.s3_connection, path_to_bucket)
            if len(s3_objects) > 0:
                start_time = time.perf_counter()
                process_s3_objects(active_threads, s3_objects, buckets_cfg, device, embedding_backend, key, update_when_done=False, force_recompute=True)
                duration = round((time.perf_counter() - start_time),3)
            else:
                duration = 0

            recompute_dict[path_to_bucket] = {
                "bucket_alias": entry.bucket_alias,
                "duration": duration,
                "nb_objects": len(s3_objects),
                "modality_name": entry.modality_name,
                "s3_username": entry.s3_username,
                "total_size_MB": round(total_size / (1024**2), 3),
                "model": entry.model
            }

    summary = pretty_print_recompute(recompute_dict, print_output=False)
    if log_output:
        print(summary)

    if active_threads.is_worker_thread() :
        active_threads.update_status_to_done()
        users = {entry.s3_username for entry in buckets_cfg.values()}
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome = {"summary": summary, "data": recompute_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key)
    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)



def update_objects(active_threads, username, buckets_cfg, device, embedding_backend, key, verbose=False, log_output=True, update_status_callback=None):
    if log_output:
        print(f"\n######################\nUpdating missing/mismatched objects in all buckets\n######################\n")

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    update_dict = {}

    for path_to_bucket, entry in buckets_cfg.items():
       with entry.reentrant_lock :
            # skip none matching username
            if username is not None and username != entry.s3_username:
                continue
            s3_objects, total_size, checksums = list_s3_bucket(entry.s3_connection, path_to_bucket)
            status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, buckets_cfg)

            to_update_objects = []
            for i, s3_object in enumerate(s3_objects):
                s = status[i]['status']
                if verbose:
                    print(f"{s3_object}: {s}")
                if s != "match":
                    to_update_objects.append(s3_object)

            if len(to_update_objects) > 0:
                start_time = time.perf_counter()
                process_s3_objects(active_threads, to_update_objects, buckets_cfg, device, embedding_backend, key, update_when_done=False)
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
                "model": entry.model
            }

    summary = pretty_print_recompute(update_dict, update=True, print_output=False)
    if log_output:
        print(summary)

    if active_threads.is_worker_thread() :
        active_threads.update_status_to_done()
        users = {entry.s3_username for entry in buckets_cfg.values()}
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    outcome= {"summary": summary, "data": update_dict}
    filter_outcome(active_threads, outcome, buckets_cfg, key)

    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)



def check_objects(active_threads, buckets_cfg, key, verbose=False, log_output=True, update_status_callback=None):
    if log_output:
        print(f"\n######################\nChecking all objects in all buckets\n######################\n")

    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)

    perform_update = False
    check_dict = {}
    for path_to_bucket, entry in buckets_cfg.items():
        nb_not_found, nb_match, nb_mismatch, nb_error = 0, 0, 0, 0
        s3_objects, total_size, checksums = list_s3_bucket(entry.s3_connection, path_to_bucket)
        status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, buckets_cfg)
        for i, s3_object in enumerate(s3_objects):
            s = status[i]['status']
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
            "Errors": nb_error
        }

    if active_threads.is_worker_thread() :
        active_threads.update_status_to_done()
        active_threads.remove_from_running_list()

    summary = pretty_print_check(check_dict, print_output=False)
    if log_output:
        print(summary)
    if perform_update:
        note = "\n⚠️ Warning: The database is not fully consistent with the objects stored in S3.\n→ To recompute and update the missing or mismatched entries, restart the server with the `--force-update` option."
    else:
        note = "\n ✅ No problem: The database is fully consistent with the objects stored in S3."
    if log_output:
        print(note)
    summary_with_note = summary + note
    outcome= {"summary": summary_with_note, "data": check_dict, "requires_update": perform_update}
    filter_outcome(active_threads, outcome, buckets_cfg, key)


    # execute callback to update task status to completed
    if update_status_callback:
        update_status_callback(TaskStatus.COMPLETED)


def _handle_maintenance_command(active_threads, cmd, client_id, buckets_cfg, device, embedding_backend,
                                configure_file, user_config_map, key, update_status_callback=None):

    # print(f"handle {cmd=} {client_id=}")

    if cmd == TaskCmd.RECOMPUTE:
        return active_threads.launch_recompute_all_objects(client_id, buckets_cfg, device, embedding_backend, key, log_output=ENABLE_PRETTY_PRINT, update_status_callback=update_status_callback)

    elif cmd == TaskCmd.UPDATE:
        return active_threads.launch_update_objects(client_id, buckets_cfg, device, embedding_backend, key, verbose=False, log_output=ENABLE_PRETTY_PRINT, update_status_callback=update_status_callback)
    elif cmd == TaskCmd.CHECK:
        return active_threads.launch_check_objects(buckets_cfg, key, verbose=False, log_output=ENABLE_PRETTY_PRINT, update_status_callback=update_status_callback)
    elif cmd == TaskCmd.RELOAD:
        outcome = reload_configuration(configure_file, buckets_cfg, user_config_map, log_output=ENABLE_PRETTY_PRINT)
        filter_outcome(active_threads, outcome, buckets_cfg, key)
        return None
    else:
        print(f"====> Error: unknown command '{cmd}' returned by maintenance handler")
        return None



def filter_outcome(active_threads, outcome, buckets_cfg, key):
    # For each user in the configuration, store its relevant data
    # to be later visible via STATUS
    users = {entry.s3_username for entry in buckets_cfg.values()}
    for user in users:
        filtered_outcome = _filter_outcome_for_client(key, outcome, user, buckets_cfg)
        active_threads.record_status(user, key, filtered_outcome)


def _filter_dict_by_user(data_dict, client_id):
    """Return a shallow copy of data_dict entries that belong to client_id (match on 's3_username')."""
    if not isinstance(data_dict, dict):
        return {}
    return {k: v for k, v in data_dict.items() if isinstance(v, dict) and v.get('s3_username') == client_id}


def _filter_outcome_for_client(cmd, outcome, client_id, buckets_cfg):
    """Filter an operation outcome so it only contains information for the requesting client_id.

    For recompute/update/check: filter the 'data' dict by s3_username then rebuild the summary.
    For reload: build a per-user configuration summary from the current buckets_cfg.
    """


    try:
        if cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE):
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            summary = pretty_print_recompute(filtered, update=(cmd == "update"), print_output=False)
            return {"summary": summary, "data": filtered}

        if cmd == TaskCmd.CHECK:
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            # print(f"{full=}\n{filtered=}")
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
            return {"summary": summary, "data": filtered, "requires_update": requires_update}

        if cmd == TaskCmd.RELOAD:
            # Derive a per-user configuration view from the live buckets_cfg
            filtered_store = buckets_cfg.build_dict_for_user(client_id)
            summary = pretty_print_cfg({"buckets_cfg": filtered_store}, print_output=False)
            return {"summary": summary}
    except Exception:
        # If filtering fails for any reason, fall back to original outcome but do not raise
        pass
    return outcome


# --------------------------- Boot and hot-reload ----------------------------
def check_duplicated_database(users):
    databases_name_dict = {}
    for user in users:
        username = user["username"]
        for xp in user["experiences"]:
            sql_username = xp["sql_username"]
            xp_name = xp["xp_name"]
            db_name = create_db_name(sql_username, xp_name)

            if db_name in databases_name_dict:
                sys.exit(f"\t❌ Error: duplicated database '{db_name}' for users '{username}' and '{databases_name_dict[db_name]}'\n\tChange xp_name or sql_username in json file.\n\tExiting...")

            databases_name_dict[db_name] = username


def perform_initial_config(configure_file, status_store):
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")

    admin_s3_configuration_file, users, server_cfg = load_server_cfg_file(configure_file)
    check_duplicated_database(users)
    print("Configuration loaded successfully.")

    # Get the directory containing configure_file
    umrs_dir = os.path.dirname(configure_file)

    # Define the temp directory path inside that directory
    tmp_dir = os.path.join(umrs_dir, get_default_tmp_subdir())

    # Create the tmp_dir if it does not exist
    os.makedirs(tmp_dir, exist_ok=True)
    control_mq = MessageQueue(role='server', queue_type='control', timeout_ms=1000, tmp_dir = tmp_dir, status_store=status_store)
    # Align MessageQueue debug with server pretty-print setting (off by default)
    try:
        control_mq.set_debug(ENABLE_PRETTY_PRINT)
    except Exception:
        # Debug flag is optional; ignore failures setting it
        pass
    print ("Subscribed to zeroMQ message queue")

    for user in users:
        create_s3_sql_users(user, admin_s3_configuration_file)

    server_cfg = ServerCfg()
    for user in users:
        configure_user(user, server_cfg)

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    embedding_backend = LocalEmbeddingBackend()
    print(f"\n######################\nServer configuration done!\n######################\n")

    return control_mq, server_cfg, device, embedding_backend


def reload_configuration(configure_file, buckets_cfg, user_config_map, log_output=True):
    if log_output:
        print("\n######################\nReloading configuration...\n######################\n")
    try:
        new_admin_s3_cfg, new_users, _ = load_server_cfg_file(configure_file)
        check_duplicated_database(new_users)
        for entry in buckets_cfg.values():
            try:
                conn = entry.s3_connection
                if conn:
                    conn.close()
            except Exception:
                pass

        for u in new_users:
            create_s3_sql_users(u, new_admin_s3_cfg)

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
        return {"summary": summary}
    except Exception as e:
        error_msg = f"\n❌ Reload failed: {e}"
        if log_output:
            print(error_msg)
        return {"summary": error_msg, "error": str(e)}


def _ensure_user_file_in_conf(conf_path, username, canonical_user_path):
    canonical_user_path = os.path.expanduser(canonical_user_path)

    try:
        with open(conf_path, 'r', encoding='utf-8') as f:
            conf = json.load(f)
    except Exception as e:
        print(f"Warning: failed to read conf '{conf_path}': {e}")
        return False

    users = conf.get('users', [])

    # Build a new list that keeps only one entry for this username (the canonical path)
    new_users = []
    removed = []
    for p in users:
        p = os.path.expanduser(p)
        if p == canonical_user_path:
            # keep canonical path if already present
            if p not in new_users:
                new_users.append(p)
            continue
        # Try to read file and inspect username; if mismatch or file missing, keep as-is
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            u = data.get('username')
        except Exception:
            u = None
        if u == username:
            # Another config for the same user; drop it
            removed.append(p)
            continue
        # Different user or unreadable -> keep entry
        if p not in new_users:
            new_users.append(p)

    # Ensure canonical path present
    if canonical_user_path not in new_users:
        new_users.append(canonical_user_path)

    if removed:
        print(f"Updated server conf: removed {len(removed)} stale user config(s) for '{username}': {removed}")

    # change absolute path to '~' if possible
    home = os.path.expanduser('~')
    new_users = [path.replace(home, '~',1) for path in new_users]

    conf['users'] = new_users
    try:
        with open(conf_path, 'w', encoding='utf-8') as f:
            json.dump(conf, f, indent=2)
        print(f"Updated server conf: ensured unique config for '{username}' -> {canonical_user_path}")
    except Exception as e:
        print(f"Warning: failed to update conf '{conf_path}': {e}")
        return False
    # Best-effort cleanup: remove pruned files under user_cfg_path only
    user_cfg_path = os.path.join(os.path.dirname(conf_path), get_users_default_cfg_subdir())
    for stale in removed:
        try:
            if stale.startswith(user_cfg_path) and stale != canonical_user_path and os.path.isfile(stale):
                os.remove(stale)
                print(f"Deleted stale user config file: {stale}")
        except Exception as e:
            print(f"Warning: failed to delete stale user config '{stale}': {e}")
    return True


def _handle_upload_workflow(client_id, zip_path, workflow_name):
    """Validate and extract an uploaded workflow zip to the user_workflows directory.

    The zip must contain a ``workflow.json`` at its root (or one level deep)
    with a parseable :class:`WorkflowDefinition`.

    Returns ``(success: bool, dest_dir: str | None, error: str | None)``.
    """
    import zipfile
    from umrs.pipeline_sdk.workflow import WorkflowDefinition

    if not os.path.isfile(zip_path):
        return False, None, f"Uploaded file not found: {zip_path}"

    # Basic size sanity check (reject > 500 MB)
    max_size = 500 * 1024 * 1024
    if os.path.getsize(zip_path) > max_size:
        return False, None, f"Zip file too large ({os.path.getsize(zip_path)} bytes, max {max_size})"

    if not zipfile.is_zipfile(zip_path):
        return False, None, f"Uploaded file is not a valid zip archive: {zip_path}"

    # Determine destination
    workflows_root = os.path.join(get_default_tmp_subdir(), "user_workflows", client_id)
    dest_dir = os.path.join(workflows_root, workflow_name)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Security: reject paths that escape the extraction directory
            for member in zf.namelist():
                resolved = os.path.realpath(os.path.join(dest_dir, member))
                if not resolved.startswith(os.path.realpath(dest_dir)):
                    return False, None, f"Zip contains unsafe path: {member}"

            # Look for workflow.json at root or one level deep
            candidates = [n for n in zf.namelist() if os.path.basename(n) == "workflow.json"]
            if not candidates:
                return False, None, "Zip archive does not contain a workflow.json"

            # Validate workflow.json contents before extracting
            wf_entry = candidates[0]
            try:
                wf_data = json.loads(zf.read(wf_entry))
                wf = WorkflowDefinition.from_dict(wf_data)
                # Override source to UPLOADED type pointing to dest_dir
                # (validation of original source is skipped — we rewrite it)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                return False, None, f"Invalid workflow.json: {e}"

            # Extract
            os.makedirs(dest_dir, exist_ok=True)
            zf.extractall(dest_dir)

        # If workflow.json was nested one level (e.g., myworkflow/workflow.json),
        # move contents up so dest_dir directly contains workflow.json
        actual_wf = os.path.join(dest_dir, "workflow.json")
        if not os.path.isfile(actual_wf):
            # Try to find it one level deep
            for name in os.listdir(dest_dir):
                candidate = os.path.join(dest_dir, name, "workflow.json")
                if os.path.isfile(candidate):
                    # Move everything from the subdirectory up
                    sub_dir = os.path.join(dest_dir, name)
                    import shutil
                    for item in os.listdir(sub_dir):
                        shutil.move(os.path.join(sub_dir, item), dest_dir)
                    os.rmdir(sub_dir)
                    break

        # Rewrite workflow.json source to UPLOADED type
        actual_wf = os.path.join(dest_dir, "workflow.json")
        if os.path.isfile(actual_wf):
            with open(actual_wf, "r", encoding="utf-8") as fh:
                wf_data = json.load(fh)
            wf_data["source"] = {"type": "uploaded", "path": dest_dir}
            with open(actual_wf, "w", encoding="utf-8") as fh:
                json.dump(wf_data, fh, indent=2)
    except zipfile.BadZipFile:
        return False, None, f"Corrupted zip archive: {zip_path}"
    except Exception as e:
        return False, None, f"Failed to extract workflow: {e}"
    finally:
        # Clean up the uploaded temp zip
        try:
            os.remove(zip_path)
        except Exception:
            pass

    print(f"Workflow '{workflow_name}' for user '{client_id}' extracted to {dest_dir}")
    return True, dest_dir, None


def _persist_uploaded_user_config(tmp_path, configure_file, user_config_map, expected_username=None):
    """Persist an uploaded user config JSON: replace create the dir and file and ensure listed in conf.

    Returns (success: bool, username: str | None, dest_path: str | None)
    """
    try:
        # Validate against client schema (strict) and obtain data
        data = load_client_cfg_file(tmp_path)
        username = data.get('username')
        if not username:
            print(f"Warning: uploaded file '{tmp_path}' missing 'username' field; ignoring")
            return False, None, None
        if expected_username and username != expected_username:
            print(f"Warning: uploaded config username '{username}' does not match client identity '{expected_username}'; ignoring file")
            return False, None, None
    except SystemExit:
        # load_client_cfg_file may sys.exit() on validation error – treat as validation failure
        print(f"Warning: uploaded file '{tmp_path}' failed validation; ignoring")
        return False, None, None
    except Exception as e:
        print(f"Warning: cannot parse uploaded config '{tmp_path}': {e}")
        return False, None, None

    # look up existing users configuration path
    dest_path = user_config_map.get(username, None)

    if not dest_path:
        # Destination path (canonical)
        umrs_dir = os.path.dirname(configure_file)
        user_dir = os.path.join(umrs_dir, get_users_default_cfg_subdir())
        try:
            os.makedirs(user_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: cannot ensure user files directory '{user_dir}': {e}")
            return False, username, None
        dest_path = os.path.join(user_dir, f"{username}.json")

    dest_path = os.path.expanduser(dest_path)

    # Atomically replace the destination file
    try:
        with open(tmp_path, 'r', encoding='utf-8') as src:
            content = src.read()
        tmp_out = dest_path + ".tmp"
        with open(tmp_out, 'w', encoding='utf-8') as out:
            out.write(content)
        os.replace(tmp_out, dest_path)
        # Remove original temp upload
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        print(f"Stored updated user config for '{username}' at '{dest_path}'")
    except Exception as e:
        print(f"Warning: failed to persist uploaded config to '{dest_path}': {e}")
        return False, username, None

    # Ensure conf contains this user file path
    conf_path = os.path.abspath(configure_file)
    _ensure_user_file_in_conf(conf_path, username, dest_path)
    return True, username, dest_path


#----------------------------
# Process S3 objects
#----------------------------
def process_s3_objects(active_threads, s3_objects, buckets_cfg, device, embedding_backend, key, update_when_done = True, update_status_callback=None, force_recompute=False):
    nb_objects = len(s3_objects)
    keys_per_bucket = defaultdict(list)
    users = []
    print(f"\t-->Detected {nb_objects} new file(s):")

    for s3_object in s3_objects:
        print(f"\t\t-->{s3_object}")
        path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
        if s3_key not in keys_per_bucket[path_to_bucket]:
            keys_per_bucket[path_to_bucket].append(s3_key)

    active_threads.update_thread_dict(buckets_cfg, keys_per_bucket, key)


    exit_status = TaskStatus.COMPLETED
    for path_to_bucket in keys_per_bucket.keys():
        selected_fields = ["s3_username", "data_base", "table_name", "model", "store_embeddings_flag", "modality_name", "task", "s3_connection", "output_path_to_bucket", "reentrant_lock"]
        user, data_base, table_name, model, store_embeddings_flag, modality_name, task, s3_connection, output_bucket, reentrant_lock = buckets_cfg.get(path_to_bucket, filter=selected_fields)

        users.append(user)
        s3_keys = keys_per_bucket[path_to_bucket]
        if model.type == "rivault":
            from umrs.server.rivault_launcher import launch_rivault_model
            try:
                if update_status_callback:
                    update_status_callback(TaskStatus.INITIALIZING)
                # launch_rivault_model is not reentrant, so we use a lock to ensure only one thread is running it at a time
                with rivault_lock:
                    model = launch_rivault_model(model.model.copy())
            except Exception as e:
                import traceback
                print(f"Error creating the RiVault model: {e}")
                traceback.print_exc()
                exit_status = TaskStatus.ERROR
                continue



        with reentrant_lock:
            should_process = store_embeddings_flag or (model is not None and getattr(model, 'return_type', None) == 'file')
            if should_process:
                if exit_status != TaskStatus.ERROR:
                    if update_status_callback:
                        update_status_callback(TaskStatus.RUNNING)
                    print(f"\t--> Processing in {table_name}")
                    try:
                        embedding_backend.compute_and_store(
                            path_to_bucket,
                            s3_keys,
                            data_base,
                            table_name,
                            model,
                            device,
                            modality_name,
                            s3_connection,
                            output_bucket,
                            task=task,
                            skip_existing=not force_recompute,
                        )
                    except Exception as e:
                        import traceback
                        print(f"Error computing/storing embeddings for bucket '{path_to_bucket}': {e}")
                        traceback.print_exc()
                        exit_status = TaskStatus.ERROR
            else:
                print(f"Error: We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")\n\tUser is probably not configured")
            buckets_cfg.update_num_objects(path_to_bucket)

    if update_when_done:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    # --- Service lifecycle: auto-release backend jobs when idle ---
    service_lifecycle = os.environ.get("UMRS_SERVICE_LIFECYCLE", "auto_release")
    if service_lifecycle == "auto_release":
        with active_threads.lock:
            still_running = sum(
                1 for t in active_threads.thread_list
                if t.is_alive() and t is not threading.current_thread()
            )
        if still_running == 0:
            from umrs.server.rivault_launcher import release_all_backend_jobs
            release_all_backend_jobs()

    if update_status_callback:
        update_status_callback(exit_status)

    print(f"\t--> Processing {nb_objects} file(s) done with status: {exit_status}!")
    with active_threads.lock:
        pretty_print_thread_status(active_threads.thread_dict, running_only=False, print_output=ENABLE_PRETTY_PRINT)
    print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")

##################################################################################
#########################        Main server loop        #########################
##################################################################################

def make_update_status_callback(task_manager, task_id):
    """
    Return a callback function that updates the task status for the given task_id.
    The returned function accepts a `status` argument when called.
    """
    def callback(status):
        task_manager.tasks_status.update_status(task_id, status)
    return callback

def server_event_loop(task_manager, control_mq, server_cfg, device, embedding_backend, configure_file):
    buckets_cfg = server_cfg.buckets_cfg
    user_config_map = server_cfg.user_config_map
    status_store = control_mq.get_status_store()
    assert status_store is not None
    assert status_store == task_manager.active_threads.get_status_store()

    try:
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######", flush=True)
            task_manager.tasks_status.pretty_print_tasks(limit=20)
            try:
                result = control_mq.server_control_queue_loop()
            except Exception as e:
                print(f"ERROR in control queue loop: {e}", flush=True)
                continue
            if not result:
                continue
            cmd, payload = result

            task_id = task_manager.tasks_status.create_task(cmd, payload)

            # create a callback with task_id embedded
            update_status_func = make_update_status_callback(task_manager, task_id)

            if cmd == TaskCmd.PROCESS_OBJECTS:
                client_id, s3_objects = payload
                thread_ident = task_manager.active_threads.launch_s3_threads(s3_objects,
                                                                             buckets_cfg, device, embedding_backend,
                                                                             cmd.lower(),
                                                                             update_status_callback=update_status_func)
                task_manager.associate_thread_with_task(thread_ident, task_id)

            elif cmd == TaskCmd.FILE:
                if payload:
                    client_id, files = payload if isinstance(payload, tuple) else (None, payload)
                    for file in files:
                        print(f"Received file: {file}")
                        ok, uname, dest = _persist_uploaded_user_config(file, configure_file, user_config_map, expected_username=client_id)
                        if ok:
                            print(f"User config persisted for '{uname}' -> {dest}")
                        else:
                            print(f"Note: file '{file}' not recognized as valid user config or identity mismatch; left in place.")
                else:
                    print("Warning: FILE command received with no payload")
                update_status_func(TaskStatus.COMPLETED)
            elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK, TaskCmd.RELOAD):
                client_id = payload
                thread_ident = _handle_maintenance_command(task_manager.active_threads, cmd, client_id, buckets_cfg,
                                                            device, embedding_backend, configure_file, user_config_map, cmd.lower(),
                                                            update_status_callback=update_status_func)
                if thread_ident is None:
                    update_status_func(TaskStatus.COMPLETED)
                else:
                    task_manager.associate_thread_with_task(thread_ident, task_id)

            elif cmd == TaskCmd.UPLOAD_WORKFLOW:
                client_id, files, workflow_name = payload
                if not files:
                    print("Warning: UPLOAD_WORKFLOW received with no files")
                    update_status_func(TaskStatus.ERROR)
                else:
                    zip_path = files[0]
                    ok, dest, err = _handle_upload_workflow(client_id, zip_path, workflow_name)
                    if ok:
                        print(f"Workflow '{workflow_name}' uploaded for user '{client_id}' at {dest}")
                        # Auto-register in workflow catalog
                        try:
                            wf_json_path = os.path.join(dest, "workflow.json")
                            with open(wf_json_path, "r", encoding="utf-8") as fh:
                                wf_data = json.load(fh)
                            version = wf_data.get("version", "0.1")
                            register_workflow(
                                workflow_name=workflow_name,
                                version=str(version),
                                user_id=client_id,
                                source_path=dest,
                                workflow_json=wf_data,
                                scope="user",
                                description=wf_data.get("description", ""),
                            )
                        except Exception as e:
                            print(f"Warning: auto-registration of workflow '{workflow_name}' failed: {e}")
                        update_status_func(TaskStatus.COMPLETED)
                    else:
                        print(f"UPLOAD_WORKFLOW failed: {err}")
                        update_status_func(TaskStatus.ERROR)

            elif cmd == TaskCmd.LIST_WORKFLOWS:
                client_id, payload_str, zmq_sender_id = payload
                response_json = handle_workflow_command(cmd, client_id, payload_str)
                try:
                    control_mq.send_string(response_json, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error sending workflow command response: {e}")
                update_status_func(TaskStatus.COMPLETED)

            elif cmd == TaskCmd.WAIT_OUTPUT : # "WAIT_BUCKET", (client_id, bucket_name, timeout, zmq_sender_id)
                client_id, output_path_to_bucket, timeout, zmq_sender_id = payload

                path_to_bucket, entry = buckets_cfg.find_by_output_path(output_path_to_bucket)
                if entry is None:
                    msg = "NO_SUCH_BUCKET"
                    control_mq.send_string(msg, zmq_sender_id)
                    update_status_func(TaskStatus.NO_SUCH_BUCKET)
                else:
                    thread_ident = task_manager.active_threads.launch_wait_table_modification_on_db(path_to_bucket, buckets_cfg, client_id,
                                                                      entry.data_base, entry.table_name,
                                                                     timeout, control_mq, zmq_sender_id, cmd.lower(),
                                                                     update_status_callback=update_status_func)
                    task_manager.associate_thread_with_task(thread_ident, task_id)
            elif cmd == TaskCmd.EXPORT_EMBEDDINGS:
                client_id, export_args, zmq_sender_id = payload
                thread_ident = task_manager.active_threads.launch_export_embeddings(client_id, export_args, buckets_cfg, control_mq, zmq_sender_id, cmd.lower(), update_status_callback=update_status_func)
                task_manager.associate_thread_with_task(thread_ident, task_id)

            elif cmd in (TaskCmd.MODEL_REGISTER, TaskCmd.MODEL_LIST):
                client_id, payload_str, zmq_sender_id = payload
                response_json = handle_model_command(cmd, client_id, payload_str)
                try:
                    control_mq.send_string(response_json, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error sending model command response: {e}")
                update_status_func(TaskStatus.COMPLETED)

            elif cmd == TaskCmd.MODEL_DOWNLOAD:
                # Server-side download: reply immediately, run download in background thread
                client_id, payload_str, zmq_sender_id = payload
                ack = handle_model_command(cmd, client_id, payload_str)
                try:
                    control_mq.send_string(ack, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error sending model download ack: {e}")
                # The handler spawns a background thread for the actual download;
                # the ack already told the client it's accepted.
                update_status_func(TaskStatus.COMPLETED)

            elif cmd in (TaskCmd.JOB_SUBMIT, TaskCmd.JOB_CANCEL, TaskCmd.JOB_LIST):
                client_id, payload_str, zmq_sender_id = payload
                response_json = handle_job_command(cmd, client_id, payload_str)
                try:
                    control_mq.send_string(response_json, zmq_sender_id)
                except RuntimeError as e:
                    print(f"Error sending job command response: {e}")
                update_status_func(TaskStatus.COMPLETED)

            else:
                print(f"====> Error: unknown command '{cmd}' returned by server loop")
                task_manager.tasks_status.update_status(task_id, TaskStatus.UNKNOWN)

            finished_threads_ident_list =  task_manager.active_threads.clean_finished_threads()
            print(f"{finished_threads_ident_list=}")
            task_manager.update_status_by_threads_ident_list(finished_threads_ident_list, TaskStatus.COMPLETED)

            # After any event and cleanup, update per-user task status summaries
            try:
                users = set(entry.s3_username for entry in buckets_cfg.values())
                users = {u for u in users if u}
                task_manager.active_threads.record_status_for_users_by_key(users, TaskCmd.PROCESS_OBJECTS)
            except Exception as e:
                # Non-fatal: do not disrupt server loop on status recording errors
                print(f"Warning: failed to update task status: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for running tasks to finish...")
        finished_threads_ident_list = task_manager.active_threads.waitall()
        task_manager.update_status_by_threads_ident_list(finished_threads_ident_list, TaskStatus.COMPLETED)
        task_manager.active_threads.close()
        task_manager.tasks_status.close()
        print("All tasks finished. Exiting now.")
        control_mq.close()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL: Unhandled exception in server event loop: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            task_manager.active_threads.close()
            task_manager.tasks_status.close()
            control_mq.close()
        except Exception:
            pass
        sys.exit(1)

def handle_maintenance_flags(task_manager, args, buckets_cfg, device, embedding_backend, configure_file, server_cfg):
    # do in this order
    if args.force_recompute:
        recompute_all_objects(task_manager.active_threads, None, buckets_cfg, device, embedding_backend, TaskCmd.RECOMPUTE.lower(), log_output=ENABLE_PRETTY_PRINT)
    if args.force_update:
        update_objects(task_manager.active_threads, None, buckets_cfg, device, embedding_backend, TaskCmd.UPDATE.lower(), verbose=False, log_output=ENABLE_PRETTY_PRINT)
    if args.check_objects:
        check_objects(task_manager.active_threads, buckets_cfg, TaskCmd.CHECK.lower(), verbose=False, log_output=ENABLE_PRETTY_PRINT)
    if args.check_incomplete_command:
        incomplete_commands = task_manager.tasks_status.list_incomplete_tasks(mark_override=True)
        for command, info in incomplete_commands.items():
            client_id, cmd, payload = command
            task_id = int(info["id"])
            status = info["status"]

            print(f"==> Recovering {cmd} for {client_id} with status: {status}")

            # create a callback with task_id embedded
            update_status_func = make_update_status_callback(task_manager, task_id)

            if not is_client_configured(client_id, server_cfg):
                print(f"Client {client_id} is not configured. Skipping recovery.")
                update_status_func(TaskStatus.ERROR)
                continue

            if cmd == TaskCmd.PROCESS_OBJECTS:
                s3_objects = json.loads(payload)
                thread_ident = task_manager.active_threads.launch_s3_threads(s3_objects,
                                                                             buckets_cfg, device, embedding_backend,
                                                                             cmd.lower(),
                                                                             update_status_callback=update_status_func)
                task_manager.associate_thread_with_task(thread_ident, task_id)
            elif cmd in (TaskCmd.RECOMPUTE, TaskCmd.UPDATE, TaskCmd.CHECK):
                user_config_map = server_cfg.user_config_map
                thread_ident = _handle_maintenance_command(task_manager.active_threads, cmd, client_id, buckets_cfg,
                                                            device, embedding_backend, configure_file, user_config_map, cmd.lower(),
                                                            update_status_callback=update_status_func)
                if thread_ident is None:
                    update_status_func(TaskStatus.COMPLETED)
                else:
                    task_manager.associate_thread_with_task(thread_ident, task_id)

            elif cmd == TaskCmd.FILE:
                user_config_map = server_cfg.user_config_map
                files = json.loads(payload)
                for file in files:
                    if not os.path.exists(file):
                        print(f"Trying to recover file: {file}, but it no longer exists. Skipping.")
                        continue
                    print(f"Recovering file: {file}")

                    ok, uname, dest = _persist_uploaded_user_config(
                        file, configure_file, user_config_map, expected_username=client_id
                    )
                    if ok:
                        print(f"User config persisted for '{uname}' -> {dest}")
                    else:
                        print(f"Note: file '{file}' was not recognized as a valid user config or there was an identity mismatch; left in place.")

                update_status_func(TaskStatus.COMPLETED)
            else:
                print("\tNothing to do for command of type {cmd}")

def check_umrs_config_directory():
    umrs_config_dir = os.getenv(get_umrs_cfg_env_var_name())

    if umrs_config_dir:
        if os.path.isdir(umrs_config_dir):
            return umrs_config_dir
        else:
            return None
    else:
        default_dir = get_default_umrs_cfg_dir()
        if os.path.isdir(default_dir):
            return default_dir
        else:
            return None
# ---------------------------------- CLI ------------------------------------

def main():
    # check if we have a correct configuration directory
    umrs_cfg_dir = check_umrs_config_directory()
    if umrs_cfg_dir:
        default_json_server_path = f"{umrs_cfg_dir}/{get_default_json_server_file()}"
    else:
        sys.exit(f"No valid UMRS configuration directory. The default directory is {get_default_umrs_cfg_dir()} or the value of the environment variable {get_umrs_cfg_env_var_name()}. Check it exists.")

    parser = argparse.ArgumentParser(description="UMRS server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--conf', required=(default_json_server_path is None), default=default_json_server_path, help=f"Config file path. The default directory is {get_default_umrs_cfg_dir()} or the value of the environment variable {get_umrs_cfg_env_var_name()}.")
    parser.add_argument('--force-recompute', action='store_true', help='Recompute all embeddings by scanning all S3 directories. Existing entries will be fully overwritten.')
    parser.add_argument('--force-update', action='store_true', help='Recompute embeddings only for missing or mismatched entries. Only those entries will be updated in the database.')
    parser.add_argument('--check-objects', action='store_true', help='Check all the buckets to see if the corresponding objects are stored in the database')
    parser.add_argument('--pretty-print', action='store_true', help='Enable server-side pretty-printed tables and summaries (default: off)')
    parser.add_argument('--check-incomplete-command', action='store_true', help='Force checking for incomplete commands after a crash (default: off)')
    args = parser.parse_args()
    configure_file = args.conf

    global ENABLE_PRETTY_PRINT
    ENABLE_PRETTY_PRINT = args.pretty_print

    print(f"Starting UMRS server with configuration file: {configure_file}")
    status_store = diskcache.Cache(f"{get_default_tmp_subdir()}/status_store", size_limit = 500*1024*1024)

    print("Initialized status store")
    thread_dict = diskcache.Cache(f"{get_default_tmp_subdir()}/thread_dict", size_limit = 500*1024*1024)

    print("Initialized thread dictionary")
    task_manager = TaskManager(status_store, thread_dict, f"{get_default_db_subdir()}/tasks.db")

    print("Performing initial configuration...")
    control_mq, server_cfg, device, embedding_backend = perform_initial_config(configure_file, status_store)

    print("Initial configuration done.")
    reconcile_status_on_startup(task_manager.active_threads, server_cfg.buckets_cfg)

    handle_maintenance_flags(task_manager, args, server_cfg.buckets_cfg, device, embedding_backend, configure_file, server_cfg)

    print("Entering main server event loop.")
    cfg_summary = pretty_print_cfg(server_cfg, print_output=ENABLE_PRETTY_PRINT)

    print(f"Server configuration summary:\n{cfg_summary}")
    task_manager.active_threads.record_status("system", "config", {"summary": cfg_summary})

    print("Server is ready to process tasks.")
    server_event_loop(task_manager, control_mq, server_cfg, device, embedding_backend, configure_file)


if __name__ == "__main__":
    main()
