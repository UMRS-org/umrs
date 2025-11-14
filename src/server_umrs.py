#!/usr/bin/python3
import threading
import argparse
import os
import sys
from collections import defaultdict
import time
from datetime import datetime
import torch
import json


# Add the parent directory of this script (src) to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from umrs.sql import *
from umrs.s3 import *
from umrs.models_loading import load_model
from umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check, pretty_print_global_thread_status
from umrs.modality import check_or_quit
from umrs.db_store import store_db_model, query_db_model
from umrs.utils import zero_out_embeddings_for_empty_objects
from umrs.json_utils import load_server_cfg_file, load_client_cfg_file
from umrs.config import *

# a global dictionnary for easly handling thread status
global_thread_status = {}

# Module-level reference to status store so we can update client-visible TASKS status
_status_store_global = None
ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag

def _set_status_store_global(store):
    global _status_store_global
    _status_store_global = store

def _filter_thread_status_for_user(client_id):
    """Build a filtered view of global_thread_status containing only works for a given client_id."""
    filtered = {}
    for thread, info in global_thread_status.items():
        works = info.get("work", [])
        user_works = [w for w in works if w.get("user") == client_id]
        if user_works:
            # Shallow copy with filtered works
            filtered[thread] = {
                "thread_id": info.get("thread_id"),
                "status": info.get("status"),
                "work": user_works,
            }
    return filtered

def _json_safe_thread_status(ts_dict):
    """Convert thread-keyed dict into a JSON-serializable dict keyed by thread_id strings."""
    safe = {}
    for _thread, info in ts_dict.items():
        tid = info.get("thread_id")
        key = str(tid) if tid is not None else "thread"
        safe[key] = {
            "thread_id": info.get("thread_id"),
            "status": info.get("status"),
            "work": info.get("work", []),
        }
    return safe

def _record_tasks_status_for_users(users):
    """Record 'tasks' status summaries for the provided user list into the global status store."""
    if _status_store_global is None:
        return
    for user in users:
        filtered = _filter_thread_status_for_user(user)
        # Include both Running and Done tasks, preserving status column
        summary = pretty_print_global_thread_status(filtered, running_only=False, print_output=False)
        json_safe = _json_safe_thread_status(filtered)
        _record_status(_status_store_global, user, "tasks", {"summary": summary, "data": json_safe})

# ---------------------- Provisioning and configuration ----------------------

def create_s3_sql_users(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        sql_username  = xp["sql_username"]
        create_db_user(sql_username)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = create_path_to_bucket(xp_name, bucket_alias, username)
            bucket_list.append(path_to_bucket)

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, _ = connect_to_s3(admin_s3_configuration_file)
    s3_create_users_and_buckets(username, s3_connection, user_s3_configuration_file, admin_s3_configuration_file, bucket_list)
    s3_connection.close()


def configure_user(user_cfg, cfg_store):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    user_conf_file = user_cfg['config_file_path']

    if 'user_config_map' not in cfg_store:
        cfg_store['user_config_map'] = {}

    user_config_map = cfg_store['user_config_map']
    user_config_map[username] = user_conf_file

    if 'db_model_store' not in cfg_store:
        cfg_store['db_model_store'] = {}

    db_model_store = cfg_store['db_model_store']

    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, bucket_location = connect_to_s3(user_s3_configuration_file)

    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username  = xp["sql_username"]
        data_base, db_name = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            modality_name = modality["modality_name"]
            check_or_quit(modality_name)
            print(f"\t\tModality: {modality_name}")
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = configure_bucket(s3_connection, bucket_alias, xp_name, username, bucket_location)
            model = load_model(modality['model'], modality_name)
            sql_cfg = modality["sql"]
            table_name, store_embeddings_flag, metadata = configure_table(data_base, sql_cfg, modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")


# --------------------------- Embedding operations ---------------------------

def compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection):
    if modality_name in ["tomography"]: # case of models where we compute everything from the files (e.g. using data loader)
        embeddings, checksums, metadata = model.compute_embeddings(s3_connection, s3_keys, path_to_bucket)
    else: # case where we need to load data into a tensor and then compute embeddings
        objects, metadata, checksums, sizes, has_at_least_a_zero = load_data_from_s3(s3_connection, path_to_bucket, s3_keys, modality_name)
        embeddings = model.compute_embeddings(objects)
        if has_at_least_a_zero:
            zero_out_embeddings_for_empty_objects(embeddings, sizes, modality_name)

    assert(len(embeddings) == len(checksums))
    assert(len(embeddings) == len(metadata))

    store_embeddings_in_db(data_base, embeddings, table_name, path_to_bucket, s3_keys, modality_name, checksums, model.model_name, metadata_list=metadata)

def update_global_thread_status(db_model_store, keys_per_bucket, thread):
    thread_id = len(global_thread_status)
    array_of_works = []
    for path_to_bucket in keys_per_bucket.keys():
        user, _, table_name, _, _, modality_name, _, data_base = query_db_model(db_model_store, path_to_bucket)
        s3_keys = keys_per_bucket[path_to_bucket]
        work = {
            "user": user,
            "data_base": data_base,
            "table_name": table_name,
            "modality_name": modality_name,
            "bucket": path_to_bucket,
            "nb_objects": len(s3_keys),
        }
        array_of_works.append(work)
    global_thread_status[thread] = {
        "thread_id" : thread_id,
        "status": "Running",
        "work": array_of_works,
    }

def process_s3_objects(s3_objects, db_model_store, device):
    current_thread = threading.current_thread()
    nb_objects = len(s3_objects)
    keys_per_bucket = defaultdict(list)
    print(f"\t-->Detected {nb_objects} new file(s):")

    for s3_object in s3_objects:
        print(f"\t\t-->{s3_object}")
        path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
        if s3_key not in keys_per_bucket[path_to_bucket]:
            keys_per_bucket[path_to_bucket].append(s3_key)

    update_global_thread_status(db_model_store, keys_per_bucket, current_thread)

    for path_to_bucket in keys_per_bucket.keys():
        _, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection, _ = query_db_model(db_model_store, path_to_bucket)
        s3_keys = keys_per_bucket[path_to_bucket]
        if store_embeddings_flag:
            print(f"\t--> Storing to data base in {table_name}")
            compute_and_store_embedding(path_to_bucket, s3_keys, data_base, table_name, model, device, modality_name, s3_connection)
        else:
            print(f"Error: We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")\n\tUser is probably not configured")

    global_thread_status[current_thread]["status"] = "Done"
    print(f"\t--> Processing {nb_objects} file(s) done!")
    pretty_print_global_thread_status(global_thread_status, running_only=False, print_output=ENABLE_PRETTY_PRINT)
    print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")


def wait_for_key(d, key, check_interval=0.05):
    while key not in d:
        time.sleep(check_interval)  # wait for 50ms before checking again
    # key is now valid in dictionary

def launch_s3_thread(s3_objects, db_model_store, device):
    thread = threading.Thread(target=process_s3_objects, args=(s3_objects, db_model_store, device))
    thread.start()
    wait_for_key(global_thread_status, thread) # wait for th thread status to be updated
    return thread


def clean_finished_threads(threads):
    new_threads = []
    for thread in threads:
        if not thread.is_alive():
            thread.join()
        else:
            new_threads.append(thread)
    print (f"==> Running threads: {len(new_threads)}")
    pretty_print_global_thread_status(global_thread_status, running_only=True, print_output=ENABLE_PRETTY_PRINT)
    return new_threads


# --------------------------- Maintenance commands ---------------------------

def recompute_all_objects(db_model_store, device, log_output=True):
    if log_output:
        print(f"\n######################\nRecomputing all objects in all buckets\n######################\n")

    recompute_dict = {}

    for path_to_bucket, entry in db_model_store.items():
        s3_objects, total_size, _ = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        if len(s3_objects) > 0:
            start_time = time.perf_counter()
            process_s3_objects(s3_objects, db_model_store, device)
            duration = round((time.perf_counter() - start_time),3)
        else:
            duration = 0

        recompute_dict[path_to_bucket] = {
            "bucket_alias": entry["bucket_alias"],
            "duration": duration,
            "nb_objects": len(s3_objects),
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"]
        }

    summary = pretty_print_recompute(recompute_dict, print_output=False)
    if log_output:
        print(summary)
    return {"summary": summary, "data": recompute_dict}


def update_objects(db_model_store, device, verbose=False, log_output=True):
    if log_output:
        print(f"\n######################\nUpdating missing/mismatched objects in all buckets\n######################\n")

    update_dict = {}

    for path_to_bucket, entry in db_model_store.items():
        s3_objects, total_size, checksums = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, db_model_store)

        to_update_objects = []
        for i, s3_object in enumerate(s3_objects):
            s = status[i]['status']
            if verbose:
                print(f"{s3_object}: {s}")
            if s != "match":
                to_update_objects.append(s3_object)

        if len(to_update_objects) > 0:
            start_time = time.perf_counter()
            process_s3_objects(to_update_objects, db_model_store, device)
            duration = round(time.perf_counter() - start_time, 3)
        else:
            duration = 0.0

        update_dict[path_to_bucket] = {
            "bucket_alias": entry["bucket_alias"],
            "duration": duration,
            "nb_objects": f"{len(to_update_objects)}/{len(s3_objects)}",
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"]
        }

    summary = pretty_print_recompute(update_dict, update=True, print_output=False)
    if log_output:
        print(summary)
    return {"summary": summary, "data": update_dict}


def check_objects(db_model_store, verbose=False, log_output=True):
    if log_output:
        print(f"\n######################\nChecking all objects in all buckets\n######################\n")

    perform_update = False
    check_dict = {}
    for path_to_bucket, entry in db_model_store.items():
        nb_not_found, nb_match, nb_mismatch, nb_error = 0, 0, 0, 0
        s3_objects, total_size, checksums = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        status = check_S3_object_status_in_db(path_to_bucket, s3_objects, checksums, db_model_store)
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
            "bucket_alias": entry["bucket_alias"],
            "nb_objects": len(s3_objects),
            "modality_name": entry["modality_name"],
            "s3_username": entry["s3_username"],
            "total_size_MB": round(total_size / (1024**2), 3),
            "model": entry["model"],
            "matches": nb_match,
            "Mismatches": nb_mismatch,
            "Not_found": nb_not_found,
            "Errors": nb_error
        }
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
    return {"summary": summary_with_note, "data": check_dict, "requires_update": perform_update}


def _record_status(status_store, client_id, key, payload):
    if status_store is None:
        return

    owner = client_id or "system"
    entry = {
        "summary": payload.get("summary", ""),
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }
    if "data" in payload:
        entry["data"] = _sanitize_status_data(payload["data"])
    if "requires_update" in payload:
        entry["requires_update"] = payload["requires_update"]
    if "error" in payload:
        entry["error"] = payload["error"]

    status_store.setdefault(owner, {})[key] = entry


def _handle_maintenance_command(cmd, client_id, db_model_store, device, configure_file, user_config_map, status_store):
    if cmd == "RECOMPUTE":
        outcome = recompute_all_objects(db_model_store, device, log_output=ENABLE_PRETTY_PRINT)
        key = "recompute"
    elif cmd == "UPDATE":
        outcome = update_objects(db_model_store, device, log_output=ENABLE_PRETTY_PRINT)
        key = "update"
    elif cmd == "CHECK":
        outcome = check_objects(db_model_store, verbose=False, log_output=ENABLE_PRETTY_PRINT)
        key = "check"
    elif cmd == "RELOAD":
        outcome = reload_configuration(configure_file, db_model_store, user_config_map, log_output=ENABLE_PRETTY_PRINT)
        key = "reload"
    else:
        print(f"====> Error: unknown command '{cmd}' returned by maintenance handler")
        return

    # Ensure only the requesting user's data is stored and later visible via STATUS
    filtered_outcome = _filter_outcome_for_client(key, outcome, client_id, db_model_store)
    _record_status(status_store, client_id, key, filtered_outcome)


def _sanitize_status_data(data):
    if isinstance(data, dict):
        return {k: _sanitize_status_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_sanitize_status_data(item) for item in data]
    if hasattr(data, "model_name"):
        return getattr(data, "model_name")
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    return str(data)


def _filter_dict_by_user(data_dict, client_id):
    """Return a shallow copy of data_dict entries that belong to client_id (match on 's3_username')."""
    if not isinstance(data_dict, dict):
        return {}
    return {k: v for k, v in data_dict.items() if isinstance(v, dict) and v.get('s3_username') == client_id}


def _filter_outcome_for_client(key, outcome, client_id, db_model_store):
    """Filter an operation outcome so it only contains information for the requesting client_id.

    For recompute/update/check: filter the 'data' dict by s3_username then rebuild the summary.
    For reload: build a per-user configuration summary from the current db_model_store.
    """
    try:
        if key in ("recompute", "update"):
            full = outcome.get("data", {}) if isinstance(outcome, dict) else {}
            filtered = _filter_dict_by_user(full, client_id)
            summary = pretty_print_recompute(filtered, update=(key == "update"), print_output=False)
            return {"summary": summary, "data": filtered}

        if key == "check":
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
            return {"summary": summary, "data": filtered, "requires_update": requires_update}

        if key == "reload":
            # Derive a per-user configuration view from the live db_model_store
            filtered_store = {k: v for k, v in db_model_store.items() if v.get('s3_username') == client_id}
            summary = pretty_print_cfg({"db_model_store": filtered_store}, print_output=False)
            return {"summary": summary}
    except Exception:
        # If filtering fails for any reason, fall back to original outcome but do not raise
        pass
    return outcome


# --------------------------- Boot and hot-reload ----------------------------

def perform_initial_config(configure_file):
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")

    admin_s3_configuration_file, users = load_server_cfg_file(configure_file)
    print("Configuration loaded successfully.")

    # Get the directory containing configure_file
    umrs_dir = os.path.dirname(configure_file)

    # Define the temp directory path inside that directory
    tmp_dir = os.path.join(umrs_dir, get_default_tmp_subdir())

    # Create the tmp_dir if it does not exist
    os.makedirs(tmp_dir, exist_ok=True)
    control_mq = MessageQueue(role='server', queue_type='control', timeout_ms=1000, tmp_dir = tmp_dir)
    # Align MessageQueue debug with server pretty-print setting (off by default)
    try:
        control_mq.set_debug(ENABLE_PRETTY_PRINT)
    except Exception:
        pass
    print ("Subscribed to zeroMQ message queue")

    for user in users:
        create_s3_sql_users(user, admin_s3_configuration_file)

    cfg_store = defaultdict(dict)
    cfg_store['status_store'] = {}
    control_mq.set_status_store(cfg_store['status_store'])
    _set_status_store_global(cfg_store['status_store'])
    for user in users:
        configure_user(user, cfg_store)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")
    print(f"\n######################\nServer configuration done!\n######################\n")

    return control_mq, cfg_store, device


def reload_configuration(configure_file, db_model_store, user_config_map, log_output=True):
    if log_output:
        print("\n######################\nReloading configuration...\n######################\n")
    try:
        new_admin_s3_cfg, new_users = load_server_cfg_file(configure_file)
        for u in new_users:
            create_s3_sql_users(u, new_admin_s3_cfg)
        new_store = defaultdict(dict)
        for u in new_users:
            configure_user(u, new_store)
        for _, entry in db_model_store.items():
            try:
                conn = entry.get("s3_connection")
                if conn:
                    conn.close()
            except Exception:
                pass

        db_model_store.clear()
        db_model_store.update(new_store['db_model_store'])
        user_config_map.clear()
        user_config_map.update(new_store['user_config_map'])
        summary = pretty_print_cfg(new_store, print_output=False)
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
        umrs_dir = os.path.dir(configure_file)
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


def server_event_loop(control_mq, cfg_store, device, configure_file):
    active_threads = []
    db_model_store = cfg_store['db_model_store']
    user_config_map = cfg_store['user_config_map']
    status_store = cfg_store.setdefault('status_store', {})
    control_mq.set_status_store(status_store)
    _set_status_store_global(status_store)
    try:
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")
            result = control_mq.server_control_queue_loop()
            if not result:
                continue
            cmd, payload = result

            if cmd == "OBJECT":
                thread = launch_s3_thread(payload, db_model_store, device)
                active_threads.append(thread)
            elif cmd == "FILE":
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
            elif cmd in ("RECOMPUTE", "UPDATE", "CHECK", "RELOAD"):
                client_id = payload
                _handle_maintenance_command(cmd, client_id, db_model_store, device, configure_file, user_config_map, status_store)
            else:
                print(f"====> Error: unknown command '{cmd}' returned by server loop")

            active_threads = clean_finished_threads(active_threads)
            # After any event and cleanup, update per-user task status summaries
            try:
                users = set(entry.get('s3_username') for entry in db_model_store.values())
                users = {u for u in users if u}
                _record_tasks_status_for_users(users)
            except Exception:
                # Non-fatal: do not disrupt server loop on status recording errors
                pass
    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for running threads to finish...")
        for thread in active_threads:
            thread.join()
        print("All threads finished. Exiting now.")
        control_mq.close()
        sys.exit(0)

def handle_maintenance_flags(args, db_model_store, device):
    if args.force_recompute:
        recompute_all_objects(db_model_store, device, log_output=ENABLE_PRETTY_PRINT)
    if args.force_update:
        update_objects(db_model_store, device, log_output=ENABLE_PRETTY_PRINT)
    if args.check_objects:
        check_objects(db_model_store, verbose=False, log_output=ENABLE_PRETTY_PRINT)

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

    args = parser.parse_args()
    configure_file = args.conf

    global ENABLE_PRETTY_PRINT
    ENABLE_PRETTY_PRINT = bool(getattr(args, 'pretty_print', False))

    control_mq, cfg_store, device = perform_initial_config(configure_file)
    handle_maintenance_flags(args, cfg_store['db_model_store'], device)
    cfg_summary = pretty_print_cfg(cfg_store, print_output=ENABLE_PRETTY_PRINT)
    _record_status(cfg_store.get('status_store'), "system", "config", {"summary": cfg_summary})
    server_event_loop(control_mq, cfg_store, device, configure_file)


if __name__ == "__main__":
    main()
