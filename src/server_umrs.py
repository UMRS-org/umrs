#!/usr/bin/python3

import argparse
import os
import sys
from collections import defaultdict
import time
import torch
import json
from diskcache import Cache
from functools import lru_cache
from typing import Optional
from urllib.parse import quote
from pathlib import Path

import requests

# Add the parent directory of this script (src) to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))



from umrs.s3 import *
from umrs.sql import create_db_user, create_db, configure_table, check_S3_object_status_in_db
from umrs.models_loading import load_model
from umrs.MessageQueue import MessageQueue
from umrs.pretty_print import pretty_print_cfg, pretty_print_recompute, pretty_print_check, pretty_print_thread_status
from umrs.modality import check_or_quit
from umrs.db_store import store_db_model, query_db_model

from umrs.json_utils import load_server_cfg_file, load_client_cfg_file
from umrs.config import *
from umrs.UMRSTasks import UMRSTasks
from umrs.embedding_backend import LocalEmbeddingBackend, create_embedding_backend_from_config
from umrs.cache_paths import find_model_cache

_status_store_global = None
ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag


# ---------------------- Provisioning and configuration ----------------------


def _normalize_rivault_endpoint(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.rstrip("/")
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        raise RuntimeError(
            f"RIVAULT endpoint must include http/https scheme (got '{raw_endpoint}')"
        )
    return endpoint


@lru_cache(maxsize=32)
def _get_rivault_endpoint() -> Optional[str]:
    raw = os.getenv("RIVAULT_ENDPOINT")
    if not raw:
        return None
    return _normalize_rivault_endpoint(raw)


def _make_safe_model_name(model_name: str) -> str:
    # Keep behaviour in sync with rivault_service: replace '/' by '__'
    return model_name.replace("/", "__")


def _tar_model_cache(model_name: str, tmp_root: str) -> Optional[str]:
    """Create a tarball containing the cached model files for RiVault uploads."""

    import tarfile

    cache_entry = find_model_cache(model_name)
    if not cache_entry:
        return None

    safe_name = _make_safe_model_name(model_name)
    out_dir = os.path.join(tmp_root, "umrs_models", safe_name)
    os.makedirs(out_dir, exist_ok=True)
    tar_path = os.path.join(out_dir, "weights.tar")

    relative_posix = Path(cache_entry.relative_path).as_posix().lstrip("./")
    arcname = f"{cache_entry.kind}/{relative_posix}" if relative_posix else cache_entry.kind

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(cache_entry.path, arcname=arcname)
        return tar_path
    except Exception:
        # Best-effort: if we cannot create the archive, just skip
        return None


def _upload_weights_to_rivault(model_name: str, tmp_root: str) -> None:
    endpoint = _get_rivault_endpoint()
    if not endpoint:
        return

    safe_name = quote(model_name, safe="")
    weights_url = f"{endpoint}/api/models/{safe_name}/weights"

    # If weights already exist on RiVault, do nothing
    try:
        head_resp = requests.head(weights_url, timeout=5)
        if head_resp.status_code == 200:
            return
    except Exception:
        # If RiVault is unreachable, keep normal behaviour
        return

    tar_path = _tar_model_cache(model_name, tmp_root)
    if not tar_path or not os.path.exists(tar_path):
        return

    try:
        with open(tar_path, "rb") as handle:
            files = {"file": ("weights.tar", handle, "application/octet-stream")}
            resp = requests.put(weights_url, files=files, timeout=600)
            resp.raise_for_status()
    except Exception:
        # Weight upload is best-effort; failures must not break server config
        return

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
            model_cfg = modality['model']
            model = load_model(model_cfg, modality_name)
            # Best-effort upload of model weights to RiVault so that
            # compute workers can later stream them from there.
            try:
                _upload_weights_to_rivault(model_cfg.get("model_name"), os.path.dirname(user_conf_file))
            except Exception:
                # Never fail user configuration because of RiVault upload issues
                pass
            sql_cfg = modality["sql"]
            table_name, store_embeddings_flag, metadata = configure_table(data_base, sql_cfg, modality_name, model)
            store_db_model(db_model_store, path_to_bucket, bucket_alias, username, data_base, db_name, table_name, sql_username, model, store_embeddings_flag, modality_name, s3_connection, metadata, xp_name)
            print(f"\t\t\t==>Bucket alias: {bucket_alias}")





# --------------------------- Maintenance commands ---------------------------

def recompute_all_objects(active_tasks, db_model_store, device, embedding_backend, log_output=True):
    if log_output:
        print(f"\n######################\nRecomputing all objects in all buckets\n######################\n")

    recompute_dict = {}

    for path_to_bucket, entry in db_model_store.items():
        s3_objects, total_size, _ = list_s3_bucket(entry["s3_connection"], path_to_bucket)
        if len(s3_objects) > 0:
            start_time = time.perf_counter()
            process_s3_objects(active_tasks, s3_objects, db_model_store, device, embedding_backend)
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


def update_objects(active_tasks, db_model_store, device, embedding_backend, verbose=False, log_output=True):
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
            process_s3_objects(active_tasks, to_update_objects, db_model_store, device, embedding_backend)
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



def _handle_maintenance_command(active_tasks, cmd, client_id, db_model_store, device, embedding_backend,
                                configure_file, user_config_map):
    if cmd == "RECOMPUTE":
        outcome = recompute_all_objects(active_tasks, db_model_store, device, embedding_backend, log_output=ENABLE_PRETTY_PRINT)
        key = "recompute"
    elif cmd == "UPDATE":
        outcome = update_objects(active_tasks, db_model_store, device, embedding_backend, log_output=ENABLE_PRETTY_PRINT)
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
    active_tasks.record_status(client_id, key, filtered_outcome)



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

def perform_initial_config(configure_file, status_store):
    if not os.path.exists(configure_file):
        sys.exit(f"Configuration file '{configure_file}' not found.")

    admin_s3_configuration_file, users, server_cfg = load_server_cfg_file(configure_file)
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

    cfg_store = defaultdict(dict)
    for user in users:
        configure_user(user, cfg_store)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"-->  Models will be run on device: {device}")

    # Select embedding backend based on configuration (local or remote).
    try:
        embedding_backend = create_embedding_backend_from_config(server_cfg)
        backend_name = type(embedding_backend).__name__
        print(f"-->  Using embedding backend: {backend_name}")
    except Exception as exc:
        print(f"Error while creating embedding backend from config, falling back to local backend: {exc}")
        embedding_backend = LocalEmbeddingBackend()
    print(f"\n######################\nServer configuration done!\n######################\n")

    return control_mq, cfg_store, device, embedding_backend


def reload_configuration(configure_file, db_model_store, user_config_map, log_output=True):
    if log_output:
        print("\n######################\nReloading configuration...\n######################\n")
    try:
        new_admin_s3_cfg, new_users, _ = load_server_cfg_file(configure_file)
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



def process_s3_objects(active_tasks, s3_objects, db_model_store, device, embedding_backend):
    nb_objects = len(s3_objects)
    keys_per_bucket = defaultdict(list)
    users = set()
    print(f"\t-->Detected {nb_objects} new file(s):")

    for s3_object in s3_objects:
        print(f"\t\t-->{s3_object}")
        path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
        if s3_key not in keys_per_bucket[path_to_bucket]:
            keys_per_bucket[path_to_bucket].append(s3_key)

    active_tasks.update_thread_dict(db_model_store, keys_per_bucket)

    for path_to_bucket in keys_per_bucket.keys():
        user, data_base, table_name, model, store_embeddings_flag, modality_name, s3_connection, _ = query_db_model(db_model_store, path_to_bucket)
        users.add(user)
        s3_keys = keys_per_bucket[path_to_bucket]
        if store_embeddings_flag:
            print(f"\t--> Storing to data base in {table_name}")
            embedding_backend.compute_and_store(
                path_to_bucket,
                s3_keys,
                data_base,
                table_name,
                model,
                device,
                modality_name,
                s3_connection,
            )
        else:
            print(f"Error: We do not compute embeddings for this bucket \"{path_to_bucket}\" (modality = \"{modality_name}\")\n\tUser is probably not configured")

    active_tasks.update_status_to_done()
    active_tasks.record_tasks_status_for_users(list(users))


    print(f"\t--> Processing {nb_objects} file(s) done!")
    with active_tasks.lock:
        pretty_print_thread_status(active_tasks.thread_dict, running_only=False, print_output=ENABLE_PRETTY_PRINT)
    print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")


def server_event_loop(active_tasks, control_mq, cfg_store, device, embedding_backend, configure_file):
    db_model_store = cfg_store['db_model_store']
    user_config_map = cfg_store['user_config_map']
    status_store = control_mq.get_status_store()
    assert status_store is not None
    assert status_store == active_tasks.get_status_store()

    try:
        while True:
            print("####### Waiting for new objects in the S3 bucket (Press CTRL+C to exit)... #######")
            result = control_mq.server_control_queue_loop()
            if not result:
                continue
            cmd, payload = result

            if cmd == "OBJECT":
                active_tasks.launch_s3_tasks(payload, db_model_store, device, embedding_backend)
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
                _handle_maintenance_command(active_tasks, cmd, client_id, db_model_store,
                                            device, embedding_backend, configure_file, user_config_map)
            else:
                print(f"====> Error: unknown command '{cmd}' returned by server loop")

            active_tasks.clean_finished_tasks()
            # After any event and cleanup, update per-user task status summaries
            try:
                users = set(entry.get('s3_username') for entry in db_model_store.values())
                users = {u for u in users if u}
                active_tasks.record_tasks_status_for_users(users)
            except Exception as e:
                # Non-fatal: do not disrupt server loop on status recording errors
                print(f"Warning: failed to update task status: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Waiting for running tasks to finish...")
        active_tasks.waitall()
        active_tasks.close()
        print("All tasks finished. Exiting now.")
        control_mq.close()
        sys.exit(0)

def handle_maintenance_flags(active_tasks, args, db_model_store, device, embedding_backend):
    if args.force_recompute:
        recompute_all_objects(active_tasks, db_model_store, device, embedding_backend, log_output=ENABLE_PRETTY_PRINT)
    if args.force_update:
        update_objects(active_tasks, db_model_store, device, embedding_backend, log_output=ENABLE_PRETTY_PRINT)
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
    ENABLE_PRETTY_PRINT = args.pretty_print

    status_store = Cache(f"{get_default_tmp_subdir()}/status_store", size_limit = 500*1024*1024)
    thread_dict = Cache(f"{get_default_tmp_subdir()}/thread_dict", size_limit = 500*1024*1024)
    active_tasks = UMRSTasks(status_store, thread_dict)

    control_mq, cfg_store, device, embedding_backend = perform_initial_config(configure_file, status_store)
    handle_maintenance_flags(active_tasks, args, cfg_store['db_model_store'], device, embedding_backend)
    cfg_summary = pretty_print_cfg(cfg_store, print_output=ENABLE_PRETTY_PRINT)
    active_tasks.record_status("system", "config", {"summary": cfg_summary})
    server_event_loop(active_tasks, control_mq, cfg_store, device, embedding_backend, configure_file)


if __name__ == "__main__":
    main()
