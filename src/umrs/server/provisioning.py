"""Canonical server provisioning and configuration helpers."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests

from umrs.core.TaskCmdStatus import TaskCmd
from umrs.inference.loading import load_model
from umrs.server.sql import configure_table, create_db, create_db_user
from umrs.shared.buckets import BucketsCfg, ServerBucketEntry
from umrs.shared.cache_paths import find_model_cache
from umrs.shared.modality import resolve_modality_task
from umrs.shared.pretty_print import pretty_print_thread_status
from umrs.shared.storage.s3 import (
    configure_bucket,
    connect_to_s3,
    make_path_to_bucket_string,
    s3_create_users_and_buckets,
)


@dataclass
class ServerCfg:
    """All information related to server configuration."""

    user_config_map: dict = field(default_factory=dict)
    buckets_cfg: BucketsCfg = field(default_factory=BucketsCfg)


def _normalize_rivault_endpoint(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.rstrip("/")
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        raise RuntimeError(
            f"RIVAULT endpoint must include http/https scheme (got '{raw_endpoint}')"
        )
    return endpoint


@lru_cache(maxsize=32)
def _get_rivault_endpoint() -> str | None:
    raw = os.getenv("RIVAULT_ENDPOINT")
    if not raw:
        return None
    return _normalize_rivault_endpoint(raw)


def _make_safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _tar_model_cache(model_name: str, tmp_root: str) -> str | None:
    """Create a tarball containing cached model files for RiVault uploads."""

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
    except Exception as e:
        print(f"Warning: failed to create weights tar for '{model_name}': {e}")
        return None


def _upload_weights_to_rivault(model_name: str, tmp_root: str) -> None:
    endpoint = _get_rivault_endpoint()
    if not endpoint or not model_name:
        return

    safe_name = quote(model_name, safe="")
    weights_url = f"{endpoint}/api/models/{safe_name}/weights"

    try:
        head_resp = requests.head(weights_url, timeout=5)
        if head_resp.status_code == 200:
            return
    except Exception as e:
        print(f"Warning: failed to check weights endpoint for '{model_name}': {e}")
        return

    tar_path = _tar_model_cache(model_name, tmp_root)
    if not tar_path or not os.path.exists(tar_path):
        return

    try:
        with open(tar_path, "rb") as handle:
            files = {"file": ("weights.tar", handle, "application/octet-stream")}
            resp = requests.put(weights_url, files=files, timeout=600)
            resp.raise_for_status()
    except Exception as e:
        print(f"Warning: failed to upload weights for '{model_name}': {e}")
        return


def create_s3_sql_users(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    print(f"######################\nCreating {fullname} ({username})\n######################")

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        sql_username = xp["sql_username"]
        create_db_user(sql_username)
        for modality in xp["modalities"]:
            raw_modality_name = modality.get("modality_name") or modality.get("modality")
            raw_task = modality.get("task")
            resolve_modality_task(raw_modality_name, raw_task)
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            bucket_list.append(path_to_bucket)
            need_output_bucket = modality["s3"].get("need_output_bucket", False)
            if need_output_bucket:
                assert modality["model"]["return_type"] is not None, (
                    f"return_type must be set in 'model' for modality '{modality}' "
                    f"of {fullname} ({username})"
                )
                output_path_to_bucket = make_path_to_bucket_string(
                    xp_name, bucket_alias, username, output=True
                )
                bucket_list.append(output_path_to_bucket)

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, _ = connect_to_s3(admin_s3_configuration_file)
    s3_create_users_and_buckets(
        username,
        s3_connection,
        user_s3_configuration_file,
        admin_s3_configuration_file,
        bucket_list,
    )
    s3_connection.close()


def is_client_configured(client_id, server_cfg):
    return any(entry.s3_username == client_id for entry in server_cfg.buckets_cfg.values())


def reconcile_status_on_startup(active_threads, buckets_cfg):
    """Write initial PROCESS_OBJECTS status entries based on pgvector counts."""

    user_entries: dict[str, list[tuple[str, ServerBucketEntry]]] = {}
    for path_to_bucket, entry in buckets_cfg.items():
        user_entries.setdefault(entry.s3_username, []).append((path_to_bucket, entry))

    for username, entries in user_entries.items():
        synthetic_thread_status = {}
        works = []
        snapshot_time = datetime.utcnow().isoformat() + "Z"
        for path_to_bucket, entry in entries:
            works.append(
                {
                    "user": entry.s3_username,
                    "data_base": entry.db_name,
                    "table_name": entry.table_name,
                    "modality_name": entry.modality_name,
                    "bucket": path_to_bucket,
                    "work_type": TaskCmd.PROCESS_OBJECTS,
                    "nb_objects": entry.num_objects,
                }
            )
        if not works:
            continue
        synthetic_thread_status["startup"] = {
            "thread_id": "snapshot",
            "status": "Done" if any(work["nb_objects"] > 0 for work in works) else "Idle",
            "updated_at": snapshot_time,
            "work": works,
        }
        summary = pretty_print_thread_status(
            synthetic_thread_status,
            running_only=False,
            print_output=False,
        )
        json_safe = active_threads._json_safe_thread_status(synthetic_thread_status)
        active_threads.record_status(
            username,
            TaskCmd.PROCESS_OBJECTS,
            {"summary": summary, "data": json_safe, "last_updated_at": snapshot_time},
        )
        active_threads.record_status(
            username,
            TaskCmd.PROCESS_OBJECTS,
            {"summary": summary, "data": json_safe, "last_updated_at": snapshot_time},
        )
    print("Status store reconciled with pgvector counts.")


def _configure_modality_bucket(
    user_cfg,
    username,
    xp_name,
    sql_username,
    modality,
    s3_connection,
    bucket_location,
    data_base,
):
    raw_modality_name = modality.get("modality_name") or modality.get("modality")
    raw_task = modality.get("task")
    modality_name, task = resolve_modality_task(raw_modality_name, raw_task)
    print(f"\t\tModality: {modality_name} (task: {task})")

    bucket_alias = modality["s3"]["bucket_alias"]
    need_output_bucket = modality["s3"].get("need_output_bucket", False)
    path_to_bucket = configure_bucket(
        s3_connection, bucket_alias, xp_name, username, bucket_location
    )

    model_cfg = modality["model"]
    model = load_model(model_cfg, modality_name, task=task)
    try:
        _upload_weights_to_rivault(
            model_cfg.get("model_name"),
            os.path.dirname(user_cfg["config_file_path"]),
        )
    except Exception as e:
        print(f"Warning: failed to upload weights to RiVault for model '{model_cfg.get('model_name', '?')}': {e}")

    if model.return_type == "file":
        assert need_output_bucket
        output_path_to_bucket = configure_bucket(
            s3_connection,
            bucket_alias,
            xp_name,
            username,
            bucket_location,
            output=True,
        )
    else:
        output_path_to_bucket = None

    sql_cfg = modality["sql"]
    table_name, store_embeddings_flag, metadata = configure_table(
        data_base, sql_cfg, modality_name, model
    )
    num_objects = data_base.count_s3_objects(table_name)
    bucket_entry = ServerBucketEntry(
        bucket_alias,
        username,
        sql_username,
        data_base,
        data_base.db_name,
        table_name,
        model,
        store_embeddings_flag,
        modality_name,
        task,
        s3_connection,
        metadata,
        xp_name,
        num_objects,
        model.return_type,
        output_path_to_bucket,
        threading.RLock(),
    )
    return bucket_alias, path_to_bucket, num_objects, bucket_entry


def configure_user(user_cfg, server_cfg: ServerCfg):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]

    server_cfg.user_config_map[username] = user_cfg["config_file_path"]
    buckets_cfg = server_cfg.buckets_cfg

    print(f"######################\nConfiguring {fullname} ({username})\n######################")

    user_s3_configuration_file = user_cfg["s3_configuration_file"]
    s3_connection, bucket_location = connect_to_s3(user_s3_configuration_file)

    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        print(f"\t==> Experience: {xp_name}")
        sql_username = xp["sql_username"]
        data_base, _db_name = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            try:
                bucket_alias, path_to_bucket, num_objects, bucket_entry = _configure_modality_bucket(
                    user_cfg,
                    username,
                    xp_name,
                    sql_username,
                    modality,
                    s3_connection,
                    bucket_location,
                    data_base,
                )
                buckets_cfg.store(path_to_bucket, bucket_entry)
                print(f"\t\t\t==>Bucket alias: {bucket_alias} ({num_objects} objects)")
            except Exception as e:
                import traceback
                mod_name = modality.get("modality_name") or modality.get("modality", "?")
                print(f"\t\t\tWARNING: Failed to configure modality '{mod_name}' "
                      f"in experience '{xp_name}': {e}")
                traceback.print_exc()


__all__ = [
    "ServerCfg",
    "configure_user",
    "create_s3_sql_users",
    "is_client_configured",
    "reconcile_status_on_startup",
]
