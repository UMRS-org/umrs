"""Canonical server provisioning and configuration helpers."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

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
    arcname = (
        f"{cache_entry.kind}/{relative_posix}" if relative_posix else cache_entry.kind
    )

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(cache_entry.path, arcname=arcname)
        return tar_path
    except Exception as e:
        logger.info(f"Warning: failed to create weights tar for '{model_name}': {e}")
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
        logger.info(f"Warning: failed to check weights endpoint for '{model_name}': {e}")
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
        logger.info(f"Warning: failed to upload weights for '{model_name}': {e}")
        return


def _resolve_s3cfg_path(user_cfg):
    """Resolve the s3cfg path, falling back to a durable shared path if needed.

    When a remote client (e.g. Mac) has an s3cfg path like ``/Users/...`` that
    doesn't exist on the Linux server, resolve to the durable, shared
    ``~/.config/umrs/s3cfg_<username>`` location instead of the configured
    client path.

    Why this matters: a per-user MinIO secret must be created **once** and stay
    identical across reloads, container restarts, and — on umrs-front — across
    both cluster servers (rccs + ai4s) that share a single MinIO. The shared
    location is the same path ``_read_user_s3cfg`` treats as authoritative
    (priority 1) and is backed by the shared ``umrs_config`` volume in the
    umrs-front compose stack.

    The previous behaviour created the *client's* path (e.g. ``/Users/...``)
    on the server when its parent could be made. That path is container-local
    and ephemeral, so every reload/restart re-entered the "first run" branch in
    :func:`minio_s3_create_user`, generated a fresh random secret, and reset the
    shared MinIO user's password — silently invalidating credentials already
    distributed to clients (``SignatureDoesNotMatch``).
    """
    username = user_cfg["username"]
    raw_path = user_cfg["s3_configuration_file"]
    expanded = os.path.expanduser(raw_path)

    # Fast path: the configured path already exists on this server. This covers
    # single-node deployments (e.g. rccs-cloud) where the server *is* the client
    # and the configured path is a real, durable local file.
    if os.path.isfile(expanded):
        return expanded

    # Durable, shared authoritative location. Matches `_read_user_s3cfg`
    # priority-1 and the umrs-front shared `umrs_config` volume, so the secret
    # is stable across reloads, restarts and both cluster servers.
    shared = os.path.expanduser(f"~/.config/umrs/s3cfg_{username}")
    if os.path.isfile(shared):
        return shared

    # Legacy server-local fallback next to the user config file. Kept only so
    # that deployments that already provisioned a credential there keep using
    # the same (stable) file instead of regenerating it.
    config_file_path = user_cfg.get("config_file_path", "")
    if config_file_path:
        server_dir = os.path.dirname(os.path.expanduser(config_file_path))
        legacy_fallback = os.path.join(server_dir, f"s3cfg_{username}")
        if os.path.isfile(legacy_fallback):
            logger.info(
                f"Note: using server-local s3cfg '{legacy_fallback}' "
                f"(client path '{raw_path}' not found on this server)"
            )
            return legacy_fallback

    # Nothing exists yet. If the configured path's parent already exists on this
    # server, the server is acting as its own client — create the file there.
    # Otherwise (remote client path unreachable) create the durable shared path.
    # Never create the remote client's path on the server: it would be
    # container-ephemeral and trigger secret regeneration on the next reload.
    s3cfg_parent = os.path.dirname(expanded) or "."
    if os.path.isdir(s3cfg_parent):
        return expanded

    try:
        os.makedirs(os.path.dirname(shared), exist_ok=True)
    except OSError:
        pass
    logger.info(
        f"Note: client s3cfg path '{raw_path}' not present on this server; "
        f"using durable shared s3cfg '{shared}'"
    )
    return shared


def create_s3_sql_users(user_cfg, admin_s3_configuration_file):
    fullname = user_cfg["fullname"]
    username = user_cfg["username"]
    logger.info(
        f"###################### Creating {fullname} ({username}) ######################"
    )

    bucket_list = []
    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        sql_username = xp["sql_username"]
        create_db_user(sql_username)
        for modality in xp["modalities"]:
            raw_modality_name = modality.get("modality_name") or modality.get(
                "modality"
            )
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

    user_s3_configuration_file = _resolve_s3cfg_path(user_cfg)

    s3_connection, _ = connect_to_s3(admin_s3_configuration_file)
    s3cfg_content = s3_create_users_and_buckets(
        username,
        s3_connection,
        user_s3_configuration_file,
        admin_s3_configuration_file,
        bucket_list,
    )
    s3_connection.close()
    return s3cfg_content


def is_client_configured(client_id, server_cfg):
    return any(
        entry.s3_username == client_id for entry in server_cfg.buckets_cfg.values()
    )


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
            "status": "Done"
            if any(work["nb_objects"] > 0 for work in works)
            else "Idle",
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
    logger.info("Status store reconciled with pgvector counts.")


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
    logger.info(f"\t\tModality: {modality_name} (task: {task})")

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
        logger.info(
            f"Warning: failed to upload weights to RiVault for model '{model_cfg.get('model_name', '?')}': {e}"
        )

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

    logger.info(
        f"######################\nConfiguring {fullname} ({username})\n######################"
    )

    user_s3_configuration_file = _resolve_s3cfg_path(user_cfg)
    s3_connection, bucket_location = connect_to_s3(user_s3_configuration_file)

    for xp in user_cfg["experiences"]:
        xp_name = xp["xp_name"]
        logger.info(f"\t==> Experience: {xp_name}")
        sql_username = xp["sql_username"]
        data_base, _db_name = create_db(sql_username, xp_name)
        for modality in xp["modalities"]:
            try:
                bucket_alias, path_to_bucket, num_objects, bucket_entry = (
                    _configure_modality_bucket(
                        user_cfg,
                        username,
                        xp_name,
                        sql_username,
                        modality,
                        s3_connection,
                        bucket_location,
                        data_base,
                    )
                )
                buckets_cfg.store(path_to_bucket, bucket_entry)
                logger.info(f"\t\t\t==>Bucket alias: {bucket_alias} ({num_objects} objects)")
            except Exception as e:
                import traceback

                mod_name = modality.get("modality_name") or modality.get(
                    "modality", "?"
                )
                logger.info(
                    f"\t\t\tWARNING: Failed to configure modality '{mod_name}' "
                    f"in experience '{xp_name}': {e}"
                )
                traceback.print_exc()


def deprovision_user(user_cfg, admin_s3_configuration_file):
    """Full teardown of a user's provisioned resources.

    Removes:
    - All S3 buckets (input + output) and their contents
    - PostgreSQL databases per experience
    - PostgreSQL roles per experience
    - MinIO IAM user and attached policies
    - User S3 config file (~/.s3cfg_<username>)
    """
    import configparser

    from umrs.server.sql import create_db_name, drop_db, drop_db_user
    from umrs.shared.storage.minio import (
        minio_delete_user,
        minio_remove_bucket,
    )

    username = user_cfg["username"]
    logger.info(f"######################\nDeprovisioning {username}\n######################")

    # Collect all bucket names and SQL usernames.
    bucket_names = []
    sql_usernames = set()
    for xp in user_cfg.get("experiences", []):
        xp_name = xp["xp_name"]
        sql_username = xp["sql_username"]
        sql_usernames.add(sql_username)
        db_name = create_db_name(sql_username, xp_name)

        for modality in xp.get("modalities", []):
            bucket_alias = modality["s3"]["bucket_alias"]
            path_to_bucket = make_path_to_bucket_string(xp_name, bucket_alias, username)
            bucket_names.append(path_to_bucket)
            if modality["s3"].get("need_output_bucket", False):
                output_path = make_path_to_bucket_string(
                    xp_name, bucket_alias, username, output=True
                )
                bucket_names.append(output_path)

        # Drop PostgreSQL database.
        try:
            drop_db(db_name)
        except Exception as e:
            logger.info(f"  Warning: failed to drop database '{db_name}': {e}")

    # Drop PostgreSQL roles.
    for sql_user in sql_usernames:
        try:
            drop_db_user(sql_user)
        except Exception as e:
            logger.info(f"  Warning: failed to drop DB user '{sql_user}': {e}")

    # Remove S3 buckets via MinIO CLI.
    admin_cfg_path = os.path.expanduser(admin_s3_configuration_file)
    if os.path.isfile(admin_cfg_path):
        import configparser as _cp

        cfg = _cp.ConfigParser(inline_comment_prefixes=("#", ";"))
        cfg.read(admin_cfg_path)
        alias = cfg.get("default", "alias", fallback="myminio")

        for bucket in bucket_names:
            try:
                minio_remove_bucket(bucket, alias)
            except Exception as e:
                logger.info(f"  Warning: failed to remove bucket '{bucket}': {e}")

        # Remove MinIO IAM user.
        try:
            minio_delete_user(username, alias)
        except Exception as e:
            logger.info(f"  Warning: failed to remove MinIO user '{username}': {e}")
    else:
        logger.info(
            f"  Warning: admin S3 config not found at '{admin_cfg_path}', skipping S3/MinIO cleanup."
        )

    # Remove user S3 config file.
    from umrs.shared.config import default_s3cfg_path

    user_s3_cfg = os.path.expanduser(
        user_cfg.get("s3_configuration_file", default_s3cfg_path(username))
    )
    if os.path.isfile(user_s3_cfg):
        os.remove(user_s3_cfg)
        logger.info(f"==> Removed S3 config file: {user_s3_cfg}")

    logger.info(f"==> Deprovisioning of '{username}' complete.")


__all__ = [
    "ServerCfg",
    "configure_user",
    "create_s3_sql_users",
    "deprovision_user",
    "is_client_configured",
    "reconcile_status_on_startup",
]
