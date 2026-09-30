"""S3 object processing pipeline: embedding computation and storage."""

import logging
import os
import threading
from collections import defaultdict

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.shared.pretty_print import pretty_print_thread_status

logger = logging.getLogger(__name__)

# Module-level lock for rivault model launches (not re-entrant)
rivault_lock = threading.Lock()

_SELECTED_FIELDS = [
    "s3_username", "data_base", "table_name", "model",
    "store_embeddings_flag", "modality_name", "task",
    "s3_connection", "output_path_to_bucket", "reentrant_lock",
    "keep_endpoint",
]


def _build_bucket_membership(client_entries):
    """Per-bucket key lists + client membership from raw client entries."""
    keys_per_bucket: dict[str, list] = {}
    bucket_to_clients: dict[str, set] = defaultdict(set)
    for client_id, s3_objects in client_entries:
        for s3_object in s3_objects:
            logger.debug("\t\t-->%s", s3_object)
            path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
            bucket_keys = keys_per_bucket.setdefault(path_to_bucket, [])
            if s3_key not in bucket_keys:
                bucket_keys.append(s3_key)
            bucket_to_clients[path_to_bucket].add(client_id)
    return keys_per_bucket, bucket_to_clients


def _resolve_model_groups(keys_per_bucket, bucket_to_clients, buckets_cfg,
                          device, force_recompute):
    """Bucket-config resolution → (model_groups, bucket_errors).

    Buckets with no config, or without an embeddings-warranting model, are
    skipped (num_objects refreshed) instead of entering a group.
    """
    model_groups: dict[str, list] = {}
    bucket_errors: set[str] = set()
    for path_to_bucket, s3_keys in keys_per_bucket.items():
        result = buckets_cfg.get(path_to_bucket, filter=_SELECTED_FIELDS)
        if result is None:
            logger.warning("No bucket config for '%s', skipping", path_to_bucket)
            bucket_errors.add(path_to_bucket)
            continue
        slot = _pack_model_group_slot(
            result, path_to_bucket, s3_keys, device, buckets_cfg, force_recompute,
            bucket_to_clients[path_to_bucket],
        )
        if slot is None:
            continue
        model_name = getattr(slot["model"], "model_name", str(slot["model"]))
        model_groups.setdefault(model_name, []).append(slot)
    return model_groups, bucket_errors


def _pack_model_group_slot(result, path_to_bucket, s3_keys, device, buckets_cfg,
                           force_recompute, clients):
    """One bucket's resolved config → its model-group slot, or None to skip."""
    (user, data_base, table_name, model, store_embeddings_flag,
     modality_name, task, s3_connection, output_bucket, reentrant_lock,
     keep_endpoint) = result
    should_process = (
        store_embeddings_flag
        or (model is not None and getattr(model, "return_type", None) == "file")
        or (model is not None and getattr(model, "type", None) in ("user_script", "user_workflow"))
    )
    if not should_process:
        logger.info("No embeddings configured for bucket '%s', skipping", path_to_bucket)
        buckets_cfg.update_num_objects(path_to_bucket)
        return None
    return {
        "path_to_bucket": path_to_bucket,
        "s3_keys": s3_keys,
        "data_base": data_base,
        "table_name": table_name,
        "model": model,
        "device": device,
        "modality_name": modality_name,
        "task": task,
        "s3_connection": s3_connection,
        "output_bucket": output_bucket,
        "reentrant_lock": reentrant_lock,
        "skip_existing": not force_recompute,
        "_clients": clients,
        "_user": user,
        "keep_endpoint": bool(keep_endpoint),
    }


def _fire_bucket_error_callbacks(bucket_errors, bucket_to_clients, client_callbacks):
    """ERROR callbacks for every client attached to an unresolvable bucket."""
    errored_clients: set[str] = set()
    for bucket in bucket_errors:
        for cid in bucket_to_clients.get(bucket, set()):
            errored_clients.add(cid)
            cb = client_callbacks.get(cid)
            if cb:
                cb(TaskStatus.ERROR)
    return errored_clients


def _assemble_result_groups(model_groups, client_callbacks, errored_clients):
    """Per-model (items, group_callbacks, group_users) excluding errored clients."""
    result_groups = []
    for items in model_groups.values():
        group_client_ids: set[str] = set()
        group_users: list[str] = []
        for item in items:
            group_client_ids.update(item["_clients"])
            group_users.append(item["_user"])
        group_callbacks = {
            cid: client_callbacks[cid]
            for cid in group_client_ids
            if cid in client_callbacks and cid not in errored_clients
        }
        result_groups.append((items, group_callbacks, group_users))
    return result_groups


def prepare_model_groups(
    active_threads,
    client_entries,
    buckets_cfg,
    device,
    key,
    client_callbacks=None,
    force_recompute=False,
):
    """Resolve bucket configs and group items by model.

    Runs synchronously before pool submission.  Each returned group is an
    independent unit of work that can be submitted to a pool thread via
    ``process_model_group``.

    Returns a list of ``(items, group_callbacks, group_users)`` tuples, one
    per distinct model encountered across all client entries.  Clients whose
    bucket config cannot be resolved receive an ERROR callback here; they are
    excluded from all returned groups.

    ``items`` is a list of per-bucket dicts (one per bucket in the group).
    Each dict carries ``reentrant_lock`` and ``_clients`` — they are consumed
    and popped by ``process_model_group``.
    """
    if client_callbacks is None:
        client_callbacks = {}

    keys_per_bucket, bucket_to_clients = _build_bucket_membership(client_entries)
    model_groups, bucket_errors = _resolve_model_groups(
        keys_per_bucket, bucket_to_clients, buckets_cfg, device, force_recompute
    )
    errored_clients = _fire_bucket_error_callbacks(
        bucket_errors, bucket_to_clients, client_callbacks
    )
    return _assemble_result_groups(model_groups, client_callbacks, errored_clients)


def _launch_group_rivault_model(model, items, group_callbacks):
    """RiVault launch for the group; None on failure (clients already ERRORed)."""
    from umrs.server.rivault_launcher import launch_rivault_model

    for cb in group_callbacks.values():
        cb(TaskStatus.INITIALIZING)
    try:
        with rivault_lock:
            model = launch_rivault_model(model.model.copy())
    except Exception as e:  # noqa: BLE001 — launcher failure shape varies; clients get ERROR either way
        import traceback
        logger.error("Error launching RiVault model for group: %s", e)
        traceback.print_exc()
        for cb in group_callbacks.values():
            cb(TaskStatus.ERROR)
        return None
    for item in items:
        item["model"] = model
    return model


def _pop_locks_and_clients(items):
    """Pop reentrant_lock/_clients/_user from each item; returns (locks, client_sets)."""
    reentrant_locks = [item.pop("reentrant_lock") for item in items]
    client_sets = [item.pop("_clients") for item in items]
    # _user already used during prepare; pop to keep items clean for backend
    for item in items:
        item.pop("_user", None)
    return reentrant_locks, client_sets


def _compute_group_batch(embedding_backend, items, reentrant_locks):
    """Acquire every bucket lock, run one batch compute, always release."""
    for lock in reentrant_locks:
        lock.acquire()
    try:
        return embedding_backend.compute_and_store_batch(items)
    finally:
        for lock in reentrant_locks:
            lock.release()


def _report_group_results(items, results, client_sets, group_callbacks, buckets_cfg):
    """Per-bucket num_objects updates + per-client final status callbacks."""
    client_status = {cid: TaskStatus.COMPLETED for cid in group_callbacks}
    for item, ok in zip(items, results):
        buckets_cfg.update_num_objects(item["path_to_bucket"])
        if not ok:
            for cid in client_sets[items.index(item)]:
                if cid in client_status:
                    client_status[cid] = TaskStatus.ERROR
    for cid, status in client_status.items():
        cb = group_callbacks.get(cid)
        if cb:
            cb(status)


def _release_phase_for_group(active_threads, items) -> None:
    """Auto-release backend jobs on idle, unless keep_endpoint was requested.

    keep_endpoint=true leaves the mini-rivault endpoint warm; the idle
    janitor reaps it once genuinely idle (ai4s-platform port, pre-batch era).
    """
    keep_endpoint_requested = any(item.get("keep_endpoint") for item in items)
    if keep_endpoint_requested:
        logger.info(
            "[lifecycle] keep_endpoint requested — leaving backend endpoint(s) warm"
        )
        return
    if os.environ.get("UMRS_SERVICE_LIFECYCLE", "auto_release") != "auto_release":
        return
    with active_threads.lock:
        current_ident = threading.current_thread().ident
        still_running = sum(
            1 for future in active_threads.thread_list
            if not future.done() and future.ident != current_ident
        )
    if still_running == 0:
        from umrs.server.rivault_launcher import release_all_backend_jobs
        release_all_backend_jobs()


def _report_group_pretty_status(active_threads, all_users) -> None:
    """Final pretty thread-status block for the finished group."""
    logger.info("Model group done. Users: %s", all_users)
    from umrs.server.config import ENABLE_PRETTY_PRINT
    with active_threads.lock:
        pretty_print_thread_status(
            active_threads.thread_dict,
            running_only=False,
            print_output=ENABLE_PRETTY_PRINT,
        )


def process_model_group(
    active_threads,
    items,
    group_callbacks,
    buckets_cfg,
    embedding_backend,
    key,
    all_users,
    update_when_done=True,
):
    """Process one model group in a pool thread.

    Handles Rivault model launch (when needed), acquires per-bucket
    reentrant locks, runs a single ``compute_and_store_batch`` call, fires
    per-client status callbacks, and optionally updates the thread-dict
    status when done.

    ``items`` is the list of per-bucket dicts produced by
    ``prepare_model_groups``; this function pops ``reentrant_lock``,
    ``_clients``, and ``_user`` from each item before forwarding to the
    embedding backend.
    """
    # Register this thread in thread_dict so update_status_to_done() can locate it.
    # Must happen before any items are popped (s3_keys and path_to_bucket are still present).
    keys_per_bucket = {item["path_to_bucket"]: item["s3_keys"] for item in items}
    active_threads.update_thread_dict(buckets_cfg, keys_per_bucket, key)

    model_name = getattr(items[0].get("model"), "model_name", str(items[0].get("model")))
    logger.info(
        "[THREAD] Starting model='%s' buckets=%s clients=%s",
        model_name,
        [item["path_to_bucket"] for item in items],
        sorted(group_callbacks.keys()),
    )

    # Signal RUNNING to all clients in this group.
    for cb in group_callbacks.values():
        cb(TaskStatus.RUNNING)

    model = items[0]["model"]

    # --- Optional Rivault model launch ---
    if getattr(model, "type", None) == "rivault":
        model = _launch_group_rivault_model(model, items, group_callbacks)
        if model is None:
            return

    # --- Extract and acquire per-bucket locks ---
    reentrant_locks, client_sets = _pop_locks_and_clients(items)
    results = _compute_group_batch(embedding_backend, items, reentrant_locks)

    _report_group_results(items, results, client_sets, group_callbacks, buckets_cfg)

    # --- Thread-dict lifecycle ---
    if update_when_done:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key(all_users, key)
        active_threads.remove_from_running_list()

    _release_phase_for_group(active_threads, items)
    _report_group_pretty_status(active_threads, all_users)


def process_s3_objects_batch(
    active_threads,
    client_entries,
    buckets_cfg,
    device,
    embedding_backend,
    key,
    client_callbacks=None,
    force_recompute=False,
    update_when_done=True,
):
    """Sequential wrapper used by maintenance paths.

    Prepares model groups synchronously then runs each group in-thread
    (no pool submission).  For parallel execution from the ZMQ event loop
    use ``launch_s3_threads_batch`` in ``UMRSThreads``, which submits one
    pool thread per model group.
    """
    groups = prepare_model_groups(
        active_threads, client_entries, buckets_cfg, device, key,
        client_callbacks, force_recompute,
    )
    for items, group_callbacks, group_users in groups:
        process_model_group(
            active_threads, items, group_callbacks, buckets_cfg,
            embedding_backend, key, group_users, update_when_done,
        )
