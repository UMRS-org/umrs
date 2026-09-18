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
]


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

    # --- Build per-bucket key lists and client membership ---
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

    # --- Resolve bucket config and group by model ---
    model_groups: dict[str, list] = {}
    bucket_errors: set[str] = set()

    for path_to_bucket, s3_keys in keys_per_bucket.items():
        result = buckets_cfg.get(path_to_bucket, filter=_SELECTED_FIELDS)
        if result is None:
            logger.warning("No bucket config for '%s', skipping", path_to_bucket)
            bucket_errors.add(path_to_bucket)
            continue
        (user, data_base, table_name, model, store_embeddings_flag,
         modality_name, task, s3_connection, output_bucket, reentrant_lock) = result

        should_process = (
            store_embeddings_flag
            or (model is not None and getattr(model, "return_type", None) == "file")
            or (model is not None and getattr(model, "type", None) in ("user_script", "user_workflow"))
        )
        if not should_process:
            logger.info("No embeddings configured for bucket '%s', skipping", path_to_bucket)
            buckets_cfg.update_num_objects(path_to_bucket)
            continue

        model_name = getattr(model, "model_name", str(model))
        model_groups.setdefault(model_name, []).append({
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
            "_clients": bucket_to_clients[path_to_bucket],
            "_user": user,
        })

    # --- Fire ERROR callbacks for unresolvable buckets ---
    errored_clients: set[str] = set()
    for bucket in bucket_errors:
        for cid in bucket_to_clients.get(bucket, set()):
            errored_clients.add(cid)
            cb = client_callbacks.get(cid)
            if cb:
                cb(TaskStatus.ERROR)

    # --- Build per-group callback dicts (exclude errored clients) ---
    result_groups = []
    for model_name, items in model_groups.items():
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
        from umrs.server.rivault_launcher import launch_rivault_model
        for cb in group_callbacks.values():
            cb(TaskStatus.INITIALIZING)
        try:
            with rivault_lock:
                model = launch_rivault_model(model.model.copy())
            for item in items:
                item["model"] = model
        except Exception as e:
            import traceback
            logger.error("Error launching RiVault model for group: %s", e)
            traceback.print_exc()
            for cb in group_callbacks.values():
                cb(TaskStatus.ERROR)
            return

    # --- Extract and acquire per-bucket locks ---
    reentrant_locks = [item.pop("reentrant_lock") for item in items]
    client_sets = [item.pop("_clients") for item in items]
    # _user already used during prepare; pop to keep items clean for backend
    for item in items:
        item.pop("_user", None)

    for lock in reentrant_locks:
        lock.acquire()
    try:
        results = embedding_backend.compute_and_store_batch(items)
    finally:
        for lock in reentrant_locks:
            lock.release()

    # --- Per-bucket result processing ---
    client_status = {cid: TaskStatus.COMPLETED for cid in group_callbacks}
    for item, ok in zip(items, results):
        buckets_cfg.update_num_objects(item["path_to_bucket"])
        if not ok:
            for cid in client_sets[items.index(item)]:
                if cid in client_status:
                    client_status[cid] = TaskStatus.ERROR

    # --- Fire per-client final callbacks ---
    for cid, status in client_status.items():
        cb = group_callbacks.get(cid)
        if cb:
            cb(status)

    # --- Thread-dict lifecycle ---
    if update_when_done:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key(all_users, key)
        active_threads.remove_from_running_list()

    # --- Auto-release Rivault backend jobs when the server goes idle ---
    service_lifecycle = os.environ.get("UMRS_SERVICE_LIFECYCLE", "auto_release")
    if service_lifecycle == "auto_release":
        with active_threads.lock:
            current_ident = threading.current_thread().ident
            still_running = sum(
                1 for future in active_threads.thread_list
                if not future.done() and future.ident != current_ident
            )
        if still_running == 0:
            from umrs.server.rivault_launcher import release_all_backend_jobs
            release_all_backend_jobs()

    logger.info("Model group done. Users: %s", all_users)
    from umrs.server.config import ENABLE_PRETTY_PRINT
    with active_threads.lock:
        pretty_print_thread_status(
            active_threads.thread_dict,
            running_only=False,
            print_output=ENABLE_PRETTY_PRINT,
        )


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
