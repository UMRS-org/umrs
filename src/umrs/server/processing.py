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
    """Batch variant of process_s3_objects.

    ``client_entries`` is a list of ``(client_id, s3_objects)`` pairs from
    multiple clients.  Objects from all clients are resolved against
    ``buckets_cfg``, grouped by model, and processed with a single model call
    per model group.  Per-client status callbacks are fired individually so
    each client sees its own task status.

    ``client_callbacks`` is a ``{client_id: callable}`` dict.
    """
    if client_callbacks is None:
        client_callbacks = {}

    # --- 1. Resolve bucket config for every (client, object) pair ---
    # bucket_items[path_to_bucket] = list of (client_id, s3_key)
    bucket_to_keys = defaultdict(list)   # bucket → [s3_key, ...]  (deduped)
    bucket_to_clients = defaultdict(set) # bucket → {client_id, ...}

    all_users = []
    keys_per_bucket_all = defaultdict(list)

    for client_id, s3_objects in client_entries:
        for s3_object in s3_objects:
            logger.info(f"\t\t-->{s3_object}")
            path_to_bucket, s3_key = s3_object.replace("s3://", "").split("/", 1)
            if s3_key not in keys_per_bucket_all[path_to_bucket]:
                keys_per_bucket_all[path_to_bucket].append(s3_key)
            bucket_to_clients[path_to_bucket].add(client_id)

    active_threads.update_thread_dict(buckets_cfg, keys_per_bucket_all, key)

    selected_fields = [
        "s3_username", "data_base", "table_name", "model",
        "store_embeddings_flag", "modality_name", "task",
        "s3_connection", "output_path_to_bucket", "reentrant_lock",
    ]

    # Resolve config per bucket and group by model_name
    # model_group[model_name] = list of item dicts ready for compute_and_store_batch
    model_groups = defaultdict(list)
    bucket_errors = set()

    for path_to_bucket, s3_keys in keys_per_bucket_all.items():
        result = buckets_cfg.get(path_to_bucket, filter=selected_fields)
        if result is None:
            logger.warning("No bucket config for '%s', skipping", path_to_bucket)
            bucket_errors.add(path_to_bucket)
            continue
        (user, data_base, table_name, model, store_embeddings_flag,
         modality_name, task, s3_connection, output_bucket, reentrant_lock) = result

        all_users.append(user)

        should_process = (
            store_embeddings_flag
            or (model is not None and getattr(model, "return_type", None) == "file")
            or (model is not None and getattr(model, "type", None) in ("user_script", "user_workflow"))
        )
        if not should_process:
            logger.info("No embeddings configured for bucket '%s', skipping", path_to_bucket)
            buckets_cfg.update_num_objects(path_to_bucket)
            continue

        if model.type == "rivault":
            from umrs.server.rivault_launcher import launch_rivault_model
            for cid in bucket_to_clients[path_to_bucket]:
                cb = client_callbacks.get(cid)
                if cb:
                    cb(TaskStatus.INITIALIZING)
            try:
                with rivault_lock:
                    model = launch_rivault_model(model.model.copy())
            except Exception as e:
                import traceback
                logger.error("Error creating RiVault model for bucket '%s': %s", path_to_bucket, e)
                traceback.print_exc()
                bucket_errors.add(path_to_bucket)
                continue

        model_name = getattr(model, "model_name", str(model))
        model_groups[model_name].append({
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
        })

    # --- 2. For each model group, fire one batch call ---
    # Track per-client success: a client fails if ANY of its buckets fails.
    client_status = {cid: TaskStatus.COMPLETED for cid, _ in client_entries}

    # Signal RUNNING to all clients whose buckets resolved correctly
    for cid in list(client_status):
        cb = client_callbacks.get(cid)
        if cb:
            cb(TaskStatus.RUNNING)

    for bucket in bucket_errors:
        for cid in bucket_to_clients.get(bucket, set()):
            client_status[cid] = TaskStatus.ERROR

    for model_name, items in model_groups.items():
        reentrant_locks = [item.pop("reentrant_lock") for item in items]
        client_sets = [item.pop("_clients") for item in items]

        for lock in reentrant_locks:
            lock.acquire()
        try:
            results = embedding_backend.compute_and_store_batch(items)
        finally:
            for lock in reentrant_locks:
                lock.release()

        for item, ok, clients in zip(items, results, client_sets):
            buckets_cfg.update_num_objects(item["path_to_bucket"])
            if not ok:
                for cid in clients:
                    client_status[cid] = TaskStatus.ERROR

    # --- 3. Fire per-client callbacks with final status ---
    for cid, status in client_status.items():
        cb = client_callbacks.get(cid)
        if cb:
            cb(status)

    if update_when_done:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key(all_users, key)
        active_threads.remove_from_running_list()

    # --- Service lifecycle: auto-release backend jobs when idle ---
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

    logger.info("Batch processing done for %d client(s).", len(client_entries))
    from umrs.server.config import ENABLE_PRETTY_PRINT
    with active_threads.lock:
        pretty_print_thread_status(
            active_threads.thread_dict,
            running_only=False,
            print_output=ENABLE_PRETTY_PRINT,
        )
