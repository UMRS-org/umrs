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


def process_s3_objects(
    active_threads,
    s3_objects,
    buckets_cfg,
    device,
    embedding_backend,
    key,
    update_when_done=True,
    update_status_callback=None,
    force_recompute=False,
):
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
    for path_to_bucket in keys_per_bucket:
        selected_fields = [
            "s3_username",
            "data_base",
            "table_name",
            "model",
            "store_embeddings_flag",
            "modality_name",
            "task",
            "s3_connection",
            "output_path_to_bucket",
            "reentrant_lock",
        ]
        result = buckets_cfg.get(path_to_bucket, filter=selected_fields)
        if result is None:
            print(
                f"WARNING: No bucket config for '{path_to_bucket}', skipping processing"
            )
            exit_status = TaskStatus.ERROR
            continue
        (
            user,
            data_base,
            table_name,
            model,
            store_embeddings_flag,
            modality_name,
            task,
            s3_connection,
            output_bucket,
            reentrant_lock,
        ) = result

        users.append(user)
        s3_keys = keys_per_bucket[path_to_bucket]
        if model.type == "rivault":
            from umrs.server.rivault_launcher import launch_rivault_model

            try:
                if update_status_callback:
                    update_status_callback(TaskStatus.INITIALIZING)
                # launch_rivault_model is not reentrant, so we use a lock
                with rivault_lock:
                    model = launch_rivault_model(model.model.copy())
            except Exception as e:
                import traceback

                print(f"Error creating the RiVault model: {e}")
                traceback.print_exc()
                exit_status = TaskStatus.ERROR
                continue

        with reentrant_lock:
            should_process = (
                store_embeddings_flag
                or (
                    model is not None
                    and getattr(model, "return_type", None) == "file"
                )
                or (
                    model is not None
                    and getattr(model, "type", None)
                    in ("user_script", "user_workflow")
                )
            )
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

                        print(
                            f"Error computing/storing embeddings for bucket "
                            f"'{path_to_bucket}': {e}"
                        )
                        traceback.print_exc()
                        exit_status = TaskStatus.ERROR
            else:
                print(
                    f'Error: We do not compute embeddings for this bucket '
                    f'"{path_to_bucket}" (modality = "{modality_name}")\n'
                    f"\tUser is probably not configured"
                )
            buckets_cfg.update_num_objects(path_to_bucket)

    if update_when_done:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key(users, key)
        active_threads.remove_from_running_list()

    
    # --- Service lifecycle: auto-release backend jobs when idle ---
    service_lifecycle = os.environ.get("UMRS_SERVICE_LIFECYCLE", "auto_release")
    if service_lifecycle == "auto_release":
        with active_threads.lock:
            current_ident = threading.current_thread().ident
            still_running = sum(
                1
                for future in active_threads.thread_list
                if not future.done() and future.ident != current_ident
            )
        if still_running == 0:
            from umrs.server.rivault_launcher import release_all_backend_jobs

            release_all_backend_jobs()

    if update_status_callback:
        update_status_callback(exit_status)

    print(f"\t--> Processing {nb_objects} file(s) done with status: {exit_status}!")
    from umrs.server.config import ENABLE_PRETTY_PRINT

    with active_threads.lock:
        pretty_print_thread_status(
            active_threads.thread_dict,
            running_only=False,
            print_output=ENABLE_PRETTY_PRINT,
        )
    print(
        "####### Waiting for new objects in the S3 bucket "
        "(Press CTRL+C to exit)... #######"
    )
