"""Handler for PROCESS_OBJECTS_BATCH.

The ZMQ loop always wraps incoming PROCESS_OBJECTS requests into a
PROCESS_OBJECTS_BATCH payload (even for a single client), so this is the
only handler needed for ingestion.  Each client in the batch gets its own
task_id and status callback; embeddings are computed in a single model
forward pass per model group.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from psycopg2 import sql as pgsql

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def filter_already_indexed(ctx, client_entries):
    """Return client_entries with already-indexed objects removed.

    For each bucket, fires head_object calls in parallel to get ETags, then
    does a single SQL query to find objects already stored with the same checksum.
    Objects present in DB with a matching ETag are dropped from the result.
    """
    # Group all s3_uris by bucket so we can batch per (data_base, table_name)
    # s3_uri format: "s3://path_to_bucket/s3_key"
    # DB format:      "path_to_bucket/s3_key"
    bucket_items = defaultdict(list)  # path_to_bucket -> [(client_id, s3_uri, s3_key)]
    for client_id, objects in client_entries:
        for s3_uri in objects:
            path_to_bucket, s3_key = s3_uri.replace("s3://", "").split("/", 1)
            bucket_items[path_to_bucket].append((client_id, s3_uri, s3_key))

    skip_uris = set()

    for path_to_bucket, items in bucket_items.items():
        entry = ctx.buckets_cfg.get(path_to_bucket)
        if entry is None:
            continue

        s3 = entry.s3_connection
        data_base = entry.data_base
        table_name = entry.table_name

        # Parallel head_object to get ETags
        def _head(item, _bucket=path_to_bucket):
            _, s3_uri, s3_key = item
            try:
                etag = s3.head_object(Bucket=_bucket, Key=s3_key)["ETag"].strip('"')
                return s3_uri, etag
            except Exception:
                return s3_uri, None

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            etag_map = dict(pool.map(_head, items))
        t_head = time.perf_counter() - t0

        # Single SQL batch query for this bucket
        t0 = time.perf_counter()
        db_keys = [f"{path_to_bucket}/{s3_key}" for _, _, s3_key in items]
        conn = data_base.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    pgsql.SQL("SELECT s3_object, checksum FROM {} WHERE s3_object = ANY(%s)")
                    .format(pgsql.Identifier(table_name)),
                    (db_keys,),
                )
                db_checksums = {row[0]: row[1] for row in cur.fetchall()}
        finally:
            data_base.put_conn(conn)
        t_sql = time.perf_counter() - t0

        # Filter: skip if ETag matches DB checksum
        n_skipped_before = len(skip_uris)
        for _, s3_uri, s3_key in items:
            db_key = f"{path_to_bucket}/{s3_key}"
            etag = etag_map.get(s3_uri)
            db_checksum = db_checksums.get(db_key)
            if etag and db_checksum and etag == db_checksum:
                skip_uris.add(s3_uri)

        logger.info(
            "[PROCESS] already-indexed check: bucket=%s n=%d head_object=%.3fs sql=%.3fs skipped=%d",
            path_to_bucket, len(items), t_head, t_sql, len(skip_uris) - n_skipped_before,
        )

    if skip_uris:
        logger.info("filter_already_indexed: skipping %d already-indexed object(s)", len(skip_uris))

    # Rebuild client_entries without skipped objects
    filtered = []
    for client_id, objects in client_entries:
        kept = [o for o in objects if o not in skip_uris]
        if kept:
            filtered.append((client_id, kept))

    return filtered


class ProcessObjectsBatchHandler(CommandHandler):
    """Handle PROCESS_OBJECTS_BATCH — batch multi-client embedding computation.

    ``payload`` is ``[(client_id, nb_objects), ...]``.
    Each client gets its own task_id; authorization was already performed per
    client when their individual PROCESS_OBJECTS was ACKed by the ZMQ loop.
    """

    def execute(self, payload: list, ctx: HandlerContext, filter: bool = True) -> None:
        # --- Task creation ---
        #
        # The ZMQ drain may deliver several PROCESS_OBJECTS from the *same*
        # client in one batch (e.g. the client uploaded two sets of files in
        # quick succession).  We create one task_id per original ZMQ request
        # so the client sees each request as a distinct entry in STATUS.
        #
        # Because downstream functions (process_s3_objects_batch,
        # launch_s3_threads_batch) consume client_callbacks as
        # dict[client_id -> callable], we keep that interface unchanged: when
        # a client appears N times, the single callable stored in the dict is
        # a closure that fires all N sub-task callbacks at once.  Callers are
        # unaware of this multiplicity — they just call cb(status) as usual.
        per_client_cbs: dict[str, list] = {}
        task_ids: list[tuple[str, int]] = []
        # client_task_ids maps each client_id to its most-recent task_id so
        # we can link threads to tasks after launch_s3_threads_batch returns.
        client_task_ids: dict[str, int] = {}
        for client_id, nb_objects in payload:
            task_id, cb = self._make_task(ctx, TaskCmd.PROCESS_OBJECTS, (client_id, nb_objects))
            per_client_cbs.setdefault(client_id, []).append(cb)
            task_ids.append((client_id, task_id))
            client_task_ids[client_id] = task_id
        logger.info(
            "[PROCESS] Tasks created: %s",
            [(cid, tid) for cid, tid in task_ids],
        )

        # Adapter closure: dict[str, callable] interface, multi-task internals.
        # The lambda captures `cbs` by value (default-arg trick) to avoid the
        # classic late-binding pitfall in Python loop closures.
        client_callbacks: dict[str, Any] = {
            cid: (lambda status, cbs=cbs: [cb(status) for cb in cbs])
            for cid, cbs in per_client_cbs.items()
        }

        # --- S3 event collection ---
        #
        # S3 events for the same client are queued in arrival order regardless
        # of how many PROCESS_OBJECTS requests produced them.  We drain them in
        # a single wait_for_s3_events call using the merged total, which is both
        # simpler and avoids a second blocking wait on an already-drained queue.
        merged: dict[str, int] = {}
        for client_id, nb_objects in payload:
            merged[client_id] = merged.get(client_id, 0) + nb_objects

        def _fetch_one(client_id: str, nb_objects: int):
            """Collect S3 events for one client — runs in a thread pool."""
            client_entry = ctx.control_mq.client_dictionary.get(client_id, {})
            data_queue = client_entry.get("data_queue")
            if data_queue is None:
                logger.warning("No data queue for client=%s in batch", client_id)
                return client_id, None
            t0 = time.perf_counter()
            objects = data_queue.wait_for_s3_events(nb_objects, timeout_s=60)
            logger.info(
                "[PROCESS] S3 events: client=%s received %d/%d in %.3fs",
                client_id, len(objects), nb_objects, time.perf_counter() - t0,
            )
            if len(objects) < nb_objects:
                logger.warning(
                    "Received %s/%s S3 events for client=%s (timeout)",
                    len(objects), nb_objects, client_id,
                )
            return client_id, objects or None

        def _collect_and_process() -> None:
            client_entries = []
            failed = []

            # Collect S3 events for all clients in parallel so the pool thread
            # starts as soon as the LAST client responds, not sum(all waits).
            with ThreadPoolExecutor(max_workers=len(merged)) as fetcher:
                futures = {
                    fetcher.submit(_fetch_one, cid, nb): cid
                    for cid, nb in merged.items()
                }
                for future in as_completed(futures):
                    client_id, objects = future.result()
                    if objects is None:
                        failed.append(client_id)
                    else:
                        client_entries.append((client_id, objects))

            for cid in failed:
                cb = client_callbacks.get(cid)
                if cb:
                    cb(TaskStatus.ERROR)

            if not client_entries:
                return

            if filter:
                original_entries = client_entries
                client_entries = filter_already_indexed(ctx, client_entries)

                surviving_cids = {cid for cid, _ in client_entries}
                filtered_entries = [(cid, objs) for cid, objs in original_entries if cid not in surviving_cids]

                if filtered_entries:
                    # Register filtered clients in thread_dict and mark Done
                    active_threads = ctx.task_manager.active_threads
                    keys_per_bucket = {}
                    for _, objects in filtered_entries:
                        for s3_uri in objects:
                            path_to_bucket, _ = s3_uri.replace("s3://", "").split("/", 1)
                            keys_per_bucket[path_to_bucket] = keys_per_bucket.get(path_to_bucket, 0) + 1
                    active_threads.update_thread_dict(ctx.buckets_cfg, keys_per_bucket, "process_objects")
                    active_threads.update_status_to_done()
                    all_users = []
                    for path_to_bucket in keys_per_bucket:
                        entry = ctx.buckets_cfg.get(path_to_bucket)
                        if entry and entry.s3_username not in all_users:
                            all_users.append(entry.s3_username)
                    active_threads.record_status_for_users_by_key(all_users, "process_objects")
                    for cid, _ in filtered_entries:
                        if cid in client_callbacks:
                            client_callbacks[cid](TaskStatus.RUNNING)
                            client_callbacks[cid](TaskStatus.COMPLETED)

                if not client_entries:
                    logger.info("[PROCESS] All objects already indexed, nothing to do.")
                    return

            active_callbacks = {
                cid: client_callbacks[cid]
                for cid, _ in client_entries
                if cid in client_callbacks
            }

            logger.info(
                "[PROCESS] S3 events ready — launching threads for %d client(s): %s",
                len(client_entries),
                [(cid, len(objs)) for cid, objs in client_entries],
            )

            # Build original counts per bucket (before filtering) so the
            # status table shows total submitted objects, not filtered ones.
            original_counts: dict[str, int] = {}
            if filter and original_entries is not client_entries:
                for _, objects in original_entries:
                    for s3_uri in objects:
                        ptb, _ = s3_uri.replace("s3://", "").split("/", 1)
                        original_counts[ptb] = original_counts.get(ptb, 0) + 1

            try:
                client_thread_map = ctx.task_manager.active_threads.launch_s3_threads_batch(
                    client_entries,
                    ctx.buckets_cfg,
                    ctx.device,
                    ctx.embedding_backend,
                    "process_objects",
                    client_callbacks=active_callbacks,
                )
            except ServerOverloadedError as exc:
                logger.warning("Rejecting PROCESS_OBJECTS_BATCH: %s", exc)
                for cb in active_callbacks.values():
                    cb(TaskStatus.ERROR)
                return

            # Link each thread to its task(s) and patch nb_objects with the
            # original submission count (stored in the task payload as "n").
            active_threads = ctx.task_manager.active_threads
            patched: set[int] = set()
            for client_id, thread_ident in client_thread_map.items():
                task_id = client_task_ids.get(client_id)
                if task_id is not None:
                    ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
                if original_counts and thread_ident not in patched:
                    active_threads.patch_thread_nb_objects(thread_ident, original_counts)
                    patched.add(thread_ident)

        thread = threading.Thread(
            target=_collect_and_process,
            name="process-objects-batch",
            daemon=True,
        )
        thread.start()
