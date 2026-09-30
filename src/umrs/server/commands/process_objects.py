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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


class ProcessObjectsBatchHandler(CommandHandler):
    """Handle PROCESS_OBJECTS_BATCH — batch multi-client embedding computation.

    ``payload`` is ``[(client_id, nb_objects), ...]``.
    Each client gets its own task_id; authorization was already performed per
    client when their individual PROCESS_OBJECTS was ACKed by the ZMQ loop.
    """

    def execute(self, payload: list, ctx: HandlerContext) -> None:
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
        for client_id, nb_objects in payload:
            task_id, cb = self._make_task(ctx, TaskCmd.PROCESS_OBJECTS, (client_id, nb_objects))
            per_client_cbs.setdefault(client_id, []).append(cb)
            task_ids.append((client_id, task_id))
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
            objects = data_queue.wait_for_s3_events(nb_objects, timeout_s=60)
            if len(objects) < nb_objects:
                logger.warning(
                    "Received %s/%s S3 events for client=%s (timeout)",
                    len(objects), nb_objects, client_id,
                )
            return client_id, objects or None

        def _fetch_all_client_events() -> tuple[list, list]:
            """Parallel S3-event fetch → (client_entries, failed_client_ids)."""
            client_entries = []
            failed = []
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
            return client_entries, failed

        def _launch_batch_threads(client_entries, active_callbacks) -> None:
            """Pool submission honoring ServerOverloadedError (callbacks → ERROR)."""
            logger.info(
                "[PROCESS] S3 events ready — launching threads for %d client(s): %s",
                len(client_entries),
                [(cid, len(objs)) for cid, objs in client_entries],
            )
            try:
                ctx.task_manager.active_threads.launch_s3_threads_batch(
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

        def _collect_and_process() -> None:
            # Collect S3 events for all clients in parallel so the pool thread
            # starts as soon as the LAST client responds, not sum(all waits).
            client_entries, failed = _fetch_all_client_events()

            for cid in failed:
                cb = client_callbacks.get(cid)
                if cb:
                    cb(TaskStatus.ERROR)

            if not client_entries:
                return

            active_callbacks = {
                cid: client_callbacks[cid]
                for cid, _ in client_entries
                if cid in client_callbacks
            }
            _launch_batch_threads(client_entries, active_callbacks)

        thread = threading.Thread(
            target=_collect_and_process,
            name="process-objects-batch",
            daemon=True,
        )
        thread.start()
