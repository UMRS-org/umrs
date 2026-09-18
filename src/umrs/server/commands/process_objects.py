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
        client_callbacks: dict[str, Any] = {}
        for client_id, nb_objects in payload:
            task_id, cb = self._make_task(ctx, TaskCmd.PROCESS_OBJECTS, (client_id, nb_objects))
            client_callbacks[client_id] = cb

        def _collect_and_process() -> None:
            client_entries = []
            failed = []

            for client_id, nb_objects in payload:
                client_entry = ctx.control_mq.client_dictionary.get(client_id, {})
                data_queue = client_entry.get("data_queue")
                if data_queue is None:
                    logger.warning("No data queue for client=%s in batch", client_id)
                    failed.append(client_id)
                    continue

                objects = data_queue.wait_for_s3_events(nb_objects, timeout_s=60)
                if len(objects) < nb_objects:
                    logger.warning(
                        "Received %s/%s S3 events for client=%s (timeout)",
                        len(objects), nb_objects, client_id,
                    )
                if not objects:
                    logger.warning("No S3 events for client=%s in batch", client_id)
                    failed.append(client_id)
                    continue
                client_entries.append((client_id, objects))

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

        thread = threading.Thread(
            target=_collect_and_process,
            name="process-objects-batch",
            daemon=True,
        )
        thread.start()
