"""Handler for the PROCESS_OBJECTS command.

Launches a thread pool worker that fetches S3 objects and computes their
embeddings.  On success the thread identifier is associated with task_id
so the event loop can track completion.  If the thread pool is saturated
the task is immediately rejected with TaskStatus.ERROR.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


class ProcessObjectsHandler(CommandHandler):
    """Handle PROCESS_OBJECTS — compute embeddings/execure model for a set of S3 objects."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        client_id, nb_objects = payload

        # Get the data queue from the control MQ's client dictionary.
        # The S3 event messages (S3 keys) were pushed by the client
        # onto this data queue during upload.
        client_entry = ctx.control_mq.client_dictionary.get(client_id, {})
        data_queue = client_entry.get("data_queue")
        if data_queue is None:
            logger.warning("No data queue for client=%s", client_id)
            update_status(TaskStatus.ERROR)
            return

        # Collect S3 events and process them in a background thread.
        # This prevents the blocking wait_for_s3_events() call from
        # stalling the server's control queue loop, which would make
        # the server unresponsive to pings and other commands.
        def _collect_and_process() -> None:
            objects = data_queue.wait_for_s3_events(nb_objects, timeout_s=60)
            if len(objects) < nb_objects:
                logger.warning(
                    "Received %s/%s S3 events for client=%s (timeout)",
                    len(objects),
                    nb_objects,
                    client_id,
                )
            if not objects:
                logger.warning(
                    "No S3 events received for client=%s (nb_objects=%s)",
                    client_id,
                    nb_objects,
                )
                update_status(TaskStatus.ERROR)
                return

            try:
                thread_ident = ctx.task_manager.active_threads.launch_s3_threads(
                    objects,
                    ctx.buckets_cfg,
                    ctx.device,
                    ctx.embedding_backend,
                    "process_objects",
                    update_status_callback=update_status,
                )
                ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
            except ServerOverloadedError as exc:
                logger.warning("Rejecting PROCESS_OBJECTS task %s: %s", task_id, exc)
                update_status(TaskStatus.ERROR)

        thread = threading.Thread(
            target=_collect_and_process,
            name=f"process-objects-{task_id}",
            daemon=True,
        )
        thread.start()
