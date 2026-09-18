"""Handler for the EXPORT_EMBEDDINGS command.

Launches a background thread that serialises computed embedding vectors
from the database to a local file.  The thread sends a ZeroMQ reply to
the caller when the export is complete.

If the thread pool is saturated, the task is rejected immediately and the
caller receives a "SERVER_OVERLOADED" ZeroMQ reply so it is not left hanging.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


class ExportEmbeddingsHandler(CommandHandler):
    """Handle EXPORT_EMBEDDINGS — export computed vectors to a local file."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.EXPORT_EMBEDDINGS, payload)
        client_id, export_args, zmq_sender_id = payload

        try:
            thread_ident = ctx.task_manager.active_threads.launch_export_embeddings(
                client_id,
                export_args,
                ctx.buckets_cfg,
                ctx.control_mq,
                zmq_sender_id,
                "export_embeddings",
                update_status_callback=update_status,
            )
            ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
        except ServerOverloadedError as exc:
            logger.warning("Rejecting EXPORT_EMBEDDINGS task %s: %s", task_id, exc)
            update_status(TaskStatus.ERROR)
            # Notify the client so it is not left waiting indefinitely.
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string("SERVER_OVERLOADED", zmq_sender_id)
