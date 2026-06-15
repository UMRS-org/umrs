"""Handler for the WAIT_OUTPUT command.

The client calls WAIT_OUTPUT to block until a result appears in a specific
output bucket's DB table (e.g. waiting for the server to finish computing
embeddings for a file it just uploaded).

A background thread is launched to poll the database for the expected table
modification.  When the modification is detected (or the timeout expires)
the thread sends a ZeroMQ reply directly to the caller.

If the thread pool is saturated, the task is rejected immediately and the
caller receives a "SERVER_OVERLOADED" ZeroMQ reply so it is not left hanging.

If the requested output bucket does not exist, the caller receives
"NO_SUCH_BUCKET" and the task is marked with TaskStatus.NO_SUCH_BUCKET.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


class WaitOutputHandler(CommandHandler):
    """Handle WAIT_OUTPUT — block until an output bucket's DB table is updated."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        client_id, output_path_to_bucket, timeout, zmq_sender_id = payload

        path_to_bucket, entry = ctx.buckets_cfg.find_by_output_path(output_path_to_bucket)
        if entry is None:
            # Bucket alias unknown — reply immediately so the client is not hung.
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string("NO_SUCH_BUCKET", zmq_sender_id)
            update_status(TaskStatus.NO_SUCH_BUCKET)
            return

        try:
            thread_ident = ctx.task_manager.active_threads.launch_wait_table_modification_on_db(
                path_to_bucket,
                ctx.buckets_cfg,
                client_id,
                entry.data_base,
                entry.table_name,
                timeout,
                ctx.control_mq,
                zmq_sender_id,
                "wait_output",
                update_status_callback=update_status,
            )
            ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
        except ServerOverloadedError as exc:
            logger.warning("Rejecting WAIT_OUTPUT task %s: %s", task_id, exc)
            update_status(TaskStatus.ERROR)
            # Notify the client so it is not left waiting indefinitely.
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string("SERVER_OVERLOADED", zmq_sender_id)
