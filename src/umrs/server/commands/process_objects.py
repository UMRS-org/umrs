"""Handler for the PROCESS_OBJECTS command.

Launches a thread pool worker that fetches S3 objects and computes their
embeddings.  On success the thread identifier is associated with task_id
so the event loop can track completion.  If the thread pool is saturated
the task is immediately rejected with TaskStatus.ERROR.
"""

from __future__ import annotations

import logging
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
        client_id, s3_objects = payload
        try:
            thread_ident = ctx.task_manager.active_threads.launch_s3_threads(
                s3_objects,
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
