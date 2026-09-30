"""Handler for the KNN command.

Launches a background thread that:
  1. Loads the query object from S3 (reuses the cached embedding when the
     ETag matches, otherwise computes it via the bucket's model).
  2. Performs a cosine-distance kNN search via pgvector.
  3. Sends the JSON-encoded result list back to the caller over ZMQ.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


class KnnHandler(CommandHandler):
    """Handle KNN — k-nearest-neighbour search on a user's embedding table."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.KNN, payload)
        client_id, args_str, zmq_sender_id = payload

        try:
            args = json.loads(args_str)
            n = int(args["n"])
            s3_object = str(args["object"])
            bucket_name = str(args["bucket"])
            subdir = args.get("subdir") or None
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("KNN bad params from %s: %s", client_id, exc)
            update_status(TaskStatus.ERROR)
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string(
                    json.dumps({"status": "error", "message": f"bad params: {exc}"}),
                    zmq_sender_id,
                )
            return

        try:
            thread_ident = ctx.task_manager.active_threads.launch_knn(
                client_id,
                n,
                s3_object,
                bucket_name,
                ctx.buckets_cfg,
                ctx.control_mq,
                zmq_sender_id,
                "knn",
                subdir=subdir,
                update_status_callback=update_status,
            )
            ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
        except ServerOverloadedError as exc:
            logger.warning("Rejecting KNN task %s: %s", task_id, exc)
            update_status(TaskStatus.ERROR)
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string("SERVER_OVERLOADED", zmq_sender_id)
