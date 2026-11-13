"""Handler for the KNN command (port of upstream 1798ac05 / e04b3d4e; plan U1 A6).

``KNN:<client_id>:{"n": <k>, "object": "<s3 key>", "bucket": "<path_to_bucket>"
[, "subdir": "<prefix>"]}``. The registry has already checked the command
and any project the payload names (ADR-097 D-7). This handler validates the
request and the caller's ownership of the bucket on the event loop, so a
refused request takes no pool slot, then runs the search on the pool
(``umrs.server.knn_worker``), which replies with
``{"status": "ok", "results": [{"object", "score"}, ...], ...}``. Every
refusal, here or in the worker, is ``{"status": "error", "code", "message"}``.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError
from umrs.server.knn_worker import (
    KnnError,
    error_reply,
    parse_knn_request,
    searchable_entry,
)

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def _reply(ctx: HandlerContext, zmq_sender_id: Any, message: str) -> None:
    with contextlib.suppress(RuntimeError):
        ctx.control_mq.send_string(message, zmq_sender_id)


class KnnHandler(CommandHandler):
    """Handle KNN — k nearest neighbours in the caller's own bucket."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.KNN, payload)
        client_id, args_str, zmq_sender_id = payload

        try:
            request = parse_knn_request(args_str)
            entry = searchable_entry(ctx.buckets_cfg, request.bucket, client_id)
        except KnnError as exc:
            logger.warning("KNN refused for %s: %s", client_id, exc)
            update_status(TaskStatus.ERROR)
            _reply(ctx, zmq_sender_id, json.dumps(error_reply(exc.code, str(exc))))
            return

        try:
            thread_ident = ctx.task_manager.active_threads.launch_knn(
                client_id,
                request,
                entry,
                ctx.buckets_cfg,
                ctx.control_mq,
                zmq_sender_id,
                "knn",
                update_status_callback=update_status,
            )
            ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
        except ServerOverloadedError as exc:
            logger.warning("Rejecting KNN task %s: %s", task_id, exc)
            update_status(TaskStatus.ERROR)
            _reply(ctx, zmq_sender_id, "SERVER_OVERLOADED")
