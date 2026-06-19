"""Handler for PURGE_BUCKETS command.

Deletes user-owned extra S3 buckets using admin credentials.
The client falls back to this server-side purge when local
``DeleteBucket`` is denied (the typical case for remote users).
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.maintenance import handle_purge_buckets

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


def _send_reply(
    ctx: HandlerContext, zmq_sender_id: Any, response_json: str, label: str
) -> None:
    """Send *response_json* back to the ZeroMQ caller.  Log but never raise."""
    try:
        ctx.control_mq.send_string(response_json, zmq_sender_id)
    except RuntimeError as exc:
        logger.error("Error sending %s response: %s", label, exc)


class PurgeBucketsHandler(CommandHandler):
    """Handle PURGE_BUCKETS — delete user-owned extra S3 buckets."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_purge_buckets(
            "PURGE_BUCKETS",
            client_id,
            payload_str,
            configure_file=ctx.configure_file,
        )
        _send_reply(ctx, zmq_sender_id, response_json, "PURGE_BUCKETS")
        update_status(TaskStatus.COMPLETED)
