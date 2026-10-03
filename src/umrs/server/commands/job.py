"""Handlers for JOB_SUBMIT, JOB_CANCEL and JOB_LIST commands."""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.job_tracking_handler import handle_job_command

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def _caller_is_admin(ctx: HandlerContext, client_id: str, _zmq_sender_id: Any) -> bool:
    """Admin from the server's own verification, never the payload — the same
    rule RUN_WORKFLOW and the command registry use (``get_auth_context``: the
    sending CURVE session's grants or an admin token it presented itself)."""
    get_auth = getattr(ctx.control_mq, "get_auth_context", None)
    auth = get_auth(client_id) if callable(get_auth) else None
    return getattr(auth, "is_admin", False) is True


def _send_reply(
    ctx: HandlerContext, zmq_sender_id: Any, response_json: str, label: str
) -> None:
    """Send *response_json* back to the ZeroMQ caller. Log but never raise."""
    try:
        ctx.control_mq.send_string(response_json, zmq_sender_id)
    except RuntimeError as exc:
        logger.error("Error sending %s response: %s", label, exc)


class JobSubmitHandler(CommandHandler):
    """Handle JOB_SUBMIT — submit a new durable job with retry config."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.JOB_SUBMIT, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_job_command(
            "JOB_SUBMIT",
            client_id,
            payload_str,
            backend=ctx.job_command_backend,
        )
        _send_reply(ctx, zmq_sender_id, response_json, "JOB_SUBMIT")
        update_status(TaskStatus.COMPLETED)


class JobCancelHandler(CommandHandler):
    """Handle JOB_CANCEL — cancel a submitted job by ID."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.JOB_CANCEL, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_job_command(
            "JOB_CANCEL",
            client_id,
            payload_str,
            backend=ctx.job_command_backend,
            caller_is_admin=_caller_is_admin(ctx, client_id, zmq_sender_id),
        )
        _send_reply(ctx, zmq_sender_id, response_json, "JOB_CANCEL")
        update_status(TaskStatus.COMPLETED)


class JobListHandler(CommandHandler):
    """Handle JOB_LIST — list durable jobs with optional filters."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.JOB_LIST, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_job_command(
            "JOB_LIST",
            client_id,
            payload_str,
            backend=ctx.job_command_backend,
            caller_is_admin=_caller_is_admin(ctx, client_id, zmq_sender_id),
        )
        _send_reply(ctx, zmq_sender_id, response_json, "JOB_LIST")
        update_status(TaskStatus.COMPLETED)
