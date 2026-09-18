"""Handlers for MODEL_REGISTER, MODEL_LIST and MODEL_DOWNLOAD commands."""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.model_registry_handler import handle_model_command

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def _send_reply(ctx: HandlerContext, zmq_sender_id: Any, response_json: str, label: str) -> None:
    try:
        ctx.control_mq.send_string(response_json, zmq_sender_id)
    except RuntimeError as exc:
        logger.error("Error sending %s response: %s", label, exc)


class ModelRegisterHandler(CommandHandler):
    """Handle MODEL_REGISTER — register a local model with UMRS."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.MODEL_REGISTER, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_model_command("MODEL_REGISTER", client_id, payload_str, backend=ctx.model_command_backend)
        _send_reply(ctx, zmq_sender_id, response_json, "MODEL_REGISTER")
        update_status(TaskStatus.COMPLETED)


class ModelListHandler(CommandHandler):
    """Handle MODEL_LIST — list registered models."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.MODEL_LIST, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_model_command("MODEL_LIST", client_id, payload_str, backend=ctx.model_command_backend)
        _send_reply(ctx, zmq_sender_id, response_json, "MODEL_LIST")
        update_status(TaskStatus.COMPLETED)


class ModelDownloadHandler(CommandHandler):
    """Handle MODEL_DOWNLOAD — server-side download, reply with ACK immediately."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.MODEL_DOWNLOAD, payload)
        client_id, payload_str, zmq_sender_id = payload
        ack = handle_model_command("MODEL_DOWNLOAD", client_id, payload_str, backend=ctx.model_command_backend)
        _send_reply(ctx, zmq_sender_id, ack, "MODEL_DOWNLOAD ack")
        update_status(TaskStatus.COMPLETED)
