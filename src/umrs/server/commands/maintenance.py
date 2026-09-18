"""Handlers for RECOMPUTE, UPDATE, CHECK and RELOAD commands.

All four delegate to ``handle_maintenance_command()`` in
``umrs.server.maintenance``, which returns:
  - an int   → thread was launched; associate with task_id
  - None     → ran synchronously and succeeded; mark COMPLETED
  - False    → error during execution; mark ERROR

On ServerOverloadedError the task is immediately rejected.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.thread_pool import ServerOverloadedError
from umrs.server.maintenance import handle_maintenance_command

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def _dispatch_maintenance(cmd: str, payload: Any, ctx: HandlerContext) -> None:
    """Shared logic for all four maintenance commands."""
    task_id, update_status = CommandHandler._make_task(ctx, cmd, payload)
    client_id, _, zmq_sender_id = payload
    with contextlib.suppress(RuntimeError):
        ctx.control_mq.send_string("ACKNOWLEDGE", zmq_sender_id)
    try:
        thread_ident = handle_maintenance_command(
            ctx.task_manager.active_threads,
            cmd,
            client_id,
            ctx.buckets_cfg,
            ctx.device,
            ctx.embedding_backend,
            ctx.configure_file,
            ctx.user_config_map,
            cmd,
            update_status_callback=update_status,
        )
        # handle_maintenance_command return values:
        #   None  — ran synchronously and succeeded
        #   False — ran synchronously but encountered an error
        #   int   — background thread launched; associate with task_id
        if thread_ident is None:
            update_status(TaskStatus.COMPLETED)
        elif thread_ident is False:
            update_status(TaskStatus.ERROR)
        else:
            ctx.task_manager.associate_thread_with_task(thread_ident, task_id)
    except ServerOverloadedError as exc:
        logger.warning("Rejecting %s task %s: %s", cmd, task_id, exc)
        update_status(TaskStatus.ERROR)


class RecomputeHandler(CommandHandler):
    """Handle RECOMPUTE — recompute all embeddings for the user's buckets."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        _dispatch_maintenance(TaskCmd.RECOMPUTE, payload, ctx)


class UpdateHandler(CommandHandler):
    """Handle UPDATE — update missing/mismatched embeddings for the user's buckets."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        _dispatch_maintenance(TaskCmd.UPDATE, payload, ctx)


class CheckHandler(CommandHandler):
    """Handle CHECK — check S3/DB consistency (admin: all users)."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        _dispatch_maintenance(TaskCmd.CHECK, payload, ctx)


class ReloadHandler(CommandHandler):
    """Handle RELOAD — hot-reload server config without restart (admin only)."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        _dispatch_maintenance(TaskCmd.RELOAD, payload, ctx)
