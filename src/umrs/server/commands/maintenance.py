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

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.core.thread_pool import ServerOverloadedError
from umrs.server.maintenance import handle_maintenance_command

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


def _dispatch_maintenance(
    cmd: str,
    payload: Any,
    update_status: StatusCallback,
    ctx: HandlerContext,
    task_id: int,
) -> None:
    """Shared logic for all four maintenance commands."""
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
            cmd.lower(),
            update_status_callback=update_status,
        )
        # handle_maintenance_command has three possible return values depending
        # on the execution context:
        #
        #   None  — ran synchronously and succeeded.  This happens when the
        #           command is called at server bootstrap (handle_maintenance_flags),
        #           where execution is blocking by design: the server must finish
        #           recomputing / reloading before entering the event loop.
        #
        #   False — ran synchronously but encountered an error (e.g. RELOAD
        #           failed to parse the new config); mark the task as ERROR.
        #
        #   int   — a background thread was launched.  This is the normal path
        #           when the command is triggered by a client at runtime: the
        #           server must stay responsive, so the work runs in a thread.
        #           Associate the thread ident with task_id so the event loop
        #           can mark the task COMPLETED when the thread finishes.
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

    def execute(self, payload, update_status: StatusCallback, ctx: HandlerContext, task_id: int) -> None:
        _dispatch_maintenance("RECOMPUTE", payload, update_status, ctx, task_id)


class UpdateHandler(CommandHandler):
    """Handle UPDATE — update missing/mismatched embeddings for the user's buckets."""

    def execute(self, payload, update_status: StatusCallback, ctx: HandlerContext, task_id: int) -> None:
        _dispatch_maintenance("UPDATE", payload, update_status, ctx, task_id)


class CheckHandler(CommandHandler):
    """Handle CHECK — check S3/DB consistency (admin: all users)."""

    def execute(self, payload, update_status: StatusCallback, ctx: HandlerContext, task_id: int) -> None:
        _dispatch_maintenance("CHECK", payload, update_status, ctx, task_id)


class ReloadHandler(CommandHandler):
    """Handle RELOAD — hot-reload server config without restart (admin only)."""

    def execute(self, payload, update_status: StatusCallback, ctx: HandlerContext, task_id: int) -> None:
        _dispatch_maintenance("RELOAD", payload, update_status, ctx, task_id)
