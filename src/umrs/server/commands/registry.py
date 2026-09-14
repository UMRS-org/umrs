"""CommandRegistry — dispatches a TaskCmd to its registered handler.

The registry is the only place that knows the mapping cmd → handler.
The event loop just calls ``registry.dispatch(...)`` and is otherwise
unaware of any individual command's logic.

Authorization is centralized here: every payload is expected to carry
``client_id`` as its first element (string or first tuple item).  The
registry checks authorization before delegating to the handler, so
handlers never need to repeat that check.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


def _extract_client_id(payload: Any) -> str | None:
    """Return the client_id from a raw ZeroMQ payload.

    Payloads follow one of two shapes:
    - ``str``               → the client_id itself  (e.g. RECOMPUTE)
    - ``tuple``             → ``(client_id, ...)``
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, tuple) and payload:
        return payload[0]
    return None


def _extract_zmq_sender_id(payload: Any) -> Any | None:
    """Return the ZeroMQ sender id when present in the payload.

    Payloads that need a reply carry the sender id as the *last*
    element of a 3- or 4-tuple: ``(client_id, data, zmq_sender_id)``.
    """
    if isinstance(payload, tuple) and len(payload) >= 3:
        return payload[-1]
    return None


class CommandRegistry:
    """Maps TaskCmd strings to CommandHandler instances."""

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cmd: str, handler: CommandHandler) -> None:
        """Register *handler* for *cmd*.  Overwrites any previous mapping."""
        self._handlers[cmd] = handler

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        cmd: str,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        """Authorize and execute the handler for *cmd*.

        1. Look up the handler — mark UNKNOWN and return if not found.
        2. Extract client_id from payload and check authorization.
        3. Call handler.execute(), forwarding task_id.
        """
        handler = self._handlers.get(cmd)
        zmq_sender_id = _extract_zmq_sender_id(payload)
        if handler is None:
            logger.error("Unknown command '%s' — no handler registered", cmd)
            # Reply so request/reply clients fail fast with a clear message
            # instead of blocking until their receive timeout expires (which
            # is what happens on a client/server version mismatch).
            self._reply_error(
                ctx,
                zmq_sender_id,
                code="UNKNOWN_COMMAND",
                message=(
                    f"Server does not support command '{cmd}'. The server may "
                    f"be running an older version — upgrade it to use this "
                    f"feature."
                ),
            )
            update_status(TaskStatus.UNKNOWN)
            return

        client_id = _extract_client_id(payload)

        if not self._authorize(cmd, client_id, zmq_sender_id, ctx):
            update_status(TaskStatus.ERROR)
            return

        try:
            handler.execute(payload, update_status, ctx, task_id)
        except Exception:
            logger.exception("Handler for '%s' raised an unhandled exception", cmd)
            # If the handler failed before replying, the request/reply client
            # would otherwise hang until its receive timeout.  Send an error
            # reply as a safety net (a no-op for fire-and-forget commands that
            # carry no sender id).
            self._reply_error(
                ctx,
                zmq_sender_id,
                code="HANDLER_ERROR",
                message=f"Server error while handling '{cmd}'.",
            )
            update_status(TaskStatus.ERROR)

    def _reply_error(
        self,
        ctx: HandlerContext,
        zmq_sender_id: Any | None,
        *,
        code: str,
        message: str,
    ) -> None:
        """Send a JSON error reply to a request/reply client.

        No-op when ``zmq_sender_id`` is absent (fire-and-forget commands do
        not expect a reply).  Never raises — a failed send must not crash the
        event loop.
        """
        if zmq_sender_id is None:
            return
        response = json.dumps({"status": "error", "code": code, "message": message})
        with contextlib.suppress(RuntimeError):
            ctx.control_mq.send_string(response, zmq_sender_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _authorize(
        self,
        cmd: str,
        client_id: str | None,
        zmq_sender_id: Any | None,
        ctx: HandlerContext,
    ) -> bool:
        """Return True when *client_id* is allowed to run *cmd*.

        On failure, sends an UNAUTHORIZED reply when a ZMQ sender id is
        available so the client is not left hanging.
        """
        if client_id is None:
            logger.warning("Cannot authorize cmd='%s': no client_id in payload", cmd)
            if zmq_sender_id is not None:
                with contextlib.suppress(RuntimeError):
                    ctx.control_mq.send_string(f"UNAUTHORIZED:{cmd}", zmq_sender_id)
            return False

        auth_ctx = ctx.control_mq.get_auth_context(client_id)
        if auth_ctx.can_execute_command(cmd, target_user=client_id):
            return True

        logger.warning(
            "UNAUTHORIZED: user=%s role=%s cmd=%s",
            client_id,
            auth_ctx.role.value,
            cmd,
        )
        if zmq_sender_id is not None:
            with contextlib.suppress(RuntimeError):
                ctx.control_mq.send_string(f"UNAUTHORIZED:{cmd}", zmq_sender_id)
        return False
