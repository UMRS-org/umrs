"""CommandRegistry — dispatches a TaskCmd to its registered handler.

The registry is the only place that knows the mapping cmd → handler.
The event loop just calls ``registry.dispatch(...)`` and is otherwise
unaware of any individual command's logic.

Authorization is centralized here: every payload is expected to carry
``client_id`` as its first element (string or first tuple item).  The
registry checks authorization before delegating to the handler, so
handlers never need to repeat that check. Since 2026-10-22 that includes
the project check (ADR-097 D-7, ``project_check``): a payload's project
must be one of the caller's live memberships unless the caller is an admin,
and a client-supplied ``_identity_verified`` never reaches a handler.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

from .base import CommandHandler, HandlerContext
from .project_check import project_refusal, strip_identity_claim

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
        self._internal_cmds: set[str] = set()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cmd: str, handler: CommandHandler) -> None:
        """Register *handler* for *cmd*.  Overwrites any previous mapping."""
        self._handlers[cmd] = handler

    def register_internal(self, cmd: str, handler: CommandHandler) -> None:
        """Register an internally-generated command that bypasses _authorize.

        Use this for commands that are never received directly from a ZMQ
        client — e.g. PROCESS_OBJECTS_BATCH, which is synthesised by the
        server's ZMQ drain loop from already-authorised PROCESS_OBJECTS
        requests.
        """
        self._handlers[cmd] = handler
        self._internal_cmds.add(cmd)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        cmd: str,
        payload: Any,
        ctx: HandlerContext,
    ) -> None:
        """Authorize and execute the handler for *cmd*.

        1. Look up the handler — reply UNKNOWN_COMMAND and return if not found.
        2. Extract client_id from payload and check authorization.
        3. Drop a client-supplied ``_identity_verified`` and check every
           project the payload names against the caller's live memberships
           (ADR-097 D-7).
        4. Call handler.execute(payload, ctx); the handler creates its own task_id(s).
        """
        handler = self._handlers.get(cmd)
        zmq_sender_id = _extract_zmq_sender_id(payload)
        if handler is None:
            logger.error("Unknown command '%s' — no handler registered", cmd)
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
            return

        if cmd not in self._internal_cmds:
            client_id = _extract_client_id(payload)
            if not self._authorize(cmd, client_id, zmq_sender_id, ctx):
                return
            payload = strip_identity_claim(payload)
            if not self._authorize_project(cmd, client_id, payload, zmq_sender_id, ctx):
                return

        try:
            handler.execute(payload, ctx)
        except Exception:  # noqa: BLE001 - handlers must not crash the loop
            logger.exception("Handler for '%s' raised an unhandled exception", cmd)
            self._reply_error(
                ctx,
                zmq_sender_id,
                code="HANDLER_ERROR",
                message=f"Server error while handling '{cmd}'.",
            )

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

    def _authorize_project(
        self,
        cmd: str,
        client_id: str,
        payload: Any,
        zmq_sender_id: Any | None,
        ctx: HandlerContext,
    ) -> bool:
        """ADR-097 D-7: True when every project *payload* names is one the
        caller may act in; else a typed JSON error reply (the gateway's
        codes: ``project_not_member``, ``project_membership_revoked``,
        ``membership_unavailable``)."""
        refusal = project_refusal(
            cmd, client_id, payload, ctx.control_mq, zmq_sender_id
        )
        if refusal is None:
            return True
        code, message = refusal
        logger.warning("ZMQ %s refused: user=%s %s (%s)", cmd, client_id, code, message)
        self._reply_error(ctx, zmq_sender_id, code=code, message=message)
        return False
