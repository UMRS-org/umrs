"""Tests for CommandRegistry.dispatch reply-on-failure behaviour.

Regression coverage for the robustness fix that prevents request/reply
clients from hanging until their receive timeout when:
  - the server does not recognise a command (version mismatch), or
  - a handler raises before sending its own reply.

In both cases the registry must send a JSON error reply to the ZMQ sender
so the client fails fast with a clear message.
"""

from __future__ import annotations

import json

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.commands.base import CommandHandler, HandlerContext
from umrs.server.commands.registry import CommandRegistry


class _RecordingControlMQ:
    """Fake control_mq that records send_string calls and always authorizes."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    def send_string(self, message, sender_id=None):
        self.sent.append((message, sender_id))

    def get_auth_context(self, client_id):
        return self

    def can_execute_command(self, command, target_user=None):
        return True


class _StatusRecorder:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def __call__(self, status: str) -> None:
        self.statuses.append(status)


class _RaisingHandler(CommandHandler):
    def execute(self, payload, update_status, ctx, task_id):
        raise RuntimeError("boom")


def _make_ctx(control_mq) -> HandlerContext:
    return HandlerContext(
        task_manager=None,
        control_mq=control_mq,
        server_cfg=None,
        device="cpu",
        embedding_backend=None,
        configure_file="",
    )


def test_unknown_command_replies_with_error() -> None:
    registry = CommandRegistry()
    mq = _RecordingControlMQ()
    ctx = _make_ctx(mq)
    status = _StatusRecorder()

    # 3-tuple payload: (client_id, payload_str, zmq_sender_id)
    payload = ("alice", "{}", b"sender-1")
    registry.dispatch("NO_SUCH_CMD", payload, status, ctx, task_id=1)

    assert len(mq.sent) == 1
    message, sender = mq.sent[0]
    assert sender == b"sender-1"
    body = json.loads(message)
    assert body["status"] == "error"
    assert body["code"] == "UNKNOWN_COMMAND"
    assert "NO_SUCH_CMD" in body["message"]
    assert TaskStatus.UNKNOWN in status.statuses


def test_unknown_command_without_sender_does_not_reply() -> None:
    registry = CommandRegistry()
    mq = _RecordingControlMQ()
    ctx = _make_ctx(mq)
    status = _StatusRecorder()

    # Fire-and-forget payload (plain str) carries no sender id.
    registry.dispatch("NO_SUCH_CMD", "alice", status, ctx, task_id=1)

    assert mq.sent == []
    assert TaskStatus.UNKNOWN in status.statuses


def test_handler_exception_replies_with_error() -> None:
    registry = CommandRegistry()
    registry.register("BOOM", _RaisingHandler())
    mq = _RecordingControlMQ()
    ctx = _make_ctx(mq)
    status = _StatusRecorder()

    payload = ("alice", "{}", b"sender-2")
    registry.dispatch("BOOM", payload, status, ctx, task_id=1)

    assert len(mq.sent) == 1
    message, sender = mq.sent[0]
    assert sender == b"sender-2"
    body = json.loads(message)
    assert body["status"] == "error"
    assert body["code"] == "HANDLER_ERROR"
    assert TaskStatus.ERROR in status.statuses
