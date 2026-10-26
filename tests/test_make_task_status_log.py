"""_make_task's status log must format when the command keeps no task row.

``TasksStatus.create_task`` returns ``None`` for read-only commands (JOB_LIST,
WORKFLOW_*). The status callback logged that id with ``%d``, so every job or
workflow poll raised a logging TypeError (ops-watch #551, its call stack #552).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.commands.base import CommandHandler


def _ctx_without_task_row() -> SimpleNamespace:
    tasks_status = SimpleNamespace(
        create_task=lambda _cmd, _payload: None,
        update_status=lambda _task_id, _status: None,
    )
    return SimpleNamespace(task_manager=SimpleNamespace(tasks_status=tasks_status))


def test_status_log_formats_when_command_has_no_task_id(caplog) -> None:
    task_id, update_status = CommandHandler._make_task(
        _ctx_without_task_row(), TaskCmd.JOB_LIST, ("alice", "{}", b"sender")
    )
    assert task_id is None
    with caplog.at_level(logging.INFO, logger="umrs.server.commands.base"):
        update_status(TaskStatus.COMPLETED)
    messages = [record.getMessage() for record in caplog.records]
    assert any("task_id=None" in m for m in messages)
