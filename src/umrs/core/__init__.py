"""Core task and message orchestration modules.

This package is the canonical home for the ZeroMQ/task/thread management
cluster. Legacy top-level modules under ``umrs`` re-export these symbols to
preserve compatibility during the tree reconstruction.
"""

from __future__ import annotations

# Import order matters for Pylance static analysis: lightweight modules first,
# MessageQueue last to avoid circular-import noise breaking Go-to-Definition.
from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.core.UMRSTask import UMRSTask
from umrs.core.TaskManager import TaskManager
from umrs.core.UMRSThreads import ThreadStatus, UMRSThreads
from umrs.core.MessageQueue import BASE_PORT, DATA_BASE_PORT, DEFAULT_NB_MSG, MessageQueue

__all__ = [
    "BASE_PORT",
    "DATA_BASE_PORT",
    "DEFAULT_NB_MSG",
    "MessageQueue",
    "TaskCmd",
    "TaskStatus",
    "TaskManager",
    "UMRSTask",
    "ThreadStatus",
    "UMRSThreads",
]


def __getattr__(name: str):
    if name in {"BASE_PORT", "DATA_BASE_PORT", "DEFAULT_NB_MSG", "MessageQueue"}:
        from umrs.core.MessageQueue import (
            BASE_PORT,
            DATA_BASE_PORT,
            DEFAULT_NB_MSG,
            MessageQueue,
        )

        return {
            "BASE_PORT": BASE_PORT,
            "DATA_BASE_PORT": DATA_BASE_PORT,
            "DEFAULT_NB_MSG": DEFAULT_NB_MSG,
            "MessageQueue": MessageQueue,
        }[name]

    if name in {"TaskCmd", "TaskStatus"}:
        from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus

        return {"TaskCmd": TaskCmd, "TaskStatus": TaskStatus}[name]

    if name == "TaskManager":
        from umrs.core.TaskManager import TaskManager

        return TaskManager

    if name == "UMRSTask":
        from umrs.core.UMRSTask import UMRSTask

        return UMRSTask

    if name in {"ThreadStatus", "UMRSThreads"}:
        from umrs.core.UMRSThreads import ThreadStatus, UMRSThreads

        return {"ThreadStatus": ThreadStatus, "UMRSThreads": UMRSThreads}[name]

    raise AttributeError(f"module 'umrs.core' has no attribute {name!r}")
