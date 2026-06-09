"""Core task and message orchestration modules.

This package is the canonical home for the ZeroMQ/task/thread management
cluster. Legacy top-level modules under ``umrs`` re-export these symbols to
preserve compatibility during the tree reconstruction.
"""

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
