"""Core task and message orchestration modules.

This package is the canonical home for the ZeroMQ/task/thread management
cluster. Legacy top-level modules under ``umrs`` re-export these symbols to
preserve compatibility during the tree reconstruction.
"""

from __future__ import annotations

# Lazy imports via __getattr__ keep lightweight client commands (e.g. `umrs-client
# info`) from pulling server-only deps (psycopg2, tabulate, diskcache) at startup.

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


# Maps each exported symbol to the submodule that defines it. Submodules are
# imported on first access only. Resolved values are cached into this module's
# globals so that names colliding with submodule names (e.g. ``MessageQueue``,
# ``TaskManager``, ``UMRSTask``, ``UMRSThreads``) resolve to the class/value
# rather than the submodule object, matching the previous eager-import behavior.
_SYMBOL_SOURCES = {
    "BASE_PORT": "umrs.core.MessageQueue",
    "DATA_BASE_PORT": "umrs.core.MessageQueue",
    "DEFAULT_NB_MSG": "umrs.core.MessageQueue",
    "MessageQueue": "umrs.core.MessageQueue",
    "TaskCmd": "umrs.core.TaskCmdStatus",
    "TaskStatus": "umrs.core.TaskCmdStatus",
    "TaskManager": "umrs.core.TaskManager",
    "UMRSTask": "umrs.core.UMRSTask",
    "ThreadStatus": "umrs.core.UMRSThreads",
    "UMRSThreads": "umrs.core.UMRSThreads",
}


def __getattr__(name: str):
    module_path = _SYMBOL_SOURCES.get(name)
    if module_path is None:
        raise AttributeError(f"module 'umrs.core' has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
