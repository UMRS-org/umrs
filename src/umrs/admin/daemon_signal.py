"""PID file + SIGHUP signalling between ``umrs-server admin`` CLI and the daemon.

The offline admin CLI mutates ``user_files/*.json`` directly. For the running
``umrs-server`` daemon to pick up the new records without a restart, the admin
CLI sends ``SIGHUP``; the daemon installs a handler that flips a flag the event
loop checks, and then calls :func:`umrs.server.maintenance.reload_configuration`.

This module is intentionally dependency-free so it can be imported by both the
daemon process and the offline CLI without dragging in ZMQ/server internals.
"""

from __future__ import annotations

import errno
import os
import signal
from dataclasses import dataclass
from typing import Optional

from umrs.shared.config import get_resolved_umrs_cfg_dir


PID_FILE_ENV_VAR = "UMRS_SERVER_PID_FILE"
DEFAULT_PID_FILE_NAME = "umrs-server.pid"


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------


def resolve_pid_file_path(cfg_dir: Optional[str] = None) -> str:
    """Return the canonical path for the daemon's PID file.

    Resolution order:
    1. ``UMRS_SERVER_PID_FILE`` environment variable (absolute path).
    2. ``<cfg_dir>/umrs-server.pid`` where ``cfg_dir`` defaults to
       :func:`get_resolved_umrs_cfg_dir`.
    """
    override = os.environ.get(PID_FILE_ENV_VAR)
    if override:
        return os.path.realpath(os.path.expanduser(override))
    base = cfg_dir or get_resolved_umrs_cfg_dir()
    return os.path.realpath(os.path.join(base, DEFAULT_PID_FILE_NAME))


# ---------------------------------------------------------------------------
# daemon-side helpers
# ---------------------------------------------------------------------------


def write_pid_file(path: Optional[str] = None) -> str:
    """Write the current process PID to *path* (or the resolved default).

    Creates the parent directory if missing. Written atomically (tmp + rename)
    with mode 0o600. Returns the realpath of the file.
    """
    target = path or resolve_pid_file_path()
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\n")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, target)
    return target


def remove_pid_file(path: Optional[str] = None, only_if_ours: bool = True) -> bool:
    """Remove the PID file. Returns True if a file was actually removed.

    When *only_if_ours* is True (default), the file is only removed if it
    contains the current process's PID. This prevents a crashing shutdown
    from deleting a freshly-written sibling daemon's PID file.
    """
    target = path or resolve_pid_file_path()
    if not os.path.exists(target):
        return False
    if only_if_ours:
        pid = read_pid_file(target)
        if pid != os.getpid():
            return False
    try:
        os.remove(target)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def read_pid_file(path: Optional[str] = None) -> Optional[int]:
    """Read a PID from the pid file. Returns None if missing or malformed."""
    target = path or resolve_pid_file_path()
    try:
        with open(target, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        pid = int(raw.splitlines()[0])
    except (ValueError, IndexError):
        return None
    return pid if pid > 0 else None


def is_process_alive(pid: int) -> bool:
    """Best-effort liveness probe using ``kill(pid, 0)``."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it; treat as alive.
        return True
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


# ---------------------------------------------------------------------------
# admin-CLI-side: signal the daemon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalResult:
    sent: bool
    pid: Optional[int]
    reason: str

    def to_dict(self) -> dict:
        return {"sent": self.sent, "pid": self.pid, "reason": self.reason}


def signal_reload(path: Optional[str] = None) -> SignalResult:
    """Send ``SIGHUP`` to the daemon identified by the PID file.

    Best-effort: returns a :class:`SignalResult` describing what happened
    instead of raising. The caller is expected to print a compact note.
    """
    target = path or resolve_pid_file_path()
    pid = read_pid_file(target)
    if pid is None:
        return SignalResult(False, None, f"no pid file at {target}")
    if not is_process_alive(pid):
        return SignalResult(False, pid, f"stale pid file at {target} (pid {pid} not running)")
    try:
        os.kill(pid, signal.SIGHUP)
    except PermissionError:
        return SignalResult(False, pid, f"permission denied sending SIGHUP to pid {pid}")
    except ProcessLookupError:
        return SignalResult(False, pid, f"pid {pid} disappeared before SIGHUP")
    except OSError as exc:
        return SignalResult(False, pid, f"SIGHUP failed: {exc}")
    return SignalResult(True, pid, f"SIGHUP sent to pid {pid}")


# ---------------------------------------------------------------------------
# daemon-side: install SIGHUP handler
# ---------------------------------------------------------------------------


# Module-level flag inspected by the server event loop. Using a plain bool
# is safe because Python only runs signal handlers between bytecode ops in
# the main thread, and the event loop reads+clears the flag in the same
# thread.
_reload_requested: bool = False


def install_reload_signal_handler() -> None:
    """Install a SIGHUP handler that sets :data:`_reload_requested`.

    Safe to call multiple times; later calls replace the previous handler.
    Must be called from the main thread (Python requirement for ``signal``).
    """

    def _handler(signum, frame):  # noqa: ARG001 - signal handler signature
        global _reload_requested
        _reload_requested = True

    signal.signal(signal.SIGHUP, _handler)


def consume_reload_request() -> bool:
    """Return True exactly once per received SIGHUP, resetting the flag."""
    global _reload_requested
    if _reload_requested:
        _reload_requested = False
        return True
    return False


__all__ = [
    "PID_FILE_ENV_VAR",
    "DEFAULT_PID_FILE_NAME",
    "SignalResult",
    "consume_reload_request",
    "install_reload_signal_handler",
    "is_process_alive",
    "read_pid_file",
    "remove_pid_file",
    "resolve_pid_file_path",
    "signal_reload",
    "write_pid_file",
]
