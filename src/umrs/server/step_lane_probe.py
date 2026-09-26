"""Probe step of the step-lane smoke (receipt R2, rccs step-lane spec §11).

Runs inside the step container on a compute node and reports what the
hardened pod leaves it: its uid, whether the root filesystem is read-only,
the names (never the values) of its environment, whether it can open a TCP
connection to each target it is given, and the GPU device nodes it sees.
Standard library only: the step image carries no server dependencies.
"""

from __future__ import annotations

import glob
import os
import platform
import resource
import socket
import time
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

_CONNECT_TIMEOUT = 3.0
#: Printed (to the job's log) before a sleeping probe sleeps: the smoke's
#: kill check cancels only once the step container runs.
SLEEP_MARKER = "umrs-probe: sleeping"


class ProbeStep(Step):
    """``config["connect"]``: ``[[host, port], ...]`` that must be unreachable;
    ``config["sleep_seconds"]``: sleep that long first (the kill check)."""

    def setup(self, config: dict) -> None:
        self._targets = [(str(h), int(p)) for h, p in config.get("connect") or []]
        self._sleep = float(config.get("sleep_seconds") or 0)

    def run(self, ctx: StepContext) -> StepResult:
        if self._sleep > 0:
            print(SLEEP_MARKER, flush=True)
            time.sleep(self._sleep)
        return StepResult(extra={"probe": probe(self._targets)})

    def teardown(self) -> None:
        return None


def probe(targets: list[tuple[str, int]]) -> dict[str, Any]:
    return {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "rootfs_read_only": _read_only("/"),
        "env_keys": sorted(os.environ),
        "connect": {f"{h}:{p}": _connect(h, p) for h, p in targets},
        "gpu_devices": sorted(glob.glob("/dev/nvidia*")),
        "pids_max": _read("/sys/fs/cgroup/pids.max"),
        "nproc_soft": resource.getrlimit(resource.RLIMIT_NPROC)[0],
        "arch": platform.machine(),
    }


def _read_only(directory: str) -> bool:
    path = os.path.join(directory, f".umrs-probe-{os.getpid()}")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError:
        return True
    os.close(fd)
    os.unlink(path)
    return False


def _connect(host: str, port: int) -> str:
    """The outcome: open when a TCP connection succeeds, else the error's
    class name (gaierror, OSError for an unreachable network, ...)."""
    try:
        with socket.create_connection((host, port), timeout=_CONNECT_TIMEOUT):
            return "open"
    except OSError as exc:
        return type(exc).__name__


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read(64).strip()
    except OSError:
        return ""
