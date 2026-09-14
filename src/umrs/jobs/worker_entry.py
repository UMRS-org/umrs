"""libexec worker entry point for the companion venue (wave T3b m4).

Spawned by `CompanionWorkerSupervisor` as
``python -m umrs.jobs.worker_entry``. Responsibilities in m4 are thin by
design — the worker holds the session, beats liveness to the supervisor,
and lives until signalled. Turn payloads arrive at the HTTP surface
(T3b m5) or over the ACP rail; this entry only carries liveness.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import time

logger = logging.getLogger(__name__)

_STOP = False


def _handle_signal(*_args) -> None:
    global _STOP
    _STOP = True


def _required_env(name: str) -> str:
    value = os.environ.get(name) or ""
    if not value.strip():
        raise RuntimeError(f"missing required env: {name}")
    return value


def _beat_once(socket_path: str, registry_key: str) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        client.connect(socket_path)
        client.sendall(f"beat {registry_key}\n".encode())


def main() -> int:
    if os.environ.get("R4_COMPANION_WORKER") != "1":
        print(
            "umrs companion worker spawned unsupervised (not via the venue)",
            file=sys.stderr,
        )
        return 2
    registry_key = _required_env("R4_COMPANION_REGISTRY_KEY")
    socket_path = _required_env("R4_COMPANION_SOCKET_PATH")
    beat_sec = float(os.environ.get("R4_COMPANION_BEAT_SEC", "5.0"))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    logger.info(
        "companion worker starting: %s (beat %.1fs over %s)",
        registry_key,
        beat_sec,
        socket_path,
    )

    # Worker waits for the supervisor's socket listener to come up, then beats
    # on schedule forever. The supervisor consumes beats into the manager's
    # heartbeat path (same push mechanism as budget_state events).
    next_beat = 0.0
    while not _STOP:
        now = time.monotonic()
        if now >= next_beat:
            try:
                _beat_once(socket_path, registry_key)
                next_beat = now + beat_sec
            except OSError:
                # Supervisor listener not up yet — retry sooner than the beat.
                next_beat = now + min(beat_sec, 1.0)
        time.sleep(min(0.25, beat_sec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
