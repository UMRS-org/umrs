"""
Minimal non-blocking keyboard input for the TUI dashboard.

Works on Unix/macOS by switching stdin to cbreak mode in a background thread.
Falls back gracefully on Windows (no keyboard support, just auto-refresh).
"""

from __future__ import annotations

import select
import sys
import threading
from collections.abc import Callable


class KeyReader:
    """Read single keystrokes from stdin without blocking the main thread."""

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            import termios
            import tty
        except ImportError:
            # Windows — no raw mode; dashboard still works via auto-refresh
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Switch input processing immediately so single-key navigation
            # works even while the live dashboard is continuously rendering.
            tty.setcbreak(fd, termios.TCSANOW)
            new = termios.tcgetattr(fd)
            new[3] &= ~termios.ISIG
            termios.tcsetattr(fd, termios.TCSANOW, new)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    break
                # Handle escape sequences (arrow keys) vs bare Esc
                if ch == "\x1b":
                    # Wait briefly for additional escape sequence chars
                    ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if ready:
                        seq = sys.stdin.read(2)
                        _ARROW_MAP = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}
                        key = _ARROW_MAP.get(seq, "esc")
                    else:
                        # No follow-up chars: bare Esc key
                        key = "esc"
                else:
                    key = ch
                self._on_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old_settings)
