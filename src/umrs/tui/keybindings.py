"""
Minimal non-blocking keyboard input for the TUI dashboard.

Works on Unix/macOS by switching stdin to raw mode in a background thread.
Falls back gracefully on Windows (no keyboard support, just auto-refresh).
"""

from __future__ import annotations

import sys
import threading
from typing import Callable, Optional


class KeyReader:
    """Read single keystrokes from stdin without blocking the main thread."""

    def __init__(self, on_key: Callable[[str], None]):
        self._on_key = on_key
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            import termios
        except ImportError:
            # Windows — no raw mode; dashboard still works via auto-refresh
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            # Use cbreak-like mode that preserves output processing (OPOST)
            # so Rich's terminal rendering is not corrupted.  Disable ISIG
            # so Ctrl+C is readable as '\x03' instead of raising SIGINT.
            new = termios.tcgetattr(fd)
            new[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
            new[6][termios.VMIN] = 1
            new[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, new)
            while not self._stop.is_set():
                ch = sys.stdin.read(1)
                if not ch:
                    break
                # Handle escape sequences (arrow keys)
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    _ARROW_MAP = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}
                    key = _ARROW_MAP.get(seq, "esc")
                else:
                    key = ch
                self._on_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
