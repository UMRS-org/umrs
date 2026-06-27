"""Sparkline widget for ping latency / queue depth trends.

Wraps :class:`textual.widgets.Sparkline` so the connection tab can show
a rolling latency history.  Falls back gracefully if the installed
Textual version lacks ``Sparkline`` (it landed in 0.39).
"""

from __future__ import annotations

from typing import Any

try:
    from textual.widgets import Sparkline
    _HAS_SPARKLINE = True
except ImportError:  # pragma: no cover — older textual
    from textual.widgets import Static
    _HAS_SPARKLINE = False


class PingSparkline(Sparkline if _HAS_SPARKLINE else Static):  # type: ignore[misc]
    """Rolling latency sparkline."""

    def __init__(self, *, max_points: int = 60) -> None:
        if _HAS_SPARKLINE:
            super().__init__()
        else:
            super().__init__("")
        self._max_points = max_points
        self._data: list[float] = []

    def push(self, value: float) -> None:
        self._data.append(value)
        if len(self._data) > self._max_points:
            self._data = self._data[-self._max_points :]
        if _HAS_SPARKLINE:
            self.data = self._data
        else:
            self.update(f"latency: {value:.0f} ms")
