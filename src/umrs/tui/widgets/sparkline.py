"""Sparkline widget for ping latency / queue depth trends.

Uses a plain ``Static`` widget to avoid segfaults caused by the
``Sparkline`` C extension on some platforms (notably macOS + Textual 8.x).
Falls back to the same ``Static`` widget if ``Sparkline`` is unavailable.
"""

from __future__ import annotations

from textual.widgets import Static


class PingSparkline(Static):
    """Rolling latency display (plain Static, no Sparkline segfault risk)."""

    def __init__(self, *, max_points: int = 60) -> None:
        super().__init__("")
        self._max_points = max_points
        self._data: list[float] = []

    def push(self, value: float) -> None:
        self._data.append(value)
        if len(self._data) > self._max_points:
            self._data = self._data[-self._max_points :]
        recent = self._data[-10:]
        bars = "".join(self._spark_char(v) for v in recent)
        self.update(f"[dim]latency:[/dim] {bars} {value:.0f} ms")

    @staticmethod
    def _spark_char(v: float) -> str:
        if v < 10:
            return "▁"
        if v < 30:
            return "▂"
        if v < 50:
            return "▃"
        if v < 80:
            return "▄"
        if v < 120:
            return "▅"
        if v < 200:
            return "▆"
        if v < 500:
            return "▇"
        return "█"
