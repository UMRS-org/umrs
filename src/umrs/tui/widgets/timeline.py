"""Gantt-style attempt timeline widget.

Renders a horizontal bar per attempt, proportional to its duration
(``started_at`` → ``ended_at``).  Used in the job detail panel.

The widget is a plain ``Static`` so it renders identically on every
Textual version; we hand-roll the bar from unicode block characters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from textual.widgets import Static

_BAR_CHARS = " ▏▎▍▌▋▊▉█"


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _render_bar(frac: float, width: int = 20) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    remainder = frac * width - filled
    bar = "█" * filled
    if filled < width:
        idx = int(remainder * (len(_BAR_CHARS) - 1))
        bar += _BAR_CHARS[idx]
    bar += "░" * (width - filled - (1 if filled < width else 0))
    return bar


class AttemptTimeline(Static):
    """Render an attempt list as Gantt bars."""

    def __init__(self) -> None:
        super().__init__("")
        self._attempts: list[dict[str, Any]] = []

    def update_attempts(self, attempts: list[dict[str, Any]]) -> None:
        self._attempts = attempts
        self.update(self._render())

    def _render(self) -> str:
        if not self._attempts:
            return "(no attempts)"
        lines: list[str] = []
        starts = [_parse_dt(a.get("started_at")) for a in self._attempts]
        ends = [_parse_dt(a.get("ended_at")) for a in self._attempts]
        t_min = min(s for s in starts if s) if any(starts) else datetime.now()
        t_max = max(
            (e for e in ends if e),
            default=t_min,
        )
        span = (t_max - t_min).total_seconds() or 1.0
        for a in self._attempts:
            start = _parse_dt(a.get("started_at"))
            end = _parse_dt(a.get("ended_at"))
            s_off = (start - t_min).total_seconds() / span if start else 0.0
            e_off = (end - t_min).total_seconds() / span if end else 0.0
            bar = _render_bar(e_off - s_off if e_off > s_off else 0.05)
            label = f"#{a.get('attempt_no', '?')} {a.get('status', '?')}"
            lines.append(f"{label:<24} {bar}")
        return "\n".join(lines)
