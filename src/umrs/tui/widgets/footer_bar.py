"""Footer widget showing keybindings and last refresh time."""

from __future__ import annotations

import time

from textual.widgets import Static

_HELP_TEXT = (
    "[dim]1-7 tabs │ r refresh │ q quit │ ? help │ "
    "Ctrl+E export │ Ctrl+P palette[/dim]"
)


class FooterBar(Static):
    """Bottom-of-screen footer showing keybindings and last refresh time."""

    DEFAULT_CSS = """
    FooterBar {
        dock: bottom;
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 2;
    }
    """

    def __init__(self) -> None:
        super().__init__(_HELP_TEXT)
        self._last_refresh: float = 0.0

    def set_last_refresh(self, ts: float) -> None:
        self._last_refresh = ts
        refresh_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "—"
        self.update(
            f"[dim]1-7 tabs │ r refresh │ q quit │ ? help │ "
            f"Ctrl+E export │ Ctrl+P palette  Last refresh: {refresh_str}[/dim]"
        )
