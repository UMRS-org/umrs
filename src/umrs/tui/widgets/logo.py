"""UMRS ASCII art logo widget."""

from __future__ import annotations

from textual.widgets import Static

UMRS_LOGO = r"""
 ██╗   ██╗███╗   ███╗██████╗ ███████╗
 ██║   ██║████╗ ████║██╔══██╗██╔════╝
 ██║   ██║██╔████╔██║██████╔╝███████╗
 ██║   ██║██║╚██╔╝██║██╔══██║██╔════╝
 ╚██████╔╝██║ ╚═╝ ██║██║  ██║███████║
  ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝
"""


class LogoWidget(Static):
    """Renders the UMRS ASCII art logo."""

    DEFAULT_CSS = """
    LogoWidget {
        dock: top;
        height: 7;
        padding: 0 2;
        background: $surface;
        color: $primary;
        text-align: center;
        border-bottom: solid $primary;
    }
    """

    def __init__(self, subtitle: str = "") -> None:
        text = UMRS_LOGO
        if subtitle:
            text += f"\n{subtitle}"
        super().__init__(text)
