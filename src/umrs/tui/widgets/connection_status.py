"""Connection status indicator widget."""

from __future__ import annotations

from textual.widgets import Static


class ConnectionStatus(Static):
    """Small widget showing connection status with a coloured dot."""

    DEFAULT_CSS = """
    ConnectionStatus {
        height: 1;
        padding: 0 2;
        background: $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__("[dim]● Unknown[/dim]")

    def set_connected(self, connected: bool | None) -> None:
        if connected is True:
            self.update("[green]● Connected[/green]")
        elif connected is False:
            self.update("[red]● Disconnected[/red]")
        else:
            self.update("[dim]● Unknown[/dim]")
