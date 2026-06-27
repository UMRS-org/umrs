"""Connection tab pane — connection profile + server health + ping sparkline."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from ..widgets.sparkline import PingSparkline


class ConnectionPane(Static):
    """Connection tab pane."""

    DEFAULT_CSS = """
    ConnectionPane { layout: vertical; padding: 0 1; }
    ConnectionPane > .summary { height: 1; color: $text; }
    ConnectionPane > .detail { height: 1fr; border: round $primary; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._ping_ok: bool | None = None
        self._ping_ms: float | None = None
        self._connection_info: dict[str, Any] = {}
        self._summary_widget = Static("", classes="summary")
        self._detail = Static("", classes="detail")
        self._sparkline = PingSparkline()

    def compose(self):
        yield self._summary_widget
        yield self._detail

    def update_data(self, data: Any) -> None:
        if data is None:
            self._summary_widget.update("[red]No data[/red]")
            return
        self._ping_ok = data.get("ping_ok")
        self._ping_ms = data.get("ping_ms")
        if self._ping_ms is not None:
            self._sparkline.push(self._ping_ms)
        self._render_summary()
        self._render_detail()

    def set_connection_info(self, info: dict[str, Any]) -> None:
        self._connection_info = info
        self._render_detail()

    def _render_summary(self) -> None:
        if self._ping_ok is True:
            status = "[green]● READY[/green]"
        elif self._ping_ok is False:
            status = "[red]● UNREACHABLE[/red]"
        else:
            status = "[dim]● unknown[/dim]"
        latency = f"{self._ping_ms:.0f} ms" if self._ping_ms is not None else "—"
        self._summary_widget.update(f"{status}  latency={latency}")

    def _render_detail(self) -> None:
        info = self._connection_info
        lines = ["[bold]Connection Profile[/bold]"]
        rows = [
            ("Profile", info.get("profile_name", "—")),
            ("Mode", info.get("mode", "—")),
            ("Server Host", info.get("server_host", "127.0.0.1")),
            ("Control Port", str(info.get("control_port", 5555))),
            ("Data Base Port", str(info.get("data_base_port", 5556))),
            ("Tunnel Mode", str(info.get("tunnel_mode", False))),
            ("Health Timeout (ms)", str(info.get("health_check_timeout_ms", 5000))),
            ("MinIO URL", info.get("minio_url", "—")),
            ("RAG URL", info.get("rag_url", "—")),
        ]
        ssh_host = info.get("ssh_host")
        if ssh_host:
            rows.append(("SSH Host", ssh_host))
        for label, value in rows:
            lines.append(f"  {label}: {value}")
        forwards = info.get("local_forwards", [])
        if forwards:
            lines.append("\n[bold]SSH Port Forwards[/bold]")
            for fwd in forwards:
                lines.append(
                    f"  {fwd.get('local_port', '?')} → "
                    f"{fwd.get('remote_host', '?')}:{fwd.get('remote_port', '?')}"
                )
        self._detail.update("\n".join(lines))
