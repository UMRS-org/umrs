"""Connection tab pane v2."""

from __future__ import annotations

from typing import Any

from textual.widgets import Static


class ConnectionPane(Static):
    """Connection tab pane v2."""

    DEFAULT_CSS = """
    ConnectionPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    ConnectionPane > .content {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._content = Static("", classes="content")

    def compose(self):
        yield self._content

    def set_connection_info(self, info: dict[str, Any]) -> None:
        self._connection_info = info
        self._render_connection()

    def update_data(self, data: Any) -> None:
        self._connection_data = data or {}
        self._render_connection()

    def _render_connection(self) -> None:
        info = getattr(self, "_connection_info", {})
        data = getattr(self, "_connection_data", {})

        # Prefer connection_info from data if available (new format)
        conn_info = data.get("connection_info", {}) if isinstance(data, dict) else info

        lines = ["[bold]Connection Profile[/bold]"]
        profile = conn_info.get("profile", info.get("profile", info.get("name", "?")))
        mode = conn_info.get("mode", conn_info.get("zmq", {}).get("tunnel_mode", "?"))
        lines.append(f"  Profile: {profile}")
        lines.append(f"  Mode: {mode}")

        zmq = conn_info.get("zmq", info.get("zmq", conn_info))
        lines.append("")
        lines.append("[bold]ZMQ[/bold]")
        lines.append(f"  Host: {zmq.get('host', '?')}")
        lines.append(f"  Control port: {zmq.get('control_port', '?')}")
        lines.append(f"  Data port range: {zmq.get('data_port_range', '?')}")
        lines.append(f"  Tunnel: {zmq.get('tunnel_mode', False)}")

        services = conn_info.get("services", info.get("services", {}))
        if services:
            lines.append("")
            lines.append("[bold]Services[/bold]")
            lines.append(f"  MinIO: {services.get('minio_url', '?')}")
            lines.append(f"  RAG: {services.get('rag_url', '?')}")

        gw_url = conn_info.get("gateway_url") or info.get("gateway_url")
        if gw_url:
            lines.append("")
            lines.append("[bold]HTTP Gateway[/bold]")
            lines.append(f"  URL: {gw_url}")

        ping_ok = conn_info.get("ping_ok") if isinstance(conn_info, dict) else None
        if ping_ok is None:
            ping_ok = data.get("ping_ok") if isinstance(data, dict) else None
        if ping_ok is not None:
            status = "● Reachable" if ping_ok else "● Unreachable"
            lines.append("")
            lines.append(f"[bold]ZMQ Ping[/bold]: {status}")

        self._content.update("\n".join(lines))
