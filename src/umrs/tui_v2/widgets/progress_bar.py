"""Progress bar widget for background operations."""

from __future__ import annotations

from textual.widget import Widget


class ProgressBar(Widget):
    """Animated progress bar with percentage and ETA."""

    DEFAULT_CSS = """
    ProgressBar {
        height: 3;
        width: 100%;
        padding: 0 1;
    }
    ProgressBar > #label {
        width: 20;
        text-align: right;
        color: $text;
    }
    ProgressBar > #track {
        width: 1fr;
        height: 1;
        background: $boost;
    }
    ProgressBar > #fill {
        height: 1;
        background: $primary;
        width: 0%;
        text-style: bold;
        color: $text-on-primary;
    }
    ProgressBar > #eta {
        width: 8;
        text-align: right;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._operation = None

    def bind_operation(self, operation) -> None:
        """Bind a tui_v2.models.Operation to this progress bar."""
        self._operation = operation
        self.refresh()

    def render(self) -> str:
        if self._operation is None:
            return " " * 80

        op = self._operation
        pct = op.progress.percent
        bar = op.progress.bar
        msg = op.progress.message

        # Format ETA
        eta_str = ""
        if op.progress.eta_seconds and op.progress.eta_seconds > 0:
            s = op.progress.eta_seconds
            if s < 60:
                eta_str = f"{int(s)}s"
            else:
                eta_str = f"{int(s // 60)}m{s % 60:02d}s"

        # Build bar
        bar_str = f"[{bar}]"
        pct_str = f"{pct:5.1f}%"
        status_str = f" {msg}"

        return f" {status_str} {bar_str} {pct_str}" + (f" {eta_str}" if eta_str else "")
