"""Data transfer tab v2 — upload and download files to/from S3 buckets."""

from __future__ import annotations

from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    Static,
)

from ..forms.screens import UploadFormScreen


class DataTransferPane(Static):
    """Data transfer tab v2."""

    DEFAULT_CSS = """
    DataTransferPane {
        layout: vertical;
        padding: 0 1;
        height: 1fr;
    }
    DataTransferPane > .summary { height: 1; color: $text; }
    DataTransferPane DataTable { height: 1fr; }
    DataTransferPane > .form-section {
        height: auto;
        border: round $primary;
        padding: 0 1;
        margin-top: 1;
    }
    DataTransferPane > .actions-bar {
        height: 1;
        background: $boost;
        color: $text-muted;
        padding: 0 1;
        margin-top: 1;
    }
    DataTransferPane .form-row {
        height: 3;
        layout: horizontal;
        margin-bottom: 1;
    }
    DataTransferPane .form-row Label {
        width: 15;
        text-align: right;
        padding-right: 1;
    }
    DataTransferPane .form-row Input {
        width: 40;
    }
    DataTransferPane .form-row Button {
        width: 12;
        margin-left: 2;
    }
    """

    BINDINGS = [
        ("u", "show_upload", "Upload"),
        ("d", "show_download", "Download"),
    ]

    def __init__(self) -> None:
        super().__init__("")
        self._summary_widget = Static("", classes="summary")
        self._history_table = DataTable(id="history-table", cursor_type="row")
        self._history: list[dict[str, Any]] = []

        # Upload form
        self._upload_bucket = Input(placeholder="bucket_alias", id="upload-bucket")
        self._upload_path = Input(placeholder="/path/to/file", id="upload-path")
        self._upload_btn = Button("Upload", id="btn-upload", variant="primary")

        # Download form
        self._download_bucket = Input(placeholder="bucket_alias", id="download-bucket")
        self._download_path = Input(placeholder="/local/path", id="download-path")
        self._download_btn = Button("Download", id="btn-download", variant="primary")

        self._actions_bar = Static(
            "  Actions: u=Upload form  d=Download form",
            classes="actions-bar",
        )

    def compose(self) -> ComposeResult:
        yield self._summary_widget
        with Vertical():
            with Horizontal(classes="form-row"):
                yield Label("Upload bucket:")
                yield self._upload_bucket
                yield self._upload_btn
            with Horizontal(classes="form-row"):
                yield Label("Upload path:")
                yield self._upload_path
        with Vertical():
            with Horizontal(classes="form-row"):
                yield Label("Download bucket:")
                yield self._download_bucket
                yield self._download_btn
            with Horizontal(classes="form-row"):
                yield Label("Download path:")
                yield self._download_path
        yield self._history_table
        yield self._actions_bar

    def on_mount(self) -> None:
        self._upload_btn.on_click(self._on_upload_click)
        self._download_btn.on_click(self._on_download_click)
        self._render_history()

    def update_data(self, data: Any) -> None:
        self._summary_widget.update("  Data Transfer — use forms below or Enter for quick actions")

    def _render_history(self) -> None:
        self._history_table.clear(columns=True)
        self._history_table.add_column("#", width=3)
        self._history_table.add_column("Time", width=8)
        self._history_table.add_column("Action", width=8)
        self._history_table.add_column("Bucket", width=16)
        self._history_table.add_column("Path", width=20)
        self._history_table.add_column("Status", width=10)
        for idx, h in enumerate(self._history[-20:]):
            self._history_table.add_row(
                str(idx + 1),
                h.get("time", "?"),
                h.get("action", "?"),
                h.get("bucket", "?"),
                h.get("path", "?")[:20],
                h.get("status", "?"),
            )

    def _on_upload_click(self) -> None:
        bucket = self._upload_bucket.value
        path = self._upload_path.value
        if not bucket or not path:
            self.notify("Fill both bucket and path fields", severity="warning")
            return
        self._record_action("upload", bucket, "queued", path)
        self.app.notify(f"Upload queued: {path} → {bucket}")

    def _on_download_click(self) -> None:
        bucket = self._download_bucket.value
        path = self._download_path.value
        if not bucket or not path:
            self.notify("Fill both bucket and path fields", severity="warning")
            return
        self._record_action("download", bucket, "queued", path)
        self.app.notify(f"Download queued: {bucket} → {path}")

    def _record_action(self, action: str, bucket: str, status: str, path: str) -> None:
        import time
        self._history.append({
            "time": time.strftime("%H:%M"),
            "action": action,
            "bucket": bucket,
            "path": path,
            "status": status,
        })
        self._render_history()

    def action_show_upload(self) -> None:
        """Show upload confirmation dialog."""
        self.push_screen(
            UploadFormScreen(kind="upload"),
            self._on_upload_form_result,
        )

    def action_show_download(self) -> None:
        """Show download confirmation dialog."""
        self.push_screen(
            UploadFormScreen(kind="download"),
            self._on_download_form_result,
        )

    def _on_upload_form_result(self, result: dict | None) -> None:
        if result is None:
            return
        self.app.notify(
            f"Upload queued: {result.get('path', '?')} → {result.get('bucket', '?')}"
        )

    def _on_download_form_result(self, result: dict | None) -> None:
        if result is None:
            return
        self.app.notify(
            f"Download queued: {result.get('bucket', '?')} → {result.get('path', '?')}"
        )
