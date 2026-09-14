"""Summary stat-bar widget for the header.

Renders a compact one-line summary of counts-by-status (jobs running,
failed, etc.).  The bar is a plain ``Static`` updated via ``update()``.
"""

from __future__ import annotations

from textual.widgets import Static


def _render_summary(stats: dict) -> str:
    """Render a compact stats summary line."""
    total_jobs = stats.get("total_jobs", 0)
    counts = stats.get("job_counts", {})
    running = counts.get("running", 0)
    failed = counts.get("failed", 0)
    endpoints = stats.get("total_endpoints", 0)
    models = stats.get("total_models", 0)
    workflows = stats.get("total_workflows", 0)

    parts: list[str] = []
    # Jobs summary
    jobs_str = f"Jobs: {total_jobs}"
    if running:
        jobs_str += f" ({running} running)"
    if failed:
        jobs_str += f" ({failed} failed)"
    parts.append(jobs_str)
    parts.append(f"Endpoints: {endpoints}")
    parts.append(f"Models: {models}")
    parts.append(f"Workflows: {workflows}")
    return "  │  ".join(parts)


class StatBar(Static):
    """One-line summary of jobs/endpoints/models/workflows counts."""

    DEFAULT_CSS = """
    StatBar {
        dock: top;
        height: 1;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self) -> None:
        super().__init__("")
        self._stats: dict = {}

    def update_stats(self, stats: dict) -> None:
        self._stats = stats
        self.update(_render_summary(stats))
